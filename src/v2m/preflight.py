"""Resource preflight: refuse a run that cannot fit, before it starts.

docs/ARCHITECTURE.md Section 3.2: "estimate peak RAM from frame count x
resolution; refuse and suggest smaller settings rather than OOM mid-run",
and Section 3.5: "Disk exhaustion -> preflight estimate". A 40-minute
pipeline dying at minute 35 is the failure this exists to prevent.

The two dominant RAM terms (the TSDF volume and Poisson) are
measurements on this project's stack; the smaller ones are conservative
estimates, and each comment below says which is which. Peak RAM is the
max over pending phases, not the sum: phases run one after another in
one process and each frees its large buffers on return.

Two policies, deliberately different:
- RAM: refuse only when the machine's *total* memory is below the
  estimate (it physically cannot fit). When *currently available* memory
  is short, warn instead -- macOS compresses and swaps rather than
  OOM-killing, and other apps' memory is usually reclaimable.
- Disk: refuse when free space is below the estimate. Running out of
  disk mid-phase fails hard, with no reclaimable slack to fall back on.
"""

from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass
from pathlib import Path

import psutil

from v2m import capability
from v2m.config import PipelineConfig
from v2m.errors import CapabilityError
from v2m.phase2_sfm.dense.monodepth_tsdf import MAX_TSDF_RESOLUTION, TSDF_BYTES_PER_VOXEL
from v2m.types import PhaseName

logger = logging.getLogger("v2m.preflight")

GB = 1e9

# Interpreter with numpy/open3d/pycolmap/cv2/trimesh imported: 0.38GB RSS
# (measured), rounded up for allocator slack and torch when it loads.
_BASELINE_BYTES = 1.0 * GB
# pycolmap SIFT extraction + sequential matching at a 1600px cap: well
# under this in practice.
_SFM_BYTES = 0.8 * GB
# Depth-Anything-V2-Small (25M params) weights + activations at its 518px
# input; ~1GB is the commonly reported footprint. Not measured here
# (huggingface.co is blocked in this sandbox -- see CLAUDE.md).
_DEPTH_MODEL_BYTES = 1.0 * GB
# open3d screened Poisson, depth 10, 1M points: 1.9GB peak (measured). Each
# extra octree level roughly quadruples the surface-node count.
_POISSON_BYTES_PER_MILLION_POINTS_D10 = 1.9 * GB
# Mesh/output phases write a few hundred MB at most (300k-face budget).
_MESH_AND_OUTPUT_DISK_BYTES = 0.3 * GB
# A JPEG at quality 95 of natural footage compresses ~8-10x; 8x is the
# pessimistic end.
_JPEG_COMPRESSION = 8.0
# One SIFT keypoint in COLMAP's database: 128B descriptor + ~24B geometry.
_SIFT_BYTES_PER_FEATURE = 152
_ASPECT = 0.75  # 4:3 worst case of a cap applied to the longest edge


@dataclass
class Estimate:
    peak_ram_bytes: float
    disk_bytes: float
    dense_backend: str


def estimate(cfg: PipelineConfig, pending: list[PhaseName]) -> Estimate:
    backend = cfg.dense.backend
    if backend == "auto":
        backend = capability.probe().dense_backend

    frames = cfg.ingest.frame_budget
    pixels = cfg.ingest.resolution_cap_px**2 * _ASPECT
    jpeg_bytes = pixels * 3 / _JPEG_COMPRESSION

    phase_ram: dict[PhaseName, float] = {
        PhaseName.INGEST: 0.2 * GB,
        PhaseName.SFM_SPARSE: _SFM_BYTES,
        PhaseName.SFM_DENSE: (
            _DEPTH_MODEL_BYTES + MAX_TSDF_RESOLUTION**3 * TSDF_BYTES_PER_VOXEL
            if backend == "monodepth_tsdf"
            else 0.2 * GB
        ),
        # The dense cloud is voxel-downsampled before Poisson; ~1M points
        # is typical for an object capture. Scale by 4x per octree level
        # above 10.
        PhaseName.MESH: _POISSON_BYTES_PER_MILLION_POINTS_D10
        * 4 ** max(0, cfg.dense.poisson_depth_low - 10),
        PhaseName.PRINT_PREP: 1.0 * GB,
    }
    phase_disk: dict[PhaseName, float] = {
        PhaseName.INGEST: frames * jpeg_bytes,
        # undistorted image copies + the feature database
        PhaseName.SFM_SPARSE: frames
        * (jpeg_bytes + cfg.sfm.sift_max_num_features * _SIFT_BYTES_PER_FEATURE),
        # one float32 depth map per frame, saved as .npy
        PhaseName.SFM_DENSE: frames * pixels * 4 if backend == "monodepth_tsdf" else 0.05 * GB,
        PhaseName.MESH: _MESH_AND_OUTPUT_DISK_BYTES / 2,
        PhaseName.PRINT_PREP: _MESH_AND_OUTPUT_DISK_BYTES / 2,
    }
    return Estimate(
        peak_ram_bytes=_BASELINE_BYTES + max(phase_ram[p] for p in pending),
        disk_bytes=sum(phase_disk[p] for p in pending),
        dense_backend=backend,
    )


def check(cfg: PipelineConfig, run_dir: Path, pending: list[PhaseName]) -> list[str]:
    """Raise `CapabilityError` if the pending phases cannot fit on this
    machine; return (and log) warnings for merely-tight situations."""
    if not pending:
        return []
    est = estimate(cfg, pending)
    memory = psutil.virtual_memory()
    run_dir.mkdir(parents=True, exist_ok=True)
    free_disk = shutil.disk_usage(run_dir).free
    warnings: list[str] = []

    smaller_settings = (
        "Try --preset fast (sparse-only dense, 800px frames), or lower "
        "ingest.frame_budget / ingest.resolution_cap_px with --set."
    )
    if memory.total < est.peak_ram_bytes:
        raise CapabilityError(
            f"This run needs ~{est.peak_ram_bytes / GB:.1f}GB of RAM at its peak "
            f"(dense backend '{est.dense_backend}'), but this machine has "
            f"{memory.total / GB:.1f}GB in total.",
            remedy=smaller_settings,
        )
    if memory.available < est.peak_ram_bytes:
        warnings.append(
            f"Only {memory.available / GB:.1f}GB of RAM is free right now and this run peaks "
            f"at ~{est.peak_ram_bytes / GB:.1f}GB -- expect swapping. Close other apps if it "
            "slows to a crawl."
        )
    if free_disk < est.disk_bytes:
        raise CapabilityError(
            f"This run needs ~{est.disk_bytes / GB:.1f}GB of disk, but only "
            f"{free_disk / GB:.1f}GB is free at {run_dir}.",
            remedy="Free some space (old runs under runs/ are safe to delete), or "
            + smaller_settings[0].lower()
            + smaller_settings[1:],
        )

    for warning in warnings:
        logger.warning(warning)
    logger.info(
        "Preflight: peak RAM ~%.1fGB of %.1fGB, disk ~%.1fGB of %.1fGB free.",
        est.peak_ram_bytes / GB,
        memory.total / GB,
        est.disk_bytes / GB,
        free_disk / GB,
    )
    return warnings
