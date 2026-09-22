"""Always-works dense-stage fallback: re-exports the sparse point cloud.

Selected by `capability.Capabilities.dense_backend` when torch or open3d
is unavailable, and by the `fast` preset, where a full monodepth+TSDF
pass isn't worth the time for a quick capture-quality check
(docs/ARCHITECTURE.md Section 4, M3: "sparse_only.py first -- trivial,
unblocks M4").

Assumes open3d itself is present -- unlike torch/transformers, open3d is
a core dependency (`capability.py`'s `_CORE_LIBS`) needed unconditionally
by Phase 3's Poisson meshing, so there is no real scenario where this
backend runs without it.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import open3d as o3d

from v2m.config import DenseConfig
from v2m.errors import SfMError
from v2m.types import DenseResult


def densify(sfm_dir: Path, output_dir: Path, config: DenseConfig) -> DenseResult:
    sparse_ply = sfm_dir / "sparse.ply"
    if not sparse_ply.exists():
        raise SfMError(
            f"No sparse.ply found in {sfm_dir}.",
            remedy="Run `v2m sfm` first to produce a sparse reconstruction.",
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    dense_ply = output_dir / "dense.ply"
    shutil.copyfile(sparse_ply, dense_ply)

    cloud = o3d.io.read_point_cloud(str(sparse_ply))
    return DenseResult(
        backend="sparse_only",
        num_points=len(cloud.points),
        dense_points_path=str(dense_ply.relative_to(output_dir)),
    )
