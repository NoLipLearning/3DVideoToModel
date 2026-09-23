"""Optional dense backend (M9): OpenMVS multi-view stereo, on the CPU.

docs/ARCHITECTURE.md Section 2: OpenMVS is an external C++ program, not
pip-installable, and strictly optional. `DensifyPointCloud --cuda-device
-2` forces its CPU path, which is what makes it usable on Apple Silicon
at all, unlike COLMAP's CUDA-only patch_match_stereo (never called
anywhere in this project).

Opt-in only (`--set dense.backend=openmvs`). "auto" never picks it, even
when the binary is installed. Real MVS is better than monocular depth
on well-textured surfaces and worse on blank ones, and blank surfaces
are the failure this pipeline's default exists to handle (Section 3.1).

Two OpenMVS tools run against the undistorted COLMAP workspace Phase 2a
already writes (`sfm/undistorted/{images,sparse}`, COLMAP's standard
dense-workspace layout, which OpenMVS reads directly):

    InterfaceCOLMAP   -i sfm/undistorted -o scene.mvs --image-folder .../images
    DensifyPointCloud scene.mvs -o scene_dense.mvs --cuda-device -2

DensifyPointCloud writes `scene_dense.ply` next to its .mvs output. That
PLY becomes `dense/dense.ply`, the same contract as every other backend.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path

import open3d as o3d

from v2m.config import DenseConfig
from v2m.errors import CapabilityError, SfMError
from v2m.types import DenseResult

logger = logging.getLogger("v2m.phase2_sfm.dense.openmvs")

_TOOLS = ("InterfaceCOLMAP", "DensifyPointCloud")


def find_tools() -> dict[str, str] | None:
    found = {tool: shutil.which(tool) for tool in _TOOLS}
    return found if all(found.values()) else None


def _run(command: list[str], workdir: Path, step: str) -> None:
    logger.info("OpenMVS: %s", " ".join(command))
    result = subprocess.run(command, cwd=workdir, capture_output=True, text=True)
    (workdir / f"{step}.log").write_text(result.stdout + result.stderr)
    if result.returncode != 0:
        tail = "\n".join((result.stdout + result.stderr).strip().splitlines()[-8:])
        raise SfMError(
            f"OpenMVS {step} failed (exit {result.returncode}):\n{tail}",
            remedy=f"See {workdir / (step + '.log')}. The default backend avoids OpenMVS "
            "entirely: --set dense.backend=monodepth_tsdf.",
        )


def densify(sfm_dir: Path, output_dir: Path, config: DenseConfig) -> DenseResult:
    tools = find_tools()
    if tools is None:
        raise CapabilityError(
            "dense.backend=openmvs, but OpenMVS (InterfaceCOLMAP + DensifyPointCloud) isn't "
            "on PATH.",
            remedy="Build OpenMVS from https://github.com/cdcseacave/openMVS (cmake), or use "
            "the default backend: --set dense.backend=monodepth_tsdf.",
        )
    workspace = sfm_dir / "undistorted"
    if not (workspace / "sparse").exists() or not (workspace / "images").exists():
        raise SfMError(
            f"No undistorted COLMAP workspace in {workspace}.",
            remedy="Run `v2m sfm` first.",
        )

    workdir = output_dir / "openmvs"
    if workdir.exists():
        shutil.rmtree(workdir)
    workdir.mkdir(parents=True)

    _run(
        [
            tools["InterfaceCOLMAP"],
            "-i",
            str(workspace.resolve()),
            "-o",
            "scene.mvs",
            "--image-folder",
            str((workspace / "images").resolve()),
        ],
        workdir,
        "interface",
    )
    _run(
        [
            tools["DensifyPointCloud"],
            "scene.mvs",
            "-o",
            "scene_dense.mvs",
            "--cuda-device",
            str(config.openmvs_cuda_device),
        ],
        workdir,
        "densify",
    )

    produced = workdir / "scene_dense.ply"
    cloud = o3d.io.read_point_cloud(str(produced)) if produced.exists() else None
    if cloud is None or len(cloud.points) == 0:
        raise SfMError(
            "OpenMVS finished but produced no dense points.",
            remedy=f"See {workdir}/densify.log; low-texture captures suit the default "
            "monodepth_tsdf backend better.",
        )
    dense_ply = output_dir / "dense.ply"
    o3d.io.write_point_cloud(str(dense_ply), cloud)
    logger.info("OpenMVS dense cloud: %d points.", len(cloud.points))
    return DenseResult(
        backend="openmvs",
        num_points=len(cloud.points),
        dense_points_path=str(dense_ply.relative_to(output_dir)),
    )
