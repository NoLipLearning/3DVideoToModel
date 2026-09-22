"""Phase 3: point cloud -> raw surface mesh (screened Poisson).

Implemented in M4. See docs/ARCHITECTURE.md Section 4.

`build_mesh()` is this package's orchestrator: normals.py's camera-aware
orientation, poisson.py's screened reconstruction + density-quantile
crop, cleanup.py's component/floater removal and face-budget decimation,
in that order. Kept here (like `dense/__init__.py`'s backend dispatch)
rather than a `sfm.py`-style sibling file -- unlike Phase 2's
multi-backend dispatch, there's exactly one pipeline here, nothing to
choose between.
"""

from __future__ import annotations

from pathlib import Path

import open3d as o3d

from v2m.config import DenseConfig, MeshConfig
from v2m.errors import MeshError
from v2m.phase3_mesh import cleanup, normals, poisson
from v2m.types import MeshReport


def build_mesh(
    dense_dir: Path,
    sfm_dir: Path,
    output_dir: Path,
    dense_config: DenseConfig,
    mesh_config: MeshConfig,
) -> MeshReport:
    """Writes `<output_dir>/{raw.ply, cleaned.ply}`. `raw.ply` is the
    Poisson output straight after the density crop, before component/
    floater removal and decimation; `cleaned.ply` (this phase's
    deliverable, per the M4 verify step) is after both."""
    dense_ply = dense_dir / "dense.ply"
    cameras_json = sfm_dir / "cameras.json"
    if not dense_ply.exists():
        raise MeshError(
            f"No dense point cloud found at {dense_ply}.",
            remedy="Run `v2m dense` first to produce dense/dense.ply.",
        )
    if not cameras_json.exists():
        raise MeshError(
            f"No camera poses found at {cameras_json}.",
            remedy="Run `v2m sfm` first to produce sfm/cameras.json.",
        )

    cloud = o3d.io.read_point_cloud(str(dense_ply))
    if len(cloud.points) == 0:
        raise MeshError(
            f"{dense_ply} has no points.",
            remedy="Re-run `v2m dense` and check dense/alignment_report.json for per-image "
            "failures -- the capture may be too weak for dense reconstruction.",
        )

    # Section 3.2's memory guard: voxel-downsample before Poisson, at the
    # same voxel size Phase 2b's TSDF volume used (a no-op in practice
    # for TSDF-fused input, which is already on that grid; it matters
    # for sparse_only's un-downsampled copy of the sparse cloud).
    cloud = cloud.voxel_down_sample(dense_config.tsdf_voxel_size_m)

    camera_centers = normals.load_camera_centers(cameras_json)
    if len(camera_centers) == 0:
        raise MeshError(
            f"{cameras_json} has no registered images.",
            remedy="Run `v2m sfm` first to produce a valid sparse reconstruction.",
        )
    normals.estimate_and_orient_normals(cloud, camera_centers)

    mesh = poisson.reconstruct_surface(cloud, dense_config, mesh_config)

    output_dir.mkdir(parents=True, exist_ok=True)
    raw_mesh_path = output_dir / "raw.ply"
    o3d.io.write_triangle_mesh(str(raw_mesh_path), mesh)

    mesh = cleanup.clean_mesh(mesh, mesh_config)
    if len(mesh.triangles) == 0:
        raise MeshError(
            "Poisson reconstruction + cleanup produced an empty mesh.",
            remedy=f"Inspect {raw_mesh_path} -- if it has geometry but cleaned.ply doesn't, "
            "min_component_volume_ratio is likely too strict for this capture.",
        )

    cleaned_mesh_path = output_dir / "cleaned.ply"
    o3d.io.write_triangle_mesh(str(cleaned_mesh_path), mesh)

    return MeshReport(
        num_vertices=len(mesh.vertices),
        num_faces=len(mesh.triangles),
        is_manifold=mesh.is_edge_manifold() and mesh.is_vertex_manifold(),
        mesh_path=str(cleaned_mesh_path.relative_to(output_dir)),
    )
