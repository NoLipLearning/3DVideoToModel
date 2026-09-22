"""Screened Poisson surface reconstruction + density-quantile crop.

docs/ARCHITECTURE.md Section 4 (M4): "screened Poisson + density-quantile
crop at 0.03." Section 3.2's memory-guard table specifies the adaptive
depth ("<=10 for <2M pts, <=11 above") via what M0 named
`DenseConfig.poisson_point_threshold`/`poisson_depth_low`/
`poisson_depth_high` -- Phase 2b's config, not Phase 3's `MeshConfig`,
matching where the doc's own table put these knobs.
"""

from __future__ import annotations

import numpy as np
import open3d as o3d

from v2m.config import DenseConfig, MeshConfig


def poisson_depth_for_point_count(num_points: int, config: DenseConfig) -> int:
    if num_points < config.poisson_point_threshold:
        return config.poisson_depth_low
    return config.poisson_depth_high


def reconstruct_surface(
    cloud: o3d.geometry.PointCloud, dense_config: DenseConfig, mesh_config: MeshConfig
) -> o3d.geometry.TriangleMesh:
    """Screened Poisson reconstruction, then crop away the low-density
    "bubble" surface Poisson extrapolates beyond the actual point
    support -- vertices below the `poisson_density_quantile_crop`
    quantile of the per-vertex density Poisson itself reports."""
    depth = poisson_depth_for_point_count(len(cloud.points), dense_config)
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(cloud, depth=depth)

    densities = np.asarray(densities)
    threshold = np.quantile(densities, mesh_config.poisson_density_quantile_crop)
    mesh.remove_vertices_by_mask(densities < threshold)

    return mesh
