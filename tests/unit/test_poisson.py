"""Poisson depth selection and reconstruction against synthetic point clouds.

A sphere primitive gives cheap, analytically-normaled point samples, so
(like test_normals.py) this needs no COLMAP/sfm_fixture and stays fast.
"""

import numpy as np
import open3d as o3d

from v2m.config import DenseConfig, MeshConfig
from v2m.phase3_mesh import poisson


def _sample_sphere(num_points: int, radius: float = 1.0) -> o3d.geometry.PointCloud:
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius, resolution=30)
    sphere.compute_vertex_normals()
    return sphere.sample_points_uniformly(number_of_points=num_points)


def test_poisson_depth_for_point_count_uses_low_below_threshold():
    config = DenseConfig(
        poisson_point_threshold=2_000_000, poisson_depth_low=10, poisson_depth_high=11
    )
    assert poisson.poisson_depth_for_point_count(1_000_000, config) == 10


def test_poisson_depth_for_point_count_uses_high_at_threshold():
    config = DenseConfig(
        poisson_point_threshold=2_000_000, poisson_depth_low=10, poisson_depth_high=11
    )
    assert poisson.poisson_depth_for_point_count(2_000_000, config) == 11


def test_poisson_depth_for_point_count_uses_high_above_threshold():
    config = DenseConfig(
        poisson_point_threshold=2_000_000, poisson_depth_low=10, poisson_depth_high=11
    )
    assert poisson.poisson_depth_for_point_count(5_000_000, config) == 11


def test_reconstruct_surface_produces_a_nonempty_mesh_spanning_the_input():
    cloud = _sample_sphere(3000, radius=1.0)
    mesh = poisson.reconstruct_surface(cloud, DenseConfig(), MeshConfig())

    assert len(mesh.vertices) > 0
    assert len(mesh.triangles) > 0

    extent = np.asarray(mesh.vertices).max(axis=0) - np.asarray(mesh.vertices).min(axis=0)
    # A radius-1 sphere is ~2.0 across; Poisson's extrapolated-then-cropped
    # surface should stay in the same ballpark, not collapse or balloon.
    assert np.all(extent > 1.5)
    assert np.all(extent < 2.5)


def test_higher_density_crop_quantile_keeps_fewer_vertices():
    cloud = _sample_sphere(2000, radius=1.0)
    low_crop = poisson.reconstruct_surface(
        cloud, DenseConfig(), MeshConfig(poisson_density_quantile_crop=0.0)
    )
    high_crop = poisson.reconstruct_surface(
        cloud, DenseConfig(), MeshConfig(poisson_density_quantile_crop=0.5)
    )
    assert len(high_crop.vertices) < len(low_crop.vertices)
