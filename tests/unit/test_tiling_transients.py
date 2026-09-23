"""M9: TSDF tiling geometry and moving-object suppression, on
hand-built geometry (fast). The end-to-end tiled densify on the fixture
is in test_m9_backends.py (slow)."""

import numpy as np
import pytest

from v2m.config import DenseConfig
from v2m.phase2_sfm.dense import tiling, transients


def test_a_small_scene_is_one_tile():
    assert tiling.tile_counts(
        np.array([1.0, 1.0, 1.0]), voxel=0.01, max_resolution=400, max_tiles=27
    ) == [1, 1, 1]


def test_a_long_scene_is_tiled_along_its_long_axis():
    # 10 units long at 0.01/voxel = 1000 voxels > 400 -> 3 cells along x.
    counts = tiling.tile_counts(
        np.array([10.0, 2.0, 1.0]), voxel=0.01, max_resolution=400, max_tiles=27
    )
    assert counts == [3, 1, 1]


def test_tile_count_is_capped_by_shrinking_the_largest_axis():
    counts = tiling.tile_counts(
        np.array([40.0, 40.0, 1.0]), voxel=0.01, max_resolution=400, max_tiles=27
    )
    assert np.prod(counts) <= 27
    assert counts[2] == 1


def test_tile_cores_partition_space_without_gaps_or_overlaps():
    config = DenseConfig(max_tiles=27)
    tiles = tiling.plan_tiles(
        np.zeros(3),
        np.array([10.0, 5.0, 1.0]),
        voxel=0.01,
        config=config,
        max_resolution=400,
        margin_voxels=8,
    )
    assert len(tiles) == 3 * 2 * 1
    rng = np.random.default_rng(0)
    # Including points slightly outside the box: edge cells own them too.
    points = rng.uniform([-0.5, -0.5, -0.5], [10.5, 5.5, 1.5], size=(20000, 3))
    owners = np.sum([tile.contains(points) for tile in tiles], axis=0)
    assert np.all(owners == 1)
    for tile in tiles:
        assert tile.resolution <= 400
        # Each volume covers its own core plus a margin.
        finite_core_min = np.where(np.isfinite(tile.core_min), tile.core_min, -np.inf)
        assert np.all(tile.origin <= np.maximum(finite_core_min, tile.origin))


def test_desired_voxel_is_a_ratio_of_the_capture_distance():
    object_cfg = DenseConfig()  # 4mm at 0.5m
    # Units cancel: the same capture in "COLMAP units" 3x larger wants a 3x voxel.
    assert tiling.desired_voxel(3.0, object_cfg) == pytest.approx(
        3 * tiling.desired_voxel(1.0, object_cfg)
    )
    assert tiling.desired_voxel(0.5, object_cfg) == pytest.approx(0.004)


def test_sample_world_points_inverts_the_projection():
    intrinsic = np.array([[100.0, 0, 32], [0, 100.0, 24], [0, 0, 1]])
    cam_from_world = np.eye(4)
    cam_from_world[:3, 3] = [0.0, 0.0, 2.0]  # world origin 2 units in front of the camera
    depth = np.full((48, 64), 2.0)
    points = tiling.sample_world_points(depth, intrinsic, cam_from_world, stride=4)
    np.testing.assert_allclose(points[:, 2], 0.0, atol=1e-9)  # the plane z=0 in world


# -- transient suppression -------------------------------------------------------------


def _views(box_present: list[bool]):
    """Cameras along x looking down +z at a wall at z=10; in some frames a
    box stands at z=6 in front of the wall around x in [-1, 1]."""
    intrinsic = np.array([[200.0, 0, 100], [0, 200.0, 100], [0, 0, 1]])
    views = []
    for i, present in enumerate(box_present):
        cam_x = -2.0 + 4.0 * i / (len(box_present) - 1)
        cam_from_world = np.eye(4)
        cam_from_world[:3, 3] = [-cam_x, 0.0, 0.0]
        rows, cols = np.mgrid[0:200, 0:200]
        # World x of the wall and box points under each pixel.
        depth = np.full((200, 200), 10.0)
        if present:
            x_at_box = (cols - 100) * 6.0 / 200.0 + cam_x
            y_at_box = (rows - 100) * 6.0 / 200.0
            on_box = (np.abs(x_at_box) <= 1.0) & (np.abs(y_at_box) <= 1.0)
            depth[on_box] = 6.0
        views.append(transients.DepthView(depth, intrinsic, cam_from_world))
    return views


def test_a_surface_seen_through_by_most_frames_is_removed():
    wall = np.column_stack([np.linspace(-3, 3, 50), np.zeros(50), np.full(50, 10.0)])
    box = np.column_stack([np.linspace(-0.8, 0.8, 20), np.zeros(20), np.full(20, 6.0)])
    points = np.vstack([wall, box])
    present = [True] * 4 + [False] * 16  # the box was there for 4 of 20 frames

    mask = transients.transient_mask(points, _views(present), tolerance=0.3, min_views=2)

    assert mask[len(wall) :].all()  # the box goes
    assert not mask[: len(wall)].any()  # the wall stays -- occluded, never seen through


def test_a_static_object_is_kept():
    box = np.column_stack([np.linspace(-0.8, 0.8, 20), np.zeros(20), np.full(20, 6.0)])
    mask = transients.transient_mask(box, _views([True] * 20), tolerance=0.3, min_views=2)
    assert not mask.any()
