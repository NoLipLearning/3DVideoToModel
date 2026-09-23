"""ground.py: gravity prior, ground-plane detection/disambiguation,
canonical re-orientation, and bed placement -- against hand-built point
clouds and cameras.json files, so fast and unmarked."""

import json

import numpy as np
import open3d as o3d
import pytest
import trimesh
from scipy.spatial.transform import Rotation

from v2m.config import PrintPrepConfig
from v2m.phase4_print import ground

UP = np.array([0.0, 0.0, 1.0])


def _cloud(points: np.ndarray) -> o3d.geometry.PointCloud:
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    return cloud


def _grid_plane(size: float, spacing: float, axis: int, offset: float) -> np.ndarray:
    ticks = np.arange(-size / 2, size / 2, spacing)
    u, v = np.meshgrid(ticks, ticks)
    columns = [u.ravel(), v.ravel()]
    columns.insert(axis, np.full(u.size, offset))
    return np.column_stack(columns)


def _box_surface(center, extents, count, seed=0) -> np.ndarray:
    box = trimesh.creation.box(extents=extents)
    box.apply_translation(center)
    points, _ = trimesh.sample.sample_surface(box, count, seed=seed)
    return np.asarray(points)


def _write_cameras(path, rotations) -> None:
    images = {
        f"{i:06d}.jpg": {
            "camera_id": 1,
            "rotation_quat_xyzw": rotation.as_quat().tolist(),
            "translation": [0.0, 0.0, 5.0],
        }
        for i, rotation in enumerate(rotations)
    }
    path.write_text(json.dumps({"cameras": {}, "images": images}))


def test_compute_gravity_up_for_an_upright_camera(tmp_path):
    # An OpenCV camera with R=I looks down world +Z with image-down along
    # world +Y, so its "up" is world -Y.
    cameras = tmp_path / "cameras.json"
    _write_cameras(cameras, [Rotation.identity()])
    np.testing.assert_allclose(ground.compute_gravity_up(cameras), [0.0, -1.0, 0.0], atol=1e-9)


def test_compute_gravity_up_averages_out_symmetric_tilts(tmp_path):
    # Cameras pitched +/-20 degrees about the view's own X axis: each "up"
    # is tilted, but the tilts cancel in the average.
    cameras = tmp_path / "cameras.json"
    _write_cameras(
        cameras,
        [Rotation.from_euler("x", 20, degrees=True), Rotation.from_euler("x", -20, degrees=True)],
    )
    np.testing.assert_allclose(ground.compute_gravity_up(cameras), [0.0, -1.0, 0.0], atol=1e-9)


def test_camera_center_from_pose_inverts_cam_from_world():
    rotation = Rotation.from_euler("z", 90, degrees=True)
    center = np.array([1.0, 2.0, 3.0])
    translation = -rotation.as_matrix() @ center
    np.testing.assert_allclose(
        ground.camera_center_from_pose(rotation.as_quat().tolist(), translation.tolist()), center
    )


def test_find_ground_plane_finds_the_floor_under_an_object():
    floor = _grid_plane(size=4.0, spacing=0.02, axis=2, offset=0.0)
    box = _box_surface(center=[0.0, 0.0, 0.5], extents=[1.0, 1.0, 1.0], count=4000)
    plane = ground.find_ground_plane(
        _cloud(np.vstack([floor, box])), UP, PrintPrepConfig(), "object"
    )

    assert plane.source == "ransac"
    np.testing.assert_allclose(plane.normal, UP, atol=1e-3)
    assert abs(plane.point[2]) < 1e-3
    assert plane.tolerance > 0


def test_find_ground_plane_prefers_the_floor_over_a_larger_wall():
    # A wall with more points than the floor: largest-plane-wins RANSAC
    # would pick it, but it faces sideways -- the gravity prior must win.
    floor = _grid_plane(size=3.0, spacing=0.03, axis=2, offset=0.0)
    wall = _grid_plane(size=4.0, spacing=0.02, axis=0, offset=-2.0)
    wall[:, 2] += 2.0  # stand it on the floor
    plane = ground.find_ground_plane(
        _cloud(np.vstack([floor, wall])), UP, PrintPrepConfig(), "object"
    )

    assert plane.source == "ransac"
    np.testing.assert_allclose(plane.normal, UP, atol=1e-3)


@pytest.mark.parametrize(
    ("mode", "source"), [("object", "fallback_object"), ("scene", "fallback_scene")]
)
def test_find_ground_plane_falls_back_without_a_gravity_aligned_plane(mode, source):
    # A free-floating box balanced on a corner (body diagonal vertical):
    # every face normal sits 54.7 degrees off vertical, a gravity
    # agreement of 0.577 -- below the 0.7 bar, however many points each
    # face has.
    points = _box_surface(center=[0.0, 0.0, 0.0], extents=[1.0, 1.0, 1.0], count=6000)
    on_corner, _ = Rotation.align_vectors([[0.0, 0.0, 1.0]], [[1.0, 1.0, 1.0]])
    points = on_corner.apply(points)
    plane = ground.find_ground_plane(_cloud(points), UP, PrintPrepConfig(), mode)
    assert plane.source == source


def test_canonical_transform_maps_plane_normal_to_z_and_point_to_origin():
    normal = np.array([0.3, -0.2, 0.9])
    normal /= np.linalg.norm(normal)
    plane = ground.GroundPlane(
        normal=normal,
        point=np.array([1.0, 2.0, 3.0]),
        inlier_ratio=0.5,
        gravity_agreement=1.0,
        source="ransac",
        tolerance=0.01,
    )
    transform = ground.canonical_transform(plane)
    np.testing.assert_allclose(transform[:3, :3] @ normal, UP, atol=1e-9)
    np.testing.assert_allclose(transform @ np.append(plane.point, 1.0), [0, 0, 0, 1], atol=1e-9)


def test_place_on_bed_aligns_footprint_and_drops_to_z_zero():
    box = trimesh.creation.box(extents=[4.0, 2.0, 1.0])
    box.apply_transform(trimesh.transformations.rotation_matrix(np.radians(37), UP))
    box.apply_translation([10.0, -5.0, 3.0])

    ground.place_on_bed(box)

    np.testing.assert_allclose(sorted(box.extents), [1.0, 2.0, 4.0], atol=1e-6)
    assert box.bounds[0][2] == pytest.approx(0.0, abs=1e-9)
    np.testing.assert_allclose((box.bounds[0][:2] + box.bounds[1][:2]) / 2, [0.0, 0.0], atol=1e-6)
