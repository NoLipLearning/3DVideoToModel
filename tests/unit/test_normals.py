"""Camera-center loading and nearest-camera normal orientation.

Pure math against hand-built cameras.json/point-cloud fixtures -- no
COLMAP/sfm_fixture needed, so unlike test_mesh_build.py this stays fast
and unmarked (same posture as test_sparse_only.py's open3d-only tests).
"""

import json

import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation

from v2m.phase3_mesh import normals


def _write_cameras_json(path, images: dict) -> None:
    path.write_text(json.dumps({"cameras": {}, "images": images}))


def test_load_camera_centers_recovers_a_known_position_with_identity_rotation(tmp_path):
    # cam_from_world: p_cam = R @ p_world + t. With R=I, the camera center
    # (where p_cam=0) is simply -t.
    cameras_json = tmp_path / "cameras.json"
    _write_cameras_json(
        cameras_json,
        {
            "000000.jpg": {
                "camera_id": 1,
                "rotation_quat_xyzw": [0.0, 0.0, 0.0, 1.0],
                "translation": [1.0, 2.0, 3.0],
            }
        },
    )
    centers = normals.load_camera_centers(cameras_json)
    assert centers.shape == (1, 3)
    np.testing.assert_allclose(centers[0], [-1.0, -2.0, -3.0])


def test_load_camera_centers_recovers_a_known_position_with_rotation(tmp_path):
    # A non-identity rotation exercises R^T rather than R -- using R
    # instead (an easy sign/transpose slip) would recover the wrong point.
    rotation = Rotation.from_euler("z", 90, degrees=True)
    true_center = np.array([5.0, 0.0, 0.0])
    translation = -rotation.as_matrix() @ true_center

    cameras_json = tmp_path / "cameras.json"
    _write_cameras_json(
        cameras_json,
        {
            "000000.jpg": {
                "camera_id": 1,
                "rotation_quat_xyzw": rotation.as_quat().tolist(),
                "translation": translation.tolist(),
            }
        },
    )
    centers = normals.load_camera_centers(cameras_json)
    np.testing.assert_allclose(centers[0], true_center, atol=1e-9)


def test_load_camera_centers_returns_one_row_per_image(tmp_path):
    cameras_json = tmp_path / "cameras.json"
    _write_cameras_json(
        cameras_json,
        {
            f"{i:06d}.jpg": {
                "camera_id": 1,
                "rotation_quat_xyzw": [0.0, 0.0, 0.0, 1.0],
                "translation": [float(i), 0.0, 0.0],
            }
            for i in range(4)
        },
    )
    centers = normals.load_camera_centers(cameras_json)
    assert centers.shape == (4, 3)


def test_orient_normals_towards_nearest_camera_uses_the_closer_camera():
    # Two points far apart, each near a different camera -- a bug that
    # orients every point towards the *first* or an averaged camera
    # rather than its own nearest one would flip one of these two.
    points = np.array([[-5.0, 0.0, 0.0], [5.0, 0.0, 0.0]])
    wrong_normals = np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    cloud.normals = o3d.utility.Vector3dVector(wrong_normals)

    camera_centers = np.array([[-10.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
    normals.orient_normals_towards_nearest_camera(cloud, camera_centers)

    result = np.asarray(cloud.normals)
    assert result[0][0] < 0  # point at x=-5, nearest camera at x=-10
    assert result[1][0] > 0  # point at x=5, nearest camera at x=10


def test_orient_normals_leaves_already_correct_normals_unchanged():
    points = np.array([[0.0, 0.0, 0.0]])
    correct_normals = np.array([[0.0, 0.0, 1.0]])
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    cloud.normals = o3d.utility.Vector3dVector(correct_normals)

    normals.orient_normals_towards_nearest_camera(cloud, np.array([[0.0, 0.0, 5.0]]))

    np.testing.assert_allclose(np.asarray(cloud.normals), correct_normals)


def test_estimate_and_orient_normals_orients_a_flat_patch_towards_one_camera():
    # PCA-based normal estimation on a flat patch has an arbitrary
    # per-point sign (no consistent "up") until oriented -- a camera
    # straight above should make every resulting normal point +Z.
    rng = np.random.default_rng(0)
    xy = rng.uniform(-1.0, 1.0, size=(200, 2))
    points = np.column_stack([xy, np.zeros(200)])
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)

    normals.estimate_and_orient_normals(cloud, np.array([[0.0, 0.0, 10.0]]))

    result = np.asarray(cloud.normals)
    assert np.all(result[:, 2] > 0)
