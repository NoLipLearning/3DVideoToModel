"""Camera-aware normal orientation.

docs/ARCHITECTURE.md Section 4 (M4): "orient toward the camera that
observed each point -- far more reliable than tangent-plane
propagation." Neither dense backend's output (dense.ply, from TSDF
fusion or the sparse_only copy) carries a per-point "which camera saw
this" tag, so the practical form of "the camera that observed it" is
each point's *nearest* reconstructed camera center -- for a
full-coverage orbiting capture that's very likely a camera that actually
saw it, and unlike open3d's tangent-plane MST propagation
(`orient_normals_consistent_tangent_plane`), it can't flip on
thin/closely-spaced surfaces since every point is judged independently
against known camera geometry rather than its neighbors' (possibly
already-wrong) orientation.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation


def load_camera_centers(cameras_json_path: Path) -> np.ndarray:
    """World-space camera centers from sfm/cameras.json.

    `cam_from_world` maps `p_cam = R @ p_world + t` (sfm.py's
    `_export_cameras_json` docstring), so the camera center in world
    space (where `p_cam = 0`) is `C = -R^T @ t`. Rotation is stored as an
    (x, y, z, w) quaternion -- COLMAP/Eigen's convention, which is also
    exactly `scipy.spatial.transform.Rotation.from_quat`'s default order.
    """
    data = json.loads(cameras_json_path.read_text())
    centers = []
    for image in data["images"].values():
        rotation = Rotation.from_quat(image["rotation_quat_xyzw"]).as_matrix()
        translation = np.array(image["translation"], dtype=np.float64)
        centers.append(-rotation.T @ translation)
    return np.array(centers, dtype=np.float64)


def orient_normals_towards_nearest_camera(
    cloud: o3d.geometry.PointCloud, camera_centers: np.ndarray
) -> None:
    """Flip each point's normal to face its nearest camera center.

    Mutates `cloud` in place, matching open3d's own
    `orient_normals_*` method conventions.
    """
    points = np.asarray(cloud.points)
    normals = np.asarray(cloud.normals)

    tree = cKDTree(camera_centers)
    _, nearest_idx = tree.query(points)
    to_camera = camera_centers[nearest_idx] - points

    flip_mask = np.einsum("ij,ij->i", normals, to_camera) < 0.0
    normals[flip_mask] *= -1.0
    cloud.normals = o3d.utility.Vector3dVector(normals)


def estimate_and_orient_normals(cloud: o3d.geometry.PointCloud, camera_centers: np.ndarray) -> None:
    """Estimate unoriented normals (open3d's own default: 30-nearest-
    neighbor PCA, scale-free -- no absolute radius to tune per capture),
    then orient them per `orient_normals_towards_nearest_camera`."""
    cloud.estimate_normals()
    orient_normals_towards_nearest_camera(cloud, camera_centers)
