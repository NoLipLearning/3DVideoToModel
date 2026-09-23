"""Ground-plane detection, gravity disambiguation, canonical re-orientation.

docs/ARCHITECTURE.md Section 3.3: find the plane, disambiguate against a
gravity prior, re-orient the whole scene so the ground normal is +Z and
the plane sits at z=0. Every downstream Phase 4 step (watertight.py's
half-space cut / slab union, validate.py's bbox check) assumes this
canonical frame.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import open3d as o3d
import trimesh
from scipy.spatial.transform import Rotation

from v2m.config import PrintPrepConfig

logger = logging.getLogger("v2m.phase4_print.ground")

_MAX_PLANE_CANDIDATES = 5


@dataclass
class GroundPlane:
    normal: np.ndarray  # unit vector, oriented to agree with the gravity prior
    point: np.ndarray  # a point on the plane
    inlier_ratio: float  # 0.0 for a fallback plane (no RANSAC support behind it)
    gravity_agreement: float  # |dot(normal, gravity_up)|
    source: str  # "ransac", "fallback_object", or "fallback_scene"
    tolerance: float  # RANSAC inlier distance used, in the cloud's own units


def compute_gravity_up(cameras_json_path: Path) -> np.ndarray:
    """The world-space "up" direction implied by how the camera was held,
    averaged across every registered image (Section 3.3: "gravity prior
    comes from the median camera 'up' vector across poses -- people hold
    cameras roughly upright"). A mean-then-renormalize stands in for a
    true geometric median here: cheap, and for an orbiting capture the
    per-frame tilts largely cancel (this project's fixture bobs the
    camera up to ~29 degrees of elevation either way over its orbit, and
    the mean still lands within ~1 degree of true vertical), without
    needing a Weiszfeld-style iterative solve for a single averaged
    direction vector.

    `cam_from_world` maps `p_cam = R @ p_world + t`, so a *direction* (no
    translation) in camera space maps to world space via `R^T @
    direction_cam`. OpenCV's camera convention is X-right, Y-down,
    Z-forward, so the camera's own "up" in its local frame is `-Y`, i.e.
    `[0, -1, 0]`.
    """
    data = json.loads(cameras_json_path.read_text())
    ups = []
    for image in data["images"].values():
        rotation = Rotation.from_quat(image["rotation_quat_xyzw"]).as_matrix()
        ups.append(rotation.T @ np.array([0.0, -1.0, 0.0]))
    mean_up = np.mean(ups, axis=0)
    return mean_up / np.linalg.norm(mean_up)


def camera_center_from_pose(
    rotation_quat_xyzw: list[float], translation: list[float]
) -> np.ndarray:
    """World-space camera center for one `cameras.json` image entry.
    `cam_from_world` maps `p_cam = R @ p_world + t`; the camera center is
    where `p_cam = 0`, i.e. `C = -R^T @ t`."""
    rotation = Rotation.from_quat(rotation_quat_xyzw).as_matrix()
    return -rotation.T @ np.array(translation, dtype=np.float64)


def load_camera_center(cameras_json_path: Path, image_name: str) -> np.ndarray:
    """`camera_center_from_pose`, looked up by image name from a
    `cameras.json` file (scale.py's ArUco wiring needs one specific
    image's camera center, not every image's like `compute_gravity_up`)."""
    data = json.loads(cameras_json_path.read_text())
    image = data["images"][image_name]
    return camera_center_from_pose(image["rotation_quat_xyzw"], image["translation"])


def _signed_distances(
    points: np.ndarray, normal: np.ndarray, point_on_plane: np.ndarray
) -> np.ndarray:
    return (points - point_on_plane) @ normal


def _fallback_object(
    points: np.ndarray, gravity_up: np.ndarray, min_inlier_ratio: float, tolerance: float
) -> GroundPlane:
    """Section 3.3's fallback when no plane clears `ground_min_inlier_ratio`:
    "cutting at the 2nd-percentile Z of the main component" -- using the
    gravity prior itself as the ground normal (still the best available
    "up" reference) and a robust near-minimum, rather than the true min,
    of the point cloud's extent along it as the plane's height."""
    heights = points @ gravity_up
    offset = np.percentile(heights, 2.0)
    logger.warning(
        "No ground plane reached the %.0f%% inlier-ratio bar; falling back to the gravity "
        "prior's direction and the 2nd-percentile height of the point cloud as the base.",
        min_inlier_ratio * 100.0,
    )
    return GroundPlane(
        normal=gravity_up,
        point=offset * gravity_up,
        inlier_ratio=0.0,
        gravity_agreement=1.0,
        source="fallback_object",
        tolerance=tolerance,
    )


def _fallback_scene(
    cloud: o3d.geometry.PointCloud,
    gravity_up: np.ndarray,
    min_inlier_ratio: float,
    tolerance: float,
) -> GroundPlane:
    """Section 3.3's scene-mode fallback: "the oriented-bounding-box
    minimum face" -- whichever OBB axis agrees best with the gravity
    prior is treated as vertical, and the face at its low end is the
    ground."""
    obb = cloud.get_oriented_bounding_box()
    axes = np.asarray(obb.R)  # columns are the OBB's local axes in world space
    alignments = [abs(float(np.dot(axes[:, i], gravity_up))) for i in range(3)]
    vertical_axis = int(np.argmax(alignments))

    axis_dir = axes[:, vertical_axis]
    if np.dot(axis_dir, gravity_up) < 0:
        axis_dir = -axis_dir

    half_extent = obb.extent[vertical_axis] / 2.0
    bottom_point = np.asarray(obb.center) - half_extent * axis_dir
    logger.warning(
        "No ground plane reached the %.0f%% inlier-ratio bar; falling back to the oriented "
        "bounding box's face most aligned with the gravity prior.",
        min_inlier_ratio * 100.0,
    )
    return GroundPlane(
        normal=axis_dir,
        point=bottom_point,
        inlier_ratio=0.0,
        gravity_agreement=alignments[vertical_axis],
        source="fallback_scene",
        tolerance=tolerance,
    )


def find_ground_plane(
    cloud: o3d.geometry.PointCloud,
    gravity_up: np.ndarray,
    config: PrintPrepConfig,
    mode: str,
) -> GroundPlane:
    """RANSAC-segments up to `_MAX_PLANE_CANDIDATES` candidate planes
    (iteratively removing each found plane's inliers before searching
    for the next, since a single RANSAC pass only ever finds the single
    largest one and a capture can have several comparably-sized planar
    regions -- e.g. a wall behind the actual ground). Each candidate is
    scored by inlier ratio, agreement with the gravity prior, and how
    much of the *entire* cloud lies on one side of it (Section 3.3's
    three disambiguation criteria).

    The winner needs *both* `ground_min_inlier_ratio` and
    `ground_min_gravity_agreement` to be trusted, else `mode`'s fallback
    applies. The doc's original wording only conditions the fallback on
    inlier ratio; gravity agreement turned out to matter too, discovered
    against this project's own cube fixture (an object with no real
    floor in view -- see tests/fixtures/make_synthetic_video.py): every
    RANSAC candidate there is a side face of comparable size to the
    others, all scoring alike on inlier ratio while sharing a near-zero
    gravity agreement (~0.005) -- since the scoring is a product,
    picking the highest-scoring candidate still surfaces a real plane
    with a large, confident inlier ratio, just one that is clearly not
    "the ground" once you look at which way it faces. Requiring a
    minimum gravity agreement independently of the score catches
    exactly this case.

    Section 3.3's RANSAC distance is "2x voxel". The voxel meant is the
    dense cloud's own sampling pitch, measured here as its median
    nearest-neighbour spacing rather than read from
    `DenseConfig.tsdf_voxel_size_m`: that value is in nominal metres, and
    at this point in the pipeline the cloud is still in COLMAP's
    arbitrary units (same scale-ambiguity issue as monodepth_tsdf.py's
    `_build_tsdf_volume`), so using it directly would size the inlier
    band wrong by whatever factor COLMAP's unit differs from a metre.
    """
    points = np.asarray(cloud.points)
    spacing = float(np.median(np.asarray(cloud.compute_nearest_neighbor_distance())))
    distance_threshold = config.ground_ransac_distance_multiplier * spacing

    remaining = cloud
    candidates: list[tuple[np.ndarray, np.ndarray, int]] = []
    for _ in range(_MAX_PLANE_CANDIDATES):
        if len(remaining.points) < config.ground_ransac_n:
            break
        plane_model, inlier_idx = remaining.segment_plane(
            distance_threshold, config.ground_ransac_n, config.ground_ransac_iterations
        )
        if len(inlier_idx) < config.ground_ransac_n:
            break
        a, b, c, d = plane_model
        normal = np.array([a, b, c])
        normal = normal / np.linalg.norm(normal)
        point_on_plane = -d * normal
        candidates.append((normal, point_on_plane, len(inlier_idx)))
        remaining = remaining.select_by_index(inlier_idx, invert=True)

    best: GroundPlane | None = None
    best_score = -1.0
    for normal, point_on_plane, inlier_count in candidates:
        if np.dot(normal, gravity_up) < 0:
            normal = -normal

        inlier_ratio = inlier_count / len(points)
        gravity_agreement = abs(float(np.dot(normal, gravity_up)))
        side_balance = float(np.mean(_signed_distances(points, normal, point_on_plane) >= 0.0))
        score = inlier_ratio * gravity_agreement * side_balance

        if score > best_score:
            best_score = score
            # Re-center the in-plane reference point at the cloud
            # centroid's own projection, purely for a tidier canonical
            # origin -- doesn't change which plane this is.
            centroid = points.mean(axis=0)
            centered_point = centroid - float(np.dot(centroid - point_on_plane, normal)) * normal
            best = GroundPlane(
                normal=normal,
                point=centered_point,
                inlier_ratio=inlier_ratio,
                gravity_agreement=gravity_agreement,
                source="ransac",
                tolerance=distance_threshold,
            )

    if (
        best is not None
        and best.inlier_ratio >= config.ground_min_inlier_ratio
        and best.gravity_agreement >= config.ground_min_gravity_agreement
    ):
        return best

    if mode == "scene":
        return _fallback_scene(
            cloud, gravity_up, config.ground_min_inlier_ratio, distance_threshold
        )
    return _fallback_object(points, gravity_up, config.ground_min_inlier_ratio, distance_threshold)


def canonical_transform(ground_plane: GroundPlane) -> np.ndarray:
    """4x4 transform mapping world space into the canonical frame: the
    ground plane's normal becomes +Z and its own reference point maps to
    the origin (so the plane sits at z=0)."""
    rotation, _ = Rotation.align_vectors([[0.0, 0.0, 1.0]], [ground_plane.normal])
    rotation_matrix = rotation.as_matrix()

    transform = np.eye(4)
    transform[:3, :3] = rotation_matrix
    transform[:3, 3] = -rotation_matrix @ ground_plane.point
    return transform


def place_on_bed(mesh: trimesh.Trimesh) -> None:
    """Rotate `mesh` about Z so its footprint's minimum-area bounding
    rectangle lines up with X/Y, then translate it to sit centered on
    the XY origin with its lowest point at z=0. Mutates in place.

    Ground-plane detection only fixes which way is *up*; the in-plane
    heading stays whatever COLMAP's arbitrary frame left it at. A cube
    at 45 degrees reports a ~41% larger axis-aligned footprint than its
    true size (seen directly on this project's cube fixture: 282 x 285
    mm for a 200mm cube), which would misstate the size in
    print_report.json and trip validate.py's build-volume check on a
    model that actually fits.
    """
    to_rect_2d, _extents = trimesh.bounds.oriented_bounds_2D(mesh.vertices[:, :2])
    rotation = np.eye(4)
    rotation[:2, :2] = to_rect_2d[:2, :2]
    mesh.apply_transform(rotation)

    bounds = mesh.bounds
    center_xy = (bounds[0, :2] + bounds[1, :2]) / 2.0
    mesh.apply_translation([-center_xy[0], -center_xy[1], -bounds[0, 2]])
