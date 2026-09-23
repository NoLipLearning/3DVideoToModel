"""Metric scale resolution: SfM is scale-blind, so a real-world size
has to come from somewhere else.

docs/ARCHITECTURE.md Section 3.4, in priority order (first success
wins): (1) an ArUco marker of known size, (2) a user-supplied real
distance between two points, (3) a manual scale factor, (4) fit the
longest axis to a target size (dimensionally meaningless, flagged as
such). Every function here returns a uniform scale factor: multiply the
mesh's own (COLMAP-arbitrary-unit) coordinates by it to get millimetres.
"""

from __future__ import annotations

import cv2
import numpy as np
import trimesh

_ARUCO_OBJECT_POINTS_UNIT = np.array(
    [
        [-0.5, 0.5, 0.0],
        [0.5, 0.5, 0.0],
        [0.5, -0.5, 0.0],
        [-0.5, -0.5, 0.0],
    ]
)  # cv2.aruco's own corner order (clockwise from top-left); scaled by marker_length_mm below


def resolve_scale_aruco(
    image_bgr: np.ndarray,
    camera_matrix: np.ndarray,
    marker_length_mm: float,
    colmap_distance_mm: float,
    aruco_dict: int = cv2.aruco.DICT_4X4_50,
) -> float | None:
    """Detects an ArUco marker in `image_bgr`, recovers its *metric*
    camera-to-marker distance via `solvePnP` (using the known real
    `marker_length_mm`), and divides by `colmap_distance_mm` -- the same
    camera's distance to some nearby reconstructed 3D point, in COLMAP's
    own arbitrary units -- to get a scale factor. Returns `None` if no
    marker is found. `aruco_dict` defaults to `cv2.aruco.DICT_4X4_50`
    (id 0); pass a different `cv2.aruco.DICT_*` id if the printed marker
    uses another dictionary.

    Caller supplies `colmap_distance_mm` (badly named for a not-yet-
    metric quantity, but keeping symmetry with the "_mm" naming
    elsewhere): it depends on the reconstruction, and computing it here
    would require this function to know about pycolmap's Reconstruction/
    Image types, which nothing else in this module needs.

    UNVERIFIED end-to-end against a real capture: this project's own
    synthetic fixture has no marker in it (rendering one in would need
    real, in-scene 3D placement, not just an image overlay), so this is
    checked here only against the isolated math -- a synthetic
    projectPoints/solvePnP round-trip (tests/unit/test_scale.py) -- not
    the full detect-a-real-marker-in-a-real-photo path.
    """
    dictionary = cv2.aruco.getPredefinedDictionary(aruco_dict)
    detector = cv2.aruco.ArucoDetector(dictionary)
    corners, ids, _ = detector.detectMarkers(image_bgr)
    if ids is None or len(ids) == 0:
        return None

    object_points = _ARUCO_OBJECT_POINTS_UNIT * marker_length_mm
    success, _rvec, tvec = cv2.solvePnP(object_points, corners[0][0], camera_matrix, np.zeros(5))
    if not success:
        return None

    metric_distance_mm = float(np.linalg.norm(tvec))
    if colmap_distance_mm <= 0:
        return None
    return metric_distance_mm / colmap_distance_mm


def resolve_scale_two_point(
    point_a: np.ndarray, point_b: np.ndarray, real_distance_mm: float
) -> float:
    """User picked two points (in the mesh's current, arbitrary-unit
    coordinates) and supplied the real distance between them, in mm."""
    current_distance = float(np.linalg.norm(np.asarray(point_a) - np.asarray(point_b)))
    if current_distance <= 0:
        raise ValueError("The two selected points are coincident; cannot derive a scale from them.")
    return real_distance_mm / current_distance


def resolve_scale_manual(scale_factor: float) -> float:
    """A user-supplied `--scale-factor` (mm per current unit), taken as-is."""
    return scale_factor


def resolve_scale_fit_to_build_volume(mesh: trimesh.Trimesh, target_size_mm: float) -> float:
    """Fallback when no metric reference is available: normalize the
    mesh's longest axis to `target_size_mm`. Dimensionally meaningless
    (Section 3.4: "flagged prominently in the report") but always
    produces *something* printable."""
    longest_extent = float(mesh.extents.max())
    return target_size_mm / longest_extent
