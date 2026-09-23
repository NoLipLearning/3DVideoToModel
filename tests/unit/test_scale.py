"""scale.py: the four metric-scale methods of docs/ARCHITECTURE.md
Section 3.4. The ArUco path is exercised against a synthetic,
fronto-parallel marker image with a known camera, since this project's
video fixture has no marker in it."""

import cv2
import numpy as np
import pytest
import trimesh

from v2m.phase4_print import scale

_CAMERA = np.array([[800.0, 0.0, 320.0], [0.0, 800.0, 240.0], [0.0, 0.0, 1.0]])


def _marker_scene(marker_mm: float, distance_mm: float) -> np.ndarray:
    """A 640x480 white image with an ArUco marker (DICT_4X4_50, id 0)
    exactly where a `marker_mm`-wide marker `distance_mm` in front of
    `_CAMERA` would project."""
    half = marker_mm / 2.0
    corners_3d = np.array([[-half, half, 0], [half, half, 0], [half, -half, 0], [-half, -half, 0]])
    projected, _ = cv2.projectPoints(
        corners_3d, np.zeros(3), np.array([0.0, 0.0, distance_mm]), _CAMERA, np.zeros(5)
    )
    projected = projected.reshape(-1, 2)
    x0, y0 = np.round(projected.min(axis=0)).astype(int)
    size = int(np.round(projected.max(axis=0)[0])) - x0

    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    marker = cv2.aruco.generateImageMarker(dictionary, 0, size, borderBits=1)
    scene = np.full((480, 640, 3), 255, dtype=np.uint8)
    scene[y0 : y0 + size, x0 : x0 + size] = cv2.cvtColor(marker, cv2.COLOR_GRAY2BGR)
    return scene


def test_fit_to_build_volume_normalizes_the_longest_axis():
    mesh = trimesh.creation.box(extents=[1.0, 2.5, 0.5])
    assert scale.resolve_scale_fit_to_build_volume(mesh, 150.0) == pytest.approx(60.0)


def test_two_point_uses_the_real_distance():
    factor = scale.resolve_scale_two_point(np.zeros(3), np.array([0.0, 2.0, 0.0]), 50.0)
    assert factor == pytest.approx(25.0)


def test_two_point_rejects_coincident_points():
    with pytest.raises(ValueError):
        scale.resolve_scale_two_point(np.ones(3), np.ones(3), 50.0)


def test_manual_factor_passes_through():
    assert scale.resolve_scale_manual(3.7) == 3.7


def test_aruco_recovers_the_metric_distance():
    image = _marker_scene(marker_mm=100.0, distance_mm=300.0)
    factor = scale.resolve_scale_aruco(image, _CAMERA, 100.0, colmap_distance_mm=150.0)
    assert factor is not None
    # 300mm real vs 150 reconstruction units; the only error is the
    # marker's integer-pixel placement in the synthetic image.
    assert factor == pytest.approx(2.0, rel=0.01)


def test_aruco_returns_none_without_a_marker():
    blank = np.full((480, 640, 3), 255, dtype=np.uint8)
    assert scale.resolve_scale_aruco(blank, _CAMERA, 100.0, colmap_distance_mm=150.0) is None
