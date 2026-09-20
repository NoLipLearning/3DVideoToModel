"""Blur scoring: synthetic sharp vs. blurred image pairs.

Per docs/ARCHITECTURE.md M1 verification: "Unit-test blur scoring on
synthetic sharp/blurred pairs."
"""

import cv2
import numpy as np

from v2m.phase1_ingest import quality


def _checkerboard(size: int = 200, cell: int = 10) -> np.ndarray:
    board = np.zeros((size, size), dtype=np.uint8)
    for y in range(0, size, cell):
        for x in range(0, size, cell):
            if ((x // cell) + (y // cell)) % 2 == 0:
                board[y : y + cell, x : x + cell] = 255
    return board


def test_laplacian_variance_drops_after_blur():
    sharp = _checkerboard()
    blurred = cv2.GaussianBlur(sharp, (15, 15), 0)

    sharp_var = quality.laplacian_variance(sharp)
    blurred_var = quality.laplacian_variance(blurred)

    assert sharp_var > blurred_var * 5  # a strong, unambiguous drop


def test_laplacian_variance_flat_image_is_near_zero():
    flat = np.full((100, 100), 128, dtype=np.uint8)
    assert quality.laplacian_variance(flat) < 1e-6


def test_to_grayscale_passes_through_2d_and_converts_3d():
    gray = _checkerboard()
    assert quality.to_grayscale(gray) is gray

    bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    converted = quality.to_grayscale(bgr)
    assert converted.ndim == 2
    assert converted.shape == gray.shape


def test_compute_blur_threshold_formula():
    variances = [10.0, 20.0, 30.0, 40.0, 50.0]  # median = 30
    threshold = quality.compute_blur_threshold(variances, min_threshold=5.0, adaptive_factor=0.6)
    assert threshold == 18.0  # max(5.0, 0.6 * 30) = 18.0


def test_compute_blur_threshold_floor_wins_on_dark_low_variance_video():
    variances = [1.0, 2.0, 3.0]  # median = 2 -> 0.6*2 = 1.2
    threshold = quality.compute_blur_threshold(variances, min_threshold=30.0, adaptive_factor=0.6)
    assert threshold == 30.0  # the floor wins


def test_compute_blur_threshold_empty_list_returns_floor():
    assert quality.compute_blur_threshold([], min_threshold=30.0, adaptive_factor=0.6) == 30.0


def test_is_degenerate_frame_detects_uniform_image():
    flat = np.full((50, 50), 100, dtype=np.uint8)
    assert quality.is_degenerate_frame(flat) is True


def test_is_degenerate_frame_false_for_textured_image():
    assert quality.is_degenerate_frame(_checkerboard()) is False
