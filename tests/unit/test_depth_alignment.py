"""Pure numerical tests for the robust affine disparity->inverse-depth fit.

docs/ARCHITECTURE.md Section 4 (M3): "Unit-test the affine alignment
against synthetic depth with known a,b + outliers." No SfM/COLMAP/open3d
involved -- these are plain numpy checks against invented (a, b) pairs,
so unlike the rest of this milestone's tests, none of this needs `slow`.
"""

import numpy as np
import pytest

from v2m.phase2_sfm.dense import depth_alignment


def test_fit_recovers_exact_affine_with_no_noise():
    rng = np.random.default_rng(1)
    disparities = rng.uniform(0.1, 1.0, size=25)
    true_a, true_b = 2.5, 0.3
    inverse_depths = true_a * disparities + true_b

    a, b, mask = depth_alignment.fit_affine_ransac(
        disparities, inverse_depths, inlier_threshold=1e-6
    )

    assert a == pytest.approx(true_a)
    assert b == pytest.approx(true_b)
    assert mask.all()
    assert depth_alignment.rmse(disparities, inverse_depths, a, b) < 1e-8


def test_fit_is_robust_to_outliers():
    rng = np.random.default_rng(2)
    disparities = rng.uniform(0.1, 1.0, size=40)
    true_a, true_b = -1.8, 0.9
    inverse_depths = true_a * disparities + true_b

    # Corrupt a minority of points far off the true line -- e.g. a
    # triangulation error or a pixel the depth model got wrong (module
    # docstring's stated reason for RANSAC over a plain least-squares fit).
    outlier_idx = np.array([0, 5, 10, 15])
    inverse_depths = inverse_depths.copy()
    inverse_depths[outlier_idx] += 5.0

    a, b, mask = depth_alignment.fit_affine_ransac(
        disparities, inverse_depths, inlier_threshold=0.01
    )

    assert a == pytest.approx(true_a, abs=1e-3)
    assert b == pytest.approx(true_b, abs=1e-3)
    assert not mask[outlier_idx].any()
    assert mask.sum() == len(disparities) - len(outlier_idx)

    inlier_rmse = depth_alignment.rmse(disparities[mask], inverse_depths[mask], a, b)
    assert inlier_rmse < 1e-3


def test_rmse_zero_for_a_perfect_fit():
    disparities = np.array([0.1, 0.4, 0.7, 1.0])
    a, b = 3.0, -0.2
    inverse_depths = a * disparities + b
    assert depth_alignment.rmse(disparities, inverse_depths, a, b) == pytest.approx(0.0)


def test_rmse_of_empty_arrays_is_zero():
    empty = np.zeros(0)
    assert depth_alignment.rmse(empty, empty, 1.0, 0.0) == 0.0


def test_fit_affine_ransac_empty_input_returns_zeros():
    empty = np.zeros(0)
    a, b, mask = depth_alignment.fit_affine_ransac(empty, empty, inlier_threshold=0.1)
    assert (a, b) == (0.0, 0.0)
    assert mask.shape == (0,)


def test_fit_affine_ransac_single_point_assumes_unit_slope():
    # _least_squares_fit's documented n==1 special case: no slope is
    # determined by one point, so a=1.0 and b is solved to pass through it.
    disparities = np.array([0.5])
    inverse_depths = np.array([2.0])
    a, b, mask = depth_alignment.fit_affine_ransac(
        disparities, inverse_depths, inlier_threshold=0.1
    )
    assert a == 1.0
    assert b == pytest.approx(1.5)
    assert mask.tolist() == [True]


def test_fit_affine_ransac_degenerate_disparities_falls_back_to_least_squares():
    # Every disparity identical -> every 2-point RANSAC sample has dx=0 and
    # is skipped -- no trial ever finds a slope, so every point must still
    # come back as an inlier via the least-squares fallback rather than an
    # empty winning set.
    disparities = np.full(10, 0.5)
    inverse_depths = np.linspace(1.0, 2.0, 10)
    a, b, mask = depth_alignment.fit_affine_ransac(
        disparities, inverse_depths, inlier_threshold=1e-9, max_iterations=50
    )
    assert mask.all()


def test_fit_affine_ransac_deterministic_for_a_fixed_seed():
    rng = np.random.default_rng(3)
    disparities = rng.uniform(0.1, 1.0, size=30)
    inverse_depths = 1.2 * disparities + 0.1
    inverse_depths[3] += 2.0  # one outlier

    first = depth_alignment.fit_affine_ransac(disparities, inverse_depths, inlier_threshold=0.05)
    second = depth_alignment.fit_affine_ransac(disparities, inverse_depths, inlier_threshold=0.05)

    assert first[0] == second[0]
    assert first[1] == second[1]
    assert (first[2] == second[2]).all()
