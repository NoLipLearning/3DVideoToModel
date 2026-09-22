"""Robust affine alignment: relative disparity -> metric inverse depth.

docs/ARCHITECTURE.md Section 2 ("Default dense strategy"): a monocular
depth model predicts *relative* inverse depth (disparity), not metric
depth. COLMAP's own sparse points give a handful of known-metric
inverse-depth values per image (`1/z_camera` at each 2D-3D
correspondence). Fitting `a*disparity + b ~= 1/z_camera` over those
correspondences lifts the whole disparity map to metric scale.

RANSAC, not a single least-squares fit (Section 2: "Huber/RANSAC"),
because SfM correspondences can include outliers -- a triangulation
error, or a pixel the depth model genuinely got wrong -- that would
otherwise skew the whole frame's depth scale.
"""

from __future__ import annotations

import numpy as np

_MIN_SAMPLE_SIZE = 2  # a line is determined by 2 points


def _least_squares_fit(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    if len(x) == 1:
        # A single point has no defined slope; assume disparity and
        # inverse depth already share scale (a=1) and solve for the shift.
        return 1.0, float(y[0] - x[0])
    a, b = np.polyfit(x, y, 1)
    return float(a), float(b)


def fit_affine_ransac(
    disparities: np.ndarray,
    inverse_depths: np.ndarray,
    *,
    inlier_threshold: float,
    max_iterations: int = 200,
    random_seed: int = 0,
) -> tuple[float, float, np.ndarray]:
    """Fit `inverse_depths ~= a*disparities + b` via RANSAC.

    Returns `(a, b, inlier_mask)`. `inlier_threshold` is in the same
    units as `inverse_depths` (1/meters) -- a correspondence counts as an
    inlier when `|a*disparity + b - inverse_depth| <= inlier_threshold`.

    Never raises. Falls back to a plain least-squares fit over every
    point (inlier_mask all True) when there are fewer than
    `_MIN_SAMPLE_SIZE` points, or when every RANSAC trial degenerates
    (e.g. all sampled disparities are equal) -- callers decide whether
    the resulting fit is trustworthy at all (docs/ARCHITECTURE.md
    Section 2: "require >=30 correspondences" before trusting a fit).
    """
    n = len(disparities)
    if n < _MIN_SAMPLE_SIZE:
        if n == 0:
            return 0.0, 0.0, np.zeros(0, dtype=bool)
        a, b = _least_squares_fit(disparities, inverse_depths)
        return a, b, np.ones(n, dtype=bool)

    rng = np.random.default_rng(random_seed)
    best_inlier_count = -1
    best_mask = np.zeros(n, dtype=bool)

    for _ in range(max_iterations):
        i, j = rng.choice(n, size=2, replace=False)
        dx = disparities[j] - disparities[i]
        if abs(dx) < 1e-12:
            continue  # degenerate sample, can't determine a slope
        a = (inverse_depths[j] - inverse_depths[i]) / dx
        b = inverse_depths[i] - a * disparities[i]

        residuals = np.abs(a * disparities + b - inverse_depths)
        mask = residuals <= inlier_threshold
        count = int(mask.sum())
        if count > best_inlier_count:
            best_inlier_count = count
            best_mask = mask

    if best_inlier_count < _MIN_SAMPLE_SIZE:
        # Every trial degenerated -- fall back to trusting every point
        # rather than returning a fit from zero real samples.
        a, b = _least_squares_fit(disparities, inverse_depths)
        return a, b, np.ones(n, dtype=bool)

    # Refit on the winning inlier set for the final, precise (a, b).
    a, b = _least_squares_fit(disparities[best_mask], inverse_depths[best_mask])
    return a, b, best_mask


def rmse(disparities: np.ndarray, inverse_depths: np.ndarray, a: float, b: float) -> float:
    """RMSE of `a*disparity + b` against `inverse_depths`, in 1/meters."""
    if len(disparities) == 0:
        return 0.0
    predicted = a * disparities + b
    return float(np.sqrt(np.mean((predicted - inverse_depths) ** 2)))
