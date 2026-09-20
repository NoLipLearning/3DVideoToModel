"""Phase 1: frame sharpness scoring.

Laplacian variance is the sharpness metric (docs/ARCHITECTURE.md Sections
2 and 3.2): a sharp, in-focus frame has high-frequency edge content, which
shows up as high variance in the Laplacian of the grayscale image; a
blurred frame is smoothed and has low variance.

The accept/reject threshold is *adaptive* per video --
`max(blur_min_threshold, blur_adaptive_factor * median(variances))` --
because "sharp" looks different for a video shot in bright daylight vs.
dim indoor light; a fixed threshold would either accept everything or
reject everything depending on lighting.

Note: the architecture doc's directory-tree blurb for this module also
names "exposure" and "motion-blur direction" as candidates. Exposure gets
one small, hardcoded (not user-tunable) safety check below for truly
degenerate frames. Direction estimation is NOT implemented here -- it
would only matter for a deblurring step this pipeline doesn't attempt,
and Section 3 doesn't specify a threshold for it, so inventing one would
be scope creep rather than following the plan.
"""

from __future__ import annotations

import cv2
import numpy as np

# A near-uniform frame (lens covered, pointed at a blown-out light, or a
# solid wall pressed against the lens) can occasionally still show
# moderate Laplacian variance from sensor noise alone. This is a cheap,
# conservative backstop -- not a tunable, since real footage should
# essentially never trip it.
_DEGENERATE_STD_THRESHOLD = 2.0


def to_grayscale(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def laplacian_variance(gray: np.ndarray) -> float:
    """Sharpness score: variance of the Laplacian. Higher = sharper."""
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def is_degenerate_frame(gray: np.ndarray) -> bool:
    """True for a near-uniform frame (see module docstring)."""
    return float(gray.std()) < _DEGENERATE_STD_THRESHOLD


def compute_blur_threshold(
    variances: list[float], *, min_threshold: float, adaptive_factor: float
) -> float:
    """`max(min_threshold, adaptive_factor * median(variances))`.

    See docs/ARCHITECTURE.md Section 3.2. Returns `min_threshold` outright
    for an empty list rather than raising -- an empty video is caught
    earlier, with a clearer error, by the caller.
    """
    if not variances:
        return min_threshold
    return max(min_threshold, adaptive_factor * float(np.median(variances)))
