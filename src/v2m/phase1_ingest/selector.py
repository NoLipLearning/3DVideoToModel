"""Phase 1: redundancy gating and frame-budget subsampling.

Two independent jobs, both described in docs/ARCHITECTURE.md Section 4
(M1) and Section 3.2:

1. Redundancy gate -- drop a frame that looks nearly identical to the
   last *kept* frame, measured by ORB-feature overlap surviving a
   homography-RANSAC consistency check. This is what turns "a phone held
   still for 3 seconds" into one frame instead of ninety.
2. Frame-budget subsampling -- if more frames survive gating than the
   configured budget, keep an evenly time-spread subset rather than
   (say) the first N, so the surviving set still covers the whole capture
   instead of just its beginning.
"""

from __future__ import annotations

import cv2
import numpy as np

_ORB = cv2.ORB_create(nfeatures=1000)
_MIN_MATCHES_FOR_OVERLAP = 8


def detect_orb(gray: np.ndarray) -> tuple[tuple, np.ndarray | None]:
    keypoints, descriptors = _ORB.detectAndCompute(gray, None)
    return keypoints, descriptors


def orb_overlap_ratio(
    keypoints_a: tuple,
    descriptors_a: np.ndarray | None,
    keypoints_b: tuple,
    descriptors_b: np.ndarray | None,
) -> float:
    """Fraction of the smaller keypoint set that matches consistently.

    `inlier_count / min(len(keypoints_a), len(keypoints_b))`, where
    inliers are ORB matches that survive a homography-RANSAC check.

    Near 1.0: frame B shows almost the same thing as frame A (redundant).
    Near/at 0.0: frame B adds new coverage, or there wasn't enough to
    match on at all -- an under-featured frame is never treated as a
    duplicate just because it has few keypoints.
    """
    if (
        descriptors_a is None
        or descriptors_b is None
        or len(keypoints_a) < _MIN_MATCHES_FOR_OVERLAP
        or len(keypoints_b) < _MIN_MATCHES_FOR_OVERLAP
    ):
        return 0.0

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(descriptors_a, descriptors_b)
    if len(matches) < _MIN_MATCHES_FOR_OVERLAP:
        return 0.0

    pts_a = np.float32([keypoints_a[m.queryIdx].pt for m in matches])
    pts_b = np.float32([keypoints_b[m.trainIdx].pt for m in matches])
    _, mask = cv2.findHomography(pts_a, pts_b, cv2.RANSAC, 5.0)
    if mask is None:
        return 0.0
    inliers = int(mask.sum())
    return inliers / min(len(keypoints_a), len(keypoints_b))


def subsample_to_budget(indices: list[int], budget: int) -> tuple[list[int], list[int]]:
    """Evenly spread `budget` picks across `indices` (ascending, unique).

    Returns (kept, dropped). Keeps everything if already at or under
    budget. Always keeps the first and last element when trimming --
    `linspace` guarantees the endpoints -- so the surviving set spans the
    full capture rather than clustering at one end.
    """
    if len(indices) <= budget:
        return list(indices), []
    if budget <= 0:
        return [], list(indices)

    positions = np.linspace(0, len(indices) - 1, num=budget)
    chosen = sorted({int(round(p)) for p in positions})

    # Rounding collisions can under-fill the budget by a handful of
    # slots; top up from the largest remaining gaps until we hit it (or
    # run out of room to add anything new).
    while len(chosen) < budget:
        gaps = [(chosen[i + 1] - chosen[i], i) for i in range(len(chosen) - 1)]
        if not gaps:
            break
        _, gap_start = max(gaps)
        candidate = (chosen[gap_start] + chosen[gap_start + 1]) // 2
        if candidate in chosen:
            break
        chosen.append(candidate)
        chosen.sort()

    kept = [indices[i] for i in chosen]
    kept_set = set(kept)
    dropped = [i for i in indices if i not in kept_set]
    return kept, dropped
