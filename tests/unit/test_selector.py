"""ORB redundancy overlap and frame-budget subsampling."""

import cv2
import numpy as np

from v2m.phase1_ingest import selector


def _textured_image(size: int = 300, seed: int = 0) -> np.ndarray:
    """A structured (not random-noise) grayscale image with plenty of
    corner-like features for ORB -- a grid of circles at jittered
    positions, so shifting the seed gives a genuinely different scene
    rather than just re-shuffled noise ORB can't reliably describe."""
    rng = np.random.default_rng(seed)
    image = np.full((size, size), 60, dtype=np.uint8)
    for cy in range(20, size - 20, 25):
        for cx in range(20, size - 20, 25):
            jitter = rng.integers(-5, 6, size=2)
            center = (cx + int(jitter[0]), cy + int(jitter[1]))
            cv2.circle(image, center, 8, 220, thickness=-1)
    return image


def test_orb_overlap_ratio_high_for_identical_image():
    image = _textured_image(seed=1)
    kp, des = selector.detect_orb(image)
    overlap = selector.orb_overlap_ratio(kp, des, kp, des)
    assert overlap > 0.9


def test_orb_overlap_ratio_low_for_different_scenes():
    image_a = _textured_image(seed=1)
    image_b = _textured_image(seed=2)
    kp_a, des_a = selector.detect_orb(image_a)
    kp_b, des_b = selector.detect_orb(image_b)
    overlap = selector.orb_overlap_ratio(kp_a, des_a, kp_b, des_b)
    assert overlap < 0.5


def test_orb_overlap_ratio_zero_when_too_few_keypoints():
    blank = np.full((50, 50), 128, dtype=np.uint8)
    kp, des = selector.detect_orb(blank)
    assert selector.orb_overlap_ratio(kp, des, kp, des) == 0.0


def test_subsample_to_budget_keeps_all_when_under_budget():
    indices = [0, 5, 10]
    kept, dropped = selector.subsample_to_budget(indices, budget=5)
    assert kept == indices
    assert dropped == []


def test_subsample_to_budget_exact_count_and_endpoints():
    indices = list(range(100))
    kept, dropped = selector.subsample_to_budget(indices, budget=10)
    assert len(kept) == 10
    assert kept[0] == 0
    assert kept[-1] == 99
    assert sorted(kept) == kept  # stays in ascending order
    assert len(kept) + len(dropped) == len(indices)


def test_subsample_to_budget_zero_budget_drops_everything():
    indices = [1, 2, 3]
    kept, dropped = selector.subsample_to_budget(indices, budget=0)
    assert kept == []
    assert dropped == indices
