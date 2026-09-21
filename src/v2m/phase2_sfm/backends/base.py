"""SfmBackend protocol.

Deliberately scoped to just the *swappable* part of sparse SfM: turning a
directory of images into a populated COLMAP database (keypoints,
descriptors, verified two-view matches). Incremental mapping,
undistortion, and export are backend-agnostic COLMAP calls and live in
`sfm.py`, applied uniformly regardless of which backend populated the
database -- that's what lets `hloc_backend.py` (M9, optional: ALIKED/DISK
+ LightGlue for low-texture scenes) slot in later without touching
`sfm.py` at all.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from v2m.config import SfmConfig


class SfmBackend(Protocol):
    """Populates a COLMAP database for every image in `images_dir`."""

    def extract_and_match(
        self, images_dir: Path, database_path: Path, config: SfmConfig
    ) -> None: ...
