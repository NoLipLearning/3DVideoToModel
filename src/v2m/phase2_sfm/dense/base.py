"""DenseBackend protocol.

Every backend takes the Phase 2a output directory (COLMAP's own
reconstruction at `sfm/sparse/final/`, plus `sfm/undistorted/images/`)
and produces a dense point cloud. `monodepth_tsdf.py` is the default on
Apple Silicon (no CUDA, ever, on this project's target -- see
capability.py); `sparse_only.py` is the always-works fallback and the
`fast` preset's backend; `openmvs.py` (M9, optional) is only reachable
when CUDA happens to be present.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from v2m.config import DenseConfig
from v2m.types import DenseResult


class DenseBackend(Protocol):
    """Turns a completed sparse reconstruction into a dense point cloud."""

    def densify(self, sfm_dir: Path, output_dir: Path, config: DenseConfig) -> DenseResult: ...
