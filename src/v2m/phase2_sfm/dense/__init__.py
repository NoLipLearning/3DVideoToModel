"""Dense-reconstruction backends.

`monodepth_tsdf.py` (M3) is the default on Apple Silicon (no CUDA).
`openmvs.py` (M9, optional) requires the external OpenMVS binary and is
only reachable when CUDA happens to be present. `sparse_only.py` (M3) is
the always-works fallback and the `fast` preset's backend.

`densify()` below is this subpackage's public entry point -- the `dense/`
analogue of `phase2_sfm/sfm.py`'s backend dispatch, kept here rather than
as a sibling file since (unlike sparse SfM's single always-used
`colmap_backend`) there's no shared orchestration logic to justify one --
just a `config.backend`/capability-probe lookup and a call-through.
"""

from __future__ import annotations

from pathlib import Path

from v2m import capability
from v2m.config import DenseConfig
from v2m.errors import SfMError
from v2m.phase2_sfm.dense import monodepth_tsdf, sparse_only
from v2m.types import DenseResult


def densify(sfm_dir: Path, output_dir: Path, config: DenseConfig) -> DenseResult:
    """Resolve `config.backend` (probing hardware/libraries when "auto",
    same as `capability.py`'s own `dense_backend` property -- see
    CLAUDE.md's CUDA constraint) and dispatch to the matching backend."""
    backend = config.backend
    if backend == "auto":
        backend = capability.probe().dense_backend

    if backend == "monodepth_tsdf":
        return monodepth_tsdf.densify(sfm_dir, output_dir, config)
    if backend == "sparse_only":
        return sparse_only.densify(sfm_dir, output_dir, config)
    if backend == "openmvs":
        raise SfMError(
            "The openmvs dense backend is not implemented yet (M9).",
            remedy="Set dense.backend to 'monodepth_tsdf' or 'sparse_only', or use --preset fast.",
        )
    raise SfMError(
        f"Unknown dense backend '{backend}'.",
        remedy="Set dense.backend to one of: auto, monodepth_tsdf, sparse_only.",
    )
