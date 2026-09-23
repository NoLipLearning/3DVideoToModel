"""Dense-reconstruction backends.

`monodepth_tsdf.py` (M3) is the default on Apple Silicon (no CUDA).
`openmvs.py` (M9, optional) runs the external OpenMVS binaries on the CPU
and is opt-in only (`dense.backend=openmvs`); "auto" never selects it. `sparse_only.py` (M3) is
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
from v2m.phase2_sfm.dense import monodepth_tsdf, openmvs, sparse_only
from v2m.types import DenseResult


def densify(
    sfm_dir: Path,
    output_dir: Path,
    config: DenseConfig,
    *,
    depth_estimator: monodepth_tsdf.DepthEstimator | None = None,
) -> DenseResult:
    """Resolve `config.backend` (probing hardware/libraries when "auto",
    same as `capability.py`'s own `dense_backend` property -- see
    CLAUDE.md's CUDA constraint) and dispatch to the matching backend.

    `depth_estimator` is passed through to `monodepth_tsdf` (and ignored
    by every other backend) so end-to-end tests can inject the same
    geometric test double the M3 tests use.
    """
    backend = config.backend
    if backend == "auto":
        backend = capability.probe().dense_backend

    if backend == "monodepth_tsdf":
        return monodepth_tsdf.densify(sfm_dir, output_dir, config, depth_estimator=depth_estimator)
    if backend == "sparse_only":
        return sparse_only.densify(sfm_dir, output_dir, config)
    if backend == "openmvs":
        return openmvs.densify(sfm_dir, output_dir, config)
    raise SfMError(
        f"Unknown dense backend '{backend}'.",
        remedy="Set dense.backend to one of: auto, monodepth_tsdf, sparse_only, openmvs.",
    )
