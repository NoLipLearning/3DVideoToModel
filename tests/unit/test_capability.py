"""Capability probe sanity checks.

These must never require CUDA, pycolmap, or torch to actually be
installed -- capability.probe() has to work (and report absence
correctly) on a bare-bones dev machine as well as on the real target.
"""

from v2m import capability


def test_probe_never_raises():
    caps = capability.probe()
    assert caps.platform_system
    assert isinstance(caps.has_cuda, bool)
    assert isinstance(caps.has_mps, bool)


def test_missing_core_detects_absent_library():
    caps = capability.probe()
    caps.libs["numpy"].available = False
    assert "numpy" in caps.missing_core()


def test_dense_backend_never_selects_openmvs_without_cuda():
    caps = capability.probe()
    if not caps.has_cuda:
        assert caps.has_colmap_dense is False
        assert caps.dense_backend != "openmvs"


def test_dense_backend_falls_back_to_sparse_only_when_torch_or_open3d_absent():
    caps = capability.probe()
    caps.has_cuda = False
    caps.libs["torch"].available = False
    assert caps.dense_backend == "sparse_only"


def test_hevc_decode_check_returns_none_without_probe_asset():
    caps = capability.probe()
    # No probe clip is shipped before M1 -- this must read as "unknown",
    # never as a false failure.
    assert capability.hevc_decode_check(caps) is None
