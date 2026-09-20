"""Dense-reconstruction backends.

`monodepth_tsdf.py` (M3) is the default on Apple Silicon (no CUDA).
`openmvs.py` (M9, optional) requires the external OpenMVS binary and is
only reachable when CUDA happens to be present. `sparse_only.py` (M3) is
the always-works fallback and the `fast` preset's backend.
"""
