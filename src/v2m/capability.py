"""Hardware, binary, and library capability probing.

`probe()` builds a `Capabilities` snapshot once at startup. Every phase
backend that has a hard requirement (e.g. "dense reconstruction needs
open3d + torch") should read this snapshot rather than probing itself, so
the whole pipeline agrees on what's available and never crashes deep
inside a run because of a decision it could have made up front.

CRITICAL: `patch_match_stereo` (COLMAP's CUDA dense-MVS stage) has no
macOS build and no CPU fallback -- see docs/ARCHITECTURE.md Section 2.
`has_colmap_dense` is therefore hardcoded to track CUDA availability, full
stop. This project's target (Apple Silicon) never has CUDA, so it must
always be False there. Never remove this gate to "try it anyway"; it
raises inside COLMAP's C++ layer, not a Python exception you can catch
cleanly.
"""

from __future__ import annotations

import importlib
import platform
import shutil
from dataclasses import dataclass, field
from pathlib import Path

# Import name -> probed. Keys are the actual `import` name, which is not
# always the PyPI package name (opencv-contrib-python -> cv2, pyyaml ->
# yaml, pillow -> PIL). cli.py maps these back to friendly names for
# display.
_CORE_LIBS = [
    "numpy",
    "cv2",
    "scipy",
    "trimesh",
    "manifold3d",
    "open3d",
    "pydantic",
    "typer",
    "yaml",
    "rich",
    "psutil",
]
_SFM_LIBS = ["pycolmap"]
_DEPTH_LIBS = ["torch", "torchvision", "transformers", "PIL", "safetensors"]
_MESH_LIBS = ["pymeshlab"]
_WEB_LIBS = ["fastapi", "uvicorn"]
_OPTIONAL_LIBS = ["av"]

ALL_PROBED_LIBS = _CORE_LIBS + _SFM_LIBS + _DEPTH_LIBS + _MESH_LIBS + _WEB_LIBS + _OPTIONAL_LIBS


def _import_ok(module_name: str) -> tuple[bool, str | None]:
    """Return (available, version_or_none) without ever raising."""
    try:
        mod = importlib.import_module(module_name)
    except Exception:
        return False, None
    return True, getattr(mod, "__version__", None)


def _binary_on_path(name: str) -> str | None:
    return shutil.which(name)


@dataclass
class LibStatus:
    name: str
    available: bool
    version: str | None = None


@dataclass
class Capabilities:
    platform_system: str
    platform_machine: str
    python_version: str

    has_cuda: bool
    has_mps: bool
    cuda_device_name: str | None

    libs: dict[str, LibStatus] = field(default_factory=dict)

    colmap_binary: str | None = None
    openmvs_binary: str | None = None
    ffmpeg_binary: str | None = None

    @property
    def has_colmap_dense(self) -> bool:
        """COLMAP `patch_match_stereo` requires CUDA. No exceptions, ever."""
        return self.has_cuda

    @property
    def dense_backend(self) -> str:
        """Which Phase 2b backend `sfm.py` should select when config says "auto"."""
        if self.has_colmap_dense and self.openmvs_binary:
            # Only reachable with CUDA present -- never the Apple Silicon path.
            return "openmvs"
        torch_ok = self.libs.get("torch", LibStatus("torch", False)).available
        open3d_ok = self.libs.get("open3d", LibStatus("open3d", False)).available
        if torch_ok and open3d_ok:
            return "monodepth_tsdf"
        return "sparse_only"

    def missing_core(self) -> list[str]:
        return [
            name for name in _CORE_LIBS if not self.libs.get(name, LibStatus(name, False)).available
        ]


def probe() -> Capabilities:
    """Probe the current environment. Never raises -- every check is guarded."""
    libs: dict[str, LibStatus] = {}
    for name in ALL_PROBED_LIBS:
        ok, version = _import_ok(name)
        libs[name] = LibStatus(name=name, available=ok, version=version)

    has_cuda = False
    has_mps = False
    cuda_device_name: str | None = None
    if libs["torch"].available:
        try:
            import torch

            has_cuda = bool(torch.cuda.is_available())
            if has_cuda:
                cuda_device_name = torch.cuda.get_device_name(0)
            mps_backend = getattr(torch.backends, "mps", None)
            has_mps = bool(mps_backend and mps_backend.is_available())
        except Exception:
            # A broken torch install shouldn't take down `doctor` -- report
            # it as no-accelerator-available rather than crashing.
            pass

    return Capabilities(
        platform_system=platform.system(),
        platform_machine=platform.machine(),
        python_version=platform.python_version(),
        has_cuda=has_cuda,
        has_mps=has_mps,
        cuda_device_name=cuda_device_name,
        libs=libs,
        colmap_binary=_binary_on_path("colmap"),
        openmvs_binary=_binary_on_path("DensifyPointCloud"),
        ffmpeg_binary=_binary_on_path("ffmpeg"),
    )


def hevc_decode_check(capabilities: Capabilities) -> bool | None:
    """Actually try to decode one frame of a tiny bundled HEVC clip via OpenCV.

    Trusting build flags is not good enough -- `doctor` should prove it
    works. Returns:
      * True/False if a probe clip exists and decoding did/didn't work.
      * None if there's nothing to test yet (no probe asset shipped before
        M1's synthetic-video fixture exists) -- this is "unknown", not a
        failure, and callers should display it as such.
    """
    if not capabilities.libs["cv2"].available:
        return False
    sample = Path(__file__).parent / "assets" / "hevc_probe.mp4"
    if not sample.exists():
        return None
    try:
        import cv2

        cap = cv2.VideoCapture(str(sample))
        ok, _ = cap.read()
        cap.release()
        return bool(ok)
    except Exception:
        return False
