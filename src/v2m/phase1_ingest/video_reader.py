"""Phase 1: streaming video decode.

Reads a video frame-by-frame -- never loading the whole file into memory
(docs/ARCHITECTURE.md Section 3.2, "never load whole video") -- and
applies the resolution cap during read so nothing downstream ever sees a
frame larger than `resolution_cap_px` on its long edge.

PyAV is tried first because it decodes the container's display-matrix
rotation (`frame.rotation`, verified against the installed av==18
package) and gives correct per-frame presentation timestamps even for
variable-frame-rate footage. `cv2.VideoCapture` is the fallback when PyAV
is unavailable; it is known to mishandle both of those (see
docs/ARCHITECTURE.md Section 2), so a warning is logged once when
falling back.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from v2m.errors import IngestError

logger = logging.getLogger("v2m.phase1_ingest.video_reader")

_ROTATE_MAP = {
    90: cv2.ROTATE_90_CLOCKWISE,
    180: cv2.ROTATE_180,
    270: cv2.ROTATE_90_COUNTERCLOCKWISE,
}


@dataclass
class RawFrame:
    image: np.ndarray  # BGR, already rotated + resolution-capped
    frame_index: int
    timestamp_s: float


def _apply_rotation(image: np.ndarray, degrees: int) -> np.ndarray:
    rotate_code = _ROTATE_MAP.get(degrees % 360)
    if rotate_code is None:
        return image
    return cv2.rotate(image, rotate_code)


def _cap_resolution(image: np.ndarray, resolution_cap_px: int) -> np.ndarray:
    height, width = image.shape[:2]
    long_edge = max(height, width)
    if long_edge <= resolution_cap_px:
        return image
    scale = resolution_cap_px / long_edge
    new_size = (max(1, round(width * scale)), max(1, round(height * scale)))
    return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)


def _iter_frames_pyav(video_path: Path, resolution_cap_px: int) -> Iterator[RawFrame]:
    import av

    container = av.open(str(video_path))
    try:
        if not container.streams.video:
            raise IngestError(
                f"{video_path} has no video stream.",
                remedy="Check the file is a real video, not audio-only or corrupted.",
            )
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        average_rate = float(stream.average_rate) if stream.average_rate else 30.0

        for frame_index, frame in enumerate(container.decode(stream)):
            image = frame.to_ndarray(format="bgr24")
            rotation = int(frame.rotation or 0)
            if rotation:
                image = _apply_rotation(image, rotation)
            image = _cap_resolution(image, resolution_cap_px)
            timestamp_s = (
                float(frame.time) if frame.time is not None else frame_index / average_rate
            )
            yield RawFrame(image=image, frame_index=frame_index, timestamp_s=timestamp_s)
    finally:
        container.close()


def _iter_frames_opencv(video_path: Path, resolution_cap_px: int) -> Iterator[RawFrame]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        cap.release()
        raise IngestError(
            f"OpenCV could not open {video_path}.",
            remedy="Confirm the file exists and is a supported video container/codec.",
        )
    logger.warning(
        "Reading %s via OpenCV (PyAV unavailable or failed to start) -- iPhone rotation "
        "metadata will NOT be corrected. See docs/ARCHITECTURE.md Section 2.",
        video_path,
    )
    frame_index = 0
    try:
        while True:
            ok, image = cap.read()
            if not ok:
                break
            image = _cap_resolution(image, resolution_cap_px)
            timestamp_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
            timestamp_s = timestamp_ms / 1000.0 if timestamp_ms > 0 else frame_index / 30.0
            yield RawFrame(image=image, frame_index=frame_index, timestamp_s=timestamp_s)
            frame_index += 1
    finally:
        cap.release()


def iter_frames(
    video_path: Path, *, resolution_cap_px: int, use_pyav: bool = True
) -> Iterator[RawFrame]:
    """Stream frames from `video_path`, one at a time, O(1) memory.

    Tries PyAV first when `use_pyav` is true and the `av` package is
    importable. Falls back to OpenCV only if PyAV fails *before yielding
    any frame* (missing package, corrupt header, unsupported container) --
    a failure partway through a stream is NOT retried via OpenCV, since
    restarting from frame 0 there would silently duplicate every frame
    already handed to the caller.
    """
    if not video_path.exists():
        raise IngestError(
            f"Video file not found: {video_path}",
            remedy="Check the path is correct and the file was fully copied over.",
        )

    if use_pyav:
        try:
            import av  # noqa: F401
        except ImportError:
            logger.info("PyAV not installed; using OpenCV for %s.", video_path)
        else:
            yielded_any = False
            try:
                for raw_frame in _iter_frames_pyav(video_path, resolution_cap_px):
                    yielded_any = True
                    yield raw_frame
                return
            except Exception:
                if yielded_any:
                    raise
                logger.warning(
                    "PyAV failed to open %s; falling back to OpenCV.", video_path, exc_info=True
                )

    yield from _iter_frames_opencv(video_path, resolution_cap_px)


def probe_frame_count(video_path: Path) -> int | None:
    """Best-effort total frame count, for progress reporting only.

    Returns None when the container doesn't expose a reliable count
    (common for some VFR streams) -- callers must not depend on this for
    correctness, only for a progress bar / sanity display.
    """
    try:
        import av

        with av.open(str(video_path)) as container:
            if container.streams.video:
                frames = container.streams.video[0].frames
                if frames:
                    return int(frames)
    except Exception:
        pass

    cap = cv2.VideoCapture(str(video_path))
    try:
        if not cap.isOpened():
            return None
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        return count if count > 0 else None
    finally:
        cap.release()
