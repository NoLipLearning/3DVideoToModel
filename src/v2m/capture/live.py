"""Guided live capture (M8): a camera loop with a quality HUD that keeps
only the frames reconstruction can use, writing them straight to disk.

docs/ARCHITECTURE.md M8: "OpenCV camera loop with a quality HUD
(sharpness bar, accepted-frame counter, baseline/coverage indicator),
writing accepted frames directly to frames/ so it enters the pipeline at
M2 -- no SLAM." Each frame goes through the same gates Phase 1 applies
to a video, decided live instead of in three passes:

- Sharpness: Laplacian variance against an adaptive threshold. Phase 1
  takes `max(min, factor * median)` over the whole video; live, the
  median is over a rolling window of recent frames, so the threshold
  follows the lighting as the operator moves.
- Degenerate frames (lens covered, blown out) are rejected outright.
- Redundancy: ORB overlap with the last *accepted* frame. At or above
  `redundancy_overlap_max` the view isn't new enough to keep. That same
  number, as 1 - overlap, is the HUD's "baseline" bar: how much new view
  has accumulated since the last keep, which is exactly what tells the
  operator to keep moving or to slow down.
- The frame budget caps the capture; at the budget it tells the operator
  they're done.

Honest limits: there is no pose tracking, so "coverage" is the accepted
count against the budget, not a map of which sides have been seen. The
HUD says "walk all the way around" instead of pretending to know.

The output folder (frames + frames.json + capture.json) is what
`v2m run --from-frames <folder>` takes (phase1_ingest/from_frames.py
keeps the capture's own accept decisions rather than re-gating them).
"""

from __future__ import annotations

import json
import logging
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from v2m.config import IngestConfig
from v2m.errors import IngestError
from v2m.phase1_ingest import quality, selector
from v2m.phase1_ingest.from_frames import CAPTURE_MARKER
from v2m.phase1_ingest.video_reader import _cap_resolution
from v2m.types import FrameRecord

logger = logging.getLogger("v2m.capture")

_SHARPNESS_WINDOW = 60  # frames in the rolling median (~2s at 30fps)
_WARMUP_FRAMES = 10  # before this many, only the fixed minimum threshold applies
_LOW_OVERLAP = 0.15  # below this vs. the last kept view, SfM may fail to link them
_DARK, _BRIGHT = 40.0, 225.0  # mean gray level bounds for an exposure warning
_WINDOW_NAME = "v2m capture  (space: pause, q: finish)"

GOOD, WARN, BAD = (80, 200, 80), (0, 190, 255), (70, 70, 230)  # BGR


@dataclass
class CaptureState:
    frames_seen: int = 0
    accepted: int = 0
    budget: int = 0
    sharpness: float = 0.0
    threshold: float = 0.0
    novelty: float = 0.0  # 1 - ORB overlap with the last accepted frame
    novelty_needed: float = 0.25  # 1 - redundancy_overlap_max
    brightness: float = 128.0
    message: str = "Point the camera at the subject"
    tone: tuple[int, int, int] = WARN
    just_accepted: bool = False
    paused: bool = False

    @property
    def done(self) -> bool:
        return self.accepted >= self.budget


class LiveCapture:
    """The gating logic, independent of any camera or window, so it can be
    driven by a video file in tests exactly as by a webcam."""

    def __init__(self, output_dir: Path, config: IngestConfig) -> None:
        self.output_dir = output_dir
        self.config = config
        output_dir.mkdir(parents=True, exist_ok=True)
        for stale in output_dir.glob("*.jpg"):
            stale.unlink()
        self.records: list[FrameRecord] = []
        self._sharpness: deque[float] = deque(maxlen=_SHARPNESS_WINDOW)
        self._last_kp: tuple | None = None
        self._last_des: np.ndarray | None = None
        self.state = CaptureState(
            budget=config.frame_budget, novelty_needed=1.0 - config.redundancy_overlap_max
        )

    def process(self, frame_bgr: np.ndarray, timestamp_s: float) -> CaptureState:
        state = self.state
        state.just_accepted = False
        index = state.frames_seen
        state.frames_seen += 1
        if state.done:
            state.message, state.tone = "Done -- press q to finish", GOOD
            return state

        image = _cap_resolution(frame_bgr, self.config.resolution_cap_px)
        gray = quality.to_grayscale(image)
        state.brightness = float(gray.mean())
        state.sharpness = quality.laplacian_variance(gray)
        self._sharpness.append(state.sharpness)
        state.threshold = (
            quality.compute_blur_threshold(
                list(self._sharpness),
                min_threshold=self.config.blur_min_threshold,
                adaptive_factor=self.config.blur_adaptive_factor,
            )
            if len(self._sharpness) >= _WARMUP_FRAMES
            else self.config.blur_min_threshold
        )

        def reject(reason: str, message: str, tone) -> CaptureState:
            state.message, state.tone = message, tone
            self.records.append(
                FrameRecord(
                    path="",
                    frame_index=index,
                    timestamp_s=timestamp_s,
                    laplacian_var=state.sharpness,
                    accepted=False,
                    reject_reason=reason,
                )
            )
            return state

        if quality.is_degenerate_frame(gray):
            return reject(
                "degenerate (near-uniform frame)", "Nothing to see -- uncover the lens", BAD
            )
        if state.sharpness < state.threshold:
            hint = "Too dark -- add light" if state.brightness < _DARK else "Blurry -- hold steady"
            return reject(
                f"blur (variance {state.sharpness:.1f} < threshold {state.threshold:.1f})",
                hint,
                BAD,
            )

        keypoints, descriptors = selector.detect_orb(gray)
        overlap = 0.0
        if self._last_kp is not None:
            overlap = selector.orb_overlap_ratio(
                self._last_kp, self._last_des, keypoints, descriptors
            )
        state.novelty = 1.0 - overlap if self._last_kp is not None else 1.0
        if self._last_kp is not None and overlap >= self.config.redundancy_overlap_max:
            return reject(
                f"redundant (overlap {overlap:.2f} with last-kept frame)",
                "Keep moving around the subject",
                WARN,
            )

        name = f"{state.accepted:06d}.jpg"
        cv2.imwrite(str(self.output_dir / name), image)
        self.records.append(
            FrameRecord(
                path=name,
                frame_index=index,
                timestamp_s=timestamp_s,
                laplacian_var=state.sharpness,
                accepted=True,
            )
        )
        self._last_kp, self._last_des = keypoints, descriptors
        state.accepted += 1
        state.just_accepted = True
        if state.accepted > 1 and overlap < _LOW_OVERLAP:
            state.message, state.tone = "Slow down -- keep some of the last view in frame", WARN
        elif state.brightness > _BRIGHT:
            state.message, state.tone = "Very bright -- avoid pointing at lights", WARN
        else:
            state.message, state.tone = "Good -- keep going", GOOD
        if state.done:
            state.message, state.tone = "Done -- press q to finish", GOOD
        return state

    def finish(self) -> dict:
        """Write frames.json + capture.json; returns the capture summary."""
        (self.output_dir / "frames.json").write_text(
            json.dumps([r.model_dump() for r in self.records], indent=2)
        )
        summary = {
            "frames_seen": self.state.frames_seen,
            "accepted": self.state.accepted,
            "budget": self.state.budget,
            "final_blur_threshold": self.state.threshold,
        }
        (self.output_dir / CAPTURE_MARKER).write_text(json.dumps(summary, indent=2))
        return summary


# -- HUD --------------------------------------------------------------------------------


def _bar(canvas, x, y, width, fraction, color, label, tick=None) -> None:
    height = 12
    cv2.rectangle(canvas, (x, y), (x + width, y + height), (90, 90, 90), -1)
    fill = int(width * float(np.clip(fraction, 0.0, 1.0)))
    if fill:
        cv2.rectangle(canvas, (x, y), (x + fill, y + height), color, -1)
    if tick is not None:
        tx = x + int(width * float(np.clip(tick, 0.0, 1.0)))
        cv2.line(canvas, (tx, y - 3), (tx, y + height + 3), (255, 255, 255), 2)
    cv2.putText(
        canvas, label, (x, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (235, 235, 235), 1, cv2.LINE_AA
    )


def draw_hud(frame_bgr: np.ndarray, state: CaptureState) -> np.ndarray:
    """The frame with the quality HUD drawn over it (a new array)."""
    canvas = frame_bgr.copy()
    height, width = canvas.shape[:2]
    scale = max(1.0, width / 1280)
    panel_w, panel_h = int(300 * scale), int(150 * scale)
    overlay = canvas.copy()
    cv2.rectangle(overlay, (0, 0), (panel_w, panel_h), (20, 20, 20), -1)
    canvas = cv2.addWeighted(overlay, 0.6, canvas, 0.4, 0)

    x, bar_w = int(14 * scale), int(270 * scale)
    # Sharpness is shown on a scale of 2x the current threshold, with the
    # threshold itself as the tick -- "past the line" reads as "sharp enough".
    sharp_full = max(state.threshold * 2.0, 1.0)
    sharp_ok = state.sharpness >= state.threshold
    _bar(
        canvas,
        x,
        int(34 * scale),
        bar_w,
        state.sharpness / sharp_full,
        GOOD if sharp_ok else BAD,
        "Sharpness",
        tick=0.5,
    )
    novelty_ok = state.novelty >= state.novelty_needed
    _bar(
        canvas,
        x,
        int(76 * scale),
        bar_w,
        state.novelty,
        GOOD if novelty_ok else WARN,
        "New view since last keep",
        tick=state.novelty_needed,
    )
    _bar(
        canvas,
        x,
        int(118 * scale),
        bar_w,
        state.accepted / max(state.budget, 1),
        GOOD,
        f"Frames kept {state.accepted}/{state.budget}",
    )

    message = "PAUSED -- space to resume" if state.paused else state.message
    tone = WARN if state.paused else state.tone
    font_scale = 0.8 * scale
    (text_w, text_h), _ = cv2.getTextSize(message, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)
    tx, ty = max(10, (width - text_w) // 2), height - int(24 * scale)
    cv2.rectangle(
        canvas, (tx - 12, ty - text_h - 12), (tx + text_w + 12, ty + 12), (20, 20, 20), -1
    )
    cv2.putText(
        canvas, message, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, font_scale, tone, 2, cv2.LINE_AA
    )
    if state.just_accepted:
        cv2.rectangle(canvas, (0, 0), (width - 1, height - 1), GOOD, int(6 * scale))
    return canvas


# -- the camera loop -------------------------------------------------------------------------


def _open_source(source: str | int) -> cv2.VideoCapture:
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        raise IngestError(
            f"Could not open camera/source {source!r}.",
            remedy="Check the camera index (--camera 0, 1, ...). On macOS, allow camera access "
            "for your terminal app in System Settings > Privacy & Security > Camera.",
        )
    return capture


def run_capture(
    source: str | int,
    output_dir: Path,
    config: IngestConfig,
    *,
    show: bool = True,
    max_seconds: float | None = None,
) -> dict:
    """Run the capture loop until the operator quits (q/Esc), the budget
    fills, the source ends, or `max_seconds` pass. `show=False` runs
    without a window (headless machines, tests, replaying a recording)."""
    capture = _open_source(source)
    live = LiveCapture(output_dir, config)
    is_file = isinstance(source, str) and Path(source).exists()
    started = time.monotonic()
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            timestamp = (
                capture.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                if is_file
                else time.monotonic() - started
            )
            state = live.state if live.state.paused else live.process(frame, timestamp)
            if show:
                try:
                    cv2.imshow(_WINDOW_NAME, draw_hud(frame, state))
                except cv2.error as exc:
                    raise IngestError(
                        "Can't open a preview window on this machine.",
                        remedy="Run with --no-preview (headless machine or no display).",
                    ) from exc
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key == ord(" "):
                    live.state.paused = not live.state.paused
            elif state.done:
                break
            if max_seconds is not None and time.monotonic() - started > max_seconds:
                break
    finally:
        capture.release()
        if show:
            cv2.destroyAllWindows()
    summary = live.finish()
    logger.info(
        "Capture finished: kept %d of %d frames seen.", summary["accepted"], summary["frames_seen"]
    )
    return summary
