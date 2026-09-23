"""Phase 1 from a folder of images instead of a video (M8, `--from-frames`).

Two kinds of folder come in here:

- A `v2m capture` folder. It already holds only frames the live HUD
  accepted, plus the capture's own `frames.json`. Those decisions are
  kept as they are: the capture gated each frame against the view the
  operator was actually holding, which a second pass after the fact
  can't reproduce.
- Any other folder of photos or frames (jpg/jpeg/png). These go through
  the same sharpness and degenerate-frame gates as a video (adaptive
  threshold, quality.py) and the same even-spread frame-budget trim
  (selector.py). The ORB redundancy gate is skipped: a photo set is
  deliberate, not 30 near-identical frames per second, and COLMAP
  benefits from every distinct view it gets. So is the ">60% blurry"
  abort, which is about shaky video.

Either way the result has the same shape as `extract.run_extract`:
`<output_dir>/frames/000000.jpg ...` renumbered in order, plus
`frames.json`, downscaled to the resolution cap. Phase 2 can't tell
which path produced it.
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path

import cv2

from v2m.config import IngestConfig
from v2m.errors import IngestError
from v2m.phase1_ingest import quality, selector
from v2m.phase1_ingest.extract import _clear_stale_frames
from v2m.phase1_ingest.video_reader import _cap_resolution
from v2m.types import FrameRecord, IngestSummary

logger = logging.getLogger("v2m.phase1_ingest.from_frames")

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
CAPTURE_MARKER = "capture.json"  # written by capture/live.py next to its frames.json


def import_frames(source_dir: Path, output_dir: Path, config: IngestConfig) -> IngestSummary:
    frames_dir = output_dir / "frames"
    if source_dir.resolve() == frames_dir.resolve():
        raise IngestError(
            f"{source_dir} is already this run's frames directory.",
            remedy="Point --from-frames at the capture/photo folder, not at the run itself.",
        )
    images = sorted(p for p in source_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    if not images:
        raise IngestError(
            f"No .jpg/.jpeg/.png images in {source_dir}.",
            remedy="Point --from-frames at a folder of frames (e.g. one written by `v2m capture`).",
        )
    _clear_stale_frames(frames_dir)

    if (source_dir / CAPTURE_MARKER).exists() and (source_dir / "frames.json").exists():
        return _import_capture(source_dir, output_dir, config)
    return _import_images(images, output_dir, config)


def _import_capture(source_dir: Path, output_dir: Path, config: IngestConfig) -> IngestSummary:
    records = [FrameRecord(**r) for r in json.loads((source_dir / "frames.json").read_text())]
    capture = json.loads((source_dir / CAPTURE_MARKER).read_text())
    frames_dir = output_dir / "frames"
    order = 0
    kept = []
    for record in records:
        if not record.accepted:
            continue
        src = source_dir / record.path
        if not src.exists():
            record.accepted, record.reject_reason, record.path = False, "missing file", ""
            continue
        name = f"{order:06d}.jpg"
        image = cv2.imread(str(src))
        capped = _cap_resolution(image, config.resolution_cap_px)
        if capped is image and src.suffix.lower() in {".jpg", ".jpeg"}:
            shutil.copyfile(src, frames_dir / name)
        else:
            cv2.imwrite(str(frames_dir / name), capped)
        record.path = name
        kept.append(record)
        order += 1
    return _write(
        records,
        output_dir,
        rejected_blur=sum(1 for r in records if (r.reject_reason or "").startswith("blur")),
        rejected_redundant=sum(
            1 for r in records if (r.reject_reason or "").startswith("redundant")
        ),
        rejected_budget=0,
        blur_threshold=float(capture.get("final_blur_threshold", config.blur_min_threshold)),
        total_frames=int(capture.get("frames_seen", len(records))),
    )


def _import_images(images: list[Path], output_dir: Path, config: IngestConfig) -> IngestSummary:
    frames_dir = output_dir / "frames"
    variances: dict[int, float] = {}
    reasons: dict[int, str] = {}
    for index, path in enumerate(images):
        image = cv2.imread(str(path))
        if image is None:
            reasons[index] = "unreadable image"
            continue
        gray = quality.to_grayscale(_cap_resolution(image, config.resolution_cap_px))
        variances[index] = quality.laplacian_variance(gray)
        if quality.is_degenerate_frame(gray):
            reasons[index] = "degenerate (near-uniform frame; lens covered or blown out)"

    threshold = quality.compute_blur_threshold(
        list(variances.values()),
        min_threshold=config.blur_min_threshold,
        adaptive_factor=config.blur_adaptive_factor,
    )
    rejected_blur = 0
    for index, variance in variances.items():
        if index not in reasons and variance < threshold:
            reasons[index] = f"blur (variance {variance:.1f} < threshold {threshold:.1f})"
            rejected_blur += 1

    survivors = [i for i in range(len(images)) if i not in reasons]
    kept, dropped = selector.subsample_to_budget(survivors, config.frame_budget)
    for index in dropped:
        reasons[index] = "frame_budget (not selected in even subsample)"

    written: dict[int, str] = {}
    for order, index in enumerate(kept):
        image = _cap_resolution(cv2.imread(str(images[index])), config.resolution_cap_px)
        written[index] = f"{order:06d}.jpg"
        cv2.imwrite(str(frames_dir / written[index]), image)

    records = [
        FrameRecord(
            path=written.get(index, ""),
            frame_index=index,
            timestamp_s=float(index),
            laplacian_var=variances.get(index, 0.0),
            accepted=index in written,
            reject_reason=reasons.get(index),
        )
        for index in range(len(images))
    ]
    return _write(
        records,
        output_dir,
        rejected_blur=rejected_blur,
        rejected_redundant=0,
        rejected_budget=len(dropped),
        blur_threshold=threshold,
        total_frames=len(images),
    )


def _write(
    records: list[FrameRecord],
    output_dir: Path,
    *,
    rejected_blur: int,
    rejected_redundant: int,
    rejected_budget: int,
    blur_threshold: float,
    total_frames: int,
) -> IngestSummary:
    accepted = sum(1 for r in records if r.accepted)
    if accepted < 3:
        raise IngestError(
            f"Only {accepted} usable frame(s) -- reconstruction needs many views.",
            remedy="Capture more frames: walk all the way around the subject.",
        )
    frames_json = output_dir / "frames" / "frames.json"
    frames_json.write_text(json.dumps([r.model_dump() for r in records], indent=2))
    summary = IngestSummary(
        total_frames=total_frames,
        accepted=accepted,
        rejected_blur=rejected_blur,
        rejected_redundant=rejected_redundant,
        rejected_budget=rejected_budget,
        blur_threshold=blur_threshold,
        frames_json_path=str(frames_json.relative_to(output_dir)),
    )
    logger.info("Imported %d frames (%d seen).", accepted, total_frames)
    return summary
