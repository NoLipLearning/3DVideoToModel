"""Phase 1 orchestrator: video -> filtered frame set.

Three passes over the source video, chosen deliberately to keep memory at
O(1) frames regardless of how many survive gating (docs/ARCHITECTURE.md
Section 3.2, "never load whole video into memory" -- read literally: not
just "don't decode the whole file at once", but "never hold more than a
handful of decoded frames at a time," which a naive two-pass design
violates whenever redundancy rejection doesn't shrink the survivor set
much before the budget trim).

  Pass 1: stream every frame, score sharpness only, discard the pixels.
          This is what makes the *adaptive* blur threshold possible --
          the whole distribution is needed before a percentile-relative
          cutoff can be chosen -- while a float-per-frame is negligible
          memory even for a long video.
  Pass 2: stream again; frames that pass the blur gate get a degenerate-
          frame check and an ORB redundancy check against the last KEPT
          frame. Only keypoints/descriptors (small) are retained between
          frames -- never pixel data -- so this pass is also O(1).
          Produces the final set of redundancy-survivor indices, which is
          then subsampled down to the configured frame budget.
  Pass 3: stream a third time; only frames selected by the budget
          subsample get written to disk, as densely-numbered JPEGs
          (000000.jpg, 000001.jpg, ...) in capture order.

This costs 3x the raw decode time of a single pass. That trade is
intentional here: Phase 1's failure mode of concern is memory exhaustion
on long/large videos (Section 3.2), not raw speed -- the real time sink
in this pipeline is Phase 2/3, not this one. If ingest speed ever needs
to improve, the likely win is estimating the blur threshold from a random
sample instead of a full Pass-1 scan, not parallelizing passes 2/3.

If more than 60% of frames fail the blur gate, this aborts before any
Pass-2 work at all -- see docs/ARCHITECTURE.md Section 3.5 ("Rolling
shutter / motion blur"). That ratio is scoped to blur rejections
specifically, not overall attrition: redundancy- and budget-driven
rejection are *expected*, often the majority of frames, and are not a
sign of shaky footage.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import cv2
import numpy as np

from v2m.config import IngestConfig
from v2m.errors import IngestError
from v2m.phase1_ingest import quality, selector, video_reader
from v2m.types import FrameRecord, IngestSummary

logger = logging.getLogger("v2m.phase1_ingest.extract")

_BLUR_ABORT_RATIO = 0.6


def _clear_stale_frames(frames_dir: Path) -> None:
    """Remove leftover JPEGs from a previous run in this directory.

    Without this, re-running extract with a smaller accepted-count than a
    prior run would leave orphaned, higher-numbered files that
    `frames.json` no longer references.
    """
    if frames_dir.exists():
        for stale in frames_dir.glob("*.jpg"):
            stale.unlink()
    else:
        frames_dir.mkdir(parents=True)


def run_extract(video_path: Path, output_dir: Path, config: IngestConfig) -> IngestSummary:
    """Run Phase 1 end to end.

    Writes `<output_dir>/frames/*.jpg` and `<output_dir>/frames/frames.json`.
    Returns a summary for the caller to log / attach to a run manifest.

    Deliberately RunContext-agnostic: `output_dir` is just a directory.
    The CLI (and, from M6, `v2m run`) own manifest bookkeeping around this
    call, not this module.
    """
    frames_dir = output_dir / "frames"
    _clear_stale_frames(frames_dir)

    # -- Pass 1: score every frame's sharpness, keep nothing else -------
    variances: list[float] = []
    total_frames = 0
    for raw in video_reader.iter_frames(
        video_path, resolution_cap_px=config.resolution_cap_px, use_pyav=config.use_pyav
    ):
        gray = quality.to_grayscale(raw.image)
        variances.append(quality.laplacian_variance(gray))
        total_frames += 1

    if total_frames == 0:
        raise IngestError(
            f"No frames could be decoded from {video_path}.",
            remedy="Confirm the file is a valid, non-empty video.",
        )

    blur_threshold = quality.compute_blur_threshold(
        variances,
        min_threshold=config.blur_min_threshold,
        adaptive_factor=config.blur_adaptive_factor,
    )
    blur_pass = [v >= blur_threshold for v in variances]
    blur_rejected_count = blur_pass.count(False)

    if blur_rejected_count / total_frames > _BLUR_ABORT_RATIO:
        raise IngestError(
            f"{blur_rejected_count}/{total_frames} frames "
            f"({blur_rejected_count / total_frames:.0%}) were too blurry to use "
            f"(threshold {blur_threshold:.1f}).",
            remedy="Footage is too shaky or out of focus -- re-shoot walking more "
            "slowly and holding the camera steadier.",
        )

    # -- Pass 2: blur-accepted frames go through degenerate + redundancy
    #    gates. Only the current and last-kept ORB descriptors are held
    #    in memory at any time -- never pixel data. ------------------------
    timestamps: list[float] = [0.0] * total_frames
    reasons: dict[int, str] = {}
    redundancy_survivor_indices: list[int] = []
    last_kept_kp: tuple | None = None
    last_kept_des: np.ndarray | None = None

    for raw in video_reader.iter_frames(
        video_path, resolution_cap_px=config.resolution_cap_px, use_pyav=config.use_pyav
    ):
        idx = raw.frame_index
        timestamps[idx] = raw.timestamp_s

        if not blur_pass[idx]:
            reasons[idx] = f"blur (variance {variances[idx]:.1f} < threshold {blur_threshold:.1f})"
            continue

        gray = quality.to_grayscale(raw.image)
        if quality.is_degenerate_frame(gray):
            reasons[idx] = "degenerate (near-uniform frame; lens covered or blown out)"
            continue

        keypoints, descriptors = selector.detect_orb(gray)
        overlap = 0.0
        if last_kept_kp is not None:
            overlap = selector.orb_overlap_ratio(
                last_kept_kp, last_kept_des, keypoints, descriptors
            )
        if last_kept_kp is not None and overlap >= config.redundancy_overlap_max:
            reasons[idx] = f"redundant (overlap {overlap:.2f} with last-kept frame)"
            continue

        last_kept_kp, last_kept_des = keypoints, descriptors
        redundancy_survivor_indices.append(idx)

    # -- Frame-budget subsampling over redundancy survivors ---------------
    kept_indices, dropped_indices = selector.subsample_to_budget(
        redundancy_survivor_indices, config.frame_budget
    )
    for idx in dropped_indices:
        reasons[idx] = "frame_budget (not selected in even temporal subsample)"
    kept_order = {idx: order for order, idx in enumerate(kept_indices)}

    # -- Pass 3: write only the final accepted set to disk -----------------
    written_paths: dict[int, str] = {}
    for raw in video_reader.iter_frames(
        video_path, resolution_cap_px=config.resolution_cap_px, use_pyav=config.use_pyav
    ):
        idx = raw.frame_index
        if idx not in kept_order:
            continue
        filename = f"{kept_order[idx]:06d}.jpg"
        cv2.imwrite(str(frames_dir / filename), raw.image)
        written_paths[idx] = filename

    # -- Assemble frames.json ------------------------------------------------
    records = [
        FrameRecord(
            path=written_paths.get(idx, ""),
            frame_index=idx,
            timestamp_s=timestamps[idx],
            laplacian_var=variances[idx],
            accepted=idx in kept_order,
            reject_reason=reasons.get(idx),
        )
        for idx in range(total_frames)
    ]
    frames_json_path = frames_dir / "frames.json"
    frames_json_path.write_text(json.dumps([r.model_dump() for r in records], indent=2))

    redundant_count = sum(1 for reason in reasons.values() if reason.startswith("redundant"))
    summary = IngestSummary(
        total_frames=total_frames,
        accepted=len(kept_indices),
        rejected_blur=blur_rejected_count,
        rejected_redundant=redundant_count,
        rejected_budget=len(dropped_indices),
        blur_threshold=blur_threshold,
        frames_json_path=str(frames_json_path.relative_to(output_dir)),
    )
    logger.info(
        "Ingest complete: %d/%d frames accepted (blur -%d, redundant -%d, budget -%d).",
        summary.accepted,
        summary.total_frames,
        summary.rejected_blur,
        summary.rejected_redundant,
        summary.rejected_budget,
    )
    return summary
