"""M8: live-capture gating, the HUD, and `--from-frames` import.

The capture loop is driven by frames we construct (or by the fixture
video standing in for a camera) -- the gating logic is independent of
where frames come from, and no display is needed."""

import json
from pathlib import Path

import cv2
import numpy as np
import pytest
import trimesh

from v2m import pipeline
from v2m.capture.live import CaptureState, LiveCapture, draw_hud, run_capture
from v2m.config import IngestConfig, load_config
from v2m.errors import IngestError
from v2m.phase1_ingest.from_frames import import_frames
from v2m.types import PhaseName, PhaseStatus

VIDEO = Path(__file__).parent.parent / "fixtures" / "sample.mp4"


def _texture(seed=0, size=(480, 1200)) -> np.ndarray:
    """A wide, richly textured strip to pan across."""
    rng = np.random.default_rng(seed)
    image = np.full((*size, 3), 128, np.uint8)
    for _ in range(900):
        center = (int(rng.integers(0, size[1])), int(rng.integers(0, size[0])))
        color = tuple(int(c) for c in rng.integers(0, 255, 3))
        cv2.circle(image, center, int(rng.integers(3, 18)), color, -1)
    return image


def _view(strip: np.ndarray, offset: int, width: int = 640) -> np.ndarray:
    return np.ascontiguousarray(strip[:, offset : offset + width])


def test_panning_views_are_kept_and_repeats_are_rejected(tmp_path):
    strip = _texture()
    live = LiveCapture(tmp_path, IngestConfig())
    for step in range(12):
        live.process(_view(strip, step * 40), step / 30)  # slow pan: 40px per frame
    for _ in range(5):
        live.process(_view(strip, 11 * 40), 1.0)  # camera held still

    reasons = [r.reject_reason or "accepted" for r in live.records]
    assert live.state.accepted >= 3
    assert any(reason.startswith("redundant") for reason in reasons)
    # Held still at the end: nothing new, so none of those are kept.
    assert not any(r.accepted for r in live.records[-5:])


def test_blurry_and_blank_frames_are_rejected_with_a_reason(tmp_path):
    strip = _texture()
    live = LiveCapture(tmp_path, IngestConfig())
    live.process(_view(strip, 0), 0.0)
    blurred = cv2.GaussianBlur(_view(strip, 200), (0, 0), 12)
    state = live.process(blurred, 0.1)
    assert live.records[-1].reject_reason.startswith("blur")
    assert "hold steady" in state.message.lower()

    live.process(np.full((480, 640, 3), 90, np.uint8), 0.2)
    assert live.records[-1].reject_reason.startswith("degenerate")


def test_the_budget_ends_the_capture(tmp_path):
    strip = _texture()
    live = LiveCapture(tmp_path, IngestConfig(frame_budget=3))
    for step in range(12):
        live.process(_view(strip, step * 180), step)
    assert live.state.accepted == 3
    assert live.state.done
    assert "Done" in live.state.message
    assert len(list(tmp_path.glob("*.jpg"))) == 3


def test_hud_draws_over_the_frame_without_resizing_it():
    frame = _view(_texture(), 0)
    state = CaptureState(accepted=12, budget=150, sharpness=300, threshold=120, novelty=0.4)
    hud = draw_hud(frame, state)
    assert hud.shape == frame.shape
    assert not np.array_equal(hud, frame)
    assert np.array_equal(frame, _view(_texture(), 0))  # input untouched


def test_run_capture_replays_a_video_as_the_camera(tmp_path):
    summary = run_capture(str(VIDEO), tmp_path / "cap", IngestConfig(), show=False)
    assert summary["frames_seen"] > 50
    assert summary["accepted"] >= 20
    records = json.loads((tmp_path / "cap" / "frames.json").read_text())
    assert sum(r["accepted"] for r in records) == summary["accepted"]
    assert (tmp_path / "cap" / "capture.json").exists()


def test_a_missing_camera_says_how_to_fix_it(tmp_path):
    with pytest.raises(IngestError) as exc_info:
        run_capture(
            str(tmp_path / "no-such-camera.mp4"), tmp_path / "cap", IngestConfig(), show=False
        )
    assert "Camera" in exc_info.value.remedy


# -- --from-frames ---------------------------------------------------------------------------


def test_a_capture_folder_is_imported_with_its_own_decisions(tmp_path):
    strip = _texture()
    live = LiveCapture(tmp_path / "cap", IngestConfig())
    for step in range(10):
        live.process(_view(strip, step * 55), step)
    live.process(cv2.GaussianBlur(_view(strip, 300), (0, 0), 12), 11)
    live.finish()

    summary = import_frames(tmp_path / "cap", tmp_path / "run", IngestConfig())

    assert summary.accepted == live.state.accepted
    assert summary.rejected_blur >= 1
    written = sorted((tmp_path / "run" / "frames").glob("*.jpg"))
    assert [p.name for p in written] == [f"{i:06d}.jpg" for i in range(summary.accepted)]


def test_a_plain_photo_folder_is_gated_and_trimmed_to_budget(tmp_path):
    strip = _texture()
    photos = tmp_path / "photos"
    photos.mkdir()
    for i in range(8):
        cv2.imwrite(str(photos / f"IMG_{i:04d}.png"), _view(strip, i * 70))
    cv2.imwrite(str(photos / "IMG_blurry.png"), cv2.GaussianBlur(_view(strip, 0), (0, 0), 15))

    summary = import_frames(photos, tmp_path / "run", IngestConfig(frame_budget=5))

    assert summary.total_frames == 9
    assert summary.rejected_blur == 1
    assert summary.accepted == 5 and summary.rejected_budget == 3


def test_too_few_frames_is_an_error_with_a_remedy(tmp_path):
    photos = tmp_path / "photos"
    photos.mkdir()
    cv2.imwrite(str(photos / "only.jpg"), _view(_texture(), 0))
    with pytest.raises(IngestError) as exc_info:
        import_frames(photos, tmp_path / "run", IngestConfig())
    assert exc_info.value.remedy


def test_a_run_started_from_frames_records_its_source(tmp_path):
    strip = _texture()
    photos = tmp_path / "photos"
    photos.mkdir()
    for i in range(6):
        cv2.imwrite(str(photos / f"{i}.jpg"), _view(strip, i * 90))

    ctx, cfg = pipeline.start_run(None, "fast", frames_dir=photos, run_dir=tmp_path / "run")
    pipeline.run_phase(ctx, cfg, PhaseName.INGEST)

    assert ctx.manifest.source_video is None
    assert ctx.manifest.source_frames == str(photos.resolve())
    assert ctx.manifest.phases[PhaseName.INGEST].status == PhaseStatus.COMPLETE
    assert ctx.manifest.phases[PhaseName.INGEST].summary["accepted"] == 6


def test_start_run_wants_exactly_one_source(tmp_path):
    with pytest.raises(IngestError):
        pipeline.start_run(None, "fast")
    with pytest.raises(IngestError):
        pipeline.start_run(VIDEO, "fast", frames_dir=tmp_path)


@pytest.mark.slow
def test_capture_then_reconstruct_from_frames(tmp_path):
    """M8 verify, with the fixture video standing in for the camera:
    capture -> `--from-frames` run -> printable model."""
    cfg = load_config("fast")
    run_capture(str(VIDEO), tmp_path / "cap", cfg.ingest, show=False)

    ctx, cfg = pipeline.start_run(None, "fast", frames_dir=tmp_path / "cap", run_dir=tmp_path / "r")
    result = pipeline.run_pipeline(ctx, cfg)

    assert result.phases_run == list(PhaseName)
    assert trimesh.load(str(ctx.output_dir / "model.stl")).is_watertight
