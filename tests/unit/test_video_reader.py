"""Streaming decode against the synthetic fixture (tests/fixtures/sample.mp4).

Regenerate the fixture with:
    uv run python tests/fixtures/make_synthetic_video.py
"""

import json
from pathlib import Path

import pytest

from v2m.errors import IngestError
from v2m.phase1_ingest import video_reader

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"
SAMPLE_VIDEO = FIXTURES_DIR / "sample.mp4"
GROUND_TRUTH = json.loads((FIXTURES_DIR / "sample_ground_truth.json").read_text())


def test_iter_frames_reads_expected_frame_count():
    frames = list(video_reader.iter_frames(SAMPLE_VIDEO, resolution_cap_px=1600))
    assert len(frames) == GROUND_TRUTH["total_frames"]
    assert [f.frame_index for f in frames] == list(range(GROUND_TRUTH["total_frames"]))


def test_iter_frames_applies_resolution_cap():
    frames = list(video_reader.iter_frames(SAMPLE_VIDEO, resolution_cap_px=100))
    for raw in frames:
        assert max(raw.image.shape[:2]) <= 100


def test_iter_frames_no_cap_needed_keeps_original_size():
    frames = list(video_reader.iter_frames(SAMPLE_VIDEO, resolution_cap_px=1600))
    assert frames[0].image.shape[:2] == (
        GROUND_TRUTH["image_size_px"],
        GROUND_TRUTH["image_size_px"],
    )


def test_iter_frames_timestamps_are_monotonic():
    frames = list(video_reader.iter_frames(SAMPLE_VIDEO, resolution_cap_px=1600))
    timestamps = [f.timestamp_s for f in frames]
    assert timestamps == sorted(timestamps)
    assert timestamps[0] >= 0.0


def test_iter_frames_missing_file_raises_ingest_error(tmp_path):
    with pytest.raises(IngestError):
        list(video_reader.iter_frames(tmp_path / "does_not_exist.mp4", resolution_cap_px=1600))


def test_probe_frame_count_close_to_ground_truth():
    count = video_reader.probe_frame_count(SAMPLE_VIDEO)
    assert count is not None
    # Container metadata can be off by a frame or two.
    assert abs(count - GROUND_TRUTH["total_frames"]) <= 2
