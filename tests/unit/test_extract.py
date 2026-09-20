"""End-to-end Phase 1 orchestration against the synthetic fixture.

Runs the real `run_extract()` against tests/fixtures/sample.mp4 and checks
its output against tests/fixtures/sample_ground_truth.json's deliberately
injected blur and duplicate frames. This is the same code path the `v2m
extract` CLI command uses (docs/ARCHITECTURE.md M1 verification), just
invoked directly instead of through Typer.
"""

import json
from pathlib import Path

import pytest

from v2m.config import IngestConfig
from v2m.errors import IngestError
from v2m.phase1_ingest.extract import run_extract

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"
SAMPLE_VIDEO = FIXTURES_DIR / "sample.mp4"
GROUND_TRUTH = json.loads((FIXTURES_DIR / "sample_ground_truth.json").read_text())


@pytest.fixture
def default_ingest_config() -> IngestConfig:
    # frame_budget deliberately left at the default (150), well above this
    # fixture's 70 frames, so nothing is budget-trimmed and every
    # assertion below is a clean read on the blur/redundancy gates alone.
    return IngestConfig(resolution_cap_px=1600)


def test_run_extract_produces_frames_json_and_images(tmp_path, default_ingest_config):
    summary = run_extract(SAMPLE_VIDEO, tmp_path, default_ingest_config)

    frames_json = tmp_path / "frames" / "frames.json"
    assert frames_json.exists()
    assert summary.frames_json_path == "frames/frames.json"

    records = json.loads(frames_json.read_text())
    assert len(records) == GROUND_TRUTH["total_frames"]
    assert summary.total_frames == GROUND_TRUTH["total_frames"]
    assert summary.accepted + summary.rejected_blur + summary.rejected_redundant == len(records)

    accepted_records = [r for r in records if r["accepted"]]
    assert len(accepted_records) == summary.accepted
    for record in accepted_records:
        assert (tmp_path / "frames" / record["path"]).exists()
        assert record["reject_reason"] is None

    # Accepted files are densely numbered from 000000.jpg with no gaps.
    accepted_names = sorted(r["path"] for r in accepted_records)
    expected_names = [f"{i:06d}.jpg" for i in range(len(accepted_records))]
    assert accepted_names == expected_names


def test_run_extract_rejects_synthetic_blur_frames(tmp_path, default_ingest_config):
    run_extract(SAMPLE_VIDEO, tmp_path, default_ingest_config)
    records = {
        r["frame_index"]: r for r in json.loads((tmp_path / "frames" / "frames.json").read_text())
    }
    for idx in GROUND_TRUTH["synthetic_blur_frame_indices"]:
        record = records[idx]
        assert record["accepted"] is False
        assert record["reject_reason"].startswith("blur")


def test_run_extract_rejects_synthetic_duplicate_frames_as_redundant(
    tmp_path, default_ingest_config
):
    run_extract(SAMPLE_VIDEO, tmp_path, default_ingest_config)
    records = {
        r["frame_index"]: r for r in json.loads((tmp_path / "frames" / "frames.json").read_text())
    }
    for idx in GROUND_TRUTH["synthetic_duplicate_frame_indices"]:
        record = records[idx]
        assert record["accepted"] is False
        assert record["reject_reason"].startswith("redundant")


def test_run_extract_every_rejected_frame_has_a_reason(tmp_path, default_ingest_config):
    run_extract(SAMPLE_VIDEO, tmp_path, default_ingest_config)
    records = json.loads((tmp_path / "frames" / "frames.json").read_text())
    for record in records:
        if not record["accepted"]:
            assert record["reject_reason"]
        else:
            assert record["reject_reason"] is None


def test_run_extract_enforces_frame_budget(tmp_path, default_ingest_config):
    config = default_ingest_config.model_copy(update={"frame_budget": 20})
    summary = run_extract(SAMPLE_VIDEO, tmp_path, config)
    assert summary.accepted == 20
    assert summary.rejected_budget > 0


def test_run_extract_is_idempotent_and_cleans_stale_files(tmp_path, default_ingest_config):
    wide_budget_config = default_ingest_config.model_copy(update={"frame_budget": 150})
    first = run_extract(SAMPLE_VIDEO, tmp_path, wide_budget_config)

    narrow_budget_config = default_ingest_config.model_copy(update={"frame_budget": 5})
    second = run_extract(SAMPLE_VIDEO, tmp_path, narrow_budget_config)

    assert second.accepted == 5
    assert second.accepted < first.accepted
    remaining_jpgs = sorted((tmp_path / "frames").glob("*.jpg"))
    assert len(remaining_jpgs) == 5  # no orphaned files from the first, wider run


def test_run_extract_missing_video_raises(tmp_path, default_ingest_config):
    with pytest.raises(IngestError):
        run_extract(tmp_path / "nope.mp4", tmp_path, default_ingest_config)
