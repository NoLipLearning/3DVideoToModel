"""RunContext: manifest lifecycle and resume, in a throwaway directory."""

import pytest

from v2m import run_context as rc
from v2m.errors import ResumeError
from v2m.types import PhaseName, PhaseStatus


@pytest.fixture
def isolated_runs_root(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(rc, "RUNS_ROOT", tmp_path / "runs")
    return tmp_path / "runs"


def test_create_makes_expected_layout(isolated_runs_root):
    ctx = rc.RunContext.create(preset="default", config_snapshot={"mode": "object"})
    assert ctx.frames_dir.exists()
    assert ctx.sfm_dir.exists()
    assert ctx.dense_dir.exists()
    assert ctx.mesh_dir.exists()
    assert ctx.output_dir.exists()
    assert ctx.manifest_path.exists()
    assert ctx.manifest.phases[PhaseName.INGEST].status == PhaseStatus.PENDING


def test_phase_lifecycle_updates_and_persists(isolated_runs_root):
    ctx = rc.RunContext.create(preset="default", config_snapshot={})
    ctx.start_phase(PhaseName.INGEST, input_hash="abc123")
    assert ctx.manifest.phases[PhaseName.INGEST].status == PhaseStatus.RUNNING

    ctx.complete_phase(PhaseName.INGEST, artifacts={"frames_json": "frames/frames.json"})

    reloaded = rc.RunContext.resume(ctx.run_dir)
    ingest_record = reloaded.manifest.phases[PhaseName.INGEST]
    assert ingest_record.status == PhaseStatus.COMPLETE
    assert ingest_record.duration_s is not None
    assert ingest_record.artifacts["frames_json"] == "frames/frames.json"
    assert reloaded.next_incomplete_phase() == PhaseName.SFM_SPARSE


def test_fail_phase_records_error(isolated_runs_root):
    ctx = rc.RunContext.create(preset="default", config_snapshot={})
    ctx.start_phase(PhaseName.SFM_SPARSE)
    ctx.fail_phase(PhaseName.SFM_SPARSE, "registration rate too low")

    reloaded = rc.RunContext.resume(ctx.run_dir)
    record = reloaded.manifest.phases[PhaseName.SFM_SPARSE]
    assert record.status == PhaseStatus.FAILED
    assert record.error == "registration rate too low"


def test_resume_without_manifest_raises(tmp_path):
    with pytest.raises(ResumeError):
        rc.RunContext.resume(tmp_path / "nonexistent")
