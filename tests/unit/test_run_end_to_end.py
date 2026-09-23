"""M6 end to end: the real five phases through `run_pipeline()` and the
`v2m run` CLI, on tests/fixtures/sample.mp4.

docs/ARCHITECTURE.md Section 4 (M6) verify: "one command ... ->
output/model.stl. Kill mid-run, re-run with --resume, confirm it
restarts at the right phase." The kill itself is simulated in
test_pipeline.py (a phase left RUNNING) and was exercised for real with
SIGKILL during development -- see the M6 notes in docs/ARCHITECTURE.md.
Here: a complete default-preset run, a resume that changes a Phase 4
setting and re-runs only Phase 4, and a fresh CLI run on the fast preset.

The default preset's dense backend needs Depth-Anything-V2 weights,
which this sandbox can't download (see CLAUDE.md), so the API-level run
injects the same geometric depth test double the M3 tests use. The CLI
can't take a test double, so the CLI-level fresh run uses `--preset
fast` (sparse-only dense, no depth model).
"""

import json
from pathlib import Path

import pytest
import trimesh
from typer.testing import CliRunner

from v2m import pipeline
from v2m.cli import app
from v2m.run_context import RunContext
from v2m.types import PhaseName, PhaseStatus

pytestmark = pytest.mark.slow

VIDEO = Path(__file__).parent.parent / "fixtures" / "sample.mp4"


class _LazyGeometricEstimator:
    """The geometric test double needs the run's own frames.json, which
    doesn't exist until Phase 1 has run inside the same pipeline call."""

    def __init__(self, run_dir: Path, estimator_cls):
        self._run_dir = run_dir
        self._cls = estimator_cls
        self._inner = None

    def predict_disparity(self, image_bgr, image_name=None):
        if self._inner is None:
            frames = json.loads((self._run_dir / "frames" / "frames.json").read_text())
            self._inner = self._cls({f["path"]: f["frame_index"] for f in frames if f["accepted"]})
        return self._inner.predict_disparity(image_bgr, image_name)


@pytest.fixture(scope="module")
def full_run(tmp_path_factory, geometric_depth_estimator_cls):
    run_dir = tmp_path_factory.mktemp("e2e") / "run"
    ctx, cfg = pipeline.start_run(VIDEO, "object", run_dir=run_dir)
    result = pipeline.run_pipeline(
        ctx,
        cfg,
        depth_estimator=_LazyGeometricEstimator(run_dir, geometric_depth_estimator_cls),
    )
    return {"ctx": ctx, "result": result}


def test_one_call_turns_the_video_into_a_printable_stl(full_run):
    ctx = full_run["ctx"]
    assert full_run["result"].phases_run == list(PhaseName)
    model = trimesh.load(str(ctx.output_dir / "model.stl"))
    assert model.is_watertight
    assert model.volume > 0
    assert full_run["result"].print_report.watertight


def test_the_manifest_and_report_describe_every_phase(full_run):
    ctx = RunContext.resume(full_run["ctx"].run_dir)
    for phase in PhaseName:
        record = ctx.manifest.phases[phase]
        assert record.status == PhaseStatus.COMPLETE
        assert record.summary, f"{phase} has no summary"
        assert record.duration_s is not None
    assert ctx.manifest.phases[PhaseName.SFM_DENSE].summary["backend"] == "monodepth_tsdf"

    report = ctx.report_path.read_text()
    assert "Complete" in report
    assert "data:image/png;base64," in report  # the model preview
    assert "data:image/jpeg;base64," in report  # frame thumbnails


def test_cli_resume_with_a_phase4_setting_reruns_only_phase4(full_run):
    run_dir = full_run["ctx"].run_dir
    before = RunContext.resume(run_dir).manifest.phases

    outcome = CliRunner().invoke(
        app, ["run", "--resume", str(run_dir), "--set", "print_prep.scale_target_size_mm=100"]
    )

    assert outcome.exit_code == 0, outcome.output
    after = RunContext.resume(run_dir).manifest.phases
    for phase in (PhaseName.INGEST, PhaseName.SFM_SPARSE, PhaseName.SFM_DENSE, PhaseName.MESH):
        assert after[phase].started_at == before[phase].started_at, f"{phase} re-ran"
    assert after[PhaseName.PRINT_PREP].started_at != before[PhaseName.PRINT_PREP].started_at
    assert max(after[PhaseName.PRINT_PREP].summary["bbox_mm"]) == pytest.approx(100.0)
    assert "already complete, skipped" in outcome.output


def test_cli_fresh_run_on_the_fast_preset(tmp_path):
    run_dir = tmp_path / "fast_run"
    outcome = CliRunner().invoke(
        app, ["run", str(VIDEO), "--preset", "fast", "--run-dir", str(run_dir)]
    )

    assert outcome.exit_code == 0, outcome.output
    assert trimesh.load(str(run_dir / "output" / "model.stl")).is_watertight
    manifest = RunContext.resume(run_dir).manifest
    assert manifest.phases[PhaseName.SFM_DENSE].summary["backend"] == "sparse_only"
    assert (run_dir / "report.html").exists()


def test_cli_refuses_to_start_without_a_video():
    outcome = CliRunner().invoke(app, ["run"])
    assert outcome.exit_code == 1
    assert "--resume" in outcome.output
