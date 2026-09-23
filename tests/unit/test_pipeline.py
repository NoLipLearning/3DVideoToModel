"""pipeline.py orchestration, resume, and invalidation -- with the five
phases replaced by instant fakes, so these run in milliseconds. The real
phases end to end are tests/unit/test_run_end_to_end.py (slow)."""

from collections import namedtuple

import numpy as np
import pytest

from v2m import pipeline, preflight
from v2m.config import PipelineConfig, config_from_snapshot, load_config
from v2m.errors import CapabilityError, ConfigError, IngestError, ResumeError, SfMError
from v2m.run_context import RunContext
from v2m.types import (
    DenseResult,
    IngestSummary,
    MeshReport,
    PhaseName,
    PhaseStatus,
    PrintReport,
    SfmResult,
)

_FAKE_RESULTS = {
    PhaseName.INGEST: IngestSummary(
        total_frames=10,
        accepted=8,
        rejected_blur=1,
        rejected_redundant=1,
        rejected_budget=0,
        blur_threshold=30.0,
        frames_json_path="frames/frames.json",
    ),
    PhaseName.SFM_SPARSE: SfmResult(
        num_images_registered=8,
        num_images_total=8,
        mean_reprojection_error_px=0.5,
        mean_track_length=4.0,
        sparse_points_path="sparse.ply",
        cameras_path="cameras.json",
    ),
    PhaseName.SFM_DENSE: DenseResult(
        backend="sparse_only", num_points=1000, dense_points_path="dense.ply"
    ),
    PhaseName.MESH: MeshReport(
        num_vertices=100, num_faces=196, is_manifold=True, mesh_path="cleaned.ply"
    ),
    PhaseName.PRINT_PREP: PrintReport(
        watertight=True,
        volume_mm3=1000.0,
        bbox_mm=(10.0, 10.0, 10.0),
        repair_rung_used=1,
        scale_method="manual",
    ),
}


class FakePhases:
    """Stands in for pipeline._execute; records calls and can fail once."""

    def __init__(self):
        self.calls: list[PhaseName] = []
        self.fail_at: PhaseName | None = None
        self.fail_with: BaseException | None = None
        self.print_options: list[pipeline.PrintOptions] = []

    def __call__(self, ctx, cfg, phase, print_options, depth_estimator):
        self.calls.append(phase)
        if phase == PhaseName.PRINT_PREP:
            self.print_options.append(print_options)
        if phase == self.fail_at:
            self.fail_at = None
            raise self.fail_with
        return _FAKE_RESULTS[phase], {"artifact": "x"}


@pytest.fixture
def fake(monkeypatch):
    phases = FakePhases()
    monkeypatch.setattr(pipeline, "_execute", phases)
    return phases


@pytest.fixture
def video(tmp_path):
    path = tmp_path / "clip.mp4"
    path.write_bytes(b"not really a video, but hashable")
    return path


def _new_run(tmp_path, video, **kwargs):
    return pipeline.start_run(video, "object", run_dir=tmp_path / "run", **kwargs)


def _statuses(ctx):
    return {p: ctx.manifest.phases[p].status for p in PhaseName}


# -- overrides / config --------------------------------------------------------------


def test_parse_overrides_types_values():
    parsed = pipeline.parse_overrides(
        ["print_prep.slab_thickness_mm=5", "sfm.use_clahe=true", "dense.backend=sparse_only"]
    )
    assert parsed == {
        "print_prep.slab_thickness_mm": 5,
        "sfm.use_clahe": True,
        "dense.backend": "sparse_only",
    }


def test_parse_overrides_rejects_a_missing_equals():
    with pytest.raises(ConfigError):
        pipeline.parse_overrides(["print_prep.slab_thickness_mm"])


def test_unknown_override_key_is_rejected_not_ignored():
    with pytest.raises(ConfigError) as exc_info:
        load_config("object", {"print_prep.slab_thickness": 5})
    assert exc_info.value.remedy
    with pytest.raises(ConfigError):
        config_from_snapshot(PipelineConfig().model_dump(mode="json"), {"nope.key": 1})


# -- full runs ----------------------------------------------------------------------------


def test_fresh_run_executes_every_phase_in_order_and_writes_a_report(tmp_path, video, fake):
    ctx, cfg = _new_run(tmp_path, video)
    result = pipeline.run_pipeline(ctx, cfg, check_resources=False)

    assert fake.calls == list(PhaseName)
    assert all(s == PhaseStatus.COMPLETE for s in _statuses(ctx).values())
    assert result.print_report == _FAKE_RESULTS[PhaseName.PRINT_PREP]
    assert result.report_path.exists()
    # Each phase's result model is kept in the manifest.
    reloaded = RunContext.resume(ctx.run_dir)
    assert reloaded.manifest.phases[PhaseName.MESH].summary["num_faces"] == 196
    assert reloaded.manifest.phases[PhaseName.INGEST].input_hash is not None


def test_a_failed_run_records_the_remedy_and_resume_skips_finished_phases(tmp_path, video, fake):
    ctx, cfg = _new_run(tmp_path, video)
    fake.fail_at = PhaseName.SFM_DENSE
    fake.fail_with = SfMError("too few correspondences", remedy="re-shoot with more texture")

    with pytest.raises(SfMError):
        pipeline.run_pipeline(ctx, cfg, check_resources=False)

    record = ctx.manifest.phases[PhaseName.SFM_DENSE]
    assert record.status == PhaseStatus.FAILED
    assert record.remedy == "re-shoot with more texture"
    assert ctx.manifest.phases[PhaseName.MESH].status == PhaseStatus.PENDING
    report = ctx.report_path.read_text()
    assert "too few correspondences" in report and "re-shoot with more texture" in report

    fake.calls.clear()
    ctx, cfg = pipeline.prepare_resume(ctx.run_dir)
    result = pipeline.run_pipeline(ctx, cfg, check_resources=False)

    assert fake.calls == [PhaseName.SFM_DENSE, PhaseName.MESH, PhaseName.PRINT_PREP]
    assert result.phases_skipped == [PhaseName.INGEST, PhaseName.SFM_SPARSE]


def test_a_phase_left_running_by_a_hard_kill_is_rerun(tmp_path, video, fake, caplog):
    ctx, cfg = _new_run(tmp_path, video)
    pipeline.run_phase(ctx, cfg, PhaseName.INGEST)
    ctx.start_phase(PhaseName.SFM_SPARSE)  # ... and then SIGKILL: nothing records the end

    fake.calls.clear()
    ctx, cfg = pipeline.prepare_resume(ctx.run_dir)
    pipeline.run_pipeline(ctx, cfg, check_resources=False)

    assert fake.calls[0] == PhaseName.SFM_SPARSE
    assert PhaseName.INGEST not in fake.calls
    assert "interrupted mid-run" in caplog.text


def test_ctrl_c_is_recorded_as_an_interruption(tmp_path, video, fake):
    ctx, cfg = _new_run(tmp_path, video)
    fake.fail_at = PhaseName.MESH
    fake.fail_with = KeyboardInterrupt()

    with pytest.raises(KeyboardInterrupt):
        pipeline.run_pipeline(ctx, cfg, check_resources=False)

    assert ctx.manifest.phases[PhaseName.MESH].status == PhaseStatus.FAILED
    assert ctx.manifest.phases[PhaseName.MESH].error == "interrupted by user"


def test_resume_of_a_finished_run_does_nothing(tmp_path, video, fake):
    ctx, cfg = _new_run(tmp_path, video)
    pipeline.run_pipeline(ctx, cfg, check_resources=False)
    fake.calls.clear()

    ctx, cfg = pipeline.prepare_resume(ctx.run_dir)
    result = pipeline.run_pipeline(ctx, cfg, check_resources=False)

    assert fake.calls == []
    assert result.print_report is None or result.print_report.scale_method == "manual"


# -- invalidation on resume ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("override", "first_rerun"),
    [
        ("print_prep.slab_thickness_mm=5", PhaseName.PRINT_PREP),
        ("mode=scene", PhaseName.PRINT_PREP),
        ("mesh.max_faces=1000", PhaseName.MESH),
        ("dense.backend=sparse_only", PhaseName.SFM_DENSE),
        ("sfm.matching_overlap=20", PhaseName.SFM_SPARSE),
        ("ingest.frame_budget=50", PhaseName.INGEST),
    ],
)
def test_a_config_change_reruns_exactly_the_phases_that_read_it(
    tmp_path, video, fake, override, first_rerun
):
    ctx, cfg = _new_run(tmp_path, video)
    pipeline.run_pipeline(ctx, cfg, check_resources=False)
    fake.calls.clear()

    ctx, cfg = pipeline.prepare_resume(ctx.run_dir, overrides=pipeline.parse_overrides([override]))
    pipeline.run_pipeline(ctx, cfg, check_resources=False)

    order = list(PhaseName)
    assert fake.calls == order[order.index(first_rerun) :]
    ((key, value),) = pipeline.parse_overrides([override]).items()
    section, _, field = key.partition(".")
    saved = RunContext.resume(ctx.run_dir).manifest.config
    assert (saved[section][field] if field else saved[section]) == value


def test_an_override_equal_to_the_current_value_reruns_nothing(tmp_path, video, fake):
    ctx, cfg = _new_run(tmp_path, video)
    pipeline.run_pipeline(ctx, cfg, check_resources=False)
    fake.calls.clear()

    current = cfg.print_prep.slab_thickness_mm
    ctx, cfg = pipeline.prepare_resume(
        ctx.run_dir, overrides={"print_prep.slab_thickness_mm": current}
    )
    pipeline.run_pipeline(ctx, cfg, check_resources=False)
    assert fake.calls == []


def test_rerun_from_resets_that_phase_onward(tmp_path, video, fake):
    ctx, cfg = _new_run(tmp_path, video)
    pipeline.run_pipeline(ctx, cfg, check_resources=False)
    fake.calls.clear()

    ctx, cfg = pipeline.prepare_resume(ctx.run_dir, rerun_from=PhaseName.MESH)
    pipeline.run_pipeline(ctx, cfg, check_resources=False)
    assert fake.calls == [PhaseName.MESH, PhaseName.PRINT_PREP]


def test_scale_options_persist_across_resume_and_a_new_one_reruns_phase_4(tmp_path, video, fake):
    options = pipeline.PrintOptions(scale_two_point=(np.zeros(3), np.array([1.0, 0.0, 0.0]), 50.0))
    ctx, cfg = _new_run(tmp_path, video, print_options=options)
    fake.fail_at = PhaseName.MESH
    fake.fail_with = SfMError("boom")
    with pytest.raises(SfMError):
        pipeline.run_pipeline(ctx, cfg, check_resources=False)

    # Resumed without scale flags: the original two-point reference is used.
    ctx, cfg = pipeline.prepare_resume(ctx.run_dir)
    pipeline.run_pipeline(ctx, cfg, check_resources=False)
    used = fake.print_options[-1]
    assert used.scale_two_point[2] == 50.0
    np.testing.assert_allclose(used.scale_two_point[1], [1.0, 0.0, 0.0])

    # A different scale reference on resume re-runs only Phase 4.
    fake.calls.clear()
    ctx, cfg = pipeline.prepare_resume(
        ctx.run_dir, print_options=pipeline.PrintOptions(scale_factor=2.0)
    )
    pipeline.run_pipeline(ctx, cfg, check_resources=False)
    assert fake.calls == [PhaseName.PRINT_PREP]
    assert fake.print_options[-1].scale_factor == 2.0


def test_resuming_with_a_different_video_is_refused(tmp_path, video, fake):
    ctx, cfg = _new_run(tmp_path, video)
    pipeline.run_pipeline(ctx, cfg, check_resources=False)
    other = tmp_path / "other.mp4"
    other.write_bytes(b"a different clip")

    with pytest.raises(ResumeError) as exc_info:
        pipeline.prepare_resume(ctx.run_dir, video=other)
    assert "--rerun-from ingest" in exc_info.value.remedy

    # ... unless the user asks to rebuild from it.
    fake.calls.clear()
    ctx, cfg = pipeline.prepare_resume(ctx.run_dir, video=other, rerun_from=PhaseName.INGEST)
    pipeline.run_pipeline(ctx, cfg, check_resources=False)
    assert fake.calls == list(PhaseName)


def test_starting_over_an_existing_run_dir_is_refused(tmp_path, video):
    _new_run(tmp_path, video)
    with pytest.raises(ResumeError):
        _new_run(tmp_path, video)


def test_missing_source_video_is_a_recorded_failure_with_a_remedy(tmp_path, video, fake):
    ctx, cfg = _new_run(tmp_path, video)
    video.unlink()
    with pytest.raises(IngestError) as exc_info:
        pipeline.run_pipeline(ctx, cfg, check_resources=False)
    assert "no longer exists" in exc_info.value.message
    record = ctx.manifest.phases[PhaseName.INGEST]
    assert record.status == PhaseStatus.FAILED
    assert "--resume" in record.remedy


# -- preflight --------------------------------------------------------------------------------

_Memory = namedtuple("_Memory", "total available")


def _cfg(**overrides):
    return load_config("object", {"dense.backend": "monodepth_tsdf", **overrides})


def test_preflight_estimate_is_dominated_by_the_tsdf_volume():
    estimate = preflight.estimate(_cfg(), list(PhaseName))
    # 400^3 voxels x 48B = 3.1GB, + depth model + baseline.
    assert 4.5e9 < estimate.peak_ram_bytes < 6e9
    fast = preflight.estimate(load_config("fast"), list(PhaseName))
    assert fast.peak_ram_bytes < estimate.peak_ram_bytes
    assert fast.disk_bytes < estimate.disk_bytes


def test_preflight_refuses_a_machine_that_is_too_small(tmp_path, monkeypatch):
    monkeypatch.setattr(preflight.psutil, "virtual_memory", lambda: _Memory(2e9, 2e9))
    with pytest.raises(CapabilityError) as exc_info:
        preflight.check(_cfg(), tmp_path, list(PhaseName))
    assert "--preset fast" in exc_info.value.remedy


def test_preflight_only_warns_when_memory_is_merely_busy(tmp_path, monkeypatch):
    monkeypatch.setattr(preflight.psutil, "virtual_memory", lambda: _Memory(64e9, 1e9))
    warnings = preflight.check(_cfg(), tmp_path, list(PhaseName))
    assert any("swapping" in w for w in warnings)


def test_preflight_refuses_when_the_disk_is_full(tmp_path, monkeypatch):
    Usage = namedtuple("Usage", "total used free")
    monkeypatch.setattr(preflight.shutil, "disk_usage", lambda _: Usage(1e12, 1e12, 1e6))
    with pytest.raises(CapabilityError) as exc_info:
        preflight.check(_cfg(), tmp_path, list(PhaseName))
    assert "disk" in exc_info.value.message


def test_preflight_only_counts_pending_phases():
    everything = preflight.estimate(_cfg(), list(PhaseName))
    phase4_only = preflight.estimate(_cfg(), [PhaseName.PRINT_PREP])
    assert phase4_only.disk_bytes < everything.disk_bytes / 10
    assert phase4_only.peak_ram_bytes < everything.peak_ram_bytes
