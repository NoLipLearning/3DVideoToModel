"""The M7 web app: upload -> queue -> SSE progress -> result -> download.

The job manager and pipeline bookkeeping are real; only the five phases
are swapped for instant fakes (same approach as test_pipeline.py), so the
HTTP surface is tested in seconds. `test_real_pipeline_through_the_web`
(slow) runs the actual phases on the fixture video."""

import json
import threading
import time
from pathlib import Path

import pytest
import trimesh
from fastapi.testclient import TestClient

from v2m import pipeline
from v2m.errors import SfMError
from v2m.phase4_print import export
from v2m.types import (
    DenseResult,
    IngestSummary,
    MeshReport,
    PhaseName,
    PrintReport,
    SfmResult,
)
from v2m.web.app import create_app

VIDEO = Path(__file__).parent.parent / "fixtures" / "sample.mp4"

_RESULTS = {
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
    PhaseName.SFM_DENSE: DenseResult(backend="fake", num_points=10, dense_points_path="dense.ply"),
    PhaseName.MESH: MeshReport(num_vertices=8, num_faces=12, is_manifold=True, mesh_path="x.ply"),
}


class FakePhases:
    def __init__(self):
        self.fail_at = None
        self.gate: threading.Event | None = None

    def __call__(self, ctx, cfg, phase, print_options, depth_estimator):
        if self.gate is not None and phase == PhaseName.INGEST:
            assert self.gate.wait(10)
        if phase == self.fail_at:
            self.fail_at = None
            raise SfMError("not enough texture", remedy="film something with more detail")
        if phase != PhaseName.PRINT_PREP:
            return _RESULTS[phase], {}
        box = trimesh.creation.box(extents=[40, 30, 20])
        box.apply_translation([0, 0, 10])
        export.export_mesh(box, ctx.output_dir)
        report = PrintReport(
            watertight=True,
            volume_mm3=float(box.volume),
            bbox_mm=(40.0, 30.0, 20.0),
            repair_rung_used=1,
            scale_method="aruco" if print_options.aruco_marker_mm else "fit_to_build_volume",
        )
        (ctx.output_dir / "print_report.json").write_text(report.model_dump_json())
        return report, {}


@pytest.fixture
def fake(monkeypatch):
    phases = FakePhases()
    monkeypatch.setattr(pipeline, "_execute", phases)
    return phases


@pytest.fixture
def client(tmp_path, fake):
    def runner(ctx, cfg, on_progress):
        return pipeline.run_pipeline(ctx, cfg, on_progress=on_progress, check_resources=False)

    with TestClient(create_app(tmp_path / "runs", runner=runner)) as test_client:
        yield test_client


def _upload(client, name="clip.mp4", **form):
    return client.post(
        "/api/jobs",
        files={"file": (name, b"fake video bytes", "video/mp4")},
        data={"preset": "object", **form},
    )


def _wait(client, run_id, timeout=15):
    deadline = time.time() + timeout
    while time.time() < deadline:
        detail = client.get(f"/api/runs/{run_id}").json()
        if detail["job"] and detail["job"]["status"] in {"complete", "failed"}:
            return detail
        time.sleep(0.05)
    raise AssertionError("job did not finish")


def test_index_page_is_served(client):
    page = client.get("/")
    assert page.status_code == 200
    assert "<model-viewer" in page.text
    assert client.get("/app.js").status_code == 200


def test_presets_are_listed_without_the_base_file(client):
    assert set(client.get("/api/presets").json()) == {"object", "scene", "fast"}


def test_upload_runs_and_the_result_is_downloadable(client):
    response = _upload(client, aruco_marker_mm="100")
    assert response.status_code == 200
    run_id = response.json()["run_id"]

    detail = _wait(client, run_id)
    assert detail["status"] == "complete"
    assert [p["status"] for p in detail["phases"]] == ["complete"] * 5
    assert detail["print_report"]["scale_method"] == "aruco"  # the form option reached Phase 4

    stl = client.get(detail["files"]["model.stl"])
    assert stl.status_code == 200
    assert f"{run_id}_model.stl" in stl.headers["content-disposition"]
    assert stl.content[:5] != b"solid"  # binary STL
    assert client.get(detail["files"]["model.glb"]).status_code == 200
    assert client.get(detail["report_url"]).status_code == 200

    runs = client.get("/api/runs").json()
    assert runs[0]["run_id"] == run_id and runs[0]["source_name"] == "clip.mp4"


def test_progress_streams_as_server_sent_events(client):
    run_id = _upload(client).json()["run_id"]
    _wait(client, run_id)

    stream = client.get(f"/api/runs/{run_id}/events")
    assert stream.headers["content-type"].startswith("text/event-stream")
    events = [
        json.loads(line[len("data: ") :])
        for line in stream.text.splitlines()
        if line.startswith("data: ")
    ]
    phase_events = [(e["phase"], e["status"]) for e in events if e["type"] == "phase"]
    assert phase_events[:2] == [("ingest", "running"), ("ingest", "complete")]
    assert ("print_prep", "complete") in phase_events
    assert events[-1] == {**events[-1], "type": "status", "status": "complete"}

    # A reconnect with Last-Event-ID only replays what came after it.
    resumed = client.get(f"/api/runs/{run_id}/events", headers={"Last-Event-ID": "3"})
    ids = [int(line[4:]) for line in resumed.text.splitlines() if line.startswith("id: ")]
    assert ids and min(ids) == 4


def test_a_failure_shows_its_remedy_and_can_be_resumed(client, fake):
    fake.fail_at = PhaseName.SFM_DENSE
    run_id = _upload(client).json()["run_id"]

    detail = _wait(client, run_id)
    assert detail["status"] == "failed"
    dense = next(p for p in detail["phases"] if p["name"] == "sfm_dense")
    assert dense["remedy"] == "film something with more detail"
    assert detail["job"]["remedy"] == "film something with more detail"
    assert client.get(f"/runs/{run_id}/output/model.stl").status_code == 404

    assert client.post(f"/api/runs/{run_id}/resume").status_code == 200
    detail = _wait(client, run_id)
    assert detail["status"] == "complete"
    statuses = {p["name"]: p["status"] for p in detail["phases"]}
    assert statuses["ingest"] == "complete"


def test_uploads_queue_behind_the_running_job(client, fake):
    fake.gate = threading.Event()
    first = _upload(client).json()["run_id"]
    second = _upload(client).json()["run_id"]
    try:
        detail = client.get(f"/api/runs/{second}").json()
        assert detail["job"]["status"] == "queued"
        assert detail["job"]["jobs_ahead"] == 1
    finally:
        fake.gate.set()
    assert _wait(client, first)["status"] == "complete"
    assert _wait(client, second)["status"] == "complete"


def test_bad_uploads_are_rejected_with_a_reason(client):
    not_video = _upload(client, name="notes.txt")
    assert not_video.status_code == 400
    assert "doesn't look like a video" in not_video.json()["error"]
    assert not_video.json()["remedy"]

    bad_preset = client.post(
        "/api/jobs", files={"file": ("a.mp4", b"x", "video/mp4")}, data={"preset": "nope"}
    )
    assert bad_preset.status_code == 400
    assert client.get("/api/runs").json() == []  # nothing half-created on disk


@pytest.mark.parametrize(
    "path",
    [
        "/api/runs/does-not-exist",
        "/api/runs/..%2F..%2Fetc",
        "/runs/..%2F..%2Fetc/output/model.stl",
        "/runs/x/output/manifest.json",
    ],
)
def test_unknown_or_escaping_paths_are_404(client, path):
    assert client.get(path).status_code == 404


@pytest.mark.slow
def test_real_pipeline_through_the_web(tmp_path):
    with TestClient(create_app(tmp_path / "runs")) as real_client:
        response = real_client.post(
            "/api/jobs",
            files={"file": ("sample.mp4", VIDEO.read_bytes(), "video/mp4")},
            data={"preset": "fast"},
        )
        run_id = response.json()["run_id"]
        detail = _wait(real_client, run_id, timeout=600)
        assert detail["status"] == "complete", detail
        stl = real_client.get(detail["files"]["model.stl"]).content
        path = tmp_path / "downloaded.stl"
        path.write_bytes(stl)
        assert trimesh.load(str(path)).is_watertight
