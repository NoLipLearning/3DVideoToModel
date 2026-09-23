"""Local web UI (M7): upload a video, watch it reconstruct, preview and
download the printable model.

    uv run v2m serve            # http://127.0.0.1:8000

No JS build step: `static/` is plain HTML/CSS/JS, and the 3D preview is
Google's `<model-viewer>` web component loaded from Google's CDN. The
server binds to 127.0.0.1 by default -- it is a local tool with no
authentication, and the upload endpoint writes to disk.

Routes:
  GET  /                              the app (static/index.html)
  GET  /api/presets                   preset names
  GET  /api/runs                      every run under runs_root, newest first
  GET  /api/runs/{id}                 one run: phases, results, live job state
  POST /api/jobs                      upload a video + options -> queued run
  POST /api/runs/{id}/resume          re-queue a failed/interrupted run
  GET  /api/runs/{id}/events          Server-Sent Events: live progress
  GET  /runs/{id}/report.html         the run's report
  GET  /runs/{id}/output/{file}       model.stl / .obj / .glb / print_report.json
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Annotated, Any

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from v2m import pipeline
from v2m.config import list_presets
from v2m.errors import V2MError
from v2m.run_context import RunContext
from v2m.types import PhaseName
from v2m.web.jobs import JobManager, Runner

STATIC_DIR = Path(__file__).parent / "static"
_OUTPUT_FILES = {
    "model.stl": "model/stl",
    "model.obj": "text/plain",
    "model.glb": "model/gltf-binary",
    "print_report.json": "application/json",
}
_SSE_POLL_S = 0.4


def create_app(runs_root: Path, *, runner: Runner | None = None) -> FastAPI:
    runs_root = runs_root.resolve()
    runs_root.mkdir(parents=True, exist_ok=True)
    manager = JobManager(runs_root, runner=runner)
    app = FastAPI(title="v2m", docs_url=None, redoc_url=None)
    app.state.jobs = manager

    def _run_dir_or_404(run_id: str) -> Path:
        run_dir = manager.run_dir(run_id)
        if run_dir is None:
            raise HTTPException(status_code=404, detail=f"No run '{run_id}'.")
        return run_dir

    @app.exception_handler(V2MError)
    async def _v2m_error(_request: Request, exc: V2MError) -> JSONResponse:
        return JSONResponse(status_code=400, content={"error": exc.message, "remedy": exc.remedy})

    @app.get("/api/presets")
    def presets() -> list[str]:
        return [p for p in list_presets() if p != "default"]

    @app.get("/api/runs")
    def runs() -> list[dict[str, Any]]:
        return manager.list_runs()

    @app.get("/api/runs/{run_id}")
    def run_detail(run_id: str) -> dict[str, Any]:
        return _describe(manager, run_id, _run_dir_or_404(run_id))

    @app.post("/api/jobs")
    def create_job(
        file: Annotated[UploadFile, File()],
        preset: Annotated[str, Form()] = "object",
        aruco_marker_mm: Annotated[float | None, Form()] = None,
        target_size_mm: Annotated[float | None, Form()] = None,
    ) -> dict[str, Any]:
        if preset not in list_presets():
            raise HTTPException(status_code=400, detail=f"Unknown preset '{preset}'.")
        overrides = {}
        if target_size_mm is not None:
            overrides["print_prep.scale_target_size_mm"] = target_size_mm
        job = manager.submit_upload(
            file.filename or "upload.mp4",
            file.file,
            preset,
            overrides=overrides,
            print_options=pipeline.PrintOptions(aruco_marker_mm=aruco_marker_mm),
        )
        return {"run_id": job.run_id, "status": job.status}

    @app.post("/api/runs/{run_id}/resume")
    def resume(run_id: str) -> dict[str, Any]:
        _run_dir_or_404(run_id)
        job = manager.resume(run_id)
        return {"run_id": job.run_id, "status": job.status}

    @app.get("/api/runs/{run_id}/events")
    async def events(run_id: str, request: Request) -> StreamingResponse:
        _run_dir_or_404(run_id)
        job = manager.get(run_id)
        if job is None:
            raise HTTPException(status_code=404, detail="No live job for this run.")
        last_id = request.headers.get("last-event-id")
        start = int(last_id) + 1 if last_id and last_id.isdigit() else 0

        async def stream():
            seq = start
            while True:
                if await request.is_disconnected():
                    return
                finished = job.finished  # read before draining, so no event is missed
                for event in job.events_since(seq):
                    seq = event["seq"] + 1
                    yield (
                        f"id: {event['seq']}\nevent: {event['type']}\ndata: {json.dumps(event)}\n\n"
                    )
                if finished:
                    return
                await asyncio.sleep(_SSE_POLL_S)

        return StreamingResponse(
            stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.get("/runs/{run_id}/report.html")
    def report(run_id: str) -> FileResponse:
        path = _run_dir_or_404(run_id) / "report.html"
        if not path.exists():
            raise HTTPException(status_code=404, detail="No report yet.")
        return FileResponse(path, media_type="text/html")

    @app.get("/runs/{run_id}/output/{name}")
    def output_file(run_id: str, name: str) -> FileResponse:
        if name not in _OUTPUT_FILES:
            raise HTTPException(status_code=404, detail=f"No such output '{name}'.")
        run_dir = _run_dir_or_404(run_id)
        if not RunContext.resume(run_dir).is_complete(PhaseName.PRINT_PREP):
            raise HTTPException(status_code=404, detail="This run has no finished model yet.")
        path = run_dir / "output" / name
        if not path.exists():
            raise HTTPException(status_code=404, detail=f"{name} is missing.")
        # STL/OBJ download under a name that says which run it came from;
        # the GLB is fetched inline by the preview.
        download = f"{run_id}_{name}" if name in {"model.stl", "model.obj"} else None
        return FileResponse(path, media_type=_OUTPUT_FILES[name], filename=download)

    app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
    return app


def _describe(manager: JobManager, run_id: str, run_dir: Path) -> dict[str, Any]:
    from v2m.report.html import overall_status

    ctx = RunContext.resume(run_dir)
    status_class, status_text = overall_status(ctx)
    job = manager.get(run_id)
    print_report = None
    if ctx.is_complete(PhaseName.PRINT_PREP) and (ctx.output_dir / "print_report.json").exists():
        print_report = json.loads((ctx.output_dir / "print_report.json").read_text())
    return {
        "run_id": run_id,
        "created_at": ctx.manifest.created_at.isoformat(),
        "preset": ctx.manifest.preset,
        "mode": ctx.manifest.config.get("mode"),
        "status": status_class,
        "status_text": status_text,
        "phases": [
            {
                "name": phase.value,
                "label": pipeline.PHASE_LABELS[phase],
                "status": ctx.manifest.phases[phase].status.value,
                "duration_s": ctx.manifest.phases[phase].duration_s,
                "error": ctx.manifest.phases[phase].error,
                "remedy": ctx.manifest.phases[phase].remedy,
            }
            for phase in PhaseName
        ],
        "print_report": print_report,
        "files": (
            {name: f"/runs/{run_id}/output/{name}" for name in _OUTPUT_FILES}
            if print_report
            else {}
        ),
        "report_url": f"/runs/{run_id}/report.html" if ctx.report_path.exists() else None,
        "job": (
            {
                "status": job.status,
                "error": job.error,
                "remedy": job.remedy,
                "jobs_ahead": manager.jobs_ahead(run_id),
            }
            if job
            else None
        ),
    }
