"""Single-worker job queue for the web UI (M7).

One pipeline at a time, deliberately: every heavy phase already uses all
cores (COLMAP matching, TSDF integration, Poisson), and the preflight RAM
budget (preflight.py) assumes one run per machine. A second upload waits
in the queue and the UI shows its position.

A job is a thin live wrapper around a run directory: everything durable
(phase status, errors, results) is in that run's manifest.json, written
by `pipeline.run_phase()` exactly as for `v2m run`. What the job adds is
an in-memory event list -- phase transitions plus the pipeline's own log
lines -- that the SSE endpoint streams to the browser. After a server
restart the events are gone but the runs are not: `list_runs()` rebuilds
the history from disk, and a failed or interrupted run can be resumed
from the UI (`resume()`), which is `v2m run --resume` under the hood.
"""

from __future__ import annotations

import logging
import re
import shutil
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Any

from v2m import pipeline
from v2m.config import PipelineConfig
from v2m.errors import V2MError
from v2m.logging_setup import setup_logging
from v2m.run_context import RunContext, new_run_id
from v2m.types import PhaseName, PhaseStatus

logger = logging.getLogger("v2m.web.jobs")

VIDEO_SUFFIXES = {".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm", ".hevc"}
_RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9_\-]+$")
_MAX_LOG_EVENTS = 2000

# (ctx, cfg, on_progress) -> PipelineResult. Injectable so tests can swap
# the real five phases for instant fakes.
Runner = Callable[[RunContext, PipelineConfig, pipeline.ProgressCallback], Any]


def _default_runner(ctx, cfg, on_progress):
    return pipeline.run_pipeline(ctx, cfg, on_progress=on_progress)


@dataclass
class Job:
    run_id: str
    run_dir: Path
    status: str = "queued"  # queued | running | complete | failed
    error: str | None = None
    remedy: str | None = None
    events: list[dict[str, Any]] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    @property
    def finished(self) -> bool:
        return self.status in {"complete", "failed"}

    def add_event(self, kind: str, **data: Any) -> None:
        with self._lock:
            seq = self.events[-1]["seq"] + 1 if self.events else 0
            self.events.append({"seq": seq, "type": kind, "time": time.time(), **data})
            if len(self.events) > _MAX_LOG_EVENTS:
                # Keep phase/status events; drop the oldest log lines.
                for index, event in enumerate(self.events):
                    if event["type"] == "log":
                        del self.events[index]
                        break

    def events_since(self, seq: int) -> list[dict[str, Any]]:
        with self._lock:
            return [e for e in self.events if e["seq"] >= seq]


class _JobLogHandler(logging.Handler):
    """Mirrors the pipeline's `v2m.*` log records into the job's event
    stream, so the browser shows the same lines the terminal would."""

    def __init__(self, job: Job) -> None:
        super().__init__(level=logging.INFO)
        self.job = job

    def emit(self, record: logging.LogRecord) -> None:
        self.job.add_event("log", level=record.levelname, message=record.getMessage())


class JobManager:
    def __init__(self, runs_root: Path, runner: Runner | None = None) -> None:
        self.runs_root = runs_root
        self._runner = runner or _default_runner
        self._jobs: dict[str, Job] = {}
        self._pending: list[str] = []
        self._cond = threading.Condition()
        self._worker: threading.Thread | None = None

    # -- lookup ------------------------------------------------------------------

    def run_dir(self, run_id: str) -> Path | None:
        """The run's directory, or None for an id that is malformed or
        would escape runs_root (it arrives straight from a URL)."""
        if not _RUN_ID_PATTERN.match(run_id):
            return None
        path = (self.runs_root / run_id).resolve()
        if path.parent != self.runs_root.resolve() or not (path / "manifest.json").exists():
            return None
        return path

    def get(self, run_id: str) -> Job | None:
        return self._jobs.get(run_id)

    def jobs_ahead(self, run_id: str) -> int | None:
        """How many jobs will run before this one starts (0 once it is
        running), or None if it isn't queued."""
        with self._cond:
            return self._pending.index(run_id) if run_id in self._pending else None

    def list_runs(self) -> list[dict[str, Any]]:
        from v2m.report.html import overall_status

        runs = []
        if not self.runs_root.exists():
            return runs
        for manifest_path in self.runs_root.glob("*/manifest.json"):
            try:
                ctx = RunContext.resume(manifest_path.parent)
            except Exception:
                continue  # a half-written or foreign directory -- not ours to show
            status_class, status_text = overall_status(ctx)
            job = self._jobs.get(ctx.manifest.run_id)
            runs.append(
                {
                    "run_id": ctx.manifest.run_id,
                    "created_at": ctx.manifest.created_at.isoformat(),
                    "preset": ctx.manifest.preset,
                    "source_name": _source_name(ctx),
                    "status": job.status if job and not job.finished else status_class,
                    "status_text": status_text,
                }
            )
        runs.sort(key=lambda r: r["created_at"], reverse=True)
        return runs

    # -- submission --------------------------------------------------------------------

    def submit_upload(
        self,
        filename: str,
        stream: IO[bytes],
        preset: str,
        *,
        overrides: dict[str, Any] | None = None,
        print_options: pipeline.PrintOptions | None = None,
    ) -> Job:
        suffix = Path(filename).suffix.lower()
        if suffix not in VIDEO_SUFFIXES:
            raise V2MError(
                f"'{filename}' doesn't look like a video.",
                remedy=f"Upload one of: {', '.join(sorted(VIDEO_SUFFIXES))}.",
            )
        run_id = new_run_id()
        run_dir = self.runs_root / run_id
        video = run_dir / "input" / f"source{suffix}"
        video.parent.mkdir(parents=True)
        with video.open("wb") as out:
            shutil.copyfileobj(stream, out, length=1 << 20)
        try:
            pipeline.start_run(
                video,
                preset,
                run_dir=run_dir,
                overrides=overrides,
                print_options=print_options,
            )
        except Exception:
            shutil.rmtree(run_dir, ignore_errors=True)
            raise
        (run_dir / "input" / "original_name.txt").write_text(filename)
        return self._enqueue(run_id, run_dir)

    def resume(self, run_id: str) -> Job:
        run_dir = self.run_dir(run_id)
        if run_dir is None:
            raise V2MError(f"No run '{run_id}'.", remedy="Pick a run from the list.")
        existing = self._jobs.get(run_id)
        if existing is not None and not existing.finished:
            return existing
        return self._enqueue(run_id, run_dir)

    def _enqueue(self, run_id: str, run_dir: Path) -> Job:
        job = Job(run_id=run_id, run_dir=run_dir)
        job.add_event("status", status="queued")
        with self._cond:
            self._jobs[run_id] = job
            self._pending.append(run_id)
            self._cond.notify()
            if self._worker is None:
                self._worker = threading.Thread(target=self._work, name="v2m-worker", daemon=True)
                self._worker.start()
        return job

    # -- worker ------------------------------------------------------------------------

    def _work(self) -> None:
        while True:
            with self._cond:
                while not self._pending:
                    self._cond.wait()
                run_id = self._pending[0]
            job = self._jobs[run_id]
            try:
                self._run(job)
            finally:
                with self._cond:
                    self._pending.remove(run_id)

    def _run(self, job: Job) -> None:
        job.status = "running"
        job.add_event("status", status="running")
        handler = _JobLogHandler(job)
        try:
            ctx, cfg = pipeline.prepare_resume(job.run_dir)
            setup_logging(run_log_path=ctx.log_path)
            logging.getLogger("v2m").addHandler(handler)

            def on_progress(phase: PhaseName, status: PhaseStatus, message: str | None) -> None:
                record = ctx.manifest.phases[phase]
                job.add_event(
                    "phase",
                    phase=phase.value,
                    status=status.value,
                    duration_s=record.duration_s if status == PhaseStatus.COMPLETE else None,
                    message=message,
                )

            self._runner(ctx, cfg, on_progress)
        except V2MError as exc:
            job.status, job.error, job.remedy = "failed", exc.message, exc.remedy
        except Exception as exc:  # the server must outlive any one run
            logger.exception("Run %s crashed.", job.run_id)
            job.status, job.error = "failed", f"unexpected {type(exc).__name__}: {exc}"
        else:
            job.status = "complete"
        finally:
            logging.getLogger("v2m").removeHandler(handler)
        job.add_event("status", status=job.status, error=job.error, remedy=job.remedy)


def _source_name(ctx: RunContext) -> str | None:
    original = ctx.run_dir / "input" / "original_name.txt"
    if original.exists():
        return original.read_text().strip()
    return Path(ctx.manifest.source_video).name if ctx.manifest.source_video else None
