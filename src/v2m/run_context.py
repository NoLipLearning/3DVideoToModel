"""RunContext: the on-disk home for one pipeline execution.

Every phase is expensive (minutes to hours), so every run gets its own
directory under `runs/<timestamp>_<id>/` with a `manifest.json` that
records per-phase status. `--resume` reads that manifest and re-enters at
the first incomplete phase, so a parameter tweak in Phase 4 never forces
Phases 1-3 to re-run. See docs/ARCHITECTURE.md, "Why runs/<id>/ with a
manifest".
"""

from __future__ import annotations

import hashlib
import os
import secrets
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from v2m.errors import ResumeError
from v2m.types import PhaseName, PhaseRecord, PhaseStatus, RunManifest

RUNS_ROOT = Path("runs")


def new_run_id() -> str:
    stamp = datetime.now(UTC).strftime("%Y-%m-%d_%H%M%S")
    return f"{stamp}_{secrets.token_hex(2)}"


def hash_file(path: Path, chunk_size: int = 1 << 20) -> str:
    """Cheap content fingerprint used to detect stale inputs on --resume."""
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()[:16]


class RunContext:
    """Owns one run directory: layout, manifest persistence, and resume."""

    def __init__(self, run_dir: Path, manifest: RunManifest) -> None:
        self.run_dir = run_dir
        self.manifest = manifest

    # -- construction ---------------------------------------------------

    @classmethod
    def create(
        cls,
        *,
        preset: str,
        config_snapshot: dict[str, Any],
        source_video: str | None = None,
    ) -> RunContext:
        run_id = new_run_id()
        run_dir = RUNS_ROOT / run_id
        manifest = RunManifest(
            run_id=run_id,
            created_at=datetime.now(UTC),
            source_video=source_video,
            preset=preset,
            config=config_snapshot,
        )
        ctx = cls(run_dir, manifest)
        ctx._make_dirs()
        ctx.save()
        return ctx

    @classmethod
    def at(
        cls,
        run_dir: Path,
        *,
        preset: str,
        config_snapshot: dict[str, Any],
        source_video: str | None = None,
    ) -> RunContext:
        """Create a new run rooted at an explicit directory, or attach to
        an existing manifest already there.

        `create()` always invents a fresh `runs/<timestamp>_<id>/` path,
        which is what `v2m run` (M6) will use for full end-to-end
        orchestration. Before that exists, the per-phase CLI commands
        (`extract`, `sfm`, `dense`, ...) need to run and re-run against a
        directory the *caller* names, while still getting manifest
        bookkeeping for free -- that's what this gives them. Idempotent:
        calling it twice on the same directory attaches to the same
        manifest rather than resetting it.
        """
        manifest_path = run_dir / "manifest.json"
        if manifest_path.exists():
            return cls.resume(run_dir)

        manifest = RunManifest(
            run_id=run_dir.name,
            created_at=datetime.now(UTC),
            source_video=source_video,
            preset=preset,
            config=config_snapshot,
        )
        ctx = cls(run_dir, manifest)
        ctx._make_dirs()
        ctx.save()
        return ctx

    @classmethod
    def resume(cls, run_dir: Path) -> RunContext:
        manifest_path = run_dir / "manifest.json"
        if not manifest_path.exists():
            raise ResumeError(
                f"No manifest.json found in {run_dir}.",
                remedy="Start a fresh run instead of --resume, or point at the correct "
                "run directory.",
            )
        manifest = RunManifest.model_validate_json(manifest_path.read_text())
        return cls(run_dir, manifest)

    # -- layout -----------------------------------------------------------

    def _make_dirs(self) -> None:
        for sub in ("frames", "sfm", "dense", "mesh", "output"):
            (self.run_dir / sub).mkdir(parents=True, exist_ok=True)

    @property
    def frames_dir(self) -> Path:
        return self.run_dir / "frames"

    @property
    def sfm_dir(self) -> Path:
        return self.run_dir / "sfm"

    @property
    def dense_dir(self) -> Path:
        return self.run_dir / "dense"

    @property
    def mesh_dir(self) -> Path:
        return self.run_dir / "mesh"

    @property
    def output_dir(self) -> Path:
        return self.run_dir / "output"

    @property
    def log_path(self) -> Path:
        return self.run_dir / "run.log.jsonl"

    @property
    def report_path(self) -> Path:
        return self.run_dir / "report.html"

    @property
    def manifest_path(self) -> Path:
        return self.run_dir / "manifest.json"

    # -- manifest lifecycle -----------------------------------------------

    def save(self) -> None:
        """Atomic: write a temp file, then rename it over manifest.json.
        A plain `write_text` truncates first, so a reader (the web UI polls
        manifests while the worker writes them) could see an empty file,
        and a kill mid-write would leave a corrupt manifest that makes
        --resume impossible -- the one thing the manifest exists for."""
        tmp = self.manifest_path.with_name(f".manifest.{os.getpid()}.{threading.get_ident()}.tmp")
        tmp.write_text(self.manifest.model_dump_json(indent=2))
        os.replace(tmp, self.manifest_path)

    def start_phase(self, phase: PhaseName, *, input_hash: str | None = None) -> None:
        record = self.manifest.phases[phase]
        record.status = PhaseStatus.RUNNING
        record.input_hash = input_hash
        record.started_at = datetime.now(UTC)
        record.error = None
        record.remedy = None
        self.save()

    def complete_phase(
        self,
        phase: PhaseName,
        *,
        artifacts: dict[str, str] | None = None,
        summary: dict[str, Any] | None = None,
    ) -> None:
        record = self.manifest.phases[phase]
        record.status = PhaseStatus.COMPLETE
        record.finished_at = datetime.now(UTC)
        if record.started_at is not None:
            record.duration_s = (record.finished_at - record.started_at).total_seconds()
        if artifacts:
            record.artifacts.update(artifacts)
        if summary is not None:
            record.summary = summary
        self.save()

    def fail_phase(self, phase: PhaseName, error: str, *, remedy: str | None = None) -> None:
        record = self.manifest.phases[phase]
        record.status = PhaseStatus.FAILED
        record.finished_at = datetime.now(UTC)
        record.error = error
        record.remedy = remedy
        self.save()

    def invalidate_from(self, phase: PhaseName) -> list[PhaseName]:
        """Reset `phase` and every phase after it to PENDING, so the next
        run re-executes them. Returns the phases that were actually
        reset (the ones that weren't already PENDING).

        Used when an input a phase depends on changes -- a config value
        on `--resume --set`, or an explicit `--rerun-from`. Artifacts on
        disk are left alone: each phase already clears its own stale
        outputs when it starts.
        """
        order = list(PhaseName)
        reset = []
        for later in order[order.index(phase) :]:
            record = self.manifest.phases[later]
            if record.status != PhaseStatus.PENDING:
                reset.append(later)
            self.manifest.phases[later] = PhaseRecord()
        self.save()
        return reset

    def is_complete(self, phase: PhaseName) -> bool:
        return self.manifest.phases[phase].status == PhaseStatus.COMPLETE

    def next_incomplete_phase(self) -> PhaseName | None:
        for phase in PhaseName:
            if not self.is_complete(phase):
                return phase
        return None
