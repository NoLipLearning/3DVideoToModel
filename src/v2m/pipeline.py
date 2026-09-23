"""End-to-end orchestration (M6): run the five phases over one RunContext.

`run_phase()` is the single place that does per-phase bookkeeping --
`start_phase` / `complete_phase` / `fail_phase` plus the phase's result
model saved into the manifest as its `summary` -- so the per-phase CLI
commands (`extract`, `sfm`, ...), `v2m run`, and the web UI's job worker
(M7) all leave identical manifests behind.

`run_pipeline()` chains them, skipping every phase the manifest already
records as COMPLETE. That is what `--resume` is: a phase left RUNNING by
a hard kill (SIGKILL, power loss -- nothing gets a chance to write
FAILED) is not COMPLETE, so it re-runs from scratch, and every phase
already clears its own stale outputs when it starts (extract.py's
`_clear_stale_frames`, sfm.py's database/scratch reset, ...), so no
partial artifact from the killed attempt leaks into the retry.

Config changes on resume go through `prepare_resume()`: a `--set
print_prep.slab_thickness_mm=5` only invalidates Phase 4, a `--set
sfm.matching_overlap=20` invalidates Phase 2a onward. This is the whole
point of docs/ARCHITECTURE.md's "runs/<id>/ with a manifest": a Phase 4
tweak must never force Phases 1-3 to re-run.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel

from v2m import preflight
from v2m.config import PipelineConfig, config_from_snapshot, load_config
from v2m.errors import ConfigError, IngestError, ResumeError, V2MError
from v2m.phase1_ingest.extract import run_extract
from v2m.phase2_sfm.dense import densify
from v2m.phase2_sfm.dense.monodepth_tsdf import DepthEstimator
from v2m.phase2_sfm.sfm import run_sparse_sfm
from v2m.phase3_mesh import build_mesh
from v2m.phase4_print import run_print_prep
from v2m.run_context import RunContext, hash_file
from v2m.types import PhaseName, PhaseStatus, PrintReport

logger = logging.getLogger("v2m.pipeline")

# The earliest phase that reads each top-level config section. Changing a
# value in that section invalidates this phase and everything after it.
# `dense` is also read by Phase 3 (Poisson depth), which is downstream of
# SFM_DENSE anyway. `mode` only changes Phase 4's base (cut vs. slab).
_SECTION_FIRST_PHASE: dict[str, PhaseName] = {
    "ingest": PhaseName.INGEST,
    "sfm": PhaseName.SFM_SPARSE,
    "dense": PhaseName.SFM_DENSE,
    "mesh": PhaseName.MESH,
    "print_prep": PhaseName.PRINT_PREP,
    "mode": PhaseName.PRINT_PREP,
}

PHASE_LABELS: dict[PhaseName, str] = {
    PhaseName.INGEST: "Phase 1 — ingest",
    PhaseName.SFM_SPARSE: "Phase 2a — sparse SfM",
    PhaseName.SFM_DENSE: "Phase 2b — dense",
    PhaseName.MESH: "Phase 3 — mesh",
    PhaseName.PRINT_PREP: "Phase 4 — print prep",
}

# (phase, status, message) -- message is the error text on FAILED, else None.
ProgressCallback = Callable[[PhaseName, PhaseStatus, str | None], None]


@dataclass
class PrintOptions:
    """Phase 4's per-invocation scale inputs. These are deliberately not
    part of `PipelineConfig`: they describe *this capture* (where the
    marker is, what the two picked points measure), not a reusable
    preset, and a numpy point pair doesn't belong in a YAML file."""

    scale_factor: float | None = None
    scale_two_point: tuple[np.ndarray, np.ndarray, float] | None = None
    aruco_image_name: str | None = None
    aruco_marker_mm: float | None = None

    def is_empty(self) -> bool:
        return self.to_json() == {}

    def to_json(self) -> dict[str, Any]:
        data: dict[str, Any] = {}
        if self.scale_factor is not None:
            data["scale_factor"] = self.scale_factor
        if self.scale_two_point is not None:
            a, b, distance = self.scale_two_point
            data["scale_two_point"] = [np.asarray(a).tolist(), np.asarray(b).tolist(), distance]
        if self.aruco_image_name is not None:
            data["aruco_image_name"] = self.aruco_image_name
        if self.aruco_marker_mm is not None:
            data["aruco_marker_mm"] = self.aruco_marker_mm
        return data

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> PrintOptions:
        two_point = data.get("scale_two_point")
        return cls(
            scale_factor=data.get("scale_factor"),
            scale_two_point=(
                (np.array(two_point[0]), np.array(two_point[1]), float(two_point[2]))
                if two_point
                else None
            ),
            aruco_image_name=data.get("aruco_image_name"),
            aruco_marker_mm=data.get("aruco_marker_mm"),
        )


@dataclass
class PipelineResult:
    run_dir: Path
    phases_run: list[PhaseName] = field(default_factory=list)
    phases_skipped: list[PhaseName] = field(default_factory=list)
    print_report: PrintReport | None = None
    report_path: Path | None = None


# -- config / resume ----------------------------------------------------------


def parse_overrides(assignments: list[str]) -> dict[str, Any]:
    """`["print_prep.slab_thickness_mm=5", "dense.backend=sparse_only"]` ->
    a dot-path override dict. Values are parsed as YAML scalars, so
    numbers, booleans and lists come through typed (`5` -> int,
    `true` -> bool, `[1,2,3]` -> list)."""
    import yaml

    overrides: dict[str, Any] = {}
    for assignment in assignments:
        key, sep, raw = assignment.partition("=")
        key = key.strip()
        if not sep or not key:
            raise ConfigError(
                f"Can't parse override '{assignment}'.",
                remedy="Use --set section.key=value, e.g. --set print_prep.slab_thickness_mm=5.",
            )
        overrides[key] = yaml.safe_load(raw)
    return overrides


def _earliest_affected_phase(changed_keys: list[str]) -> PhaseName | None:
    order = list(PhaseName)
    affected = []
    for key in changed_keys:
        section = key.split(".", 1)[0]
        if section in _SECTION_FIRST_PHASE:
            affected.append(_SECTION_FIRST_PHASE[section])
    return min(affected, key=order.index) if affected else None


def start_run(
    video: Path,
    preset: str,
    *,
    overrides: dict[str, Any] | None = None,
    run_dir: Path | None = None,
    print_options: PrintOptions | None = None,
) -> tuple[RunContext, PipelineConfig]:
    """Create a fresh run for `video`. `run_dir=None` invents
    `runs/<timestamp>_<id>/`."""
    if not video.exists():
        raise IngestError(
            f"Video not found: {video}", remedy="Check the path, or drag the file in again."
        )
    cfg = load_config(preset, overrides)
    snapshot = cfg.model_dump(mode="json")
    if run_dir is None:
        ctx = RunContext.create(preset=preset, config_snapshot=snapshot, source_video=str(video))
    else:
        if (run_dir / "manifest.json").exists():
            raise ResumeError(
                f"{run_dir} already holds a run.",
                remedy=f"Use `v2m run --resume {run_dir}` to continue it, or pick a new --run-dir.",
            )
        ctx = RunContext.at(
            run_dir, preset=preset, config_snapshot=snapshot, source_video=str(video)
        )
    if print_options is not None:
        ctx.manifest.print_options = print_options.to_json()
        ctx.save()
    return ctx, cfg


def prepare_resume(
    run_dir: Path,
    *,
    overrides: dict[str, Any] | None = None,
    rerun_from: PhaseName | None = None,
    video: Path | None = None,
    print_options: PrintOptions | None = None,
) -> tuple[RunContext, PipelineConfig]:
    """Attach to an existing run, apply config changes, and invalidate
    exactly the phases those changes affect.

    `print_options`, when given and different from the ones stored in
    the manifest, replace them and invalidate Phase 4 (a new scale
    reference means a new model.stl); when omitted, the stored ones are
    kept.

    The config comes from the manifest's own snapshot, not from
    `configs/*.yaml` -- editing a preset file after a run started never
    silently changes what its resume does (see `RunManifest`'s docstring).
    """
    ctx = RunContext.resume(run_dir)

    if video is not None:
        ingest = ctx.manifest.phases[PhaseName.INGEST]
        ingested_other_video = (
            ingest.status == PhaseStatus.COMPLETE
            and ingest.input_hash is not None
            and hash_file(video) != ingest.input_hash
        )
        if ingested_other_video and rerun_from != PhaseName.INGEST:
            raise ResumeError(
                f"{video} is not the video this run was built from ({ctx.manifest.source_video}).",
                remedy="Start a fresh run for a different video, or add "
                "--rerun-from ingest to rebuild this run from the new one.",
            )
        ctx.manifest.source_video = str(video)

    overrides = overrides or {}
    cfg = config_from_snapshot(ctx.manifest.config, overrides)
    new_snapshot = cfg.model_dump(mode="json")
    changed = [
        key for key in overrides if _lookup(ctx.manifest.config, key) != _lookup(new_snapshot, key)
    ]
    ctx.manifest.config = new_snapshot
    scale_changed = False
    if print_options is not None and not print_options.is_empty():
        scale_changed = print_options.to_json() != ctx.manifest.print_options
        ctx.manifest.print_options = print_options.to_json()
    ctx.save()

    order = list(PhaseName)
    candidates = [p for p in (rerun_from, _earliest_affected_phase(changed)) if p is not None]
    if scale_changed:
        candidates.append(PhaseName.PRINT_PREP)
        changed.append("scale reference")
    if candidates:
        first = min(candidates, key=order.index)
        reset = ctx.invalidate_from(first)
        if reset:
            logger.info(
                "Re-running from %s (%s).",
                PHASE_LABELS[first],
                "requested" if first == rerun_from else f"config changed: {', '.join(changed)}",
            )
    return ctx, cfg


def _lookup(data: dict[str, Any], dotted_key: str) -> Any:
    node: Any = data
    for part in dotted_key.split("."):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node


# -- phase execution -------------------------------------------------------------


def _execute(
    ctx: RunContext,
    cfg: PipelineConfig,
    phase: PhaseName,
    print_options: PrintOptions,
    depth_estimator: DepthEstimator | None,
) -> tuple[BaseModel, dict[str, str]]:
    if phase == PhaseName.INGEST:
        video = _source_video(ctx)
        summary = run_extract(video, ctx.run_dir, cfg.ingest)
        return summary, {"frames_json": summary.frames_json_path}
    if phase == PhaseName.SFM_SPARSE:
        result = run_sparse_sfm(ctx.frames_dir, ctx.sfm_dir, cfg.sfm)
        return result, {"sparse_points": result.sparse_points_path, "cameras": result.cameras_path}
    if phase == PhaseName.SFM_DENSE:
        result = densify(ctx.sfm_dir, ctx.dense_dir, cfg.dense, depth_estimator=depth_estimator)
        return result, {"dense_points": result.dense_points_path}
    if phase == PhaseName.MESH:
        result = build_mesh(ctx.dense_dir, ctx.sfm_dir, ctx.mesh_dir, cfg.dense, cfg.mesh)
        return result, {"mesh": result.mesh_path}
    if phase == PhaseName.PRINT_PREP:
        report = run_print_prep(
            ctx.mesh_dir,
            ctx.dense_dir,
            ctx.sfm_dir,
            ctx.output_dir,
            cfg.mesh,
            cfg.print_prep,
            cfg.mode,
            scale_factor=print_options.scale_factor,
            scale_two_point=print_options.scale_two_point,
            aruco_image_name=print_options.aruco_image_name,
            aruco_marker_mm=print_options.aruco_marker_mm,
        )
        return report, {
            "stl": "model.stl",
            "obj": "model.obj",
            "glb": "model.glb",
            "print_report": "print_report.json",
        }
    raise ValueError(f"unknown phase {phase}")


def _source_video(ctx: RunContext) -> Path:
    if ctx.manifest.source_video is None:
        raise IngestError(
            "This run has no source video recorded, so Phase 1 can't run.",
            remedy="Start a fresh run with a video, or populate frames/ with `v2m capture`.",
        )
    video = Path(ctx.manifest.source_video)
    if not video.exists():
        raise IngestError(
            f"The run's source video {video} no longer exists.",
            remedy="Restore the file, or pass the video again: `v2m run <video> --resume "
            f"{ctx.run_dir}`.",
        )
    return video


def run_phase(
    ctx: RunContext,
    cfg: PipelineConfig,
    phase: PhaseName,
    *,
    print_options: PrintOptions | None = None,
    depth_estimator: DepthEstimator | None = None,
    on_progress: ProgressCallback | None = None,
) -> BaseModel:
    """Run one phase with full manifest bookkeeping; returns the phase's
    result model. On failure the manifest records FAILED plus the error
    and the exception propagates unchanged."""
    print_options = print_options or PrintOptions()
    previous = ctx.manifest.phases[phase].status
    if previous == PhaseStatus.RUNNING:
        logger.warning(
            "%s was interrupted mid-run last time (the process was killed before it could "
            "record success or failure); re-running it from scratch.",
            PHASE_LABELS[phase],
        )

    ctx.start_phase(phase)
    if on_progress:
        on_progress(phase, PhaseStatus.RUNNING, None)
    try:
        if phase == PhaseName.INGEST:
            # The video's fingerprint, so a later --resume can tell whether
            # it is being pointed at the same clip.
            ctx.manifest.phases[phase].input_hash = hash_file(_source_video(ctx))
            ctx.save()
        result, artifacts = _execute(ctx, cfg, phase, print_options, depth_estimator)
    except V2MError as exc:
        ctx.fail_phase(phase, exc.message, remedy=exc.remedy)
        if on_progress:
            on_progress(phase, PhaseStatus.FAILED, exc.message)
        raise
    except BaseException as exc:
        # Includes KeyboardInterrupt: a Ctrl-C is recorded as a failure
        # with a clear message rather than left looking like a hard kill.
        message = (
            "interrupted by user"
            if isinstance(exc, KeyboardInterrupt)
            else f"unexpected {type(exc).__name__}: {exc}"
        )
        ctx.fail_phase(phase, message)
        if on_progress:
            on_progress(phase, PhaseStatus.FAILED, message)
        raise

    ctx.complete_phase(phase, artifacts=artifacts, summary=result.model_dump(mode="json"))
    if on_progress:
        on_progress(phase, PhaseStatus.COMPLETE, None)
    return result


def run_pipeline(
    ctx: RunContext,
    cfg: PipelineConfig,
    *,
    print_options: PrintOptions | None = None,
    depth_estimator: DepthEstimator | None = None,
    on_progress: ProgressCallback | None = None,
    check_resources: bool = True,
) -> PipelineResult:
    """Run every phase not already COMPLETE, in order, then write
    `report.html`. The report is written on failure too -- that is when
    it is most useful -- and a report-writing problem never masks the
    pipeline's own error.

    `print_options=None` uses the ones stored in the manifest (see
    `start_run` / `prepare_resume`)."""
    from v2m.report.html import write_report

    if print_options is None:
        print_options = PrintOptions.from_json(ctx.manifest.print_options)
    result = PipelineResult(run_dir=ctx.run_dir)
    pending = [p for p in PhaseName if not ctx.is_complete(p)]
    if check_resources and pending:
        preflight.check(cfg, ctx.run_dir, pending)

    try:
        for phase in PhaseName:
            if ctx.is_complete(phase):
                result.phases_skipped.append(phase)
                if on_progress:
                    on_progress(phase, PhaseStatus.SKIPPED, None)
                continue
            phase_result = run_phase(
                ctx,
                cfg,
                phase,
                print_options=print_options,
                depth_estimator=depth_estimator,
                on_progress=on_progress,
            )
            result.phases_run.append(phase)
            if isinstance(phase_result, PrintReport):
                result.print_report = phase_result
    finally:
        try:
            result.report_path = write_report(ctx)
        except Exception:
            logger.exception("Could not write report.html (the run itself is unaffected).")

    if result.print_report is None and ctx.is_complete(PhaseName.PRINT_PREP):
        report_json = ctx.output_dir / "print_report.json"
        if report_json.exists():
            result.print_report = PrintReport.model_validate_json(report_json.read_text())
    return result
