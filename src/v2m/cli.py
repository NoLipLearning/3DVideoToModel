"""v2m command-line interface.

Milestone map (see docs/ARCHITECTURE.md Section 4 for detail):
  M0 - doctor, presets                (this milestone)
  M1 - extract                        (Phase 1: ingest)
  M2 - sfm                            (Phase 2a: sparse)
  M3 - dense                          (Phase 2b: dense)
  M4 - mesh                           (Phase 3)
  M5 - printprep                      (Phase 4)
  M6 - run                            (end-to-end + --resume)
  M7 - serve                          (local web UI)
  M8 - capture                        (guided live capture)
"""

import json
from pathlib import Path

import numpy as np
import typer
from rich.console import Console
from rich.table import Table

from v2m import capability, pipeline
from v2m import run_context as rc
from v2m.config import list_presets, load_config
from v2m.errors import V2MError
from v2m.logging_setup import setup_logging
from v2m.types import PhaseName, PhaseStatus

app = typer.Typer(add_completion=False, no_args_is_help=True)
console = Console()

# Import name -> friendly package name, for display only.
_FRIENDLY_NAMES = {
    "cv2": "opencv-contrib-python",
    "yaml": "pyyaml",
    "PIL": "pillow",
}

_LIB_GROUPS = {
    "core": [
        "numpy",
        "cv2",
        "scipy",
        "trimesh",
        "manifold3d",
        "open3d",
        "pydantic",
        "typer",
        "yaml",
        "rich",
        "psutil",
    ],
    "sfm": ["pycolmap"],
    "depth": ["torch", "torchvision", "transformers", "PIL", "safetensors"],
    "mesh (optional)": ["pymeshlab"],
    "web": ["fastapi", "uvicorn"],
    "dev/optional": ["av"],
}


def _not_implemented(command: str, milestone: str) -> None:
    console.print(
        f"[yellow]`v2m {command}` is not implemented yet.[/yellow] "
        f"Coming in milestone {milestone} -- see docs/ARCHITECTURE.md Section 4."
    )
    raise typer.Exit(code=1)


@app.command()
def doctor() -> None:
    """Probe hardware, binaries, and libraries; report what the pipeline can actually run."""
    caps = capability.probe()

    info = Table(title="Platform", show_header=False)
    info.add_row("OS", caps.platform_system)
    info.add_row("Architecture", caps.platform_machine)
    info.add_row("Python", caps.python_version)
    info.add_row("CUDA available", "yes" if caps.has_cuda else "no")
    if caps.has_cuda:
        info.add_row("CUDA device", caps.cuda_device_name or "unknown")
    info.add_row("MPS (Apple GPU) available", "yes" if caps.has_mps else "no")
    console.print(info)

    for group_name, lib_names in _LIB_GROUPS.items():
        table = Table(title=f"Libraries: {group_name}")
        table.add_column("Package")
        table.add_column("Status")
        table.add_column("Version")
        for name in lib_names:
            status = caps.libs.get(name)
            display_name = _FRIENDLY_NAMES.get(name, name)
            if status and status.available:
                table.add_row(display_name, "[green]present[/green]", status.version or "?")
            else:
                table.add_row(display_name, "[red]absent[/red]", "-")
        console.print(table)

    bins = Table(title="External binaries")
    bins.add_column("Binary")
    bins.add_column("Path")
    bins.add_row("colmap", caps.colmap_binary or "[red]not found[/red]")
    bins.add_row(
        "OpenMVS (DensifyPointCloud)",
        caps.openmvs_binary or "[yellow]not found (optional)[/yellow]",
    )
    bins.add_row(
        "ffmpeg",
        caps.ffmpeg_binary or "[yellow]not found (optional; OpenCV bundles its own)[/yellow]",
    )
    console.print(bins)

    hevc_ok = capability.hevc_decode_check(caps)
    if hevc_ok is None:
        hevc_display = "not tested yet (no probe clip shipped before M1)"
    elif hevc_ok:
        hevc_display = "[green]ok[/green]"
    else:
        hevc_display = "[red]failed[/red]"
    console.print(f"HEVC decode check: {hevc_display}")

    console.print()
    if caps.has_cuda:
        console.print(
            "[yellow]CUDA detected.[/yellow] This is NOT the target platform (Apple Silicon, no "
            "CUDA) -- COLMAP dense MVS becomes technically available here, but the pipeline still "
            f"defaults to '{caps.dense_backend}' unless config explicitly opts into an "
            "OpenMVS/CUDA path."
        )
    else:
        console.print(
            "COLMAP dense MVS (patch_match_stereo) is [bold]unavailable[/bold] -- it requires "
            "CUDA, which does not exist on macOS or this machine. This is expected and by "
            f"design: the pipeline uses dense backend '[bold]{caps.dense_backend}[/bold]' instead."
        )

    missing = caps.missing_core()
    if missing:
        friendly_missing = [_FRIENDLY_NAMES.get(name, name) for name in missing]
        console.print(f"\n[red]Missing core dependencies:[/red] {', '.join(friendly_missing)}")
        console.print("Run `uv sync` from the repo root, then re-run `v2m doctor`.")
        raise typer.Exit(code=1)

    console.print(
        "\n[green]Core dependencies OK.[/green] Ready for milestone-by-milestone development."
    )


@app.command()
def presets() -> None:
    """List available config presets."""
    for name in list_presets():
        console.print(f"- {name}")


def _attach(run_dir: Path, preset: str, *, source_video: str | None = None):
    """Attach a per-phase command to `run_dir` (creating the run if it's
    new), set up logging, and note a preset mismatch."""
    cfg = load_config(preset)
    ctx = rc.RunContext.at(
        run_dir,
        preset=preset,
        config_snapshot=cfg.model_dump(mode="json"),
        source_video=source_video,
    )
    if source_video is not None and ctx.manifest.source_video != source_video:
        ctx.manifest.source_video = source_video
        ctx.save()
    if ctx.manifest.preset != preset:
        console.print(
            f"[yellow]Note:[/yellow] existing run at {run_dir} was created with preset "
            f"'{ctx.manifest.preset}'; ignoring --preset {preset} for this attach. Delete "
            "the directory (or use a new one) to start fresh with a different preset."
        )
    setup_logging(run_log_path=ctx.log_path)
    return ctx, cfg


def _run_single(ctx, cfg, phase: PhaseName, **kwargs):
    """One phase with manifest bookkeeping; prints the error + remedy and
    exits non-zero on a V2MError."""
    try:
        return pipeline.run_phase(ctx, cfg, phase, **kwargs)
    except V2MError as exc:
        _print_error(f"{pipeline.PHASE_LABELS[phase]} failed", exc)
        if phase == PhaseName.SFM_SPARSE:
            console.print(f"See {ctx.sfm_dir / 'diagnostics.json'} for details.")
        raise typer.Exit(code=1) from exc


def _print_error(title: str, exc: V2MError) -> None:
    console.print(f"[red]{title}:[/red] {exc.message}")
    if exc.remedy:
        console.print(f"  [yellow]→[/yellow] {exc.remedy}")


@app.command()
def extract(
    video: str,
    output: str = typer.Option(..., "--output", "-o"),
    preset: str = typer.Option("object", "--preset"),
) -> None:
    """Phase 1: extract a filtered frame set from a video."""
    video_path = Path(video)
    if not video_path.exists():
        console.print(f"[red]Video not found:[/red] {video_path}")
        raise typer.Exit(code=1)

    ctx, cfg = _attach(Path(output), preset, source_video=str(video_path))
    summary = _run_single(ctx, cfg, PhaseName.INGEST)
    _print_phase_summary(PhaseName.INGEST, summary)
    console.print(f"Run directory: [bold]{ctx.run_dir}[/bold]")


@app.command(name="sfm")
def sfm_cmd(
    run_dir: str,
    preset: str = typer.Option("object", "--preset"),
) -> None:
    """Phase 2a: sparse structure-from-motion. Expects frames/ already populated by `extract`."""
    ctx, cfg = _attach(Path(run_dir), preset)
    result = _run_single(ctx, cfg, PhaseName.SFM_SPARSE)
    _print_phase_summary(PhaseName.SFM_SPARSE, result)

    diagnostics_path = ctx.sfm_dir / "diagnostics.json"
    if diagnostics_path.exists():
        diagnostics = json.loads(diagnostics_path.read_text())
        for warning in diagnostics.get("warnings", []):
            console.print(f"[yellow]Warning:[/yellow] {warning}")
    console.print(f"Run directory: [bold]{ctx.run_dir}[/bold]")


@app.command()
def dense(
    run_dir: str,
    preset: str = typer.Option("object", "--preset"),
) -> None:
    """Phase 2b: dense point cloud. Expects sfm/ already populated by `sfm`."""
    ctx, cfg = _attach(Path(run_dir), preset)
    result = _run_single(ctx, cfg, PhaseName.SFM_DENSE)
    _print_phase_summary(PhaseName.SFM_DENSE, result)
    console.print(f"Run directory: [bold]{ctx.run_dir}[/bold]")


@app.command()
def mesh(
    run_dir: str,
    preset: str = typer.Option("object", "--preset"),
) -> None:
    """Phase 3: raw surface mesh. Expects dense/ already populated by `dense`."""
    ctx, cfg = _attach(Path(run_dir), preset)
    result = _run_single(ctx, cfg, PhaseName.MESH)
    _print_phase_summary(PhaseName.MESH, result)
    console.print(f"Run directory: [bold]{ctx.run_dir}[/bold]")


def _print_options(
    scale_factor: float | None,
    scale_points: str | None,
    scale_distance_mm: float | None,
    aruco_image: str | None,
    aruco_marker_mm: float | None,
) -> pipeline.PrintOptions:
    two_point = None
    if scale_points is not None or scale_distance_mm is not None:
        if scale_points is None or scale_distance_mm is None:
            console.print("[red]--scale-points and --scale-distance-mm go together.[/red]")
            raise typer.Exit(code=1)
        try:
            point_a, point_b = (
                np.array([float(v) for v in part.split(",")]) for part in scale_points.split(";")
            )
        except ValueError as exc:
            console.print(f'[red]--scale-points must look like "x,y,z;x,y,z":[/red] {exc}')
            raise typer.Exit(code=1) from exc
        two_point = (point_a, point_b, scale_distance_mm)
    if aruco_image is not None and aruco_marker_mm is None:
        console.print("[red]--aruco-image needs --aruco-marker-mm (the printed size).[/red]")
        raise typer.Exit(code=1)
    return pipeline.PrintOptions(
        scale_factor=scale_factor,
        scale_two_point=two_point,
        aruco_image_name=aruco_image,
        aruco_marker_mm=aruco_marker_mm,
    )


_SCALE_FACTOR_OPT = typer.Option(
    None, "--scale-factor", help="mm per reconstruction unit, if already known."
)
_SCALE_POINTS_OPT = typer.Option(
    None,
    "--scale-points",
    help='Two points in mesh/cleaned.ply coords, "x,y,z;x,y,z" (with --scale-distance-mm).',
)
_SCALE_DISTANCE_OPT = typer.Option(
    None, "--scale-distance-mm", help="Real distance between the two --scale-points."
)
_ARUCO_IMAGE_OPT = typer.Option(
    None,
    "--aruco-image",
    help="Frame (e.g. 000012.jpg) to read the marker from. Default: the frame where it "
    "appears largest.",
)
_ARUCO_MM_OPT = typer.Option(
    None,
    "--aruco-marker-mm",
    help="Printed edge length of an ArUco marker (DICT_4X4_50) placed in the scene.",
)
_TARGET_SIZE_OPT = typer.Option(
    None, "--target-size", help="Longest-axis size in mm when no metric reference is given."
)


@app.command()
def printprep(
    run_dir: str,
    preset: str = typer.Option("object", "--preset"),
    scale_factor: float | None = _SCALE_FACTOR_OPT,
    scale_points: str | None = _SCALE_POINTS_OPT,
    scale_distance_mm: float | None = _SCALE_DISTANCE_OPT,
    aruco_image: str | None = _ARUCO_IMAGE_OPT,
    aruco_marker_mm: float | None = _ARUCO_MM_OPT,
    target_size_mm: float | None = _TARGET_SIZE_OPT,
) -> None:
    """Phase 4: watertight, scaled, print-ready export. Expects mesh/ populated by `mesh`."""
    options = _print_options(
        scale_factor, scale_points, scale_distance_mm, aruco_image, aruco_marker_mm
    )
    ctx, cfg = _attach(Path(run_dir), preset)
    if target_size_mm is not None:
        cfg.print_prep.scale_target_size_mm = target_size_mm
    report = _run_single(ctx, cfg, PhaseName.PRINT_PREP, print_options=options)
    _print_print_report(report)
    console.print(f"Run directory: [bold]{ctx.run_dir}[/bold]")


def _print_phase_summary(phase: PhaseName, result) -> None:
    table = Table(title=f"{pipeline.PHASE_LABELS[phase]} summary")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    if phase == PhaseName.INGEST:
        table.add_row("Total frames decoded", str(result.total_frames))
        table.add_row("Accepted", f"[green]{result.accepted}[/green]")
        table.add_row("Rejected: blur", str(result.rejected_blur))
        table.add_row("Rejected: redundant", str(result.rejected_redundant))
        table.add_row("Rejected: frame budget", str(result.rejected_budget))
        table.add_row("Blur threshold used", f"{result.blur_threshold:.1f}")
    elif phase == PhaseName.SFM_SPARSE:
        rate = (
            result.num_images_registered / result.num_images_total
            if result.num_images_total
            else 0.0
        )
        table.add_row(
            "Registered images",
            f"[green]{result.num_images_registered}/{result.num_images_total}[/green] ({rate:.0%})",
        )
        table.add_row("Mean reprojection error", f"{result.mean_reprojection_error_px:.3f} px")
        table.add_row("Mean track length", f"{result.mean_track_length:.2f}")
    elif phase == PhaseName.SFM_DENSE:
        table.add_row("Backend", result.backend)
        table.add_row("Dense points", f"[green]{result.num_points:,}[/green]")
        if result.alignment_rmse is not None:
            table.add_row("Mean alignment RMSE (1/units)", f"{result.alignment_rmse:.6g}")
    elif phase == PhaseName.MESH:
        table.add_row("Vertices", f"{result.num_vertices:,}")
        table.add_row("Faces", f"{result.num_faces:,}")
        table.add_row(
            "Edge/vertex manifold",
            "[green]yes[/green]" if result.is_manifold else "[yellow]no[/yellow]",
        )
    console.print(table)


def _print_print_report(report) -> None:
    table = Table(title="Print-ready model")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    table.add_row("Watertight", "[green]yes[/green]" if report.watertight else "[red]no[/red]")
    table.add_row("Size (mm)", " x ".join(f"{v:.1f}" for v in report.bbox_mm))
    table.add_row("Volume (cm³)", f"{report.volume_mm3 / 1000:.1f}")
    table.add_row("Repair rung used", str(report.repair_rung_used))
    table.add_row("Scale method", report.scale_method)
    console.print(table)
    for warning in report.warnings:
        console.print(f"[yellow]Warning:[/yellow] {warning}")


def _progress_printer(ctx):
    def on_progress(phase: PhaseName, status: PhaseStatus, message: str | None) -> None:
        label = pipeline.PHASE_LABELS[phase]
        if status == PhaseStatus.RUNNING:
            console.print(f"[bold]▶ {label}[/bold]")
        elif status == PhaseStatus.COMPLETE:
            duration = ctx.manifest.phases[phase].duration_s or 0.0
            console.print(f"[green]✓ {label}[/green] ({duration:.1f}s)")
        elif status == PhaseStatus.SKIPPED:
            console.print(f"[dim]↷ {label} — already complete, skipped[/dim]")

    return on_progress


_RERUN_FROM_OPT = typer.Option(
    None, "--rerun-from", help="With --resume: redo this phase and everything after it."
)
_SET_OPT = typer.Option(
    [],
    "--set",
    help="Config override, e.g. --set print_prep.slab_thickness_mm=5 (repeatable). On "
    "--resume, only the phases that read the changed section re-run.",
)


@app.command(name="run")
def run_cmd(
    video: str | None = typer.Argument(
        None, help="Input video. Optional with --resume (pass it if the file moved)."
    ),
    preset: str | None = typer.Option(
        None, "--preset", help="object | scene | fast (default: object). Fixed once a run starts."
    ),
    resume: str | None = typer.Option(
        None, "--resume", help="Continue an existing run directory from its first unfinished phase."
    ),
    run_dir: str | None = typer.Option(
        None, "--run-dir", help="Where to create a new run (default: runs/<timestamp>_<id>/)."
    ),
    from_frames: str | None = typer.Option(
        None,
        "--from-frames",
        help="Start from a folder of frames instead of a video (e.g. from `v2m capture`).",
    ),
    rerun_from: PhaseName | None = _RERUN_FROM_OPT,
    overrides: list[str] = _SET_OPT,
    scale_factor: float | None = _SCALE_FACTOR_OPT,
    scale_points: str | None = _SCALE_POINTS_OPT,
    scale_distance_mm: float | None = _SCALE_DISTANCE_OPT,
    aruco_image: str | None = _ARUCO_IMAGE_OPT,
    aruco_marker_mm: float | None = _ARUCO_MM_OPT,
    target_size_mm: float | None = _TARGET_SIZE_OPT,
    skip_preflight: bool = typer.Option(
        False, "--skip-preflight", help="Don't refuse a run the RAM/disk estimate says won't fit."
    ),
) -> None:
    """End-to-end pipeline: video in, output/model.stl out. Resumable."""
    options = _print_options(
        scale_factor, scale_points, scale_distance_mm, aruco_image, aruco_marker_mm
    )
    try:
        override_dict = pipeline.parse_overrides(overrides)
        if target_size_mm is not None:
            override_dict["print_prep.scale_target_size_mm"] = target_size_mm

        if resume is not None:
            if run_dir is not None:
                console.print("[red]--resume and --run-dir are mutually exclusive.[/red]")
                raise typer.Exit(code=1)
            ctx, cfg = pipeline.prepare_resume(
                Path(resume),
                overrides=override_dict,
                rerun_from=rerun_from,
                video=Path(video) if video else None,
                print_options=options,
            )
            if preset is not None and preset != ctx.manifest.preset:
                console.print(
                    f"[yellow]Note:[/yellow] this run uses preset '{ctx.manifest.preset}'; "
                    f"--preset {preset} is ignored on --resume. Use --set to change settings."
                )
        else:
            if (video is None) == (from_frames is None):
                console.print(
                    "[red]Give a video or --from-frames <folder> to start a run, or --resume "
                    "<run_dir>.[/red]"
                )
                raise typer.Exit(code=1)
            if rerun_from is not None:
                console.print("[red]--rerun-from only applies with --resume.[/red]")
                raise typer.Exit(code=1)
            ctx, cfg = pipeline.start_run(
                Path(video) if video else None,
                preset or "object",
                frames_dir=Path(from_frames) if from_frames else None,
                overrides=override_dict,
                run_dir=Path(run_dir) if run_dir else None,
                print_options=options,
            )
    except V2MError as exc:
        _print_error("Can't start", exc)
        raise typer.Exit(code=1) from exc
    _execute_run(ctx, cfg, skip_preflight=skip_preflight)


def _execute_run(ctx, cfg, *, skip_preflight: bool = False) -> None:
    """Run the pipeline with live progress lines, then print the result."""
    setup_logging(run_log_path=ctx.log_path)
    console.print(f"Run directory: [bold]{ctx.run_dir}[/bold]  (preset {ctx.manifest.preset})")
    try:
        result = pipeline.run_pipeline(
            ctx,
            cfg,
            on_progress=_progress_printer(ctx),
            check_resources=not skip_preflight,
        )
    except V2MError as exc:
        _print_error("Run failed", exc)
        console.print(f"Report: [bold]{ctx.report_path}[/bold]")
        console.print(f"Fix the cause, then: [bold]v2m run --resume {ctx.run_dir}[/bold]")
        raise typer.Exit(code=1) from exc
    except KeyboardInterrupt:
        console.print(
            f"\n[yellow]Interrupted.[/yellow] Continue with: v2m run --resume {ctx.run_dir}"
        )
        raise typer.Exit(code=130) from None

    if result.print_report is not None:
        _print_print_report(result.print_report)
    console.print(f"Model: [bold]{ctx.output_dir / 'model.stl'}[/bold]")
    if result.report_path is not None:
        console.print(f"Report: [bold]{result.report_path}[/bold]")


@app.command()
def serve(
    port: int = typer.Option(8000, "--port"),
    host: str = typer.Option(
        "127.0.0.1",
        "--host",
        help="Interface to bind. The UI has no authentication -- only expose it beyond "
        "localhost on a network you trust.",
    ),
    runs_dir: str = typer.Option(str(rc.RUNS_ROOT), "--runs-dir", help="Where runs are stored."),
) -> None:
    """Local web UI: upload a video, watch progress, preview and download the model."""
    import uvicorn

    from v2m.web.app import create_app

    setup_logging()
    console.print(f"v2m web UI on [bold]http://{host}:{port}[/bold]  (runs in {runs_dir})")
    uvicorn.run(create_app(Path(runs_dir)), host=host, port=port, log_level="warning")


@app.command()
def capture(
    output: str = typer.Option(..., "--output", "-o", help="Folder to write accepted frames to."),
    camera: int = typer.Option(0, "--camera", help="Camera index (0 = default camera)."),
    source: str | None = typer.Option(
        None, "--source", help="A video file or stream URL to use instead of a camera."
    ),
    preset: str = typer.Option("object", "--preset", help="Sets the frame budget and resolution."),
    no_preview: bool = typer.Option(
        False, "--no-preview", help="No HUD window (headless machine, or replaying a file)."
    ),
    max_seconds: float | None = typer.Option(None, "--max-seconds", help="Stop after this long."),
    then_run: bool = typer.Option(False, "--run", help="Reconstruct as soon as capture ends."),
) -> None:
    """Guided live capture: a HUD that keeps only usable frames. Space pauses, q finishes."""
    from v2m.capture.live import run_capture

    setup_logging()
    try:
        cfg = load_config(preset)
        summary = run_capture(
            source if source is not None else camera,
            Path(output),
            cfg.ingest,
            show=not no_preview,
            max_seconds=max_seconds,
        )
    except V2MError as exc:
        _print_error("Capture failed", exc)
        raise typer.Exit(code=1) from exc

    console.print(
        f"Kept [green]{summary['accepted']}[/green] of {summary['frames_seen']} frames "
        f"(budget {summary['budget']}) in [bold]{output}[/bold]"
    )
    if summary["accepted"] < 20:
        console.print(
            "[yellow]That's few views for a reconstruction -- walk all the way around the "
            "subject, slowly, and capture again if the result has holes.[/yellow]"
        )
    if not then_run:
        console.print(f"Next: [bold]v2m run --from-frames {output} --preset {preset}[/bold]")
        return
    try:
        ctx, run_cfg = pipeline.start_run(None, preset, frames_dir=Path(output))
    except V2MError as exc:
        _print_error("Can't start", exc)
        raise typer.Exit(code=1) from exc
    _execute_run(ctx, run_cfg)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
