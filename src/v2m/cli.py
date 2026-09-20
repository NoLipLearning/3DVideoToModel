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

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from v2m import capability
from v2m import run_context as rc
from v2m.config import list_presets, load_config
from v2m.errors import V2MError
from v2m.logging_setup import setup_logging
from v2m.phase1_ingest.extract import run_extract
from v2m.types import PhaseName

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


@app.command()
def extract(
    video: str,
    output: str = typer.Option(..., "--output", "-o"),
    preset: str = typer.Option("object", "--preset"),
) -> None:
    """Phase 1: extract a filtered frame set from a video."""
    video_path = Path(video)
    output_dir = Path(output)

    if not video_path.exists():
        console.print(f"[red]Video not found:[/red] {video_path}")
        raise typer.Exit(code=1)

    cfg = load_config(preset)
    ctx = rc.RunContext.at(
        output_dir,
        preset=preset,
        config_snapshot=cfg.model_dump(),
        source_video=str(video_path),
    )
    if ctx.manifest.preset != preset:
        console.print(
            f"[yellow]Note:[/yellow] existing run at {output_dir} was created with preset "
            f"'{ctx.manifest.preset}'; ignoring --preset {preset} for this attach. Delete "
            "the directory (or use a new one) to start fresh with a different preset."
        )
    setup_logging(run_log_path=ctx.log_path)

    ctx.start_phase(PhaseName.INGEST, input_hash=rc.hash_file(video_path))
    try:
        summary = run_extract(video_path, ctx.run_dir, cfg.ingest)
    except V2MError as exc:
        ctx.fail_phase(PhaseName.INGEST, exc.message)
        console.print(f"[red]Ingest failed:[/red] {exc.message}")
        if exc.remedy:
            console.print(f"  [yellow]→[/yellow] {exc.remedy}")
        raise typer.Exit(code=1) from exc

    ctx.complete_phase(PhaseName.INGEST, artifacts={"frames_json": summary.frames_json_path})

    table = Table(title="Ingest summary")
    table.add_column("Metric")
    table.add_column("Count", justify="right")
    table.add_row("Total frames decoded", str(summary.total_frames))
    table.add_row("Accepted", f"[green]{summary.accepted}[/green]")
    table.add_row("Rejected: blur", str(summary.rejected_blur))
    table.add_row("Rejected: redundant", str(summary.rejected_redundant))
    table.add_row("Rejected: frame budget", str(summary.rejected_budget))
    table.add_row("Blur threshold used", f"{summary.blur_threshold:.1f}")
    console.print(table)
    console.print(f"Run directory: [bold]{ctx.run_dir}[/bold]")


@app.command(name="sfm")
def sfm_cmd(frames_dir: str) -> None:
    """Phase 2a: sparse structure-from-motion. (M2)"""
    _not_implemented("sfm", "M2")


@app.command()
def dense(run_dir: str) -> None:
    """Phase 2b: dense point cloud. (M3)"""
    _not_implemented("dense", "M3")


@app.command()
def mesh(run_dir: str) -> None:
    """Phase 3: raw surface mesh. (M4)"""
    _not_implemented("mesh", "M4")


@app.command()
def printprep(run_dir: str) -> None:
    """Phase 4: watertight, scaled, print-ready export. (M5)"""
    _not_implemented("printprep", "M5")


@app.command(name="run")
def run_cmd(
    video: str,
    preset: str = typer.Option("object", "--preset"),
    resume: str | None = typer.Option(None, "--resume"),
) -> None:
    """End-to-end pipeline: video in, model.stl out. (M6)"""
    _not_implemented("run", "M6")


@app.command()
def serve(port: int = typer.Option(8000, "--port")) -> None:
    """Local web UI for upload + progress + preview. (M7)"""
    _not_implemented("serve", "M7")


@app.command()
def capture(output: str = typer.Option(..., "--output", "-o")) -> None:
    """Guided live capture session. (M8)"""
    _not_implemented("capture", "M8")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
