"""Per-run HTML report: `<run_dir>/report.html`.

One self-contained file (thumbnails and the model preview are embedded as
base64, CSS is inline, no scripts): it opens from disk, can be emailed,
and still works after the run directory moves. Written at the end of
every `run_pipeline()`, on failure as well as success -- a failed run is
when a report is most needed.

docs/ARCHITECTURE.md Section 3.1: "Never fail silently. If a region
reconstructs poorly, report.html shows it and states the capture fix."
So besides numbers, `capture_advice()` turns each diagnostic signal the
pipeline already records into the concrete re-shoot instruction that
fixes it.

Everything here reads files the phases already write (manifest.json,
frames/frames.json, sfm/diagnostics.json, output/print_report.json,
run.log.jsonl) -- it never re-derives a phase's result, and every read
tolerates the file being absent, because a failed run stops partway.
"""

from __future__ import annotations

import base64
import html
import json
import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from v2m.pipeline import PHASE_LABELS
from v2m.run_context import RunContext
from v2m.types import PhaseName, PhaseStatus

logger = logging.getLogger("v2m.report")

_THUMBNAIL_COUNT = 12
_THUMBNAIL_WIDTH = 160
_LOG_TAIL = 25

_LABELS = PHASE_LABELS


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def _print_report(ctx: RunContext) -> dict[str, Any] | None:
    if not ctx.is_complete(PhaseName.PRINT_PREP):
        return None
    return _read_json(ctx.output_dir / "print_report.json")


def _e(value: Any) -> str:
    return html.escape(str(value))


# -- data gathering ------------------------------------------------------------------


def overall_status(ctx: RunContext) -> tuple[str, str]:
    """(css class, human text) for the run as a whole."""
    phases = ctx.manifest.phases
    for phase in PhaseName:
        if phases[phase].status == PhaseStatus.FAILED:
            return "failed", f"Failed at {_LABELS[phase]}"
    if all(phases[p].status == PhaseStatus.COMPLETE for p in PhaseName):
        return "complete", "Complete"
    for phase in PhaseName:
        if phases[phase].status == PhaseStatus.RUNNING:
            return "running", f"Running (or interrupted) at {_LABELS[phase]}"
    return "pending", "Incomplete"


def capture_advice(ctx: RunContext) -> list[str]:
    """The re-shoot / re-run instructions this run's diagnostics call for."""
    advice: list[str] = []
    config = ctx.manifest.config
    ingest = ctx.manifest.phases[PhaseName.INGEST].summary
    diagnostics = _read_json(ctx.sfm_dir / "diagnostics.json") or {}
    print_report = _print_report(ctx) or {}

    total = ingest.get("total_frames") or 0
    if total and ingest.get("rejected_blur", 0) / total > 0.3:
        advice.append(
            f"{ingest['rejected_blur']} of {total} frames were too blurry to use. Move the "
            "camera more slowly, add light, or lock focus/exposure."
        )

    rate = diagnostics.get("registration_rate")
    rate_min = config.get("sfm", {}).get("registration_rate_min", 0.7)
    if rate is not None and rate < rate_min:
        advice.append(
            f"Only {rate:.0%} of frames could be placed in 3D. Move more slowly so consecutive "
            "views overlap by about 70%, and avoid sudden turns."
        )
    low_texture = diagnostics.get("low_keypoint_images") or []
    if low_texture:
        advice.append(
            f"{len(low_texture)} frames had too little texture to track (blank walls, sky, "
            "glass, glossy surfaces). Keep textured areas in frame, or add texture — tape, "
            "newspaper, or a patterned mat under the object."
        )
    angle = diagnostics.get("median_triangulation_angle_deg")
    angle_min = config.get("sfm", {}).get("min_triangulation_angle_deg", 2.0)
    if angle is not None and angle < angle_min:
        advice.append(
            "The camera mostly rotated in place, so depth couldn't be triangulated. Walk around "
            "the subject instead of pivoting on the spot."
        )

    warnings = print_report.get("warnings", [])
    if any("ground plane" in w for w in warnings):
        advice.append(
            "No floor was found to put the model's base on, so the base was estimated. Keep "
            "some of the table or floor around the object in view."
        )
    if print_report.get("scale_method") == "fit_to_build_volume":
        advice.append(
            "The model's size is not real-world: it was scaled to fit the printer. For true "
            "dimensions, lay a printed 100 mm ArUco marker (DICT_4X4_50) in the scene and pass "
            "--aruco-image / --aruco-marker-mm, or give --scale-factor or --scale-points."
        )
    if print_report.get("repair_rung_used") == 6:
        advice.append(
            "The surface had gaps the repair ladder could only close by voxel remeshing, which "
            "loses fine detail. Capture more angles — especially from above and low down."
        )
    return advice


def _thumbnails(ctx: RunContext) -> list[tuple[str, str]]:
    """(data URI, caption) for up to _THUMBNAIL_COUNT accepted frames,
    evenly spread over the capture."""
    frames = _read_json(ctx.frames_dir / "frames.json") or []
    accepted = [f for f in frames if f.get("accepted")]
    if not accepted:
        return []
    picks = np.linspace(0, len(accepted) - 1, min(_THUMBNAIL_COUNT, len(accepted))).astype(int)
    thumbs = []
    for index in sorted(set(picks.tolist())):
        record = accepted[index]
        image = cv2.imread(str(ctx.frames_dir / record["path"]))
        if image is None:
            continue
        height = int(image.shape[0] * _THUMBNAIL_WIDTH / image.shape[1])
        small = cv2.resize(image, (_THUMBNAIL_WIDTH, height), interpolation=cv2.INTER_AREA)
        ok, encoded = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ok:
            data = base64.b64encode(encoded.tobytes()).decode("ascii")
            thumbs.append(
                (
                    f"data:image/jpeg;base64,{data}",
                    f"#{record['frame_index']} · sharpness {record['laplacian_var']:.0f}",
                )
            )
    return thumbs


def _model_preview(ctx: RunContext) -> str | None:
    """A data URI of the final model (or, if Phase 4 didn't finish, the
    raw Phase 3 mesh) -- None if neither exists or rendering fails."""
    import trimesh

    from v2m.report.preview import render_preview

    # Only a COMPLETE phase's output counts: after a failed re-run, an
    # older model.stl may still be sitting in output/.
    candidates = [
        (PhaseName.PRINT_PREP, ctx.output_dir / "model.stl"),
        (PhaseName.MESH, ctx.mesh_dir / "cleaned.ply"),
    ]
    for phase, path in candidates:
        if not ctx.is_complete(phase) or not path.exists():
            continue
        try:
            mesh = trimesh.load(str(path), force="mesh")
            if len(mesh.faces) == 0:
                continue
            png = render_preview(mesh)
        except Exception:
            logger.exception("Could not render a preview of %s.", path)
            return None
        return "data:image/png;base64," + base64.b64encode(png).decode("ascii")
    return None


def _log_tail(ctx: RunContext) -> list[dict[str, str]]:
    if not ctx.log_path.exists():
        return []
    entries = []
    for line in ctx.log_path.read_text().splitlines():
        try:
            entry = json.loads(line)
        except ValueError:
            continue
        if entry.get("level") in {"WARNING", "ERROR", "CRITICAL"}:
            entries.append(entry)
    return entries[-_LOG_TAIL:]


def _phase_numbers(phase: PhaseName, summary: dict[str, Any]) -> str:
    """One short line of the numbers that matter for each phase."""
    if not summary:
        return ""
    if phase == PhaseName.INGEST:
        return (
            f"{summary['accepted']} of {summary['total_frames']} frames kept "
            f"(blurry {summary['rejected_blur']}, redundant {summary['rejected_redundant']}, "
            f"over budget {summary['rejected_budget']})"
        )
    if phase == PhaseName.SFM_SPARSE:
        total = summary["num_images_total"] or 1
        return (
            f"{summary['num_images_registered']}/{summary['num_images_total']} registered "
            f"({summary['num_images_registered'] / total:.0%}), reprojection "
            f"{summary['mean_reprojection_error_px']:.2f}px, track length "
            f"{summary['mean_track_length']:.1f}"
        )
    if phase == PhaseName.SFM_DENSE:
        rmse = summary.get("alignment_rmse")
        extra = f", alignment RMSE {rmse:.3g}" if rmse is not None else ""
        return f"{summary['backend']}: {summary['num_points']:,} points{extra}"
    if phase == PhaseName.MESH:
        return (
            f"{summary['num_vertices']:,} vertices, {summary['num_faces']:,} faces"
            f"{'' if summary['is_manifold'] else ' (not yet manifold — Phase 4 repairs it)'}"
        )
    if phase == PhaseName.PRINT_PREP:
        size = " × ".join(f"{v:.1f}" for v in summary["bbox_mm"])
        return (
            f"{size} mm, repair rung {summary['repair_rung_used']}, scale {summary['scale_method']}"
        )
    return ""


# -- rendering -------------------------------------------------------------------------

_CSS = """
:root { --bg:#f7f7f5; --fg:#1d1d1f; --muted:#6b6b70; --card:#fff; --line:#e3e3e0;
  --ok:#1f7a3f; --bad:#b3261e; --warn:#8a5a00; --run:#1d5fb4; }
@media (prefers-color-scheme: dark) { :root { --bg:#161618; --fg:#ececef; --muted:#9a9aa2;
  --card:#1f1f22; --line:#2e2e33; --ok:#5ccf85; --bad:#ff8a80; --warn:#f0b64a; --run:#7fb2ff; } }
* { box-sizing:border-box; }
body { margin:0; background:var(--bg); color:var(--fg);
  font:15px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif; }
main { max-width:960px; margin:0 auto; padding:24px 16px 64px; }
h1 { font-size:22px; margin:0 0 4px; } h2 { font-size:17px; margin:32px 0 10px; }
.meta { color:var(--muted); font-size:13px; }
.badge { display:inline-block; padding:2px 10px; border-radius:999px; font-weight:600;
  font-size:13px; border:1px solid currentColor; }
.complete { color:var(--ok); } .failed { color:var(--bad); } .running { color:var(--run); }
.pending,.skipped { color:var(--muted); }
.card { background:var(--card); border:1px solid var(--line); border-radius:10px;
  padding:14px 16px; margin:12px 0; }
.error { border-color:var(--bad); } .error strong { color:var(--bad); }
.result { display:flex; gap:20px; flex-wrap:wrap; align-items:center; }
.result img { width:320px; max-width:100%; }
.result dl { display:grid; grid-template-columns:auto auto; gap:4px 16px; margin:0; }
.result dt { color:var(--muted); }
table { width:100%; border-collapse:collapse; font-size:14px; }
th,td { text-align:left; padding:6px 8px; border-bottom:1px solid var(--line);
  vertical-align:top; }
th { color:var(--muted); font-weight:500; }
ul { margin:6px 0; padding-left:20px; } li { margin:4px 0; }
.warn li { color:var(--warn); }
.thumbs { display:flex; flex-wrap:wrap; gap:8px; }
.thumbs figure { margin:0; font-size:11px; color:var(--muted); }
.thumbs img { display:block; width:160px; border-radius:4px; }
code, .log { font-family:ui-monospace,SFMono-Regular,Menlo,monospace; font-size:12px; }
.log { white-space:pre-wrap; overflow-wrap:anywhere; }
a { color:var(--run); }
"""


def render(ctx: RunContext) -> str:
    manifest = ctx.manifest
    status_class, status_text = overall_status(ctx)
    config = manifest.config
    parts: list[str] = []
    add = parts.append

    add("<!doctype html><html lang='en'><head><meta charset='utf-8'>")
    add("<meta name='viewport' content='width=device-width,initial-scale=1'>")
    add(f"<title>v2m run {_e(manifest.run_id)}</title><style>{_CSS}</style></head><body><main>")
    add(f"<h1>v2m run <code>{_e(manifest.run_id)}</code></h1>")
    add(
        f"<div class='meta'>preset <b>{_e(manifest.preset)}</b> · mode "
        f"<b>{_e(config.get('mode', '?'))}</b> · source "
        f"<code>{_e(manifest.source_video or '(frames)')}</code> · created "
        f"{_e(manifest.created_at.strftime('%Y-%m-%d %H:%M UTC'))}</div>"
    )
    add(f"<p><span class='badge {status_class}'>{_e(status_text)}</span></p>")

    for phase in PhaseName:
        record = manifest.phases[phase]
        if record.status == PhaseStatus.FAILED:
            add(f"<div class='card error'><strong>{_e(_LABELS[phase])} failed:</strong> ")
            add(f"{_e(record.error)}")
            if record.remedy:
                add(f"<br>→ {_e(record.remedy)}")
            add(
                f"<div class='meta'>Fix the cause, then continue with <code>v2m run --resume "
                f"{_e(ctx.run_dir)}</code> — completed phases are not re-run.</div></div>"
            )

    print_report = _print_report(ctx)
    preview = _model_preview(ctx)
    if print_report or preview:
        add("<h2>Model</h2><div class='card result'>")
        if preview:
            add(f"<img src='{preview}' alt='Shaded preview of the model'>")
        if print_report:
            size = " × ".join(f"{v:.1f}" for v in print_report["bbox_mm"])
            add("<dl>")
            add(f"<dt>Size</dt><dd>{_e(size)} mm</dd>")
            add(f"<dt>Volume</dt><dd>{print_report['volume_mm3'] / 1000:.1f} cm³</dd>")
            watertight = "yes" if print_report["watertight"] else "no"
            add(f"<dt>Watertight</dt><dd>{watertight}</dd>")
            add(f"<dt>Repair rung</dt><dd>{print_report['repair_rung_used']} of 6</dd>")
            add(f"<dt>Scale</dt><dd>{_e(print_report['scale_method'])}</dd>")
            add(
                "<dt>Files</dt><dd><a href='output/model.stl'>model.stl</a> · "
                "<a href='output/model.obj'>model.obj</a> · "
                "<a href='output/model.glb'>model.glb</a></dd>"
            )
            add("</dl>")
        else:
            add("<p class='meta'>Raw Phase 3 surface — not yet print-ready.</p>")
        add("</div>")

    advice = capture_advice(ctx)
    if advice:
        add("<h2>How to get a better result</h2><div class='card'><ul>")
        parts.extend(f"<li>{_e(a)}</li>" for a in advice)
        add("</ul></div>")

    add("<h2>Phases</h2><div class='card'><table>")
    add("<tr><th>Phase</th><th>Status</th><th>Time</th><th>Result</th></tr>")
    for phase in PhaseName:
        record = manifest.phases[phase]
        duration = f"{record.duration_s:.1f}s" if record.duration_s is not None else ""
        try:
            numbers = _phase_numbers(phase, record.summary)
        except (KeyError, TypeError, ZeroDivisionError):
            numbers = ""
        add(
            f"<tr><td>{_e(_LABELS[phase])}</td><td class='{record.status.value}'>"
            f"{_e(record.status.value)}</td><td>{duration}</td><td>{_e(numbers)}</td></tr>"
        )
    add("</table></div>")

    warnings: list[str] = []
    diagnostics = _read_json(ctx.sfm_dir / "diagnostics.json") or {}
    warnings += [f"SfM: {w}" for w in diagnostics.get("warnings", [])]
    if print_report:
        warnings += [f"Print prep: {w}" for w in print_report.get("warnings", [])]
    if warnings:
        add("<h2>Warnings</h2><div class='card warn'><ul>")
        parts.extend(f"<li>{_e(w)}</li>" for w in warnings)
        add("</ul></div>")

    low_texture = diagnostics.get("low_keypoint_images") or []
    if low_texture:
        add("<h2>Low-texture frames</h2><div class='card'><p class='meta'>Fewer than ")
        add(
            f"{_e(config.get('sfm', {}).get('min_keypoints_per_image', '?'))} keypoints each "
            "— these are the views where the surface gave the reconstructor nothing to hold "
            "on to.</p><p>"
        )
        add(", ".join(f"<code>{_e(i['name'])}</code> ({i['keypoint_count']})" for i in low_texture))
        add("</p></div>")

    thumbs = _thumbnails(ctx)
    if thumbs:
        add("<h2>Frames used</h2><div class='card thumbs'>")
        for uri, caption in thumbs:
            add(f"<figure><img src='{uri}' alt=''><figcaption>{_e(caption)}</figcaption></figure>")
        add("</div>")

    log_tail = _log_tail(ctx)
    if log_tail:
        add("<h2>Log (warnings and errors)</h2><div class='card log'>")
        for entry in log_tail:
            add(
                f"{_e(entry.get('ts', '')[:19])} {_e(entry.get('level'))} "
                f"{_e(entry.get('message'))}\n"
            )
        add("</div>")

    add(
        "<p class='meta'>Full detail: <code>manifest.json</code>, <code>run.log.jsonl</code>, "
        "<code>sfm/diagnostics.json</code>, <code>output/print_report.json</code>.</p>"
    )
    add("</main></body></html>")
    return "".join(parts)


def write_report(ctx: RunContext) -> Path:
    ctx.report_path.write_text(render(ctx), encoding="utf-8")
    return ctx.report_path
