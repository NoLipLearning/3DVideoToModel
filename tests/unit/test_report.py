"""report/html.py and report/preview.py against hand-built run directories."""

import io
import json

import numpy as np
import trimesh
from PIL import Image

from v2m.config import load_config
from v2m.report import html as report_html
from v2m.report.preview import render_preview
from v2m.run_context import RunContext
from v2m.types import PhaseName, PrintReport


def _ctx(tmp_path):
    cfg = load_config("object")
    return RunContext.at(
        tmp_path / "run", preset="object", config_snapshot=cfg.model_dump(mode="json")
    )


def test_an_empty_run_still_renders(tmp_path):
    ctx = _ctx(tmp_path)
    path = report_html.write_report(ctx)
    text = path.read_text()
    assert "Incomplete" in text
    assert "<script" not in text  # self-contained and script-free


def test_errors_are_escaped_and_the_resume_command_is_shown(tmp_path):
    ctx = _ctx(tmp_path)
    ctx.start_phase(PhaseName.INGEST)
    ctx.fail_phase(PhaseName.INGEST, "<script>alert(1)</script>", remedy="re-shoot & retry")
    text = report_html.render(ctx)
    assert "<script>alert" not in text
    assert "&lt;script&gt;" in text
    assert "re-shoot &amp; retry" in text
    assert "v2m run --resume" in text
    assert report_html.overall_status(ctx)[0] == "failed"


def test_capture_advice_follows_the_diagnostics(tmp_path):
    ctx = _ctx(tmp_path)
    (ctx.sfm_dir / "diagnostics.json").write_text(
        json.dumps(
            {
                "registration_rate": 0.4,
                "median_triangulation_angle_deg": 0.5,
                "low_keypoint_images": [{"name": "000001.jpg", "keypoint_count": 12}],
                "warnings": [],
            }
        )
    )
    report = PrintReport(
        watertight=True,
        volume_mm3=1.0,
        bbox_mm=(1.0, 1.0, 1.0),
        repair_rung_used=6,
        scale_method="fit_to_build_volume",
        warnings=["No trustworthy ground plane was found (fallback_object)"],
    )
    (ctx.output_dir / "print_report.json").write_text(report.model_dump_json())
    for phase in PhaseName:
        ctx.start_phase(phase)
        ctx.complete_phase(phase)

    advice = " ".join(report_html.capture_advice(ctx))
    assert "overlap" in advice  # low registration
    assert "Walk around" in advice  # pure rotation
    assert "texture" in advice  # low-keypoint frames
    assert "floor" in advice  # ground fallback
    assert "ArUco" in advice  # no metric scale
    assert "voxel remeshing" in advice  # rung 6


def test_stale_outputs_from_an_unfinished_phase_are_not_reported(tmp_path):
    ctx = _ctx(tmp_path)
    trimesh.creation.box(extents=[10, 10, 10]).export(ctx.output_dir / "model.stl")
    (ctx.output_dir / "print_report.json").write_text("{}")
    # Phase 4 never completed in this run, so those files are leftovers.
    text = report_html.render(ctx)
    assert "model.stl</a>" not in text


def test_preview_renders_a_shaded_image_of_the_mesh():
    png = render_preview(trimesh.creation.box(extents=[10, 20, 30]), size=200)
    image = np.asarray(Image.open(io.BytesIO(png)))
    assert image.shape == (200, 200, 4)
    drawn = image[:, :, 3] > 0
    assert 0.1 < drawn.mean() < 0.9  # something drawn, not a full flood
    # Lambert shading: the three visible faces get different brightness.
    assert len(np.unique(image[drawn][:, :3], axis=0)) >= 3
