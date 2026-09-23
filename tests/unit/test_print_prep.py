"""Phase 4 end-to-end: run_print_prep() on the synthetic cube.

docs/ARCHITECTURE.md Section 4 (M5) verify: "Synthetic cube round-trips to
within 2% of ground-truth dimensions. print_report.json records which rung
was used and every scale decision."

The fixture has no marker in it, so the metric scale comes from a
reference the fixture *does* have: every camera's ground-truth position
in millimetres (tests/fixtures/sample_ground_truth.json) against the same
camera's reconstructed position -- the median ratio of pairwise camera
distances, passed in as a manual scale factor. That exercises the same
code path a user's `--scale-factor` takes and leaves the pipeline's own
geometry as the only thing being measured.

Dimensions are compared rotation-invariantly: the model's heading about
Z is arbitrary, so axis-aligned extents alone would conflate "wrong
size" with "turned 45 degrees".
"""

import itertools
import json
from pathlib import Path

import numpy as np
import pytest
import trimesh

from v2m.config import MeshConfig, PrintPrepConfig
from v2m.errors import PrintPrepError
from v2m.phase4_print import ground, run_print_prep

pytestmark = pytest.mark.slow

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"
CUBE_SIDE_MM = 200.0


def _ground_truth_scale(sfm_fixture) -> float:
    ground_truth = json.loads((FIXTURES_DIR / "sample_ground_truth.json").read_text())
    frames = json.loads((sfm_fixture["run_dir"] / "frames" / "frames.json").read_text())
    frame_index = {f["path"]: f["frame_index"] for f in frames if f.get("accepted")}
    true_center = {
        p["frame_index"]: np.array(p["camera_center_world_mm"]) for p in ground_truth["poses"]
    }
    cameras = json.loads((sfm_fixture["run_dir"] / "sfm" / "cameras.json").read_text())["images"]
    reconstructed = {
        name: ground.camera_center_from_pose(c["rotation_quat_xyzw"], c["translation"])
        for name, c in cameras.items()
    }
    ratios = [
        np.linalg.norm(true_center[frame_index[a]] - true_center[frame_index[b]])
        / np.linalg.norm(reconstructed[a] - reconstructed[b])
        for a, b in itertools.combinations(sorted(reconstructed), 2)
    ]
    return float(np.median(ratios))


@pytest.fixture(scope="module")
def metric_run(sfm_fixture, dense_fixture, mesh_fixture, tmp_path_factory) -> dict:
    output_dir = tmp_path_factory.mktemp("print_prep_metric")
    report = run_print_prep(
        mesh_fixture["output_dir"],
        dense_fixture["output_dir"],
        sfm_fixture["run_dir"] / "sfm",
        output_dir,
        MeshConfig(),
        PrintPrepConfig(),
        "object",
        scale_factor=_ground_truth_scale(sfm_fixture),
    )
    return {"report": report, "output_dir": output_dir}


def test_output_is_a_valid_printable_solid(metric_run):
    model = trimesh.load(str(metric_run["output_dir"] / "model.stl"))
    assert model.is_watertight
    assert model.is_winding_consistent
    assert model.volume > 0
    assert metric_run["report"].watertight


def test_cube_volume_round_trips_within_two_percent(metric_run):
    model = trimesh.load(str(metric_run["output_dir"] / "model.stl"))
    assert model.volume ** (1 / 3) == pytest.approx(CUBE_SIDE_MM, rel=0.02)


def test_cube_sides_round_trip_within_tolerance(metric_run):
    # An oriented bounding box is an upper envelope -- any surface bump
    # pushes it outward -- so it sits slightly above the volume-based
    # measure (measured: 201-204mm); 3% keeps it a real check without
    # failing on that one-sided bias.
    model = trimesh.load(str(metric_run["output_dir"] / "model.stl"))
    sides = np.sort(model.bounding_box_oriented.primitive.extents)
    np.testing.assert_allclose(sides, [CUBE_SIDE_MM] * 3, rtol=0.03)


def test_model_sits_on_the_bed(metric_run):
    model = trimesh.load(str(metric_run["output_dir"] / "model.stl"))
    assert model.bounds[0][2] == pytest.approx(0.0, abs=1e-6)


def test_print_report_records_rung_and_scale_decision(metric_run):
    written = json.loads((metric_run["output_dir"] / "print_report.json").read_text())
    assert written["scale_method"] == "manual"
    assert 1 <= written["repair_rung_used"] <= 6
    assert written["watertight"] is True
    # This fixture has no floor in view, so the base placement is a
    # gravity-prior fallback -- and the report must say so.
    assert any("ground plane" in w for w in written["warnings"])


def test_all_export_formats_are_written(metric_run):
    for name in ("model.stl", "model.obj", "model.glb", "print_report.json"):
        assert (metric_run["output_dir"] / name).stat().st_size > 0


def test_without_a_scale_reference_it_fits_the_target_size_and_says_so(
    sfm_fixture, dense_fixture, mesh_fixture, tmp_path
):
    config = PrintPrepConfig()
    report = run_print_prep(
        mesh_fixture["output_dir"],
        dense_fixture["output_dir"],
        sfm_fixture["run_dir"] / "sfm",
        tmp_path,
        MeshConfig(),
        config,
        "object",
    )
    assert report.scale_method == "fit_to_build_volume"
    assert max(report.bbox_mm) == pytest.approx(config.scale_target_size_mm, rel=1e-6)
    assert any("NOT physically meaningful" in w for w in report.warnings)


def test_missing_inputs_raise_with_a_remedy(tmp_path):
    with pytest.raises(PrintPrepError) as exc_info:
        run_print_prep(
            tmp_path / "mesh",
            tmp_path / "dense",
            tmp_path / "sfm",
            tmp_path / "output",
            MeshConfig(),
            PrintPrepConfig(),
            "object",
        )
    assert exc_info.value.remedy
