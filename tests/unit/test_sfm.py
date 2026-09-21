"""Phase 2a orchestration: sparse SfM against the synthetic fixture.

Most assertions here share one expensive real-COLMAP run via the
session-scoped `sfm_fixture` conftest fixture; a few tests that need a
*different* config (forced failure, forced retry, CLAHE) run their own
pipeline pass. All are marked `slow` (pytest.ini: "integration tests
requiring colmap/heavy deps").
"""

import json
from pathlib import Path

import pytest

from v2m.config import SfmConfig
from v2m.errors import SfMError
from v2m.phase2_sfm.sfm import run_sparse_sfm

pytestmark = pytest.mark.slow


def test_registration_rate_meets_architecture_bar(sfm_fixture):
    """docs/ARCHITECTURE.md Section 4 (M2): "registration rate >= 0.9"."""
    result = sfm_fixture["result"]
    rate = result.num_images_registered / result.num_images_total
    assert rate >= 0.9


def test_reprojection_error_meets_architecture_bar(sfm_fixture):
    """docs/ARCHITECTURE.md Section 4 (M2): "reprojection error < 1.5 px"."""
    result = sfm_fixture["result"]
    assert result.mean_reprojection_error_px < 1.5


def test_mean_track_length_is_healthy(sfm_fixture):
    result = sfm_fixture["result"]
    assert result.mean_track_length > 2.0


def test_output_files_exist(sfm_fixture):
    sfm_dir = sfm_fixture["run_dir"] / "sfm"
    result = sfm_fixture["result"]
    assert (sfm_dir / result.sparse_points_path).exists()
    assert (sfm_dir / result.cameras_path).exists()
    assert (sfm_dir / "diagnostics.json").exists()
    assert (sfm_dir / "sparse" / "final").is_dir()
    assert (sfm_dir / "undistorted" / "images").is_dir()


def test_diagnostics_json_matches_result(sfm_fixture):
    sfm_dir = sfm_fixture["run_dir"] / "sfm"
    diagnostics = json.loads((sfm_dir / "diagnostics.json").read_text())
    result = sfm_fixture["result"]
    assert diagnostics["num_images_registered"] == result.num_images_registered
    assert diagnostics["num_images_total"] == result.num_images_total
    assert diagnostics["mean_reprojection_error_px"] == pytest.approx(
        result.mean_reprojection_error_px
    )


def test_cameras_json_structure(sfm_fixture):
    sfm_dir = sfm_fixture["run_dir"] / "sfm"
    cameras = json.loads((sfm_dir / "cameras.json").read_text())
    assert "convention" in cameras
    assert len(cameras["cameras"]) == 1  # CameraMode.SINGLE: one physical camera
    camera = next(iter(cameras["cameras"].values()))
    assert camera["width"] == camera["height"]  # our fixture renders square frames
    assert len(camera["params"]) >= 3  # at least fx/fy, cx, cy

    result = sfm_fixture["result"]
    assert len(cameras["images"]) == result.num_images_registered
    sample_pose = next(iter(cameras["images"].values()))
    assert len(sample_pose["rotation_quat_xyzw"]) == 4
    assert len(sample_pose["translation"]) == 3


def test_missing_frames_raises_sfm_error_with_remedy(tmp_path):
    empty_images_dir = tmp_path / "frames"
    empty_images_dir.mkdir()
    with pytest.raises(SfMError) as exc_info:
        run_sparse_sfm(empty_images_dir, tmp_path / "sfm", SfmConfig())
    assert exc_info.value.remedy is not None
    assert "extract" in exc_info.value.remedy


def test_total_failure_writes_diagnostics_before_raising(sfm_fixture, tmp_path):
    images_dir = sfm_fixture["images_dir"]
    # An impossibly strict threshold on both passes rejects virtually
    # every feature -> zero registered images on both attempts.
    config = SfmConfig(sift_peak_threshold=5.0, sift_peak_threshold_retry=5.0)
    output_dir = tmp_path / "sfm_fail"

    with pytest.raises(SfMError) as exc_info:
        run_sparse_sfm(images_dir, output_dir, config)
    assert "0/" in exc_info.value.message

    diagnostics = json.loads((output_dir / "diagnostics.json").read_text())
    assert diagnostics["num_images_registered"] == 0
    assert any("registration_rate" in w for w in diagnostics["warnings"])


def test_retry_recovers_from_a_too_strict_initial_threshold(sfm_fixture, tmp_path):
    images_dir = sfm_fixture["images_dir"]
    config = SfmConfig(sift_peak_threshold=5.0, sift_peak_threshold_retry=0.0066)
    output_dir = tmp_path / "sfm_retry"

    result = run_sparse_sfm(images_dir, output_dir, config)

    assert result.num_images_registered / result.num_images_total >= 0.9
    diagnostics = json.loads((output_dir / "diagnostics.json").read_text())
    assert diagnostics["attempt"] == 2  # the retry pass won


def test_clahe_preprocessing_produces_a_valid_reconstruction(sfm_fixture, tmp_path):
    images_dir = sfm_fixture["images_dir"]
    config = SfmConfig(use_clahe=True)
    output_dir = tmp_path / "sfm_clahe"

    result = run_sparse_sfm(images_dir, output_dir, config)

    assert result.num_images_registered / result.num_images_total >= 0.9
    assert (output_dir / "clahe_frames").is_dir()
    assert len(list((output_dir / "clahe_frames").glob("*.jpg"))) == len(
        list(Path(images_dir).glob("*.jpg"))
    )
