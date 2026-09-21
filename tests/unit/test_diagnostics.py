"""diagnostics.py: registration-rate/track-length/triangulation-angle math
and the warning thresholds.

Most tests here assert against the real reconstruction produced by the
shared, session-scoped `sfm_fixture` (see tests/conftest.py) and are
marked individually `slow` (pytest.ini: "integration tests requiring
colmap/heavy deps"). `test_summarize_handles_none_reconstruction` doesn't
need a real reconstruction at all -- just a valid (if empty) COLMAP
database -- so it stays fast and unmarked.
"""

from pathlib import Path

import pycolmap
import pytest

from v2m.config import SfmConfig
from v2m.phase2_sfm import diagnostics


def _load_reconstruction(sfm_fixture) -> pycolmap.Reconstruction:
    return pycolmap.Reconstruction(sfm_fixture["run_dir"] / "sfm" / "sparse" / "final")


@pytest.mark.slow
def test_low_keypoint_images_matches_database_counts(sfm_fixture):
    database_path = sfm_fixture["run_dir"] / "sfm" / "database.db"
    flagged = diagnostics.low_keypoint_images(database_path, min_keypoints=600)
    assert all(item.keypoint_count < 600 for item in flagged)

    # Cross-check one flagged name directly against the database.
    with pycolmap.Database.open(database_path) as db:
        for item in flagged:
            image = next(img for img in db.read_all_images() if img.name == item.name)
            assert db.num_keypoints_for_image(image.image_id) == item.keypoint_count


@pytest.mark.slow
def test_low_keypoint_images_empty_with_a_low_enough_threshold(sfm_fixture):
    database_path = sfm_fixture["run_dir"] / "sfm" / "database.db"
    assert diagnostics.low_keypoint_images(database_path, min_keypoints=1) == []


@pytest.mark.slow
def test_median_triangulation_angle_is_positive_and_reasonable(sfm_fixture):
    reconstruction = _load_reconstruction(sfm_fixture)
    angle = diagnostics.median_triangulation_angle_deg(reconstruction)
    assert angle is not None
    # A real orbiting capture triangulates well above the "pure rotation"
    # floor (2 deg, Section 3.5) but well under a physically implausible
    # value for a smooth, small-step orbit.
    assert 2.0 < angle < 90.0


def test_median_triangulation_angle_none_for_empty_reconstruction():
    empty = pycolmap.Reconstruction()
    assert diagnostics.median_triangulation_angle_deg(empty) is None


@pytest.mark.slow
def test_summarize_flags_low_registration_rate(sfm_fixture):
    reconstruction = _load_reconstruction(sfm_fixture)
    database_path = sfm_fixture["run_dir"] / "sfm" / "database.db"
    # An artificially strict minimum makes an otherwise-healthy
    # reconstruction "fail" the registration-rate check.
    strict_config = SfmConfig(registration_rate_min=0.999)
    diag = diagnostics.summarize(
        database_path=database_path,
        reconstruction=reconstruction,
        num_images_total=sfm_fixture["result"].num_images_total,
        config=strict_config,
    )
    assert any("registration_rate" in w for w in diag.warnings)


@pytest.mark.slow
def test_summarize_no_registration_warning_when_threshold_is_lenient(sfm_fixture):
    reconstruction = _load_reconstruction(sfm_fixture)
    database_path = sfm_fixture["run_dir"] / "sfm" / "database.db"
    lenient_config = SfmConfig(registration_rate_min=0.0)
    diag = diagnostics.summarize(
        database_path=database_path,
        reconstruction=reconstruction,
        num_images_total=sfm_fixture["result"].num_images_total,
        config=lenient_config,
    )
    assert not any("registration_rate" in w for w in diag.warnings)


@pytest.mark.slow
def test_summarize_flags_low_track_length(sfm_fixture):
    reconstruction = _load_reconstruction(sfm_fixture)
    database_path = sfm_fixture["run_dir"] / "sfm" / "database.db"
    strict_config = SfmConfig(mean_track_length_min=1000.0)
    diag = diagnostics.summarize(
        database_path=database_path,
        reconstruction=reconstruction,
        num_images_total=sfm_fixture["result"].num_images_total,
        config=strict_config,
    )
    assert any("mean_track_length" in w for w in diag.warnings)


@pytest.mark.slow
def test_summarize_flags_pure_rotation_when_angle_threshold_is_high(sfm_fixture):
    reconstruction = _load_reconstruction(sfm_fixture)
    database_path = sfm_fixture["run_dir"] / "sfm" / "database.db"
    strict_config = SfmConfig(min_triangulation_angle_deg=89.0)
    diag = diagnostics.summarize(
        database_path=database_path,
        reconstruction=reconstruction,
        num_images_total=sfm_fixture["result"].num_images_total,
        config=strict_config,
    )
    assert any("triangulation angle" in w for w in diag.warnings)


def test_summarize_handles_none_reconstruction(tmp_path: Path):
    # A real, valid (just empty) COLMAP database -- summarize() always
    # reads low_keypoint_images() from a real file, even when there's no
    # reconstruction to report on.
    database_path = tmp_path / "empty.db"
    with pycolmap.Database.open(database_path):
        pass

    diag = diagnostics.summarize(
        database_path=database_path,
        reconstruction=None,
        num_images_total=10,
        config=SfmConfig(),
    )
    assert diag.num_images_registered == 0
    assert diag.registration_rate == 0.0
    assert diag.mean_reprojection_error_px is None
    assert diag.mean_track_length is None
    assert any("registration_rate" in w for w in diag.warnings)
