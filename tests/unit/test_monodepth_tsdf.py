"""Phase 2b end-to-end: monodepth_tsdf against the synthetic fixture.

Everything here needs a real sparse reconstruction (the shared
session-scoped `sfm_fixture`/`dense_fixture`, see conftest.py) plus
open3d/pycolmap, so it's uniformly `slow` (pytest.ini: "integration
tests requiring colmap/heavy deps"). huggingface.co is
network-policy-blocked in this sandbox (see CLAUDE.md), so `dense_fixture`
injects `GeometricDepthEstimator` -- a `DepthEstimator` built from the
fixture's own known ground-truth geometry
(tests/fixtures/make_synthetic_video.py's `render_true_depth`) -- instead
of the real, untested-here `TransformersDepthEstimator`.
"""

import json

import open3d as o3d
import pycolmap
import pytest

from v2m.config import DenseConfig
from v2m.errors import SfMError
from v2m.phase2_sfm.dense import monodepth_tsdf

pytestmark = pytest.mark.slow


def test_dense_point_count_meets_architecture_bar(sfm_fixture, dense_fixture):
    """docs/ARCHITECTURE.md Section 4 (M3): "dense.ply has >=20x the sparse point count"."""
    sparse_cloud = o3d.io.read_point_cloud(str(sfm_fixture["run_dir"] / "sfm" / "sparse.ply"))
    assert dense_fixture["result"].num_points >= 20 * len(sparse_cloud.points)


def test_dense_ply_is_written_and_matches_reported_count(dense_fixture):
    dense_ply = dense_fixture["output_dir"] / "dense.ply"
    assert dense_ply.exists()
    cloud = o3d.io.read_point_cloud(str(dense_ply))
    assert len(cloud.points) == dense_fixture["result"].num_points


def test_every_registered_image_was_integrated(sfm_fixture, dense_fixture):
    # The fixture's clean 100% registration + generous correspondence
    # counts (M2's rotation-handedness regression fix) mean every image
    # should have enough 2D-3D correspondences to align -- none skipped.
    assert dense_fixture["estimator"].calls == sfm_fixture["result"].num_images_registered

    alignment_report = json.loads(
        (dense_fixture["output_dir"] / "alignment_report.json").read_text()
    )
    assert [r for r in alignment_report if "skipped" in r] == []


def test_alignment_rmse_is_small(dense_fixture):
    # In depth_alignment.rmse's native units (1/reconstruction-scale,
    # COLMAP's own arbitrary scale -- see _build_tsdf_volume's
    # docstring). This checks the affine fit tracks COLMAP's own points
    # well; recovering a literal metric depth RMSE against known ground
    # truth is test_depth_alignment.py's job, which controls (a, b)
    # directly instead of depending on whatever scale a real SfM run
    # happens to produce.
    assert dense_fixture["result"].alignment_rmse is not None
    assert dense_fixture["result"].alignment_rmse < 0.01


def test_result_backend_is_monodepth_tsdf(dense_fixture):
    assert dense_fixture["result"].backend == "monodepth_tsdf"


def test_depth_npy_files_are_written_per_integrated_image(dense_fixture, sfm_fixture):
    depth_dir = dense_fixture["output_dir"] / "depth"
    npy_files = list(depth_dir.glob("*.npy"))
    assert len(npy_files) == sfm_fixture["result"].num_images_registered


def test_densify_raises_when_no_sparse_reconstruction(tmp_path):
    empty_sfm_dir = tmp_path / "sfm"
    empty_sfm_dir.mkdir()
    with pytest.raises(SfMError) as exc_info:
        monodepth_tsdf.densify(empty_sfm_dir, tmp_path / "dense", DenseConfig())
    assert "v2m sfm" in exc_info.value.remedy


def test_build_tsdf_volume_resolution_is_within_configured_bounds(sfm_fixture):
    reconstruction = pycolmap.Reconstruction(sfm_fixture["run_dir"] / "sfm" / "sparse" / "final")
    volume = monodepth_tsdf._build_tsdf_volume(reconstruction, DenseConfig())
    assert (
        monodepth_tsdf._MIN_TSDF_RESOLUTION
        <= volume.resolution
        <= monodepth_tsdf.MAX_TSDF_RESOLUTION
    )


def test_build_tsdf_volume_sdf_trunc_preserves_configured_voxel_ratio(sfm_fixture):
    # This milestone's fix for COLMAP's scale ambiguity (see
    # _build_tsdf_volume's docstring): sdf_trunc is sized as a multiple
    # of the *actual* per-voxel size, not DenseConfig's literal value,
    # since that value assumes real metres and COLMAP's own
    # reconstruction scale at this point in the pipeline is arbitrary.
    config = DenseConfig()
    reconstruction = pycolmap.Reconstruction(sfm_fixture["run_dir"] / "sfm" / "sparse" / "final")
    volume = monodepth_tsdf._build_tsdf_volume(reconstruction, config)
    voxel_size_actual = volume.length / volume.resolution
    expected_ratio = config.tsdf_sdf_trunc_m / config.tsdf_voxel_size_m
    assert volume.sdf_trunc == pytest.approx(expected_ratio * voxel_size_actual)
