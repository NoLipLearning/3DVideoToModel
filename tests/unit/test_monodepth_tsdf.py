"""Phase 2b end-to-end: monodepth_tsdf against the synthetic fixture.

Everything here needs a real sparse reconstruction (the shared
session-scoped `sfm_fixture`, see conftest.py) plus open3d/pycolmap, so
it's uniformly `slow` (pytest.ini: "integration tests requiring
colmap/heavy deps"). huggingface.co is network-policy-blocked in this
sandbox (see CLAUDE.md), so every test injects `GeometricDepthEstimator`
below -- a `DepthEstimator` built from the fixture's own known
ground-truth geometry (tests/fixtures/make_synthetic_video.py's
`render_true_depth`) -- instead of the real, untested-here
`TransformersDepthEstimator`.
"""

import json
import sys
from pathlib import Path

import numpy as np
import open3d as o3d
import pycolmap
import pytest

from v2m.config import DenseConfig
from v2m.errors import SfMError
from v2m.phase2_sfm.dense import monodepth_tsdf

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"
sys.path.insert(0, str(FIXTURES_DIR))
import make_synthetic_video as msv  # noqa: E402

pytestmark = pytest.mark.slow

GROUND_TRUTH = json.loads((FIXTURES_DIR / "sample_ground_truth.json").read_text())
_POSES_BY_FRAME_INDEX = {p["frame_index"]: p for p in GROUND_TRUTH["poses"]}
_CAMERA_MATRIX = np.array(GROUND_TRUTH["camera_matrix"], dtype=np.float64)


class GeometricDepthEstimator:
    """Renders each fixture frame's *exact* known depth via
    `render_true_depth`, keyed by the accepted-frame `image_name` COLMAP
    itself uses, standing in for a real (network-dependent) depth model.
    """

    def __init__(self, path_to_frame_index: dict[str, int]):
        self._path_to_frame_index = path_to_frame_index
        self._faces = msv._build_faces()
        self.calls = 0

    def predict_disparity(self, image_bgr: np.ndarray, image_name: str | None = None) -> np.ndarray:
        self.calls += 1
        pose = _POSES_BY_FRAME_INDEX[self._path_to_frame_index[image_name]]
        rvec = np.array(pose["rvec"], dtype=np.float64)
        tvec = np.array(pose["tvec"], dtype=np.float64)
        cam_center = np.array(pose["camera_center_world_mm"], dtype=np.float64)
        depth_mm = msv.render_true_depth(self._faces, rvec, tvec, cam_center, _CAMERA_MATRIX)
        with np.errstate(divide="ignore", invalid="ignore"):
            disparity = np.where(depth_mm > 1e-6, 1.0 / depth_mm, 0.0)
        return disparity.astype(np.float32)


@pytest.fixture(scope="module")
def dense_result(sfm_fixture, tmp_path_factory) -> dict:
    """Runs densify() once (module-scoped: a real TSDF pass, even against
    this small fixture, is too expensive to repeat per-test)."""
    frames = json.loads((sfm_fixture["run_dir"] / "frames" / "frames.json").read_text())
    path_to_frame_index = {f["path"]: f["frame_index"] for f in frames if f.get("accepted")}
    estimator = GeometricDepthEstimator(path_to_frame_index)

    output_dir = tmp_path_factory.mktemp("dense_fixture")
    result = monodepth_tsdf.densify(
        sfm_fixture["run_dir"] / "sfm",
        output_dir,
        DenseConfig(),
        depth_estimator=estimator,
    )
    return {"result": result, "output_dir": output_dir, "estimator": estimator}


def test_dense_point_count_meets_architecture_bar(sfm_fixture, dense_result):
    """docs/ARCHITECTURE.md Section 4 (M3): "dense.ply has >=20x the sparse point count"."""
    sparse_cloud = o3d.io.read_point_cloud(str(sfm_fixture["run_dir"] / "sfm" / "sparse.ply"))
    assert dense_result["result"].num_points >= 20 * len(sparse_cloud.points)


def test_dense_ply_is_written_and_matches_reported_count(dense_result):
    dense_ply = dense_result["output_dir"] / "dense.ply"
    assert dense_ply.exists()
    cloud = o3d.io.read_point_cloud(str(dense_ply))
    assert len(cloud.points) == dense_result["result"].num_points


def test_every_registered_image_was_integrated(sfm_fixture, dense_result):
    # The fixture's clean 100% registration + generous correspondence
    # counts (M2's rotation-handedness regression fix) mean every image
    # should have enough 2D-3D correspondences to align -- none skipped.
    assert dense_result["estimator"].calls == sfm_fixture["result"].num_images_registered

    alignment_report = json.loads(
        (dense_result["output_dir"] / "alignment_report.json").read_text()
    )
    assert [r for r in alignment_report if "skipped" in r] == []


def test_alignment_rmse_is_small(dense_result):
    # In depth_alignment.rmse's native units (1/reconstruction-scale,
    # COLMAP's own arbitrary scale -- see _build_tsdf_volume's
    # docstring). This checks the affine fit tracks COLMAP's own points
    # well; recovering a literal metric depth RMSE against known ground
    # truth is test_depth_alignment.py's job, which controls (a, b)
    # directly instead of depending on whatever scale a real SfM run
    # happens to produce.
    assert dense_result["result"].alignment_rmse is not None
    assert dense_result["result"].alignment_rmse < 0.01


def test_result_backend_is_monodepth_tsdf(dense_result):
    assert dense_result["result"].backend == "monodepth_tsdf"


def test_depth_npy_files_are_written_per_integrated_image(dense_result, sfm_fixture):
    depth_dir = dense_result["output_dir"] / "depth"
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
        <= monodepth_tsdf._MAX_TSDF_RESOLUTION
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
