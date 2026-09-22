"""Shared pytest fixtures."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures"
sys.path.insert(0, str(FIXTURES_DIR))
import make_synthetic_video as msv  # noqa: E402

GROUND_TRUTH = json.loads((FIXTURES_DIR / "sample_ground_truth.json").read_text())
_POSES_BY_FRAME_INDEX = {p["frame_index"]: p for p in GROUND_TRUTH["poses"]}
_CAMERA_MATRIX = np.array(GROUND_TRUTH["camera_matrix"], dtype=np.float64)


class GeometricDepthEstimator:
    """A `DepthEstimator` (see phase2_sfm/dense/monodepth_tsdf.py) that
    renders each fixture frame's *exact* known depth via
    `render_true_depth`, keyed by the accepted-frame `image_name` COLMAP
    itself uses, standing in for a real (network-dependent, untestable in
    this sandbox -- huggingface.co is network-policy-blocked, see
    CLAUDE.md) depth model.
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


@pytest.fixture(scope="session")
def sfm_fixture(tmp_path_factory) -> dict:
    """Runs Phase 1 + Phase 2a (default config) once for the whole test
    session against tests/fixtures/sample.mp4.

    Several M2 test modules (test_sfm.py, test_diagnostics.py,
    test_colmap_backend.py) assert against this single, expensive
    (~20-30s) real-COLMAP result rather than each re-running the pipeline
    -- consistent with marking every test that consumes it `slow`
    (pytest.ini: "integration tests requiring colmap/heavy deps").
    """
    from v2m.config import IngestConfig, SfmConfig
    from v2m.phase1_ingest.extract import run_extract
    from v2m.phase2_sfm.sfm import run_sparse_sfm

    run_dir = tmp_path_factory.mktemp("sfm_fixture")
    run_extract(FIXTURES_DIR / "sample.mp4", run_dir, IngestConfig())
    result = run_sparse_sfm(run_dir / "frames", run_dir / "sfm", SfmConfig())
    return {"run_dir": run_dir, "images_dir": run_dir / "frames", "result": result}


@pytest.fixture(scope="session")
def dense_fixture(sfm_fixture, tmp_path_factory) -> dict:
    """Runs Phase 2b (monodepth_tsdf, default config) once for the whole
    test session on top of `sfm_fixture`, using `GeometricDepthEstimator`
    in place of the real (untestable-here) `TransformersDepthEstimator`.

    Session-scoped for the same reason as `sfm_fixture`: a real TSDF pass
    is too expensive to repeat per-test, and test_monodepth_tsdf.py's own
    tests plus test_mesh_build.py's M4 integration tests both need it.
    """
    from v2m.config import DenseConfig
    from v2m.phase2_sfm.dense import monodepth_tsdf

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
