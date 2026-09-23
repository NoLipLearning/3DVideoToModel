"""Shared pytest fixtures."""

import json
import sys
from pathlib import Path

import cv2
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
        disparity = disparity.astype(np.float32)

        # render_true_depth always renders at the fixture's fixed
        # IMAGE_SIZE; COLMAP's per-image undistortion can crop to a
        # slightly different size when its estimated SIMPLE_RADIAL `k`
        # is non-trivial (steeper viewing angles -- e.g. the poles of an
        # orbited object -- make this more likely). The real
        # TransformersDepthEstimator always interpolates back to
        # image_bgr's own resolution (see monodepth_tsdf.py), so this
        # double must match that behavior to stand in for it faithfully.
        if disparity.shape != image_bgr.shape[:2]:
            disparity = cv2.resize(
                disparity, (image_bgr.shape[1], image_bgr.shape[0]), interpolation=cv2.INTER_LINEAR
            )
        return disparity


@pytest.fixture(scope="session")
def geometric_depth_estimator_cls():
    """`GeometricDepthEstimator` itself, for tests that build their own
    (e.g. once a pipeline run has produced its own frames.json)."""
    return GeometricDepthEstimator


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


@pytest.fixture(scope="session")
def mesh_fixture(sfm_fixture, dense_fixture, tmp_path_factory) -> dict:
    """Runs Phase 3 once for the whole test session on top of
    `dense_fixture` -- shared by test_mesh_build.py (M4) and
    test_print_prep.py (M5)."""
    from v2m.config import DenseConfig, MeshConfig
    from v2m.phase3_mesh import build_mesh

    output_dir = tmp_path_factory.mktemp("mesh_fixture")
    report = build_mesh(
        dense_fixture["output_dir"],
        sfm_fixture["run_dir"] / "sfm",
        output_dir,
        DenseConfig(),
        MeshConfig(),
    )
    return {"report": report, "output_dir": output_dir}
