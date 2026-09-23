"""M9 optional backends: OpenMVS orchestration (against stand-in
binaries -- OpenMVS isn't installed here), learned-feature SfM and the
tiled TSDF on the real fixture (slow)."""

import json
import os
import stat
from pathlib import Path

import numpy as np
import open3d as o3d
import pytest

from v2m import capability
from v2m.config import DenseConfig, IngestConfig, SfmConfig
from v2m.errors import CapabilityError, SfMError
from v2m.phase2_sfm.backends import hloc_backend
from v2m.phase2_sfm.dense import densify, monodepth_tsdf, openmvs

FAKE_INTERFACE = """#!/bin/sh
echo "InterfaceCOLMAP $@" >> calls.log
touch scene.mvs
"""
FAKE_DENSIFY = """#!/bin/sh
echo "DensifyPointCloud $@" >> calls.log
cat > scene_dense.ply <<'PLY'
ply
format ascii 1.0
element vertex 3
property float x
property float y
property float z
end_header
0 0 0
1 0 0
0 1 0
PLY
"""


def _install_fake_openmvs(bin_dir: Path, monkeypatch, densify_script=FAKE_DENSIFY) -> None:
    bin_dir.mkdir()
    for name, script in (
        ("InterfaceCOLMAP", FAKE_INTERFACE),
        ("DensifyPointCloud", densify_script),
    ):
        path = bin_dir / name
        path.write_text(script)
        path.chmod(path.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")


def _fake_sfm_workspace(tmp_path: Path) -> Path:
    sfm = tmp_path / "sfm"
    (sfm / "undistorted" / "sparse").mkdir(parents=True)
    (sfm / "undistorted" / "images").mkdir()
    return sfm


def test_openmvs_runs_both_tools_on_the_cpu_and_returns_the_cloud(tmp_path, monkeypatch):
    _install_fake_openmvs(tmp_path / "bin", monkeypatch)
    sfm = _fake_sfm_workspace(tmp_path)

    result = densify(sfm, tmp_path / "dense", DenseConfig(backend="openmvs"))

    assert result.backend == "openmvs" and result.num_points == 3
    assert len(o3d.io.read_point_cloud(str(tmp_path / "dense" / "dense.ply")).points) == 3
    calls = (tmp_path / "dense" / "openmvs" / "calls.log").read_text().splitlines()
    assert calls[0].startswith("InterfaceCOLMAP -i ")
    assert "--cuda-device -2" in calls[1]  # CPU, always, unless configured otherwise


def test_openmvs_failure_surfaces_its_log(tmp_path, monkeypatch):
    _install_fake_openmvs(
        tmp_path / "bin",
        monkeypatch,
        densify_script="#!/bin/sh\necho 'out of memory' >&2\nexit 3\n",
    )
    with pytest.raises(SfMError) as exc_info:
        openmvs.densify(_fake_sfm_workspace(tmp_path), tmp_path / "dense", DenseConfig())
    assert "out of memory" in exc_info.value.message
    assert "densify.log" in exc_info.value.remedy


def test_openmvs_absent_is_a_capability_error_with_a_remedy(tmp_path, monkeypatch):
    monkeypatch.setenv("PATH", str(tmp_path))
    with pytest.raises(CapabilityError) as exc_info:
        openmvs.densify(_fake_sfm_workspace(tmp_path), tmp_path / "dense", DenseConfig())
    assert "monodepth_tsdf" in exc_info.value.remedy


def test_auto_never_selects_openmvs_even_when_installed():
    caps = capability.probe()
    caps.openmvs_binary = "/usr/local/bin/DensifyPointCloud"
    caps.has_cuda = True
    assert caps.dense_backend != "openmvs"


def test_learned_pairs_are_a_cyclic_window():
    pairs = hloc_backend._pairs(10, SfmConfig(matching_overlap=2), exhaustive=False)
    assert (0, 1) in pairs and (0, 2) in pairs and (8, 9) in pairs
    assert (0, 9) in pairs and (0, 8) in pairs  # wraps around: closes an orbit
    assert (0, 3) not in pairs
    assert len(pairs) == 20
    assert len(hloc_backend._pairs(10, SfmConfig(), exhaustive=True)) == 45


# -- slow: the real thing on the fixture -----------------------------------------------------


@pytest.mark.slow
def test_learned_features_reconstruct_the_fixture(tmp_path):
    pytest.importorskip("kornia")
    from v2m.phase1_ingest.extract import run_extract
    from v2m.phase2_sfm.sfm import run_sparse_sfm

    run_extract(
        Path(__file__).parent.parent / "fixtures" / "sample.mp4",
        tmp_path,
        IngestConfig(frame_budget=30),
    )
    result = run_sparse_sfm(
        tmp_path / "frames",
        tmp_path / "sfm",
        SfmConfig(
            feature_backend="disk",
            matching_overlap=5,
            learned_max_keypoints=1024,
            learned_resize_px=640,
        ),
    )
    assert result.num_images_registered / result.num_images_total >= 0.9
    assert result.mean_reprojection_error_px < 1.5


@pytest.mark.slow
def test_learned_fallback_rescues_a_failed_sift_run(sfm_fixture, tmp_path):
    pytest.importorskip("kornia")
    from v2m.phase2_sfm.sfm import run_sparse_sfm

    # Both SIFT passes are made to find nothing; DISK + LightGlue steps in.
    config = SfmConfig(
        sift_peak_threshold=5.0,
        sift_peak_threshold_retry=5.0,
        matching_overlap=5,
        learned_max_keypoints=1024,
        learned_resize_px=640,
    )
    result = run_sparse_sfm(sfm_fixture["images_dir"], tmp_path / "sfm", config)

    assert result.num_images_registered / result.num_images_total >= 0.9
    diagnostics = json.loads((tmp_path / "sfm" / "diagnostics.json").read_text())
    assert diagnostics["attempt"] == 3


@pytest.mark.slow
def test_tiled_tsdf_matches_the_single_volume(
    sfm_fixture, dense_fixture, geometric_depth_estimator_cls, tmp_path
):
    single = o3d.io.read_point_cloud(str(dense_fixture["output_dir"] / "dense.ply"))
    # Pretend the capture wanted far finer voxels than one volume holds:
    # the voxel/distance ratio shrinks 6x, so the cube needs ~2 tiles/axis.
    config = DenseConfig(nominal_capture_distance_m=3.0, max_tiles=8)
    # A fresh estimator: dense_fixture's is session-wide, and its call
    # count is asserted elsewhere.
    frames = json.loads((sfm_fixture["run_dir"] / "frames" / "frames.json").read_text())
    estimator = geometric_depth_estimator_cls(
        {f["path"]: f["frame_index"] for f in frames if f["accepted"]}
    )
    result = monodepth_tsdf.densify(
        sfm_fixture["run_dir"] / "sfm", tmp_path, config, depth_estimator=estimator
    )
    tiled = o3d.io.read_point_cloud(str(tmp_path / "dense.ply"))
    log = json.loads((tmp_path / "alignment_report.json").read_text())
    assert log  # frames were aligned and integrated

    single_pts, tiled_pts = np.asarray(single.points), np.asarray(tiled.points)
    # Same object, same place...
    np.testing.assert_allclose(
        np.percentile(tiled_pts, [2, 98], axis=0),
        np.percentile(single_pts, [2, 98], axis=0),
        atol=0.05 * np.ptp(single_pts, axis=0).max(),
    )
    # ...no duplicated seams: every tiled point sits within one fine voxel of
    # a single-volume surface point, and no two tiled points coincide.
    distances = np.asarray(tiled.compute_point_cloud_distance(single))
    assert np.median(distances) < 0.01 * np.ptp(single_pts, axis=0).max()
    assert len(np.unique(np.round(tiled_pts, 9), axis=0)) == len(tiled_pts)
    assert result.num_points > 0
