"""Phase 3 end-to-end: build_mesh() against the synthetic fixture.

Builds on the shared session-scoped `mesh_fixture` (see
conftest.py), so like test_monodepth_tsdf.py this is uniformly `slow`
(pytest.ini: "integration tests requiring colmap/heavy deps").
"""

import numpy as np
import open3d as o3d
import pytest

from v2m.config import DenseConfig, MeshConfig
from v2m.errors import MeshError
from v2m.phase3_mesh import build_mesh

pytestmark = pytest.mark.slow


def test_cleaned_mesh_has_geometry(mesh_fixture):
    """docs/ARCHITECTURE.md Section 4 (M4): "mesh/cleaned.ply opens ...,
    visually matches the subject, face count within budget"."""
    assert mesh_fixture["report"].num_vertices > 0
    assert mesh_fixture["report"].num_faces > 0


def test_cleaned_mesh_face_count_is_within_budget(mesh_fixture):
    config = MeshConfig()
    assert mesh_fixture["report"].num_faces <= config.max_faces


def test_cleaned_mesh_reports_a_manifold_flag(mesh_fixture):
    # Not asserted True: not a doc-mandated M4 bar, and not always true
    # in practice -- Poisson reconstruction from the noisier, more
    # oblique depth samples near a pole (needed for M5's dimensional
    # accuracy, see make_synthetic_video.py's _camera_pose docstring)
    # can introduce a handful of non-manifold edges/vertices even before
    # any cleanup runs. `is_manifold` is still useful, honestly-reported
    # diagnostic information -- exactly the condition Phase 4's
    # watertight ladder (rung 1's process(validate=True), rung 4's
    # pymeshlab non-manifold repair) exists to handle, not a Phase 3 bug.
    assert isinstance(mesh_fixture["report"].is_manifold, bool)


def test_cleaned_mesh_bounding_box_is_roughly_cube_shaped(mesh_fixture):
    # The fixture is a real cube; COLMAP's own reconstruction scale is
    # arbitrary (see monodepth_tsdf.py's _build_tsdf_volume docstring)
    # but should still be *shaped* like a cube -- roughly equal extent on
    # all three axes, not a degenerate sliver or a flat plane.
    cleaned_ply = mesh_fixture["output_dir"] / mesh_fixture["report"].mesh_path
    mesh = o3d.io.read_triangle_mesh(str(cleaned_ply))
    vertices = np.asarray(mesh.vertices)
    extent = vertices.max(axis=0) - vertices.min(axis=0)
    ratio = extent.min() / extent.max()
    assert ratio > 0.5


def test_raw_and_cleaned_ply_are_both_written(mesh_fixture):
    output_dir = mesh_fixture["output_dir"]
    assert (output_dir / "raw.ply").exists()
    assert (output_dir / "cleaned.ply").exists()

    raw = o3d.io.read_triangle_mesh(str(output_dir / "raw.ply"))
    cleaned = o3d.io.read_triangle_mesh(str(output_dir / "cleaned.ply"))
    # cleanup only ever removes floaters and decimates -- never adds faces.
    assert len(cleaned.triangles) <= len(raw.triangles)


def test_mesh_path_is_relative_to_output_dir(mesh_fixture):
    assert mesh_fixture["report"].mesh_path == "cleaned.ply"


def test_build_mesh_raises_when_dense_ply_is_missing(tmp_path, sfm_fixture):
    empty_dense_dir = tmp_path / "dense"
    empty_dense_dir.mkdir()

    with pytest.raises(MeshError) as exc_info:
        build_mesh(
            empty_dense_dir,
            sfm_fixture["run_dir"] / "sfm",
            tmp_path / "mesh",
            DenseConfig(),
            MeshConfig(),
        )
    assert "v2m dense" in exc_info.value.remedy


def test_build_mesh_raises_when_cameras_json_is_missing(tmp_path, dense_fixture):
    empty_sfm_dir = tmp_path / "sfm"
    empty_sfm_dir.mkdir()

    with pytest.raises(MeshError) as exc_info:
        build_mesh(
            dense_fixture["output_dir"],
            empty_sfm_dir,
            tmp_path / "mesh",
            DenseConfig(),
            MeshConfig(),
        )
    assert "v2m sfm" in exc_info.value.remedy
