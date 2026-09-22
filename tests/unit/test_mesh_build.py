"""Phase 3 end-to-end: build_mesh() against the synthetic fixture.

Builds on the shared session-scoped `sfm_fixture`/`dense_fixture` (see
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


@pytest.fixture(scope="module")
def mesh_result(sfm_fixture, dense_fixture, tmp_path_factory) -> dict:
    output_dir = tmp_path_factory.mktemp("mesh_fixture")
    report = build_mesh(
        dense_fixture["output_dir"],
        sfm_fixture["run_dir"] / "sfm",
        output_dir,
        DenseConfig(),
        MeshConfig(),
    )
    return {"report": report, "output_dir": output_dir}


def test_cleaned_mesh_has_geometry(mesh_result):
    """docs/ARCHITECTURE.md Section 4 (M4): "mesh/cleaned.ply opens ...,
    visually matches the subject, face count within budget"."""
    assert mesh_result["report"].num_vertices > 0
    assert mesh_result["report"].num_faces > 0


def test_cleaned_mesh_face_count_is_within_budget(mesh_result):
    config = MeshConfig()
    assert mesh_result["report"].num_faces <= config.max_faces


def test_cleaned_mesh_is_edge_and_vertex_manifold(mesh_result):
    # Not the same claim as Phase 4's watertightness -- a raw Poisson
    # mesh can (and typically does) still have boundary holes -- but a
    # healthy one should already have no non-manifold edges/vertices.
    assert mesh_result["report"].is_manifold


def test_cleaned_mesh_bounding_box_is_roughly_cube_shaped(mesh_result):
    # The fixture is a real cube; COLMAP's own reconstruction scale is
    # arbitrary (see monodepth_tsdf.py's _build_tsdf_volume docstring)
    # but should still be *shaped* like a cube -- roughly equal extent on
    # all three axes, not a degenerate sliver or a flat plane.
    cleaned_ply = mesh_result["output_dir"] / mesh_result["report"].mesh_path
    mesh = o3d.io.read_triangle_mesh(str(cleaned_ply))
    vertices = np.asarray(mesh.vertices)
    extent = vertices.max(axis=0) - vertices.min(axis=0)
    ratio = extent.min() / extent.max()
    assert ratio > 0.5


def test_raw_and_cleaned_ply_are_both_written(mesh_result):
    output_dir = mesh_result["output_dir"]
    assert (output_dir / "raw.ply").exists()
    assert (output_dir / "cleaned.ply").exists()

    raw = o3d.io.read_triangle_mesh(str(output_dir / "raw.ply"))
    cleaned = o3d.io.read_triangle_mesh(str(output_dir / "cleaned.ply"))
    # cleanup only ever removes floaters and decimates -- never adds faces.
    assert len(cleaned.triangles) <= len(raw.triangles)


def test_mesh_path_is_relative_to_output_dir(mesh_result):
    assert mesh_result["report"].mesh_path == "cleaned.ply"


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
