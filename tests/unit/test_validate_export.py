"""validate.py's printability checks and export.py's file output."""

import numpy as np
import pytest
import trimesh

from v2m.config import PrintPrepConfig
from v2m.errors import PrintPrepError
from v2m.phase4_print import export, validate


def test_a_clean_printable_cube_has_no_warnings():
    assert (
        validate.validate_mesh(trimesh.creation.box(extents=[150, 150, 150]), PrintPrepConfig())
        == []
    )


def test_sharp_edges_are_not_mistaken_for_thin_walls():
    # Regression: trimesh's default max_sphere thickness reads ~0.09mm
    # next to any convex edge of a clean cube.
    cube = trimesh.creation.box(extents=[20, 20, 20])
    warnings = validate.validate_mesh(cube, PrintPrepConfig())
    assert not any("wall thickness" in w for w in warnings)


def test_a_thin_slab_is_flagged():
    warnings = validate.validate_mesh(
        trimesh.creation.box(extents=[100, 100, 0.3]), PrintPrepConfig()
    )
    assert any("wall thickness" in w for w in warnings)


def test_an_oversized_model_is_flagged():
    warnings = validate.validate_mesh(
        trimesh.creation.box(extents=[300, 100, 100]), PrintPrepConfig()
    )
    assert any("build volume" in w for w in warnings)


def test_a_torus_is_flagged_for_its_topology_but_not_rejected():
    torus = trimesh.creation.torus(major_radius=40, minor_radius=10)
    warnings = validate.validate_mesh(torus, PrintPrepConfig())
    assert any("Euler number" in w for w in warnings)


def test_a_non_watertight_mesh_is_rejected():
    box = trimesh.creation.box(extents=[10, 10, 10])
    open_box = trimesh.Trimesh(vertices=box.vertices, faces=box.faces[:-1], process=False)
    with pytest.raises(PrintPrepError):
        validate.validate_mesh(open_box, PrintPrepConfig())


def test_an_inside_out_mesh_is_rejected():
    box = trimesh.creation.box(extents=[10, 10, 10])
    inverted = trimesh.Trimesh(vertices=box.vertices, faces=np.fliplr(box.faces), process=False)
    with pytest.raises(PrintPrepError):
        validate.validate_mesh(inverted, PrintPrepConfig())


def test_export_writes_binary_stl_obj_and_glb(tmp_path):
    cube = trimesh.creation.box(extents=[20, 30, 40])
    written = export.export_mesh(cube, tmp_path)

    assert set(written) == {"stl", "obj", "glb"}
    for filename in written.values():
        assert (tmp_path / filename).stat().st_size > 0
    assert (tmp_path / "model.stl").read_bytes()[:5] != b"solid"  # binary, not ASCII

    reloaded = trimesh.load(str(tmp_path / "model.stl"))
    assert reloaded.is_watertight
    assert reloaded.volume == pytest.approx(20 * 30 * 40)
