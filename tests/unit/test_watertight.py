"""The watertight repair ladder against tests/fixtures/broken_meshes/.

docs/ARCHITECTURE.md Section 4 (M5) verify: "tests/fixtures/broken_meshes/*
all become watertight." Regenerate the fixtures with
`uv run python tests/fixtures/make_broken_meshes.py`.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import trimesh

from v2m.config import MeshConfig, PrintPrepConfig
from v2m.phase4_print import watertight

BROKEN_MESHES = sorted((Path(__file__).parent.parent / "fixtures" / "broken_meshes").glob("*.obj"))

# Volume of the part of a unit sphere centered at z=0.3 that lies above
# z=0 -- a spherical cap of height h=1.3: pi*h^2*(3r-h)/3.
_OPEN_BOTTOM_CAP_VOLUME = np.pi * 1.3**2 * (3 - 1.3) / 3


def _load(name: str) -> trimesh.Trimesh:
    path = Path(__file__).parent.parent / "fixtures" / "broken_meshes" / f"{name}.obj"
    return trimesh.load(str(path), process=False, force="mesh")


def test_broken_mesh_fixtures_exist():
    assert len(BROKEN_MESHES) >= 8


@pytest.mark.parametrize("path", BROKEN_MESHES, ids=lambda p: p.stem)
def test_every_broken_mesh_becomes_a_valid_solid(path):
    mesh = trimesh.load(str(path), process=False, force="mesh")
    assert not mesh.is_watertight or mesh.volume < 0  # each fixture is genuinely broken

    repaired, rung, _warnings = watertight.make_watertight(
        mesh, "object", MeshConfig(), PrintPrepConfig()
    )

    assert repaired.is_watertight
    assert repaired.is_winding_consistent
    assert repaired.volume > 0
    assert 1 <= rung <= 6


def test_rung1_merges_a_triangle_soup():
    mesh = _load("unmerged_vertices")
    assert len(mesh.vertices) == 36
    watertight.rung1_basic_repair(mesh)
    assert len(mesh.vertices) == 8
    assert mesh.is_watertight


def test_rung1_fixes_inverted_winding():
    mesh = _load("inverted_winding")
    assert mesh.volume < 0
    watertight.rung1_basic_repair(mesh)
    assert mesh.volume == pytest.approx(8.0)


def test_rung2_drops_floating_debris():
    mesh = _load("floating_debris")
    repaired = watertight.rung2_drop_small_components(mesh, MeshConfig())
    assert repaired.bounds[1][0] < 2.0  # the debris sat at x=5
    assert repaired.is_watertight


def test_rung3_fills_small_holes_but_leaves_ones_over_the_limit():
    filled = watertight.rung3_fill_small_holes(_load("small_holes"), PrintPrepConfig())
    assert filled.is_watertight

    too_big = watertight.rung3_fill_small_holes(
        _load("open_bottom"), PrintPrepConfig(max_hole_size_triangles=10)
    )
    assert not too_big.is_watertight


@pytest.mark.skipif(
    importlib.util.find_spec("pymeshlab") is not None,
    reason="pymeshlab is installed, so the no-op path isn't reachable",
)
def test_rung4_is_a_noop_without_pymeshlab():
    mesh = _load("small_holes")
    result, ran = watertight.rung4_remove_nonmanifold(mesh)
    assert ran is False
    assert result is mesh


def test_open_bottom_reaches_rung5_and_is_solid_not_a_shell():
    # With a tight hole-size limit the underside can't be fan-filled, so
    # the ladder has to close it at the ground. Regression: an earlier
    # fill-before-cut version produced a hollow one-voxel shell here
    # (volume ~0.08) that still reported itself watertight.
    repaired, rung, _ = watertight.make_watertight(
        _load("open_bottom"), "object", MeshConfig(), PrintPrepConfig(max_hole_size_triangles=10)
    )
    assert rung == 5
    assert repaired.is_watertight
    assert repaired.volume == pytest.approx(_OPEN_BOTTOM_CAP_VOLUME, rel=0.03)
    pitch = repaired.extents.max() / 256
    assert repaired.bounds[0][2] == pytest.approx(0.0, abs=pitch)


def test_rung5_object_mode_cuts_at_the_ground_tolerance():
    repaired = watertight.rung5_ground_cap(
        _load("open_bottom"), "object", PrintPrepConfig(), ground_tolerance=0.2
    )
    pitch = repaired.extents.max() / 256
    assert repaired.bounds[0][2] == pytest.approx(0.2, abs=pitch)


def test_rung5_scene_mode_adds_a_slab_below_the_ground():
    config = PrintPrepConfig()
    mesh = _load("open_bottom")
    expected_thickness = config.slab_thickness_mm / config.scale_target_size_mm * mesh.extents.max()

    repaired = watertight.rung5_ground_cap(mesh, "scene", config)

    assert repaired.is_watertight
    assert repaired.bounds[0][2] == pytest.approx(-expected_thickness, rel=0.05)


def test_rung6_keeps_the_mesh_where_it_was():
    # Regression: VoxelGrid.marching_cubes returns voxel-index coordinates;
    # without re-applying the grid transform the result lands at the
    # origin at ~resolution x the size.
    sphere = trimesh.creation.icosphere(subdivisions=3, radius=2.0)
    sphere.apply_translation([100.0, -50.0, 30.0])

    remeshed = watertight.rung6_voxel_remesh(sphere)

    assert remeshed.is_watertight
    np.testing.assert_allclose(remeshed.bounds, sphere.bounds, atol=4.0 / 256 * 2)
