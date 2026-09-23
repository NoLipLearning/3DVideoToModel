"""Generate tests/fixtures/broken_meshes/*.obj -- one deliberately broken
mesh per defect class the watertight repair ladder (phase4_print/
watertight.py) has to handle. docs/ARCHITECTURE.md Section 4, M5 verify:
"tests/fixtures/broken_meshes/* all become watertight."

Every mesh sits in phase4_print's canonical frame (ground at z=0, +Z up)
with its bulk above z=0, so rung 5's ground cut has something sensible to
keep. Each is built from trimesh primitives rather than hand-typed, so
the defect is exactly the one named and nothing else.

Regenerate with: `uv run python tests/fixtures/make_broken_meshes.py`
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import trimesh

OUTPUT_DIR = Path(__file__).parent / "broken_meshes"


def _sphere(center_z: float = 1.2) -> trimesh.Trimesh:
    sphere = trimesh.creation.icosphere(subdivisions=3, radius=1.0)
    sphere.apply_translation([0.0, 0.0, center_z])
    return sphere


def unmerged_vertices() -> trimesh.Trimesh:
    """Every face owns its own three vertices (a "triangle soup" as many
    exporters write it): watertight only once coincident vertices are
    merged -- rung 1."""
    box = trimesh.creation.box(extents=[2.0, 2.0, 2.0])
    box.apply_translation([0.0, 0.0, 1.0])
    soup = box.triangles.reshape(-1, 3)
    return trimesh.Trimesh(vertices=soup, faces=np.arange(len(soup)).reshape(-1, 3), process=False)


def inverted_winding() -> trimesh.Trimesh:
    """Closed, but every face wound inward (negative volume) -- rung 1's
    winding/normal fix."""
    box = trimesh.creation.box(extents=[2.0, 2.0, 2.0])
    box.apply_translation([0.0, 0.0, 1.0])
    return trimesh.Trimesh(vertices=box.vertices, faces=np.fliplr(box.faces), process=False)


def duplicate_and_degenerate_faces() -> trimesh.Trimesh:
    """A closed sphere plus exact duplicates of some faces and some
    zero-area faces -- rung 1's validate pass."""
    sphere = _sphere()
    duplicates = sphere.faces[:10]
    degenerate = np.array([[0, 0, 1], [2, 2, 2], [3, 4, 3]])
    faces = np.vstack([sphere.faces, duplicates, degenerate])
    return trimesh.Trimesh(vertices=sphere.vertices, faces=faces, process=False)


def floating_debris() -> trimesh.Trimesh:
    """A closed sphere plus a tiny, open (non-watertight) triangle patch
    floating well away from it -- rung 2's component filter."""
    sphere = _sphere()
    patch = trimesh.creation.box(extents=[0.05, 0.05, 0.05])
    patch.apply_translation([5.0, 5.0, 3.0])
    open_patch = trimesh.Trimesh(vertices=patch.vertices, faces=patch.faces[:-2], process=False)
    return trimesh.util.concatenate([sphere, open_patch])


def small_holes() -> trimesh.Trimesh:
    """A sphere with a few scattered single faces removed -- small enough
    for rung 3's size-gated fill."""
    sphere = _sphere()
    keep = np.ones(len(sphere.faces), dtype=bool)
    keep[[0, 200, 400, 600]] = False
    mesh = trimesh.Trimesh(vertices=sphere.vertices, faces=sphere.faces[keep], process=False)
    mesh.remove_unreferenced_vertices()
    return mesh


def open_bottom() -> trimesh.Trimesh:
    """The Poisson-mesh failure Section 3.3 is built around: the whole
    underside was never seen, leaving one large, open boundary loop just
    below the ground plane. The loop is convex (40 edges), so at the
    default `max_hole_size_triangles` rung 3's fan fill closes it safely;
    tests also run it with a tight hole-size limit to force rung 5's
    ground cut, the path a large non-convex opening would take."""
    sphere = _sphere(center_z=0.3)
    centroids = sphere.triangles.mean(axis=1)
    mesh = trimesh.Trimesh(
        vertices=sphere.vertices, faces=sphere.faces[centroids[:, 2] > -0.4], process=False
    )
    mesh.remove_unreferenced_vertices()
    return mesh


def nonmanifold_edge() -> trimesh.Trimesh:
    """Two closed boxes touching along exactly one shared edge (a
    "bowtie"): that edge is used by four faces. Rung 2's component split
    separates the two boxes by face adjacency -- each comes back as its
    own closed body with its own copy of the shared edge -- so this is
    repaired without any re-meshing."""
    box_a = trimesh.creation.box(extents=[1.0, 1.0, 1.0])
    box_a.apply_translation([0.5, 0.5, 0.5])
    box_b = trimesh.creation.box(extents=[1.0, 1.0, 1.0])
    box_b.apply_translation([-0.5, -0.5, 0.5])
    combined = trimesh.util.concatenate([box_a, box_b])
    combined.merge_vertices()
    return combined


def nonmanifold_fin() -> trimesh.Trimesh:
    """A closed sphere with one extra triangle hanging off an existing
    edge, so that edge is used by three faces. trimesh's face adjacency
    only links faces across an edge shared by exactly two of them, so
    rung 2's component split separates the fin as its own tiny body and
    drops it -- no hole filling or re-meshing involved."""
    sphere = _sphere()
    a, b = sphere.faces[0][:2]
    midpoint = (sphere.vertices[a] + sphere.vertices[b]) / 2.0
    apex = midpoint + (midpoint - sphere.vertices.mean(axis=0)) * 0.3
    vertices = np.vstack([sphere.vertices, apex])
    faces = np.vstack([sphere.faces, [[a, b, len(vertices) - 1]]])
    return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)


FIXTURES = {
    "unmerged_vertices": unmerged_vertices,
    "inverted_winding": inverted_winding,
    "duplicate_and_degenerate_faces": duplicate_and_degenerate_faces,
    "floating_debris": floating_debris,
    "small_holes": small_holes,
    "open_bottom": open_bottom,
    "nonmanifold_edge": nonmanifold_edge,
    "nonmanifold_fin": nonmanifold_fin,
}


def generate() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    for name, builder in FIXTURES.items():
        mesh = builder()
        (OUTPUT_DIR / f"{name}.obj").write_text(
            trimesh.exchange.obj.export_obj(mesh, include_normals=False, include_color=False)
        )
        print(
            f"{name}.obj: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces, "
            f"watertight={mesh.is_watertight}"
        )


if __name__ == "__main__":
    generate()
