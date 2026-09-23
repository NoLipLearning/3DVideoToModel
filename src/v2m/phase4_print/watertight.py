"""The 6-rung watertight repair ladder.

docs/ARCHITECTURE.md Section 4 (M5) / Section 3.3: applied in order,
stopping at the first rung that yields a watertight solid. Rungs 1-4 are
progressively more aggressive general-purpose cleanup; by the time a
mesh reaches Phase 4 it almost always still has at least one large,
*intentional* opening (Section 3.3: "a Poisson mesh... has holes where
the camera never looked... has no bottom") that no amount of generic
repair closes, which is why rung 5 exists as a deliberate, mode-specific
architectural decision (ground.py's canonical frame) rather than a
repair heuristic, and rung 6 exists as an always-succeeds last resort.
"""

from __future__ import annotations

import logging

import networkx as nx
import numpy as np
import trimesh
from scipy.ndimage import binary_fill_holes
from scipy.spatial import ConvexHull
from shapely.geometry import Polygon
from trimesh.repair import faces_to_edges, group_rows, hashable_rows, triangulate_quads

from v2m.config import MeshConfig, PrintPrepConfig

logger = logging.getLogger("v2m.phase4_print.watertight")

_VOXEL_REMESH_RESOLUTION = 256  # resolution**3 grid (~16M cells as bools)


def rung1_basic_repair(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Merge vertices, drop degenerate/duplicate faces, fix winding and
    normals -- all of which `Trimesh.process(validate=True)` does in one
    call (confirmed directly against the installed trimesh==5.1.0: its
    docstring's steps 3-5 are exactly this rung)."""
    mesh.process(validate=True)
    return mesh


def rung2_drop_small_components(mesh: trimesh.Trimesh, config: MeshConfig) -> trimesh.Trimesh:
    """Keep the largest connected component, plus any other whose
    bounding-box volume is at least `min_component_volume_ratio` of it.

    Reuses `MeshConfig.min_component_volume_ratio` rather than a new
    Phase-4-specific field -- same operation, same meaning, as Phase 3's
    `cleanup.remove_small_components` (which operates on open3d meshes;
    this is trimesh's native equivalent, needed since Phase 4 works in
    trimesh throughout). Bounding-box volume, not `Trimesh.volume`: the
    mesh isn't watertight yet at this rung, so per-component `.volume`
    isn't meaningful.
    """
    components = mesh.split(only_watertight=False)
    if len(components) <= 1:
        return mesh

    bbox_volumes = np.array([np.prod(component.extents) for component in components])
    largest = bbox_volumes.max()
    keep = [
        component
        for component, volume in zip(components, bbox_volumes, strict=True)
        if volume >= config.min_component_volume_ratio * largest
    ]
    return trimesh.util.concatenate(keep)


def rung3_fill_small_holes(mesh: trimesh.Trimesh, config: PrintPrepConfig) -> trimesh.Trimesh:
    """Fill only the boundary loops small enough that fan triangulation
    is trustworthy (Section 3.3: fan-filling "may result in bad answers
    if the holes are non convex" -- exactly the risk with a large hole,
    which is also always the *intentional* floor opening rung 5 exists
    to close properly, not paper over with a fan).

    Reimplements `trimesh.repair.fill_holes`'s own algorithm (confirmed
    directly against trimesh==5.1.0's source) with an added size filter,
    since trimesh's built-in version has no such option -- it fills
    every boundary loop unconditionally.
    """
    if mesh.is_watertight:
        return mesh

    boundary_groups = group_rows(mesh.edges_sorted, require_count=1)
    if len(boundary_groups) < 3:
        return mesh

    boundary = mesh.edges[boundary_groups]
    holes = nx.cycle_basis(nx.from_edgelist(boundary))
    # An N-vertex boundary loop fan-triangulates into N-2 triangles.
    small_holes = [hole for hole in holes if (len(hole) - 2) <= config.max_hole_size_triangles]
    if not small_holes:
        return mesh

    new_faces = triangulate_quads(small_holes, use_fan=True)
    if len(new_faces) == 0:
        return mesh

    new_edges = faces_to_edges(new_faces)
    hashable_new = hashable_rows(new_edges)
    hashable_old = hashable_rows(boundary)
    needs_reverse = np.isin(hashable_new, hashable_old).reshape((-1, 3)).any(axis=1)
    new_faces[needs_reverse] = np.fliplr(new_faces[needs_reverse])

    mesh.extend_faces(new_faces)
    return mesh


def rung4_remove_nonmanifold(mesh: trimesh.Trimesh) -> tuple[trimesh.Trimesh, bool]:
    """Best-effort non-manifold edge repair via pymeshlab. Returns
    `(mesh, ran)`; `ran=False` (a no-op) when pymeshlab is absent --
    CLAUDE.md: "pymeshlab is optional... always provide an open3d/trimesh
    fallback path", same posture as every other optional-dependency call
    site in this codebase.

    UNVERIFIED in this sandbox: pymeshlab is confirmed absent here (this
    project's own dev environment), so unlike this codebase's other
    library integrations, `repair_non_manifold_edges` could not be
    checked directly against `help(pymeshlab.MeshSet)` on an actual
    installation (CLAUDE.md's own established practice for pycolmap).
    Verify it before trusting this blindly. By design this rung is
    non-load-bearing: rung 5 re-meshes from a voxel occupancy grid, which
    discards any non-manifold edge along with everything else, so skipping
    this rung costs surface detail (the mesh reaches rung 5 instead of
    stopping here), never correctness.
    """
    try:
        import pymeshlab
    except ImportError:
        return mesh, False

    mesh_set = pymeshlab.MeshSet()
    mesh_set.add_mesh(pymeshlab.Mesh(vertex_matrix=mesh.vertices, face_matrix=mesh.faces))
    try:
        mesh_set.repair_non_manifold_edges()
    except AttributeError:
        logger.warning(
            "pymeshlab is installed but repair_non_manifold_edges() was not found "
            "(API may have changed) -- skipping rung 4."
        )
        return mesh, False

    repaired = mesh_set.current_mesh()
    return (
        trimesh.Trimesh(
            vertices=repaired.vertex_matrix(), faces=repaired.face_matrix(), process=False
        ),
        True,
    )


def _solid_voxels_above(mesh: trimesh.Trimesh, cut_z: float) -> trimesh.voxel.VoxelGrid:
    """Solid occupancy grid of `mesh` (already ground-oriented: +Z up),
    keeping only voxels centered at or above `cut_z`.

    The interior is found *after* sealing the grid at the cut, not before:
    a layer of fully occupied voxels is laid across the whole grid just
    below `cut_z`, then everything enclosed by the surface plus that
    floor is flood-filled (`scipy.ndimage.binary_fill_holes` -- any empty
    region not connected to the padded grid's border). Filling first
    (`VoxelGrid.fill()`) was tried and fails on exactly the case this
    rung exists for: confirmed directly on
    tests/fixtures/broken_meshes/open_bottom.obj, the interior of a
    surface with an open underside leaks straight out through the
    opening, so nothing gets filled and the cut leaves a hollow
    one-voxel shell (volume 0.084 against ~3.0 expected).
    """
    pitch = float(mesh.extents.max()) / _VOXEL_REMESH_RESOLUTION
    surface = mesh.voxelized(pitch=pitch)

    # Pad one empty voxel on every side so "outside" is always connected
    # to the array border, as binary_fill_holes requires.
    matrix = np.pad(surface.matrix, 1)
    transform = surface.transform @ trimesh.transformations.translation_matrix([-1, -1, -1])
    layer_z = transform[2, 2] * np.arange(matrix.shape[2]) + transform[2, 3]

    below = layer_z < cut_z
    if below.any():
        floor_layer = int(np.flatnonzero(below).max())
        matrix[:, :, floor_layer] = True
    matrix = binary_fill_holes(matrix)
    matrix[:, :, below] = False

    return trimesh.voxel.VoxelGrid(
        trimesh.voxel.encoding.DenseEncoding(matrix), transform=transform
    )


def rung5_ground_cap(
    mesh: trimesh.Trimesh, mode: str, config: PrintPrepConfig, ground_tolerance: float = 0.0
) -> trimesh.Trimesh:
    """Close the ground-facing opening at z=0, assuming `mesh` is already
    in ground.py's canonical frame (ground at z=0, +Z up).

    Section 4 M5's ladder table describes this as a half-space boolean
    via manifold3d, "capped automatically" -- but manifold3d's `Manifold`
    constructor (what `trimesh.boolean.*(engine="manifold")` builds from
    each operand) requires an already-*closed* 2-manifold, confirmed
    directly: it errors on open boundary edges rather than auto-capping
    them. Forcing the mesh closed first via `trimesh.repair.fill_holes`
    doesn't reliably substitute -- confirmed directly too: fan
    triangulation of one large, non-convex "floor" boundary loop (the
    normal case reaching this rung) produced wildly wrong geometry
    reaching far outside the mesh's own bounds, and the non-fan mode
    silently leaves large holes unfilled rather than closing them badly.

    So instead this voxelizes `mesh`, seals and fills it at the cut (see
    `_solid_voxels_above`), and clears every voxel below the cut directly
    on the occupancy grid -- a boolean at heart, just array-based rather
    than routed through manifold3d, and
    exactly as robust as rung 6 (which this shares its voxel/marching-
    cubes machinery with) rather than the least robust link in the
    ladder. OBJECT mode cuts at `ground_tolerance` above the plane
    rather than exactly at it (Section 3.3: "discard points below
    z = epsilon"): a reconstructed table surface around the object sits
    within that band and would otherwise survive as a thin base plate
    under it. SCENE mode cuts at exactly z=0 -- there the ground *is*
    the subject -- and additionally
    unions in a slab prism (still via manifold3d, safely this time since
    both operands -- the now ground-cut, watertight mesh and the
    extruded prism -- are genuinely closed already); if that union still
    fails for some other reason, the cut mesh alone (no slab, but
    already watertight) is returned rather than raising.

    `slab_thickness_mm`/`scale_target_size_mm` are real-world-unit config
    values, but metric scale isn't established until *after* the repair
    ladder (scale.py runs later in Phase 4) -- same scale-ambiguity issue
    as monodepth_tsdf.py's `_build_tsdf_volume` at M3, and the same fix:
    only their *ratio* is trusted, applied to this mesh's own longest
    extent in whatever units it's currently in.

    `_VOXEL_REMESH_RESOLUTION` sets the accuracy/runtime trade. Measured
    on this project's 200mm cube fixture: at 128, the result's oriented
    bounding box came out 2.2-2.7% oversized (marching cubes on a filled
    grid sits about half a voxel outside the true surface); at 256,
    1.2-2.1%, with cube-root-of-volume within 0.4% -- in ~45s end to end
    on a 300k-face input.
    """
    cut_voxels = _solid_voxels_above(mesh, ground_tolerance if mode == "object" else 0.0)
    cut_mesh = cut_voxels.marching_cubes
    cut_mesh.apply_transform(cut_voxels.transform)

    if mode == "object" or not cut_mesh.is_watertight:
        return cut_mesh

    longest_extent = float(mesh.extents.max())
    thickness = (config.slab_thickness_mm / config.scale_target_size_mm) * longest_extent
    xy = mesh.vertices[:, :2]
    hull_2d = ConvexHull(xy)
    footprint = Polygon(xy[hull_2d.vertices])
    slab = trimesh.creation.extrude_polygon(
        footprint,
        height=thickness,
        transform=trimesh.transformations.translation_matrix([0.0, 0.0, -thickness]),
    )
    try:
        return trimesh.boolean.union([cut_mesh, slab], engine="manifold")
    except ValueError:
        return cut_mesh


def rung6_voxel_remesh(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Last resort: voxelize and re-extract via marching cubes. Always
    produces a watertight, manifold mesh by construction (a filled voxel
    grid has no boundary), at the cost of surface detail -- callers must
    log this as a quality warning (Section 4 M5's ladder table).

    `VoxelGrid.marching_cubes` returns the result in raw voxel-index
    coordinates, not world space (confirmed directly: it's
    `ops.matrix_to_marching_cubes(matrix=self.matrix)`, nothing else --
    no transform applied). Skipping `apply_transform` here would silently
    return a mesh at entirely the wrong position and scale.
    """
    pitch = float(mesh.extents.max()) / _VOXEL_REMESH_RESOLUTION
    voxels = mesh.voxelized(pitch=pitch).fill()
    result = voxels.marching_cubes
    result.apply_transform(voxels.transform)
    return result


def make_watertight(
    mesh: trimesh.Trimesh,
    mode: str,
    mesh_config: MeshConfig,
    print_config: PrintPrepConfig,
    ground_tolerance: float = 0.0,
) -> tuple[trimesh.Trimesh, int, list[str]]:
    """Runs the ladder, stopping at the first rung whose result is
    watertight. `mesh` must already be in ground.py's canonical frame
    (rung 5 cuts at the ground plane; `ground_tolerance` is ground.py's
    own plane-fit tolerance, in the mesh's units). Returns `(mesh,
    rung_used, warnings)`."""
    warnings: list[str] = []

    mesh = rung1_basic_repair(mesh)
    if mesh.is_watertight:
        return mesh, 1, warnings

    mesh = rung2_drop_small_components(mesh, mesh_config)
    if mesh.is_watertight:
        return mesh, 2, warnings

    mesh = rung3_fill_small_holes(mesh, print_config)
    if mesh.is_watertight:
        return mesh, 3, warnings

    mesh, rung4_ran = rung4_remove_nonmanifold(mesh)
    if not rung4_ran:
        warnings.append("pymeshlab absent -- skipped rung 4 (non-manifold edge repair)")
    if mesh.is_watertight:
        return mesh, 4, warnings

    mesh = rung5_ground_cap(mesh, mode, print_config, ground_tolerance)
    if mesh.is_watertight:
        warnings.append(
            f"closed via rung 5 (voxel ground cut at resolution {_VOXEL_REMESH_RESOLUTION}) -- "
            "fine surface detail below the voxel size was smoothed out"
        )
        return mesh, 5, warnings

    warnings.append(
        "fell back to rung 6 (voxel remesh) to reach a watertight solid -- "
        "surface detail was reduced"
    )
    mesh = rung6_voxel_remesh(mesh)
    return mesh, 6, warnings
