"""Largest-component / floater removal + face-budget decimation.

docs/ARCHITECTURE.md Section 4 (M4): "largest-component / floater
removal, decimate to <=300k faces." Watertightness isn't established
until Phase 4 (Section 3.3), so a raw Poisson component here can't be
assumed manifold enough to have a real enclosed volume -- despite
`MeshConfig.min_component_volume_ratio`'s name, "volume" below means
each component's axis-aligned bounding-box volume, the cheapest measure
that's always well-defined regardless of watertightness and still
distinguishes real geometry from small noise-driven bubbles.
"""

from __future__ import annotations

import numpy as np
import open3d as o3d

from v2m.config import MeshConfig


def remove_small_components(
    mesh: o3d.geometry.TriangleMesh, config: MeshConfig
) -> o3d.geometry.TriangleMesh:
    """Keep the largest connected component, plus any other component
    whose bounding-box volume is at least `min_component_volume_ratio`
    of it -- drops small floating debris (isolated bubbles Poisson
    sometimes creates in noisy regions) without being so strict that a
    legitimate disjoint part of a real subject (e.g. a two-piece object)
    vanishes."""
    triangle_clusters, cluster_n_triangles, _cluster_area = mesh.cluster_connected_triangles()
    if len(cluster_n_triangles) <= 1:
        return mesh

    triangle_clusters = np.asarray(triangle_clusters)
    triangles = np.asarray(mesh.triangles)
    vertices = np.asarray(mesh.vertices)

    cluster_volumes = np.zeros(len(cluster_n_triangles))
    for cluster_id in range(len(cluster_n_triangles)):
        vertex_ids = np.unique(triangles[triangle_clusters == cluster_id])
        extent = vertices[vertex_ids].max(axis=0) - vertices[vertex_ids].min(axis=0)
        cluster_volumes[cluster_id] = np.prod(np.maximum(extent, 1e-9))

    largest_volume = cluster_volumes.max()
    keep_clusters = np.flatnonzero(
        cluster_volumes >= config.min_component_volume_ratio * largest_volume
    )
    remove_mask = ~np.isin(triangle_clusters, keep_clusters)

    mesh.remove_triangles_by_mask(remove_mask)
    mesh.remove_unreferenced_vertices()
    return mesh


def decimate_to_face_budget(
    mesh: o3d.geometry.TriangleMesh, config: MeshConfig
) -> o3d.geometry.TriangleMesh:
    """Quadric-decimate down to `max_faces`, leaving the mesh untouched
    if it's already within budget (decimation only ever removes
    detail -- never worth paying for when it wouldn't do anything)."""
    if len(mesh.triangles) <= config.max_faces:
        return mesh
    return mesh.simplify_quadric_decimation(target_number_of_triangles=config.max_faces)


def clean_mesh(mesh: o3d.geometry.TriangleMesh, config: MeshConfig) -> o3d.geometry.TriangleMesh:
    """Both cleanup steps, in the order the architecture doc lists them:
    drop floaters first so decimation's face budget is spent entirely on
    real geometry."""
    mesh = remove_small_components(mesh, config)
    mesh = decimate_to_face_budget(mesh, config)
    return mesh
