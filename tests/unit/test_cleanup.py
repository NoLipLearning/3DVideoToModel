"""Component/floater removal and face-budget decimation, against
hand-built box meshes -- fast and unmarked, same posture as
test_normals.py/test_poisson.py.
"""

import open3d as o3d

from v2m.config import MeshConfig
from v2m.phase3_mesh import cleanup


def _box_at(size: float, translation) -> o3d.geometry.TriangleMesh:
    box = o3d.geometry.TriangleMesh.create_box(width=size, height=size, depth=size)
    box.translate(translation)
    return box


def test_remove_small_components_keeps_a_single_component_untouched():
    mesh = _box_at(1.0, [0.0, 0.0, 0.0])
    original_triangle_count = len(mesh.triangles)

    cleaned = cleanup.remove_small_components(mesh, MeshConfig())

    assert len(cleaned.triangles) == original_triangle_count


def test_remove_small_components_drops_a_tiny_floater():
    big_box = _box_at(10.0, [0.0, 0.0, 0.0])
    tiny_floater = _box_at(0.1, [100.0, 100.0, 100.0])  # bbox volume ratio: 0.1^3/10^3 = 1e-4
    combined = big_box + tiny_floater

    cleaned = cleanup.remove_small_components(combined, MeshConfig(min_component_volume_ratio=0.01))

    clusters, n_triangles, _area = cleaned.cluster_connected_triangles()
    assert len(n_triangles) == 1
    assert len(cleaned.triangles) == len(big_box.triangles)


def test_remove_small_components_keeps_comparably_sized_components():
    box_a = _box_at(1.0, [0.0, 0.0, 0.0])
    box_b = _box_at(1.0, [100.0, 0.0, 0.0])  # same size, far away -- e.g. a two-piece subject
    combined = box_a + box_b

    cleaned = cleanup.remove_small_components(combined, MeshConfig(min_component_volume_ratio=0.01))

    assert len(cleaned.triangles) == len(combined.triangles)


def test_decimate_to_face_budget_is_a_noop_under_budget():
    mesh = _box_at(1.0, [0.0, 0.0, 0.0])
    original_triangle_count = len(mesh.triangles)

    result = cleanup.decimate_to_face_budget(
        mesh, MeshConfig(max_faces=original_triangle_count + 100)
    )

    assert len(result.triangles) == original_triangle_count


def test_decimate_to_face_budget_reduces_triangle_count_over_budget():
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=1.0, resolution=40)
    assert len(sphere.triangles) > 500

    result = cleanup.decimate_to_face_budget(sphere, MeshConfig(max_faces=500))

    assert len(result.triangles) <= 500


def test_clean_mesh_runs_both_steps_in_order():
    big_box = _box_at(10.0, [0.0, 0.0, 0.0])
    tiny_floater = _box_at(0.1, [100.0, 100.0, 100.0])
    combined = big_box + tiny_floater

    result = cleanup.clean_mesh(combined, MeshConfig(min_component_volume_ratio=0.01, max_faces=6))

    clusters, n_triangles, _area = result.cluster_connected_triangles()
    assert len(n_triangles) == 1  # floater dropped
    assert len(result.triangles) <= 6  # then decimated
