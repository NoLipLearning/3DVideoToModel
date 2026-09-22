"""sparse_only.py: the always-works dense-stage fallback.

No COLMAP/SfM involved -- just a hand-built sparse.ply -- so this stays
fast and unmarked, unlike the monodepth_tsdf tests.
"""

import numpy as np
import open3d as o3d
import pytest

from v2m.config import DenseConfig
from v2m.errors import SfMError
from v2m.phase2_sfm.dense import sparse_only


def _write_point_cloud(path, num_points: int) -> np.ndarray:
    rng = np.random.default_rng(0)
    points = rng.uniform(-1.0, 1.0, size=(num_points, 3))
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    o3d.io.write_point_cloud(str(path), cloud)
    return points


def test_densify_copies_sparse_cloud_and_reports_point_count(tmp_path):
    sfm_dir = tmp_path / "sfm"
    sfm_dir.mkdir()
    _write_point_cloud(sfm_dir / "sparse.ply", num_points=17)

    output_dir = tmp_path / "dense"
    result = sparse_only.densify(sfm_dir, output_dir, DenseConfig())

    assert result.backend == "sparse_only"
    assert result.num_points == 17
    assert result.dense_points_path == "dense.ply"

    dense_ply = output_dir / "dense.ply"
    assert dense_ply.exists()
    dense_cloud = o3d.io.read_point_cloud(str(dense_ply))
    assert len(dense_cloud.points) == 17


def test_densify_raises_when_sparse_ply_is_missing(tmp_path):
    sfm_dir = tmp_path / "sfm"
    sfm_dir.mkdir()  # no sparse.ply written

    with pytest.raises(SfMError) as exc_info:
        sparse_only.densify(sfm_dir, tmp_path / "dense", DenseConfig())

    assert "v2m sfm" in exc_info.value.remedy
