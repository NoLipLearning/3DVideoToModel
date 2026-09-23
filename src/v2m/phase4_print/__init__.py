"""Phase 4: surface -> watertight, scaled, printable solid.

This is the milestone that determines whether the whole project succeeds
-- see docs/ARCHITECTURE.md Section 3.3 and the M5 entry in Section 4.
Implemented in M5.

`run_print_prep()` is this package's orchestrator, tying together (in
the doc's own order): ground.py's plane detection + canonical
re-orientation, watertight.py's 6-rung repair ladder, scale.py's
metric-scale resolution, validate.py's printability checks, and
export.py's STL/OBJ/GLB output.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
import pycolmap
import trimesh

from v2m.config import MeshConfig, PrintPrepConfig
from v2m.errors import PrintPrepError
from v2m.phase4_print import export, ground, scale, validate, watertight
from v2m.types import PrintReport


def _camera_matrix_and_center_for_image(
    undistorted_sparse_dir: Path, image_name: str
) -> tuple[np.ndarray, np.ndarray]:
    """A 3x3 intrinsics matrix and world-space camera center for one
    image, for `scale.py`'s ArUco `solvePnP` call.

    Reads the *undistorted* reconstruction, not `cameras.json` (which
    `sfm.py` exports from the pre-undistortion one): `solvePnP` is being
    run against the undistorted image on disk (`sfm_dir/undistorted/
    images/`), so it needs that image's own camera model, not the
    original distorted one -- the same original-vs-undistorted mismatch
    `monodepth_tsdf.py`'s `densify()` had (see its own docstring for the
    full story). Camera *centers* happen to be identical either way
    (confirmed directly: undistortion changes each image's 2D keypoints
    and camera intrinsics, never poses or 3D points) but reading both
    from the same reconstruction here keeps this function's own
    correctness independent of that fact.
    """
    reconstruction = pycolmap.Reconstruction(undistorted_sparse_dir)
    image = next(img for img in reconstruction.images.values() if img.name == image_name)
    camera = image.camera
    camera_matrix = np.array(
        [
            [camera.focal_length_x, 0.0, camera.principal_point_x],
            [0.0, camera.focal_length_y, camera.principal_point_y],
            [0.0, 0.0, 1.0],
        ]
    )
    pose = image.cam_from_world()
    camera_center = -pose.rotation.matrix().T @ pose.translation
    return camera_matrix, camera_center


def run_print_prep(
    mesh_dir: Path,
    dense_dir: Path,
    sfm_dir: Path,
    output_dir: Path,
    mesh_config: MeshConfig,
    print_config: PrintPrepConfig,
    mode: str,
    *,
    scale_factor: float | None = None,
    scale_two_point: tuple[np.ndarray, np.ndarray, float] | None = None,
    aruco_image_name: str | None = None,
    aruco_marker_mm: float | None = None,
) -> PrintReport:
    """Writes `<output_dir>/{model.stl, model.obj, model.glb,
    print_report.json}`.

    Scale methods are applied lowest-to-highest priority so the last
    write wins, landing on Section 3.4's stated order (ArUco > two-point
    > manual > fit-to-build-volume) applied highest-first: each later
    block only overwrites `factor` when its own input was actually
    supplied (and, for ArUco, only when a marker was actually detected),
    so an unsupplied or failed higher-priority method correctly falls
    through to whatever a lower-priority one already set.
    """
    cleaned_ply = mesh_dir / "cleaned.ply"
    dense_ply = dense_dir / "dense.ply"
    cameras_json = sfm_dir / "cameras.json"
    for path, label in (
        (cleaned_ply, "mesh/cleaned.ply"),
        (dense_ply, "dense/dense.ply"),
        (cameras_json, "sfm/cameras.json"),
    ):
        if not path.exists():
            raise PrintPrepError(
                f"Missing required input: {label}.",
                remedy="Run `v2m extract`, `sfm`, `dense`, and `mesh` first.",
            )

    mesh = trimesh.load(str(cleaned_ply), process=False, force="mesh")
    cloud = o3d.io.read_point_cloud(str(dense_ply))

    gravity_up = ground.compute_gravity_up(cameras_json)
    ground_plane = ground.find_ground_plane(cloud, gravity_up, print_config, mode)
    mesh.apply_transform(ground.canonical_transform(ground_plane))

    if ground_plane.source != "ransac":
        warnings_pre = [
            f"No trustworthy ground plane was found ({ground_plane.source}) -- the base was "
            "placed from the camera gravity prior instead; check the model's orientation."
        ]
    else:
        warnings_pre = []

    mesh, rung_used, warnings = watertight.make_watertight(
        mesh, mode, mesh_config, print_config, ground_plane.tolerance
    )
    warnings = warnings_pre + warnings
    ground.place_on_bed(mesh)

    scale_method = "fit_to_build_volume"
    factor = scale.resolve_scale_fit_to_build_volume(mesh, print_config.scale_target_size_mm)

    if scale_factor is not None:
        factor = scale.resolve_scale_manual(scale_factor)
        scale_method = "manual"

    if scale_two_point is not None:
        point_a, point_b, real_distance_mm = scale_two_point
        factor = scale.resolve_scale_two_point(point_a, point_b, real_distance_mm)
        scale_method = "two_point"

    if aruco_marker_mm is not None and aruco_image_name is None:
        aruco_image_name = scale.find_aruco_image(sfm_dir / "undistorted" / "images")
        if aruco_image_name is None:
            warnings.append(
                "ArUco: --aruco-marker-mm was given but no DICT_4X4_50 marker was found in any "
                "frame; skipping."
            )

    if aruco_image_name is not None and aruco_marker_mm is not None:
        image_path = sfm_dir / "undistorted" / "images" / aruco_image_name
        image_bgr = cv2.imread(str(image_path))
        if image_bgr is None:
            warnings.append(f"ArUco: could not read {image_path}; skipping.")
        else:
            camera_matrix, camera_center = _camera_matrix_and_center_for_image(
                sfm_dir / "undistorted" / "sparse", aruco_image_name
            )
            # The marker is expected near the subject (Section 3.4: "lay
            # it in the scene"), so the scene centroid's own distance
            # from this camera stands in for "the marker's distance", in
            # COLMAP's arbitrary units -- avoiding a full per-pixel
            # 2D-3D correspondence lookup for one approximate reference.
            scene_centroid = np.asarray(cloud.points).mean(axis=0)
            colmap_distance = float(np.linalg.norm(camera_center - scene_centroid))
            aruco_factor = scale.resolve_scale_aruco(
                image_bgr, camera_matrix, aruco_marker_mm, colmap_distance
            )
            if aruco_factor is not None:
                factor = aruco_factor
                scale_method = "aruco"
            else:
                warnings.append(f"ArUco: no marker detected in {aruco_image_name}; skipping.")

    mesh.apply_scale(factor)
    if scale_method == "fit_to_build_volume":
        warnings.append(
            "No metric scale reference was available -- dimensions are normalized to "
            "a target size and are NOT physically meaningful."
        )

    warnings.extend(validate.validate_mesh(mesh, print_config))

    export.export_mesh(mesh, output_dir)

    report = PrintReport(
        watertight=mesh.is_watertight,
        volume_mm3=float(mesh.volume),
        bbox_mm=tuple(mesh.extents.tolist()),
        repair_rung_used=rung_used,
        scale_method=scale_method,
        warnings=warnings,
    )
    (output_dir / "print_report.json").write_text(report.model_dump_json(indent=2))
    return report
