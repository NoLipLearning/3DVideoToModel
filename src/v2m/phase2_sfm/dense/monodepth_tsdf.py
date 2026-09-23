"""Default dense backend on Apple Silicon: monocular depth + TSDF fusion.

docs/ARCHITECTURE.md Section 2 ("Default dense strategy -- monodepth_tsdf"):
  1. COLMAP gives per-image pose + intrinsics + sparse 3D points.
  2. A monocular depth model gives *relative* inverse depth (disparity)
     per frame -- dense and smooth even on blank walls.
  3. Project that frame's visible sparse points into the image (here:
     use the 2D-3D correspondences COLMAP already established, which are
     exact by construction, not a re-derived approximation); fit a
     robust affine `a*disparity + b ~= 1/z_sfm` to lift disparity to
     metric inverse depth.
  4. Integrate the aligned depth maps into an
     `open3d.pipelines.integration.UniformTSDFVolume`, sized from the
     sparse reconstruction's own bounding box, using the known poses.
     (docs/ARCHITECTURE.md Section 2 specifies the auto-expanding
     `ScalableTSDFVolume` instead -- see `_build_tsdf_volume()` below for
     why this project switched to a `UniformTSDFVolume`.)
  5. Extract the point cloud.

The depth-inference step is behind the `DepthEstimator` protocol so it
can be swapped out in tests: this project's sandbox cannot download
Depth-Anything-V2's weights (huggingface.co is network-policy-blocked,
confirmed directly -- see CLAUDE.md), so integration tests inject a
geometric stand-in built from known fixture geometry
(`tests/fixtures/make_synthetic_video.py::render_true_depth`) instead of
`TransformersDepthEstimator`. `TransformersDepthEstimator` itself follows
the standard HF depth-estimation pattern but has not been run end-to-end
in this sandbox for the same reason -- verify it on a machine that can
reach huggingface.co before trusting it blindly.
"""

from __future__ import annotations

import json
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np
import open3d as o3d
import pycolmap

from v2m.config import DenseConfig
from v2m.errors import SfMError
from v2m.phase2_sfm.dense import depth_alignment, tiling, transients
from v2m.types import DenseResult

logger = logging.getLogger("v2m.phase2_sfm.dense.monodepth_tsdf")

# How far past the farthest *reliable* (inlier) sparse-point depth in an
# image to trust the aligned depth map before treating it as background/
# extrapolation noise. Not itself specified numerically in the
# architecture doc (only "Huber/RANSAC" for the fit itself) -- documented
# here rather than silently invented elsewhere.
_DEPTH_TRUNC_SAFETY_FACTOR = 3.0
_INLIER_THRESHOLD_FRACTION = 0.1  # of the correspondence set's inverse-depth spread


class DepthEstimator(Protocol):
    def predict_disparity(self, image_bgr: np.ndarray, image_name: str | None = None) -> np.ndarray:
        """Relative inverse depth (higher = closer): an HxW float32 array
        matching `image_bgr`'s height and width.

        `image_name` is an optional hint a real model ignores -- it
        exists so a test double can look up which known pose/geometry an
        input frame corresponds to (see
        tests/unit/test_monodepth_tsdf.py's GeometricDepthEstimator),
        without inventing content-based matching against COLMAP's
        undistorted (and therefore not byte-identical) image copies.
        """
        ...


class TransformersDepthEstimator:
    """Depth Anything V2 via `transformers`. Lazy-loads the model on
    first use, so constructing this object (or importing this module)
    never requires torch/transformers to actually download or load
    anything until a real prediction is requested.
    """

    def __init__(self, model_name: str, device: str | None = None) -> None:
        self._model_name = model_name
        self._requested_device = device
        self._model = None
        self._processor = None
        self._device = "cpu"

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation

        if self._requested_device:
            self._device = self._requested_device
        elif torch.backends.mps.is_available():
            self._device = "mps"
        elif torch.cuda.is_available():
            self._device = "cuda"
        else:
            self._device = "cpu"

        try:
            self._processor = AutoImageProcessor.from_pretrained(self._model_name)
            self._model = AutoModelForDepthEstimation.from_pretrained(self._model_name)
        except OSError as exc:
            # huggingface_hub raises a plain OSError for both "no network"
            # and "no local cache" -- either way this is a remedy-worthy
            # user-facing failure, not a bare traceback (CLAUDE.md: "Raise
            # V2MError subclasses with a remedy=, not bare exceptions").
            raise SfMError(
                f"Could not load depth model '{self._model_name}': {exc}",
                remedy="Check network access to huggingface.co (needed on first run to "
                "download weights; cached afterward), or switch to dense.backend: "
                "sparse_only for a lower-quality preview that needs no depth model.",
            ) from exc
        self._model.to(self._device)
        self._model.eval()

    def predict_disparity(self, image_bgr: np.ndarray, image_name: str | None = None) -> np.ndarray:
        del image_name  # unused: a real model only ever sees pixels
        self._ensure_loaded()
        import torch
        from PIL import Image as PILImage

        pil_image = PILImage.fromarray(image_bgr[:, :, ::-1])  # BGR -> RGB
        inputs = self._processor(images=pil_image, return_tensors="pt")
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._model(**inputs)

        predicted = outputs.predicted_depth  # (1, h', w') at the model's internal resolution
        resized = torch.nn.functional.interpolate(
            predicted.unsqueeze(1),
            size=image_bgr.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()
        return resized.detach().cpu().numpy().astype(np.float32)


def _gather_correspondences(
    image: pycolmap.Image,
    reconstruction: pycolmap.Reconstruction,
    rotation: np.ndarray,
    translation: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pixel coords and camera-space inverse depths for this image's
    2D-3D correspondences. These are COLMAP's own established
    correspondences -- exact by construction, not a re-projection
    approximation. `rotation`/`translation` are this image's
    `cam_from_world()`, passed in so the caller (which also needs them
    for TSDF integration) only computes it once.

    Returns (pixel_x, pixel_y, inverse_depths) as parallel float arrays.
    """
    pixel_x: list[float] = []
    pixel_y: list[float] = []
    inverse_depths: list[float] = []

    for point2d in image.points2D:
        if not point2d.has_point3D():
            continue
        point3d = reconstruction.points3D[point2d.point3D_id]
        z_cam = (rotation @ point3d.xyz + translation)[2]
        if z_cam <= 1e-6:
            continue
        x_val, y_val = point2d.xy
        pixel_x.append(float(x_val))
        pixel_y.append(float(y_val))
        inverse_depths.append(1.0 / z_cam)

    return np.array(pixel_x), np.array(pixel_y), np.array(inverse_depths)


def _sample_disparity(
    disparity: np.ndarray, pixel_x: np.ndarray, pixel_y: np.ndarray
) -> np.ndarray:
    """Nearest-neighbor sample of `disparity` at each (x, y) -- COLMAP's
    keypoint coordinates are sub-pixel, but the depth model's disparity
    map is already a fairly smooth field, so nearest-neighbor sampling is
    an acceptable, simple choice here (bilinear would be a marginal
    refinement, not a correctness fix)."""
    height, width = disparity.shape
    xi = np.clip(np.round(pixel_x).astype(int), 0, width - 1)
    yi = np.clip(np.round(pixel_y).astype(int), 0, height - 1)
    return disparity[yi, xi]


_MIN_TSDF_RESOLUTION = 32
# resolution**3 voxels are allocated upfront. Measured on open3d 0.20:
# 48 bytes/voxel RSS (a 200**3 volume costs 384MB), so 400**3 is ~3.1GB --
# preflight.py budgets for exactly this.
MAX_TSDF_RESOLUTION = 400
TSDF_BYTES_PER_VOXEL = 48
_BBOX_TRIM_PERCENTILE = 0.02  # trim the outer 2% of points on each side before sizing the volume
_BBOX_SAFETY_MARGIN = 1.3


def _build_tsdf_volume(
    reconstruction: pycolmap.Reconstruction, config: DenseConfig
) -> o3d.pipelines.integration.UniformTSDFVolume:
    """A `UniformTSDFVolume` sized from the reconstruction's own sparse
    point-cloud extent, at `config.tsdf_voxel_size_m` resolution.

    docs/ARCHITECTURE.md Section 2 specifies `ScalableTSDFVolume`
    (auto-expanding, no size to compute up front). This project's
    installed open3d==0.20.0 CPU build has a confirmed-broken
    `ScalableTSDFVolume.integrate()` -- verified three independent ways,
    including Open3D's own official RGBD-integration tutorial parameters
    verbatim: `integrate()` runs without error, but
    `extract_point_cloud()`, `extract_voxel_point_cloud()`, and
    `extract_triangle_mesh()` all consistently return empty results, on
    both this project's data and a trivial synthetic flat-plane case.
    `UniformTSDFVolume` with otherwise-identical inputs works correctly.
    This is very possibly a Linux-CPU-wheel-specific issue rather than a
    real bug in Open3D generally -- if you're on macOS (this project's
    actual target) and can confirm `ScalableTSDFVolume` works there,
    switching back gets the better behavior on a very large/unbounded
    "scene" capture that might not fit comfortably in a fixed-size
    volume; this function's call site is the only place that would need
    to change.

    `config.tsdf_voxel_size_m`/`tsdf_sdf_trunc_m` are named (and
    documented in docs/ARCHITECTURE.md Section 3.2) as literal metres,
    but at this point in the pipeline that unit doesn't exist yet:
    monocular SfM is scale-ambiguous, and COLMAP normalizes the first
    registered image pair's baseline to an arbitrary length, not a
    real-world one -- true metric scale isn't established until Phase 4's
    `scale.py` (Section 3.4), which rescales the *final mesh*, long after
    this volume is built and discarded. Treating `tsdf_sdf_trunc_m` as an
    absolute value here would size the truncation band wrong by whatever
    unknown factor COLMAP's unit differs from a real metre -- for a
    tightly-spaced capture that factor can be tiny, making the band
    thinner than a single voxel. So only the *ratio* between the two
    config values (default 0.02/0.004 = 5 voxels of truncation) is
    trusted as meaningful; it's applied to `voxel_size_actual`, the
    per-voxel size this reconstruction's own extent actually resolves to
    after the resolution clamp above -- whatever scale that turns out to
    be. `resolution` itself needs no equivalent correction: clamping it
    to [`_MIN_TSDF_RESOLUTION`, `MAX_TSDF_RESOLUTION`] already absorbs
    an arbitrary length/voxel_size_m ratio without reference to real
    units.
    """
    bbox = reconstruction.compute_bounding_box(_BBOX_TRIM_PERCENTILE, 1.0 - _BBOX_TRIM_PERCENTILE)
    extent = np.asarray(bbox.max) - np.asarray(bbox.min)
    center = (np.asarray(bbox.max) + np.asarray(bbox.min)) / 2.0

    if not np.all(np.isfinite(extent)) or np.any(extent < 0):
        # Degenerate/near-empty point cloud -- fall back to a modest
        # default volume rather than raising here; densify() already
        # requires a real reconstruction to reach this point, so this is
        # a last-resort safety net, not the expected path. There's no
        # reconstruction-derived scale to anchor to at all in this case,
        # so the config value is used as a literal (best-effort) length.
        extent = np.full(3, config.tsdf_voxel_size_m * _MIN_TSDF_RESOLUTION)
        center = np.zeros(3)

    half_diagonal = float(np.linalg.norm(extent)) / 2.0
    length = max(
        2.0 * half_diagonal * _BBOX_SAFETY_MARGIN, config.tsdf_voxel_size_m * _MIN_TSDF_RESOLUTION
    )
    resolution = int(
        np.clip(round(length / config.tsdf_voxel_size_m), _MIN_TSDF_RESOLUTION, MAX_TSDF_RESOLUTION)
    )
    origin = center - length / 2.0

    voxel_size_actual = length / resolution
    trunc_voxel_multiple = config.tsdf_sdf_trunc_m / config.tsdf_voxel_size_m
    sdf_trunc = trunc_voxel_multiple * voxel_size_actual

    return o3d.pipelines.integration.UniformTSDFVolume(
        length=length,
        resolution=resolution,
        sdf_trunc=sdf_trunc,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
        origin=origin,
    )


# Transient points must be seen through by at least this many voxels' worth
# of depth before a frame counts as a free-space violation -- the same
# order as the TSDF truncation band (5 voxels by default), a little under
# it so that ordinary depth noise on a real surface never votes against it.
_TRANSIENT_TOLERANCE_VOXELS = 3.0
_TILE_MARGIN_VOXELS = 8


@dataclass
class _AlignedFrame:
    """One image's metric depth map, saved to disk, plus what's needed to
    integrate it -- so tiles can re-read frames instead of holding them all."""

    image_path: Path
    depth_path: Path
    depth_trunc: float
    intrinsic: np.ndarray
    size: tuple[int, int]
    extrinsic: np.ndarray

    def load_depth(self) -> np.ndarray:
        depth = np.load(self.depth_path)
        return np.where(depth <= self.depth_trunc, depth, 0.0).astype(np.float32)

    def rgbd(self) -> o3d.geometry.RGBDImage:
        color_rgb = np.ascontiguousarray(cv2.imread(str(self.image_path))[:, :, ::-1])
        return o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(color_rgb),
            o3d.geometry.Image(self.load_depth()),
            depth_scale=1.0,
            depth_trunc=self.depth_trunc,
            convert_rgb_to_intensity=False,
        )

    def o3d_intrinsic(self) -> o3d.camera.PinholeCameraIntrinsic:
        k = self.intrinsic
        return o3d.camera.PinholeCameraIntrinsic(
            self.size[0], self.size[1], k[0, 0], k[1, 1], k[0, 2], k[1, 2]
        )


def _integrate(
    reconstruction: pycolmap.Reconstruction,
    frames: list[_AlignedFrame],
    config: DenseConfig,
    median_depth: float,
) -> tuple[o3d.geometry.PointCloud, int, float]:
    """TSDF-integrate `frames`; returns (points, tiles used, voxel size).

    One volume when the scene fits at the detail this capture supports
    (always, for an object) -- built by `_build_tsdf_volume` exactly as
    before tiling existed. Otherwise a grid of tiles (tiling.py), each
    integrated from only the frames that see it and cropped to its core.
    """
    bbox = reconstruction.compute_bounding_box(_BBOX_TRIM_PERCENTILE, 1.0 - _BBOX_TRIM_PERCENTILE)
    bbox_min, bbox_max = np.asarray(bbox.min), np.asarray(bbox.max)
    voxel = tiling.desired_voxel(median_depth, config)
    tiles = []
    if np.all(np.isfinite(bbox_max - bbox_min)) and np.all(bbox_max > bbox_min):
        center = (bbox_min + bbox_max) / 2
        half = (bbox_max - bbox_min) / 2 * _BBOX_SAFETY_MARGIN
        tiles = tiling.plan_tiles(
            center - half, center + half, voxel, config, MAX_TSDF_RESOLUTION, _TILE_MARGIN_VOXELS
        )

    if len(tiles) <= 1:
        volume = _build_tsdf_volume(reconstruction, config)
        for frame in frames:
            volume.integrate(frame.rgbd(), frame.o3d_intrinsic(), frame.extrinsic)
        return volume.extract_point_cloud(), 1, voxel

    trunc_ratio = config.tsdf_sdf_trunc_m / config.tsdf_voxel_size_m
    footprints = [
        tiling.sample_world_points(f.load_depth(), f.intrinsic, f.extrinsic) for f in frames
    ]
    merged = o3d.geometry.PointCloud()
    for tile in tiles:
        members = [f for f, pts in zip(frames, footprints, strict=True) if tile.touches(pts)]
        if not members:
            continue
        volume = o3d.pipelines.integration.UniformTSDFVolume(
            length=tile.length,
            resolution=tile.resolution,
            sdf_trunc=trunc_ratio * tile.length / tile.resolution,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
            origin=tile.origin,
        )
        for frame in members:
            volume.integrate(frame.rgbd(), frame.o3d_intrinsic(), frame.extrinsic)
        part = volume.extract_point_cloud()
        keep = tile.contains(np.asarray(part.points))
        merged += part.select_by_index(np.flatnonzero(keep))
        del volume
    logger.info(
        "Scene tiled into %d TSDF volumes (voxel %.4g units, median depth %.4g).",
        len(tiles),
        voxel,
        median_depth,
    )
    return merged, len(tiles), voxel


def densify(
    sfm_dir: Path,
    output_dir: Path,
    config: DenseConfig,
    *,
    depth_estimator: DepthEstimator | None = None,
) -> DenseResult:
    """Run Phase 2b end to end: sparse reconstruction -> dense point cloud.

    Writes `<output_dir>/{depth/*.npy, dense.ply, alignment_report.json}`.
    `depth_estimator` defaults to the real `TransformersDepthEstimator`;
    tests inject a synthetic stand-in (see module docstring).
    """
    # The *undistorted* reconstruction, not sparse/final/: pycolmap's
    # undistort_images() writes a second copy of the reconstruction here
    # with each image's 2D keypoints reprojected into the undistorted
    # pixel space and camera models updated to match (confirmed
    # directly -- 3D points and poses are untouched, only per-image
    # intrinsics/2D coordinates change). sparse/final/'s keypoint pixel
    # coordinates are in the *original* (distorted) image space, which
    # only happens to line up with the undistorted images on disk when a
    # camera's estimated distortion is negligible. This project's own
    # M1/M2 fixture initially had exactly that -- near-zero estimated
    # distortion for every camera on a shallow orbit -- which is why
    # this went uncaught until M5's fixture rework gave the poles enough
    # elevation to make COLMAP's per-camera distortion estimate (and
    # hence the undistorted canvas size/keypoint positions) actually
    # diverge from the original.
    undistorted_sparse_dir = sfm_dir / "undistorted" / "sparse"
    images_dir = sfm_dir / "undistorted" / "images"
    if not undistorted_sparse_dir.exists() or not images_dir.exists():
        raise SfMError(
            f"No completed sparse reconstruction found in {sfm_dir}.",
            remedy="Run `v2m sfm` first to produce undistorted/sparse/ and undistorted/images/.",
        )

    reconstruction = pycolmap.Reconstruction(undistorted_sparse_dir)
    if depth_estimator is None:
        depth_estimator = TransformersDepthEstimator(config.depth_model)

    output_dir.mkdir(parents=True, exist_ok=True)
    depth_dir = output_dir / "depth"
    if depth_dir.exists():
        shutil.rmtree(depth_dir)
    depth_dir.mkdir(parents=True)

    per_image_reports: list[dict] = []
    frames: list[_AlignedFrame] = []
    all_rmses: list[float] = []
    median_depths: list[float] = []

    for image in reconstruction.images.values():
        image_path = images_dir / image.name
        color_bgr = cv2.imread(str(image_path)) if image_path.exists() else None
        report: dict = {"name": image.name}

        if color_bgr is None:
            report["skipped"] = "image file not found"
            per_image_reports.append(report)
            continue

        pose = image.cam_from_world()
        rotation = pose.rotation.matrix()
        translation = pose.translation

        pixel_x, pixel_y, inverse_depths = _gather_correspondences(
            image, reconstruction, rotation, translation
        )
        report["num_correspondences"] = len(inverse_depths)

        if len(inverse_depths) < config.min_alignment_correspondences:
            report["skipped"] = (
                f"only {len(inverse_depths)} correspondences "
                f"(need >= {config.min_alignment_correspondences})"
            )
            per_image_reports.append(report)
            continue

        disparity = depth_estimator.predict_disparity(color_bgr, image.name)
        disparities_at_points = _sample_disparity(disparity, pixel_x, pixel_y)

        depth_spread = float(np.ptp(inverse_depths)) or 1.0
        inlier_threshold = _INLIER_THRESHOLD_FRACTION * depth_spread
        a, b, inlier_mask = depth_alignment.fit_affine_ransac(
            disparities_at_points, inverse_depths, inlier_threshold=inlier_threshold
        )
        fit_rmse = depth_alignment.rmse(
            disparities_at_points[inlier_mask], inverse_depths[inlier_mask], a, b
        )
        report.update(
            {
                "num_inliers": int(inlier_mask.sum()),
                "affine_a": a,
                "affine_b": b,
                "alignment_rmse_inv_m": fit_rmse,
            }
        )
        all_rmses.append(fit_rmse)
        per_image_reports.append(report)

        metric_inverse_depth = a * disparity + b
        with np.errstate(divide="ignore", invalid="ignore"):
            metric_depth = np.where(metric_inverse_depth > 1e-6, 1.0 / metric_inverse_depth, 0.0)
        depth_path = depth_dir / f"{Path(image.name).stem}.npy"
        np.save(depth_path, metric_depth.astype(np.float32))

        inlier_depths = 1.0 / inverse_depths[inlier_mask]
        median_depths.append(float(np.median(inlier_depths)))
        camera = image.camera
        extrinsic = np.eye(4)
        extrinsic[:3, :3] = rotation
        extrinsic[:3, 3] = translation
        frames.append(
            _AlignedFrame(
                image_path=image_path,
                depth_path=depth_path,
                depth_trunc=_DEPTH_TRUNC_SAFETY_FACTOR * float(np.max(inlier_depths)),
                intrinsic=np.array(
                    [
                        [camera.focal_length_x, 0.0, camera.principal_point_x],
                        [0.0, camera.focal_length_y, camera.principal_point_y],
                        [0.0, 0.0, 1.0],
                    ]
                ),
                size=(camera.width, camera.height),
                extrinsic=extrinsic,
            )
        )

    (output_dir / "alignment_report.json").write_text(json.dumps(per_image_reports, indent=2))

    if not frames:
        raise SfMError(
            "No image had enough sparse correspondences to align a metric depth map "
            f"(need >= {config.min_alignment_correspondences} per image).",
            remedy="Check sfm/diagnostics.json -- the sparse reconstruction is likely too "
            "weak for dense reconstruction. Re-shoot with more texture/parallax.",
        )

    cloud, num_tiles, voxel = _integrate(
        reconstruction, frames, config, float(np.median(median_depths))
    )
    removed = 0
    if config.suppress_transients and len(cloud.points):
        points = np.asarray(cloud.points)
        views = (transients.DepthView(f.load_depth(), f.intrinsic, f.extrinsic) for f in frames)
        mask = transients.transient_mask(
            points,
            views,
            tolerance=_TRANSIENT_TOLERANCE_VOXELS * voxel,
            min_views=config.transient_min_views,
        )
        removed = int(mask.sum())
        cloud = cloud.select_by_index(np.flatnonzero(~mask))
        logger.info("Transient suppression removed %d of %d points.", removed, len(points))

    dense_ply_path = output_dir / "dense.ply"
    o3d.io.write_point_cloud(str(dense_ply_path), cloud)

    overall_rmse = float(np.mean(all_rmses)) if all_rmses else None
    logger.info(
        "Dense reconstruction: %d/%d images integrated into %d TSDF tile(s), %d points, "
        "mean alignment RMSE %s.",
        len(frames),
        len(reconstruction.images),
        num_tiles,
        len(cloud.points),
        f"{overall_rmse:.4f}" if overall_rmse is not None else "n/a",
    )

    return DenseResult(
        backend="monodepth_tsdf",
        num_points=len(cloud.points),
        dense_points_path=str(dense_ply_path.relative_to(output_dir)),
        alignment_rmse=overall_rmse,
    )
