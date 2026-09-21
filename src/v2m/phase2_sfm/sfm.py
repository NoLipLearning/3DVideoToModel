"""Phase 2a orchestrator: sparse structure-from-motion.

Ties together backend feature/match (backends/colmap_backend.py),
COLMAP's own incremental mapper (shared across backends, per
backends/base.py's split), diagnostics, the Section 3.1 low-texture
retry, and export (cameras.json, sparse.ply, undistorted images). Read
docs/ARCHITECTURE.md Sections 2 and 3.1 before changing the
retry/backend-selection logic.
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path

import cv2
import pycolmap

from v2m.config import SfmConfig
from v2m.errors import SfMError
from v2m.phase2_sfm import diagnostics as diagnostics_module
from v2m.phase2_sfm.backends import colmap_backend
from v2m.types import SfmDiagnostics, SfmResult

logger = logging.getLogger("v2m.phase2_sfm.sfm")


def _apply_clahe(images_dir: Path, staging_dir: Path, config: SfmConfig) -> Path:
    """CLAHE-enhance every frame's luminance channel into `staging_dir`
    (Section 3.1 mitigation #1: "recovers detail on near-uniform
    surfaces"). Applied in LAB space so only local contrast is boosted --
    color is preserved."""
    if staging_dir.exists():
        shutil.rmtree(staging_dir)
    staging_dir.mkdir(parents=True)

    clahe = cv2.createCLAHE(
        clipLimit=config.clahe_clip_limit,
        tileGridSize=(config.clahe_tile_grid, config.clahe_tile_grid),
    )
    for image_path in sorted(images_dir.glob("*.jpg")):
        image = cv2.imread(str(image_path))
        l_channel, a_channel, b_channel = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2LAB))
        enhanced_l = clahe.apply(l_channel)
        enhanced = cv2.cvtColor(cv2.merge((enhanced_l, a_channel, b_channel)), cv2.COLOR_LAB2BGR)
        cv2.imwrite(str(staging_dir / image_path.name), enhanced)
    return staging_dir


def _pick_largest_reconstruction(
    reconstructions: dict[int, pycolmap.Reconstruction],
) -> pycolmap.Reconstruction | None:
    if not reconstructions:
        return None
    best_key = max(reconstructions, key=lambda key: reconstructions[key].num_reg_images())
    return reconstructions[best_key]


def _run_one_attempt(
    images_dir: Path, sfm_dir: Path, config: SfmConfig, *, attempt: int
) -> tuple[pycolmap.Reconstruction | None, SfmDiagnostics]:
    """One full extract -> match -> map pass.

    Clears and recreates the database and sparse-model scratch space on
    every call, including a retry -- COLMAP's feature extractor skips
    images that already have features in the database, so a retry with a
    different SIFT threshold needs a fresh database to actually take
    effect. The returned `Reconstruction` is a live in-memory object and
    survives this cleanup regardless of what happens to these files on a
    later attempt; only the *chosen* attempt's reconstruction is written
    back to a stable path afterward, in `run_sparse_sfm`.
    """
    database_path = sfm_dir / "database.db"
    sparse_scratch = sfm_dir / "sparse"
    if database_path.exists():
        database_path.unlink()
    if sparse_scratch.exists():
        shutil.rmtree(sparse_scratch)
    sparse_scratch.mkdir(parents=True)

    num_images_total = len(list(images_dir.glob("*.jpg")))

    colmap_backend.extract_and_match(images_dir, database_path, config)

    mapper_options = pycolmap.IncrementalPipelineOptions()
    reconstructions = pycolmap.incremental_mapping(
        database_path=database_path,
        image_path=images_dir,
        output_path=sparse_scratch,
        options=mapper_options,
    )
    best = _pick_largest_reconstruction(reconstructions)

    diag = diagnostics_module.summarize(
        database_path=database_path,
        reconstruction=best,
        num_images_total=num_images_total,
        config=config,
        attempt=attempt,
    )
    return best, diag


def _export_cameras_json(reconstruction: pycolmap.Reconstruction, output_path: Path) -> None:
    """A simplified, JSON-friendly camera/pose export.

    Rotation is a quaternion in (x, y, z, w) order -- Eigen/COLMAP's
    convention, confirmed directly against pycolmap's own
    `Rigid3d.todict()` (an identity rotation serializes as
    quat=[0, 0, 0, 1]). `cam_from_world` maps a WORLD point into CAMERA
    coordinates: `p_cam = R @ p_world + t`.
    """
    cameras = {
        str(camera_id): {
            "model": camera.model.name,
            "width": camera.width,
            "height": camera.height,
            "params": camera.params.tolist(),
        }
        for camera_id, camera in reconstruction.cameras.items()
    }
    images = {}
    for image in reconstruction.images.values():
        pose = image.cam_from_world()
        images[image.name] = {
            "camera_id": image.camera_id,
            "rotation_quat_xyzw": pose.rotation.quat.tolist(),
            "translation": pose.translation.tolist(),
        }
    payload = {
        "convention": "cam_from_world: p_cam = R @ p_world + t; quat is (x, y, z, w)",
        "cameras": cameras,
        "images": images,
    }
    output_path.write_text(json.dumps(payload, indent=2))


def run_sparse_sfm(images_dir: Path, output_dir: Path, config: SfmConfig) -> SfmResult:
    """Run Phase 2a end to end: images -> sparse reconstruction.

    Writes `<output_dir>/{database.db, sparse/final/, undistorted/,
    cameras.json, sparse.ply, diagnostics.json}`. Deliberately
    RunContext-agnostic like `phase1_ingest.extract.run_extract` --
    `output_dir` is just a directory.

    (`sparse/final/` rather than the architecture doc's illustrative
    `sparse/0/`: COLMAP numbers *candidate* models per attempt, and after
    a Section 3.1 retry the winning reconstruction may not be "0" from
    either attempt -- "final" names the one this function actually chose,
    unambiguously.)
    """
    if not any(images_dir.glob("*.jpg")):
        raise SfMError(
            f"No .jpg frames found in {images_dir}.",
            remedy="Run `v2m extract` first to populate frames/ before `v2m sfm`.",
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    effective_images_dir = images_dir
    if config.use_clahe:
        effective_images_dir = _apply_clahe(images_dir, output_dir / "clahe_frames", config)

    reconstruction, diag = _run_one_attempt(effective_images_dir, output_dir, config, attempt=1)

    needs_retry = reconstruction is None or diag.registration_rate < config.registration_rate_min
    if needs_retry:
        logger.warning(
            "Initial sparse SfM pass registered %d/%d images (rate %.2f) -- retrying with a "
            "lowered SIFT peak_threshold (%.4f -> %.4f) per docs/ARCHITECTURE.md Section 3.1.",
            diag.num_images_registered,
            diag.num_images_total,
            diag.registration_rate,
            config.sift_peak_threshold,
            config.sift_peak_threshold_retry,
        )
        retry_config = config.model_copy(
            update={"sift_peak_threshold": config.sift_peak_threshold_retry}
        )
        retry_reconstruction, retry_diag = _run_one_attempt(
            effective_images_dir, output_dir, retry_config, attempt=2
        )
        if retry_diag.num_images_registered >= diag.num_images_registered:
            reconstruction, diag = retry_reconstruction, retry_diag

    diagnostics_module.write_diagnostics(diag, output_dir / "diagnostics.json")

    if reconstruction is None or diag.num_images_registered == 0:
        raise SfMError(
            f"Sparse reconstruction registered 0/{diag.num_images_total} images.",
            remedy="Footage likely lacks parallax or texture -- walk around the subject with "
            "a steady, textured background, and check diagnostics.json for details.",
        )

    final_sparse_dir = output_dir / "sparse" / "final"
    if final_sparse_dir.exists():
        shutil.rmtree(final_sparse_dir)
    final_sparse_dir.mkdir(parents=True)
    reconstruction.write(final_sparse_dir)

    undistorted_dir = output_dir / "undistorted"
    if undistorted_dir.exists():
        shutil.rmtree(undistorted_dir)
    pycolmap.undistort_images(
        output_path=undistorted_dir,
        input_path=final_sparse_dir,
        image_path=effective_images_dir,
    )

    sparse_ply_path = output_dir / "sparse.ply"
    reconstruction.export_PLY(str(sparse_ply_path))

    cameras_json_path = output_dir / "cameras.json"
    _export_cameras_json(reconstruction, cameras_json_path)

    return SfmResult(
        num_images_registered=diag.num_images_registered,
        num_images_total=diag.num_images_total,
        mean_reprojection_error_px=diag.mean_reprojection_error_px or 0.0,
        mean_track_length=diag.mean_track_length or 0.0,
        sparse_points_path=str(sparse_ply_path.relative_to(output_dir)),
        cameras_path=str(cameras_json_path.relative_to(output_dir)),
    )
