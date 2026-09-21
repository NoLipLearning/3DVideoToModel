"""Phase 2a diagnostics: registration rate, track length, low-texture
detection, and the "pure rotation" abort check.

docs/ARCHITECTURE.md Section 3.1 ("Detect -- in phase2_sfm/diagnostics.py")
and Section 3.5 ("Pure rotation / no parallax"). A diagnostics JSON is
written even when reconstruction fails outright (zero registered images)
so a bad capture always leaves a concrete, inspectable reason behind --
see `write_diagnostics`.
"""

from __future__ import annotations

import math
from pathlib import Path

import pycolmap

from v2m.config import SfmConfig
from v2m.types import LowKeypointImage, SfmDiagnostics


def low_keypoint_images(database_path: Path, min_keypoints: int) -> list[LowKeypointImage]:
    """Every image whose extracted keypoint count is below `min_keypoints`
    (Section 3.1's textureless-frame flag)."""
    flagged = []
    with pycolmap.Database.open(database_path) as db:
        for image in db.read_all_images():
            count = db.num_keypoints_for_image(image.image_id)
            if count < min_keypoints:
                flagged.append(LowKeypointImage(name=image.name, keypoint_count=count))
    return flagged


def median_triangulation_angle_deg(reconstruction: pycolmap.Reconstruction) -> float | None:
    """Median triangulation angle (degrees) across every 3D point's first
    two track observations.

    Section 3.5: a low median angle suggests the camera orbited in place
    without real parallax ("spun in place" rather than walked around the
    subject) -- SfM cannot recover reliable depth from that, even if
    matching itself looks fine. Returns None for an empty/failed
    reconstruction (nothing to measure).
    """
    angles: list[float] = []
    for point in reconstruction.points3D.values():
        elements = point.track.elements
        if len(elements) < 2:
            continue
        image_a = reconstruction.images[elements[0].image_id]
        image_b = reconstruction.images[elements[1].image_id]
        angle_rad = pycolmap.calculate_triangulation_angle(
            image_a.projection_center(), image_b.projection_center(), point.xyz
        )
        angles.append(math.degrees(angle_rad))

    if not angles:
        return None
    angles.sort()
    mid = len(angles) // 2
    if len(angles) % 2:
        return angles[mid]
    return (angles[mid - 1] + angles[mid]) / 2.0


def summarize(
    *,
    database_path: Path,
    reconstruction: pycolmap.Reconstruction | None,
    num_images_total: int,
    config: SfmConfig,
    attempt: int = 1,
) -> SfmDiagnostics:
    """Build the full diagnostics record for one reconstruction attempt.

    `reconstruction` is None when `incremental_mapping` produced nothing
    registrable at all -- diagnostics are still built and returned, just
    with `num_images_registered=0`, per the "written even on failure"
    requirement.
    """
    num_registered = reconstruction.num_reg_images() if reconstruction is not None else 0
    registration_rate = num_registered / num_images_total if num_images_total else 0.0

    diagnostics = SfmDiagnostics(
        num_images_total=num_images_total,
        num_images_registered=num_registered,
        registration_rate=registration_rate,
        attempt=attempt,
        low_keypoint_images=low_keypoint_images(database_path, config.min_keypoints_per_image),
    )

    if reconstruction is not None and num_registered > 0:
        diagnostics.mean_reprojection_error_px = reconstruction.compute_mean_reprojection_error()
        diagnostics.mean_track_length = reconstruction.compute_mean_track_length()
        diagnostics.median_triangulation_angle_deg = median_triangulation_angle_deg(reconstruction)

    if registration_rate < config.registration_rate_min:
        diagnostics.warnings.append(
            f"registration_rate {registration_rate:.2f} is below the configured minimum "
            f"{config.registration_rate_min:.2f}"
        )
    if (
        diagnostics.mean_track_length is not None
        and diagnostics.mean_track_length < config.mean_track_length_min
    ):
        diagnostics.warnings.append(
            f"mean_track_length {diagnostics.mean_track_length:.2f} is below the configured "
            f"minimum {config.mean_track_length_min:.2f} -- weak geometry"
        )
    if diagnostics.low_keypoint_images:
        diagnostics.warnings.append(
            f"{len(diagnostics.low_keypoint_images)} image(s) had fewer than "
            f"{config.min_keypoints_per_image} keypoints -- likely low-texture regions "
            "(blank walls, sky, glass)"
        )
    if (
        diagnostics.median_triangulation_angle_deg is not None
        and diagnostics.median_triangulation_angle_deg < config.min_triangulation_angle_deg
    ):
        diagnostics.warnings.append(
            f"median triangulation angle {diagnostics.median_triangulation_angle_deg:.1f} deg "
            f"is below {config.min_triangulation_angle_deg} deg -- capture may be pure rotation "
            "with little parallax (walk around the subject rather than pivoting in place)"
        )

    return diagnostics


def write_diagnostics(diagnostics: SfmDiagnostics, output_path: Path) -> None:
    output_path.write_text(diagnostics.model_dump_json(indent=2))
