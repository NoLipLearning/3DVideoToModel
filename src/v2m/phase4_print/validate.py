"""Printability assertions -- the last check before export.

docs/ARCHITECTURE.md Section 4 (M5): assert is_watertight,
is_winding_consistent, volume > 0, euler_number sane, min wall
thickness, bbox within build volume. Runs *after* scale.py (Section 4's
own M5 ordering), so `mesh` here is already in real millimetres --
`min_wall_thickness_mm`/`build_volume_mm` are compared directly, no unit
conversion needed (unlike watertight.py's rung 5, which runs before
scale is known).

Only the first three are hard failures: a mesh reaching this function is
expected to already be `make_watertight()`'s output, so failing them
indicates a bug in the repair ladder, not a recoverable capture-quality
issue. The rest are recorded as warnings for the user to see before
printing, not reasons to discard otherwise-usable output.
"""

from __future__ import annotations

import numpy as np
import trimesh

from v2m.config import PrintPrepConfig
from v2m.errors import PrintPrepError

# method="ray" (cast inward along the surface normal, measure the distance
# to the opposite wall), not trimesh's default "max_sphere": confirmed
# directly, max_sphere reports ~0.09mm on a perfectly clean 150mm cube --
# near any convex edge the largest sphere tangent at a sample point is
# limited by the adjacent face, not the opposite wall, so every sharp
# edge reads as a thin wall. The ray method reads 150mm there and still
# flags a genuine 0.3mm slab. Each sample costs a nearest-face lookup
# plus a ray cast, so the count stays modest.
_THICKNESS_SAMPLE_COUNT = 200


def validate_mesh(mesh: trimesh.Trimesh, config: PrintPrepConfig) -> list[str]:
    if not mesh.is_watertight:
        raise PrintPrepError(
            "The repaired mesh is not watertight.",
            remedy="This indicates a bug in the repair ladder (watertight.py), which is "
            "supposed to guarantee this -- please report it.",
        )
    if not mesh.is_winding_consistent:
        raise PrintPrepError(
            "The repaired mesh has inconsistent face winding.",
            remedy="This indicates a bug in the repair ladder (watertight.py).",
        )
    if mesh.volume <= 0:
        raise PrintPrepError(
            f"The repaired mesh has non-positive volume ({mesh.volume:.6g}).",
            remedy="This indicates inverted normals somewhere in the repair ladder.",
        )

    warnings: list[str] = []

    if mesh.euler_number != 2:
        warnings.append(
            f"Euler number is {mesh.euler_number}, not 2 -- the model has handles, "
            "tunnels, or internal cavities (not necessarily a problem, but worth a "
            "visual check before printing)."
        )

    sample_count = min(_THICKNESS_SAMPLE_COUNT, len(mesh.faces))
    sample_points, _ = trimesh.sample.sample_surface(mesh, sample_count)
    thicknesses = trimesh.proximity.thickness(mesh, sample_points, method="ray")
    finite = thicknesses[np.isfinite(thicknesses)]
    min_thickness = float(finite.min()) if len(finite) else float("inf")
    if min_thickness < config.min_wall_thickness_mm:
        warnings.append(
            f"Minimum wall thickness ({min_thickness:.2f}mm) is below the printable "
            f"minimum ({config.min_wall_thickness_mm}mm) -- thin walls may fail to "
            "print or break easily."
        )

    extents = mesh.extents
    build_volume = np.array(config.build_volume_mm)
    if np.any(extents > build_volume):
        warnings.append(
            f"Model bounding box {extents.tolist()}mm exceeds the configured build "
            f"volume {list(config.build_volume_mm)}mm -- it will not fit the printer as scaled."
        )

    return warnings
