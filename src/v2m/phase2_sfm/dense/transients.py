"""Moving-object suppression (M9, docs/ARCHITECTURE.md Section 3.5: "TSDF
integration naturally suppresses transient surfaces; additionally flag
high per-voxel depth variance").

A parked car that drove off, or a person who walked through, is solid in
a few frames and empty in the rest. TSDF averaging thins it out but
often not completely: a few frames of strong evidence survive against
many frames that merely saw past it. This pass makes that explicit. For
every dense point and every frame whose (aligned, metric) depth map
covers it:

- **support**: the frame's depth there matches the point's depth (within
  `tolerance`), so the frame saw this surface;
- **free-space violation**: the frame's depth there is *beyond* the point
  by more than `tolerance`, so the frame looked straight through where
  the point claims to be;
- anything nearer (the point is behind something from this view) is
  occlusion and says nothing either way.

A point with more violations than support, and at least `min_views`
violations, was seen through more often than it was seen: it is
transient, and removed. That is the Section 3.5 "depth variance" signal
made concrete per point, with occlusion handled explicitly, since
occlusion is the thing that makes a plain variance threshold remove real
geometry.

Off by default for objects (a turntable-style object capture has no
passers-by) and on in the scene preset.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np


@dataclass
class DepthView:
    depth: np.ndarray  # HxW metric depth, 0 = no data
    intrinsic: np.ndarray  # 3x3
    cam_from_world: np.ndarray  # 4x4


def transient_mask(
    points: np.ndarray, views: Iterable[DepthView], tolerance: float, min_views: int
) -> np.ndarray:
    """Boolean mask over `points`: True = transient (remove)."""
    support = np.zeros(len(points), dtype=np.int32)
    violations = np.zeros(len(points), dtype=np.int32)
    for view in views:
        rotation, translation = view.cam_from_world[:3, :3], view.cam_from_world[:3, 3]
        cam = points @ rotation.T + translation
        z = cam[:, 2]
        in_front = z > 1e-9
        u = np.full(len(points), -1, dtype=np.int64)
        v = np.full(len(points), -1, dtype=np.int64)
        u[in_front] = np.round(
            view.intrinsic[0, 0] * cam[in_front, 0] / z[in_front] + view.intrinsic[0, 2]
        ).astype(np.int64)
        v[in_front] = np.round(
            view.intrinsic[1, 1] * cam[in_front, 1] / z[in_front] + view.intrinsic[1, 2]
        ).astype(np.int64)
        height, width = view.depth.shape
        visible = in_front & (u >= 0) & (u < width) & (v >= 0) & (v < height)
        observed = np.zeros(len(points))
        observed[visible] = view.depth[v[visible], u[visible]]
        has_data = visible & (observed > 0)
        support += has_data & (np.abs(observed - z) <= tolerance)
        violations += has_data & (observed > z + tolerance)
    return (violations > support) & (violations >= min_views)
