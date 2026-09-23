"""Spatial tiling for the TSDF (M9, docs/ARCHITECTURE.md Section 3.2:
"if bbox diagonal > tile_threshold, TSDF-integrate per spatial tile").

One `UniformTSDFVolume` is capped at MAX_TSDF_RESOLUTION^3 voxels (~3.1GB
at 400). A street or a room at the detail its capture supports can need
far more than that along its long axis. Tiling splits the scene's
bounding box into a grid of cells, integrates each cell in its own
volume (only one alive at a time, so peak memory doesn't grow), and
concatenates the extracted points. Each cell's volume overlaps its
neighbours by a margin so surfaces near a seam are fully integrated;
only points inside the cell's own core are kept, so seams produce no
duplicates.

**When to tile has to be scale-free.** The doc's "30 m" threshold can't
be applied here: COLMAP's units are arbitrary until Phase 4. What the
capture itself does tell us is how far the camera was from the surfaces
(the median depth of the sparse points, in COLMAP's own units). The
presets already say what detail they want, as metres of voxel at metres
of typical distance (object: 4mm at 0.5m; scene: 2cm at 5m). So the
ratio `tsdf_voxel_size_m / nominal_capture_distance_m`, applied to the
median depth, is the voxel size this capture deserves, whatever the
units. Tiles are only used when the bounding box needs more than one
volume's worth of those voxels along some axis. An object capture never
does, so its single-volume path is exactly the pre-M9 code.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from v2m.config import DenseConfig


@dataclass(frozen=True)
class Tile:
    core_min: np.ndarray  # this tile owns points in [core_min, core_max)
    core_max: np.ndarray
    origin: np.ndarray  # the TSDF cube: origin + length, including the margin
    length: float
    resolution: int

    def contains(self, points: np.ndarray) -> np.ndarray:
        return np.all((points >= self.core_min) & (points < self.core_max), axis=1)

    def touches(self, points: np.ndarray) -> bool:
        """Whether any of `points` falls inside this tile's (margin-
        expanded) volume -- i.e. whether a frame should be integrated here."""
        inside = np.all((points >= self.origin) & (points < self.origin + self.length), axis=1)
        return bool(inside.any())


def desired_voxel(median_depth: float, config: DenseConfig) -> float:
    return median_depth * config.tsdf_voxel_size_m / config.nominal_capture_distance_m


def tile_counts(extent: np.ndarray, voxel: float, max_resolution: int, max_tiles: int) -> list[int]:
    """Cells per axis so each cell fits in one volume at `voxel`, reduced
    (largest axis first) until the total is within `max_tiles` -- past
    that, the scene is integrated coarser rather than refused."""
    counts = [max(1, math.ceil(float(e) / (voxel * max_resolution))) for e in extent]
    while math.prod(counts) > max_tiles:
        axis = int(np.argmax(counts))
        counts[axis] -= 1
    return counts


def plan_tiles(
    bbox_min: np.ndarray,
    bbox_max: np.ndarray,
    voxel: float,
    config: DenseConfig,
    max_resolution: int,
    margin_voxels: int,
) -> list[Tile]:
    extent = bbox_max - bbox_min
    counts = tile_counts(extent, voxel, max_resolution, config.max_tiles)
    cell = extent / np.array(counts)
    tiles = []
    for index in np.ndindex(*counts):
        core_min = bbox_min + cell * np.array(index)
        core_max = core_min + cell
        # Open the outer faces of edge cells, so no point is lost to
        # rounding right at the scene's bounding box.
        core_min = np.where(np.array(index) == 0, -np.inf, core_min)
        core_max = np.where(np.array(index) == np.array(counts) - 1, np.inf, core_max)
        finite_min = bbox_min + cell * np.array(index)
        side = float(cell.max())
        resolution = int(np.clip(math.ceil(side / voxel), 32, max_resolution))
        margin = margin_voxels * side / resolution
        length = side + 2 * margin
        center = finite_min + cell / 2
        tiles.append(
            Tile(
                core_min=core_min,
                core_max=core_max,
                origin=center - length / 2,
                length=length,
                resolution=resolution,
            )
        )
    return tiles


def sample_world_points(
    depth: np.ndarray, intrinsic: np.ndarray, cam_from_world: np.ndarray, stride: int = 8
) -> np.ndarray:
    """A sparse grid of a depth map's valid pixels, back-projected to
    world coordinates -- enough to tell which tiles a frame sees."""
    rows, cols = np.mgrid[0 : depth.shape[0] : stride, 0 : depth.shape[1] : stride]
    z = depth[rows, cols]
    valid = z > 0
    z, u, v = z[valid], cols[valid], rows[valid]
    fx, fy, cx, cy = intrinsic[0, 0], intrinsic[1, 1], intrinsic[0, 2], intrinsic[1, 2]
    cam_points = np.column_stack([(u - cx) * z / fx, (v - cy) * z / fy, z])
    rotation, translation = cam_from_world[:3, :3], cam_from_world[:3, 3]
    return (cam_points - translation) @ rotation  # R^T (p - t), row-vector form
