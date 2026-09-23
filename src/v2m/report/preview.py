"""A static, shaded preview image of a mesh, for report.html.

No GPU or offscreen OpenGL context is assumed (open3d's offscreen
renderer needs EGL, which a headless box usually lacks, and matplotlib is
not a dependency), so this is a small software rasterizer: back-face
culling, painter's-algorithm ordering, and Lambert shading, drawn with
PIL. Roughly a second for a 300k-face mesh -- fine for a once-per-run
report. The web UI (M7) shows the real interactive GLB instead.
"""

from __future__ import annotations

import io

import numpy as np
import trimesh
from PIL import Image, ImageDraw

_BASE_COLOR = np.array([96.0, 150.0, 210.0])
_AMBIENT = 0.3


def render_preview(
    mesh: trimesh.Trimesh,
    size: int = 480,
    azimuth_deg: float = 35.0,
    elevation_deg: float = 25.0,
    background: tuple[int, int, int, int] = (0, 0, 0, 0),
) -> bytes:
    """PNG bytes of `mesh` seen from (azimuth, elevation), +Z up."""
    azimuth, elevation = np.radians(azimuth_deg), np.radians(elevation_deg)
    to_camera = np.array(
        [
            np.cos(elevation) * np.cos(azimuth),
            np.cos(elevation) * np.sin(azimuth),
            np.sin(elevation),
        ]
    )
    right = np.cross(-to_camera, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, -to_camera)

    vertices = mesh.vertices - mesh.bounds.mean(axis=0)
    screen = np.column_stack([vertices @ right, vertices @ up])
    depth = vertices @ to_camera

    extent = float(np.abs(screen).max()) or 1.0
    pixels = screen / extent * (size * 0.45)
    pixels[:, 1] *= -1  # image y grows downward
    pixels += size / 2

    normals = mesh.face_normals
    facing = normals @ to_camera > 0
    faces = mesh.faces[facing]
    face_normals = normals[facing]

    light = to_camera + np.array([0.3, -0.2, 0.6])
    light /= np.linalg.norm(light)
    shade = _AMBIENT + (1 - _AMBIENT) * np.clip(face_normals @ light, 0.0, 1.0)
    colors = np.clip(shade[:, None] * _BASE_COLOR, 0, 255).astype(np.uint8)

    order = np.argsort(depth[faces].mean(axis=1))  # farthest first
    image = Image.new("RGBA", (size, size), background)
    draw = ImageDraw.Draw(image)
    corners = pixels[faces]
    for index in order:
        triangle = corners[index]
        color = tuple(int(c) for c in colors[index])
        draw.polygon(
            [
                (triangle[0, 0], triangle[0, 1]),
                (triangle[1, 0], triangle[1, 1]),
                (triangle[2, 0], triangle[2, 1]),
            ],
            fill=color,
            outline=color,
        )

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()
