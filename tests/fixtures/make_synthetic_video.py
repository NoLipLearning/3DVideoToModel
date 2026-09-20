"""Generate the ground-truth synthetic fixture used across tests.

Renders a camera orbiting a 200mm cube whose six faces are covered in a
checkerboard of colored points (ORB-friendly corner-like features,
without needing real polygon rasterization / occlusion handling -- each
point is individually visibility-culled by its face normal, then drawn as
a filled circle). This is the "textured cube -> video with ground truth"
fixture docs/ARCHITECTURE.md's Verification Strategy section calls for,
scoped for M1's needs (Phase 1 ingest); later milestones (M2+) can read
the same ground-truth JSON for pose/reconstruction accuracy checks.

Deliberately injects two kinds of "bad" frames so Phase 1's gates have
something real to reject:
  - a Gaussian-blurred frame every `blur_every` frames (simulates shake)
  - a short run of frozen (repeated) camera poses (simulates the user
    holding the phone still) -- the redundancy gate should collapse these
    down to ~1 kept frame

Regenerate with: `uv run python tests/fixtures/make_synthetic_video.py`
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

FIXTURES_DIR = Path(__file__).parent
OUTPUT_VIDEO = FIXTURES_DIR / "sample.mp4"
OUTPUT_GROUND_TRUTH = FIXTURES_DIR / "sample_ground_truth.json"

CUBE_SIDE_MM = 200.0
IMAGE_SIZE = 480
FX = FY = 420.0
CAMERA_RADIUS_MM = 550.0
ORBIT_STEPS = 70  # ~5 degrees apart, just under a full 360 degree loop
FPS = 15
POINTS_PER_FACE_EDGE = 8  # 8x8 grid per face -> 384 points total
FREEZE_RUN_START = 30  # index (in the orbit sequence) where the "held still" run begins
FREEZE_RUN_LENGTH = 5
BLUR_EVERY = 9  # every 9th frame gets a synthetic motion-blur pass

# Face definition: (normal, u_axis, v_axis, base_color BGR). u/v span the
# face in [-1, 1] and are scaled by CUBE_SIDE_MM/2.
_HALF = CUBE_SIDE_MM / 2.0
# Face normal, u-axis, v-axis, base BGR color.
_X, _Y, _Z = np.array([1.0, 0, 0]), np.array([0, 1.0, 0]), np.array([0, 0, 1.0])
_FACES = [
    (_Z, _X, _Y, (60, 60, 220)),  # +Z red-ish
    (-_Z, _X, _Y, (60, 200, 60)),  # -Z green-ish
    (_X, _Z, _Y, (220, 60, 60)),  # +X blue-ish
    (-_X, _Z, _Y, (60, 200, 220)),  # -X yellow-ish
    (_Y, _X, _Z, (220, 60, 200)),  # +Y magenta-ish
    (-_Y, _X, _Z, (200, 220, 60)),  # -Y cyan-ish
]


def _build_cube_points() -> tuple[np.ndarray, np.ndarray, list[tuple]]:
    """Returns (points_3d [N,3], normals [N,3], colors [N] as BGR tuples)."""
    points = []
    normals = []
    colors = []
    steps = np.linspace(-1.0, 1.0, POINTS_PER_FACE_EDGE)
    for normal, u_axis, v_axis, base_color in _FACES:
        for i, u in enumerate(steps):
            for j, v in enumerate(steps):
                point = normal * _HALF + (u * u_axis + v * v_axis) * _HALF
                points.append(point)
                normals.append(normal)
                # Checkerboard shading for extra corner-like contrast.
                if (i + j) % 2 == 0:
                    color = base_color
                else:
                    color = tuple(max(0, c - 90) for c in base_color)
                colors.append(color)
    return np.array(points, dtype=np.float64), np.array(normals, dtype=np.float64), colors


def _camera_pose(azimuth_deg: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (rvec, tvec, camera_center_world) for a camera on a ring
    around the cube at `azimuth_deg`, always looking at the origin, using
    OpenCV's convention (camera looks down +Z, X right, Y down)."""
    azimuth = np.radians(azimuth_deg)
    height = 60.0 * np.sin(azimuth * 2.0)  # gentle bob, avoids a perfectly flat orbit
    eye = np.array([CAMERA_RADIUS_MM * np.cos(azimuth), height, CAMERA_RADIUS_MM * np.sin(azimuth)])
    target = np.zeros(3)
    world_up = np.array([0.0, 1.0, 0.0])

    forward = target - eye
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, world_up)
    right /= np.linalg.norm(right)
    cam_up = np.cross(forward, right)

    # OpenCV camera axes: row0=right, row1=down(-up), row2=forward.
    rotation = np.stack([right, -cam_up, forward], axis=0)
    tvec = -rotation @ eye
    rvec, _ = cv2.Rodrigues(rotation)
    return rvec.reshape(3), tvec.reshape(3), eye


def _camera_matrix() -> np.ndarray:
    cx = cy = IMAGE_SIZE / 2.0
    return np.array([[FX, 0, cx], [0, FY, cy], [0, 0, 1]], dtype=np.float64)


def _render_frame(points, normals, colors, rvec, tvec, cam_center, camera_matrix) -> np.ndarray:
    image = np.full((IMAGE_SIZE, IMAGE_SIZE, 3), 120, dtype=np.uint8)  # neutral gray background

    view_dirs = points - cam_center
    view_dirs /= np.linalg.norm(view_dirs, axis=1, keepdims=True)
    visible = np.einsum("ij,ij->i", normals, -view_dirs) > 0.05

    if not np.any(visible):
        return image

    visible_points = points[visible]
    visible_colors = [c for c, v in zip(colors, visible, strict=True) if v]

    projected, _ = cv2.projectPoints(visible_points, rvec, tvec, camera_matrix, None)
    projected = projected.reshape(-1, 2)

    # Depth-sort so nearer points draw last (on top) -- purely cosmetic,
    # since points don't actually occlude each other in this simplified
    # renderer, but it keeps overlapping circles looking sane.
    depths = np.linalg.norm(visible_points - cam_center, axis=1)
    order = np.argsort(-depths)

    for idx in order:
        x, y = projected[idx]
        if -20 <= x <= IMAGE_SIZE + 20 and -20 <= y <= IMAGE_SIZE + 20:
            cv2.circle(image, (int(round(x)), int(round(y))), 6, visible_colors[idx], thickness=-1)

    return image


def generate() -> None:
    points, normals, colors = _build_cube_points()
    camera_matrix = _camera_matrix()

    azimuths = np.linspace(0, 360, ORBIT_STEPS, endpoint=False)
    # Freeze the camera pose for a short run, simulating "held still".
    frozen_azimuth = azimuths[FREEZE_RUN_START]
    for offset in range(FREEZE_RUN_LENGTH):
        azimuths[FREEZE_RUN_START + offset] = frozen_azimuth

    writer = cv2.VideoWriter(
        str(OUTPUT_VIDEO),
        cv2.VideoWriter_fourcc(*"mp4v"),
        FPS,
        (IMAGE_SIZE, IMAGE_SIZE),
    )

    blur_frame_indices: list[int] = []
    duplicate_frame_indices: list[int] = [
        FREEZE_RUN_START + offset for offset in range(1, FREEZE_RUN_LENGTH)
    ]
    poses = []

    for i, azimuth in enumerate(azimuths):
        rvec, tvec, cam_center = _camera_pose(float(azimuth))
        frame = _render_frame(points, normals, colors, rvec, tvec, cam_center, camera_matrix)

        if i > 0 and i % BLUR_EVERY == 0:
            frame = cv2.GaussianBlur(frame, (21, 21), 0)
            blur_frame_indices.append(i)

        writer.write(frame)
        poses.append(
            {
                "frame_index": i,
                "azimuth_deg": float(azimuth),
                "rvec": rvec.tolist(),
                "tvec": tvec.tolist(),
                "camera_center_world_mm": cam_center.tolist(),
            }
        )

    writer.release()

    ground_truth = {
        "cube_side_length_mm": CUBE_SIDE_MM,
        "image_size_px": IMAGE_SIZE,
        "camera_matrix": camera_matrix.tolist(),
        "fps": FPS,
        "total_frames": len(azimuths),
        "synthetic_blur_frame_indices": blur_frame_indices,
        "synthetic_duplicate_frame_indices": duplicate_frame_indices,
        "poses": poses,
    }
    OUTPUT_GROUND_TRUTH.write_text(json.dumps(ground_truth, indent=2))
    print(f"Wrote {OUTPUT_VIDEO} ({len(azimuths)} frames) and {OUTPUT_GROUND_TRUTH}")


if __name__ == "__main__":
    generate()
