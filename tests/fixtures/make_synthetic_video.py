"""Generate the ground-truth synthetic fixture used across tests.

Renders a camera orbiting a 200mm cube whose six faces are covered with a
real warped texture (a face-unique noise pattern, homography-mapped from
the texture image onto each face's projected quad -- not a sparse grid of
discrete points). This is the "textured cube -> video with ground truth"
fixture docs/ARCHITECTURE.md's Verification Strategy section calls for;
M1 (Phase 1 ingest) and M2 (sparse SfM registration) both verify against
it.

IMPORTANT -- why scattered shapes, not pasted points or per-pixel noise:
three earlier versions of this fixture didn't give COLMAP enough to work
with, each failing for a different, non-obvious reason:
  v1: flat-colored circles (2 shades/face). SIFT keys off grayscale
      gradients, not hue -- converted to grayscale the whole 480x480
      image had only 5 unique intensity values, so same-shade points were
      pixel-identical to the descriptor. Registered 2/50 images.
  v2: unique random noise *patches* pasted at each of ~64 points/face.
      Fixed the descriptor-ambiguity problem (100+ unique grayscale
      values per patch) but the points were still sparse decals that
      didn't warp with viewpoint. Still 2/54.
  v3: each face as one continuous per-pixel random-noise texture,
      correctly homography-warped from its 3D quad into every frame.
      Looked right and had plenty of local gradient -- but COLMAP's own
      verbose mapper log (`incremental_pipeline.cc`) showed every single
      candidate seed pair failing to grow past 2 registered images, even
      ones with 100+ raw 2D-3D correspondences. Root cause: per-pixel
      white noise is statistically self-similar everywhere -- different
      patches of uniform noise can look alike to a descriptor, so many
      "matches" are false positives that don't survive PnP RANSAC when
      registering a third image. Still 2/49.
  v4 (this version): each face's texture is a sparse scatter of
      randomly-sized, randomly-colored circles/rectangles over a tinted
      background -- genuinely distinctive, non-repeating macro structure
      (closer to how a real cluttered surface or a fiducial-marker board
      behaves) rather than statistically-uniform noise. Warped onto each
      face's 3D quad exactly as in v3.

Deliberately injects two kinds of "bad" frames so Phase 1's gates have
something real to reject:
  - a Gaussian-blurred frame every `blur_every` frames (simulates shake)
  - a short run of frozen (repeated) camera poses (simulates the user
    holding the phone still) -- the redundancy gate should collapse these
    down to ~1 kept frame

Also fixed at M3: `_camera_pose`'s rotation matrix was a *reflection*
(det=-1), not a proper rotation, from v1 onward -- see the comment at its
`down = cross(forward, right)` line for the full story. It never affected
M1/M2 (COLMAP only needs the rendered frames to be *self*-consistent, not
match any particular "intended" camera path) but would have silently
broken M3's ray-traced ground-truth depth, which needs the actual camera
model to be geometrically correct.

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
IMAGE_SIZE = 640
FX = FY = 560.0
CAMERA_RADIUS_MM = 400.0
ORBIT_STEPS = 70  # ~5 degrees apart, just under a full 360 degree loop
FPS = 15
FACE_TEXTURE_SIZE = 384
FREEZE_RUN_START = 30  # index (in the orbit sequence) where the "held still" run begins
FREEZE_RUN_LENGTH = 5
BLUR_EVERY = 9  # every 9th frame gets a synthetic motion-blur pass
_TEXTURE_RNG_SEED = 20260921  # fixed seed -> the fixture is reproducible

# Face definition: (normal, u_axis, v_axis, base_color BGR). u/v span the
# face in [-1, 1] and are scaled by CUBE_SIDE_MM/2.
_HALF = CUBE_SIDE_MM / 2.0
_X, _Y, _Z = np.array([1.0, 0, 0]), np.array([0, 1.0, 0]), np.array([0, 0, 1.0])
_FACES = [
    (_Z, _X, _Y, (60, 60, 220)),  # +Z red-ish
    (-_Z, _X, _Y, (60, 200, 60)),  # -Z green-ish
    (_X, _Z, _Y, (220, 60, 60)),  # +X blue-ish
    (-_X, _Z, _Y, (60, 200, 220)),  # -X yellow-ish
    (_Y, _X, _Z, (220, 60, 200)),  # +Y magenta-ish
    (-_Y, _X, _Z, (200, 220, 60)),  # -Y cyan-ish
]

# (0,0) -> (W,0) -> (W,H) -> (0,H): TL, TR, BR, BL in the texture image.
_TEXTURE_CORNERS_2D = np.array(
    [
        [0, 0],
        [FACE_TEXTURE_SIZE - 1, 0],
        [FACE_TEXTURE_SIZE - 1, FACE_TEXTURE_SIZE - 1],
        [0, FACE_TEXTURE_SIZE - 1],
    ],
    dtype=np.float32,
)
# Matching (u, v) corners in face-local coordinates, same TL/TR/BR/BL
# winding so index i in one array corresponds to index i in the other.
_CORNER_UV = [(-1, -1), (1, -1), (1, 1), (-1, 1)]


_SHAPES_PER_FACE = 320
_SHAPE_RADIUS_RANGE = (4, 20)
_SHAPE_COLOR_JITTER = 110


def _make_face_texture(rng: np.random.Generator, base_color: tuple) -> np.ndarray:
    """A face-unique texture: a scatter of randomly sized/placed/colored
    circles and rectangles over a tinted background.

    Deliberately NOT per-pixel noise -- see the module docstring's v3/v4
    comparison. Distinct macro-scale shapes at varying scales give SIFT
    genuinely unique, well-localized keypoints (shape boundaries); a
    statistically-uniform noise field does not, no matter how much local
    gradient it has.
    """
    base = np.array(base_color, dtype=np.float32)
    texture = np.full((FACE_TEXTURE_SIZE, FACE_TEXTURE_SIZE, 3), base * 0.7, dtype=np.uint8)
    for _ in range(_SHAPES_PER_FACE):
        radius = int(rng.integers(*_SHAPE_RADIUS_RANGE))
        center = rng.integers(0, FACE_TEXTURE_SIZE, size=2)
        jitter = rng.integers(-_SHAPE_COLOR_JITTER, _SHAPE_COLOR_JITTER, size=3)
        color = tuple(int(c) for c in np.clip(base + jitter, 0, 255))
        if rng.random() < 0.5:
            cv2.circle(texture, tuple(int(c) for c in center), radius, color, thickness=-1)
        else:
            offset = rng.integers(-radius, radius, size=2)
            corner = center + offset
            cv2.rectangle(
                texture,
                tuple(int(c) for c in center),
                tuple(int(c) for c in corner),
                color,
                thickness=-1,
            )
    # A light blur softens hard shape edges slightly without erasing the
    # macro-scale structure that makes each region distinctive.
    return cv2.GaussianBlur(texture, (3, 3), 0)


def _build_faces() -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Returns a list of (normal, corners_3d [4,3], texture [H,W,3])."""
    rng = np.random.default_rng(_TEXTURE_RNG_SEED)
    faces = []
    for normal, u_axis, v_axis, base_color in _FACES:
        corners = np.array(
            [normal * _HALF + (u * u_axis + v * v_axis) * _HALF for u, v in _CORNER_UV],
            dtype=np.float64,
        )
        texture = _make_face_texture(rng, base_color)
        faces.append((normal, corners, texture))
    return faces


def _camera_pose(azimuth_deg: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (rvec, tvec, camera_center_world) for a camera on a ring
    around the cube at `azimuth_deg`, always looking at the origin, using
    OpenCV's convention (camera looks down +Z, X right, Y down)."""
    azimuth = np.radians(azimuth_deg)
    # Bob amplitude matters beyond "avoid a perfectly flat orbit": a
    # shallow bob (originally 60mm, ~8.6 degrees of elevation) leaves the
    # top/bottom faces almost edge-on for the whole orbit, so Phase 2b's
    # dense reconstruction reconstructs them far less completely than the
    # four equatorial side faces -- invisible to M1-M4's own verification
    # (registration rate, mesh manifold-ness), but it surfaced directly at
    # M5 as a ~27% foreshortened axis after watertight repair, large enough
    # to fail the architecture doc's M5 bar ("cube round-trips to within 2%
    # of ground-truth dimensions"). 220mm (~28.8 degrees of elevation) gives
    # real, if still oblique, coverage of both poles across the orbit's two
    # up/down cycles, without going so steep it weakens the equatorial
    # faces' own already-verified registration.
    height = 220.0 * np.sin(azimuth * 2.0)
    eye = np.array([CAMERA_RADIUS_MM * np.cos(azimuth), height, CAMERA_RADIUS_MM * np.sin(azimuth)])
    target = np.zeros(3)
    world_up = np.array([0.0, 1.0, 0.0])

    forward = target - eye
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, world_up)
    right /= np.linalg.norm(right)
    # For a proper (det=+1) right-handed (right, down, forward) triple,
    # `down` must be `cross(forward, right)` -- by the BAC-CAB identity
    # `right x cross(forward, right) = forward*(right.right) -
    # right*(right.forward) = forward` exactly (right and forward are
    # orthonormal), i.e. right x down = forward, the defining property of
    # a right-handed system. An earlier version of this function negated
    # this vector (thinking of it as "-cam_up"), which silently flipped
    # the handedness into a *reflection* (det=-1) -- cv2.Rodrigues still
    # accepts it and returns *some* rvec, but that rvec does not
    # round-trip back to the same matrix, since Rodrigues' formula can
    # only ever represent a proper rotation. Every frame this fixture
    # ever rendered was still self-consistent (cv2.projectPoints uses
    # Rodrigues(rvec) internally too, so encoding and rendering agreed
    # with *each other*), which is why M1's gating logic and M2's SfM
    # registration both worked fine regardless -- but it breaks
    # render_true_depth below, which needs the actual camera model to be
    # geometrically correct, not just self-consistent.
    down = np.cross(forward, right)

    # OpenCV camera axes: row0=right, row1=down, row2=forward.
    rotation = np.stack([right, down, forward], axis=0)
    tvec = -rotation @ eye
    rvec, _ = cv2.Rodrigues(rotation)
    return rvec.reshape(3), tvec.reshape(3), eye


def _camera_matrix() -> np.ndarray:
    cx = cy = IMAGE_SIZE / 2.0
    return np.array([[FX, 0, cx], [0, FY, cy], [0, 0, 1]], dtype=np.float64)


def _render_frame(faces, rvec, tvec, cam_center, camera_matrix) -> np.ndarray:
    image = np.full((IMAGE_SIZE, IMAGE_SIZE, 3), 120, dtype=np.uint8)  # neutral gray background

    visible = []
    for normal, corners, texture in faces:
        face_center = corners.mean(axis=0)
        view_dir = face_center - cam_center
        view_dir /= np.linalg.norm(view_dir)
        if np.dot(normal, -view_dir) > 0.05:
            depth = float(np.linalg.norm(face_center - cam_center))
            visible.append((depth, corners, texture))

    # Painter's algorithm, farthest first: visible cube faces never
    # actually overlap on screen when correctly projected, but this is
    # cheap insurance against edge-case projection artifacts.
    visible.sort(key=lambda item: -item[0])

    for _, corners, texture in visible:
        projected, _ = cv2.projectPoints(corners, rvec, tvec, camera_matrix, None)
        projected = projected.reshape(-1, 2).astype(np.float32)

        homography = cv2.getPerspectiveTransform(_TEXTURE_CORNERS_2D, projected)
        warped = cv2.warpPerspective(texture, homography, (IMAGE_SIZE, IMAGE_SIZE))

        mask = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
        cv2.fillConvexPoly(mask, projected.astype(np.int32), 255)
        image[mask > 0] = warped[mask > 0]

    return image


def render_true_depth(faces, rvec, tvec, cam_center, camera_matrix) -> np.ndarray:
    """Exact camera-space depth (millimeters) for the frame `_render_frame`
    would produce from the same faces/pose -- ray-plane intersection
    against each visible face's actual 3D quad, using the same
    visibility test and farthest-first compositing order as
    `_render_frame`, so the two stay pixel-consistent.

    Returns an (IMAGE_SIZE, IMAGE_SIZE) float32 array; 0.0 where no face
    is visible (background). This is the ground truth Phase 2b's
    depth-alignment and TSDF-integration code is tested against, standing
    in for a real monocular depth model: this project's sandbox cannot
    download Depth-Anything-V2's weights (huggingface.co is
    network-policy-blocked, confirmed directly -- see CLAUDE.md), so
    end-to-end dense-reconstruction tests use this exact, known-correct
    depth instead of running the real model.
    """
    rotation, _ = cv2.Rodrigues(rvec)
    k_inv = np.linalg.inv(camera_matrix)

    ys, xs = np.mgrid[0:IMAGE_SIZE, 0:IMAGE_SIZE]
    pixels_h = np.stack([xs.ravel(), ys.ravel(), np.ones(xs.size)], axis=0).astype(np.float64)
    dirs_cam = k_inv @ pixels_h  # (3, N): each column is (X, Y, 1) in camera space
    dirs_world = (rotation.T @ dirs_cam).T  # (N, 3)

    depth = np.zeros(IMAGE_SIZE * IMAGE_SIZE, dtype=np.float64)

    visible = []
    for normal, corners, _texture in faces:
        face_center = corners.mean(axis=0)
        view_dir = face_center - cam_center
        view_dir = view_dir / np.linalg.norm(view_dir)
        if np.dot(normal, -view_dir) > 0.05:
            face_depth = float(np.linalg.norm(face_center - cam_center))
            visible.append((face_depth, normal, corners))
    # Farthest first: nearer faces overwrite in the loop below, matching
    # _render_frame's painter's-algorithm compositing.
    visible.sort(key=lambda item: -item[0])

    for _, normal, corners in visible:
        center = corners.mean(axis=0)
        # Recover the u/v axes used to build `corners` from _CORNER_UV's
        # winding (see _build_faces): corners[1]-corners[0] = 2*u_axis*_HALF,
        # corners[3]-corners[0] = 2*v_axis*_HALF.
        u_axis = (corners[1] - corners[0]) / (2 * _HALF)
        v_axis = (corners[3] - corners[0]) / (2 * _HALF)

        denom = dirs_world @ normal  # (N,)
        numerator = np.dot(corners[0] - cam_center, normal)
        with np.errstate(divide="ignore", invalid="ignore"):
            s = numerator / denom
        valid = (np.abs(denom) > 1e-9) & (s > 0)

        hit_points = cam_center[None, :] + s[:, None] * dirs_world  # (N, 3)
        rel = hit_points - center[None, :]
        u_coord = (rel @ u_axis) / _HALF
        v_coord = (rel @ v_axis) / _HALF
        in_quad = valid & (np.abs(u_coord) <= 1.0) & (np.abs(v_coord) <= 1.0)

        depth[in_quad] = s[in_quad]

    return depth.reshape(IMAGE_SIZE, IMAGE_SIZE).astype(np.float32)


def generate() -> None:
    faces = _build_faces()
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
        frame = _render_frame(faces, rvec, tvec, cam_center, camera_matrix)

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
