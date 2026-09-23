"""Learned-feature SfM backend (M9): DISK or ALIKED keypoints + LightGlue
matching, written into COLMAP's database so the *same* incremental
mapper, undistortion and export in `sfm.py` run on top of it.

docs/ARCHITECTURE.md Section 3.1 #4: "learned features are far better
on low texture". The file keeps the doc's name, `hloc_backend.py`, but
uses kornia's implementations of these models, not the `hloc` package.
hloc isn't on PyPI and pulls in a research-code dependency tree, while
kornia is one pip install (the optional `learned` extra) and ships the
same DISK, ALIKED and LightGlue networks. The recipe is hloc's: extract
per image, match candidate pairs, write keypoints and raw matches into
the database, and let COLMAP's two-view geometry verification decide
which matches survive.

Licensing (Section 3.1's warning): DISK, ALIKED and LightGlue are
permissively licensed. SuperPoint is not offered at all. Its weights are
Magic Leap research/non-commercial, and kornia ships no SuperPoint
extractor, so "opt-in with a warning" would have meant vendoring
non-commercial code into this repo.

Pairing: the frame sequence is treated as *cyclic*. Each frame is
matched against the next `matching_overlap` frames, wrapping from the
last frame back to the first, which closes the loop of an orbit capture
without a vocabulary tree. Pairs that don't overlap simply fail
verification. With a GPU (CUDA/MPS) and at most
`exhaustive_matching_max_images` frames, every pair is matched instead,
the same rule as the SIFT backend. On CPU, LightGlue is ~0.5-1.5s per
pair, so exhaustive matching of 80 frames (3160 pairs) would take an hour.
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np
import pycolmap

from v2m.config import SfmConfig
from v2m.errors import CapabilityError, SfMError

logger = logging.getLogger("v2m.phase2_sfm.backends.hloc_backend")

LEARNED_FEATURES = ("disk", "aliked")


def _require_kornia():
    try:
        import kornia.feature as KF  # noqa: N812 -- kornia's own convention
        import torch
    except ImportError as exc:
        raise CapabilityError(
            "Learned features need kornia, which isn't installed.",
            remedy="uv sync --extra learned  (or set sfm.feature_backend=sift)",
        ) from exc
    return KF, torch


def _device(torch):
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class _Extractor:
    """One network for the whole image set (weights load once)."""

    def __init__(self, name: str, max_keypoints: int, KF, torch, device) -> None:  # noqa: N803
        self.name, self.max_keypoints = name, max_keypoints
        self.torch, self.device = torch, device
        try:
            if name == "disk":
                self.model = KF.DISK.from_pretrained("depth", device=device)
            else:
                self.model = KF.ALIKED.from_pretrained(
                    "aliked-n16", max_num_keypoints=max_keypoints, device=device
                )
        except Exception as exc:  # network / cache failures surface as many types
            raise SfMError(
                f"Could not load the {name} feature network's weights: {exc}",
                remedy="The first use downloads them (to ~/.cache/torch/hub). Check network "
                "access to github.com, or use sfm.feature_backend=sift.",
            ) from exc
        self.model.eval()

    def __call__(self, image_bgr: np.ndarray) -> tuple:
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        tensor = self.torch.from_numpy(rgb).permute(2, 0, 1).float()[None].to(self.device) / 255.0
        with self.torch.inference_mode():
            if self.name == "disk":
                feats = self.model(
                    tensor, n=self.max_keypoints, window_size=5, pad_if_not_divisible=True
                )[0]
            else:
                feats = self.model(tensor)[0]
        return feats.keypoints, feats.descriptors


def _pairs(n: int, config: SfmConfig, exhaustive: bool) -> list[tuple[int, int]]:
    if exhaustive:
        return [(i, j) for i in range(n) for j in range(i + 1, n)]
    window = min(config.matching_overlap, max(1, n // 2))
    pairs = set()
    for i in range(n):
        for step in range(1, window + 1):
            j = (i + step) % n
            if i != j:
                pairs.add((min(i, j), max(i, j)))
    return sorted(pairs)


def extract_and_match(images_dir: Path, database_path: Path, config: SfmConfig) -> None:
    """Same contract as `colmap_backend.extract_and_match`: populate the
    database with keypoints and verified two-view geometries."""
    KF, torch = _require_kornia()  # noqa: N806
    names = sorted(p.name for p in images_dir.glob("*.jpg"))
    if not names:
        return
    device = _device(torch)
    exhaustive = device.type != "cpu" and len(names) <= config.exhaustive_matching_max_images
    logger.info(
        "Learned features: %s + LightGlue on %s, %d images, %s pairing.",
        config.feature_backend,
        device.type,
        len(names),
        "exhaustive" if exhaustive else f"cyclic window {config.matching_overlap}",
    )

    pycolmap.Database.open(database_path).close()  # import_images needs it to exist
    pycolmap.import_images(
        database_path=database_path,
        image_path=images_dir,
        camera_mode=pycolmap.CameraMode.SINGLE,
        image_names=names,
    )
    extractor = _Extractor(config.feature_backend, config.learned_max_keypoints, KF, torch, device)
    matcher = KF.LightGlueMatcher(config.feature_backend).to(device).eval()

    database = pycolmap.Database.open(database_path)
    try:
        image_ids = {image.name: image.image_id for image in database.read_all_images()}
        features = []
        for name in names:
            image = cv2.imread(str(images_dir / name))
            height, width = image.shape[:2]
            scale = min(1.0, config.learned_resize_px / max(height, width))
            small = (
                cv2.resize(image, (round(width * scale), round(height * scale)), cv2.INTER_AREA)
                if scale < 1.0
                else image
            )
            keypoints, descriptors = extractor(small)
            # Back to full-resolution pixels, then COLMAP's convention
            # (pixel centres at +0.5; kornia's are at integer coordinates).
            full_res = keypoints.cpu().numpy() / scale + 0.5
            database.write_keypoints(image_ids[name], full_res.astype(np.float32))
            lafs = KF.laf_from_center_scale_ori(
                keypoints[None], torch.ones(1, len(keypoints), 1, 1, device=device)
            )
            features.append((descriptors, lafs, small.shape[:2]))

        pair_indices = _pairs(len(names), config, exhaustive)
        for i, j in pair_indices:
            desc_i, lafs_i, hw_i = features[i]
            desc_j, lafs_j, hw_j = features[j]
            with torch.inference_mode():
                _, matches = matcher(desc_i, desc_j, lafs_i, lafs_j, hw1=hw_i, hw2=hw_j)
            if len(matches):
                database.write_matches(
                    image_ids[names[i]],
                    image_ids[names[j]],
                    matches.cpu().numpy().astype(np.uint32),
                )
    finally:
        database.close()

    pairs_path = database_path.with_name("learned_pairs.txt")
    pairs_path.write_text("".join(f"{names[i]} {names[j]}\n" for i, j in pair_indices))
    pycolmap.verify_matches(database_path=database_path, pairs_path=pairs_path)
