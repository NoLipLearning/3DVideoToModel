"""COLMAP SfM backend: SIFT feature extraction + matching.

Verified against the actually-installed pycolmap==4.2.0 API (the modern
`colmap/colmap/python` bindings -- not the deprecated `colmap/pycolmap`
package; its function surface changed substantially, so nothing here was
guessed from older documentation or training-data recall).

Matching strategy (docs/ARCHITECTURE.md Section 3.2 and 4/M2):
  - `num_images <= config.exhaustive_matching_max_images` (80): exhaustive
    matching. At this scale it is cheap and gives the incremental mapper
    the widest possible choice of a good, well-triangulated seed pair --
    important in practice: sequential-only matching with a narrow overlap
    window starved the mapper of any wide-baseline pair and produced a
    2-image reconstruction against this project's own synthetic fixture.
  - Above that: sequential matching with `config.matching_overlap`, plus
    vocab-tree loop detection when `config.loop_detection` is true AND a
    vocabulary tree file is actually available. COLMAP's own vocab-tree
    hosts (e.g. demuc.de) are blocked by this project's sandboxed network
    policy, confirmed via a direct reachability check, so loop detection
    degrades to plain sequential matching with a logged warning rather
    than failing -- exactly the "optional, degrades with a log line"
    posture used elsewhere in this codebase (pymeshlab, OpenMVS).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import pycolmap

from v2m.config import SfmConfig

logger = logging.getLogger("v2m.phase2_sfm.backends.colmap_backend")


def _find_vocab_tree() -> Path | None:
    """A user-supplied vocabulary tree file, if one has been configured.

    COLMAP does not ship one and this project cannot download one in a
    network-restricted environment (see module docstring) -- set
    V2M_VOCAB_TREE_PATH to a real .bin file (from
    https://demuc.de/colmap/, fetched wherever your network allows it) to
    enable loop detection.
    """
    env_path = os.environ.get("V2M_VOCAB_TREE_PATH")
    if env_path and Path(env_path).is_file():
        return Path(env_path)
    return None


def _build_extraction_options(config: SfmConfig) -> pycolmap.FeatureExtractionOptions:
    options = pycolmap.FeatureExtractionOptions()
    options.sift.peak_threshold = config.sift_peak_threshold
    options.sift.max_num_features = config.sift_max_num_features
    return options


def extract_and_match(images_dir: Path, database_path: Path, config: SfmConfig) -> None:
    """Extract SIFT features and run matching into `database_path`.

    `images_dir` must contain only the images to reconstruct -- COLMAP
    scans it by name, and a stray non-image file (frames.json alongside
    the JPEGs) is misread as a corrupt bitmap otherwise. Callers pass an
    explicit `image_names` list rather than relying on directory-only
    scanning, precisely to avoid that.
    """
    image_names = sorted(p.name for p in images_dir.glob("*.jpg"))
    if not image_names:
        # sfm.py raises a proper SfMError with a remedy; this backend
        # only owns the COLMAP calls themselves.
        return

    pycolmap.extract_features(
        database_path=database_path,
        image_path=images_dir,
        image_names=image_names,
        camera_mode=pycolmap.CameraMode.SINGLE,
        extraction_options=_build_extraction_options(config),
    )

    if len(image_names) <= config.exhaustive_matching_max_images:
        pycolmap.match_exhaustive(database_path=database_path)
        return

    pairing_options = pycolmap.SequentialPairingOptions(overlap=config.matching_overlap)
    if config.loop_detection:
        vocab_tree_path = _find_vocab_tree()
        if vocab_tree_path is not None:
            pairing_options.loop_detection = True
            pairing_options.vocab_tree_path = str(vocab_tree_path)
        else:
            logger.warning(
                "loop_detection is enabled but no vocabulary tree file is configured "
                "(set V2M_VOCAB_TREE_PATH) -- falling back to sequential-only matching. "
                "Loop closure on a return-to-start capture will be weaker without it."
            )
    pycolmap.match_sequential(database_path=database_path, pairing_options=pairing_options)
