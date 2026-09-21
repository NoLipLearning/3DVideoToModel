"""colmap_backend.py: vocab-tree discovery and matching-strategy dispatch."""

import pycolmap
import pytest

from v2m.config import SfmConfig
from v2m.phase2_sfm.backends import colmap_backend


def test_find_vocab_tree_returns_none_without_env_var(monkeypatch):
    monkeypatch.delenv("V2M_VOCAB_TREE_PATH", raising=False)
    assert colmap_backend._find_vocab_tree() is None


def test_find_vocab_tree_returns_none_for_nonexistent_file(monkeypatch, tmp_path):
    monkeypatch.setenv("V2M_VOCAB_TREE_PATH", str(tmp_path / "does_not_exist.bin"))
    assert colmap_backend._find_vocab_tree() is None


def test_find_vocab_tree_returns_path_when_file_exists(monkeypatch, tmp_path):
    vocab_file = tmp_path / "vocab.bin"
    vocab_file.write_bytes(b"not a real vocab tree, just needs to exist")
    monkeypatch.setenv("V2M_VOCAB_TREE_PATH", str(vocab_file))
    assert colmap_backend._find_vocab_tree() == vocab_file


def test_extract_and_match_no_op_on_empty_directory(tmp_path):
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    database_path = tmp_path / "database.db"
    # Should return cleanly without creating a database or raising --
    # sfm.py owns turning "nothing to reconstruct" into a proper SfMError.
    colmap_backend.extract_and_match(empty_dir, database_path, SfmConfig())
    assert not database_path.exists()


@pytest.mark.slow
def test_extract_and_match_populates_database_exhaustive(sfm_fixture, tmp_path):
    database_path = tmp_path / "database.db"
    # Default exhaustive_matching_max_images (80) exceeds this fixture's
    # image count -> exhaustive matching.
    colmap_backend.extract_and_match(sfm_fixture["images_dir"], database_path, SfmConfig())
    with pycolmap.Database.open(database_path) as db:
        assert db.num_images() == len(list(sfm_fixture["images_dir"].glob("*.jpg")))
        assert db.num_keypoints() > 0
        assert db.num_verified_image_pairs() > 0


@pytest.mark.slow
def test_extract_and_match_sequential_path_degrades_without_vocab_tree(
    sfm_fixture, tmp_path, monkeypatch, caplog
):
    monkeypatch.delenv("V2M_VOCAB_TREE_PATH", raising=False)
    database_path = tmp_path / "database.db"
    # Force the sequential+loop_detection path regardless of image count,
    # to exercise the graceful "no vocab tree available" fallback --
    # this sandbox's network policy blocks COLMAP's own vocab-tree hosts
    # (see colmap_backend.py's module docstring).
    config = SfmConfig(exhaustive_matching_max_images=0, loop_detection=True)
    with caplog.at_level("WARNING"):
        colmap_backend.extract_and_match(sfm_fixture["images_dir"], database_path, config)

    assert "vocabulary tree" in caplog.text
    with pycolmap.Database.open(database_path) as db:
        assert db.num_verified_image_pairs() > 0
