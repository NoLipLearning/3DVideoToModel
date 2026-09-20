"""Config loading: presets merge over defaults, dot-path overrides apply."""

import pytest

from v2m.config import list_presets, load_config
from v2m.errors import ConfigError


def test_default_preset_loads():
    cfg = load_config("default")
    assert cfg.mode == "object"
    assert cfg.ingest.frame_budget == 150


def test_object_preset_matches_default_shape():
    cfg = load_config("object")
    assert cfg.mode == "object"
    assert cfg.ingest.frame_budget == 150


def test_scene_preset_overrides_expected_fields():
    cfg = load_config("scene")
    assert cfg.mode == "scene"
    assert cfg.ingest.frame_budget == 300
    assert cfg.sfm.use_clahe is True
    assert cfg.dense.tsdf_voxel_size_m == 0.02
    assert cfg.dense.tsdf_sdf_trunc_m == 0.06


def test_fast_preset_uses_sparse_only_and_lower_resolution():
    cfg = load_config("fast")
    assert cfg.dense.backend == "sparse_only"
    assert cfg.ingest.resolution_cap_px == 800
    assert cfg.mesh.max_faces == 50_000


def test_unknown_preset_raises_configerror_with_remedy():
    with pytest.raises(ConfigError) as exc_info:
        load_config("does-not-exist")
    assert exc_info.value.remedy is not None


def test_dot_path_override_applies():
    cfg = load_config("default", overrides={"sfm.registration_rate_min": 0.42})
    assert cfg.sfm.registration_rate_min == 0.42
    # sibling fields must survive the merge untouched
    assert cfg.sfm.mean_track_length_min == 3.0


def test_list_presets_finds_all_four():
    names = list_presets()
    assert {"default", "object", "scene", "fast"}.issubset(set(names))
