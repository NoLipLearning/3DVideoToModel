"""Pipeline configuration: Pydantic models + YAML loading + CLI overrides.

Resolution order (later wins):
  1. Field defaults declared on the models below (belt-and-suspenders --
     the pipeline still works even if `configs/` can't be found).
  2. `configs/default.yaml`
  3. `configs/<preset>.yaml`, when preset != "default"
  4. an explicit `overrides` dict passed by the CLI, using dot-path keys
     like `{"sfm.registration_rate_min": 0.6}`

Every numeric default here matches Sections 2 and 3 of
docs/ARCHITECTURE.md -- do not silently change one without updating the
corresponding comment in configs/default.yaml.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field

from v2m.errors import ConfigError


def _repo_config_dir() -> Path:
    env = os.environ.get("V2M_CONFIG_DIR")
    if env:
        return Path(env)
    # src/v2m/config.py -> src/v2m -> src -> <repo root> -> configs/
    return Path(__file__).resolve().parents[2] / "configs"


CONFIG_DIR = _repo_config_dir()


class IngestConfig(BaseModel):
    frame_budget: int = 150
    resolution_cap_px: int = 1600
    blur_min_threshold: float = 30.0
    blur_adaptive_factor: float = 0.6
    redundancy_overlap_max: float = 0.75
    use_pyav: bool = True


class SfmConfig(BaseModel):
    sift_peak_threshold: float = 0.0066
    sift_peak_threshold_retry: float = 0.002
    sift_max_num_features: int = 8192
    min_keypoints_per_image: int = 600
    matching_overlap: int = 10
    loop_detection: bool = True
    exhaustive_matching_max_images: int = 80
    registration_rate_min: float = 0.7
    mean_track_length_min: float = 3.0
    use_clahe: bool = False
    clahe_clip_limit: float = 2.0
    clahe_tile_grid: int = 8
    min_triangulation_angle_deg: float = 2.0


class DenseConfig(BaseModel):
    backend: Literal["auto", "monodepth_tsdf", "openmvs", "sparse_only"] = "auto"
    depth_model: str = "depth-anything/Depth-Anything-V2-Small-hf"
    min_alignment_correspondences: int = 30
    tsdf_voxel_size_m: float = 0.004
    tsdf_sdf_trunc_m: float = 0.02
    poisson_point_threshold: int = 2_000_000
    poisson_depth_low: int = 10
    poisson_depth_high: int = 11
    scene_tile_threshold_m: float = 30.0
    openmvs_cuda_device: int = -2


class MeshConfig(BaseModel):
    poisson_density_quantile_crop: float = 0.03
    max_faces: int = 300_000
    min_component_volume_ratio: float = 0.01


class PrintPrepConfig(BaseModel):
    ground_ransac_distance_multiplier: float = 2.0
    ground_ransac_n: int = 3
    ground_ransac_iterations: int = 1000
    ground_min_inlier_ratio: float = 0.15
    ground_min_gravity_agreement: float = 0.7
    min_wall_thickness_mm: float = 0.8
    slab_thickness_mm: float = 3.0
    max_hole_size_triangles: int = 1000
    scale_target_size_mm: float = 150.0
    build_volume_mm: tuple[float, float, float] = (220.0, 220.0, 250.0)


class PipelineConfig(BaseModel):
    preset: str = "default"
    mode: Literal["object", "scene"] = "object"
    ingest: IngestConfig = Field(default_factory=IngestConfig)
    sfm: SfmConfig = Field(default_factory=SfmConfig)
    dense: DenseConfig = Field(default_factory=DenseConfig)
    mesh: MeshConfig = Field(default_factory=MeshConfig)
    print_prep: PrintPrepConfig = Field(default_factory=PrintPrepConfig)


def list_presets() -> list[str]:
    """Names of every `configs/*.yaml` file, sorted, e.g. for `v2m presets`."""
    if not CONFIG_DIR.exists():
        return []
    return sorted(p.stem for p in CONFIG_DIR.glob("*.yaml"))


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open() as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ConfigError(
            f"{path} does not contain a YAML mapping at the top level.",
            remedy=f"Check the YAML syntax in {path} -- it must be a set of key: value pairs.",
        )
    return data


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _set_dot_path(data: dict[str, Any], dotted_key: str, value: Any) -> None:
    parts = dotted_key.split(".")
    node = data
    for part in parts[:-1]:
        node = node.setdefault(part, {})
    node[parts[-1]] = value


def load_config(preset: str = "default", overrides: dict[str, Any] | None = None) -> PipelineConfig:
    """Load `preset` merged over `default.yaml`, then apply dot-path `overrides`."""
    merged = _load_yaml(CONFIG_DIR / "default.yaml")

    if preset != "default":
        preset_path = CONFIG_DIR / f"{preset}.yaml"
        if not preset_path.exists():
            available = ", ".join(list_presets()) or "(none found)"
            raise ConfigError(
                f"Unknown preset '{preset}'.",
                remedy=f"Available presets: {available}. Run `v2m presets` to list them.",
            )
        merged = _deep_merge(merged, _load_yaml(preset_path))

    merged.setdefault("preset", preset)

    if overrides:
        for dotted_key, value in overrides.items():
            _set_dot_path(merged, dotted_key, value)

    return PipelineConfig.model_validate(merged)
