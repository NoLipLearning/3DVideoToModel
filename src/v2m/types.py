"""Shared data types passed between phases and persisted to disk.

Every type here is a Pydantic model so it round-trips through
`model_dump_json` / `model_validate_json` cleanly -- these are exactly the
shapes written into `manifest.json`, `frames.json`, and `print_report.json`.

The *behavior* around these types (creating run directories, updating
phase status, deciding when a run is resumable) lives in `run_context.py`,
not here -- this module is data only, per docs/ARCHITECTURE.md's directory
structure.

`FrameRecord`, `SfmResult`, `DenseResult`, and `MeshReport` are scaffolded
now but populated starting at M1/M2/M3/M4 respectively; `PrintReport` at
M5. Field shapes may grow as those milestones land, but should not need to
shrink -- they already reflect the metrics named in docs/ARCHITECTURE.md
Sections 3 and 4 (registration rate, reprojection error, track length,
watertightness, volume, bbox, repair rung, scale method).
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class PhaseName(StrEnum):
    """The five phases a run's manifest tracks, in execution order."""

    INGEST = "ingest"
    SFM_SPARSE = "sfm_sparse"
    SFM_DENSE = "sfm_dense"
    MESH = "mesh"
    PRINT_PREP = "print_prep"


class PhaseStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETE = "complete"
    FAILED = "failed"
    SKIPPED = "skipped"


class PhaseRecord(BaseModel):
    """One phase's entry in a run manifest."""

    status: PhaseStatus = PhaseStatus.PENDING
    input_hash: str | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None
    duration_s: float | None = None
    artifacts: dict[str, str] = Field(default_factory=dict)
    error: str | None = None
    remedy: str | None = None
    # The phase's own result model (IngestSummary, SfmResult, ...) as
    # plain JSON, so report/html.py and the web UI can show every phase's
    # numbers from the manifest alone. Added at M6; older manifests
    # without it load with an empty dict.
    summary: dict[str, Any] = Field(default_factory=dict)


class RunManifest(BaseModel):
    """The full state of one run, persisted at `<run_dir>/manifest.json`.

    `config` is a snapshot (via `PipelineConfig.model_dump()`) of the
    resolved configuration at run creation time -- not a live reference --
    so a later change to configs/*.yaml never silently changes what a
    `--resume` does.
    """

    run_id: str
    created_at: datetime
    source_video: str | None = None
    # Set instead of source_video for a run built from a folder of frames
    # (`v2m run --from-frames`, e.g. a `v2m capture` folder). Added at M8.
    source_frames: str | None = None
    preset: str
    config: dict[str, Any]
    phases: dict[PhaseName, PhaseRecord] = Field(
        default_factory=lambda: {phase: PhaseRecord() for phase in PhaseName}
    )
    # Phase 4's per-capture scale inputs (--scale-factor, --scale-points,
    # --aruco-*), kept so a --resume that reaches Phase 4 uses the same
    # scale reference the run was started with. See pipeline.PrintOptions.
    print_options: dict[str, Any] = Field(default_factory=dict)


class FrameRecord(BaseModel):
    """One candidate frame from Phase 1. Populated starting at M1.

    `path` is the filename inside `frames/` (e.g. "000000.jpg") for an
    accepted frame -- an empty string for a rejected one, which is never
    written to disk.
    """

    path: str
    frame_index: int
    timestamp_s: float
    laplacian_var: float
    accepted: bool
    reject_reason: str | None = None


class IngestSummary(BaseModel):
    """Returned by `phase1_ingest.extract.run_extract()`; the CLI attaches
    it to the manifest's INGEST phase artifacts. Introduced at M1.
    """

    total_frames: int
    accepted: int
    rejected_blur: int
    rejected_redundant: int
    rejected_budget: int
    blur_threshold: float
    frames_json_path: str


class SfmResult(BaseModel):
    """Sparse reconstruction summary. Populated starting at M2."""

    num_images_registered: int
    num_images_total: int
    mean_reprojection_error_px: float
    mean_track_length: float
    sparse_points_path: str
    cameras_path: str


class LowKeypointImage(BaseModel):
    """One image flagged by diagnostics for too little texture to key off
    of (docs/ARCHITECTURE.md Section 3.1)."""

    name: str
    keypoint_count: int


class SfmDiagnostics(BaseModel):
    """Written to `sfm/diagnostics.json` on every sparse-SfM attempt,
    success or failure -- docs/ARCHITECTURE.md Section 4 (M2): "Diagnostics
    JSON written even on failure." `attempt` is 1 for the initial pass,
    2 for the Section 3.1 retry (lowered SIFT peak_threshold), and 3 for
    the M9 learned-feature fallback (DISK + LightGlue).
    """

    num_images_total: int
    num_images_registered: int
    registration_rate: float
    attempt: int = 1
    mean_reprojection_error_px: float | None = None
    mean_track_length: float | None = None
    median_triangulation_angle_deg: float | None = None
    low_keypoint_images: list[LowKeypointImage] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class DenseResult(BaseModel):
    """Dense point cloud summary. Populated starting at M3."""

    backend: str
    num_points: int
    dense_points_path: str
    alignment_rmse: float | None = None


class MeshReport(BaseModel):
    """Raw surface mesh summary. Populated starting at M4."""

    num_vertices: int
    num_faces: int
    is_manifold: bool
    mesh_path: str


class PrintReport(BaseModel):
    """Final print-readiness summary. Populated starting at M5."""

    watertight: bool
    volume_mm3: float
    bbox_mm: tuple[float, float, float]
    repair_rung_used: int
    scale_method: str
    warnings: list[str] = Field(default_factory=list)
