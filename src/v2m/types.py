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
    preset: str
    config: dict[str, Any]
    phases: dict[PhaseName, PhaseRecord] = Field(
        default_factory=lambda: {phase: PhaseRecord() for phase in PhaseName}
    )


class FrameRecord(BaseModel):
    """One candidate frame from Phase 1. Populated starting at M1."""

    path: str
    frame_index: int
    timestamp_s: float
    laplacian_var: float
    accepted: bool
    reject_reason: str | None = None


class SfmResult(BaseModel):
    """Sparse reconstruction summary. Populated starting at M2."""

    num_images_registered: int
    num_images_total: int
    mean_reprojection_error_px: float
    mean_track_length: float
    sparse_points_path: str
    cameras_path: str


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
