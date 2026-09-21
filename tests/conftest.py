"""Shared pytest fixtures."""

from pathlib import Path

import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def sfm_fixture(tmp_path_factory) -> dict:
    """Runs Phase 1 + Phase 2a (default config) once for the whole test
    session against tests/fixtures/sample.mp4.

    Several M2 test modules (test_sfm.py, test_diagnostics.py,
    test_colmap_backend.py) assert against this single, expensive
    (~20-30s) real-COLMAP result rather than each re-running the pipeline
    -- consistent with marking every test that consumes it `slow`
    (pytest.ini: "integration tests requiring colmap/heavy deps").
    """
    from v2m.config import IngestConfig, SfmConfig
    from v2m.phase1_ingest.extract import run_extract
    from v2m.phase2_sfm.sfm import run_sparse_sfm

    run_dir = tmp_path_factory.mktemp("sfm_fixture")
    run_extract(FIXTURES_DIR / "sample.mp4", run_dir, IngestConfig())
    result = run_sparse_sfm(run_dir / "frames", run_dir / "sfm", SfmConfig())
    return {"run_dir": run_dir, "images_dir": run_dir / "frames", "result": result}
