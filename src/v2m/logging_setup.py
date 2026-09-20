"""Console + per-run JSONL logging.

Console output goes through `rich` for a readable CLI experience. Every
run additionally gets a flat JSONL log at `<run_dir>/run.log.jsonl` so a
failure can be inspected without re-running the pipeline, and so
`report/html.py` (M6) can pull events back out for the per-run report.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path

from rich.logging import RichHandler


class _JsonlHandler(logging.Handler):
    """Appends one JSON object per line to a run's log file."""

    def __init__(self, path: Path) -> None:
        super().__init__()
        self.path = path

    def emit(self, record: logging.LogRecord) -> None:
        entry: dict[str, str] = {
            "ts": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            entry["exc_info"] = self.formatException(record.exc_info)
        with self.path.open("a") as f:
            f.write(json.dumps(entry) + "\n")


def setup_logging(*, verbose: bool = False, run_log_path: Path | None = None) -> logging.Logger:
    """Configure the `v2m` logger.

    Safe to call more than once per process (e.g. once per run): handlers
    are replaced, not stacked.
    """
    logger = logging.getLogger("v2m")
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    console_handler = RichHandler(rich_tracebacks=True, show_path=False)
    console_handler.setLevel(logging.DEBUG if verbose else logging.INFO)
    logger.addHandler(console_handler)

    if run_log_path is not None:
        run_log_path.parent.mkdir(parents=True, exist_ok=True)
        jsonl_handler = _JsonlHandler(run_log_path)
        jsonl_handler.setLevel(logging.DEBUG)
        logger.addHandler(jsonl_handler)

    return logger
