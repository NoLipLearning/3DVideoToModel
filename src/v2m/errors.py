"""Error hierarchy for v2m.

Every `V2MError` carries a short, user-facing `remedy` in addition to the
technical message, so CLI failures tell the operator what to actually do
next ("re-shoot walking slower") instead of just a stack trace. See
docs/ARCHITECTURE.md, "Notes for the Implementing Model", point 6.
"""

from __future__ import annotations


class V2MError(Exception):
    """Base class for all v2m errors.

    Args:
        message: technical description of what failed.
        remedy: a concrete, user-facing suggestion for how to fix it.
            Optional, but every call site that surfaces to the CLI should
            provide one.
    """

    def __init__(self, message: str, *, remedy: str | None = None) -> None:
        self.message = message
        self.remedy = remedy
        full = message if remedy is None else f"{message}\n  → {remedy}"
        super().__init__(full)


class CapabilityError(V2MError):
    """A required capability (library, binary, hardware feature) is missing."""


class ConfigError(V2MError):
    """Configuration is missing, malformed, or internally inconsistent."""


class IngestError(V2MError):
    """Phase 1: video reading, quality filtering, or frame selection failed."""


class SfMError(V2MError):
    """Phase 2: structure-from-motion or dense reconstruction failed."""


class MeshError(V2MError):
    """Phase 3: surface meshing failed."""


class PrintPrepError(V2MError):
    """Phase 4: watertight repair, scaling, or export failed."""


class ResumeError(V2MError):
    """A --resume was requested but the run manifest is missing or inconsistent."""
