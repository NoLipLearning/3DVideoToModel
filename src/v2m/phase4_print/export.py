"""Final export: binary STL (mm) + OBJ + GLB.

docs/ARCHITECTURE.md's directory tree: `output/model.stl, model.obj,
model.glb, print_report.json`. STL is the slicer-ready deliverable
(confirmed directly: `Trimesh.export(path)` with a `.stl` extension
defaults to binary, not ASCII); OBJ and GLB are secondary preview/
interchange formats.
"""

from __future__ import annotations

from pathlib import Path

import trimesh

_EXPORT_FORMATS = {"stl": "model.stl", "obj": "model.obj", "glb": "model.glb"}


def export_mesh(mesh: trimesh.Trimesh, output_dir: Path) -> dict[str, str]:
    """Writes model.{stl,obj,glb} to `output_dir`. Returns {format:
    filename} (relative to output_dir) for the caller to record as
    manifest artifacts.

    OBJ export embeds any per-vertex color directly in the file's `v`
    lines -- trimesh's default behavior for a vertex-colored, non-UV-
    mapped mesh (this pipeline's typical output, carried through from
    Phase 2b's RGBD integration) -- rather than writing a companion
    `.mtl`, which trimesh only emits for an actual UV-mapped texture.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    for file_type, filename in _EXPORT_FORMATS.items():
        mesh.export(str(output_dir / filename), file_type=file_type)
    return dict(_EXPORT_FORMATS)
