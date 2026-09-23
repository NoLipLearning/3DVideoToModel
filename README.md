# 3DVideoToModel (v2m)

Turn a video of an object or a scene into a manifold, watertight,
print-ready 3D model (STL/OBJ) — no custom reconstruction math, just
careful orchestration of COLMAP, a monocular depth model, Open3D, and
Trimesh.

See [`CLAUDE.md`](CLAUDE.md) for build/test commands, the module map, and
the constraints that shape this codebase (most importantly: **no CUDA on
the target platform, ever**). The full design document — library
justification, edge-case handling, milestone-by-milestone execution plan —
lives in [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

## Status

M0–M6 are complete: the whole video → STL pipeline runs end to end, is
resumable, and writes a per-run HTML report. See `CLAUDE.md` for the
milestone checklist.

## Quickstart

```bash
uv sync
uv run v2m doctor
```

`doctor` reports what's actually installed and working on this machine —
including whether COLMAP's dense-reconstruction stage is available (it
isn't, on Apple Silicon, by design; `doctor` explains why and what backend
is used instead).

```bash
uv run v2m presets   # list config presets: default, object, scene, fast
uv run pytest        # unit tests
uv run ruff check .  # lint
```

## Usage

```bash
uv run v2m run clip.mov --preset object      # video in -> runs/<id>/output/model.stl
uv run v2m run clip.mov --preset fast        # quick preview (sparse-only, low res)
uv run v2m run clip.mov --aruco-image 000012.jpg --aruco-marker-mm 100   # real-world scale
```

Every run lives in `runs/<timestamp>_<id>/` with a `manifest.json` and a
self-contained `report.html` (preview, per-phase numbers, and what to
re-shoot if something went wrong). Runs are resumable:

```bash
uv run v2m run --resume runs/<id>                                   # continue after a crash / Ctrl-C
uv run v2m run --resume runs/<id> --set print_prep.slab_thickness_mm=5   # re-runs only Phase 4
uv run v2m run --resume runs/<id> --rerun-from mesh                  # redo Phase 3 onward
```

Each phase can also be run on its own: `v2m extract`, `sfm`, `dense`,
`mesh`, `printprep`.
