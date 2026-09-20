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

Milestone M0 (scaffold + capability probe) is complete. Nothing beyond
`v2m doctor` and `v2m presets` runs yet — see `CLAUDE.md` for the
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
