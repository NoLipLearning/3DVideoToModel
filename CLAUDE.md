# CLAUDE.md

## What this is

v2m ("3DVideoToModel") ingests a video of an object or a scene and outputs
a manifold, watertight, metric-scaled 3D mesh (STL/OBJ) ready for a
slicer. It orchestrates mature libraries (OpenCV, COLMAP/pycolmap, a
monocular depth model, Open3D, Trimesh) rather than implementing
reconstruction math from scratch. Full architecture, library
justification, and edge-case handling:
**[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)** — read that before
touching Phase 2 (SfM/dense) or Phase 4 (print-prep) code.

## The one constraint that overrides everything else

**Target platform is Apple Silicon. There is no CUDA, ever, on this
target.** COLMAP's dense-reconstruction stage (`patch_match_stereo`) is
CUDA-only and does not exist in any macOS build. If you find yourself
about to call it, stop — you're on the wrong path. The default dense
backend is `monodepth_tsdf` (Depth Anything V2 + TSDF fusion,
MPS-accelerated); see `docs/ARCHITECTURE.md` Section 2 for why, and
`src/v2m/capability.py` for how the pipeline decides this at runtime.

## Build & test commands

```bash
uv sync                    # install all dependencies (torch resolves from default PyPI -- see the comment in pyproject.toml [tool.uv] for the Linux/Windows CPU-only opt-in)
uv run v2m doctor          # capability probe: what's installed, what backend will be used, why
uv run v2m presets         # list config presets (default, object, scene, fast)
uv run pytest              # unit tests; slow/integration tests are marked `slow` and skip without colmap
uv run pytest -m "not slow"
uv run ruff check .        # lint
uv run ruff format .       # format
```

There is no build step beyond `uv sync` — this is a pure-Python CLI (plus
a FastAPI app from M7 onward), not a compiled package.

**Linux dev note (not needed on macOS):** open3d's compiled extension
dynamically links `libEGL.so.1` and `libusb-1.0.so.0` (EGL/OpenGL and
RealSense support respectively). A minimal/headless Linux container
usually doesn't have these and `import open3d` fails with an
`ImportError` naming the missing `.so` — `capability.py` catches this and
reports open3d as "absent" rather than crashing, but it will make
`v2m doctor` exit non-zero. Fix: `apt-get install -y libegl1 libgl1
libgomp1 libusb-1.0-0`. macOS wheels don't have this dependency at all.

## Module map

```
src/v2m/
  cli.py            Typer entrypoint -- one subcommand per phase, plus `run` (M6), `serve` (M7), `capture` (M8)
  config.py         Pydantic config models + configs/*.yaml loading + dot-path CLI overrides
  capability.py     Hardware/binary/library probe -- the ONLY place that decides "is CUDA available"
  run_context.py    RunContext: run directory layout, manifest.json read/write, --resume logic
  types.py          Shared Pydantic models: RunManifest, PhaseRecord, FrameRecord, SfmResult, DenseResult, MeshReport, PrintReport
  errors.py         V2MError hierarchy -- every error carries a user-facing remedy, not just a message
  logging_setup.py  Rich console logging + per-run JSONL log
  phase1_ingest/    Video -> filtered frame set (blur/redundancy gating)
  phase2_sfm/       Sparse SfM (COLMAP) + dense reconstruction (backends/ + dense/)
  phase3_mesh/      Point cloud -> raw surface mesh (Poisson)
  phase4_print/     Surface -> watertight, scaled, printable solid (THE hard part -- see docs/ARCHITECTURE.md Section 3.3)
  capture/          Guided live-capture HUD (M8) -- feeds phase1, not a SLAM system
  web/              FastAPI local UI (M7)
  report/           Per-run HTML report
```

## Working conventions

- **One milestone per session.** Read only the files that milestone
  touches plus this file; don't re-read the whole repo.
  `docs/ARCHITECTURE.md` Section 4 has the milestone list and
  per-milestone verification steps.
- **Every tunable default already has a value.** They're declared in
  `configs/default.yaml` with citations back to the architecture doc. Use
  them; don't re-derive defaults from scratch.
- **`pymeshlab` is optional.** It's the heaviest, least portable
  dependency in the stack. Guard every import (`capability.py` reports it
  as present/absent) and always provide an open3d/trimesh fallback path.
- **Write a manifest entry at the end of every phase**, via
  `RunContext.complete_phase()` / `.fail_phase()`. Resumability is not
  optional here — a 40-minute pipeline is unworkable to iterate on
  without it.
- **Raise `V2MError` subclasses with a `remedy=`**, not bare exceptions,
  for anything a user needs to act on (bad footage, missing capability,
  stale resume state).
- **Never hand-triangulate hole-filling for the print-ready base.**
  Boolean half-space cut via `manifold3d` (Section 3.3) guarantees
  manifold output; ad-hoc hole-filling is where these pipelines usually
  break.
- **Verify pycolmap's API against the actually-installed version before
  writing code against it.** The modern `colmap/colmap/python` bindings
  (what `pip install pycolmap` gives you now) are a different, cleaner
  API than the deprecated `colmap/pycolmap` package's — top-level
  functions like `extract_features`/`match_exhaustive`/
  `incremental_mapping`, `Database.open(path)` as a classmethod, methods
  like `image.cam_from_world()` that look like properties but aren't.
  Training-data recall of "pycolmap" is likely to be the old API. `import
  pycolmap; help(pycolmap.the_thing)` first.
- **COLMAP's vocab-tree loop detection needs a file this sandbox cannot
  download** (`demuc.de` is network-policy-blocked, confirmed directly).
  It degrades to sequential-only matching with a logged warning; set
  `V2M_VOCAB_TREE_PATH` to enable it for real on a machine that can reach
  COLMAP's site.

## Milestone status

- [x] M0 — scaffold, config, capability probe, `v2m doctor`
- [x] M1 — Phase 1: intelligent ingest
- [x] M2 — Phase 2a: sparse SfM
- [ ] M3 — Phase 2b: dense point cloud (monodepth + TSDF)
- [ ] M4 — Phase 3: raw mesh (Poisson)
- [ ] M5 — Phase 4: print-ready post-processing (the critical milestone)
- [ ] M6 — end-to-end `v2m run` + `--resume`
- [ ] M7 — local web UI
- [ ] M8 — guided live capture
- [ ] M9 — optional quality backends (OpenMVS, hloc/ALIKED, tiling)

Update the checkbox when a milestone's verification step (in
`docs/ARCHITECTURE.md`) actually passes — not just when the code is
written.
