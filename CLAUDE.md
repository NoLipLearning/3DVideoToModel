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
  pipeline.py       run_phase() (the one place phase bookkeeping happens) + run_pipeline()/prepare_resume() (M6)
  preflight.py      RAM/disk estimate before a run; refuses what can't fit (M6)
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
  report/           Per-run self-contained report.html (html.py) + software-rendered model preview (preview.py)
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
- **Write a manifest entry at the end of every phase** — in practice,
  run phases through `pipeline.run_phase()`, which does the
  start/complete/fail bookkeeping (plus the result `summary` and failure
  `remedy`) for every caller: per-phase CLI commands, `v2m run`, the web
  worker. Resumability is not optional here — a 40-minute pipeline is
  unworkable to iterate on without it. If you add a config section or a
  phase, update `pipeline._SECTION_FIRST_PHASE` so `--resume --set`
  invalidates the right phases.
- **Raise `V2MError` subclasses with a `remedy=`**, not bare exceptions,
  for anything a user needs to act on (bad footage, missing capability,
  stale resume state).
- **Never hand-triangulate hole-filling for the print-ready base.**
  `phase4_print/watertight.py`'s rung 5 closes it with a voxel ground
  cut (seal a floor layer, flood-fill, clear below the cut, marching
  cubes), which is manifold by construction. Don't reach for a
  `manifold3d` boolean on an *open* mesh: manifold3d requires closed
  input and rejects boundary edges. Fan-filling a large non-convex
  loop first to satisfy it produced garbage (both confirmed directly;
  see `docs/ARCHITECTURE.md` M5).
- **Anything that pairs images with COLMAP keypoints or intrinsics reads
  `sfm/undistorted/sparse/`, not `sfm/sparse/final/`.** The images on
  disk are the undistorted ones, and only that copy of the
  reconstruction has matching keypoints and camera models (poses and 3D
  points are identical in both). The two only coincide when a camera's
  estimated distortion is ~0, which is how this went unnoticed through
  M3.
- **`trimesh` `VoxelGrid.marching_cubes` returns voxel-index
  coordinates.** Always `apply_transform(voxels.transform)` afterwards.
- **Compare reconstructed dimensions rotation-invariantly** (volume,
  oriented bounding box). A model's heading about Z is arbitrary, so
  axis-aligned extents conflate "wrong size" with "turned 45°".
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
- **`open3d.pipelines.integration.ScalableTSDFVolume` is broken in this
  project's dev sandbox** (open3d==0.20.0, Linux CPU wheel) —
  `integrate()` succeeds silently but every extraction method returns
  empty, confirmed three ways including Open3D's own tutorial parameters
  verbatim. `dense/monodepth_tsdf.py`'s `_build_tsdf_volume()` uses
  `UniformTSDFVolume` instead, adaptively sized from the reconstruction's
  own bounding box. If you're on macOS (the actual target) and confirm
  `ScalableTSDFVolume` works there, that one function is the only call
  site that would need to change back.
- **COLMAP's own reconstruction scale is arbitrary, not metric** —
  monocular SfM normalizes the first registered pair's baseline to an
  unknown length; true scale isn't established until Phase 4's
  `scale.py`. Don't apply `DenseConfig`'s `_m`-suffixed fields (or any
  future Phase 2b/3 "physical size" default) as literal metres against a
  raw COLMAP reconstruction — see `_build_tsdf_volume()`'s docstring for
  the ratio-based pattern that stays correct regardless of what scale
  COLMAP actually produced.
- **`huggingface.co` is network-policy-blocked** in this sandbox
  (confirmed directly) — `dense/monodepth_tsdf.py`'s
  `TransformersDepthEstimator` cannot download Depth-Anything-V2's
  weights here, so it is untested end-to-end. Tests inject a
  `DepthEstimator` test double built from known fixture geometry instead
  (`tests/conftest.py`'s session-scoped `dense_fixture` /
  `GeometricDepthEstimator`). Verify the real model on a machine that can
  reach huggingface.co before trusting it blindly.
- **Neither dense backend's output carries a per-point "which camera saw
  this" tag.** `phase3_mesh/normals.py`'s "orient toward the observing
  camera" (Section 4 M4) is therefore each point's *nearest*
  reconstructed camera center, not a literal per-point visibility
  lookup — see its module docstring. Same posture as the
  `min_component_volume_ratio` note in `cleanup.py`: a config/doc name
  that's a slight simplification of what's actually measured, documented
  at the point of implementation rather than silently reinterpreted.

## Milestone status

- [x] M0 — scaffold, config, capability probe, `v2m doctor`
- [x] M1 — Phase 1: intelligent ingest
- [x] M2 — Phase 2a: sparse SfM
- [x] M3 — Phase 2b: dense point cloud (monodepth + TSDF)
- [x] M4 — Phase 3: raw mesh (Poisson)
- [x] M5 — Phase 4: print-ready post-processing (the critical milestone)
- [x] M6 — end-to-end `v2m run` + `--resume`
- [ ] M7 — local web UI
- [ ] M8 — guided live capture
- [ ] M9 — optional quality backends (OpenMVS, hloc/ALIKED, tiling)

Update the checkbox when a milestone's verification step (in
`docs/ARCHITECTURE.md`) actually passes — not just when the code is
written.
