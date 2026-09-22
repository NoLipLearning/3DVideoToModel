> This is the durable, repo-tracked copy of the architecture document
> approved for this project. It originated as a Claude Code plan-mode
> document; that file lives outside the repo and does not survive
> container reclaim, so this copy is the canonical reference for every
> future coding session. See `CLAUDE.md` for the condensed quick-start
> version of this document (build commands, module map, milestone
> checklist) -- read this file for the *why* behind each decision.

# 3DVideoToModel — Architecture & Implementation Document

## Context

**The need.** Turn ordinary video — a phone clip from the camera roll, or a live capture session — into a *manifold, watertight, metric-scaled* 3D mesh that loads straight into a slicer (Cura/PrusaSlicer/Bambu) and prints without repair. Two subject classes matter equally: **objects** (a prop, a sculpture, a part) and **scenes** (a street, a room, a facade).

**Why this is not "just run COLMAP".** Off-the-shelf photogrammetry produces *surfaces*, not *solids*. A Poisson mesh straight out of Open3D is typically non-manifold, has holes where the camera never looked, floats at arbitrary scale, and has no bottom. Slicers reject or silently mangle all four. The engineering value of this project lives almost entirely in Phase 1 (feeding the reconstructor good frames) and Phase 4 (turning a surface into a printable solid). Phases 2–3 are orchestration of mature libraries.

**The hardware constraint that shapes everything.** Target is **Apple Silicon**. macOS has no CUDA, and COLMAP's dense stage (`patch_match_stereo`) is CUDA-only — stock `pycolmap` PyPI wheels are [built without CUDA](https://github.com/colmap/colmap/blob/main/python/README.md), and `pycolmap-cuda12` is Linux-only. **COLMAP dense MVS is unavailable on this target and the code must never call it.** The default dense backend is therefore monocular-depth + TSDF fusion driven by COLMAP's poses, accelerated on MPS. This is the single most important architectural fact in this document.

**Intended outcome.** `v2m run clip.mov --preset object` produces `model.stl` that passes `trimesh.is_watertight`, has positive volume, correct winding, and real millimetre dimensions.

**Explicitly out of scope:** slicing, G-code, printer control, native mobile apps, training any neural network.

---

## 1. Directory Structure

```
3DVideoToModel/
├── pyproject.toml                  # uv-managed; optional-dependency groups
├── uv.lock
├── README.md
├── CLAUDE.md                       # written at M0; build/test commands for future sessions
├── .gitignore                      # runs/, *.ply, *.stl, models/, .venv/
├── docs/
│   └── ARCHITECTURE.md             # this document
├── configs/
│   ├── default.yaml                # base knobs — every value below is declared here
│   ├── object.yaml                 # preset: single subject, closed bottom
│   ├── scene.yaml                  # preset: large area, ground slab
│   └── fast.yaml                   # preview preset: sparse_only, low res
├── src/v2m/
│   ├── __init__.py
│   ├── cli.py                      # Typer app: run | extract | sfm | dense | mesh | printprep | doctor | serve | capture
│   ├── config.py                   # Pydantic v2 models + YAML load + CLI override merge
│   ├── capability.py               # hardware/binary probe → Capabilities
│   ├── run_context.py              # RunContext: run dir, manifest, resume, artifact paths
│   ├── types.py                    # shared dataclasses (FrameRecord, SfmResult, MeshReport…)
│   ├── errors.py                   # V2MError hierarchy — each carries a user-facing remedy
│   ├── logging_setup.py            # rich console + JSONL file log per run
│   ├── phase1_ingest/
│   │   ├── video_reader.py         # streaming decode, rotation metadata, VFR timestamps
│   │   ├── quality.py              # Laplacian variance, exposure, motion-blur direction
│   │   ├── selector.py             # redundancy + baseline gating
│   │   └── extract.py              # orchestrator → frames/ + frames.json
│   ├── phase2_sfm/
│   │   ├── backends/
│   │   │   ├── base.py             # SfmBackend protocol
│   │   │   ├── colmap_backend.py   # pycolmap: SIFT, sequential+loop matching, mapper
│   │   │   └── hloc_backend.py     # OPTIONAL (M9): ALIKED/DISK + LightGlue for low texture
│   │   ├── sfm.py                  # backend dispatch, undistortion, pose export
│   │   ├── dense/
│   │   │   ├── base.py             # DenseBackend protocol
│   │   │   ├── monodepth_tsdf.py   # DEFAULT on Apple Silicon
│   │   │   ├── openmvs.py          # OPTIONAL: external binary, --cuda-device -2
│   │   │   └── sparse_only.py      # always-works fallback / fast preview
│   │   └── diagnostics.py          # registration rate, track length, textureless report
│   ├── phase3_mesh/
│   │   ├── normals.py              # camera-aware normal orientation
│   │   ├── poisson.py              # screened Poisson + density-quantile crop
│   │   └── cleanup.py              # components, floaters, decimation
│   ├── phase4_print/
│   │   ├── ground.py               # RANSAC plane, gravity alignment, mode-specific base
│   │   ├── watertight.py           # the 6-rung repair ladder
│   │   ├── scale.py                # ArUco / two-point / manual metric scale
│   │   ├── validate.py             # printability assertions → PrintReport
│   │   └── export.py               # STL (binary, mm) + OBJ/MTL + GLB for preview
│   ├── capture/
│   │   └── live.py                 # M8: guided capture HUD → recorded frame set
│   ├── web/
│   │   ├── app.py                  # FastAPI: upload, job control, SSE progress
│   │   ├── jobs.py                 # single-worker job queue over RunContext
│   │   └── static/                 # index.html + <model-viewer> preview, no build step
│   └── report/
│       └── html.py                 # per-run HTML report: thumbnails, metrics, failures
├── tests/
│   ├── conftest.py                 # synthetic fixtures
│   ├── fixtures/
│   │   ├── make_synthetic_video.py # renders a known cube→video; ground-truth dimensions
│   │   └── broken_meshes/          # hand-made non-manifold/holed OBJs
│   ├── unit/                       # pure functions — fast, no heavy deps
│   └── integration/                # @pytest.mark.slow, gated on colmap availability
├── scripts/
│   ├── install_macos.sh            # brew + uv sync + model download
│   └── fetch_models.py             # cache Depth-Anything weights offline
└── runs/                           # gitignored; one dir per run
    └── 2026-09-20_143022_a3f1/
        ├── manifest.json           # ← the resume/state file
        ├── run.log.jsonl
        ├── frames/                 # 000000.jpg … + frames.json
        ├── sfm/                    # database.db, sparse/0/, undistorted/, cameras.json
        ├── dense/                  # depth/*.npy, dense.ply
        ├── mesh/                   # raw.ply, cleaned.ply
        ├── output/                 # model.stl, model.obj, model.glb, print_report.json
        └── report.html
```

**Why `runs/<id>/` with a manifest:** every phase is expensive (minutes to hours). The manifest records per-phase `status`, `input_hash`, `duration_s`, and `artifacts`. `--resume` re-enters at the first incomplete phase. Without this, every meshing-parameter tweak re-runs SfM, and iteration becomes impossible.

---

## 2. Library Analysis

### Phase 1 — Ingest
| Library | Role | Notes |
|---|---|---|
| **opencv-contrib-python** | decode, Laplacian, ORB, CLAHE, ArUco | `contrib` is required — `cv2.aruco` lives there. Install **only this**, never alongside `opencv-python` (they conflict). |
| **numpy** | array math | — |
| **PyAV** *(optional)* | VFR timestamps, rotation metadata | `cv2.VideoCapture` mishandles iPhone rotation metadata and variable frame rate. PyAV reads the container properly. Fall back to OpenCV if absent. |

**System dep:** none mandatory. HEVC/H.265 (iPhone default) decodes via OpenCV's bundled FFmpeg on macOS arm64. `doctor` must verify by decoding one frame, not by trusting the build flags.

### Phase 2 — SfM + Dense
| Library | Role | Notes |
|---|---|---|
| **pycolmap** | sparse SfM: SIFT, matching, incremental mapper, undistortion | PyPI wheel, macOS arm64, **CPU SIFT only** (no GPU on macOS). `github.com/colmap/pycolmap` is deprecated — bindings now live in `colmap/colmap/python`, still `pip install pycolmap`. |
| **torch + transformers** | Depth Anything V2 (`depth-anything/Depth-Anything-V2-Small-hf`) | MPS-accelerated. Set `PYTORCH_ENABLE_MPS_FALLBACK=1` — some ops still lack MPS kernels. |
| **open3d** | `ScalableTSDFVolume`, point-cloud ops | macOS arm64 wheels from 0.18+. |
| **OpenMVS** *(optional)* | true MVS densification on CPU | External C++ binary. `DensifyPointCloud --cuda-device -2` forces CPU. Not pip-installable; build via cmake or tap. Strictly optional — degrade with a log line if absent. |

**The CUDA problem, restated for the implementer:** `pycolmap.patch_match_stereo` and `colmap patch_match_stereo` **do not exist in any macOS build**. Calling them raises. `capability.py` must hard-gate this, and `colmap_backend` must never emit a dense call.

**Default dense strategy — `monodepth_tsdf`:**
1. COLMAP gives per-image pose + intrinsics + sparse 3D points.
2. Depth Anything V2 gives *relative inverse depth* (disparity) per frame — dense and smooth even on blank walls.
3. Project that frame's visible sparse points into the image; fit a robust affine `a·disparity + b ≈ 1/z_sfm` (Huber/RANSAC, require ≥30 correspondences) to lift relative depth to metric.
4. Integrate the aligned depth maps into `open3d.pipelines.integration.ScalableTSDFVolume` using the known poses.
5. Extract point cloud (and optionally mesh directly — TSDF marching cubes is already near-manifold, a useful shortcut).

This is the right default here: it runs on MPS, it is the *only* approach in this stack that produces geometry for textureless surfaces, and it degrades to a visible warning rather than a crash when alignment fails.

### Phase 3 — Meshing
| Library | Role |
|---|---|
| **open3d** | `create_from_point_cloud_poisson` (returns per-vertex densities — essential), normal estimation, clustering |
| **pymeshlab** *(optional)* | screened Poisson variant, quadric decimation, non-manifold repair |

`pymeshlab` is powerful but the heaviest install and the most likely to fail on arm64. **Every pymeshlab call must have an open3d/trimesh fallback**; it is an enhancement, never a hard dependency.

### Phase 4 — Print Prep
| Library | Role |
|---|---|
| **trimesh** | the workhorse: repair, winding, volume, export, transforms |
| **manifold3d** | trimesh's boolean engine — *guarantees manifold output*. This is how the bottom gets capped. |
| **scipy** | plane fitting, convex hull, spatial queries |

**Key insight for capping:** do not hand-triangulate boundary loops. Boolean-intersect the mesh with a half-space box via `manifold3d`. The cut face is capped automatically and the result is manifold by construction. This is more robust than any hole-filling heuristic.

### Cross-cutting
`typer` (CLI), `pydantic` v2 (config), `rich` (progress/logging), `fastapi`+`uvicorn`+`python-multipart` (web), `pytest` (+`pytest-cov`), `ruff` (lint+format).

### `pyproject.toml` dependency groups
```
core    = numpy, scipy, opencv-contrib-python, typer, pydantic, pyyaml, rich, trimesh, manifold3d, open3d
sfm     = pycolmap
depth   = torch, torchvision, transformers, pillow, safetensors
mesh    = pymeshlab          # optional, may fail on arm64
web     = fastapi, uvicorn[standard]
dev     = pytest, pytest-cov, ruff, av
```
Default `uv sync` installs `core+sfm+depth+web`. `mesh` is opt-in. Torch/torchvision resolve from default PyPI, which is correct on macOS (no CPU/CUDA split exists there — its wheels are already CPU/MPS, never CUDA). On Linux/Windows this pulls PyPI's CUDA-bundled build; `pyproject.toml` documents an opt-in platform-conditional override to `download.pytorch.org/whl/cpu` for a leaner install, left off by default because that host is blocked by some sandboxed/CI network policies.

---

## 3. Edge Case Handling

### 3.1 Featureless surfaces (blank walls, sky, glass, asphalt)
**Detect** — in `phase2_sfm/diagnostics.py`:
- keypoints/image below `min_keypoints` (600) → flag frame
- registration rate `registered/total < 0.7` → flag run
- mean track length < 3 → weak geometry
- per-image sparse-point density heat map → identify *which regions* failed

**Mitigate, in order:**
1. **CLAHE** contrast equalisation before feature extraction (`clipLimit=2.0, tile=8×8`) — recovers detail on near-uniform surfaces. On by default for `scene` preset.
2. Lower SIFT `peak_threshold` (0.0066 → 0.002) and raise `max_num_features` on a retry pass.
3. **The mono-depth backend is the real answer** — it hallucinates plausible smooth geometry across textureless regions from learned priors. This is *precisely* why it is the default rather than a fallback.
4. Optional (M9) `hloc_backend` with **ALIKED + LightGlue** — learned features are far better on low texture. ⚠️ **License note:** SuperPoint/SuperGlue weights are Magic Leap **research/non-commercial**. Use **ALIKED** or **DISK** (permissive) as the default learned detector; expose SuperPoint behind an explicit opt-in flag with the restriction logged.

**Never fail silently.** If a region reconstructs poorly, `report.html` shows it and states the capture fix ("move slower past the wall", "add texture", "avoid pointing at sky/glass").

### 3.2 Memory exhaustion
The realistic failure mode: a 4K 3-minute clip → 5400 frames → exhaustive matching (14.6M pairs) → OOM or effectively infinite runtime.

| Guard | Mechanism | Default |
|---|---|---|
| Never load whole video | stream frame-by-frame, write JPEG immediately | always |
| Frame budget | hard cap after quality filtering | 150 (object) / 300 (scene) |
| Resolution cap | downscale longest edge before SfM | 1600 px |
| **Matching strategy** | **sequential matcher + vocab-tree loop detection — never exhaustive above 80 images** | overlap=10, loop_detection=True |
| Depth map memory | process one frame at a time, write `.npy`, never hold all | always |
| TSDF budget | `ScalableTSDFVolume` is sparse-block; voxel size adapts to scene extent | 4 mm object / 2 cm scene |
| Point cloud | voxel-downsample before Poisson | 1× voxel size |
| Poisson depth | adaptive by point count (≤10 for <2M pts, ≤11 above) | 10 / 11 |
| Scene tiling | if bbox diagonal > `tile_threshold`, TSDF-integrate per spatial tile, mesh separately, union | 30 m |
| Preflight | estimate peak RAM from frame count × resolution; **refuse and suggest smaller settings** rather than OOM mid-run | via `psutil` |

### 3.3 Defining the "floor" and closing the mesh
This is the difference between a printable solid and a broken surface. `phase4_print/ground.py`:

1. **Find the plane** — `open3d.geometry.PointCloud.segment_plane` (RANSAC, `distance_threshold=2×voxel`, `ransac_n=3`, `num_iterations=1000`) on the dense cloud.
2. **Disambiguate** — a street scan may fit a *wall* instead. Score candidate planes by: inlier count, normal-vs-gravity agreement, and whether most geometry lies on one side. Gravity prior comes from the median camera "up" vector across poses (people hold cameras roughly upright) or from video metadata when present.
3. **Re-orient** — rotate the whole scene so the ground normal is `+Z` and translate so the plane sits at `z=0`. Every downstream step assumes this canonical frame.
4. **Mode-specific base:**
   - **OBJECT** — discard points below `z = ε`; boolean-intersect the mesh with a half-space box `z ≥ 0` via `manifold3d`. The cut is capped and manifold by construction. If the subject is hollow, optionally offer a shelled variant.
   - **SCENE** — keep the ground as a surface, then extrude it into a slab: build a prism from the ground footprint down to `z = −slab_thickness` (default 3 mm) and **union** it with the mesh. Result is a terrain tile with a flat printable underside.
5. **Fallbacks** — if no plane reaches `min_inlier_ratio` (0.15): OBJECT mode falls back to cutting at the 2nd-percentile Z of the main component; SCENE mode falls back to the oriented-bounding-box minimum face. Either way, **log loudly** that the floor was guessed.

### 3.4 Metric scale (SfM is scale-blind)
Resolution order, first success wins, always recorded in `print_report.json`:
1. **ArUco/ChArUco marker** — `cv2.aruco` detects a printed marker of known edge length; solve the similarity transform. Fully automatic, sub-percent accurate. *Recommended workflow: print a 100 mm ArUco tag, lay it in the scene.*
2. **Two-point reference** — user supplies a real-world distance between two picked 3D points via CLI/UI.
3. **Manual factor** — `--scale-factor`.
4. **Fit to build volume** — normalize the longest axis to `--target-size` (default 150 mm). Output is *dimensionally meaningless but printable*; flagged prominently in the report.

### 3.5 Other real-world failures
- **Rolling shutter / motion blur** → Phase 1 rejection handles it; if >60% of frames are rejected, abort early with "footage too shaky, re-shoot slower".
- **Moving objects** (cars, people in a street scan) → they produce inconsistent geometry. Mitigation: TSDF integration naturally suppresses transient surfaces; additionally flag high per-voxel depth variance. Full semantic masking is M9+.
- **Pure rotation / no parallax** (user pivots in place) → SfM degenerates. Detect via median triangulation angle < 2°; abort with "walk around the subject, don't spin on the spot".
- **HEIC/HEVC and iPhone rotation** → PyAV path; `doctor` decode-test proves it works before a long run.
- **Disk exhaustion** → preflight estimate; `runs/` cleanup command.

---

## 4. Step-by-Step Execution Plan

Each milestone is independently runnable, independently testable, and ends in a commit. **Build and verify strictly in order** — every milestone consumes the previous one's artifact.

### M0 — Scaffold & capability probe
`pyproject.toml` (uv), `config.py`, `run_context.py` + manifest, `logging_setup.py`, `errors.py`, `capability.py`, Typer skeleton, `ruff`, pytest bootstrap, `CLAUDE.md`, `.gitignore`.
**Verify:** `uv run v2m doctor` prints a table — platform, MPS availability, CUDA=False, each library present/absent, colmap/OpenMVS binary status, HEVC decode test — and exits non-zero on missing *core* deps.
**Critical:** `doctor` must state plainly "COLMAP dense MVS unavailable (no CUDA) — using monodepth_tsdf".

### M1 — Phase 1: Intelligent ingest
`video_reader.py`, `quality.py`, `selector.py`, `extract.py`.
Adaptive blur threshold `max(30, 0.6×median(laplacian_var))`; ORB-overlap redundancy gate (keep when inlier overlap vs last-kept < 0.75); enforce frame budget by even temporal spread over survivors.
**Verify:** `uv run v2m extract tests/fixtures/sample.mp4 -o runs/test` → `frames/` + `frames.json` with per-frame scores and a rejection reason for every dropped frame. Unit-test blur scoring on synthetic sharp/blurred pairs.
(Built as `tests/fixtures/make_synthetic_video.py`: a camera orbiting a 200mm textured cube, with deliberately injected blurred and held-still/duplicate frames plus a ground-truth JSON -- `.mp4`/mp4v rather than the `.mov` named above, since OpenCV muxes it far more reliably on Linux; any real container works identically through `video_reader.py`.)

### M2 — Phase 2a: Sparse SfM
`colmap_backend.py`, `sfm.py`, `diagnostics.py`. Exhaustive matching at/below `exhaustive_matching_max_images` (80) else sequential+loop-detection, incremental mapper, `undistort_images`, export poses to `cameras.json`, sparse cloud to `sparse.ply`.
**Verify:** on the synthetic fixture, registration rate ≥ 0.9 and reprojection error < 1.5 px. Diagnostics JSON written even on failure.

**Actually achieved on the fixture:** 52/53 images registered (98%), 0.41px mean reprojection error, 7.56 mean track length. Two implementation notes that turned out to matter:
- **Exhaustive, not sequential, below the threshold.** Sequential-only matching (overlap=10) starved the incremental mapper of any wide-baseline pair on an orbiting-camera capture and produced a 2-image reconstruction. Exhaustive matching at this scale gives the mapper the widest choice of a good seed pair; it's what actually gets registration above 0.9. Sequential+loop-detection remains the path above the threshold, for the memory reasons in Section 3.2.
- **Vocab-tree loop detection needs a file COLMAP doesn't ship and this project's sandbox can't download** (`demuc.de` is blocked by network policy, confirmed directly). `colmap_backend.py` checks `V2M_VOCAB_TREE_PATH`; without it, loop detection degrades to plain sequential matching with a logged warning, same posture as pymeshlab/OpenMVS elsewhere in this doc.

The winning reconstruction is written to `sfm/sparse/final/` (not the illustrative `sparse/0/` above) — after a Section 3.1 retry, the actual winner may not be "0" from either attempt, so `sfm.py` always writes the one it chose to an unambiguous, stable path. `cameras.json`'s rotation is a quaternion in **(x, y, z, w)** order (Eigen/COLMAP convention); `cam_from_world` maps `p_cam = R @ p_world + t`.

The M1 synthetic fixture needed real rework to be SfM-viable — see `tests/fixtures/make_synthetic_video.py`'s module docstring for the three failed attempts and why (in short: SIFT keys off grayscale gradients, not hue; sparse discrete points don't give the mapper enough growable 3D structure regardless of per-point texture; and per-pixel random noise is statistically self-similar and produces false correspondences that fail PnP RANSAC. What worked: real per-face texture, homography-warped onto each face's actual 3D quad, using a sparse scatter of distinctly-sized/colored/positioned shapes rather than noise).

### M3 — Phase 2b: Dense point cloud
`dense/base.py`, `sparse_only.py` first (trivial, unblocks M4), then `monodepth_tsdf.py` — depth inference, robust affine alignment to sparse points, TSDF integration, `dense.ply`.
**Verify:** `dense.ply` has ≥20× the sparse point count; aligned depth RMSE against held-out sparse points < 5% of scene extent. Unit-test the affine alignment against synthetic depth with known `a,b` + outliers.

**Actually achieved on the fixture:** 119,673 dense points from 4,171 sparse points (28.7×), all 42 registered images integrated (none skipped for too few correspondences), mean per-image alignment RMSE 0.00095 (1/reconstruction-scale units — see the scale note below). Three implementation notes that turned out to matter:
- **`ScalableTSDFVolume` is broken in this project's dev sandbox** (open3d==0.20.0, Linux CPU wheel). `integrate()` runs without error, but `extract_point_cloud()`/`extract_voxel_point_cloud()`/`extract_triangle_mesh()` all return empty — confirmed three independent ways: this project's own data, a minimal synthetic flat-plane, and Open3D's own official RGBD-integration tutorial parameters verbatim. `monodepth_tsdf.py`'s `_build_tsdf_volume()` uses `UniformTSDFVolume` instead, sized adaptively from the sparse reconstruction's own bounding box (resolution clamped to [32, 400] voxels). This is very possibly a Linux-CPU-wheel-specific issue rather than a real Open3D bug — **if you're on macOS (the actual target) and can confirm `ScalableTSDFVolume` works there, switching back gets better behavior on a large/unbounded scene capture that might not fit a fixed-size volume; `_build_tsdf_volume()`'s call site is the only place that would need to change.**
- **COLMAP's own reconstruction scale is arbitrary, not metric.** Monocular SfM normalizes the first registered image pair's baseline to an unknown length, not a real-world one — true metric scale isn't established until Phase 4's `scale.py` (Section 3.4), long after Phase 2b's TSDF volume is built and discarded. `tsdf_voxel_size_m`/`tsdf_sdf_trunc_m` are named (and defaulted, Section 3.2) as if in real metres, which would size the truncation band wrong by whatever unknown factor COLMAP's unit differs from an actual metre if applied literally — for a tightly-spaced capture that factor can make the band thinner than a single voxel. `_build_tsdf_volume()` instead treats only the *ratio* between the two config values (default 5 voxels of truncation) as meaningful, applying it to the actual per-voxel size the reconstruction resolves to. `resolution` itself needs no equivalent fix — clamping to [32, 400] already absorbs an arbitrary length/voxel-size ratio without reference to real units. This makes Phase 2b's TSDF sizing correct regardless of what scale COLMAP happens to produce, without trying to solve metric calibration early (that stays Phase 4's job, as planned).
- **`TransformersDepthEstimator` is untested end-to-end in this sandbox** (huggingface.co is network-policy-blocked, confirmed directly). All M3 tests inject a `GeometricDepthEstimator` test double (`tests/unit/test_monodepth_tsdf.py`) built from the fixture's own known ground-truth geometry (`tests/fixtures/make_synthetic_video.py::render_true_depth`) via the `DepthEstimator` protocol, rather than exercising the real model. Depth-model load failures (no network, no cache) now raise `SfMError` with a remedy instead of a bare `OSError` traceback. Verify `TransformersDepthEstimator` for real on a machine that can reach huggingface.co before trusting it blindly.

Backend dispatch (`config.backend`, including `"auto"` → `capability.py`'s `dense_backend` probe) lives in `dense/__init__.py` rather than a separate top-level file — unlike sparse SfM's `sfm.py`, there's no shared orchestration logic between `monodepth_tsdf` and `sparse_only` to justify one.

### M4 — Phase 3: Raw mesh
`normals.py` (orient toward the camera that observed each point — far more reliable than tangent-plane propagation), `poisson.py` (screened Poisson + density-quantile crop at 0.03), `cleanup.py` (largest-component / floater removal, decimate to ≤300k faces).
**Verify:** `mesh/cleaned.ply` opens in MeshLab, visually matches the subject, face count within budget.

### M5 — Phase 4: Print-ready ⭐ *the milestone that determines whether this project succeeds*
`ground.py`, then `watertight.py` — the repair ladder, applied in order, stopping at the first rung that yields a valid solid:

| # | Rung | Tool |
|---|---|---|
| 1 | merge vertices, drop degenerate/duplicate faces, fix winding & normals | trimesh |
| 2 | keep largest connected component; drop floaters below volume threshold | trimesh |
| 3 | fill holes up to `max_hole_size` | trimesh → pymeshlab if available |
| 4 | remove non-manifold edges/vertices | pymeshlab (skip if absent) |
| 5 | **half-space boolean cut + cap** — object base or scene slab | manifold3d |
| 6 | **last resort: voxel remesh** (voxelize → marching cubes) — guarantees manifold, costs detail; logged as a quality warning | open3d/trimesh |

Then `scale.py`, `validate.py` (assert `is_watertight`, `is_winding_consistent`, `volume > 0`, `euler_number` sane, min wall thickness, bbox within build volume), `export.py` (binary STL in mm + OBJ/MTL + GLB).
**Verify:** `tests/fixtures/broken_meshes/*` all become watertight. Synthetic cube round-trips to within 2% of ground-truth dimensions. `print_report.json` records which rung was used and every scale decision.

### M6 — End-to-end orchestration
`v2m run` chaining all phases with `--resume`, `--preset object|scene|fast`; `report/html.py`.
**Verify:** one command, real phone video → `output/model.stl`. Kill mid-run, re-run with `--resume`, confirm it restarts at the right phase.

### M7 — Local web UI
FastAPI: drag-and-drop upload, preset picker, single-worker job queue, SSE progress stream, `<model-viewer>` GLB preview, STL download. No JS build step — CDN only.
**Verify:** `uv run v2m serve` → browser at `:8000` → upload → watch progress → preview → download.

### M8 — Guided live capture
`capture/live.py`: OpenCV camera loop with a quality HUD (sharpness bar, accepted-frame counter, baseline/coverage indicator), writing accepted frames directly to `frames/` so it enters the pipeline at M2 — *no SLAM*.
**Verify:** capture a desk object live, reconstruct it with `--from-frames`.

### M9 — Optional quality backends
`openmvs.py` (CPU MVS when the binary exists), `hloc_backend.py` (ALIKED+LightGlue for low texture), scene tiling for large captures, moving-object suppression.

---

## Verification Strategy

**Unit** (fast, no heavy deps — these are the ones that catch real bugs): blur scoring, frame selection, affine depth alignment, plane disambiguation, scale math, each repair rung against known-broken meshes.

**Integration** (`@pytest.mark.slow`, skipped when colmap is absent): synthetic-video → STL, asserting watertightness and ground-truth dimensions within 2%.

**The ground-truth fixture is the backbone.** `tests/fixtures/make_synthetic_video.py` renders a textured cube of *exactly known size* orbited by a virtual camera. Every phase can be scored against truth — poses, depth, mesh dimensions. Build this at M0/M1; it pays for itself many times over.

**Manual acceptance:** the real test is a phone video of a real object producing an STL that a slicer accepts with zero repairs and prints.

---

## Notes for the Implementing Model (Sonnet)

1. **One milestone per session.** Read only that milestone's files. Do not re-read the whole repo — `CLAUDE.md` (written at M0) carries the shared context.
2. **Every default value is specified above.** Use them; do not re-derive or deliberate.
3. **Never call `patch_match_stereo`.** It does not exist on macOS. If you find yourself reaching for COLMAP dense, you are on the wrong path.
4. **`pymeshlab` is optional.** Guard every import; always provide a trimesh/open3d fallback.
5. **Write the manifest entry at the end of every phase.** Resumability is not a nice-to-have here — without it, iteration on a 40-minute pipeline is impossible.
6. **Fail with remedies.** Every `V2MError` carries a user-facing fix ("re-shoot walking slower"), not just a stack trace.
7. Commit after each milestone to `claude/gallant-tesla-fmpff3`.

**Sources:** [COLMAP Python bindings README](https://github.com/colmap/colmap/blob/main/python/README.md) · [pycolmap on PyPI](https://pypi.org/project/pycolmap/) · [pycolmap-cuda12](https://pypi.org/project/pycolmap-cuda12/) · [OpenMVS DensifyPointCloud](https://github.com/cdcseacave/openMVS/blob/master/apps/DensifyPointCloud/DensifyPointCloud.cpp)
