# Pipeline Consolidation — Tiers 0–2 (execution-ready design)

> **Current disposition (2026-07-15) — historical execution design.**
> [`2026-07-07-refactoring-roadmap.md`](2026-07-07-refactoring-roadmap.md) is the
> sole live status, dependency, and work-selection authority for this initiative.
> Phase 0 and Phase 2a remain unexecuted; Phase 1 has only isolated
> landed/obsolete items and still contains live defects. Commit `60d2c7c7b` landed
> an alternate dataset-backed barycentric CLI route, superseding the Phase 2b
> stitch-core recipe below. Current CI absolute scaling is governed by
> [`spec-ptycho-core.md`](../specs/spec-ptycho-core.md); the historical ~53× gap
> applies only to legacy-amplitude evidence. The task bodies are preserved for
> provenance and hazards only: they are not independently executable and do not
> create completion gates.

> **For agentic workers:** Execute phase-by-phase. Phases 0, 1, 2a are mechanical
> (deletion + local fixes) and safe to run subagent-driven. Phase 2b is
> behavior-changing and gated on a frozen-fixture baseline + a metric re-baseline
> (Gate A, designed below). Steps use `- [ ]`. Commit per step; commit messages
> carry **NO** AI/Claude attribution (repo rule).

**Parent plan:** `docs/plans/2026-07-06-pipeline-consolidation.md` (full 6-phase
overview). This document is the detailed design for the highest-value subset:
dead-code deletion (Phase 0), cheap defects (Phase 1), solver stabilization
(Phase 2a), and the CLI↔in-process inference unification (Phase 2b).

**Goal:** Remove ~3000 lines of zero-importer duplicate code, fix the latent
`TypeError` that makes CLI torch inference unrunnable, delete the three dead
VarPro solvers, and make the CLI and in-process inference paths produce the same
amplitudes for the same checkpoint by routing both through one barycentric stitch
core.

**Target branch:** `fno-stable` (HEAD `a1d52011` at design time). Do **not** check
out `main`; do not create worktrees.

**Evidence base:** grep-verified zero-importer confirmation + two design subagents
(inference frame/window; solver call-graph). All line numbers below verified
against `a1d52011`.

---

## Global Constraints (the ones that bind Tiers 0–2)

- **STABLE-CORE (CLAUDE.md §6):** `ptycho/model.py`, `ptycho/diffsim.py`,
  `ptycho/tf_helper.py` must NOT be modified. Everything here is torch-side
  (`ptycho_torch/…`, `scripts/…`, top-level `torch/`) — none of it is stable-core.
  Note: `ptycho_torch/model.py` is **not** stable-core and is editable.
- **Historical legacy-amplitude scope:** do NOT paper over the ~53× s1 gap
  (REASSEMBLY-BRIDGE-001). Tiers 0–2
  *stabilize and unify* the stitch surface; they do not touch VarPro numerics.
  `solve_lbfgs` output must be byte-frozen before any solver deletion.
  This does not govern current CI count-intensity behavior, whose physical-probe
  and VarPro scale contract lives in `spec-ptycho-core.md`.
- **Window unification shifts metrics.** CLI uses `stitch_crop_size=20`; the
  barycentric core uses `middle_trim=32`. Unifying onto the barycentric core
  changes every CLI NCC/MAE score — this is expected and must be re-baselined
  (Gate A), not "fixed."
- **Preserve `reassemble_patches_position_real_probe`** (`helper.py:197`) — gated
  live path, inert at gridsize=1 (object_big gate). Out of scope here; do not touch.

---

## Phase 0 — Delete zero-importer dead modules (effort S, risk LOW)

All targets confirmed zero-importer against `a1d52011` (precise import grep, not
substring). Re-run the grep at execution time before deleting.

**Delete (whole files):**
- `ptycho_torch/reassembly_alpha.py` — 0 importers
- `ptycho_torch/reassembly_beta.py` (1633 lines) — 0 importers
- `ptycho_torch/beta_modules/reassembly.py` — 0 importers; it imports the
  `reconstruct_image_barycentric_weighted` alias, so delete it **before** removing
  the alias (Step 0.3)
- top-level `torch/` directory (`torch/tf_helper.py` = dead line-by-line TF port;
  no `torch/__init__.py`, so it is not an importable package — unreachable). Delete
  the whole `torch/` dir including any `torch/tests/*`.
- `ptycho_torch/dset_loader_pt_mmap.py` — dead `PtychoDataset` sibling, 0 importers
- `ptycho_torch/datagen.py` (single-file) — shadowed by the `datagen/` package
  (package wins over module of the same name). The only reference,
  `ptycho_torch/notebooks/analysis.py:2 import ptycho_torch.datagen`, already
  resolves to the empty package `__init__.py`, not this file.

**Edit `ptycho_torch/reassembly.py`:**
- Remove `class VectorizedBarycentricAccumulator` (line 722) — defined but never
  instantiated in-module (the canonical accumulator is
  `VectorizedWeightedAccumulator` at 837; the only instantiations of the deleted
  class were in `reassembly_alpha/beta`, gone in this phase).
- Remove `reconstruct_image_barycentric_weighted = reconstruct_image_barycentric`
  (line 1323) — its only consumer was `beta_modules/reassembly.py:5`.

**Explicitly NOT in Phase 0** (has a live/notebook importer — defer):
- `reassembly.py:185 reconstruct_image` (used by `ptycho_torch/notebooks/analysis.py:86`).

- [ ] **0.1** Re-run zero-importer grep for every target:
      `grep -rn --include=*.py -E "(from|import).*\b<module>\b" ptycho/ ptycho_torch/ scripts/ tests/ | grep -v "/<module>.py:"`.
      Any live importer → STOP, reclassify.
- [ ] **0.2** Delete the six file targets (whole `torch/` dir included).
- [ ] **0.3** Delete `beta_modules/reassembly.py`, then remove the two symbols from
      `reassembly.py`. Grep both symbol names repo-wide → 0 remaining references.
- [ ] **0.4 Historical gate (superseded):** current work runs claim-matched
      selectors per task and `bash ci/run_ci_tests.sh` only at the boundary named
      by the live roadmap.
- [ ] **0.5** Commit: `refactor: delete zero-importer dead reassembly/loader modules`

**Exit:** ~3000 lines gone; gate green; no import errors.

---

## Phase 1 — Cheap correctness/hygiene defects (effort S, risk LOW)

Each is a self-contained commit. Verified lines below.

- [ ] **1.1 — the blocking `TypeError`.** `scripts/inference/inference.py:786`
      passes `debug_dump_dir=debug_dump_dir` to `_run_inference_and_reconstruct`,
      whose signature is
      `def _run_inference_and_reconstruct(model, raw_data, config, execution_config, device, quiet=False, intensity_scale=None)`
      (`ptycho_torch/inference.py:320`) — **no such param**. Add
      `debug_dump_dir=None` to the signature and thread it to the debug dump (or,
      if no dump is wired yet, accept-and-ignore with a `# TODO` referencing the
      Phase 2b dump). Add a smoke test that reaches this branch. **This unblocks
      Phase 2b validation.**
- [ ] **1.2** `ptycho_torch/inference.py:415` — delete the `output_scale_factor`
      blend (computed, never passed to `forward_predict`).
- [ ] **1.3** `ptycho_torch/reassembly.py:1316-1317` — delete the dead diagnostic
      clobber (`modified_scaled_canvas = modified_s1*... ` immediately overwritten
      by `texture_canvas.real + 1j*texture_canvas.imag`). Only reachable under
      `return_diagnostics=True`; the weighting on 1316 is discarded.
- [ ] **1.4** `ptycho_torch/config_params.py:238` (`output_dir="lightning_outputs"`)
      vs `:243` (`output_dir="training_outputs"`) — the second silently wins.
      Collapse to one field; grep both string values to confirm
      `training_outputs` was already dead.
- [ ] **1.5** `ptycho/params.py:124` — `def set(...)` unconditionally prints
      `DEBUG: Setting <k> to <v>`. **NOTE:** `ptycho/params.py` is not in the §6
      stable-core list, but confirm no test asserts on this stdout before editing.
      Gate it behind a logger/debug flag.
- [ ] **1.6** `ptycho_torch/reassembly.py:1082` —
      `gpu_ids = list(range(torch.cuda.device_count()))` unconditionally overwrites
      the `gpu_ids` argument. All five callers pass `None`. Drop `gpu_ids` from the
      `reconstruct_image_barycentric` signature (folds into the Phase 2b wrapper).
- [ ] **1.7** `ptycho_torch/api/api_helper.py:510 predict_only` — calls
      `forward_predict(batch)` with 1 arg vs the 4-arg contract
      `(x, positions, probe, input_scale_factor)`. Broken/dead: grep callers; delete
      or repair to the real contract. **Current override:** the API package has
      current public-contract tests, so deletion requires an explicit migration
      decision; this historical item does not authorize it.
- [ ] **Historical gate (superseded):** run claim-matched selectors per task and
      the checked-in CI harness only at the boundary named by the live roadmap.

**Exit:** CLI torch inference branch runs without `TypeError`; gate green.

---

## Phase 2a — Delete dead VarPro solvers (effort S→M, risk LOW–MED)

**Simplification vs the audit:** no shared-helper extraction is needed. The three
copies of the eigen-projection are: (1) `ptycho_torch/model.py:1502` — LIVE,
self-contained in `RectangularScaledDiffraction`, leave as-is; (2)
`beta_modules/model.py:834` — dies with `beta_modules` (already isolated, dead);
(3) inline in `VarProScaler.solve` (`reassembly.py:471-484`) — dies when `solve`
is deleted. So Phase 2a is pure deletion.

**Keep (all wired into `reconstruct_image_barycentric`):**
- `VarProScaler.solve_lbfgs` (`reassembly.py:656`) — the wired solver (called at
  `:1035`). Returns the unconstrained quartic minimizer with a global sign-flip.
- `accumulate_batch_from_basis` (`:370`, multi-mode; called at `:1216`)
- `compute_varpro_basis` (`:986`, module-level; called at `:1214`)

**Delete (grep-confirmed zero external callers):**
- `VarProScaler.solve` (`:452`) — eigen-projects onto a rank-1 cone (a *different*
  answer than `solve_lbfgs`; do not treat as interchangeable)
- `VarProScaler.solve_quadratic_direct` (`:488`)
- `VarProScaler.solve_autograd` (`:587`)
- single-mode `VarProScaler.accumulate_batch` (`:329`)

**Caveat — the ATA/ATb machinery.** `accumulate_batch` (`:329`) is the only
populator of `self.ATA/self.ATb`, which are consumed **only** by the dead solvers
plus `get_condition_number` / `get_correlation_matrix` / `swap_channels`.
`solve_lbfgs` uses only the scalar autograd stats from
`accumulate_batch_from_basis`. **Before** deleting `accumulate_batch`, grep for
`swap_channels` / `get_condition_number` on the wired path: the barycentric
`swap_detection` param defaults `'None'` and all callers pass `'None'`, so
`swap_channels` should be unreachable — but confirm, and if `swap_detection` can
reach `swap_channels`, scope that separately rather than breaking swap.

- [ ] **2a.1 — freeze first.** Write `tests/torch/test_varpro_solver_frozen.py`:
      construct a small fixed `VarProScaler` state via `compute_varpro_basis` +
      `accumulate_batch_from_basis` on a deterministic input, call `solve_lbfgs`,
      assert `(s1, s2)` within tolerance of recorded values. Run it green on
      current HEAD; record the values.
- [ ] **2a.2** Confirm zero external callers of `solve`,
      `solve_quadratic_direct`, `solve_autograd`, `accumulate_batch` (grep
      `ptycho/ ptycho_torch/ scripts/ tests/`). Confirm the `swap_channels`
      reachability per the caveat.
- [ ] **2a.3** Delete the four methods (and the now-orphaned ATA/ATb accumulation
      inside `accumulate_batch` only — leave `get_condition_number`/`swap_channels`
      if the caveat check leaves them reachable; otherwise delete them too).
- [ ] **2a.4** Run torch gate + the new frozen fixture (must pass unchanged).
      Commit: `refactor: remove dead VarPro solvers (keep solve_lbfgs)`

**Exit:** one wired solver; `solve_lbfgs` numerics provably unchanged; the two
remaining eigen-projection copies (beta, inline) gone via Phase 0 + this deletion.

---

## Phase 2b — Unify CLI ↔ in-process inference (effort L, risk HIGH — Gate A)

> **Superseded implementation recipe (2026-07-15):** `60d2c7c7b` landed
> `_resolve_reassembly_route` plus a dataset-backed
> `_run_barycentric_inference_and_reconstruct` path instead of the pure stitch-core
> extraction specified below. The steps in this section are historical. Any
> continuation must first audit the landed route, the still-separate uniform path,
> remaining direct call sites, and the CLI `debug_dump_dir` mismatch, then issue a
> fresh plan.

### The divergence (why this is the top-value item)
Same checkpoint, two amplitudes: the CLI (`_run_inference_and_reconstruct`,
`inference.py:320`) stitches **UNIFORM** (`hh.reassemble_patches_position_real`,
`:460`, window `stitch_crop_size=20`) and silently drops probe-weighting + VarPro;
the in-process path (`inference.py:186`) calls `reconstruct_image_barycentric`
(probe-weighted + VarPro, window `middle_trim=32`). Studies route around the CLI
because of this.

### Gate A — resolved design

**Coordinate frame — already compatible (no sign flip, no axis swap).** Both paths
use COM-relative `(x, y)`:
- Path A: `dx = x - mean(x)`, `dy = y - mean(y)`, `offsets = stack([dx,dy])`
  (`inference.py:432-434`).
- Path B: `relative_positions = global_coords - center_of_mass`, then
  `canvas_positions = relative_positions + canvas_center` where
  `canvas_center = [canvas_size[1]//2, canvas_size[0]//2]` and
  `canvas_size = (middle_trim + 2·ceil(max|dy|), middle_trim + 2·ceil(max|dx|))`
  (`reassembly.py:1122-1127, 1221-1224`).

Path A's `offsets` **equal** Path B's `relative_positions`. The only conversion
is the explicit `+canvas_center` shift (Path A's uniform stitcher does this
internally; the barycentric accumulator wants absolute canvas pixels).

**Window — canonicalize on `middle_trim` (32).** The barycentric core is
canonical; unify the CLI onto `middle_trim`. Map `stitch_crop_size → middle_trim`
(or deprecate `stitch_crop_size`). This changes CLI NCC/MAE — re-baseline (below).

**Real work — the input-contract mismatch.** `reconstruct_image_barycentric`
consumes a whole `PtychoDataset` (needs `coords_global`, `coords_relative`,
`rms_scaling_constant`, 5-D probe modes, `com` in `mmap_ptycho`); the CLI only has
a `RawData` + already-computed `patch_complex` from `forward_predict`. Two options:

| | Approach | Cost | Risk |
|---|---|---|---|
| **(b) recommended** | Extract the barycentric **stitch core** (`reassembly.py:1195-1258` + `1282-1289`) into a pure function `_barycentric_stitch_core(a_tilde, b_tilde, probe_modes, I_raw, canvas_positions, patch_size, *, uniform_weighting, varpro_scaling)` → `(amp, phase)`. Both `reconstruct_image_barycentric` (dataset-driven) and the CLI (RawData-driven) build its inputs and call it. | Refactor B internals once | Med — one numeric core, both callers thin |
| (a) not recommended | Make the CLI synthesize a `PtychoDataset`/tensordict from `RawData` | Replicate loader logic (coords_global, rms_scaling, com, probe modes) | High — duplicates the loader, drift-prone |

Go with **(b)**. `forward_predict` output supplies `a_tilde/b_tilde` directly;
reshape the single-mode probe to Path B's `(1,1,1,H,W)` 5-D layout (P=1).

**Double-scaling guard.** Path A applies `input_scale_factor`/`output_scale_factor`
normalization (`inference.py:414-426`); Path B applies `rms_scaling_constant` +
VarPro s1/s2. Feeding Path A's pre-scaled `patch_complex` into the VarPro core
would double-apply. Decide one scaling owner: the stitch core takes **unscaled**
`a_tilde/b_tilde` and owns rms+VarPro; the CLI stops pre-scaling patches.

### Steps

- [ ] **2b.1 — extract the core (no behavior change).** Pull `reassembly.py`
      1195-1258 + 1282-1289 into `_barycentric_stitch_core(...)`. Have
      `reconstruct_image_barycentric` call it. Run the existing barycentric tests +
      `varpro_probe_ablation_runner` smoke → identical output (byte or tight tol).
      Commit.
- [ ] **2b.2 — one wrapper for the 5 call sites.** All five sites
      (`base_api.py:1232`, `api_helper.py:524`, `workflows/components.py:1336`,
      `inference.py:186`, `varpro_probe_ablation_runner.py:413`) pass the same 6
      configs + `model` + `ptycho_dset`, with `gpu_ids=None`, `verbose=False`,
      `use_mixed_precision=True`, `swap_detection='None'` as constants. The **only**
      real variant is `return_diagnostics` (True only for the ablation runner →
      4-tuple vs 3-tuple). Add one wrapper with those defaults; repoint all five.
      Commit.
- [ ] **2b.3 — freeze the current in-process baseline.** Record
      `reconstruct_image_barycentric` amplitude output for one fixed checkpoint+dataset
      (fixture). This is the target the CLI must match.
- [ ] **2b.4 — rewire the CLI.** In `_run_inference_and_reconstruct`, replace the
      uniform `reassemble_patches_position_real` call (`:460`) with:
      build `canvas_positions` from `dx/dy` via the Gate-A transform
      (`+canvas_center`, `canvas_size` from `middle_trim`), reshape probe to 5-D,
      and call `_barycentric_stitch_core(...)` with `patch_size=middle_trim`,
      `uniform_weighting=inference_config.patch_weighting`,
      `varpro_scaling=True`. Remove the CLI patch pre-scaling (double-scaling guard).
      Keep the COM-offset front-end.
- [ ] **2b.5 — regression test.** Assert the CLI path and the in-process path
      produce matching amplitudes (within tol) for one checkpoint — the exact
      divergence this phase closes.
- [ ] **2b.6 — retire `load_and_predict`** (`inference.py:122`, MLflow, sets
      `training=True` before inference): grep callers; delete or route through the
      wrapper.
- [ ] **2b.7 — re-baseline metrics.** Re-run one inference/study smoke; record the
      NCC/MAE deltas vs the old CLI (uniform, crop-20) numbers **in this doc** —
      the shift is expected (window 20→32 + probe/VarPro now applied). Commit.

**Exit:** one inference entrypoint; CLI == in-process amplitude regression passes;
metrics re-baselined and recorded; `stitch_crop_size` mapped/deprecated.

---

## Design decisions surfaced (recommended — flag for owner veto)

1. **Phase 2b approach = (b) extract stitch core** (not (a) synthesize a dataset).
   Recommended for a single numeric core; the alternative duplicates loader logic.
2. **Unified window = `middle_trim` (32)**, canonicalizing on the barycentric
   contract. This *will* move CLI NCC/MAE numbers; Gate A's re-baseline (2b.7)
   captures it. Alternative: keep 20 and pass it as `patch_size` — cheaper but
   diverges from the canonical inference config.
3. **Scaling owner = the stitch core** (rms + VarPro); CLI stops pre-scaling.
   Alternative would require reconciling `output_scale_factor` with s1/s2, which
   reopens REASSEMBLY-BRIDGE-001 — avoid.

---

## Verification Protocol (every phase)

1. `pytest tests/torch -m "not slow"` with project deselect/ignore lists — matches
   the recorded green baseline (no new failures).
2. Phase 2a: `test_varpro_solver_frozen.py` passes unchanged.
3. Phase 2b: 2b.1 output-identity check; 2b.5 CLI==in-process regression; 2b.7
   NCC/MAE deltas recorded here before marking complete.
4. Commit per step; **no AI/Claude attribution** in commit messages.

## Sequencing

```
Phase 0 ──► Phase 1 ──► Phase 2a (freeze → delete)
                             │
                Gate A ──────►──► Phase 2b (extract core → wrapper → rewire CLI → re-baseline)
```

Phases 0, 1, 2a can run back-to-back subagent-driven (mechanical). 2b is the
single gated, behavior-changing step — do not start it until 1.1 (the `TypeError`
fix) and 2a (frozen solver) are landed and the in-process baseline (2b.3) is frozen.
