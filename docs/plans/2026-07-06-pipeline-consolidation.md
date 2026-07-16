# Pipeline Consolidation Refactor Plan

> **Current disposition (2026-07-15) — historical vertical-slice plan.**
> [`2026-07-07-refactoring-roadmap.md`](2026-07-07-refactoring-roadmap.md) is the
> sole live status, dependency, and work-selection authority for this initiative.
> Phase 0 remains unexecuted; Phase 1 contains a few landed/obsolete hygiene items
> but still includes live defects; Phase 2a and Phases 3–5 remain candidate work.
> Commit `60d2c7c7b` landed an alternate dataset-backed barycentric CLI route, so
> the Phase 2b core-extraction recipe is superseded and must be re-planned around
> that implementation. Current CI absolute scaling is governed by
> [`spec-ptycho-core.md`](../specs/spec-ptycho-core.md); the historical ~53× gap
> below applies only to legacy-amplitude evidence. The task bodies are retained for
> provenance and hazards only: they are not independently executable and do not
> create completion gates.

> **For agentic workers:** Execute phase-by-phase. Phases 0–1 are mechanical and
> safe to run subagent-driven. Phases 2–6 are behavior-changing and **gated** —
> each requires a frozen-fixture parity baseline *before* any deletion/rewrite,
> and Phase 2b + Phase 5 have open design questions that must be resolved first
> (see **Design Gates**). Steps use `- [ ]` for tracking.

**Goal:** Collapse the accumulated TF/PyTorch duplicate implementations of
reassembly, training, inference, config, and scaling down to one canonical path
per operation — deleting ~3000 lines of zero-importer dead code first, then
converging live forks behind parity fixtures — without disturbing load-bearing
per-pipeline conventions.

**Architecture:** The repo carries two backends (TF `ptycho/`, PyTorch
`ptycho_torch/`) that reimplement every operation independently, plus
branch-fork archaeology (alpha/beta/main variants committed side-by-side) and a
CLI-vs-study driver drift. Consolidation proceeds outside→in: delete dead
modules (no callers) → fix cheap correctness defects → stabilize the solver →
unify inference → unify training → unify config → retire the stitch fork. TF
stable-core stays untouched; consolidation is torch-side plus explicitly-labelled
OLD/deprecated TF siblings only.

**Tech Stack:** Python, PyTorch ≥2.2 (POLICY-001), TensorFlow (stable-core),
pytest, git plumbing for main updates.

**Source audit:** 6-dimension subagent audit (reassembly, training, inference,
config, data/coords, scaling), synthesis saved to session scratchpad
`synth.json`. Findings referenced: REASSEMBLY-BRIDGE-001, STITCH-GRIDSIZE-001,
GRIDLINES-OBJECT-BIG-001, TORCH-REASSEMBLY-NORM-001, EXEC-ACCUM-001, CONFIG-001,
POLICY-001 (all in `docs/findings.md`).

**Target branch:** `fno-stable` (torch-side work; Phase-1 work is cherry-picked
to fno-stable per branch topology, never merged to main directly). Do **not**
check out `main`; do not create worktrees.

---

## Global Constraints

These are **invariants**, not cleanup targets. Every phase's work implicitly
includes them. Violating one silently corrupts reconstructions or breaks the
parity oracle.

- **STABLE-CORE (CLAUDE.md §6):** `ptycho/model.py`, `ptycho/diffsim.py`,
  `ptycho/tf_helper.py` must NOT be modified. Consolidation touches the torch
  side and only the explicitly-labelled OLD/deprecated TF siblings
  (`shift_and_sum_old`, `reassemble_position_old`, `reassemble_whole_object`).
- **COORD SIGN FLIP is load-bearing:** `local_offset_sign = -1` (TF,
  `raw_data.py:123,941`) vs `+1` (torch, `patch_generator.py:386,403`). Do NOT
  merge the two `get_relative_coords` into one shared function — it mirror-flips
  every scan displacement in one pipeline (amp_corr 0.075 vs 0.52). Keep both
  conventions or migrate fixtures atomically.
- **COORD AXIS TRANSPOSE is load-bearing:** TF `coords_nn` is `(M,1,2,C)`
  averaged over axis=3; torch is `(M,C,1,2)` averaged over axis=1
  (REASSEMBLY-BRIDGE-001 trap#1). Any array shared TF↔torch is silently
  transposed; a bridge must transpose explicitly.
- **DIFFRACTION UNITS split is real data, not a bug:** study builders write
  `round(amp²·S)` uint16 intensity counts; the loader assumes normalized
  amplitude <1 and silently re-normalizes (`dataloader.py:703-710`). Do NOT add
  a "unit-safety" assertion to the loader — it would reject every existing
  study dataset.
- **object_big GATE must be preserved:** `training_patch_weighting` probe/uniform
  branches are inert at gridsize=1 by the `if self.object_big:` gate
  (`model.py:1584`) plus the grid_lines runner hardcoding `object_big=False`
  (GRIDLINES-OBJECT-BIG-001). Consolidating the two torch training reassemblers
  must NOT accidentally activate probe/uniform weighting. Do NOT delete
  `reassemble_patches_position_real_probe`.
- **Historical legacy-amplitude scope:** the ~53× s1 units gap
  (REASSEMBLY-BRIDGE-001) must NOT be papered over by rescaling s1. It needs a real nphotons anchor
  (`derive_intensity_scale_from_amplitudes`, `helper.py:723`) wired into
  `apply_varpro_canvas_scaling`. Blind rescale breaks the parity oracle.
  This is not the current CI contract: CI count-intensity scaling and the physical
  probe/VarPro gauge are governed by `spec-ptycho-core.md`.
- **update_legacy_dict None-skip** (`config.py:778`) is relied on across ~24
  call sites (stale-reset behavior). Audit all callers before touching.
- **Frozen-fixture parity tests pin intentional quirks:** e.g. `RectangularMAELoss`
  deliberately double-squares to match main byte-for-byte (`model.py:1731`);
  `test_cross_branch_rectangular_parity.py`, `test_rectangular_scaled_forward.py`
  fail if the squaring convention is "cleaned up." Amplitude-vs-intensity loss
  split is intentional.
- **Every behavior-changing phase re-baselines metrics.** Stitch window differs
  per path (stitch_crop_size=20 CLI, middle_trim=32 barycentric, M=20 TF); any
  unification shifts every NCC/MAE score — reconcile the window knob explicitly.

---

## Canonical Target (one path per operation)

| Operation | Canonical (keep) | Deprecate/delete |
|---|---|---|
| Reassembly — torch inference | `reassembly.py:1040 reconstruct_image_barycentric` (probe-weighted + VarPro s1/s2) | reassembly_alpha/beta, beta_modules/reassembly, in-module `VectorizedBarycentricAccumulator:722`, `reconstruct_image:185`, `_weighted` alias |
| Reassembly — torch training forward | `helper.py:38 reassemble_patches_position_real` (central_mask) | `torch/tf_helper.py:204` (dead port). **Keep** `helper.py:197 _probe` (gated live) |
| Reassembly — TF | `tf_helper.py:1335/1190/1104` (stable-core, untouched) | only OLD siblings + fix STITCH-GRIDSIZE-001 guard |
| Training driver | `workflows/components.py:810 _train_with_lightning` | `train.py:372 main_lightning`, fold+retire `train_lightning_only.py::main` |
| Inference entrypoint | `reconstruct_image_barycentric` behind one wrapper | `inference.py:320` UNIFORM stitch (rewire), `:122 load_and_predict`, `api_helper:510 predict_only` |
| Config source of truth | TF dataclasses `ptycho/config/config.py` + single `config_bridge` + write-only `params.cfg` | JSON loader `utils.py:132`, dead `.get()/.set()` consumers, duplicate optimizer homes |
| Data loader | `dataloader.py:171 PtychoDataset` | `dset_loader_pt_mmap.py:51`, single-file `datagen.py` |
| Scaling / VarPro | `solve_lbfgs:656` + `accumulate_batch_from_basis:370` + `compute_varpro_basis:986` (norm='ortho') | `solve/solve_quadratic_direct/solve_autograd`, single-mode `accumulate_batch:328`, 2 of 3 `enforce_physics_constraint` copies |

---

## Design Gates (resolve BEFORE the gated phase)

- **Gate A — inference frame+window reconciliation (blocks Phase 2b).** The CLI
  front-end uses COM-relative `dx=x-mean(x)` + window=stitch_crop_size(20); the
  barycentric core uses `coords_global-center_of_mass+canvas_center` +
  middle_trim(32). These must be reconciled and the NCC/MAE metrics re-baselined
  *before* swapping the stitch, or scores shift silently. Needs a short design
  note + baseline run.
- **Gate B — historical legacy-amplitude anchor (blocks closing the legacy scope
  of REASSEMBLY-BRIDGE-001).**
  The barycentric path has no absolute anchor (s1~0.0886 vs sensible ~4.7). Wiring
  `derive_intensity_scale_from_amplitudes` into `apply_varpro_canvas_scaling` is a
  **separate design task**, not part of the mechanical consolidation. This plan
  *stabilizes and unifies* the solve surface so the anchor has one place to land;
  it does not itself close the legacy gap. This gate does not block the current CI
  path, whose absolute-scale contract is already normative in
  `docs/specs/spec-ptycho-core.md`.
- **Gate C — training-driver convergence parity (blocks Phase 3).** patience
  (10/100/config) and find_learning_rate-applied-or-not change convergence.
  Freeze loss-curve fixtures on the current `_train_with_lightning` before
  retiring any driver.

---

## Phase 0 — Delete confirmed-dead modules (effort S, risk LOW)

**Precondition:** grep-verify zero importers against current HEAD (re-run, do not
trust the audit's snapshot).

**Files (delete):**
- `ptycho_torch/reassembly_alpha.py`
- `ptycho_torch/reassembly_beta.py` (1633 lines)
- `ptycho_torch/beta_modules/reassembly.py` (re-export shim)
- `torch/tf_helper.py` + its test `tests/torch/test_tf_helper.py` (delete together)
- `ptycho_torch/dset_loader_pt_mmap.py` (dead PtychoDataset sibling)
- `ptycho_torch/datagen.py` (single-file module shadowed by `datagen/` package)

**Files (edit):**
- `ptycho_torch/reassembly.py` — remove `reconstruct_image_barycentric_weighted`
  alias (:1323) and the never-instantiated `VectorizedBarycentricAccumulator` (:722).

- [ ] **Step 0.1:** For each target, run `grep -rn "import.*<module>" --include=*.py`
      and `grep -rn "<module>\." ` across `ptycho/ ptycho_torch/ scripts/ tests/`.
      Record the grep output. Any live importer → STOP, reclassify that file.
- [ ] **Step 0.2:** Delete the six modules + the paired dead test.
- [ ] **Step 0.3:** Remove the two symbols from `reassembly.py`; grep for their
      names repo-wide to confirm no caller.
- [ ] **Step 0.4 historical gate (superseded):** current work runs claim-matched
      selectors per task and `bash ci/run_ci_tests.sh` only at the boundary named
      by the live roadmap.
- [ ] **Step 0.5:** Commit. `refactor: delete zero-importer dead reassembly/loader modules`

**Exit criteria:** ~3000 lines gone, gate green, no import errors.

---

## Phase 1 — Cheap correctness/hygiene defects (effort S, risk LOW)

Independent single-point fixes. Each is a self-contained commit.

- [ ] **1.1** `scripts/inference/inference.py:778` — the unified CLI pytorch
      branch passes `debug_dump_dir=` to `_run_inference_and_reconstruct`
      (`inference.py:320`), which has no such param → latent `TypeError`. Add the
      param (thread it to the debug dump) or drop the arg. **This unblocks Phase 2**
      (the CLI torch branch is currently unrunnable). Add a smoke test that
      exercises the branch.
- [ ] **1.2** `ptycho_torch/inference.py:415` — delete the `output_scale_factor`
      blend (computed then never passed to `forward_predict`).
- [ ] **1.3** `ptycho_torch/reassembly.py:1316-1317` — delete the dead diagnostic
      clobber (`modified_scaled_canvas` overwritten by identity `real+1j*imag`).
- [ ] **1.4** `ptycho_torch/config_params.py:238/:243` — collapse the duplicate
      `output_dir` declaration (second silently wins) to one field. Grep both
      names to confirm the documented `training_outputs` was already dead.
- [ ] **1.5** `ptycho/params.py:124` — gate the unconditional
      `DEBUG: Setting <k>` print behind a debug flag/logger.
- [ ] **1.6** `ptycho_torch/reassembly.py:1082` — `reconstruct_image_barycentric`
      overwrites its `gpu_ids` arg with `cuda.device_count()`. All callers pass
      None; drop the param from the signature (folds into Phase 2's wrapper).
- [ ] **1.7** `ptycho_torch/api/api_helper.py:510 predict_only` — calls
      `forward_predict(batch)` with 1 arg vs the 4-arg contract
      `(x, positions, probe, input_scale_factor)`. Broken/dead: delete or repair
      to the real contract; grep for callers first. **Current override:** because
      the API package has current public-contract tests, deletion requires an
      explicit migration decision; this historical item does not authorize it.
- [ ] **Historical gate (superseded):** run claim-matched selectors per task and
      the checked-in CI harness only at the boundary named by the live roadmap.

**Exit criteria:** CLI torch inference branch runs without TypeError; gate green.

---

## Phase 2 — Stabilize solver, then unify inference

### Phase 2a — Single VarPro solve + one enforce_physics_constraint (effort M, risk MED)

**Freeze first:** before deleting any solver sibling, add a fixture test that
pins the current `solve_lbfgs` s1/s2 outputs for a fixed input (Hazard 5 — the
solvers are NOT interchangeable; `solve` eigen-projects onto a rank-1 cone,
`solve_lbfgs` returns the unconstrained minimizer).

- [ ] **2a.1** Write `tests/torch/test_varpro_solver_frozen.py`: run
      `solve_lbfgs` on a small fixed accumulator state, assert s1/s2 within tol
      of recorded values.
- [ ] **2a.2** Extract the eigen-projection `enforce_physics_constraint` into one
      shared helper; import it in `reassembly.py` and `model.py`. **Do NOT** touch
      the `beta_modules/model.py` copy (stays forked pending multi-mode parity —
      Hazard 10). Preserve fixture identity and confirm the governed numerical
      behavior at the registered tolerance via the model.py loss fixtures.
- [ ] **2a.3** Delete `VarProScaler.solve`, `solve_quadratic_direct`,
      `solve_autograd`, and single-mode `accumulate_batch:328` (keep
      `accumulate_batch_from_basis:370`). Grep for callers first.
- [ ] **2a.4** Run torch gate + the new frozen fixture. Commit.

### Phase 2b — Collapse barycentric call sites + unify CLI stitch (effort L, risk HIGH — Gate A)

> **Superseded implementation recipe (2026-07-15):** `60d2c7c7b` landed
> `_resolve_reassembly_route` plus a dataset-backed
> `_run_barycentric_inference_and_reconstruct` path instead of the pure stitch-core
> extraction specified below. The steps in this subsection are historical. Any
> continuation must first audit the landed route, the still-separate uniform path,
> remaining direct call sites, and the CLI `debug_dump_dir` mismatch, then issue a
> fresh plan.

**Precondition: Gate A resolved** (frame+window reconciled, metrics re-baselined).

- [ ] **2b.1** Introduce one wrapper (e.g. `ptycho_torch/inference.py:
      reconstruct(...)`) fronting `reconstruct_image_barycentric`. Repoint the five
      copy-paste call sites (`base_api.py:1232`, `api_helper.py:524`,
      `workflows/components.py:1336`, `inference.py:186`,
      `varpro_probe_ablation_runner.py:413`) at it. No behavior change yet —
      verify studies + api tests unchanged.
- [ ] **2b.2** Rewire `_run_inference_and_reconstruct` (`inference.py:320,460`) to
      delegate its stitch to the wrapper (honoring patch_weighting/varpro_scaling)
      instead of the UNIFORM `reassemble_patches_position_real`. Keep the COM-offset
      front-end; apply the Gate-A frame/window reconciliation.
- [ ] **2b.3** Add a regression test: CLI path and in-process path produce matching
      amplitudes for one checkpoint (this is the exact divergence the studies exist
      to characterize).
- [ ] **2b.4** Retire `load_and_predict` (`inference.py:122`, MLflow, sets
      `training=True` before inference) — grep callers, delete or route through
      the wrapper. Run gate + re-baseline NCC/MAE. Commit.

**Exit criteria:** one inference entrypoint; CLI==in-process amplitude regression
test passes; metrics re-baselined and recorded in the plan.

---

## Phase 3 — Converge training drivers (effort L, risk HIGH — Gate C)

**Precondition: Gate C resolved** (loss-curve fixtures frozen on
`_train_with_lightning`).

- [ ] **3.1** Freeze a loss-curve fixture: short deterministic run through
      `_train_with_lightning`, record per-epoch loss.
- [ ] **3.2** Port `train_lightning_only`'s rms/probe-scaling fix (the "semantic
      scaling gap", `varpro_probe_ablation_runner.py:7-9`) into
      `_train_with_lightning`. Preserve manual-optimization + EXEC-ACCUM-001 guard.
- [ ] **3.3** Reconcile the divergent knobs explicitly: patience (10/100/config →
      one config-driven value), find_learning_rate LR-rescale (applied by 2 of 4
      drivers → decide + apply uniformly), accelerator (execution_config).
      Document the chosen defaults.
- [ ] **3.4** Route studies (`grid_lines_torch_runner`, varpro runner) through the
      unified driver; retire `main_lightning` (`train.py:372`, no in-repo caller)
      and `train_lightning_only::main`.
- [ ] **3.5** Run gate + loss-curve fixture (must match within tol) + one study
      smoke. Commit.

**Exit criteria:** one training path, one set of defaults; loss fixture stable;
CLI-vs-study fork removed.

---

## Phase 4 — Config single source of truth (effort L, risk HIGH)

Can run parallel to Phase 3. **Land behind fixture parity** — any default flip
(gridsize, intensity_scale_trainable) changes trained-model behavior.

- [ ] **4.1** Delete the JSON loader path `utils.py:132 load_config_from_json` +
      `validate_and_process_config` and its dead `.get()/.set()` singleton
      consumers (they `AttributeError` if exercised — plain dataclasses now).
      Grep callers first.
- [ ] **4.2** Collapse duplicate optimizer/scheduler/clip homes
      (scheduler, gradient_clip_val/algorithm, accum_steps, learning_rate,
      plateau_*) declared on all three of TF TrainingConfig / torch TrainingConfig /
      PyTorchExecutionConfig → pick `execution_config` for runtime knobs.
- [ ] **4.3** Add unknown-key errors to `setup_configuration` and
      `update_existing_config` so typo'd knobs fail loud (currently silently
      dropped). **Guard:** do NOT tighten the `nphotons==1e5` HARD-RAISE
      (`config_bridge.py:290`) — disambiguate sentinel-vs-real 1e5 first (Hazard 9).
- [ ] **4.4** Do NOT change `update_legacy_dict` None-skip semantics
      (`config.py:778`) without auditing all ~24 call sites (Hazard 8) — out of
      scope for this phase; note as follow-up.
- [ ] **4.5** Run gate + a config-bridge parity fixture (train + inference payload
      round-trip). Commit.

**Exit criteria:** one bridge; typo'd keys fail loud; reverse-bridge allow-list
(`components.py:886-980`) no longer load-bearing.

---

## Phase 5 — Retire the stitch_predictions fork (effort M, risk MED)

**Precondition:** Phases 2–4 landed. **Care:** `stitch_predictions` carries the
`norm_Y_I` amplitude anchor (`grid_lines_workflow.py:864`) absent from
`stitch_data` — must be preserved.

- [ ] **5.1** Fix the STITCH-GRIDSIZE-001 one-line guard in
      `data_preprocessing.stitch_data:152` so `gridsize==1` is accepted.
- [ ] **5.2** Repoint grid_lines callers back to the canonical stitcher; port the
      `norm_Y_I` rescale so it is preserved.
- [ ] **5.3** Add a fixture asserting stitched output (incl. norm_Y_I scale)
      matches the pre-refactor `stitch_predictions` output. Run gate. Commit.

**Exit criteria:** one grid-stitch copy; norm_Y_I anchor preserved.

---

## Sequencing Summary

```
Phase 0 (dead code)  ──►  Phase 1 (defects) ──►  Phase 2a (solver)
                                                      │
                              Gate A ────────────────►├─► Phase 2b (inference)
Gate C ──► Phase 3 (training)  ── parallel ──  Phase 4 (config)
                                    │
                                    └──►  Phase 5 (stitch fork)
```

Phases 0–1 are safe to start immediately. 2b/3/4/5 are behavior-changing and each
gated on a frozen-fixture baseline + (2b/3) a design gate.

## Out of Scope (explicit)

- Closing the historical legacy-amplitude ~53× gap (Gate B /
  REASSEMBLY-BRIDGE-001) — separate
  design task; this plan only makes one landing site for the anchor.
- Unifying TF↔torch coords/axis/sign conventions — load-bearing invariants
  (Global Constraints); would need atomic fixture migration.
- Merging the two `RectangularScaledDiffraction` forward models — coherent vs
  incoherent, agree only at P=1; blocked on multi-mode parity fixtures (Hazard 10).
- Any modification to STABLE-CORE TF files.

## Verification Protocol (every phase)

1. `pytest tests/torch -m "not slow"` with project deselect/ignore lists — must
   match the recorded green baseline (no new failures).
2. Behavior-changing phases: the phase's frozen-fixture parity test must pass.
3. Phases 2b/3/5: re-run one study/inference smoke and record NCC/MAE deltas in
   this plan before marking the phase complete.
4. Commit per step; commit messages carry NO AI/Claude attribution (repo rule).
