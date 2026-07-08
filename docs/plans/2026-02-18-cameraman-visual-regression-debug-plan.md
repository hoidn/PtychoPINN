# Cameraman Visual Regression Debug Plan Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Isolate and fix the remaining cameraman quality regression after enforcing `shift_sum` + fail-fast OOM behavior.

**Architecture:** Treat 2026-02-18 as two rounds. Round 1 established the backend/runtime path behavior and landed a guardrail fix. Round 2 focuses on residual quality drift under a fixed runtime path, using controlled replays (fixed checkpoint/data) and narrow hypothesis tests (data semantics, coordinate frame/precision, checkpoint/stitch path).

**Tech Stack:** Python 3.11, NumPy, PyTorch/TensorFlow runtime, `scripts/studies/*`, pytest, git, existing `tmp/debug/*` artifacts.

---

## Round 1 Outcomes (Locked Baseline)

### Confirmed findings
1. Hybrid runtime path behavior is real and reproducible at hash level:
   - `auto` == `shift_sum` hash `5a5ec397...`
   - `batched` hash `0729fa0a...` (matches historical bad postfix-style artifact)
   - Evidence: `tmp/debug/hybrid_backend_matrix_2026-02-18.json`
2. Commit replay shows first hash transition at `6ccc6e2b`.
   - Evidence: `tmp/debug/replay_commit_matrix_2026-02-18.md`
3. Data semantics are a secondary contributor (non-trivial A/B delta).
   - Evidence: `tmp/debug/downsample_semantics_ablation_2026-02-18.md`

### Implemented mitigation (already landed)
- Commit `66176f37`:
  - Added `allow_oom_fallback` control to position reassembly.
  - Threaded it through hybrid cross-dataset inference + manifest.
  - Forced `allow_oom_fallback=False` in NERSC orchestration.
  - Maintains `position_reassembly_backend="shift_sum"` for study runs.

### Round 2 non-negotiables
- Hybrid debug runs must use `position_reassembly_backend="shift_sum"`.
- Hybrid debug runs must set `allow_oom_fallback=False`.
- Any OOM is a hard failure (no silent reassembly path switch).

---

## Updated Hypothesis Ledger (Round 2)

| ID | Hypothesis | Status | Why it still matters | Round 2 test |
|---|---|---|---|---|
| H1 | Hybrid visual collapse is primarily backend path switching | Mitigated | Guardrail is in, but needs full-run validation | Post-fix replay parity + manifest audit |
| H2 | Downsample/data semantics drive remaining quality drift | Open (high) | Measurable deltas persisted under fixed checkpoint | Controlled semantics matrix |
| H4 | Coordinate frame / conversion drift in adapter path | Open (high) | Could create edge artifacts independent of backend | Occupancy/OOB and centroid diagnostics |
| H5 | PtychoViT silent fallback stitch path | Open | Contract passed, but historical logs lacked stitch-path observability | Add explicit stitch-path logging + replay |
| H6 | PtychoViT checkpoint choice explains ptychovit arm degradation | Supported | Needs quantified effect under fixed data path | Fixed-data checkpoint A/B |
| H9 | Cached test bundle semantics mismatch training semantics | New | Can confound replay conclusions despite good losses | 2x2 train/infer semantics coupling matrix |
| H10 | GT selection key order (`YY_ground_truth`/`YY_full`/`objectGuess`) skews metrics | New | Can hide/overstate quality shifts | GT-source ablation metric pass |
| H11 | Offset dtype/rounding (`float32` vs `float64`) changes placement at N=128 | New | Could produce structured phase texture differences | Precision sensitivity sweep |

---

### Task 1: Post-Fix Baseline Replay (Guardrail Validation)

**Files:**
- Create: `tmp/debug/round2_postfix_baseline_2026-02-18.md`
- Create: `tmp/debug/round2_postfix_baseline_2026-02-18.json`

**Step 1: Re-run fixed-checkpoint inference with enforced strict runtime policy**
- Use the same checkpoint/test NPZ from Round 1 backend matrix.
- Force `backend=shift_sum`, `allow_oom_fallback=False`.

**Step 2: Capture manifest/runtime evidence**
- Record recon hash, metrics, and emitted manifest fields (`position_reassembly_backend`, `allow_oom_fallback`).

**Step 3: Compare with Round 1 oracle hashes**
- Confirm no batched-hash contamination in strict mode.

**Expected:** Strict-mode replay is deterministic and never emits batched-style recon hash.

---

### Task 2: Runtime Contract Hardening (TDD)

**Files:**
- Modify: `scripts/studies/hybrid_checkpoint_inference.py`
- Modify: `scripts/studies/grid_lines_torch_runner.py`
- Test: `tests/torch/test_hybrid_checkpoint_cross_dataset_inference.py`
- Test: `tests/torch/test_grid_lines_torch_runner.py`

**Step 1: Write failing tests**
- Assert manifest includes explicit runtime decision fields:
  - `requested_reassembly_backend`
  - `resolved_reassembly_backend`
  - `allow_oom_fallback`
  - `fallback_used`
- Assert strict mode (`allow_oom_fallback=False`) never reports fallback used.

**Step 2: Run failing selectors**
Run:
```bash
pytest -q tests/torch/test_hybrid_checkpoint_cross_dataset_inference.py -k "manifest and fallback"
pytest -q tests/torch/test_grid_lines_torch_runner.py -k "oom and fallback"
```
Expected: RED (missing fields/behavior).

**Step 3: Implement minimal runtime evidence wiring**
- Thread resolved backend + fallback-used flags into manifest output.

**Step 4: Re-run tests**
Expected: GREEN.

**Step 5: Commit**
```bash
git add scripts/studies/hybrid_checkpoint_inference.py scripts/studies/grid_lines_torch_runner.py \
  tests/torch/test_hybrid_checkpoint_cross_dataset_inference.py tests/torch/test_grid_lines_torch_runner.py
git commit -m "Add explicit reassembly runtime contract evidence to manifests"
```

---

### Task 3: Semantics/Cached-Bundle Coupling Matrix (H2 + H9)

**Files:**
- Create: `tmp/debug/run_semantics_cache_matrix.py`
- Create: `tmp/debug/semantics_cache_matrix_2026-02-18.json`
- Create: `tmp/debug/semantics_cache_matrix_2026-02-18.md`

**Step 1: Build 2x2 controlled matrix**
- Train semantics: `{current_binning, legacy_crop}`
- Inference cached-bundle semantics: `{current_binning, legacy_crop}`
- Hold checkpoint family and runtime path fixed (`shift_sum`, no fallback).

**Step 2: Measure outputs**
- recon hash
- `amp_mae`, `complex_mae`, wrapped phase MAE
- optional LPIPS/SSIM if available in existing study tooling

**Step 3: Rank effect size**
- Quantify train-side vs infer-side semantic contribution.

**Expected:** Clean separation between data semantics effects and runtime-path effects.

---

### Task 4: Coordinate Frame + Precision Diagnostics (H4 + H11)

**Files:**
- Create: `tmp/debug/check_coords_frame_precision_2026-02-18.py`
- Create: `tmp/debug/coords_frame_precision_2026-02-18.md`

**Step 1: Add diagnostics on fixed patches/offsets**
- occupancy heatmap summary
- OOB percentage
- patch-centroid displacement vs expected

**Step 2: Precision sweep**
- Reassemble with offsets in `float32` vs `float64`.
- Report complex/amp/phase deltas.

**Step 3: Frame sanity checks**
- Validate origin convention and axis direction for scan807 + cameraman pairs.

**Expected:** Either clear frame/precision fault signature or explicit refutation.

---

### Task 5: PtychoViT Residual Track Isolation (H5 + H6)

**Files:**
- Modify: `scripts/studies/ptychovit_bridge_entrypoint.py`
- Test: `tests/studies/test_ptychovit_bridge_entrypoint_invocation.py`
- Create: `tmp/debug/ptychovit_checkpoint_matrix_2026-02-18.md`

**Step 1: Write failing tests for bridge evidence**
- Ensure inference artifacts/manifest include:
  - checkpoint path + sha256
  - stitch path used (`position-aware` vs fallback)

**Step 2: Implement minimal bridge instrumentation**
- Emit evidence without changing reconstruction math.

**Step 3: Fixed-data checkpoint A/B replay**
- Compare `run145` vs historical “good” checkpoint on identical inputs.

**Expected:** Quantified ptychovit checkpoint effect and stitch-path observability closure.

---

### Task 6: GT-Source Ablation + Decision Gate (H10 + closeout)

**Files:**
- Create: `tmp/debug/gt_source_ablation_2026-02-18.py`
- Create: `tmp/debug/gt_source_ablation_2026-02-18.md`
- Create: `tmp/debug/round2_decision_gate_2026-02-18.md`

**Step 1: GT-source ablation**
- Evaluate metrics using each available GT key (`YY_ground_truth`, `YY_full`, `objectGuess`) where present.
- Report ranking stability across sources.

**Step 2: Final decision gate**
- Mark each open hypothesis Confirmed/Refuted/Inconclusive.
- Rank primary/secondary residual causes.

**Step 3: Write next implementation brief**
- If H2/H9 dominant: patch data prep/cached bundle semantics alignment.
- If H4/H11 dominant: patch coordinate conversion/precision contract.
- If H6 dominant: pin default checkpoint policy + manifest enforcement.

---

## Verification Checklist (Round 2)

- Strict-mode replay artifact exists and proves `shift_sum` + `allow_oom_fallback=false`.
- Runtime manifest contains explicit requested/resolved backend and fallback-used evidence.
- Semantics/cached-bundle 2x2 matrix completed with effect-size ranking.
- Coordinate frame/precision diagnostics completed with OOB/centroid evidence.
- PtychoViT checkpoint matrix produced with stitch-path observability.
- GT-source ablation completed to deconfound metric interpretation.
- Round 2 decision gate identifies primary residual cause(s) and immediate patch target.
