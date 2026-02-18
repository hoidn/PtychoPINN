# Cameraman Visual Regression Debug Plan Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Isolate the exact commit-level and runtime-path cause of the cameraman visual regression (Image #1 -> Image #2) using historical study outputs plus git versioning.

**Architecture:** Use a git-first, last-24h commit-window strategy. Treat historical outputs as oracle artifacts, then run deterministic inference-only replays and targeted ablations (backend, downsample semantics, harmonization) to identify one causal track. Keep provenance minimal and practical.

**Tech Stack:** Python 3.11, NumPy, PyTorch/TensorFlow runtime, existing study scripts (`scripts/studies/*`), pytest, git (`log`, `show`, `bisect`).

## Original Symptoms (User-Reported Contract)

The plan must preserve these concrete symptoms as the regression contract:

1. Cameraman `pinn_hybrid_resnet` phase appears checkerboard/periodic with repeated-looking local structure (suggesting patch-level placement or model-collapse artifact).
2. Cameraman `pinn_hybrid_resnet` reconstruction quality is visibly worse than the prior "good-ish" output, not just different color scaling.
3. Cameraman `pinn_ptychovit` appears overly flat/saturated in failing outputs (loss of object detail).
4. User hypothesis explicitly included possible batched-path placement bug (e.g., corner/crop-centering/coordinate-offset mishandling), so this must be tested directly rather than inferred from aggregate metrics alone.

**Primary visual references:**
- Image #1 style run: `outputs/nersc_scan807_cameraman_study_smoke/cameraman256/visuals/compare_amp_phase.png`
- Image #2 style run: `outputs/nersc_scan807_cameraman_study_downsample_flip_full_20260217_171730/cameraman256/visuals/compare_amp_phase.png`
- Original-good comparison run (historical anchor): `outputs/grid_lines_external_fly001_n128_top_train_full_test_e40_seed3_cnn_hybrid_resnet_rerun_20260216_213242_pty/`

---

## Current Hypothesis Ledger (git-window focused)

| ID | Hypothesis | Status | Evidence so far | Next elimination/confirmation test |
|---|---|---|---|---|
| H1 | Hybrid regression is reassembly-backend divergence (`auto` vs `shift_sum` vs `batched`), specifically corner/crop-centering/offset handling error in batched placement | Open / high-priority | `6ccc6e2b` changed position reassembly behavior and fallback handling; user-observed repeated/checkerboard phase is consistent with misplacement class | Run fixed-checkpoint backend matrix **plus per-patch placement parity** (same patches/coords, backend-only swap) |
| H2 | Regression is data-prep semantic change from `7df62bbf` (diffraction binning + object/probe crop) | Open / high-priority | Full run retrained on changed prep policy; this can affect both model arms | Rebuild prep payloads pre/post semantic flip and replay same checkpoints |
| H3 | Shared post-processing path (harmonize/resize) is degrading both models after inference | Open | Both models go through shared harmonization/evaluation path | Compare raw `recon.npz` vs harmonized render/metrics path from same files |
| H4 | NERSC coordinate-frame conversion introduced drift in adapter path | Open | New adapter path converts meter-space positions to pixel offsets | Run coordinate sanity diagnostics and OOB/occupancy checks for identical scans |
| H5 | PtychoViT bridge is taking fallback stitching path silently | Open | Bridge has broad exception fallback to simpler stitcher | Add stitch-path logging + run contract gate on historical runs |
| H6 | PtychoViT change is checkpoint-only (`run145` vs `ptychovit_initial_fresh_stitched`) | Supported but not isolated | Historical runs use different checkpoints and produce different hashes | Hold data fixed and swap checkpoint only |
| H7 | Hybrid change is partially runtime nondeterminism (same checkpoint, different output hash) | Open | Historical postfix run differs from full run hybrid recon despite similar setup | Force deterministic settings and replay identical checkpoint + NPZ |
| H8 | Visual severity is mostly plotting-scale/layout drift | Open / lower-priority | Image #1 vs #2 layout differs | Re-render with fixed ranges only after H1-H7 checks |

---

### Task 1: Freeze Oracle Outputs + Commit Window

**Files:**
- Create: `tmp/debug/oracle_runs_2026-02-18.md`
- Create: `tmp/debug/oracle_hashes_2026-02-18.json`

**Step 1: Capture oracle run artifacts (no broad provenance sweep)**

Run:
```bash
python - <<'PY'
import hashlib, json
from pathlib import Path

runs = {
  "smoke": "outputs/nersc_scan807_cameraman_study_smoke",
  "full": "outputs/nersc_scan807_cameraman_study_downsample_flip_full_20260217_171730",
  "postfix": "outputs/nersc_scan807_cameraman_study_downsample_flip_full_20260217_171730_postfix_20260217_214635",
  "historical_good_anchor": "outputs/grid_lines_external_fly001_n128_top_train_full_test_e40_seed3_cnn_hybrid_resnet_rerun_20260216_213242_pty",
}
targets = {}
for name, root in runs.items():
    r = Path(root)
    items = [
        r / "cameraman256/recons/pinn_ptychovit/recon.npz",
        r / "cameraman256/recons/pinn_hybrid_resnet/recon.npz",
        r / "cameraman256/metrics_by_model.json",
        r / "cameraman256/visuals/compare_amp_phase.png",
    ]
    targets[name] = {}
    for p in items:
        if p.exists():
            h = hashlib.sha256(p.read_bytes()).hexdigest()
            targets[name][str(p)] = h
Path("tmp/debug").mkdir(parents=True, exist_ok=True)
Path("tmp/debug/oracle_hashes_2026-02-18.json").write_text(json.dumps(targets, indent=2))
print("wrote tmp/debug/oracle_hashes_2026-02-18.json")
PY
```
Expected: One JSON with stable hashes for oracle files.

**Step 2: Bound commit search to last 24h high-risk changes**

Run:
```bash
git log --since='2026-02-17 00:00' --until='2026-02-18 23:59' --date=iso --oneline \
  -- scripts/studies/prepare_nersc_hybrid_dataset.py \
     scripts/studies/grid_lines_torch_runner.py \
     ptycho/tf_helper.py \
     scripts/studies/grid_lines_compare_wrapper.py \
     ptycho/workflows/grid_lines_workflow.py
```
Expected: Suspect list includes `7df62bbf`, `6ccc6e2b`, `3528933d`, `0a6693c7`.

---

### Task 2: Shared-Path Control (Raw Recon vs Harmonized Path)

**Files:**
- Create: `tmp/debug/check_shared_postprocess_path.py`
- Create: `tmp/debug/shared_path_control_2026-02-18.md`

**Step 1: Compare raw recon arrays before any harmonization**

Implement script to:
1. Load `YY_pred` from oracle `recon.npz` files.
2. Report shape, amp mean/std/q99, phase mean/std.
3. Compute pairwise MAE (smoke vs full, full vs postfix) for each model.

Expected:
- If raw arrays already collapse, issue is pre-harmonization.
- If raw arrays look reasonable but final visuals degrade, shared postprocess path is suspect.

---

### Task 3: Hybrid Reassembly Backend Matrix (H1/H7)

**Files:**
- Create: `tmp/debug/run_hybrid_backend_matrix.py`
- Create: `tmp/debug/hybrid_backend_matrix_2026-02-18.json`

**Step 1: Deterministic fixed-checkpoint replay across backends**

Run matrix for one fixed hybrid checkpoint and one fixed cached test NPZ:
- `position_reassembly_backend=shift_sum`
- `position_reassembly_backend=batched`
- `position_reassembly_backend=auto`

For each run capture:
- output recon hash
- amplitude/phase summary stats
- metrics vs fixed GT recon

**Step 2: Patch-placement parity subtest (directly targets user hypothesis)**
- On a fixed subset of predicted patches and fixed `coords_offsets`, run backend-only reassembly swap:
  - `shift_sum` reference
  - `batched` candidate
- Compare:
  - per-pixel complex delta map
  - per-patch contribution centroid/footprint parity
  - top-k worst scan indices for placement mismatch

Expected:
- Near-zero placement deltas refute corner/crop-centering offset bug class.
- Structured deltas (especially edge/corner-biased) support placement bug class.

Expected:
- Large backend-dependent deltas confirm H1.
- Identical deltas across backends refute H1 and prioritize data/checkpoint tracks.

---

### Task 4: Downsample Semantic Flip Ablation (H2/H4)

**Files:**
- Create: `tmp/debug/ablate_downsample_semantics.py`
- Create: `tmp/debug/downsample_semantics_ablation_2026-02-18.md`

**Step 1: Rebuild prep outputs with alternate semantics**

From the same paired HDF5 inputs:
1. Produce payload A (current policy: diffraction binning + object/probe center crop).
2. Produce payload B (legacy-like comparison policy).
3. Keep coords handling explicit and log frame assumptions.

**Step 2: Replay identical checkpoints on A vs B**
- PtychoViT checkpoint fixed.
- Hybrid checkpoint fixed.

Expected:
- Major A/B shift with fixed checkpoints confirms data-prep semantics as causal.

---

### Task 5: PtychoViT Stitch-Path + Contract Gate (H5/H6)

**Files:**
- Create: `tmp/debug/check_ptychovit_stitch_and_contract.py`
- Create: `tmp/debug/ptychovit_contract_gate_2026-02-18.json`

**Step 1: Run contract checks on oracle run artifacts**

Validate:
1. paired HDF5 key/shape requirements
2. position-vector scan-count parity
3. runtime normalization config keys present and valid
4. stitch path used (upstream placement vs fallback)

Expected:
- Any fallback stitch usage or contract violation elevates H5.

---

### Task 6: Commit Replay Matrix (Inference-Only)

**Files:**
- Create: `tmp/debug/replay_commit_matrix_2026-02-18.md`
- Runtime outputs: `outputs/nersc_commit_replay_<shortsha>_<timestamp>/`

**Step 1: Replay same inference inputs across suspect commits**

Suspect commits:
- `0a6693c7`
- `3528933d`
- `7df62bbf`
- `6ccc6e2b`

For each commit:
1. Run inference-only replay on fixed artifacts.
2. Record recon hash + compact metric set (`mae`, `ssim`, `ms_ssim`) for both models.

Expected:
- One or two commits show first major regression jump.

---

### Task 7: Decision Gate + Bounded `git bisect`

**Files:**
- Create: `tmp/debug/regression_decision_gate_2026-02-18.md`
- Create: `tmp/debug/bisect_oracle_2026-02-18.sh`
- Create: `tmp/debug/bisect_report_2026-02-18.md`

**Step 1: Decision gate from H1-H8**
- Mark each hypothesis Confirmed/Refuted/Inconclusive.
- Assign root-cause ranking (primary + optional secondary contributors):
  - `track-A`: reassembly/runtime backend behavior
  - `track-B`: data-prep/coordinate semantic change
  - `track-C`: checkpoint-only/model-quality shift
  - `track-D`: shared postprocessing/visualization path
- If multiple tracks contribute, explicitly quantify each contribution using fixed-input ablation metrics.

**Step 2: Bisect only if decision gate is inconclusive**

Run:
```bash
git bisect start <known-bad-sha> <known-good-sha>
git bisect run bash tmp/debug/bisect_oracle_2026-02-18.sh
git bisect reset
```
Oracle contract:
- `0` = good
- `1` = bad
- `125` = skip

Expected:
- First bad commit or explicit nondeterminism proof in bounded window.

---

## Guardrails
- Prioritize historical outputs and git code history over broad provenance paperwork.
- No fixes until one hypothesis is confirmed with reproducible evidence.
- Use fixed checkpoints and fixed NPZ/HDF5 inputs for all replay tests.
- Keep debug scripts and summaries in `tmp/debug/`.

## Verification Checklist
- Oracle hash bundle exists for smoke/full/postfix runs.
- Oracle hash bundle includes historical good-anchor run (when artifact exists).
- Backend matrix compares `shift_sum`/`batched`/`auto` under fixed checkpoint+NPZ.
- Backend matrix includes direct patch-placement parity subtest for corner/crop-centering offset hypothesis.
- Downsample semantics ablation compares A/B with identical checkpoints.
- Decision gate classifies H1-H8 and ranks primary/secondary root-cause tracks.
- Bisect report exists if and only if decision gate is inconclusive.
