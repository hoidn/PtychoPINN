# Cameraman Visual Regression Debug Plan Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Determine why cameraman recon visuals regressed from the semi-reasonable heatmaps (Image #1) to the checkerboard/flat outputs (Image #2), and isolate the exact causal code/data/config change(s).

**Architecture:** Treat this as a provenance + contract debugging problem. First lock down which runs produced Image #1 vs Image #2 and enumerate all differences (code commits, checkpoints, inputs, preprocessing, plotting). Then test one hypothesis at a time with minimal, reproducible diagnostics that either eliminate or confirm a cause.

**Tech Stack:** Python 3.11, NumPy, PyTorch/TensorFlow runtime, existing study scripts (`scripts/studies/*`), pytest, git history.

---

## Current Hypothesis Ledger (as-of now)

| ID | Hypothesis | Status | Evidence so far | Next elimination/confirmation test |
|---|---|---|---|---|
| H1 | Batched reassembly regression causes Image #2 artifacts | **Refuted (for scan807)** | `outputs/scan807_parity_postfix_20260217_215744/summary.json` shows `parity_pass=true`, `amp_mae~1.4e-8`, `complex_allclose=true` | Re-run same parity test on cameraman to fully close H1 for both datasets |
| H2 | Inference batching feeds wrong/repeated diffraction patches | **Weakly refuted** | Permutation test on cameraman inference produced very small diffs (`mean abs diff ~1e-4`, not catastrophic mismatch) | Add strict batch-routing diagnostic: compare outputs for hand-picked index set with 1-image batches vs large batches |
| H3 | Input diffraction patches are accidentally duplicated/collapsed | **Partially refuted / still open** | No exact duplicates found; but cameraman diffraction cosine similarity is very high (~0.997), so patches are distinct but highly similar | Compare raw HDF5 diffraction diversity vs cached NPZ diversity to detect preprocessing collapse |
| H4 | Coordinates/reassembly frame contract is wrong | **Open** | Reassembling cached `Y_I/Y_phi` against GT gave large error, but this test may be invalid if those tensors are not object patches in this external path | Validate exact semantic contract of `Y_I/Y_phi` in external mode; if valid, this strongly implicates data prep/coords |
| H5 | Image #1 and Image #2 came from different runs with different artifacts and are not directly comparable | **Supported** | Image sizes/layout differ (3000x1200 with probe panel vs 2250x1200 without), matching different output dirs | Identify exact source dirs for both images and pin artifact lineage in a table |
| H6 | PtychoViT regression is from checkpoint swap, not reassembly | **Supported** | Smoke run uses `/datasets/run145/best_model.pth`; full run uses `tmp/ptychovit_initial_fresh_stitched/...`; recon hashes differ | Replay full run with smoke checkpoint only; compare ptychovit recon directly |
| H7 | Hybrid regression is from checkpoint/data-prep change (downsample policy flip + retrain), not plotting | **Supported** | Smoke/full hybrid model checkpoints differ (different SHA), recons differ strongly (`amp_mae~1.23`) | Isolate factors: run full inference with smoke hybrid checkpoint on full cached NPZs |
| H9 | Image #1 -> Image #2 regression is confounded by crop/bin semantic flip (real-space crop + diffraction bin) rather than reassembly/stitching behavior | **Open / high-priority** | Policy changed between runs (`7df62bbf`) and full run was retrained on different prepared data; this can alter both ptychovit inputs and hybrid training distribution | Hold reassembly backend fixed and run a semantics ablation: rebuild cameraman/scan807 cached NPZs with pre-flip vs post-flip prep, then replay identical checkpoints on each and compare recon deltas |
| H8 | Visual regression is mostly plotting-scale/style drift | **Open** | Probe panel and colormap range differ across runs; numeric metrics do not always align with visual severity | Render both runs with a fixed color-scale plotting script and compare |

---

### Task 1: Lock Image Provenance (No ambiguity)

**Files:**
- Create: `tmp/debug/image_provenance_2026-02-18.md`

**Step 1: Record candidate source run for Image #1 and Image #2**

Run:
```bash
python - <<'PY'
from pathlib import Path
from PIL import Image
runs=[
 'outputs/nersc_scan807_cameraman_study_smoke',
 'outputs/nersc_scan807_cameraman_study_downsample_flip_full_20260217_171730',
 'outputs/nersc_scan807_cameraman_study_downsample_flip_full_20260217_171730_postfix_20260217_214635',
]
for r in runs:
    p=Path(r)/'cameraman256/visuals/compare_amp_phase.png'
    if p.exists():
        print(r, Image.open(p).size, p.stat().st_mtime)
PY
```
Expected: unique layout signature identifies which run matches each image.

**Step 2: Capture invocation + checkpoint lineage for identified runs**

Run:
```bash
cat <run>/invocation.sh
cat <run>/cameraman256/runs/pinn_ptychovit/invocation.sh
sha256sum <run>/hybrid_training/runs/pinn_hybrid_resnet/model.pt
```
Expected: explicit table of changed checkpoints/inputs.

---

### Task 2: Close H1 For Cameraman (reassembly parity)

**Files:**
- Runtime artifacts: `outputs/cameraman_parity_postfix_<timestamp>/`

**Step 1: Execute parity replay on cameraman**

Run:
```bash
PYTHONPATH=. /home/ollie/miniconda3/envs/ptycho311/bin/python \
  scripts/studies/position_reassembly_checkpoint_replay.py \
  --model-pt outputs/nersc_scan807_cameraman_study_downsample_flip_full_20260217_171730/hybrid_training/runs/pinn_hybrid_resnet/model.pt \
  --train-npz outputs/nersc_scan807_cameraman_study_downsample_flip_full_20260217_171730/hybrid_training/datasets/N128/gs1/train.npz \
  --test-npz outputs/nersc_scan807_cameraman_study_downsample_flip_full_20260217_171730/cameraman256/hybrid_cached/datasets/N128/gs1/test.npz \
  --output-dir outputs/cameraman_parity_postfix_<timestamp> \
  --dataset-name cameraman256 --architecture hybrid_resnet --n 128 --gridsize 1 \
  --batch-size 8 --learning-rate 2e-4 --infer-batch-size 128 --position-reassembly-batch-size 8 --seed 3
```
Expected: `summary.json` with pass/fail and concrete diff metrics.

**Step 2: Decide H1 state**
- If parity passes with tiny error: H1 closed as not-causal.
- If parity fails: open reassembly bug track with failing artifact.

---

### Task 3: Validate Inference Routing (H2)

**Files:**
- Create: `tmp/debug/check_inference_routing.py`
- Create: `tmp/debug/inference_routing_2026-02-18.json`

**Step 1: 1-image vs batched deterministic routing check**

Implement a diagnostic script that:
1. Picks 64 deterministic indices from cameraman cached NPZ.
2. Runs `run_torch_inference` once with `infer_batch_size=1` and once with `infer_batch_size=128` on exactly those samples.
3. Reports per-sample complex MAE and max diff.

Expected:
- Very small per-sample errors (`<1e-4`) indicates routing is correct.
- Large/sample-shifted errors indicate misindexing.

---

### Task 4: Compare Raw HDF5 vs Cached NPZ Diversity (H3)

**Files:**
- Create: `tmp/debug/compare_raw_vs_cached_diversity.py`
- Create: `tmp/debug/raw_vs_cached_diversity_2026-02-18.md`

**Step 1: Quantify diffraction diversity pre/post prep**

Compute for raw HDF5 and cached NPZ:
- per-patch mean/std distributions
- pairwise cosine similarity distribution (random subset)
- effective rank / PCA variance concentration

Expected:
- If cached NPZ diversity collapses sharply vs raw, preprocessing pipeline is suspect.
- If both are similarly homogeneous, model collapse may be expected from data regime.

---

### Task 5: Isolate Checkpoint/Data-Prep Deltas (H6, H7)

**Files:**
- Runtime artifacts under new output dirs:
  - `outputs/nersc_ablate_ptychovit_ckpt_<timestamp>/`
  - `outputs/nersc_ablate_hybrid_ckpt_<timestamp>/`

**Step 1: PtychoViT checkpoint ablation**
- Hold dataset fixed (full run working pair)
- Swap only checkpoint (`run145` vs `ptychovit_initial_fresh_stitched`)
- Compare recon metrics and visuals

**Step 2: Hybrid checkpoint ablation**
- Hold test dataset fixed (full run cached NPZ)
- Swap only hybrid checkpoint (`smoke model.pt` vs `full model.pt`)
- Compare reconstructed objects

Expected:
- Clean attribution of how much regression is from checkpoint choice alone.

---

### Task 6: Plotting-Only Control (H8)

**Files:**
- Create: `tmp/debug/render_fixed_scale_compare.py`
- Output: `tmp/debug/fixed_scale_compare_<timestamp>.png`

**Step 1: Re-render both run recons with fixed vmin/vmax**
- Use identical amplitude and phase ranges across runs/models.
- Include a side-by-side panel to remove autoscaling confounds.

Expected:
- If apparent regression shrinks under fixed scaling, plotting contributes significantly.
- If not, underlying recon differences are real.

---

### Task 7: Decision Gate and Next Fix Plan

**Files:**
- Create: `tmp/debug/regression_decision_gate_2026-02-18.md`

**Step 1: Summarize which hypotheses were eliminated/confirmed**
- Mark each H1-H8 as Confirmed/Refuted/Inconclusive with evidence path.

**Step 2: Choose one root-cause track only**
- `track-A`: data prep/contract mismatch
- `track-B`: checkpoint/model collapse
- `track-C`: visualization artifact only

**Step 3: Draft fix implementation plan**
- Only after root cause is confirmed.

---

### Task 8: Last-Resort Commit Isolation (`git bisect`)

**Trigger condition:**
- Run this task only if Tasks 1-7 do not isolate a single root cause (or produce conflicting attributions).

**Files:**
- Create: `tmp/debug/bisect_oracle_2026-02-18.sh`
- Create: `tmp/debug/bisect_report_2026-02-18.md`

**Step 1: Define a deterministic bisect oracle**
- Use fixed inputs/checkpoints/seed and one binary pass/fail metric (for example, parity/metric threshold from the decision gate).
- Oracle script exit codes:
  - `0` = good commit
  - `1` = bad commit
  - `125` = skip (commit cannot be evaluated in this environment)

**Step 2: Run bisect in a bounded commit window**
- Use the known-good vs known-bad run lineage from Task 1 provenance.

Run:
```bash
git bisect start <known-bad-sha> <known-good-sha>
git bisect run bash tmp/debug/bisect_oracle_2026-02-18.sh
git bisect reset
```

**Step 3: Record bisect outcome**
- Write `tmp/debug/bisect_report_2026-02-18.md` with:
  - known-good SHA
  - known-bad SHA
  - first bad SHA (or inconclusive result)
  - oracle command used
  - key metric deltas supporting the verdict

Expected:
- A concrete first-bad commit (or explicit proof that commit-level isolation is inconclusive due to non-determinism/environment drift).

---

## Guardrails
- Do not apply fixes until one hypothesis is confirmed with artifact evidence.
- Use dedicated output dirs per test to avoid artifact contamination.
- Keep all debugging scripts/results in `tmp/debug/`.

## Verification Checklist
- `summary.json` exists for both scan807 and cameraman parity checks.
- Provenance table links each image to a specific run dir and invocation.
- At least one ablation isolates checkpoint effect for ptychovit and hybrid independently.
- Final decision gate file clearly picks one root-cause track.
