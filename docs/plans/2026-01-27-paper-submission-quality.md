# Paper Submission Quality Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Regenerate SIM-LINES-4X metrics with the new grid-lines workflow, update the paper table/provenance, and clean up benchmark artifacts while keeping the throughput claim predict-only.

**Architecture:** Use the existing `scripts/studies/grid_lines_workflow.py` CLI to produce per-case `metrics.json` outputs, then update `paper/data/sim_lines_4x_metrics.json` and regenerate the LaTeX table with `paper/tables/scripts/generate_sim_lines_4x_metrics.py`. No core code changes are required; this is data regeneration + documentation.

**Tech Stack:** Python (TensorFlow/Keras + NumPy), existing workflow scripts, LaTeX table generator.

---

### Task 1: Confirm SIM-LINES-4X regeneration parameters

**Files:**
- Read: `paper/data/sim_lines_4x_metrics.json`
- Read: `docs/plans/2026-01-27-grid-lines-workflow.md`
- Read: `scripts/studies/grid_lines_workflow.py`

**Step 1: Capture current schema and notes**

Open `paper/data/sim_lines_4x_metrics.json` and record:
- `cases` IDs (`gs1_ideal`, `gs1_custom`, `gs2_ideal`, `gs2_custom`)
- Labels and `notes` (nsamples=1000, seed=7, registration settings)

**Step 2: Confirm CLI parameters and probe mapping**

Read `scripts/studies/grid_lines_workflow.py` and confirm the CLI flags for:
- `--N`, `--gridsize`, `--probe-npz`, `--nimgs-train`, `--nimgs-test`, `--nphotons`, `--nepochs`, `--batch-size`, `--nll-weight`, `--mae-weight`, `--realspace-weight`, `--probe-smoothing-sigma`

**Step 3: Freeze probe mapping for the four cases**

Use the following mapping unless updated by the user:
- **idealized probe:** `--probe-smoothing-sigma 0.0`
- **custom probe:** `--probe-smoothing-sigma 0.5`

If this mapping is wrong, stop and ask the user before running workflows.

---

### Task 2: Run grid-lines workflow for four cases

**Files:**
- Create (artifact dirs): `.artifacts/sim_lines_4x_metrics_2026-01-27/gs1_ideal/`
- Create (artifact dirs): `.artifacts/sim_lines_4x_metrics_2026-01-27/gs1_custom/`
- Create (artifact dirs): `.artifacts/sim_lines_4x_metrics_2026-01-27/gs2_ideal/`
- Create (artifact dirs): `.artifacts/sim_lines_4x_metrics_2026-01-27/gs2_custom/`

**Step 1: Run gs1 + idealized probe**

Run:
```bash
python scripts/studies/grid_lines_workflow.py \
  --N 64 \
  --gridsize 1 \
  --output-dir .artifacts/sim_lines_4x_metrics_2026-01-27/gs1_ideal \
  --probe-npz datasets/Run1084_recon3_postPC_shrunk_3.npz \
  --nimgs-train 2 \
  --nimgs-test 2 \
  --nphotons 1e9 \
  --nepochs 60 \
  --batch-size 16 \
  --nll-weight 0.0 \
  --mae-weight 1.0 \
  --realspace-weight 0.0 \
  --probe-smoothing-sigma 0.0
```
Expected: `metrics.json` exists under the output dir.

**Step 2: Run gs1 + custom probe**

Run:
```bash
python scripts/studies/grid_lines_workflow.py \
  --N 64 \
  --gridsize 1 \
  --output-dir .artifacts/sim_lines_4x_metrics_2026-01-27/gs1_custom \
  --probe-npz datasets/Run1084_recon3_postPC_shrunk_3.npz \
  --nimgs-train 2 \
  --nimgs-test 2 \
  --nphotons 1e9 \
  --nepochs 60 \
  --batch-size 16 \
  --nll-weight 0.0 \
  --mae-weight 1.0 \
  --realspace-weight 0.0 \
  --probe-smoothing-sigma 0.5
```
Expected: `metrics.json` exists under the output dir.

**Step 3: Run gs2 + idealized probe**

Run:
```bash
python scripts/studies/grid_lines_workflow.py \
  --N 64 \
  --gridsize 2 \
  --output-dir .artifacts/sim_lines_4x_metrics_2026-01-27/gs2_ideal \
  --probe-npz datasets/Run1084_recon3_postPC_shrunk_3.npz \
  --nimgs-train 2 \
  --nimgs-test 2 \
  --nphotons 1e9 \
  --nepochs 60 \
  --batch-size 16 \
  --nll-weight 0.0 \
  --mae-weight 1.0 \
  --realspace-weight 0.0 \
  --probe-smoothing-sigma 0.0
```
Expected: `metrics.json` exists under the output dir.

**Step 4: Run gs2 + custom probe**

Run:
```bash
python scripts/studies/grid_lines_workflow.py \
  --N 64 \
  --gridsize 2 \
  --output-dir .artifacts/sim_lines_4x_metrics_2026-01-27/gs2_custom \
  --probe-npz datasets/Run1084_recon3_postPC_shrunk_3.npz \
  --nimgs-train 2 \
  --nimgs-test 2 \
  --nphotons 1e9 \
  --nepochs 60 \
  --batch-size 16 \
  --nll-weight 0.0 \
  --mae-weight 1.0 \
  --realspace-weight 0.0 \
  --probe-smoothing-sigma 0.5
```
Expected: `metrics.json` exists under the output dir.

---

### Task 3: Update `paper/data/sim_lines_4x_metrics.json`

**Files:**
- Modify: `paper/data/sim_lines_4x_metrics.json`

**Step 1: Extract metrics from each run**

For each case, open:
- `.artifacts/sim_lines_4x_metrics_2026-01-27/gs1_ideal/metrics.json`
- `.artifacts/sim_lines_4x_metrics_2026-01-27/gs1_custom/metrics.json`
- `.artifacts/sim_lines_4x_metrics_2026-01-27/gs2_ideal/metrics.json`
- `.artifacts/sim_lines_4x_metrics_2026-01-27/gs2_custom/metrics.json`

Use the **PINN** metrics (`metrics["pinn"]`) for the table. If `registration_offset` is present in the metrics payload, carry it over; otherwise set it to `[null, null]`.

**Step 2: Update the JSON cases in-place**

Replace the `metrics` blocks for the four cases while preserving:
- `id`
- `label`
- `notes` (keep nsamples=1000, seed=7 unless the workflow uses different sampling)

---

### Task 4: Regenerate the LaTeX table

**Files:**
- Modify: `paper/tables/sim_lines_4x_metrics.tex`

**Step 1: Run the table generator**

Run:
```bash
python paper/tables/scripts/generate_sim_lines_4x_metrics.py \
  --input paper/data/sim_lines_4x_metrics.json \
  --output paper/tables/sim_lines_4x_metrics.tex
```
Expected: Table values match the updated JSON (6 decimal places).

---

### Task 5: Update provenance notes

**Files:**
- Modify: `paper/data/README.md`

**Step 1: Record new metrics sources**

Add the output paths used for SIM-LINES-4X regeneration under the metrics section, including the run date and the four case directories.

---

### Task 6: Clean up temporary benchmark artifacts

**Files:**
- Delete: `tmp/subsample_seed45_indices.txt`

**Step 1: Remove the temporary subsample indices**

Run:
```bash
rm -f tmp/subsample_seed45_indices.txt
```

---

### Task 7: Sanity check and commit

**Files:**
- Verify: `paper/data/sim_lines_4x_metrics.json`
- Verify: `paper/tables/sim_lines_4x_metrics.tex`
- Verify: `paper/data/README.md`

**Step 1: Quick diff review**

Run:
```bash
git status -sb
git diff --stat
```
Expected: Only the JSON, table, README, and cleanup file are changed.

**Step 2: (Optional) Commit**

```bash
git add paper/data/sim_lines_4x_metrics.json paper/tables/sim_lines_4x_metrics.tex paper/data/README.md
git rm tmp/subsample_seed45_indices.txt
git commit -m "docs(paper): refresh SIM-LINES-4X metrics table"
```

---

## Out of Scope / Deferred
- Provenance for `paper/figures/8192.png` (deferred by user).
- `paper/tables/metrics.tex` (explicitly accepted as-is).
- FRC vs dose plot replacement (placeholder remains).

---

Plan complete and saved to `docs/plans/2026-01-27-paper-submission-quality.md`. Two execution options:

1. Subagent-Driven (this session) — I dispatch a fresh subagent per task, review between tasks, fast iteration  
2. Parallel Session (separate) — Open new session with executing-plans, batch execution with checkpoints

Which approach?  
