# Idealized vs Custom Probe PINN Figure Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Generate a 2x2 figure (Idealized vs Custom probe × CDI vs Ptycho) showing PINN-only reconstructions at N=64 for the manuscript.

**Architecture:** Use the legacy grid-based simulation pipeline via `ptycho.workflows.grid_lines_workflow` to synthesize lines data for each condition (idealized/custom probe and gridsize 1/2). A dedicated script will run simulation → PINN training → PINN inference → stitching and save raw amplitude PNGs with a fixed colormap; existing `paper/figures/scripts/cdi_ptycho.py` will then rescale the four panels consistently.

**Tech Stack:** Python (numpy, tensorflow, matplotlib), existing PtychoPINN workflows, shell scripts.

### Task 1: Add a reproducible figure generation script

**Files:**
- Create: `scripts/paper/generate_idealized_custom_pinn_figure.py`
- Read: `ptycho/workflows/grid_lines_workflow.py`, `ptycho/probe.py`

**Step 1: Write a minimal script skeleton (no execution yet)**

```python
#!/usr/bin/env python3
"""Generate idealized vs custom PINN-only recon panels for CDI/Ptycho."""
```

**Step 2: Implement the CLI and configuration**

```python
# Arguments: --N (64), --outdir, --custom-probe-npz, --nepochs, --batch-size, --nimgs-train, --nimgs-test
# Optional: --probe-smoothing-sigma, --seed
```

**Step 3: Implement probe setup**

```python
# Idealized probe: ptycho.probe.get_default_probe(N, fmt='np')
# Custom probe: load probeGuess from provided NPZ
# Coerce to complex64 and shape (N, N)
```

**Step 4: Implement the per-condition runner**

```python
# For each condition:
# - Build GridLinesConfig with gridsize=1 or 2
# - Call simulate_grid_data() (which configures legacy params)
# - Train PINN via train_pinn_model()
# - Run inference via run_pinn_inference()
# - Stitch amplitude via stitch_predictions(..., part='amp')
# - Save a raw PNG with matplotlib.cm.get_cmap('jet') via plt.imsave
```

**Step 5: Save outputs to a stable naming scheme**

```python
# idealized_cdi.png
# idealized_ptycho.png
# hybrid_cdi.png
# hybrid_ptycho.png
```

**Step 6: Dry-run validation (no tests)**

Run: `python scripts/paper/generate_idealized_custom_pinn_figure.py --help`
Expected: CLI help text prints with required arguments.

**Step 7: Commit**

```bash
git add scripts/paper/generate_idealized_custom_pinn_figure.py
git commit -m "feat: add script to generate idealized/custom PINN recon panels"
```

### Task 2: Run the script to generate raw panels (manual execution)

**Files:**
- Read/Write: `/home/ollie/Documents/tmp/paper/figures/recon_idealized_custom_v1/*`

**Step 1: Create output directory**

Run: `mkdir -p /home/ollie/Documents/tmp/paper/figures/recon_idealized_custom_v1`
Expected: directory exists.

**Step 2: Execute the script (N=64)**

Run:
```bash
python scripts/paper/generate_idealized_custom_pinn_figure.py \
  --N 64 \
  --outdir /home/ollie/Documents/tmp/paper/figures/recon_idealized_custom_v1 \
  --custom-probe-npz datasets/Run1084_recon3_postPC_shrunk_3.npz \
  --nimgs-train 2 \
  --nimgs-test 2 \
  --nepochs 60 \
  --batch-size 16
```
Expected: four PNGs written to output directory and a small JSON log of run settings.

**Step 3: Sanity check output files**

Run: `ls -la /home/ollie/Documents/tmp/paper/figures/recon_idealized_custom_v1/*.png`
Expected: 4 files exist and are non-empty.

### Task 3: Rescale panels consistently using existing scaling script

**Files:**
- Read: `/home/ollie/Documents/tmp/paper/figures/recon_idealized_custom_v1/*.png`
- Write: `/home/ollie/Documents/tmp/paper/figures/recon_idealized_custom_v1/out_scaling_v5/*`

**Step 1: Run scaling script**

Run:
```bash
python /home/ollie/Documents/tmp/paper/figures/scripts/cdi_ptycho.py \
  --idealized-cdi /home/ollie/Documents/tmp/paper/figures/recon_idealized_custom_v1/idealized_cdi.png \
  --idealized-ptycho /home/ollie/Documents/tmp/paper/figures/recon_idealized_custom_v1/idealized_ptycho.png \
  --hybrid-cdi /home/ollie/Documents/tmp/paper/figures/recon_idealized_custom_v1/hybrid_cdi.png \
  --hybrid-ptycho /home/ollie/Documents/tmp/paper/figures/recon_idealized_custom_v1/hybrid_ptycho.png \
  --outdir /home/ollie/Documents/tmp/paper/figures/recon_idealized_custom_v1/out_scaling_v5
```
Expected: `_scaled.png`, `_scaled_small.png`, and `recon_mosaic_small.png` outputs plus `scaling_manifest.json`.

**Step 2: Optional rename for manuscript**

```bash
cp /home/ollie/Documents/tmp/paper/figures/recon_idealized_custom_v1/out_scaling_v5/idealized_cdi_scaled_small.png \
  /home/ollie/Documents/tmp/paper/figures/idealized_cdi_scaled_v5_small.png
# Repeat for the other three panels as needed.
```

### Task 4: Record outputs in the plan summary (manual)

**Files:**
- Update: `plans/active/<initiative>/summary.md` (if/when an initiative is created)

**Step 1: Add paths and a short note**

```text
Generated idealized/custom PINN recon panels with N=64.
Output dir: /home/ollie/Documents/tmp/paper/figures/recon_idealized_custom_v1
Scaled panels: out_scaling_v5/*
```

**Step 2: Commit (if repo plan or summary updated)**

```bash
git add plans/active/<initiative>/summary.md
git commit -m "docs: record idealized/custom PINN figure outputs"
```
