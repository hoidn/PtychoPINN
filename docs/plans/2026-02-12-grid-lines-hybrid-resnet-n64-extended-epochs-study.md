# Grid-Lines N64 PINN + Hybrid-ResNet E20 Study Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Run a reproducible `N=64` grid-lines study for CNN-PINN (`pinn`) and Torch Hybrid ResNet (`pinn_hybrid_resnet`) at `20` epochs using a reusable script that performs two backend calls and then renders a guaranteed combined comparison PNG.

**Architecture:** Use one launcher script (`run_study.sh`) that executes: (1) TF-side selected-model run for `pinn` via `grid_lines_compare_wrapper.py` (this also materializes the dataset), (2) `grid_lines_torch_runner.py` for `hybrid_resnet` against the same generated NPZ files, then (3) an explicit `render_grid_lines_visuals(..., order=("gt","pinn","pinn_hybrid_resnet"))` call to guarantee the final comparison visual contains exactly those models.

**Tech Stack:** Python 3, `scripts/studies/grid_lines_compare_wrapper.py`, `scripts/studies/grid_lines_torch_runner.py`, `ptycho.workflows.grid_lines_workflow.render_grid_lines_visuals`, JSON/NPZ artifacts.

---

### Task 1: Freeze Study Spec and Create Reusable Launcher

**Files:**
- Read: `scripts/studies/grid_lines_compare_wrapper.py`
- Read: `scripts/studies/grid_lines_torch_runner.py`
- Create: `.artifacts/studies/grid_lines_n64_pinn_hybrid_resnet_e20/`
- Create: `.artifacts/studies/grid_lines_n64_pinn_hybrid_resnet_e20/run_study.sh`

**Step 1: Define constants**

Set:
- `N=64`
- `GRID=1`
- `EPOCHS=20`
- `SEED=3`
- `OUTPUT_DIR=outputs/grid_lines_n64_compare_padex_lr2e4_plateau_e20_seed3_pinn_hybrid_resnet`

**Step 2: Save exact two-call + render launcher**

Create `.artifacts/studies/grid_lines_n64_pinn_hybrid_resnet_e20/run_study.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail

OUT="outputs/grid_lines_n64_compare_padex_lr2e4_plateau_e20_seed3_pinn_hybrid_resnet"

# Call 1: run TF PINN selected-model workflow (also prepares datasets under $OUT/datasets/N64/gs1/)
python scripts/studies/grid_lines_compare_wrapper.py \
  --N 64 \
  --gridsize 1 \
  --output-dir "$OUT" \
  --models pinn \
  --nimgs-train 2 \
  --nimgs-test 1 \
  --nphotons 1e9 \
  --nepochs 20 \
  --batch-size 16 \
  --seed 3 \
  --probe-source custom \
  --probe-scale-mode pad_extrapolate \
  --set-phi

# Call 2: run Torch Hybrid ResNet on the same prepared datasets
python scripts/studies/grid_lines_torch_runner.py \
  --output-dir "$OUT" \
  --architecture hybrid_resnet \
  --train-npz "$OUT/datasets/N64/gs1/train.npz" \
  --test-npz "$OUT/datasets/N64/gs1/test.npz" \
  --N 64 \
  --gridsize 1 \
  --epochs 20 \
  --batch-size 16 \
  --infer-batch-size 16 \
  --learning-rate 2e-4 \
  --scheduler ReduceLROnPlateau \
  --plateau-factor 0.5 \
  --plateau-patience 2 \
  --plateau-min-lr 1e-4 \
  --plateau-threshold 0.0 \
  --seed 3 \
  --optimizer adam \
  --weight-decay 0.0 \
  --beta1 0.9 \
  --beta2 0.999 \
  --torch-loss-mode mae \
  --output-mode real_imag \
  --probe-source custom \
  --fno-modes 12 \
  --fno-width 32 \
  --fno-blocks 4 \
  --fno-cnn-blocks 2 \
  --torch-logger mlflow

# Explicit final render to guarantee compare PNG uses gt+pinn+pinn_hybrid_resnet
python - <<'PY'
from pathlib import Path
from ptycho.workflows.grid_lines_workflow import render_grid_lines_visuals

out = Path("outputs/grid_lines_n64_compare_padex_lr2e4_plateau_e20_seed3_pinn_hybrid_resnet")
render_grid_lines_visuals(out, order=("gt", "pinn", "pinn_hybrid_resnet"))
print("Rendered visuals under", out / "visuals")
PY
```

**Step 3: Optional preflight checks**

Run:

```bash
python scripts/studies/grid_lines_compare_wrapper.py --help
python scripts/studies/grid_lines_torch_runner.py --help
```

Expected: both exit `0`.

### Task 2: Execute Launcher and Capture Log

**Files:**
- Create: `outputs/grid_lines_n64_compare_padex_lr2e4_plateau_e20_seed3_pinn_hybrid_resnet/`
- Create: `.artifacts/studies/grid_lines_n64_pinn_hybrid_resnet_e20/logs/study.log`

**Step 1: Run the study**

Run:

```bash
mkdir -p .artifacts/studies/grid_lines_n64_pinn_hybrid_resnet_e20/logs
bash .artifacts/studies/grid_lines_n64_pinn_hybrid_resnet_e20/run_study.sh \
  |& tee .artifacts/studies/grid_lines_n64_pinn_hybrid_resnet_e20/logs/study.log
```

Expected: exit `0`.

### Task 3: Verify Required Artifacts

**Files:**
- Check: `outputs/grid_lines_n64_compare_padex_lr2e4_plateau_e20_seed3_pinn_hybrid_resnet/datasets/N64/gs1/train.npz`
- Check: `outputs/grid_lines_n64_compare_padex_lr2e4_plateau_e20_seed3_pinn_hybrid_resnet/datasets/N64/gs1/test.npz`
- Check: `outputs/grid_lines_n64_compare_padex_lr2e4_plateau_e20_seed3_pinn_hybrid_resnet/recons/gt/recon.npz`
- Check: `outputs/grid_lines_n64_compare_padex_lr2e4_plateau_e20_seed3_pinn_hybrid_resnet/recons/pinn/recon.npz`
- Check: `outputs/grid_lines_n64_compare_padex_lr2e4_plateau_e20_seed3_pinn_hybrid_resnet/recons/pinn_hybrid_resnet/recon.npz`
- Check: `outputs/grid_lines_n64_compare_padex_lr2e4_plateau_e20_seed3_pinn_hybrid_resnet/runs/pinn_hybrid_resnet/metrics.json`
- Check: `outputs/grid_lines_n64_compare_padex_lr2e4_plateau_e20_seed3_pinn_hybrid_resnet/visuals/amp_phase_pinn.png`
- Check: `outputs/grid_lines_n64_compare_padex_lr2e4_plateau_e20_seed3_pinn_hybrid_resnet/visuals/amp_phase_pinn_hybrid_resnet.png`
- Check: `outputs/grid_lines_n64_compare_padex_lr2e4_plateau_e20_seed3_pinn_hybrid_resnet/visuals/compare_amp_phase.png`

**Step 1: Existence checks**

Run:

```bash
ls -lh \
  outputs/grid_lines_n64_compare_padex_lr2e4_plateau_e20_seed3_pinn_hybrid_resnet/datasets/N64/gs1/train.npz \
  outputs/grid_lines_n64_compare_padex_lr2e4_plateau_e20_seed3_pinn_hybrid_resnet/datasets/N64/gs1/test.npz \
  outputs/grid_lines_n64_compare_padex_lr2e4_plateau_e20_seed3_pinn_hybrid_resnet/recons/gt/recon.npz \
  outputs/grid_lines_n64_compare_padex_lr2e4_plateau_e20_seed3_pinn_hybrid_resnet/recons/pinn/recon.npz \
  outputs/grid_lines_n64_compare_padex_lr2e4_plateau_e20_seed3_pinn_hybrid_resnet/recons/pinn_hybrid_resnet/recon.npz \
  outputs/grid_lines_n64_compare_padex_lr2e4_plateau_e20_seed3_pinn_hybrid_resnet/runs/pinn_hybrid_resnet/metrics.json \
  outputs/grid_lines_n64_compare_padex_lr2e4_plateau_e20_seed3_pinn_hybrid_resnet/visuals/amp_phase_pinn.png \
  outputs/grid_lines_n64_compare_padex_lr2e4_plateau_e20_seed3_pinn_hybrid_resnet/visuals/amp_phase_pinn_hybrid_resnet.png \
  outputs/grid_lines_n64_compare_padex_lr2e4_plateau_e20_seed3_pinn_hybrid_resnet/visuals/compare_amp_phase.png
```

Expected: all files present and non-empty.

**Step 2: Quick metrics sanity**

Run:

```bash
python - <<'PY'
import json
from pathlib import Path

out = Path("outputs/grid_lines_n64_compare_padex_lr2e4_plateau_e20_seed3_pinn_hybrid_resnet")
pinn_metrics = json.loads((out / "metrics.json").read_text()).get("pinn")
hybrid_metrics = json.loads((out / "runs/pinn_hybrid_resnet/metrics.json").read_text())
print("pinn keys:", sorted((pinn_metrics or {}).keys()))
print("pinn_hybrid_resnet keys:", sorted(hybrid_metrics.keys()))
PY
```

Expected: both metric payloads are present.

### Task 4: Create Studies Index Entry

**Files:**
- Create: `docs/studies/index.md` (if missing)
- Modify: `docs/studies/index.md` (append this study entry)
- Modify: `docs/index.md` (add discoverability link to studies index)

**Step 1: Create index scaffold (initial contents)**

If `docs/studies/index.md` does not exist, create it with:

```markdown
# Studies Index

## Grid-Lines Studies

| Study ID | Purpose | Scripts | CLI Entry Points | Output Directory |
|---|---|---|---|---|
```

**Step 2: Add this study as first entry**

Add a row that references:
- **Study ID:** `grid-lines-n64-pinn-hybrid-resnet-e20`
- **Scripts:** `.artifacts/studies/grid_lines_n64_pinn_hybrid_resnet_e20/run_study.sh`
- **CLI Entry Points:** 
  - `python scripts/studies/grid_lines_compare_wrapper.py ... --models pinn ...`
  - `python scripts/studies/grid_lines_torch_runner.py ... --architecture hybrid_resnet ...`
  - `python -c 'from ptycho.workflows.grid_lines_workflow import render_grid_lines_visuals; ...'`
- **Output Directory:** `outputs/grid_lines_n64_compare_padex_lr2e4_plateau_e20_seed3_pinn_hybrid_resnet`

**Step 3: Verify index contains required references**

Run:

```bash
rg -n "grid-lines-n64-pinn-hybrid-resnet-e20|run_study.sh|grid_lines_compare_wrapper.py|grid_lines_torch_runner.py|grid_lines_n64_compare_padex_lr2e4_plateau_e20_seed3_pinn_hybrid_resnet" docs/studies/index.md
```

Expected: all references are present.

**Step 4: Make studies index discoverable from docs hub**

In `docs/index.md`, add a "Studies Index" entry under the "Studies & Analysis"
section linking to `studies/index.md`, with a short description indicating it is
the registry for reproducible study runbooks (scripts, CLI entry points, output
directories).

**Step 5: Verify docs hub link**

Run:

```bash
rg -n "Studies Index|studies/index.md" docs/index.md docs/studies/index.md
```

Expected:
- `docs/index.md` includes a link to `studies/index.md`
- `docs/studies/index.md` exists and contains the study row from Step 2.

### Task 5: Capture Evidence

**Files:**
- Create: `.artifacts/studies/grid_lines_n64_pinn_hybrid_resnet_e20/summary.md`

**Step 1: Record**

Include:
- launcher script contents
- output directory path
- artifact verification output
- metric snapshot (`pinn` and `pinn_hybrid_resnet`)
- any warnings/failures from `study.log`

**Step 2: Optional commit**

```bash
git add docs/plans/2026-02-12-grid-lines-hybrid-resnet-n64-extended-epochs-study.md docs/studies/index.md docs/index.md
git commit -m "docs(plan): add discoverable studies index entry for n64 pinn+hybrid_resnet e20 workflow"
```
