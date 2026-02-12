# Studies Index

## Grid-Lines Studies

### `grid-lines-n64-pinn-hybrid-resnet-e20`

- Purpose: Run `N=64` grid-lines with `pinn` (TF) and `pinn_hybrid_resnet` (Torch) at `20` epochs, then render combined visuals.
- Script: `.artifacts/studies/grid_lines_n64_pinn_hybrid_resnet_e20/run_study.sh`
- Output directory: `outputs/grid_lines_n64_compare_padex_lr2e4_plateau_e20_seed3_pinn_hybrid_resnet`
- Invocation artifacts:
  - `outputs/grid_lines_n64_compare_padex_lr2e4_plateau_e20_seed3_pinn_hybrid_resnet/invocation.json`
  - `outputs/grid_lines_n64_compare_padex_lr2e4_plateau_e20_seed3_pinn_hybrid_resnet/invocation.sh`
  - `outputs/grid_lines_n64_compare_padex_lr2e4_plateau_e20_seed3_pinn_hybrid_resnet/runs/pinn_hybrid_resnet/invocation.json`
  - `outputs/grid_lines_n64_compare_padex_lr2e4_plateau_e20_seed3_pinn_hybrid_resnet/runs/pinn_hybrid_resnet/invocation.sh`

CLI entry points (full commands):

```bash
python scripts/studies/grid_lines_compare_wrapper.py \
  --N 64 \
  --gridsize 1 \
  --output-dir outputs/grid_lines_n64_compare_padex_lr2e4_plateau_e20_seed3_pinn_hybrid_resnet \
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
```

```bash
python scripts/studies/grid_lines_torch_runner.py \
  --output-dir outputs/grid_lines_n64_compare_padex_lr2e4_plateau_e20_seed3_pinn_hybrid_resnet \
  --architecture hybrid_resnet \
  --train-npz outputs/grid_lines_n64_compare_padex_lr2e4_plateau_e20_seed3_pinn_hybrid_resnet/datasets/N64/gs1/train.npz \
  --test-npz outputs/grid_lines_n64_compare_padex_lr2e4_plateau_e20_seed3_pinn_hybrid_resnet/datasets/N64/gs1/test.npz \
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
```

```bash
python - <<'PY'
from pathlib import Path
from ptycho.workflows.grid_lines_workflow import render_grid_lines_visuals

out = Path("outputs/grid_lines_n64_compare_padex_lr2e4_plateau_e20_seed3_pinn_hybrid_resnet")
render_grid_lines_visuals(out, order=("gt", "pinn", "pinn_hybrid_resnet"))
print("Rendered visuals under", out / "visuals")
PY
```
