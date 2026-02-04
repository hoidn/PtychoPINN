# Rerun Instructions: sim_lines_4x Ideal-Probe Cases

These steps are for a NEW session with GPU working. They rerun **only** the ideal-probe cases (gs1_ideal + gs2_ideal) using the correct disk-shaped probe via `--probe-source ideal_disk`.

## Prerequisites

1. Start a session with GPU passthrough enabled.
2. Activate the correct conda env:

```bash
conda activate ptycho311
```

3. Verify GPU is visible to PyTorch (stop if `cuda_available` is False):

```bash
python - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
print("device_count", torch.cuda.device_count())
PY
```

## Rerun gs1_ideal (N=64, gridsize=1)

```bash
python scripts/studies/grid_lines_workflow.py \
  --N 64 \
  --gridsize 1 \
  --output-dir .artifacts/sim_lines_4x_metrics_2026-01-27/gs1_ideal \
  --probe-npz datasets/Run1084_recon3_postPC_shrunk_3.npz \
  --probe-source ideal_disk \
  --probe-smoothing-sigma 0.0 \
  --probe-scale-mode pad_extrapolate \
  --nimgs-train 2 \
  --nimgs-test 2 \
  --nphotons 1e9 \
  --nepochs 60 \
  --batch-size 16 \
  --nll-weight 0.0 \
  --mae-weight 1.0 \
  --realspace-weight 0.0
```

Expected output:
- `.artifacts/sim_lines_4x_metrics_2026-01-27/gs1_ideal/metrics.json`
- `.artifacts/sim_lines_4x_metrics_2026-01-27/gs1_ideal/visuals/compare_amp_phase.png`

## Rerun gs2_ideal (N=64, gridsize=2)

```bash
python scripts/studies/grid_lines_workflow.py \
  --N 64 \
  --gridsize 2 \
  --output-dir .artifacts/sim_lines_4x_metrics_2026-01-27/gs2_ideal \
  --probe-npz datasets/Run1084_recon3_postPC_shrunk_3.npz \
  --probe-source ideal_disk \
  --probe-smoothing-sigma 0.0 \
  --probe-scale-mode pad_extrapolate \
  --nimgs-train 2 \
  --nimgs-test 2 \
  --nphotons 1e9 \
  --nepochs 60 \
  --batch-size 16 \
  --nll-weight 0.0 \
  --mae-weight 1.0 \
  --realspace-weight 0.0
```

Expected output:
- `.artifacts/sim_lines_4x_metrics_2026-01-27/gs2_ideal/metrics.json`
- `.artifacts/sim_lines_4x_metrics_2026-01-27/gs2_ideal/visuals/compare_amp_phase.png`

## Notes

- The CLI default `--probe-source` is `custom`, so **`--probe-source ideal_disk` must be explicit**.
- The custom-probe cases (`gs1_custom`, `gs2_custom`) do not need rerun for this correction.
- If you want to overwrite prior outputs, ensure the output directories are writable and that you are OK replacing existing metrics/visuals.
