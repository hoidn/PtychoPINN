### Turn Summary
Performed supervisor code inspection to identify why model outputs are ~4.3× smaller than ground truth despite IntensityScaler matching perfectly.
Root cause identified: `realspace_weight=0.0` in default config means the loss function has NO direct supervision on object amplitude — model only learns to match diffraction intensity patterns (NLL loss).
Fix is straightforward: set `realspace_weight > 0` (reference: `train_pinn.py:56` uses 0.1 for PINN training) in the sim_lines pipeline config.
Artifacts: plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-21T220000Z/ (summary.md)

## D6 Root Cause Analysis

### Code Inspection Evidence

**1. Loss Function Compilation** (`ptycho/model.py:597-601`):
```python
autoencoder.compile(
    loss=[hh.realspace_loss, 'mean_absolute_error', negloglik],
    loss_weights=[realspace_weight, mae_weight, nll_weight],
)
```

**2. Training Output Labels** (`ptycho/loader.py:306-309`):
```python
# Prepare outputs: (centered_Y_I[:,:,:,:1], X*s, (X*s)^2) as tuple
Y_I_centered = hh.center_channels(Y_I_batch, coords_batch)[:, :, :, :1]
X_scaled = intensity_scale * X_batch
outputs = (Y_I_centered, X_scaled, X_scaled ** 2)
```

Loss-to-output mapping:
| Loss Function | Model Output | Training Label | Default Weight |
|--------------|--------------|----------------|----------------|
| `realspace_loss` | `trimmed_obj` | `Y_I_centered` (ground truth amplitude) | **0.0** |
| `mae` | `pred_amp_scaled` | `X_scaled` (scaled diffraction) | 0.0 |
| `negloglik` | `pred_intensity_sampled` | `X_scaled²` (scaled intensity) | 1.0 |

**3. Default Loss Weights** (`ptycho/params.py:64-65`, `ptycho/config/config.py:115-118`):
- `mae_weight: float = 0.0`
- `nll_weight: float = 1.0`
- `realspace_weight: float = 0.0` ← **ROOT CAUSE**

**4. sim_lines Pipeline** (`scripts/studies/sim_lines_4x/pipeline.py:176-203`):
- `build_training_config()` creates `TrainingConfig` without setting any loss weights
- Uses dataclass defaults → `realspace_weight=0.0`

### Root Cause

**`realspace_weight=0.0` means the model has NO direct supervision on object amplitude.**

The model only optimizes:
- NLL loss on intensity: `pred_intensity ≈ X_scaled²` (weight=1.0)
- MAE loss on amplitude: `pred_amp_scaled ≈ X_scaled` (weight=0.0)

This explains:
1. Model can successfully reproduce diffraction patterns (intensity loss is optimized)
2. Object amplitude has NO direct constraint to match ground truth
3. The ~4.3× amplitude gap exists because amplitude is only constrained implicitly through the physics forward model

### Fix Approach

Set `realspace_weight > 0` in the training configuration. Reference implementation:
- `ptycho/train_pinn.py:56` uses `realspace_weight=0.1` for PINN training

### Next Action (D6a)

Update `scripts/studies/sim_lines_4x/pipeline.py::build_training_config()` to set `realspace_weight=0.1` (or expose as configurable parameter), rerun gs2_ideal with the new loss weighting, and compare amplitude metrics against D5b baseline.
