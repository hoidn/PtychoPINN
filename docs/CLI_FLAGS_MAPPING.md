# CLI Flags Mapping: ptycho_torch/inference.py

## Overview
This document maps all command-line interface (CLI) flags defined in `ptycho_torch/inference.py` to their execution destinations, configuration objects, and current usage status.

### Legend
- **Phase E2.C2**: New Lightning checkpoint inference path (lines 293-572)
- **Legacy**: MLflow-based inference path (lines 574-600)
- **Config Object**: Target dataclass from `ptycho_torch/config_params.py`
- **Current Status**: Hardcoded or Configurable in destination
- **Notes**: Additional context on usage patterns

---

## Phase E2.C2: Lightning Checkpoint Inference (NEW CLI)

| Flag | Type | Default | Required | Destination Config | Current Status | Notes |
|------|------|---------|----------|-------------------|-----------------|-------|
| `--model_path` | str | N/A | Yes | (Direct Path) | Required Input | Path to training output dir containing `checkpoints/last.ckpt` or `wts.pt`. Passed directly to `Path()` object. No config object storage. |
| `--test_data` | str | N/A | Yes | (Direct Path) | Required Input | Path to test NPZ file. Validated against `specs/data_contracts.md` (requires 'diffraction', 'probeGuess', 'objectGuess'). Passed directly to `np.load()`. |
| `--output_dir` | str | N/A | Yes | (Direct Path) | Required Input | Directory for reconstruction outputs. Passed to `save_individual_reconstructions()`. Creates directory if missing via `mkdir(parents=True, exist_ok=True)`. |
| `--n_images` | int | 32 | No | (Direct Tensor Slicing) | Hardcoded Usage | Used at line 504 to limit diffraction batch: `diffraction[:args.n_images]`. Not stored in config object. |
| `--device` | str (choice: cpu/cuda) | 'cpu' | No | `TrainingConfig.device` | Partially Configurable | Passed to model at line 442 via `map_location=args.device` and line 447 via `model.to(args.device)`. Also used for tensor device placement (lines 494-495, 521, 525). Currently hardcoded to `args.device` in all usages; no dynamic fallback. |
| `--quiet` | bool (flag) | False | No | (Direct Control Flow) | Hardcoded Control | Controls output verbosity. Checked at lines 431, 449, 464, 527, 553, 561. Entirely hardcoded as `args.quiet` conditionals; not stored in config. |

**Execution Flow (Phase E2.C2)**:
1. CLI args parsed → Path validation
2. Checkpoint located (candidates: `checkpoints/last.ckpt`, `wts.pt`, `model.pt`)
3. Lightning module loaded via `PtychoPINN_Lightning.load_from_checkpoint()` with `map_location=args.device`
4. Model moved to device: `model.to(args.device)` + set to eval mode
5. Test data loaded from NPZ: `np.load(args.test_data)`
6. Forward pass: `model.forward_predict(diffraction, positions, probe, input_scale_factor)` with dtype enforcement (float32 for diffraction, complex64 for probe)
7. Batch limited to first `args.n_images` at line 504
8. Output saved via `save_individual_reconstructions()` to `args.output_dir`
9. Verbosity controlled by `args.quiet` flag

---

## Legacy Path: MLflow-Based Inference

| Flag | Type | Default | Required | Destination Config | Current Status | Notes |
|------|------|---------|----------|-------------------|-----------------|-------|
| `--run_id` | str | N/A | Yes (for legacy) | (MLflow URI) | Required Input | MLflow run ID used to construct `model_uri = f"runs:/{run_id}/model"` (line 127). Passed to `mlflow.pytorch.load_model()` and config loading functions. |
| `--infer_dir` | str | N/A | Yes (for legacy) | (Data Path) | Required Input | Directory containing ptychography files. Passed to `PtychoDataset()` constructor at line 149. |
| `--file_index` | int | 0 | No | `InferenceConfig.experiment_number` | Configurable | Maps to `inference_config.experiment_number` via `update_existing_config()` at line 138. Used for file selection in multi-file experiments. |
| `--config` | str | None | No | (JSON Config Path) | Configurable | Optional config override path. If provided, used in `load_all_configs()` at line 133 instead of loading from MLflow tracking server. |

**Execution Flow (Legacy)**:
1. CLI args parsed → MLflow tracking URI constructed
2. Model loaded via `mlflow.pytorch.load_model(model_uri)` with device mapping from `training_config.device`
3. Configs loaded from MLflow or from `--config` override path
4. `file_index` mapped to `inference_config.experiment_number`
5. `PtychoDataset` instantiated with `infer_dir`
6. Reconstruction via `reconstruct_image_barycentric()`
7. Results saved and plotted

---

## Hardcoded vs Configurable Summary

### Fully Hardcoded (No Config Object Storage)
- `--model_path`: Direct Path object usage
- `--test_data`: Direct np.load() usage
- `--output_dir`: Direct output path
- `--n_images`: Direct tensor slicing (line 504)
- `--quiet`: Direct control flow conditionals

### Partially Configurable
- `--device`: Used in inference but not persisted to `TrainingConfig` for later access
- `--file_index` (legacy): Maps to `InferenceConfig.experiment_number` (configurable)
- `--config` (legacy): Override config path (configurable)

### Notes on Configuration Missing Gaps
1. **No `--device` persistence**: The device is applied locally but not stored in a config object for potential replay or logging
2. **No `--n_images` config field**: Should perhaps map to `InferenceConfig` for consistency, but currently inline
3. **No verbosity config**: `--quiet` is not tracked in any config dataclass; would require adding to `InferenceConfig`

---

## Dtype Enforcement (Phase D1d)

Per lines 490-512, the following dtype conversions are **hardcoded** and not configurable:

| Input | Conversion | Destination | Specification |
|-------|-----------|-------------|---------------|
| `test_data['diffraction']` | `torch.float32` | GPU/CPU device | specs/data_contracts.md §1 (prevents Conv2d dtype mismatch) |
| `test_data['probeGuess']` | `torch.complex64` | GPU/CPU device | Legacy precision standard |
| Probe dimensions | Unsqueezed to `(1, 1, 1, H, W)` | Model input | Phase E2.C2 requirement |
| Diffraction dimensions | Permuted from `(H, W, n)` to `(n, H, W)` if needed | Model input | Standardization |

---

## Config Object Destinations Reference

### TrainingConfig
- **Field**: `device` (str, default='cuda')
- **Phase E2.C2 Usage**: Potentially stored but overridden by `args.device`
- **Legacy Usage**: Used via `training_config.device` at line 143

### InferenceConfig
- **Field**: `experiment_number` (int, default=0)
- **Legacy Usage**: Set from `--file_index` at line 138 via `update_existing_config()`
- **Phase E2.C2**: Not used

---

## Missing CLI Flags (Future Enhancements)

The following configuration parameters are **NOT exposed as CLI flags** but could be in future:

### From TrainingConfig:
- `learning_rate`, `epochs`, `batch_size`, `strategy`, `n_devices`

### From InferenceConfig:
- `middle_trim` (int, default=32)
- `batch_size` (int, default=1000) — **Note**: Distinct from training batch_size
- `pad_eval` (bool, default=True)
- `window` (int, default=20) — **Note**: Used in legacy path for plotting window padding (line 169)

### From DataConfig:
- `N` (int, default=64) — Diffraction patch size
- `normalize` (Literal, default='Batch')
- `probe_normalize` (bool, default=True)
- `data_scaling` (Literal, default='Parseval')

---

## References

- **Phase E2.C2 Implementation**: `plans/active/INTEGRATE-PYTORCH-001/phase_e2_implementation.md §E2.C2`
- **Test Contract**: `tests/torch/test_integration_workflow_torch.py`
- **Data Contract Specification**: `specs/data_contracts.md`
- **PyTorch API Spec**: `specs/ptychodus_api_spec.md §4.8`
- **Dtype Enforcement**: `ptycho_torch/inference.py` lines 490-512 (Phase D1d)
- **Config Dataclasses**: `ptycho_torch/config_params.py` lines 17-149

