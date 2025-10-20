# Phase D.A Baseline: CLI Call Graph and Current Architecture

**Initiative:** ADR-003-BACKEND-API — Standardize PyTorch backend API
**Phase:** D.A — Capture baseline evidence before refactoring CLI wrappers
**Date:** 2025-10-20
**Author:** Ralph (Engineer Loop)

---

## Executive Summary

This document captures the current CLI architecture for `ptycho_torch/train.py` and `ptycho_torch/inference.py` before Phase D refactoring. Both CLIs exhibit **dual interface patterns** (legacy + new) and contain **substantial business logic** that should be extracted to reusable helpers. The goal of Phase D is to collapse these modules into thin argument-parsing shims that delegate to shared factories and workflow components.

**Key Findings:**
1. **Training CLI (670 lines):** 377-line `cli_main()` function handles argparse, config construction, factory invocation, data loading, AND workflow orchestration — violating single-responsibility principle.
2. **Inference CLI (671 lines):** 349-line `cli_main()` function handles similar responsibilities including manual tensor operations and reconstruction logic.
3. **Execution Config Integration:** Both CLIs implement Phase C4 execution config flags (--accelerator, --deterministic, --num-workers) but with **duplicate validation and device mapping logic**.
4. **Legacy Interface Burden:** Both maintain backward-compatible paths (`--ptycho_dir`, `--config`, `--run_id`) requiring if-branch logic and separate code paths.
5. **Factory Integration Achieved:** Both CLIs successfully delegate to `create_training_payload()` and `create_inference_payload()` (Phase C4 success), but surrounding orchestration logic remains in CLI scope.

---

## Training CLI Call Graph: `ptycho_torch/train.py`

### Entry Point Chain

```
cli_main() [ptycho_torch/train.py:377]
  ↓
  argparse.ArgumentParser() [train.py:390-481]
    - New interface flags: --train_data_file, --test_data_file, --output_dir, --max_epochs, --n_images, --gridsize, --batch_size, --device
    - Execution config flags (Phase C4): --accelerator, --deterministic/--no-deterministic, --num-workers, --learning-rate [424-474]
    - Legacy interface flags: --ptycho_dir, --config [477-480]
    - Shared: --disable_mlflow [420-421]
  ↓
  Interface Resolution [train.py:484-491]
    - Detects legacy vs new interface
    - Rejects mixed usage (error exit)
  ↓
  [LEGACY PATH] main(ptycho_dir, config_path, disable_mlflow) [train.py:493-507]
    → Loads JSON config
    → Instantiates DataConfig/ModelConfig/TrainingConfig singletons
    → Calls legacy training flow (NOT REFACTORED)
  ↓
  [NEW PATH] Factory-Based Training [train.py:509-656]
    ↓
    Path Validation [train.py:522-532]
      - Validates train_data_file, test_data_file exist
      - Creates output_dir
    ↓
    Execution Config Construction [train.py:541-581]
      - Creates PyTorchExecutionConfig from CLI args
      - Maps --device to --accelerator (backward compat) [545-556]
      - Validates num_workers >= 0, learning_rate > 0 [558-564]
      - Emits deterministic+num_workers warning [566-573]
    ↓
    create_training_payload() [ptycho_torch.config_factory:create_training_payload] [train.py:598-604]
      - Factory handles CONFIG-001 compliance (update_legacy_dict)
      - Returns TrainingPayload(tf_training_config, pt_data_config, pt_model_config, pt_training_config, execution_config)
    ↓
    Config Tuple Assembly [train.py:614-622]
      - Builds existing_config tuple for legacy main() signature
      - Adds default InferenceConfig(), DatagenConfig()
    ↓
    Data Loading (DUPLICATION) [train.py:636-638]
      - RawData.from_file(train_data_file)
      - RawData.from_file(test_data_file) if present
      - ⚠️ This is duplicate logic: workflows/components.run_cdi_example_torch also loads data
    ↓
    run_cdi_example_torch() [ptycho_torch.workflows.components:run_cdi_example_torch] [train.py:641-647]
      - do_stitching=False (CLI only trains, doesn't reconstruct)
      - Orchestrates training + model persistence
      - Returns (amplitude, phase, results_dict)
    ↓
    Success Logging [train.py:649-650]
      - Prints output_dir, wts.h5.zip path
```

### Main Training Function: `main()`

**Signature:** `main(ptycho_dir, config_path=None, existing_config=None, disable_mlflow=False, output_dir=None, execution_config=None)` [train.py:144-374]

**Responsibilities:**
1. **Config Loading** [169-199]: Legacy JSON loading OR existing_config tuple acceptance
2. **Seed Management** [202]: `set_seed(42, n_devices=training_config.n_devices)`
3. **Data Module** [206-215]: Instantiates `PtychoDataModule` with `initial_remake_map=True`, `val_split=0.05`
4. **Model Creation** [218-226]: Instantiates `PtychoPINN_Lightning`, updates learning rate via `find_learning_rate()`
5. **Checkpoint Config** [233-235]: Constructs checkpoint directory from `output_dir` or `os.getcwd()` parent
6. **Callbacks** [238-258]: Creates `ModelCheckpoint` + `EarlyStopping` callbacks
7. **Trainer Config** [267-300]: Instantiates `lightning.Trainer` with execution config knobs
8. **MLflow Setup** [303-313]: Conditional autologging, experiment creation
9. **Training Loop** [311-340]: `trainer.fit(model, datamodule)` with MLflow run context
10. **Fine-Tuning** [343-349]: Optional fine-tune loop via `ModelFineTuner`
11. **Synchronization** [359-374]: Distributed barrier, return run_ids dict

**Key Dependencies:**
- `ptycho_torch.model.PtychoPINN_Lightning`
- `ptycho_torch.train_utils.PtychoDataModule`
- `ptycho_torch.train_utils.find_learning_rate`
- `lightning.pytorch.Trainer`
- `mlflow.pytorch.autolog`

**⚠️ Problem:** This function is called by BOTH legacy and new CLI paths, creating tight coupling. Legacy path directly invokes `main()`, new path delegates to `run_cdi_example_torch()` which presumably internally uses similar logic (duplication).

---

## Inference CLI Call Graph: `ptycho_torch/inference.py`

### Entry Point Chain

```
cli_main() [ptycho_torch/inference.py:293]
  ↓
  argparse.ArgumentParser() [inference.py:319-410]
    - New interface flags: --model_path, --test_data, --output_dir, --n_images, --device
    - Execution config flags (Phase C4): --accelerator, --num-workers, --inference-batch-size [380-410]
    - Shared: --quiet [375-377]
  ↓
  Execution Config Construction [inference.py:414-442]
    - Creates PyTorchExecutionConfig from CLI args
    - Maps --device to --accelerator (backward compat) [418-429]
    - Validates num_workers >= 0, inference_batch_size > 0 [431-435]
    - enable_progress_bar = not args.quiet [441]
  ↓
  create_inference_payload() [ptycho_torch.config_factory:create_inference_payload] [inference.py:477-484]
    - Factory handles CONFIG-001 compliance
    - Returns InferencePayload(tf_inference_config, pt_data_config, execution_config)
  ↓
  load_inference_bundle_torch() [ptycho_torch.workflows.components:load_inference_bundle_torch] [inference.py:510-519]
    - Loads wts.h5.zip bundle
    - Restores models_dict, params_dict
    - Returns ({'diffraction_to_obj': lightning_module}, params_dict)
  ↓
  Model Device Move [inference.py:521-533]
    - Maps execution_config.accelerator → torch device ('cpu'/'cuda'/'mps')
    - model.to(device)
  ↓
  Data Loading [inference.py:550-554]
    - RawData.from_file(test_data_path)
  ↓
  Manual Inference Loop [inference.py:565-629]
    ⚠️ This is HEAVY logic that should be in workflows/components:
    - Converts numpy → torch tensors [568-569]
    - DTYPE enforcement: .to(dtype=torch.float32) [568]
    - Transpose handling: (H,W,n) → (n,H,W) [572-574]
    - Subsetting: diffraction[:tf_inference_config.n_groups] [577]
    - Channel dimension insertion: .unsqueeze(1) [580]
    - Probe preparation: unsqueeze batch dims [587-589]
    - Dummy positions tensor [592-593]
    - model.forward_predict() call [603-608]
    - Result aggregation: np.mean(reconstruction_cpu, axis=0) [614]
    - Amplitude/phase extraction [620-621]
  ↓
  save_individual_reconstructions() [inference.py:629]
    - Generates reconstructed_amplitude.png, reconstructed_phase.png
    - Matplotlib figure creation + savefig
```

### Helper Functions

**`load_and_predict()` [inference.py:96-184]:**
- **Legacy MLflow-based inference path** (not refactored)
- Loads model via `mlflow.pytorch.load_model()`
- Uses `reconstruct_image_barycentric()` (custom reassembly)
- Generates 2x2 comparison plots via `plot_amp_and_phase()`
- **Not invoked by new CLI path** (Phase E2.C2 uses manual loop)

**`plot_amp_and_phase()` [inference.py:187-246]:**
- Creates 2x2 grid: reconstructed amp/phase, ground truth amp/phase
- Saves to SVG/PNG with timestamp filename
- **Not used by new CLI path** (uses `save_individual_reconstructions()` instead)

**`save_individual_reconstructions()` [inference.py:249-290]:**
- **Phase E2.C2 artifact generator**
- Saves amplitude/phase as separate PNG files
- Required by integration test contract (test_integration_workflow_torch.py)
- Simple matplotlib imshow + colorbar + savefig

---

## Shared Patterns and Redundancies

### Duplicate Device Mapping Logic

**Training CLI** [train.py:545-556]:
```python
resolved_accelerator = args.accelerator
if args.device and args.accelerator == 'auto':
    resolved_accelerator = 'cpu' if args.device == 'cpu' else 'gpu'
elif args.device and args.accelerator != 'auto':
    warnings.warn("--device is deprecated... Use --accelerator instead.")
```

**Inference CLI** [inference.py:418-429]:
```python
resolved_accelerator = args.accelerator
if args.device and args.accelerator == 'auto':
    resolved_accelerator = 'cpu' if args.device == 'cpu' else 'gpu'
elif args.device and args.accelerator != 'auto':
    warnings.warn("--device is deprecated... Use --accelerator instead.")
```

**⚠️ Problem:** Exact duplication. Should be extracted to shared helper: `ptycho_torch/cli/shared.py:resolve_accelerator(args)`

### Duplicate Validation Logic

Both CLIs validate `num_workers >= 0`, `learning_rate > 0` (training), `inference_batch_size > 0` (inference). This should be moved to `PyTorchExecutionConfig.__post_init__()` or factory validation layer.

### Duplicate Data Loading Calls

Training CLI loads `RawData.from_file()` at CLI level [train.py:636-638], then passes to `run_cdi_example_torch()` which may reload or reuse. Inference CLI loads at CLI level [inference.py:550-554], then manually processes tensors. This suggests:
- Training path: Factory should handle data loading (not CLI)
- Inference path: `_reassemble_cdi_image_torch()` should accept `RawData` directly

---

## Architecture Dependencies

### Import Graph: Training CLI

```
ptycho_torch/train.py
  ├─ ptycho_torch.config_params (DataConfig, ModelConfig, TrainingConfig, etc.)
  ├─ ptycho_torch.model (PtychoPINN_Lightning)
  ├─ ptycho_torch.utils (config_to_json_serializable_dict, load_config_from_json)
  ├─ ptycho_torch.train_utils (set_seed, get_training_strategy, PtychoDataModule, ModelFineTuner)
  ├─ ptycho_torch.config_factory (create_training_payload)
  ├─ ptycho.config.config (PyTorchExecutionConfig, update_legacy_dict)
  ├─ ptycho.raw_data (RawData)
  ├─ ptycho_torch.workflows.components (run_cdi_example_torch)
  ├─ lightning (L.Trainer, callbacks, strategies)
  ├─ mlflow.pytorch (autolog, experiment tracking)
  └─ torch (cuda, distributed)
```

### Import Graph: Inference CLI

```
ptycho_torch/inference.py
  ├─ ptycho_torch.config_params (DataConfig, ModelConfig, etc.) [legacy path only]
  ├─ ptycho_torch.utils (load_config_from_json) [legacy path only]
  ├─ ptycho_torch.reassembly (reconstruct_image_barycentric) [legacy path only]
  ├─ ptycho_torch.dataloader (PtychoDataset) [legacy path only]
  ├─ ptycho_torch.config_factory (create_inference_payload)
  ├─ ptycho.config.config (PyTorchExecutionConfig)
  ├─ ptycho.raw_data (RawData)
  ├─ ptycho_torch.workflows.components (load_inference_bundle_torch)
  ├─ torch (tensor ops, device management)
  ├─ matplotlib.pyplot (plot generation)
  └─ mlflow (legacy path only)
```

---

## Key File:Line References

### Training CLI (`ptycho_torch/train.py`)

| Line Range | Description | Refactor Target |
|------------|-------------|-----------------|
| 377-481    | `cli_main()` argparse definition | **Extract to `ptycho_torch/cli/training_args.py:parse_training_args()`** |
| 484-491    | Legacy vs new interface detection | **Keep in CLI wrapper** |
| 509-581    | Execution config construction + validation | **Extract to `ptycho_torch/cli/shared.py:build_execution_config_from_args()`** |
| 584-630    | Factory invocation + config assembly | **Keep in CLI wrapper (core delegation)** |
| 636-638    | Data loading | **Remove (factory should handle)** |
| 641-647    | Workflow orchestration | **Keep in CLI wrapper (core delegation)** |
| 144-374    | `main()` function | **Deprecate OR extract to `ptycho_torch/workflows/training.py:train_with_lightning()`** |

### Inference CLI (`ptycho_torch/inference.py`)

| Line Range | Description | Refactor Target |
|------------|-------------|-----------------|
| 293-410    | `cli_main()` argparse definition | **Extract to `ptycho_torch/cli/inference_args.py:parse_inference_args()`** |
| 414-442    | Execution config construction + validation | **Extract to `ptycho_torch/cli/shared.py:build_execution_config_from_args()`** |
| 463-504    | Factory invocation | **Keep in CLI wrapper** |
| 508-546    | Bundle loading + model device move | **Keep in CLI wrapper** |
| 550-554    | Data loading | **Keep in CLI wrapper (required for workflow)** |
| 565-629    | Manual inference loop (tensor ops, forward pass, aggregation) | **Extract to `ptycho_torch/workflows/components.py:run_simple_inference_torch()`** |
| 629        | Artifact generation | **Keep in CLI wrapper (test contract)** |

---

## Mock/Fixture Dependencies (Test Context)

### Training CLI Tests (`tests/torch/test_cli_train_torch.py`)

**Key Mocks:**
- `ptycho_torch.config_factory.create_training_payload` → Returns mock `TrainingPayload` with `execution_config` field
- `ptycho_torch.model_manager.save_torch_bundle` → Captures models_dict structure (dual-model contract test)
- `ptycho.raw_data.RawData.from_file` → Returns mock RawData
- `ptycho_torch.workflows.components.run_cdi_example_torch` → Simulates training without actual computation

**Fixture Patterns:**
- `minimal_train_args`: Provides `--train_data_file`, `--output_dir`, `--n_images`, `--max_epochs`
- Execution config tests add flags: `--accelerator cpu`, `--deterministic`, `--num-workers 4`, `--learning-rate 5e-4`

**Assertion Strategy:**
- Tests verify `mock_factory.call_args.kwargs['execution_config']` contains correct field values
- Bundle persistence test asserts `save_torch_bundle()` called with `{'autoencoder': ..., 'diffraction_to_obj': ...}`

### Inference CLI Tests (`tests/torch/test_cli_inference_torch.py`)

**Key Mocks:**
- `ptycho_torch.config_factory.create_inference_payload` → Returns mock `InferencePayload` with `execution_config` field
- `ptycho_torch.workflows.components.load_inference_bundle_torch` → Returns `({}, {})`

**Fixture Patterns:**
- `minimal_inference_args`: Provides `--model_path`, `--test_data`, `--output_dir`, `--n_images`
- Execution config tests add flags: `--accelerator cpu`, `--num-workers 4`, `--inference-batch-size 32`

**Assertion Strategy:**
- Tests verify `mock_factory.call_args.kwargs['execution_config']` contains correct field values
- No bundle persistence test (inference doesn't save models)

---

## Phase D Refactor Targets

### Priority 1: Extract Shared Helpers

**New Module:** `ptycho_torch/cli/shared.py`
- `resolve_accelerator(args) -> str`: Handles `--device` → `--accelerator` mapping with deprecation warning
- `build_execution_config_from_args(args, mode='training') -> PyTorchExecutionConfig`: Constructs execution config from parsed args, applies mode-specific defaults
- `validate_execution_config(exec_config) -> None`: Validates ranges (num_workers >= 0, etc.), emits warnings for deterministic+num_workers combination

### Priority 2: Thin Training CLI Wrapper

**New Structure:** `ptycho_torch/train.py`
```python
def cli_main():
    """Thin wrapper: argparse → factory → workflow delegation."""
    args = parse_training_args()  # Extracted argparse logic

    # Legacy path (unchanged)
    if args.ptycho_dir or args.config:
        main(args.ptycho_dir, args.config, disable_mlflow=args.disable_mlflow)
        return

    # New path (simplified)
    validate_paths(args.train_data_file, args.test_data_file, args.output_dir)
    execution_config = build_execution_config_from_args(args, mode='training')

    payload = create_training_payload(
        train_data_file=args.train_data_file,
        output_dir=args.output_dir,
        overrides={'n_groups': args.n_images, ...},
        execution_config=execution_config,
    )

    # Delegate to workflow (no RawData loading here)
    from ptycho_torch.workflows.components import run_cdi_example_torch
    amplitude, phase, results = run_cdi_example_torch(
        train_data=None,  # Factory handles data
        test_data=None,
        config=payload.tf_training_config,
        do_stitching=False
    )

    print(f"✓ Training completed. Outputs: {args.output_dir}/wts.h5.zip")
```

### Priority 3: Thin Inference CLI Wrapper

**New Structure:** `ptycho_torch/inference.py`
```python
def cli_main():
    """Thin wrapper: argparse → factory → workflow delegation."""
    args = parse_inference_args()  # Extracted argparse logic

    execution_config = build_execution_config_from_args(args, mode='inference')

    payload = create_inference_payload(
        model_path=args.model_path,
        test_data_file=args.test_data,
        output_dir=args.output_dir,
        overrides={'n_groups': args.n_images},
        execution_config=execution_config,
    )

    # Delegate to workflow helper (extract manual loop)
    from ptycho_torch.workflows.components import run_simple_inference_torch
    amplitude, phase = run_simple_inference_torch(
        bundle_dir=args.model_path,
        test_data_file=args.test_data,
        config=payload.tf_inference_config,
        execution_config=execution_config,
        quiet=args.quiet,
    )

    save_individual_reconstructions(amplitude, phase, args.output_dir)
    print(f"✓ Inference completed. Outputs: {args.output_dir}")
```

---

## Test Selectors for Baseline Verification

```bash
# Training CLI execution config roundtrip tests (Phase C4.B tests, currently GREEN)
CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_cli_train_torch.py::TestExecutionConfigCLI::test_accelerator_flag_roundtrip -vv
CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_cli_train_torch.py::TestExecutionConfigCLI::test_deterministic_flag_roundtrip -vv
CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_cli_train_torch.py::TestExecutionConfigCLI::test_num_workers_flag_roundtrip -vv
CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_cli_train_torch.py::TestExecutionConfigCLI::test_learning_rate_flag_roundtrip -vv
CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_cli_train_torch.py::TestExecutionConfigCLI::test_multiple_execution_config_flags -vv
CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_cli_train_torch.py::TestExecutionConfigCLI::test_bundle_persistence -vv

# Inference CLI execution config roundtrip tests (Phase C4.D tests, currently GREEN)
CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_cli_inference_torch.py::TestInferenceCLI::test_accelerator_flag_roundtrip -vv
CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_cli_inference_torch.py::TestInferenceCLI::test_num_workers_flag_roundtrip -vv
CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_cli_inference_torch.py::TestInferenceCLI::test_inference_batch_size_flag_roundtrip -vv
CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_cli_inference_torch.py::TestInferenceCLI::test_multiple_execution_config_flags -vv
```

---

## Conclusion

The current CLI architecture successfully integrates Phase C4 execution config flags and factory-based configuration, but suffers from:
1. **Monolithic CLI functions** (377 lines training, 349 lines inference)
2. **Duplicate device mapping/validation logic** between training and inference
3. **Business logic in CLI scope** (data loading, tensor operations, manual inference loop)
4. **Legacy interface burden** requiring if-branch complexity

Phase D refactoring will extract shared helpers, thin CLI wrappers to pure argument-parsing shims, and delegate orchestration to `ptycho_torch/workflows/components.py` functions. This will improve testability, reduce duplication, and align with ADR-003 thin-wrapper design principles.

**Next Steps:** Run baseline test selectors (A2) and document legacy flag deprecation strategy (A3).
