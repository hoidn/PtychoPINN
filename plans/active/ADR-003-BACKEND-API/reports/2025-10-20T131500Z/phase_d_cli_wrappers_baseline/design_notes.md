# Phase D.A Design Notes: Legacy Flag Handling and Deprecation Strategy

**Initiative:** ADR-003-BACKEND-API — Standardize PyTorch backend API
**Phase:** D.A — Baseline & Design Decisions
**Date:** 2025-10-20
**Author:** Ralph (Engineer Loop)

---

## Purpose

This document records design decisions for handling overlapping CLI flags, legacy interface paths, and deprecation timelines as we refactor `ptycho_torch/train.py` and `ptycho_torch/inference.py` into thin wrappers. These decisions will guide Phase B (training CLI refactor) and Phase C (inference CLI refactor).

---

## Decision Summary

| Decision ID | Topic | Resolution | Rationale |
|-------------|-------|------------|-----------|
| **D1** | `--device` vs `--accelerator` coexistence | **Deprecate `--device`, emit warning, prefer `--accelerator`** | Phase C4 introduced `--accelerator` with richer semantics ('auto', 'gpu', 'tpu', 'mps'). Keeping both creates user confusion. |
| **D2** | Legacy interface (`--ptycho_dir`, `--config`) | **Preserve in Phase D, mark for Phase E removal** | Low usage suspected; keeping allows graceful migration but adds if-branch complexity. |
| **D3** | `--disable_mlflow` scope | **Repurpose for progress bar control during Phase D** | MLflow not implemented in workflows; flag currently controls `enable_progress_bar` via `execution_config.enable_progress_bar = not args.disable_mlflow`. Clean semantic TBD Phase E. |
| **D4** | Duplicate validation logic (num_workers, learning_rate) | **Move to `PyTorchExecutionConfig.__post_init__()` or factory layer** | CLI should not perform business logic validation; dataclass is authoritative source. |
| **D5** | Duplicate device mapping logic | **Extract to `ptycho_torch/cli/shared.py:resolve_accelerator()`** | Exact duplication between train/inference CLIs; single shared helper ensures consistency. |
| **D6** | RawData loading in CLI scope (training) | **Remove from CLI, delegate to factory or workflow** | Factory already validates paths; duplicate loading creates coupling and potential CONFIG-001 violations. |
| **D7** | Manual inference loop in CLI scope | **Extract to `ptycho_torch/workflows/components.py:run_simple_inference_torch()`** | 65 lines of tensor operations (inference.py:565-629) belong in workflow helpers, not CLI. |
| **D8** | Legacy MLflow inference path | **Preserve separate entry point, mark for Phase E deprecation** | Used by legacy callers; refactoring not in Phase D scope. |

---

## D1: `--device` vs `--accelerator` Coexistence

### Current Behavior (Phase C4)

Both training and inference CLIs accept **two overlapping flags**:
- `--device` (legacy): Choices = `['cpu', 'cuda']`
- `--accelerator` (new): Choices = `['auto', 'cpu', 'gpu', 'cuda', 'tpu', 'mps']`

**Conflict Resolution Logic** (train.py:545-556, inference.py:418-429):
```python
resolved_accelerator = args.accelerator
if args.device and args.accelerator == 'auto':
    # Map legacy --device to --accelerator if accelerator not explicitly set
    resolved_accelerator = 'cpu' if args.device == 'cpu' else 'gpu'
elif args.device and args.accelerator != 'auto':
    # Warn if both specified
    warnings.warn(
        "--device is deprecated and will be removed in Phase D. "
        "Use --accelerator instead. Ignoring --device value.",
        DeprecationWarning
    )
```

### Problems

1. **User confusion:** Two flags for same concept
2. **Ambiguous behavior:** What happens if user specifies `--device cuda --accelerator cpu`? (Answer: accelerator wins, device ignored + warning)
3. **Test complexity:** Need to test 3 scenarios (device-only, accelerator-only, both)
4. **Documentation burden:** Must explain precedence rules

### Decision: Deprecate `--device`

**Phase D Action:**
- **Keep both flags** for backward compatibility during Phase D refactoring
- **Emit `DeprecationWarning`** when `--device` is used (already implemented)
- **Update help text** to explicitly state `--device` is deprecated:
  ```python
  parser.add_argument('--device', type=str, choices=['cpu', 'cuda'], default=None,
                     help='[DEPRECATED] Use --accelerator instead. Kept for backward compatibility only.')
  ```
- **Update docs/workflows/pytorch.md** to recommend `--accelerator` in all examples

**Phase E Action (Future):**
- **Remove `--device` argument** entirely
- **Remove conflict resolution logic** (lines 545-556 in train.py, 418-429 in inference.py)
- **Update tests** to remove `--device` usage

**Migration Path for Users:**
```bash
# Old (deprecated)
python -m ptycho_torch.train --device cuda ...

# New (preferred)
python -m ptycho_torch.train --accelerator gpu ...
```

---

## D2: Legacy Interface Preservation

### Current State

**Training CLI** supports two distinct interfaces:
1. **Legacy:** `--ptycho_dir <dir> --config <json> [--disable_mlflow]`
2. **New:** `--train_data_file <npz> --output_dir <dir> --n_images <int> ...`

**Inference CLI** supports two paths:
1. **Legacy MLflow:** `--run_id <id> --infer_dir <dir> [--file_index <int>]`
2. **New Lightning:** `--model_path <dir> --test_data <npz> --output_dir <dir> ...`

**Interface Detection:** CLIs reject mixed usage (error exit if both detected) [train.py:484-491]

### Problems

1. **Low usage suspected:** No evidence of legacy interface usage in recent tests/workflows
2. **Code complexity:** Requires if-branch logic, separate code paths, separate validation
3. **Testing burden:** Must maintain tests for both paths
4. **Migration friction:** Users may not know which interface to use

### Decision: Preserve in Phase D, Plan Phase E Removal

**Phase D Action:**
- **Keep legacy paths unchanged** (no refactoring during CLI thin-wrapper work)
- **Add usage tracking comment** to `cli_main()`:
  ```python
  # Legacy interface: Preserved for backward compatibility.
  # Scheduled for removal in Phase E (ADR-003).
  # If you are using this interface, please migrate to the new CLI flags.
  ```
- **Document migration path** in `docs/workflows/pytorch.md` §12 (CLI usage)

**Phase E Governance Question:**
- Should we deprecate legacy interfaces immediately, or run telemetry to detect active usage?
- Recommendation: Add `warnings.warn("Legacy interface deprecated...", DeprecationWarning)` in Phase D, track warnings in CI/user reports

**Migration Path for Users:**

**Training:**
```bash
# Old (legacy)
python -m ptycho_torch.train --ptycho_dir data/ --config config.json

# New (preferred)
python -m ptycho_torch.train \
  --train_data_file data/train.npz \
  --output_dir outputs/ \
  --n_images 512 \
  --max_epochs 10 \
  --gridsize 2 \
  --batch_size 4
```

**Inference:**
```bash
# Old (MLflow)
python -m ptycho_torch.inference --run_id abc123 --infer_dir data/

# New (Lightning checkpoint)
python -m ptycho_torch.inference \
  --model_path outputs/ \
  --test_data data/test.npz \
  --output_dir inference_outputs/ \
  --n_images 32
```

---

## D3: `--disable_mlflow` Semantic Overload

### Current Behavior

**Training CLI:**
- `--disable_mlflow` flag **suppresses MLflow autologging** (train.py:306-307)
- BUT also **controls progress bar** via `execution_config.enable_progress_bar = not args.disable_mlflow` (train.py:580)

**Inference CLI:**
- `--quiet` flag **controls progress bar** via `execution_config.enable_progress_bar = not args.quiet` (inference.py:441)
- No `--disable_mlflow` flag (MLflow not used in new inference path)

### Problems

1. **Semantic overload:** `--disable_mlflow` name implies MLflow-specific behavior, but affects progress bar visibility
2. **Inconsistency:** Training uses `--disable_mlflow` for progress control, inference uses `--quiet`
3. **User confusion:** Why does disabling MLflow hide progress bars?
4. **MLflow not implemented:** Workflows don't actually use MLflow (TODO comments in `run_cdi_example_torch()`)

### Decision: Repurpose for Progress Control, Add `--quiet` Alias

**Phase D Action:**
- **Keep `--disable_mlflow`** for backward compatibility
- **Add `--quiet` flag** as alias for training CLI:
  ```python
  parser.add_argument('--quiet', action='store_true',
                     help='Suppress progress bars and verbose output (alias for --disable_mlflow)')
  parser.add_argument('--disable_mlflow', action='store_true',
                     help='[DEPRECATED] Suppress MLflow tracking and progress bars. Use --quiet instead.')
  ```
- **Reconcile flag semantics:**
  ```python
  quiet_mode = args.quiet or args.disable_mlflow
  execution_config = PyTorchExecutionConfig(
      enable_progress_bar=(not quiet_mode),
      # ... other fields
  )
  ```
- **Update help text** to clarify `--quiet` is preferred

**Phase E Action (Future):**
- **Remove `--disable_mlflow`** flag entirely
- **Rename to `--verbose` / `--no-verbose`** for positive semantics?
- **Implement actual MLflow integration** OR **remove MLflow comments from code**

---

## D4: Duplicate Validation Logic

### Current Duplication

**Training CLI** (train.py:558-564):
```python
if args.num_workers < 0:
    print(f"ERROR: --num-workers must be >= 0, got {args.num_workers}")
    sys.exit(1)
if args.learning_rate <= 0:
    print(f"ERROR: --learning-rate must be > 0, got {args.learning_rate}")
    sys.exit(1)
```

**Inference CLI** (inference.py:431-435):
```python
if args.num_workers < 0:
    raise ValueError(f"--num-workers must be >= 0, got {args.num_workers}")
if args.inference_batch_size is not None and args.inference_batch_size <= 0:
    raise ValueError(f"--inference-batch-size must be > 0, got {args.inference_batch_size}")
```

### Problems

1. **DRY violation:** Same logic in two places
2. **Inconsistent error handling:** Training uses `print + sys.exit(1)`, inference uses `raise ValueError`
3. **CLI responsibility creep:** CLI should parse args, not validate business rules
4. **Dataclass is authoritative:** `PyTorchExecutionConfig` should enforce its own invariants

### Decision: Move Validation to Dataclass `__post_init__`

**Phase D Action:**
- **Add validation to `PyTorchExecutionConfig.__post_init__()`** in `ptycho/config/config.py`:
  ```python
  @dataclass
  class PyTorchExecutionConfig:
      accelerator: str = 'cpu'
      deterministic: bool = True
      num_workers: int = 0
      learning_rate: float = 1e-3
      inference_batch_size: Optional[int] = None
      enable_progress_bar: bool = True

      def __post_init__(self):
          if self.num_workers < 0:
              raise ValueError(f"num_workers must be >= 0, got {self.num_workers}")
          if self.learning_rate <= 0:
              raise ValueError(f"learning_rate must be > 0, got {self.learning_rate}")
          if self.inference_batch_size is not None and self.inference_batch_size <= 0:
              raise ValueError(f"inference_batch_size must be > 0, got {self.inference_batch_size}")
          if self.accelerator not in {'auto', 'cpu', 'gpu', 'cuda', 'tpu', 'mps'}:
              raise ValueError(f"Invalid accelerator: {self.accelerator}")
  ```
- **Remove CLI validation** (train.py:558-564, inference.py:431-435)
- **Catch `ValueError` in CLI** and re-raise with friendly message:
  ```python
  try:
      execution_config = PyTorchExecutionConfig(...)
  except ValueError as e:
      print(f"ERROR: {e}")
      sys.exit(1)
  ```

---

## D5: Duplicate Device Mapping Logic

### Current Duplication

**Training CLI** (train.py:545-556) and **Inference CLI** (inference.py:418-429) contain **identical** 12-line device mapping blocks.

### Decision: Extract to Shared Helper

**New Module:** `ptycho_torch/cli/shared.py`

```python
"""Shared CLI utilities for PyTorch backend (ADR-003 Phase D)."""

import warnings
from typing import Optional


def resolve_accelerator(
    accelerator: str = 'auto',
    device: Optional[str] = None,
) -> str:
    """
    Resolve accelerator from CLI args, handling --device backward compatibility.

    Args:
        accelerator: Value from --accelerator flag (default: 'auto')
        device: Value from --device flag (deprecated, optional)

    Returns:
        Resolved accelerator string ('cpu', 'gpu', 'cuda', 'tpu', 'mps', 'auto')

    Emits:
        DeprecationWarning if device is specified

    Examples:
        >>> resolve_accelerator('cpu', None)
        'cpu'
        >>> resolve_accelerator('auto', 'cuda')  # Legacy --device usage
        'gpu'
        >>> resolve_accelerator('cpu', 'cuda')  # Conflict: accelerator wins
        'cpu'
    """
    resolved = accelerator

    if device and accelerator == 'auto':
        # Map legacy --device to --accelerator
        warnings.warn(
            "--device is deprecated and will be removed in Phase E. "
            "Use --accelerator instead.",
            DeprecationWarning,
            stacklevel=2
        )
        resolved = 'cpu' if device == 'cpu' else 'gpu'

    elif device and accelerator != 'auto':
        # Conflict: accelerator takes precedence
        warnings.warn(
            "--device is deprecated and will be removed in Phase E. "
            "Use --accelerator instead. Ignoring --device value.",
            DeprecationWarning,
            stacklevel=2
        )
        # resolved = accelerator (no change)

    return resolved
```

**Phase D Refactor:**
- Replace train.py:545-556 with `resolved_accelerator = resolve_accelerator(args.accelerator, args.device)`
- Replace inference.py:418-429 with same call
- Add unit tests to `tests/torch/test_cli_shared.py`

---

## D6: RawData Loading in Training CLI

### Current Behavior

**Training CLI** (train.py:636-638):
```python
from ptycho.raw_data import RawData
train_data = RawData.from_file(str(train_data_file))
test_data = RawData.from_file(str(test_data_file)) if test_data_file else None
```

**Then passes to workflow** (train.py:641-647):
```python
amplitude, phase, results = run_cdi_example_torch(
    train_data=train_data,
    test_data=test_data,
    config=payload.tf_training_config,
    do_stitching=False
)
```

### Problems

1. **Duplicate loading:** Factory already validated paths; workflow may reload data internally
2. **CONFIG-001 risk:** Loading data before workflow invocation may trigger legacy module imports
3. **Coupling:** CLI now depends on `ptycho.raw_data` module
4. **Responsibility creep:** CLI should parse args and delegate, not manage data I/O

### Decision: Remove from CLI, Delegate to Workflow

**Phase B Refactor:**
- **Remove RawData loading** from CLI (train.py:636-638)
- **Update `run_cdi_example_torch()` signature** to accept file paths instead of `RawData` objects:
  ```python
  def run_cdi_example_torch(
      train_data_file: Path,  # Changed from train_data: RawData
      test_data_file: Optional[Path],  # Changed from test_data: Optional[RawData]
      config: TrainingConfig,
      do_stitching: bool = False,
  ) -> tuple[Optional[np.ndarray], Optional[np.ndarray], dict]:
      """
      Run PyTorch training workflow from NPZ file paths.
      Handles RawData loading internally after CONFIG-001 compliance.
      """
      from ptycho.raw_data import RawData
      train_data = RawData.from_file(str(train_data_file))
      test_data = RawData.from_file(str(test_data_file)) if test_data_file else None
      # ... rest of workflow
  ```

**CLI becomes:**
```python
amplitude, phase, results = run_cdi_example_torch(
    train_data_file=train_data_file,  # Path object
    test_data_file=test_data_file,    # Path object or None
    config=payload.tf_training_config,
    do_stitching=False
)
```

---

## D7: Manual Inference Loop in CLI Scope

### Current Behavior

**Inference CLI** contains **65 lines** of tensor operations (inference.py:565-629):
- Numpy → torch conversion
- DTYPE enforcement (float32)
- Transpose handling `(H,W,n) → (n,H,W)`
- Subsetting `diffraction[:n_groups]`
- Channel dimension insertion `.unsqueeze(1)`
- Probe preparation (multiple unsqueeze)
- Dummy positions tensor creation
- `model.forward_predict()` call
- Result aggregation `np.mean(reconstruction_cpu, axis=0)`
- Amplitude/phase extraction

### Problems

1. **Business logic in CLI:** This belongs in `workflows/components.py`
2. **Duplication risk:** If another caller needs inference, they must reimplement this logic
3. **Testing burden:** Must test tensor operations via CLI mock chains
4. **Maintenance:** Future changes to inference require CLI edits

### Decision: Extract to `run_simple_inference_torch()`

**New Function:** `ptycho_torch/workflows/components.py`

```python
def run_simple_inference_torch(
    bundle_dir: Path,
    test_data_file: Path,
    config: InferenceConfig,
    execution_config: PyTorchExecutionConfig,
    quiet: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run simplified PyTorch inference: load bundle, predict, extract amplitude/phase.

    This function provides a lightweight inference path for CLI and test usage,
    avoiding the full reassembly pipeline (_reassemble_cdi_image_torch).

    Args:
        bundle_dir: Directory containing wts.h5.zip checkpoint
        test_data_file: Path to test NPZ file (DATA-001 compliant)
        config: InferenceConfig with n_groups, model params
        execution_config: PyTorchExecutionConfig for device/batch settings
        quiet: If True, suppress progress output

    Returns:
        (amplitude, phase): Tuple of 2D numpy arrays

    Raises:
        RuntimeError: If bundle loading or forward pass fails
        ValueError: If test data violates DATA-001 contract

    References:
        - Phase D7 decision: plans/.../design_notes.md §D7
        - Test contract: tests/torch/test_integration_workflow_torch.py
    """
    # Move inference.py:508-629 logic here
    # Return (amplitude, phase) for CLI consumption
```

**CLI becomes:**
```python
from ptycho_torch.workflows.components import run_simple_inference_torch

amplitude, phase = run_simple_inference_torch(
    bundle_dir=args.model_path,
    test_data_file=args.test_data,
    config=payload.tf_inference_config,
    execution_config=execution_config,
    quiet=args.quiet,
)

save_individual_reconstructions(amplitude, phase, args.output_dir)
```

---

## D8: Legacy MLflow Inference Path

### Current State

Inference CLI supports two distinct entry points:
1. **New Lightning path:** `cli_main()` [inference.py:293-641]
2. **Legacy MLflow path:** `load_and_predict()` [inference.py:96-184]

Entry point selection based on `sys.argv` inspection [inference.py:645-670]:
```python
if sys.argv[1] in ['--model_path', '--help', '-h']:
    sys.exit(cli_main())  # New path
else:
    # Legacy MLflow path
    parser.add_argument('--run_id', ...)
    load_and_predict(run_id, infer_dir, ...)
```

### Problems

1. **Fragile dispatch:** Relies on first positional arg, breaks if user adds flags before subcommand
2. **Code duplication:** Two complete inference implementations
3. **Testing burden:** Must maintain tests for both paths
4. **User confusion:** Which interface should I use?

### Decision: Preserve Separate Entry Point, Plan Phase E Removal

**Phase D Action:**
- **No refactoring of legacy path** (out of scope for CLI thin-wrapper work)
- **Add deprecation warning** to legacy path:
  ```python
  else:
      # Legacy MLflow-based inference path (DEPRECATED)
      warnings.warn(
          "MLflow-based inference (--run_id) is deprecated and will be removed in Phase E. "
          "Please migrate to Lightning checkpoint inference (--model_path).",
          DeprecationWarning
      )
      # ... rest of legacy logic
  ```
- **Document migration path** in `docs/workflows/pytorch.md`

**Phase E Action (Future):**
- **Remove `load_and_predict()` function** entirely
- **Remove legacy argparse branch** (inference.py:651-670)
- **Simplify entry point** to single `cli_main()` call

---

## Implementation Checklist for Phase D Refactoring

### Phase B: Training CLI Thin Wrapper

- [ ] Extract `ptycho_torch/cli/shared.py:resolve_accelerator()`
- [ ] Extract `ptycho_torch/cli/training_args.py:parse_training_args()`
- [ ] Add `--quiet` flag as alias for `--disable_mlflow`
- [ ] Update `--device` help text to mark deprecated
- [ ] Remove RawData loading from CLI (delegate to workflow)
- [ ] Update `run_cdi_example_torch()` to accept file paths
- [ ] Add validation to `PyTorchExecutionConfig.__post_init__()`
- [ ] Remove CLI validation logic (train.py:558-564)
- [ ] Update `docs/workflows/pytorch.md` CLI examples
- [ ] Add unit tests for `resolve_accelerator()` in `tests/torch/test_cli_shared.py`

### Phase C: Inference CLI Thin Wrapper

- [ ] Extract `ptycho_torch/cli/inference_args.py:parse_inference_args()`
- [ ] Extract `run_simple_inference_torch()` to `workflows/components.py`
- [ ] Move manual inference loop (inference.py:565-629) into workflow helper
- [ ] Add deprecation warning to legacy MLflow path
- [ ] Remove CLI validation logic (inference.py:431-435)
- [ ] Update `docs/workflows/pytorch.md` inference examples

### Phase D: Smoke Tests & Documentation

- [ ] Run smoke tests with both `--device` and `--accelerator` (expect warnings)
- [ ] Verify test suite still passes (both CLI test modules)
- [ ] Update `docs/workflows/pytorch.md` §7 (CLI Reference) with deprecation notices
- [ ] Add migration guide to `docs/workflows/pytorch.md` §14 (Troubleshooting)
- [ ] Record Phase E backlog items (remove legacy paths, MLflow decision)

---

## Open Questions for Phase E Governance

1. **MLflow Integration:** Should we implement actual MLflow tracking in workflows, or remove all MLflow references?
   - **Recommendation:** Remove references; Lightning has native experiment tracking via `TensorBoardLogger`

2. **Legacy Interface Telemetry:** Should we add usage tracking before removal?
   - **Recommendation:** Add `warnings.warn()` in Phase D, monitor user reports for 1-2 cycles

3. **`--verbose` Semantics:** Should we flip `--quiet` to `--verbose` / `--no-verbose` for positive semantics?
   - **Recommendation:** Keep `--quiet` (aligns with Unix conventions: `grep --quiet`, `tar --quiet`)

4. **Execution Config Persistence:** Should `PyTorchExecutionConfig` be saved to checkpoint bundles?
   - **Recommendation:** No; execution config is runtime-specific (CPU vs GPU). Only model/data config should persist.

5. **Inference Batch Size Default:** Should `inference_batch_size` default to `training_config.batch_size` or `1`?
   - **Recommendation:** Default to `None` (use training batch_size); explicit override for inference-specific tuning

---

## Conclusion

Phase D design decisions prioritize **backward compatibility** during refactoring while establishing **clear deprecation timelines** for legacy interfaces. By extracting shared helpers (`resolve_accelerator()`, validation logic) and delegating business logic to workflow components, we reduce duplication and improve testability while preserving existing CLI contracts.

**Next Steps:**
- Implement Phase B refactoring (training CLI thin wrapper)
- Validate via existing test suite (ensure GREEN)
- Update docs with deprecation notices
- Capture Phase E backlog items for governance review
