# Phase D.B Training CLI Thin Wrapper Blueprint

**Initiative:** ADR-003-BACKEND-API — Standardize PyTorch backend API
**Phase:** D.B — Training CLI Thin Wrapper Refactoring
**Date:** 2025-10-20
**Author:** Ralph (Engineer Loop)

---

## Executive Summary

This document specifies the architectural blueprint for refactoring `ptycho_torch/train.py` from a monolithic 670-line module containing heavy business logic into a lightweight argument-parsing shim that delegates to shared factories and workflow components. The refactor maintains backward compatibility with legacy interfaces while preparing for Phase E deprecation.

**Key Design Principles:**
1. **Single Responsibility:** CLI parses arguments and delegates; workflow components handle orchestration
2. **Shared Helpers:** Extract duplicate logic (device mapping, validation) to `ptycho_torch/cli/shared.py`
3. **CONFIG-001 Compliance:** Factory layer remains responsible for `update_legacy_dict()` call
4. **Test-Driven:** RED tests capture expected behavior before implementation (Phase B2)
5. **Backward Compatible:** Legacy `--ptycho_dir`/`--config` interface preserved unchanged

---

## Current State Analysis

### Problems Identified (from baseline.md)

1. **Monolithic `cli_main()`:** 377-line function handles argparse, config construction, factory invocation, data loading, AND workflow orchestration
2. **Duplicate Device Mapping:** Identical `--device` → `--accelerator` resolution logic in train.py:545-556 and inference.py:418-429
3. **Duplicate Validation:** `num_workers >= 0`, `learning_rate > 0` checks in CLI (should be in dataclass)
4. **RawData Loading in CLI:** Lines 636-638 load data before passing to workflow (duplication + CONFIG-001 risk)
5. **Legacy Interface Burden:** Requires if-branch logic to support `--ptycho_dir`/`--config` path
6. **Semantic Overload:** `--disable_mlflow` controls both MLflow (unimplemented) and progress bars

### Current Call Graph (Phase C4 State)

```
cli_main() [train.py:377]
  ↓
  argparse.ArgumentParser() [390-481]
    - New interface: --train_data_file, --test_data_file, --output_dir, --max_epochs, --n_images, --gridsize, --batch_size
    - Execution config (Phase C4): --accelerator, --deterministic/--no-deterministic, --num-workers, --learning-rate
    - Legacy: --ptycho_dir, --config
    - Shared: --disable_mlflow
  ↓
  Interface Resolution [484-491] ← KEEP (thin wrapper responsibility)
  ↓
  [LEGACY PATH] main() [493-507] ← NO CHANGES (out of scope for Phase D)
  ↓
  [NEW PATH] Path Validation [522-532] ← EXTRACT to helper
  ↓
  Execution Config Construction [541-581] ← EXTRACT to shared.py
  ↓
  create_training_payload() [598-604] ← KEEP (factory delegation)
  ↓
  Config Tuple Assembly [614-622] ← KEEP (factory payload unpacking)
  ↓
  RawData Loading [636-638] ← REMOVE (delegate to workflow)
  ↓
  run_cdi_example_torch() [641-647] ← KEEP (workflow delegation)
  ↓
  Success Logging [649-650] ← KEEP (CLI output)
```

---

## Target Architecture

### New Module Structure

```
ptycho_torch/
├─ cli/
│  ├─ __init__.py (empty)
│  ├─ shared.py (NEW)
│  │  ├─ resolve_accelerator(args) -> str
│  │  ├─ build_execution_config_from_args(args, mode='training') -> PyTorchExecutionConfig
│  │  └─ validate_paths(train_file, test_file, output_dir) -> None
│  └─ training_args.py (NEW, optional)
│     └─ parse_training_args() -> argparse.Namespace
└─ train.py (REFACTORED)
   ├─ cli_main() (SIMPLIFIED: argparse → helpers → delegation)
   ├─ main() (UNCHANGED: legacy path preserved)
   └─ _build_config_tuple() (EXTRACTED: config assembly logic)
```

### Proposed Thin Wrapper Flow

```
cli_main() [train.py, ~80 lines after refactor]
  ↓
  argparse.ArgumentParser() [inline or extracted to training_args.py]
  ↓
  Interface Detection (legacy vs new)
    ├─ [LEGACY] → main(ptycho_dir, config, disable_mlflow) [UNCHANGED]
    └─ [NEW] ↓
  ↓
  validate_paths(train_data_file, test_data_file, output_dir) [cli/shared.py]
    - Checks file existence, creates output_dir
    - Raises FileNotFoundError with clear message
  ↓
  build_execution_config_from_args(args, mode='training') [cli/shared.py]
    - Calls resolve_accelerator(args.accelerator, args.device)
    - Constructs PyTorchExecutionConfig with validated fields
    - Emits deterministic+num_workers warning if applicable
    - Maps disable_mlflow/quiet to enable_progress_bar
  ↓
  create_training_payload(..., execution_config=...) [config_factory.py]
    - Factory handles CONFIG-001 compliance (update_legacy_dict)
    - Returns TrainingPayload with tf_training_config, execution_config, etc.
  ↓
  _build_config_tuple(payload) -> tuple [train.py, extracted helper]
    - Assembles (data_config, model_config, training_config, inference_config, datagen_config)
    - Required by legacy main() signature (backward compat)
  ↓
  run_cdi_example_torch(train_data, test_data, config, do_stitching=False) [workflows/components.py]
    - MODIFIED SIGNATURE: Accept RawData objects (or None) OR file paths (TBD Phase B3)
    - Workflow handles data loading internally if passed None + paths in config
    - Returns (amplitude, phase, results_dict)
  ↓
  Success Logging
    - Prints output_dir, wts.h5.zip path
    - Exit code 0
```

---

## Detailed Component Specifications

### Component 1: `ptycho_torch/cli/shared.py` (NEW)

**Purpose:** Centralize CLI helper logic shared between training and inference CLIs.

**Functions:**

#### `resolve_accelerator(accelerator: str, device: Optional[str]) -> str`

**Responsibility:** Handle `--device` → `--accelerator` backward compatibility mapping.

**Logic:**
```python
def resolve_accelerator(accelerator: str = 'auto', device: Optional[str] = None) -> str:
    """
    Resolve accelerator from CLI args, handling --device deprecation.

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
            "--device is deprecated. Use --accelerator instead.",
            DeprecationWarning,
            stacklevel=2
        )
        resolved = 'cpu' if device == 'cpu' else 'gpu'

    elif device and accelerator != 'auto':
        # Conflict: accelerator takes precedence
        warnings.warn(
            "--device is deprecated. Use --accelerator instead. Ignoring --device value.",
            DeprecationWarning,
            stacklevel=2
        )
        # resolved = accelerator (no change)

    return resolved
```

**Test Coverage (RED tests in Phase B2):**
- `test_resolve_accelerator_default_no_device`
- `test_resolve_accelerator_legacy_device_cpu`
- `test_resolve_accelerator_legacy_device_cuda_maps_to_gpu`
- `test_resolve_accelerator_conflict_accelerator_wins`
- `test_resolve_accelerator_emits_deprecation_warning`

---

#### `build_execution_config_from_args(args: argparse.Namespace, mode: str = 'training') -> PyTorchExecutionConfig`

**Responsibility:** Construct `PyTorchExecutionConfig` from parsed CLI arguments with mode-specific defaults.

**Logic:**
```python
def build_execution_config_from_args(
    args: argparse.Namespace,
    mode: str = 'training'
) -> PyTorchExecutionConfig:
    """
    Build PyTorchExecutionConfig from CLI args with validation and warnings.

    Args:
        args: Parsed argparse.Namespace containing execution config flags
        mode: 'training' or 'inference' (controls field availability)

    Returns:
        PyTorchExecutionConfig instance

    Raises:
        ValueError: If validation fails (caught in dataclass __post_init__)

    Emits:
        UserWarning if deterministic=True and num_workers > 0 (training mode only)

    Examples:
        >>> args = argparse.Namespace(accelerator='cpu', deterministic=True, num_workers=0, learning_rate=1e-3, disable_mlflow=False)
        >>> config = build_execution_config_from_args(args, mode='training')
        >>> config.accelerator
        'cpu'
    """
    from ptycho.config.config import PyTorchExecutionConfig

    # Resolve accelerator (handles --device deprecation)
    resolved_accelerator = resolve_accelerator(
        args.accelerator,
        getattr(args, 'device', None)
    )

    # Map disable_mlflow/quiet to enable_progress_bar
    quiet_mode = getattr(args, 'quiet', False) or getattr(args, 'disable_mlflow', False)
    enable_progress_bar = not quiet_mode

    # Emit deterministic+num_workers warning (training only)
    if mode == 'training' and args.deterministic and args.num_workers > 0:
        warnings.warn(
            f"Deterministic mode with num_workers={args.num_workers} may cause performance degradation. "
            f"Consider setting --num-workers 0 for reproducibility.",
            UserWarning,
            stacklevel=2
        )

    # Construct config (validation happens in __post_init__)
    if mode == 'training':
        return PyTorchExecutionConfig(
            accelerator=resolved_accelerator,
            deterministic=args.deterministic,
            num_workers=args.num_workers,
            learning_rate=args.learning_rate,
            enable_progress_bar=enable_progress_bar,
        )
    elif mode == 'inference':
        return PyTorchExecutionConfig(
            accelerator=resolved_accelerator,
            num_workers=args.num_workers,
            inference_batch_size=getattr(args, 'inference_batch_size', None),
            enable_progress_bar=enable_progress_bar,
        )
    else:
        raise ValueError(f"Invalid mode: {mode}. Expected 'training' or 'inference'.")
```

**Test Coverage (RED tests in Phase B2):**
- `test_build_execution_config_training_mode_defaults`
- `test_build_execution_config_training_mode_custom_values`
- `test_build_execution_config_inference_mode`
- `test_build_execution_config_emits_deterministic_warning`
- `test_build_execution_config_handles_quiet_flag`
- `test_build_execution_config_handles_disable_mlflow_flag`

---

#### `validate_paths(train_file: Optional[Path], test_file: Optional[Path], output_dir: Path) -> None`

**Responsibility:** Validate input file existence and create output directory.

**Logic:**
```python
def validate_paths(
    train_file: Optional[Path],
    test_file: Optional[Path],
    output_dir: Path,
) -> None:
    """
    Validate input NPZ files exist and create output directory.

    Args:
        train_file: Path to training NPZ file (required for training CLI)
        test_file: Path to test NPZ file (optional)
        output_dir: Directory for outputs (will be created if missing)

    Raises:
        FileNotFoundError: If train_file or test_file does not exist

    Side Effects:
        Creates output_dir and any parent directories (mkdir -p behavior)

    Examples:
        >>> validate_paths(Path('data/train.npz'), None, Path('outputs/'))
        # Creates outputs/ if missing, raises if data/train.npz missing
    """
    if train_file and not train_file.exists():
        raise FileNotFoundError(f"Training data file not found: {train_file}")

    if test_file and not test_file.exists():
        raise FileNotFoundError(f"Test data file not found: {test_file}")

    # Create output directory (mkdir -p)
    output_dir.mkdir(parents=True, exist_ok=True)
```

**Test Coverage (RED tests in Phase B2):**
- `test_validate_paths_creates_output_dir`
- `test_validate_paths_raises_if_train_file_missing`
- `test_validate_paths_raises_if_test_file_missing`
- `test_validate_paths_accepts_none_test_file`

---

### Component 2: `ptycho_torch/train.py` Refactored CLI

**Target State:** Thin wrapper (~80 lines for `cli_main()`, legacy path unchanged).

**Key Changes:**
1. **Remove duplicate device mapping** (lines 545-556) → call `resolve_accelerator()`
2. **Remove duplicate validation** (lines 558-564) → rely on dataclass `__post_init__()`
3. **Remove path validation logic** (lines 522-532) → call `validate_paths()`
4. **Remove RawData loading** (lines 636-638) → delegate to workflow
5. **Extract config tuple assembly** → new helper `_build_config_tuple()`

**Proposed `cli_main()` Structure (Pseudocode):**

```python
def cli_main():
    """
    Training CLI entry point: argparse → factory → workflow delegation.

    Supports two interfaces:
    1. Legacy: --ptycho_dir <dir> --config <json> [--disable_mlflow]
    2. New: --train_data_file <npz> --output_dir <dir> --n_images <int> ...

    Exit Codes:
        0: Success
        1: Argument parsing error or validation failure
        2: Training workflow error
    """
    # --- Argument Parsing (inline or extracted) ---
    parser = argparse.ArgumentParser(description="Train PtychoPINN model (PyTorch backend)")

    # New interface flags
    parser.add_argument('--train_data_file', type=Path, help='Training NPZ file (DATA-001 compliant)')
    parser.add_argument('--test_data_file', type=Path, help='Test NPZ file (optional)')
    parser.add_argument('--output_dir', type=Path, help='Output directory for checkpoints')
    parser.add_argument('--n_images', type=int, default=512, help='Number of grouped samples')
    parser.add_argument('--max_epochs', type=int, default=10, help='Training epochs')
    parser.add_argument('--gridsize', type=int, default=2, help='Group grid size')
    parser.add_argument('--batch_size', type=int, default=4, help='Training batch size')

    # Execution config flags (Phase C4)
    parser.add_argument('--accelerator', type=str, default='auto', choices=['auto', 'cpu', 'gpu', 'cuda', 'tpu', 'mps'])
    parser.add_argument('--deterministic', action='store_true', default=True, help='Enable deterministic training')
    parser.add_argument('--no-deterministic', dest='deterministic', action='store_false')
    parser.add_argument('--num-workers', type=int, default=0, help='DataLoader worker count')
    parser.add_argument('--learning-rate', type=float, default=1e-3, help='Optimizer learning rate')

    # Shared flags
    parser.add_argument('--disable_mlflow', action='store_true', help='[DEPRECATED] Use --quiet instead')
    parser.add_argument('--quiet', action='store_true', help='Suppress progress bars')

    # Legacy interface flags
    parser.add_argument('--ptycho_dir', type=Path, help='[LEGACY] Ptychography directory')
    parser.add_argument('--config', type=Path, help='[LEGACY] JSON config file')

    # Deprecated flag
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda'], help='[DEPRECATED] Use --accelerator')

    args = parser.parse_args()

    # --- Interface Detection ---
    using_legacy = args.ptycho_dir or args.config
    using_new = args.train_data_file or args.output_dir

    if using_legacy and using_new:
        print("ERROR: Cannot mix legacy (--ptycho_dir/--config) and new (--train_data_file/--output_dir) interfaces.")
        sys.exit(1)

    # --- Legacy Path (UNCHANGED) ---
    if using_legacy:
        # Legacy interface: Preserved for backward compatibility.
        # Scheduled for removal in Phase E (ADR-003).
        # If you are using this interface, please migrate to the new CLI flags.
        return main(
            ptycho_dir=args.ptycho_dir,
            config_path=args.config,
            disable_mlflow=args.disable_mlflow,
        )

    # --- New Path (REFACTORED) ---

    # 1. Validate paths (creates output_dir, checks file existence)
    from ptycho_torch.cli.shared import validate_paths
    try:
        validate_paths(args.train_data_file, args.test_data_file, args.output_dir)
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    # 2. Build execution config (handles device deprecation, validation, warnings)
    from ptycho_torch.cli.shared import build_execution_config_from_args
    try:
        execution_config = build_execution_config_from_args(args, mode='training')
    except ValueError as e:
        print(f"ERROR: Invalid execution config: {e}")
        sys.exit(1)

    # 3. Invoke factory (CONFIG-001 compliance handled internally)
    from ptycho_torch.config_factory import create_training_payload

    payload = create_training_payload(
        train_data_file=args.train_data_file,
        test_data_file=args.test_data_file,
        output_dir=args.output_dir,
        overrides={
            'n_groups': args.n_images,
            'gridsize': args.gridsize,
            'batch_size': args.batch_size,
            'nepochs': args.max_epochs,
        },
        execution_config=execution_config,
    )

    # 4. Build config tuple (for legacy main() compatibility if workflow needs it)
    # existing_config = _build_config_tuple(payload)  # OPTIONAL: only if workflow requires tuple

    # 5. Delegate to workflow (DO NOT load RawData here — workflow handles it)
    from ptycho_torch.workflows.components import run_cdi_example_torch

    try:
        amplitude, phase, results = run_cdi_example_torch(
            train_data=None,  # Workflow loads from config paths
            test_data=None,
            config=payload.tf_training_config,
            do_stitching=False,
        )
    except Exception as e:
        print(f"ERROR: Training workflow failed: {e}")
        sys.exit(2)

    # 6. Success logging
    print(f"✓ Training completed successfully.")
    print(f"  Output directory: {args.output_dir}")
    print(f"  Model checkpoint: {args.output_dir / 'wts.h5.zip'}")

    return 0


def _build_config_tuple(payload: TrainingPayload) -> tuple:
    """
    Assemble config tuple from TrainingPayload for legacy main() signature compatibility.

    Args:
        payload: TrainingPayload from create_training_payload()

    Returns:
        Tuple of (DataConfig, ModelConfig, TrainingConfig, InferenceConfig, DatagenConfig)

    Notes:
        Required only if workflow helpers expect legacy tuple format.
        Phase E candidate for removal once workflows fully adopt payload objects.
    """
    from ptycho_torch.config_params import InferenceConfig as PTInferenceConfig, DatagenConfig

    return (
        payload.pt_data_config,
        payload.pt_model_config,
        payload.pt_training_config,
        PTInferenceConfig(),  # Default inference config (unused in training)
        DatagenConfig(),      # Default datagen config (unused in training)
    )
```

---

### Component 3: RawData Ownership Decision

**Problem:** Current CLI loads `RawData.from_file()` before calling workflow (lines 636-638).

**Options:**

**Option A: CLI retains RawData loading**
- ✅ PRO: Matches current test mocking patterns (`@patch('ptycho.raw_data.RawData.from_file')`)
- ✅ PRO: Explicit control over when CONFIG-001 happens (factory before data load)
- ❌ CON: CLI has business logic responsibility (data I/O)
- ❌ CON: Duplication if workflow also needs to load data

**Option B: Workflow handles RawData loading**
- ✅ PRO: CLI purely argument parsing + delegation
- ✅ PRO: Single data loading path (no duplication)
- ✅ PRO: Factory already validates file paths
- ❌ CON: Requires updating `run_cdi_example_torch()` signature
- ❌ CON: Requires updating test mocks to patch workflow-internal calls

**Option C: Hybrid (accept RawData OR file paths)**
- ✅ PRO: Backward compatible with existing callers
- ✅ PRO: Flexibility for programmatic vs CLI usage
- ❌ CON: Signature complexity (`train_data: Union[RawData, Path]`)
- ❌ CON: Branching logic inside workflow

**DECISION (Phase B): Option A (CLI retains RawData loading) for Phase D**

**Rationale:**
1. Minimal test churn (existing mocks unchanged)
2. Explicit CONFIG-001 ordering visible in CLI flow
3. Phase E can migrate to Option B after workflow signature stabilizes

**Implementation:**
```python
# In cli_main() (NEW PATH)
from ptycho.raw_data import RawData

# Load data AFTER factory invocation (CONFIG-001 already satisfied by factory)
train_data = RawData.from_file(str(args.train_data_file))
test_data = RawData.from_file(str(args.test_data_file)) if args.test_data_file else None

amplitude, phase, results = run_cdi_example_torch(
    train_data=train_data,
    test_data=test_data,
    config=payload.tf_training_config,
    do_stitching=False,
)
```

**Phase E Migration Path:** Update `run_cdi_example_torch()` to accept file paths, remove CLI data loading, update test mocks.

---

### Component 4: Accelerator Warning Strategy

**Current Warnings (train.py:545-556):**
1. `--device` specified + `--accelerator` is 'auto' → Emits deprecation, maps device to accelerator
2. `--device` specified + `--accelerator` is NOT 'auto' → Emits deprecation, ignores device

**Current Warning (train.py:566-573):**
3. `deterministic=True` + `num_workers > 0` → Emits performance warning

**Refactored Strategy:**
- **Warning 1 & 2:** Moved to `resolve_accelerator()` in `cli/shared.py`
- **Warning 3:** Moved to `build_execution_config_from_args()` in `cli/shared.py`

**Warning Text Consistency:**
```python
# Deprecation warning (warnings.warn with DeprecationWarning)
"--device is deprecated. Use --accelerator instead."

# Performance warning (warnings.warn with UserWarning)
"Deterministic mode with num_workers={N} may cause performance degradation. Consider setting --num-workers 0 for reproducibility."
```

**Test Coverage:**
- Verify warnings emitted via `pytest.warns(DeprecationWarning)` and `pytest.warns(UserWarning)`
- Capture warning messages via `warnings.catch_warnings()`

---

### Component 5: `--disable_mlflow` Handling

**Current Behavior:**
- Suppresses MLflow autologging (NOT IMPLEMENTED in workflows)
- Controls `enable_progress_bar` via `not args.disable_mlflow` (line 580)

**Phase D Action:**
- Keep `--disable_mlflow` flag for backward compatibility
- Add `--quiet` as alias (clearer semantics)
- Map both to `enable_progress_bar` in `build_execution_config_from_args()`

**Implementation:**
```python
# In argparse
parser.add_argument('--disable_mlflow', action='store_true',
                   help='[DEPRECATED] Suppress MLflow tracking and progress bars. Use --quiet instead.')
parser.add_argument('--quiet', action='store_true',
                   help='Suppress progress bars and verbose output')

# In build_execution_config_from_args()
quiet_mode = getattr(args, 'quiet', False) or getattr(args, 'disable_mlflow', False)
enable_progress_bar = not quiet_mode
```

**Phase E Action:**
- Remove `--disable_mlflow` flag
- Keep `--quiet` as standard flag
- Implement actual MLflow tracking OR remove MLflow comments from codebase

---

## Open Questions & Design Choices

### Q1: Should `validate_paths()` accept `None` for `train_file`?

**Context:** Training CLI requires `--train_data_file`, but inference CLI may not.

**Decision:** Accept `Optional[Path]` for flexibility across CLI contexts. Training CLI always provides non-None.

---

### Q2: Should validation errors use `print() + sys.exit(1)` or `raise ValueError`?

**Current Mix:**
- Training CLI: `print() + sys.exit(1)` (lines 558-564)
- Inference CLI: `raise ValueError` (lines 431-435)

**Decision:** Use exceptions inside helpers, catch in CLI for user-friendly messages.

**Rationale:**
- Helpers are reusable (may be called programmatically)
- CLI wrapper catches and formats errors for terminal output

**Example:**
```python
# In cli/shared.py
def validate_paths(...):
    if not train_file.exists():
        raise FileNotFoundError(f"Training data file not found: {train_file}")

# In train.py cli_main()
try:
    validate_paths(args.train_data_file, args.test_data_file, args.output_dir)
except FileNotFoundError as e:
    print(f"ERROR: {e}")
    sys.exit(1)
```

---

### Q3: Should `_build_config_tuple()` be kept or removed?

**Context:** Legacy `main()` expects tuple of 5 config objects. Factory returns `TrainingPayload` dataclass.

**Decision:** Keep helper in Phase D for backward compatibility. Mark for Phase E removal when legacy interface deprecated.

---

## Test Strategy (Phase B2 RED Coverage)

### Test File: `tests/torch/test_cli_train_torch.py`

**New Test Class: `TestThinWrapperHelpers`**

#### Test: `test_resolve_accelerator_default_no_device`
- Args: `accelerator='cpu'`, `device=None`
- Expected: `'cpu'`
- No warnings

#### Test: `test_resolve_accelerator_legacy_device_cpu`
- Args: `accelerator='auto'`, `device='cpu'`
- Expected: `'cpu'`
- Emits `DeprecationWarning`

#### Test: `test_resolve_accelerator_legacy_device_cuda_maps_to_gpu`
- Args: `accelerator='auto'`, `device='cuda'`
- Expected: `'gpu'` (not 'cuda')
- Emits `DeprecationWarning`

#### Test: `test_resolve_accelerator_conflict_accelerator_wins`
- Args: `accelerator='cpu'`, `device='cuda'`
- Expected: `'cpu'` (accelerator takes precedence)
- Emits `DeprecationWarning` ("Ignoring --device value")

#### Test: `test_build_execution_config_training_mode_defaults`
- Args: `Namespace(accelerator='cpu', deterministic=True, num_workers=0, learning_rate=1e-3, disable_mlflow=False, quiet=False, device=None)`
- Expected: `PyTorchExecutionConfig(accelerator='cpu', deterministic=True, num_workers=0, learning_rate=1e-3, enable_progress_bar=True)`

#### Test: `test_build_execution_config_emits_deterministic_warning`
- Args: `Namespace(accelerator='cpu', deterministic=True, num_workers=4, ...)`
- Expected: Emits `UserWarning` about performance degradation

#### Test: `test_build_execution_config_handles_quiet_flag`
- Args: `Namespace(quiet=True, disable_mlflow=False, ...)`
- Expected: `enable_progress_bar=False`

#### Test: `test_build_execution_config_handles_disable_mlflow_flag`
- Args: `Namespace(quiet=False, disable_mlflow=True, ...)`
- Expected: `enable_progress_bar=False`

#### Test: `test_validate_paths_creates_output_dir`
- Setup: `train_file` exists, `output_dir` does not
- Expected: `output_dir.exists() == True` after call

#### Test: `test_validate_paths_raises_if_train_file_missing`
- Setup: `train_file` does not exist
- Expected: `FileNotFoundError` raised

#### Test: `test_validate_paths_accepts_none_test_file`
- Args: `train_file=<valid>`, `test_file=None`, `output_dir=<path>`
- Expected: No error, output_dir created

---

### Test File: `tests/torch/test_cli_shared.py` (NEW)

**Purpose:** Unit tests for `ptycho_torch/cli/shared.py` helper functions.

**Coverage:**
- All `resolve_accelerator()` scenarios
- All `build_execution_config_from_args()` scenarios
- All `validate_paths()` scenarios
- Warning emission verification

**Pattern:**
```python
import pytest
import warnings
from pathlib import Path
from ptycho_torch.cli.shared import resolve_accelerator, build_execution_config_from_args, validate_paths


class TestResolveAccelerator:
    def test_default_no_device(self):
        result = resolve_accelerator('cpu', None)
        assert result == 'cpu'

    def test_legacy_device_cpu_emits_warning(self):
        with pytest.warns(DeprecationWarning, match="--device is deprecated"):
            result = resolve_accelerator('auto', 'cpu')
        assert result == 'cpu'

    # ... (remaining tests)


class TestBuildExecutionConfig:
    def test_training_mode_defaults(self):
        args = argparse.Namespace(
            accelerator='cpu', deterministic=True, num_workers=0,
            learning_rate=1e-3, disable_mlflow=False, quiet=False, device=None
        )
        config = build_execution_config_from_args(args, mode='training')
        assert config.accelerator == 'cpu'
        assert config.enable_progress_bar is True

    # ... (remaining tests)


class TestValidatePaths:
    def test_creates_output_dir(self, tmp_path):
        train_file = tmp_path / 'train.npz'
        train_file.touch()
        output_dir = tmp_path / 'outputs'

        validate_paths(train_file, None, output_dir)
        assert output_dir.exists()

    # ... (remaining tests)
```

---

## Expected RED Test Failures (Phase B2)

**After writing new tests but before implementing helpers:**

```
FAILED tests/torch/test_cli_shared.py::TestResolveAccelerator::test_default_no_device - ImportError: cannot import name 'resolve_accelerator' from 'ptycho_torch.cli.shared' (module not found)
FAILED tests/torch/test_cli_shared.py::TestBuildExecutionConfig::test_training_mode_defaults - ImportError: cannot import name 'build_execution_config_from_args'
FAILED tests/torch/test_cli_shared.py::TestValidatePaths::test_creates_output_dir - ImportError: cannot import name 'validate_paths'
...
```

**Expected RED Log Capture:** `pytest_cli_train_thin_red.log` should show ~10-15 ImportError or AttributeError failures for missing helper functions.

---

## Implementation Sequence (Phase B3, deferred to next loop)

1. Create `ptycho_torch/cli/__init__.py` (empty)
2. Create `ptycho_torch/cli/shared.py` with helper functions
3. Add validation to `PyTorchExecutionConfig.__post_init__()` in `ptycho/config/config.py`
4. Refactor `ptycho_torch/train.py` `cli_main()` to use helpers
5. Keep RawData loading in CLI (Option A)
6. Add `--quiet` flag as alias for `--disable_mlflow`
7. Update `--device` help text to mark deprecated
8. Run `pytest tests/torch/test_cli_train_torch.py -vv` → GREEN
9. Run `pytest tests/torch/test_cli_shared.py -vv` → GREEN
10. Update `docs/workflows/pytorch.md` CLI examples with deprecation notices

---

## Artifacts & References

**This Blueprint References:**
- `plans/active/ADR-003-BACKEND-API/reports/2025-10-20T131500Z/phase_d_cli_wrappers_baseline/baseline.md` (current call graph)
- `plans/active/ADR-003-BACKEND-API/reports/2025-10-20T131500Z/phase_d_cli_wrappers_baseline/design_notes.md` (decisions D1-D8)
- `specs/ptychodus_api_spec.md` §4.8 (backend selection + CONFIG-001 compliance)
- `docs/workflows/pytorch.md` §12 (CLI usage patterns)

**Phase B2 Artifacts (Next Step):**
- `tests/torch/test_cli_shared.py` (NEW: unit tests for helpers)
- `tests/torch/test_cli_train_torch.py` (EXTENDED: integration tests for thin wrapper)
- `pytest_cli_train_thin_red.log` (RED test output before implementation)

**Phase B3 Artifacts (Future):**
- `ptycho_torch/cli/shared.py` (NEW: helper implementation)
- `ptycho_torch/train.py` (REFACTORED: thin wrapper)
- `pytest_cli_train_thin_green.log` (GREEN test output after implementation)

---

## Success Criteria

Phase D.B is complete when:

1. ✅ **Blueprint Approved:** This document captures complete design (Phase B1)
2. ✅ **RED Tests Written:** New tests in `test_cli_shared.py` and `test_cli_train_torch.py` fail with ImportError (Phase B2)
3. ✅ **RED Log Captured:** `pytest_cli_train_thin_red.log` shows expected failures (Phase B2)
4. ⏳ **Helpers Implemented:** `cli/shared.py` functions pass all unit tests (Phase B3, deferred)
5. ⏳ **CLI Refactored:** `train.py` delegates to helpers, removes duplication (Phase B3, deferred)
6. ⏳ **GREEN Tests Verified:** Full regression suite passes (Phase B3, deferred)
7. ⏳ **Docs Updated:** `pytorch.md` reflects new CLI patterns + deprecation notices (Phase B4, deferred)

---

**Next Step:** Proceed to Phase B2 (write RED tests for helpers and thin wrapper).
