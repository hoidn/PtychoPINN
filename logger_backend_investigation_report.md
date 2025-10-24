# Logger Backend Support Investigation Report

## Executive Summary

This report documents the required code changes to implement logger backend support (`--logger` and `--disable_mlflow` flags) for the PyTorch training CLI. The investigation identifies two primary failure points:

1. **CLI argparse**: Missing `--logger` and `--disable_mlflow` flags in ptycho_torch/train.py
2. **Workflow orchestration**: Missing logger instantiation in ptycho_torch/workflows/components.py:_train_with_lightning

---

## Test Failure Analysis

### RED Test Location 1: CLI Argument Parsing
**File:** `/home/ollie/Documents/PtychoPINN2/tests/torch/test_cli_train_torch.py`

**Failing Tests:**
- `test_logger_backend_csv_default` (line 621)
- `test_logger_backend_tensorboard` (line 657)
- `test_logger_backend_none` (line 693)
- `test_disable_mlflow_deprecation_warning` (line 729)

**Expected Behavior:**
- `--logger csv` should set `execution_config.logger_backend='csv'`
- `--logger tensorboard` should set `execution_config.logger_backend='tensorboard'`
- `--logger none` should set `execution_config.logger_backend=None`
- `--disable_mlflow` should emit DeprecationWarning and map to `--logger none`

**Current Failure:**
```
argparse.ArgumentError: unrecognized arguments: --logger csv
```

---

### RED Test Location 2: Logger Instantiation
**File:** `/home/ollie/Documents/PtychoPINN2/tests/torch/test_workflows_components.py`

**Failing Test:** Around line 3318

**Expected Behavior:**
- When `execution_config.logger_backend='csv'`, should instantiate CSVLogger and pass to Trainer
- When `execution_config.logger_backend=None`, should pass `logger=False` to Trainer

**Current Failure:**
- `_train_with_lightning` always passes `logger=False` (hardcoded at line 760)
- No logger instantiation logic exists

---

## Code Modification Requirements

### 1. CLI Argument Addition (ptycho_torch/train.py)

**Location:** After line 476 (after `--learning-rate` flag)

**Required Code:**
```python
parser.add_argument(
    '--logger',
    type=str,
    default='csv',
    choices=['none', 'csv', 'tensorboard', 'wandb', 'mlflow'],
    dest='logger_backend',
    help=(
        'Experiment tracking logger backend (default: csv). '
        'Choices: none (disable logging), csv (CSVLogger), tensorboard (TensorBoardLogger), '
        'wandb (WandbLogger), mlflow (MLFlowLogger). '
        'Logger outputs are saved to output_dir/logs/.'
    )
)
```

**Deprecation Flag Location:** After line 495 (after `--disable-checkpointing`)

**Required Code:**
```python
# Deprecation flag for --disable_mlflow (maps to --logger none)
parser.add_argument(
    '--disable_mlflow',
    action='store_true',
    help=(
        '[DEPRECATED] Use --logger none instead. '
        'Disable experiment tracking (equivalent to --logger none). '
        'Emits DeprecationWarning.'
    )
)
```

**Shared Helper Update Location:** `/home/ollie/Documents/PtychoPINN2/ptycho_torch/cli/shared.py`

**Function:** `build_execution_config_from_args` (line 76-146)

**Required Addition (after line 117):**
```python
# Handle --disable_mlflow deprecation (maps to --logger none)
logger_backend_value = getattr(args, 'logger_backend', 'csv')
if getattr(args, 'disable_mlflow', False):
    warnings.warn(
        "--disable_mlflow is deprecated. Use --logger none instead.",
        DeprecationWarning,
        stacklevel=2
    )
    logger_backend_value = None  # Map to logger_backend=None
elif logger_backend_value == 'none':
    logger_backend_value = None  # Normalize 'none' string to None
```

**Update PyTorchExecutionConfig construction (line 131):**
```python
return PyTorchExecutionConfig(
    accelerator=resolved_accelerator,
    deterministic=args.deterministic,
    num_workers=args.num_workers,
    learning_rate=args.learning_rate,
    enable_progress_bar=enable_progress_bar,
    logger_backend=logger_backend_value,  # ADD THIS LINE
    # ... existing fields
)
```

---

### 2. Logger Instantiation (ptycho_torch/workflows/components.py)

**Location:** `_train_with_lightning` function, before Trainer instantiation (insert after line 741, before line 743)

**Required Code:**
```python
# EB3.C: Instantiate logger based on execution_config.logger_backend (ADR-003 Phase EB3)
logger_instance = False  # Default: no logger
if execution_config.logger_backend is not None:
    # Import Lightning loggers on-demand (torch-optional)
    try:
        if execution_config.logger_backend == 'csv':
            from lightning.pytorch.loggers import CSVLogger
            logger_instance = CSVLogger(
                save_dir=str(output_dir / "logs"),
                name="pytorch_training",
            )
        elif execution_config.logger_backend == 'tensorboard':
            from lightning.pytorch.loggers import TensorBoardLogger
            logger_instance = TensorBoardLogger(
                save_dir=str(output_dir / "logs"),
                name="pytorch_training",
            )
        elif execution_config.logger_backend == 'wandb':
            from lightning.pytorch.loggers import WandbLogger
            logger_instance = WandbLogger(
                save_dir=str(output_dir / "logs"),
                project="ptychopinn",
                name="pytorch_training",
            )
        elif execution_config.logger_backend == 'mlflow':
            from lightning.pytorch.loggers import MLFlowLogger
            logger_instance = MLFlowLogger(
                experiment_name="pytorch_training",
                save_dir=str(output_dir / "logs"),
            )
        else:
            # Unrecognized backend: fall back to False (validation in PyTorchExecutionConfig should prevent this)
            logger.warning(f"Unrecognized logger_backend '{execution_config.logger_backend}', disabling logging")
            logger_instance = False
    except ImportError as e:
        # If logger backend unavailable, warn and disable logging
        logger.warning(f"Logger backend '{execution_config.logger_backend}' requires additional dependencies. "
                      f"Install with: pip install lightning[{execution_config.logger_backend}]. "
                      f"Disabling logging. Error: {e}")
        logger_instance = False
```

**Update Trainer instantiation (line 760):**
Replace:
```python
logger=False,  # Disable default logger for now; MLflow/TensorBoard added in Phase D
```

With:
```python
logger=logger_instance,  # EB3.C: Use instantiated logger or False
```

---

## Existing Code Context

### PyTorchExecutionConfig Field
**File:** `/home/ollie/Documents/PtychoPINN2/ptycho/config/config.py:242`
```python
logger_backend: Optional[str] = None  # Experiment tracking backend: None, 'tensorboard', 'wandb', 'mlflow'
```

**Validation:** No validation currently exists for logger_backend values. Should add to `__post_init__` (line 249):
```python
# Validate logger_backend (if provided)
allowed_loggers = {None, 'csv', 'tensorboard', 'wandb', 'mlflow'}
if self.logger_backend not in allowed_loggers:
    raise ValueError(
        f"logger_backend must be one of {allowed_loggers}, got '{self.logger_backend}'"
    )
```

---

### Existing MLflow Usage Pattern
**File:** `/home/ollie/Documents/PtychoPINN2/ptycho_torch/train.py`

**Line 420-421:** Deprecated `--disable_mlflow` flag already exists with deprecation message:
```python
parser.add_argument('--disable_mlflow', action='store_true',
                   help='[DEPRECATED] Use --quiet instead. Disable MLflow experiment tracking (useful for CI)')
```

**Issue:** Current deprecation message suggests using `--quiet`, but tests expect it to map to `--logger none`. The deprecation message needs updating to match test expectations.

---

## Implementation Order

### Phase 1: CLI Argument Parsing
1. Add `--logger` argument to `ptycho_torch/train.py` (after line 476)
2. Update `--disable_mlflow` help text to reference `--logger none` (line 421)
3. Update `build_execution_config_from_args` in `ptycho_torch/cli/shared.py` to:
   - Handle `--disable_mlflow` deprecation warning
   - Map `logger_backend='none'` to `None`
   - Pass `logger_backend` to PyTorchExecutionConfig constructor

### Phase 2: Logger Instantiation
1. Add logger_backend validation to `PyTorchExecutionConfig.__post_init__` (ptycho/config/config.py:249)
2. Add logger instantiation logic to `_train_with_lightning` (ptycho_torch/workflows/components.py:741)
3. Update Trainer instantiation to use `logger_instance` instead of hardcoded `False` (line 760)

### Phase 3: Test Validation
1. Run `pytest tests/torch/test_cli_train_torch.py::TestPyTorchCLITrainingInterface::test_logger_backend_csv_default -vv`
2. Run `pytest tests/torch/test_cli_train_torch.py::TestPyTorchCLITrainingInterface::test_disable_mlflow_deprecation_warning -vv`
3. Run `pytest tests/torch/test_workflows_components.py -k logger_backend -vv`

---

## Additional Notes

### Import Strategy
The logger instantiation uses lazy imports (inside if/elif blocks) to maintain torch-optional design. If a logger backend requires additional dependencies (e.g., wandb, mlflow), the code gracefully falls back to `logger=False` with a warning.

### Default Behavior
- Default: `logger_backend='csv'` (test expectation per line 727 in test_config_factory.py)
- Legacy: `--disable_mlflow` maps to `logger_backend=None` (maintains backward compatibility)
- Explicit: `--logger none` also maps to `logger_backend=None`

### File References Summary
1. **CLI Flags:** `/home/ollie/Documents/PtychoPINN2/ptycho_torch/train.py:476-495`
2. **Shared Helper:** `/home/ollie/Documents/PtychoPINN2/ptycho_torch/cli/shared.py:117-146`
3. **Config Validation:** `/home/ollie/Documents/PtychoPINN2/ptycho/config/config.py:249`
4. **Logger Instantiation:** `/home/ollie/Documents/PtychoPINN2/ptycho_torch/workflows/components.py:741-760`
5. **Test Contracts:**
   - `/home/ollie/Documents/PtychoPINN2/tests/torch/test_cli_train_torch.py:621-770`
   - `/home/ollie/Documents/PtychoPINN2/tests/torch/test_workflows_components.py:3318-3348`

---

## Root Cause Summary

### Question 1: Where to add CLI flags?
**Answer:** `/home/ollie/Documents/PtychoPINN2/ptycho_torch/train.py`
- `--logger` flag: After line 476 (after `--learning-rate`)
- `--disable_mlflow` update: Line 420-421 (existing flag, update help text)

### Question 2: Where does _train_with_lightning instantiate Trainer?
**Answer:** `/home/ollie/Documents/PtychoPINN2/ptycho_torch/workflows/components.py:744`
- Logger instantiation: Insert before line 744 (after callbacks construction at line 741)
- Trainer kwarg update: Line 760 (replace hardcoded `logger=False`)

### Question 3: Existing logger-related code?
**Answer:** None in the target files. Existing MLflow usage is in deprecated legacy code paths:
- `ptycho_torch/train.py:74-80` (MLflow imports)
- `ptycho_torch/train.py:306-340` (MLflow autolog in legacy main() function)
- New CLI path (line 604-714) does NOT use MLflow - routes through workflow components

---

**Generated:** 2025-10-23
**Author:** Claude (debugging agent)
**Task:** Logger backend implementation investigation (EB3 Phase B1)
