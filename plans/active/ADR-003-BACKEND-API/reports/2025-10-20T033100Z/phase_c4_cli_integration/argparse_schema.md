# Argparse Schema — ADR-003 Phase C4.A4

**Date:** 2025-10-20
**Purpose:** Complete argparse flag schema for Phase C4 execution config CLI exposure (5 new flags).

---

## Reading This Schema

**Table Columns:**
- **Flag:** CLI argument string(s) (primary + aliases if any)
- **Type:** Python type for argparse (str/int/float/bool/choice)
- **Default:** Default value if flag not specified
- **Dest:** argparse destination variable name
- **Action:** argparse action (store/store_true/store_false/choice)
- **Choices:** Valid values for choice-type flags
- **Help Text:** User-facing help message
- **Validation:** Post-parse validation rules
- **Mutual Exclusivity:** Flags that cannot be combined

---

## Training Execution Config Flags (4 new flags)

### 1. `--accelerator`

```python
parser.add_argument(
    '--accelerator',
    type=str,
    default='auto',
    choices=['auto', 'cpu', 'gpu', 'cuda', 'tpu', 'mps'],
    dest='accelerator',
    help=(
        'Hardware accelerator for training: '
        'auto (auto-detect, default), cpu (CPU-only), gpu (NVIDIA GPU), '
        'cuda (alias for gpu), tpu (Google TPU), mps (Apple Silicon). '
        'Default: auto.'
    )
)
```

| Attribute | Value |
|-----------|-------|
| **Flag** | `--accelerator` |
| **Type** | `str` |
| **Default** | `'auto'` |
| **Dest** | `accelerator` |
| **Action** | `store` |
| **Choices** | `['auto', 'cpu', 'gpu', 'cuda', 'tpu', 'mps']` |
| **Help Text** | "Hardware accelerator for training: auto (auto-detect, default), cpu (CPU-only), gpu (NVIDIA GPU), cuda (alias for gpu), tpu (Google TPU), mps (Apple Silicon). Default: auto." |
| **Validation** | None (argparse choices enforce validity) |
| **Mutual Exclusivity** | Mutually exclusive with legacy `--device` flag (if both specified, `--accelerator` wins with deprecation warning) |

**Implementation Notes:**
- **Backward Compatibility:** If legacy `--device` flag is used, print deprecation warning and map to `--accelerator` (cpu→cpu, cuda→gpu)
- **Lightning Mapping:** Pass directly to `Trainer(accelerator=args.accelerator)`
- **CUDA Alias:** `cuda` is accepted as alias for `gpu` (Lightning uses `gpu`, not `cuda`)
- **Future-Proof:** `tpu` and `mps` are valid choices even if not yet tested (Lightning supports them)

**Example Usage:**
```bash
python -m ptycho_torch.train --accelerator cpu ...     # CPU training
python -m ptycho_torch.train --accelerator gpu ...     # GPU training (auto-detect count)
python -m ptycho_torch.train --accelerator auto ...    # Auto-detect (default)
```

---

### 2. `--deterministic` / `--no-deterministic`

```python
parser.add_argument(
    '--deterministic',
    dest='deterministic',
    action='store_true',
    default=True,
    help=(
        'Enable deterministic training for reproducibility (default: enabled). '
        'Sets torch.use_deterministic_algorithms(True) and Lightning deterministic=True. '
        'Use --no-deterministic to disable for potential performance gains.'
    )
)
parser.add_argument(
    '--no-deterministic',
    dest='deterministic',
    action='store_false',
    help='Disable deterministic training. May improve performance but results are non-reproducible.'
)
```

| Attribute | Value |
|-----------|-------|
| **Flag** | `--deterministic` (enable) / `--no-deterministic` (disable) |
| **Type** | `bool` |
| **Default** | `True` (deterministic enabled by default) |
| **Dest** | `deterministic` |
| **Action** | `store_true` / `store_false` |
| **Choices** | N/A (boolean) |
| **Help Text** | "Enable deterministic training for reproducibility (default: enabled). Sets torch.use_deterministic_algorithms(True) and Lightning deterministic=True. Use --no-deterministic to disable for potential performance gains." |
| **Validation** | None (boolean flag) |
| **Mutual Exclusivity** | None |

**Implementation Notes:**
- **Default Behavior:** Deterministic mode is **enabled by default** to ensure reproducibility (matches current workflow behavior)
- **Opt-Out:** Users can disable with `--no-deterministic` for performance benchmarking
- **Lightning Mapping:** Pass to `Trainer(deterministic=args.deterministic)`
- **Seed Requirement:** Deterministic mode requires `subsample_seed` to be set for data sampling reproducibility (logged warning if missing)

**Example Usage:**
```bash
python -m ptycho_torch.train --deterministic ...         # Explicit enable (redundant with default)
python -m ptycho_torch.train --no-deterministic ...      # Disable for benchmarking
python -m ptycho_torch.train ...                         # Default: deterministic enabled
```

---

### 3. `--num-workers`

```python
parser.add_argument(
    '--num-workers',
    type=int,
    default=0,
    dest='num_workers',
    help=(
        'Number of DataLoader worker processes for parallel data loading (default: 0 = synchronous). '
        'Typical values: 2-8 for multi-core systems. Higher values increase data loading throughput '
        'but consume more memory. Set to 0 for single-threaded loading (safest for CI).'
    )
)
```

| Attribute | Value |
|-----------|-------|
| **Flag** | `--num-workers` |
| **Type** | `int` |
| **Default** | `0` (synchronous DataLoader) |
| **Dest** | `num_workers` |
| **Action** | `store` |
| **Choices** | N/A (integer) |
| **Help Text** | "Number of DataLoader worker processes for parallel data loading (default: 0 = synchronous). Typical values: 2-8 for multi-core systems. Higher values increase data loading throughput but consume more memory. Set to 0 for single-threaded loading (safest for CI)." |
| **Validation** | `if args.num_workers < 0: raise ValueError("--num-workers must be >= 0")` |
| **Mutual Exclusivity** | None |

**Implementation Notes:**
- **Default 0:** Synchronous data loading (safest for reproducibility and CI)
- **Validation:** Negative values are invalid; raise `ValueError` in factory
- **Lightning Mapping:** Pass to `DataLoader(num_workers=args.num_workers)` for both training and inference
- **Performance Warning:** If `num_workers > 0` with `--deterministic`, log warning: "WARNING: num_workers > 0 with deterministic mode may introduce non-determinism on some platforms"
- **Memory Impact:** Higher worker counts increase memory usage (document in help text)

**Example Usage:**
```bash
python -m ptycho_torch.train --num-workers 0 ...     # Single-threaded (default)
python -m ptycho_torch.train --num-workers 4 ...     # 4 parallel workers
python -m ptycho_torch.train --num-workers 8 ...     # 8 parallel workers (high-throughput systems)
```

---

### 4. `--learning-rate`

```python
parser.add_argument(
    '--learning-rate',
    type=float,
    default=1e-3,
    dest='learning_rate',
    help=(
        'Learning rate for Adam optimizer (default: 1e-3). '
        'Typical range: 1e-5 (slow, stable) to 1e-2 (fast, may diverge). '
        'Adjust based on convergence behavior during training.'
    )
)
```

| Attribute | Value |
|-----------|-------|
| **Flag** | `--learning-rate` |
| **Type** | `float` |
| **Default** | `1e-3` (Adam optimizer default) |
| **Dest** | `learning_rate` |
| **Action** | `store` |
| **Choices** | N/A (float) |
| **Help Text** | "Learning rate for Adam optimizer (default: 1e-3). Typical range: 1e-5 (slow, stable) to 1e-2 (fast, may diverge). Adjust based on convergence behavior during training." |
| **Validation** | `if args.learning_rate <= 0: raise ValueError("--learning-rate must be > 0")` |
| **Mutual Exclusivity** | None |

**Implementation Notes:**
- **Current Gap:** Learning rate is **hardcoded** in workflow (components.py:538 as 1e-3); this flag exposes it
- **Validation:** Zero or negative values are invalid; raise `ValueError` in factory
- **Lightning Mapping:** Pass to optimizer constructor: `torch.optim.Adam(params, lr=execution_config.learning_rate)`
- **Guidance:** Help text provides typical range based on Adam optimizer behavior
- **Future Extension:** Phase D may add scheduler flags (`--scheduler`, `--lr-decay-rate`)

**Example Usage:**
```bash
python -m ptycho_torch.train --learning-rate 1e-3 ...   # Default (redundant)
python -m ptycho_torch.train --learning-rate 5e-4 ...   # Conservative learning rate
python -m ptycho_torch.train --learning-rate 1e-2 ...   # Aggressive learning rate
python -m ptycho_torch.train --learning-rate 1e-5 ...   # Very slow, stable convergence
```

---

## Inference Execution Config Flags (1 new flag)

### 5. `--inference-batch-size`

```python
parser.add_argument(
    '--inference-batch-size',
    type=int,
    default=None,
    dest='inference_batch_size',
    help=(
        'Batch size for inference DataLoader (default: None = use training batch_size). '
        'Larger values increase inference throughput on GPU systems. '
        'Typical values: 16-64 for GPU inference, 4-8 for CPU.'
    )
)
```

| Attribute | Value |
|-----------|-------|
| **Flag** | `--inference-batch-size` |
| **Type** | `int` |
| **Default** | `None` (uses `TrainingConfig.batch_size` if None) |
| **Dest** | `inference_batch_size` |
| **Action** | `store` |
| **Choices** | N/A (integer) |
| **Help Text** | "Batch size for inference DataLoader (default: None = use training batch_size). Larger values increase inference throughput on GPU systems. Typical values: 16-64 for GPU inference, 4-8 for CPU." |
| **Validation** | `if args.inference_batch_size is not None and args.inference_batch_size <= 0: raise ValueError("--inference-batch-size must be > 0")` |
| **Mutual Exclusivity** | None |

**Implementation Notes:**
- **Current Gap:** Inference always uses training `batch_size`; this flag allows override
- **Default Behavior:** If not specified, defaults to `TrainingConfig.batch_size` (maintains current behavior)
- **Validation:** Only validate if specified (non-None); positive integer required
- **Lightning Mapping:** Pass to `DataLoader(batch_size=execution_config.inference_batch_size or config.batch_size)` in `_build_inference_dataloader`
- **Use Case:** GPU inference can use larger batches than training (no gradient storage overhead)

**Example Usage:**
```bash
# Use training batch size (default)
python -m ptycho_torch.inference --model_path outputs/ --test_data test.npz ...

# Override with larger batch size for GPU inference
python -m ptycho_torch.inference --inference-batch-size 32 ...

# CPU inference with small batch size
python -m ptycho_torch.inference --inference-batch-size 4 --accelerator cpu ...
```

---

## Shared Flags (Training + Inference)

### `--accelerator` (Shared)

**Usage in Both Scripts:**
- **Training:** `ptycho_torch/train.py` — Controls training device
- **Inference:** `ptycho_torch/inference.py` — Controls inference device

**Implementation:**
- Same argparse definition in both scripts
- Same destination field: `execution_config.accelerator`

---

### `--num-workers` (Shared)

**Usage in Both Scripts:**
- **Training:** Controls training DataLoader parallelism
- **Inference:** Controls inference DataLoader parallelism

**Implementation:**
- Same argparse definition in both scripts
- Same destination field: `execution_config.num_workers`

---

## Validation Rules (Factory Layer)

### Post-Parse Validation Logic

```python
def validate_execution_config(args):
    """
    Factory-layer validation for execution config CLI arguments.

    Raises:
        ValueError: If validation fails.
    """
    # Validate num_workers range
    if args.num_workers < 0:
        raise ValueError(f"--num-workers must be >= 0, got {args.num_workers}")

    # Validate learning_rate range
    if hasattr(args, 'learning_rate') and args.learning_rate <= 0:
        raise ValueError(f"--learning-rate must be > 0, got {args.learning_rate}")

    # Validate inference_batch_size range (if specified)
    if hasattr(args, 'inference_batch_size') and args.inference_batch_size is not None:
        if args.inference_batch_size <= 0:
            raise ValueError(f"--inference-batch-size must be > 0, got {args.inference_batch_size}")

    # Warn about num_workers + deterministic combination
    if args.num_workers > 0 and getattr(args, 'deterministic', True):
        import logging
        logging.warning(
            "num_workers > 0 with deterministic mode enabled. "
            "This may introduce non-determinism on some platforms. "
            "Consider using --num-workers 0 for strict reproducibility."
        )
```

**Validation Strategy:**
- **Argparse Layer:** Type validation, choices validation
- **Factory Layer:** Range validation, cross-flag warnings

---

## Mutual Exclusivity Rules

### Backward Compatibility: `--device` vs `--accelerator`

**Rule:** If both `--device` (legacy) and `--accelerator` (new) are specified:
1. Prefer `--accelerator` value
2. Print deprecation warning:
   ```
   DeprecationWarning: --device is deprecated and will be removed in Phase D.
   Use --accelerator instead. Ignoring --device value and using --accelerator.
   ```

**Implementation:**
```python
if args.device and args.accelerator != 'auto':
    import warnings
    warnings.warn(
        "--device is deprecated and will be removed in Phase D. "
        "Use --accelerator instead. Ignoring --device value.",
        DeprecationWarning
    )
elif args.device:
    # Map legacy --device to --accelerator
    args.accelerator = 'cpu' if args.device == 'cpu' else 'gpu'
```

---

## CLI Help Text Examples

### Training Help Output (Post-C4)

```bash
$ python -m ptycho_torch.train --help

usage: train.py [-h] [--train_data_file TRAIN_DATA_FILE] [--output_dir OUTPUT_DIR]
                [--max_epochs MAX_EPOCHS] [--n_images N_IMAGES] [--batch_size BATCH_SIZE]
                [--accelerator {auto,cpu,gpu,cuda,tpu,mps}] [--deterministic] [--no-deterministic]
                [--num-workers NUM_WORKERS] [--learning-rate LEARNING_RATE]

PyTorch Lightning training for ptychographic reconstruction

optional arguments:
  --train_data_file TRAIN_DATA_FILE
                        Path to training NPZ dataset (required)
  --output_dir OUTPUT_DIR
                        Directory for checkpoint outputs (required)
  --max_epochs MAX_EPOCHS
                        Maximum training epochs (default: 100)
  --n_images N_IMAGES   Number of diffraction groups (default: 512)
  --batch_size BATCH_SIZE
                        Training batch size (default: 4)
  --accelerator {auto,cpu,gpu,cuda,tpu,mps}
                        Hardware accelerator: auto (detect), cpu, gpu, cuda, tpu, or mps (default: auto)
  --deterministic       Enable deterministic training for reproducibility (default: enabled)
  --no-deterministic    Disable deterministic training for performance
  --num-workers NUM_WORKERS
                        Number of DataLoader worker processes (default: 0)
  --learning-rate LEARNING_RATE
                        Learning rate for optimizer (default: 1e-3, range: 1e-5 to 1e-2)
```

### Inference Help Output (Post-C4)

```bash
$ python -m ptycho_torch.inference --help

usage: inference.py [-h] --model_path MODEL_PATH --test_data TEST_DATA --output_dir OUTPUT_DIR
                    [--n_images N_IMAGES] [--accelerator {auto,cpu,gpu,cuda,tpu,mps}]
                    [--num-workers NUM_WORKERS] [--inference-batch-size INFERENCE_BATCH_SIZE]

PyTorch Lightning checkpoint inference

optional arguments:
  --model_path MODEL_PATH
                        Path to training output directory (required)
  --test_data TEST_DATA Path to test NPZ file (required)
  --output_dir OUTPUT_DIR
                        Directory for reconstruction outputs (required)
  --n_images N_IMAGES   Number of images for reconstruction (default: 32)
  --accelerator {auto,cpu,gpu,cuda,tpu,mps}
                        Hardware accelerator (default: auto)
  --num-workers NUM_WORKERS
                        DataLoader worker processes (default: 0)
  --inference-batch-size INFERENCE_BATCH_SIZE
                        Inference batch size (default: None = use training batch_size)
```

---

## Factory Integration Points

### Training Factory Call Pattern (Post-C4)

```python
# In ptycho_torch/train.py cli_main() (after argparse)
from ptycho_torch.config_factory import create_training_payload

# Validate execution config args (factory layer)
validate_execution_config(args)

# Create execution config from CLI args
execution_config = PyTorchExecutionConfig(
    accelerator=args.accelerator,
    deterministic=args.deterministic,
    num_workers=args.num_workers,
    learning_rate=args.learning_rate,
    enable_progress_bar=(not args.quiet if hasattr(args, 'quiet') else True),
)

# Call factory with execution config
payload = create_training_payload(
    train_data_file=args.train_data_file,
    output_dir=args.output_dir,
    overrides=dict(
        n_groups=args.n_images,
        test_data_file=args.test_data_file,
    ),
    execution_config=execution_config,
)

# Extract configs and workflow
tf_training_config = payload.tf_training_config
pt_configs = (payload.data_config, payload.pt_model_config, payload.pt_training_config, ...)

# Call workflow
run_cdi_example_torch(..., execution_config=payload.execution_config)
```

### Inference Factory Call Pattern (Post-C4)

```python
# In ptycho_torch/inference.py cli_main() (after argparse)
from ptycho_torch.config_factory import create_inference_payload

# Validate execution config args
validate_execution_config(args)

# Create execution config from CLI args
execution_config = PyTorchExecutionConfig(
    accelerator=args.accelerator,
    num_workers=args.num_workers,
    inference_batch_size=args.inference_batch_size,
)

# Call factory with execution config
payload = create_inference_payload(
    model_path=args.model_path,
    test_data_file=args.test_data,
    output_dir=args.output_dir,
    overrides=dict(
        n_groups=args.n_images,
    ),
    execution_config=execution_config,
)

# Extract configs and workflow
# ... (similar to training)
```

---

## TODO Markers for Phase C4.C Implementation

**CLI Updates Checklist:**
- [ ] Training CLI: Add 4 argparse arguments (--accelerator, --deterministic, --num-workers, --learning-rate) at `train.py:~370-380`
- [ ] Inference CLI: Add 3 argparse arguments (--accelerator, --num-workers, --inference-batch-size) at `inference.py:~330-340`
- [ ] Training CLI: Instantiate `PyTorchExecutionConfig` from parsed args before factory call
- [ ] Inference CLI: Instantiate `PyTorchExecutionConfig` from parsed args before factory call
- [ ] Both CLIs: Call `validate_execution_config(args)` after argparse, before factory
- [ ] Both CLIs: Add deprecation warning logic for `--device` → `--accelerator` migration
- [ ] Workflow docs: Update `docs/workflows/pytorch.md` §13 with new flag examples
- [ ] Spec docs: Add CLI reference table to `specs/ptychodus_api_spec.md` §7 (new section)

---

## References

**Factory Design:**
- `plans/active/ADR-003-BACKEND-API/reports/2025-10-19T232336Z/phase_b_factories/factory_design.md` §3 (Call Patterns)

**Execution Config Definition:**
- `ptycho/config/config.py:72-90` (PyTorchExecutionConfig dataclass)

**Phase C3 Workflow Wiring:**
- `plans/active/ADR-003-BACKEND-API/reports/2025-10-20T025643Z/phase_c3_workflow_integration/summary.md` (Trainer/DataLoader kwargs)

**Current CLI Definitions:**
- Training: `ptycho_torch/train.py:366-404`
- Inference: `ptycho_torch/inference.py:319-379`

**Lightning Trainer API:**
- https://lightning.ai/docs/pytorch/stable/common/trainer.html (accelerator, deterministic, devices)

**PyTorch DataLoader API:**
- https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader (num_workers, batch_size)

---

**Schema Complete:** 5 execution config flags documented with full argparse specs, validation rules, help text, and integration guidance.
