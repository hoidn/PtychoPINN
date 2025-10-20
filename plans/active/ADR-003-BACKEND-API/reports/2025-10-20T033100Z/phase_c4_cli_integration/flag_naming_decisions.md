# CLI Flag Naming Decisions — ADR-003 Phase C4.A3

**Date:** 2025-10-20
**Purpose:** Document naming conventions and harmonization decisions for PyTorch CLI flags with TensorFlow equivalents.

---

## Naming Convention Principles

### 1. Kebab-Case for CLI Flags

**Decision:** All CLI flags use `--kebab-case` (lowercase with hyphens)

**Rationale:**
- **Consistency:** Matches existing PyTorch CLI flags (`--train_data_file`, `--output_dir`, `--max_epochs`)
- **POSIX Convention:** Standard Unix/Linux flag naming (e.g., `gcc --output-file`, `git --no-pager`)
- **Readability:** Hyphens are easier to read than underscores in shell context

**Examples:**
```bash
--learning-rate      # Chosen
--learning_rate      # Rejected (inconsistent with existing --train_data_file)
--learningRate       # Rejected (camelCase not idiomatic for CLI)
```

**Exception:**
- Existing flags with underscores (`--train_data_file`, `--test_data_file`, `--output_dir`) are **retained for backward compatibility**
- New flags introduced in Phase C4 use hyphens

---

### 2. Dataclass Field Names: Snake_Case

**Decision:** Python dataclass field names use `snake_case` (lowercase with underscores)

**Rationale:**
- **Python Convention:** PEP 8 recommends snake_case for variable and attribute names
- **TensorFlow Parity:** Canonical TensorFlow configs use snake_case (`learning_rate`, `num_workers`, `batch_size`)
- **Automatic Mapping:** Argparse converts `--learning-rate` → `args.learning_rate` (replaces hyphens with underscores)

**Examples:**
```python
# PyTorchExecutionConfig dataclass
learning_rate: float = 1e-3        # Chosen
num_workers: int = 0                # Chosen
inference_batch_size: Optional[int] = None  # Chosen
```

---

### 3. TensorFlow Naming Harmonization

**Decision Matrix:**

| Concept | PyTorch CLI Flag | PyTorch Dataclass Field | TensorFlow Equivalent | Harmonization Decision |
|---------|-----------------|------------------------|----------------------|----------------------|
| **Training Epochs** | `--max_epochs` | `TrainingConfig.epochs` | `TrainingConfig.nepochs` | **Divergence Accepted:** PT uses "epochs" (Lightning convention), TF uses "nepochs" (legacy). Bridge handles conversion. |
| **Data Grouping Count** | `--n_images` | (bridge override) `n_groups` | `TrainingConfig.n_groups` | **CLI Rename Recommended:** Change PT flag to `--n_groups` in future for parity; currently `--n_images` maps to `n_groups` via bridge. |
| **Test Dataset** | `--test_data` | (future) `test_data_file` | `TrainingConfig.test_data_file` | **CLI Rename Recommended:** Change PT flag to `--test_data_file` for consistency; currently `--test_data` maps to `test_data_file`. |
| **Activation Function** | (not exposed) | `ModelConfig.activate` | `ModelConfig.amp_activation` | **Bridge Handled:** `activate` (PT enum name) → `amp_activation` (TF field name). No CLI flag. |
| **Grid Size** | `--gridsize` (int) | `DataConfig.grid_size` (tuple) | `ModelConfig.gridsize` (int) | **Type Conversion:** Bridge converts int → (gridsize, gridsize) tuple for PT, then back to int for TF. |
| **Device Selection** | `--device` (cpu/cuda) | `TrainingConfig.n_devices` (int) | (none) | **PT-Specific:** TF doesn't expose device selection. PT converts choice → int (1 for cpu, cuda.device_count() for cuda). |
| **Debug Mode** | `--quiet` (bool) | `InferenceConfig.debug` (inverted) | `InferenceConfig.debug` (bool) | **Logic Inversion:** `--quiet` → `debug=False`; TF uses `--debug` (positive logic). Consider adding `--debug` flag for parity. |

---

## Execution Config Flag Naming (Phase C4)

### Training Execution Config Flags

| Flag | Dataclass Field | TF Equivalent | Naming Rationale |
|------|----------------|---------------|-----------------|
| `--accelerator` | `execution_config.accelerator` | (none) | **PyTorch-Specific:** Matches Lightning API (`Trainer(accelerator=...)`) |
| `--deterministic` | `execution_config.deterministic` | (none) | **PyTorch-Specific:** Matches Lightning API (`Trainer(deterministic=...)`) |
| `--num-workers` | `execution_config.num_workers` | (none) | **PyTorch-Specific:** Matches PyTorch DataLoader API (`num_workers` kwarg). Use hyphen (kebab-case) for CLI. |
| `--learning-rate` | `execution_config.learning_rate` | `--learning_rate` (if exists) | **Harmonized:** TF likely uses `learning_rate` if exposed. Use hyphen for CLI consistency. |
| `--inference-batch-size` | `execution_config.inference_batch_size` | (none) | **PyTorch-Specific:** Decouples training/inference batch sizes. Use hyphens for readability. |

**Key Decision:**
- PyTorch-specific execution config flags (accelerator, deterministic, num_workers) have **no TensorFlow equivalents** because TF uses different execution models (Keras Fit vs Lightning Trainer)
- Use Lightning/PyTorch API naming conventions directly for these flags

---

### Inference Execution Config Flags

| Flag | Dataclass Field | TF Equivalent | Naming Rationale |
|------|----------------|---------------|-----------------|
| `--accelerator` | `execution_config.accelerator` | (none) | **Shared with Training:** Same flag name for consistency |
| `--num-workers` | `execution_config.num_workers` | (none) | **Shared with Training:** Same flag name for consistency |
| `--inference-batch-size` | `execution_config.inference_batch_size` | (none) | **Inference-Specific:** Distinguishes from training `--batch_size` |

---

## Rejected Naming Alternatives

### Alternative 1: Match TensorFlow Exactly

**Rejected Approach:** Use `--nepochs` instead of `--max_epochs`

**Rejection Rationale:**
- Lightning convention is `max_epochs` (canonical PyTorch naming)
- Breaking from Lightning API would confuse PyTorch users
- Bridge can handle `epochs` → `nepochs` conversion transparently
- **Decision:** Accept divergence; bridge handles translation

---

### Alternative 2: Camel Case Flags

**Rejected Approach:** Use `--learningRate`, `--numWorkers`

**Rejection Rationale:**
- Violates POSIX CLI conventions
- Inconsistent with existing codebase (`--train_data_file`, not `--trainDataFile`)
- Less readable in shell context
- **Decision:** Use kebab-case (`--learning-rate`)

---

### Alternative 3: Underscore Flags for Execution Config

**Rejected Approach:** Use `--num_workers`, `--learning_rate`

**Rejection Rationale:**
- Existing PyTorch CLI has **mixed convention** (`--train_data_file` with underscores, `--max_epochs` implied hyphen-friendly)
- New flags should use hyphens to gradually migrate to consistent kebab-case
- **Decision:** New Phase C4 flags use hyphens; legacy flags unchanged

---

## Boolean Flag Patterns

### Pattern: `--flag` (enable) vs `--no-flag` (disable)

**Example:** `--deterministic` / `--no-deterministic`

**Implementation:**
```python
parser.add_argument('--deterministic', dest='deterministic', action='store_true',
                   default=True, help='Enable deterministic training (default: enabled)')
parser.add_argument('--no-deterministic', dest='deterministic', action='store_false',
                   help='Disable deterministic training for performance')
```

**Rationale:**
- **Default Enabled:** Deterministic mode is default for reproducibility
- **Explicit Disable:** Users can opt-out with `--no-deterministic` for benchmarking
- **Argparse Pattern:** Standard Python argparse boolean flag idiom

**Rejected Alternative:**
- `--deterministic` (action='store_true', default=False) — would require users to always specify flag for reproducibility

---

## Special Case: Device/Accelerator Transition

### Current State (Before C4)

```bash
--device cpu         # Current flag
--device cuda        # Current flag
```

Maps to:
```python
TrainingConfig.n_devices = 1 if args.device == 'cpu' else torch.cuda.device_count()
```

### Proposed State (After C4)

```bash
--accelerator cpu    # New flag (replaces --device)
--accelerator gpu    # New flag
--accelerator cuda   # Alias for gpu
--accelerator auto   # Auto-detect (default)
--accelerator tpu    # TPU support (future)
--accelerator mps    # Apple Silicon (future)
```

Maps to:
```python
PyTorchExecutionConfig.accelerator = args.accelerator  # Passed directly to Lightning Trainer
```

**Migration Strategy:**
1. **Phase C4:** Add `--accelerator` flag
2. **Deprecation Warning:** If `--device` used, print warning: "WARNING: --device is deprecated, use --accelerator instead"
3. **Backward Compatibility:** Accept both flags; `--accelerator` takes precedence if both specified
4. **Phase D/E:** Remove `--device` flag entirely

**Naming Decision:**
- **Chosen:** `--accelerator` (matches Lightning API directly)
- **Rejected:** `--device` (too PyTorch-specific, doesn't expose TPU/MPS options)

---

## Help Text Conventions

### Template for Execution Config Flags

```python
parser.add_argument('--learning-rate', type=float, default=1e-3,
                   help='Learning rate for optimizer (default: 1e-3, typical range: 1e-5 to 1e-2)')
```

**Help Text Components:**
1. **Purpose:** "Learning rate for optimizer"
2. **Default:** "(default: 1e-3, ...)"
3. **Guidance:** "typical range: 1e-5 to 1e-2" (for numeric flags)
4. **Choices:** Listed explicitly for choice flags (e.g., accelerator: auto/cpu/gpu/cuda/tpu/mps)

**Examples:**
```python
parser.add_argument('--accelerator', type=str, default='auto',
                   choices=['auto', 'cpu', 'gpu', 'cuda', 'tpu', 'mps'],
                   help='Hardware accelerator: auto (detect), cpu, gpu, cuda, tpu, or mps (default: auto)')

parser.add_argument('--num-workers', type=int, default=0,
                   help='Number of DataLoader worker processes (default: 0 = synchronous, typical: 2-8 for multi-core systems)')

parser.add_argument('--inference-batch-size', type=int, default=None,
                   help='Batch size for inference DataLoader (default: None = use training batch_size). Larger values increase throughput.')
```

---

## Cross-Reference: TensorFlow CLI Naming

**Note:** TensorFlow workflows currently expose CLI flags via:
- `scripts/training/train.py` (TensorFlow training CLI)
- `ptycho/cli_args.py` (if it exists; not found in Phase A inventory)

**TensorFlow CLI Gap:**
- TensorFlow CLI flags are **not centrally documented** in existing codebase
- Most TF workflows use YAML config files instead of CLI flags
- PyTorch CLI is **more CLI-first** than TF workflows

**Naming Precedent (from existing TF training script):**
```python
# Assumed TF naming (not verified; TF uses YAML configs primarily)
--train_data_file    # Matches PT
--test_data_file     # Matches PT (but PT uses --test_data currently)
--output_dir         # Matches PT
--nepochs            # Diverges from PT --max_epochs
--n_groups           # Diverges from PT --n_images
--batch_size         # Matches PT
```

**Harmonization Recommendation (Phase D):**
- Add `--n_groups` to PyTorch CLI (keep `--n_images` as deprecated alias)
- Add `--test_data_file` to PyTorch CLI (keep `--test_data` as alias)
- Document divergences in `docs/workflows/pytorch.md` §13

---

## Summary Table: Final Naming Choices

| Concept | PyTorch CLI Flag (Phase C4) | Dataclass Field | TF Equivalent | Notes |
|---------|---------------------------|----------------|---------------|-------|
| Hardware Accelerator | `--accelerator` | `execution_config.accelerator` | (none) | Replaces `--device` |
| Deterministic Mode | `--deterministic` / `--no-deterministic` | `execution_config.deterministic` | (none) | Boolean flag pattern |
| DataLoader Workers | `--num-workers` | `execution_config.num_workers` | (none) | Hyphenated CLI |
| Optimizer Learning Rate | `--learning-rate` | `execution_config.learning_rate` | `--learning_rate` (assumed) | Hyphenated CLI |
| Inference Batch Size | `--inference-batch-size` | `execution_config.inference_batch_size` | (none) | Inference-specific |

**Convention Summary:**
- **CLI Flags:** `--kebab-case` (new flags), `--snake_case` (legacy flags retained)
- **Dataclass Fields:** `snake_case` (Python convention)
- **Help Text:** Purpose + Default + Guidance/Choices
- **Boolean Flags:** `--flag` (enable, default) + `--no-flag` (disable)
- **TF Parity:** Accept divergences where Lightning API differs from TF Keras API

---

## References

**Existing PyTorch CLI:**
- Training: `ptycho_torch/train.py:366-404`
- Inference: `ptycho_torch/inference.py:319-379`

**Lightning API Documentation:**
- Trainer args: https://lightning.ai/docs/pytorch/stable/common/trainer.html
- DataLoader args: https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader

**Factory Design:**
- `plans/active/ADR-003-BACKEND-API/reports/2025-10-19T232336Z/phase_b_factories/factory_design.md`

**Override Matrix:**
- `plans/active/ADR-003-BACKEND-API/reports/2025-10-19T232336Z/phase_b_factories/override_matrix.md` (naming divergences documented)

**PEP 8 Style Guide:**
- https://peps.python.org/pep-0008/ (Python naming conventions)

---

**Decision Authority:** ADR-003 Phase C4 implementation; aligns with Lightning API conventions and PEP 8 Python standards.
