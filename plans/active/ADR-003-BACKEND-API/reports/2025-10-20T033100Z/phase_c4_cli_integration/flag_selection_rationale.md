# CLI Flag Selection Rationale — ADR-003 Phase C4.A2

**Date:** 2025-10-20
**Purpose:** Justify selection of 5 high-priority execution config flags for Phase C4 CLI exposure.

---

## Selection Criteria

Flags were prioritized based on:

1. **User Impact:** How often users need to configure this parameter
2. **Runtime Safety:** Critical for reproducibility or hardware compatibility
3. **Parity with TensorFlow:** Aligns PyTorch CLI with existing TF workflows
4. **Implementation Complexity:** Low-complexity argparse-to-dataclass mapping
5. **Deferred Dependencies:** Flags requiring callback wiring or governance decisions are deferred to Phase D

---

## Selected Flags (5 Total)

### 1. `--accelerator` (Training + Inference)

**Target Field:** `PyTorchExecutionConfig.accelerator`
**Type:** `str` (choice: auto/cpu/gpu/cuda/tpu/mps)
**Default:** `'auto'`
**Priority:** **HIGH**

**Rationale:**
- **Hardware Compatibility:** Essential for users to select compute device (CPU for CI, GPU for production)
- **Lightning Requirement:** PyTorch Lightning's `Trainer(accelerator=...)` is a core parameter
- **POLICY-001 Alignment:** Explicit device selection prevents silent CUDA unavailability issues
- **User Impact:** Every deployment scenario requires device configuration (CI=cpu, local dev=auto, cluster=gpu)
- **Current Gap:** Training uses `--device` choice (cpu/cuda), but this doesn't expose Lightning's full accelerator surface (tpu/mps)

**Implementation Notes:**
- Current `--device` flag (train.py:394) converts to `n_devices` (int), not `accelerator` (str)
- Replace `--device` with `--accelerator` to match Lightning API
- Maintain backward compatibility by accepting both flags (deprecation warning for `--device`)

**Override Precedence:**
- Level 2 (Execution Config) — overrides CLI defaults, defers to explicit overrides dict

---

### 2. `--deterministic` (Training)

**Target Field:** `PyTorchExecutionConfig.deterministic`
**Type:** `bool` (action='store_true')
**Default:** `True` (enabled by default for reproducibility)
**Priority:** **HIGH**

**Rationale:**
- **Reproducibility:** Critical for scientific workflows and regression testing
- **Lightning Behavior:** `Trainer(deterministic=True)` enables `torch.use_deterministic_algorithms()`
- **Parity with Specs:** `docs/workflows/pytorch.md` §6 "Checkpoint Management and Reproducibility" mandates deterministic behavior
- **Current State:** Deterministic mode is **hardcoded** in workflows (components.py:571), not user-configurable
- **User Impact:** Researchers need to toggle determinism off for performance benchmarks

**Implementation Notes:**
- Default `True` to preserve current reproducible behavior
- Use `--deterministic` to enable (default), `--no-deterministic` to disable via argparse `action='store_true'` / `action='store_false'`
- Document that disabling determinism may yield faster training but non-reproducible results

**Override Precedence:**
- Level 2 (Execution Config) — overrides CLI defaults

---

### 3. `--num-workers` (Training + Inference)

**Target Field:** `PyTorchExecutionConfig.num_workers`
**Type:** `int`
**Default:** `0` (synchronous DataLoader)
**Priority:** **HIGH**

**Rationale:**
- **Performance Knob:** DataLoader parallelism can significantly speed up data loading (2-4x for I/O-bound datasets)
- **Cluster Compatibility:** Users on multi-core systems need to configure worker count
- **Current Gap:** Hardcoded to `0` in workflow (components.py:361); no CLI exposure
- **Lightning DataLoader:** Shared parameter across training and inference dataloaders
- **User Impact:** Default 0 (single-threaded) is safe for CI but suboptimal for production

**Implementation Notes:**
- Validate `num_workers >= 0` (negative values are invalid)
- Document that `num_workers > 0` requires `persistent_workers=True` for best performance (deferred to Phase D)
- Add warning if `num_workers > 0` with `deterministic=True` (may introduce non-determinism on some platforms)

**Override Precedence:**
- Level 2 (Execution Config) — applies to both training and inference dataloaders

---

### 4. `--learning-rate` (Training Only)

**Target Field:** `PyTorchExecutionConfig.learning_rate`
**Type:** `float`
**Default:** `1e-3` (Adam optimizer default)
**Priority:** **HIGH**

**Rationale:**
- **Hyperparameter Tuning:** Most critical training hyperparameter for convergence
- **Current Gap:** Hardcoded in workflow (components.py:538); **missing CLI flag** per `override_matrix.md` §7
- **TensorFlow Parity:** TensorFlow workflows allow learning rate configuration
- **User Impact:** Users must modify code to tune learning rate; unacceptable for standard ML workflow
- **Scientific Workflows:** Hyperparameter sweeps require CLI exposure

**Implementation Notes:**
- Validate `learning_rate > 0` (zero or negative values are invalid)
- Document range recommendations: typical range `[1e-5, 1e-2]` for Adam optimizer
- Add to optimizer configuration in `_train_with_lightning` (line 538 replacement)

**Override Precedence:**
- Level 2 (Execution Config) — overrides hardcoded default 1e-3

---

### 5. `--inference-batch-size` (Inference Only)

**Target Field:** `PyTorchExecutionConfig.inference_batch_size`
**Type:** `int`
**Default:** `None` (uses `TrainingConfig.batch_size` if None)
**Priority:** **MEDIUM** (included for workflow flexibility)

**Rationale:**
- **Memory Optimization:** Inference can use larger batch sizes than training (no gradient storage)
- **Current Gap:** Inference always uses training `batch_size`; cannot override
- **User Impact:** Users on GPU clusters can increase inference throughput by 2-4x with larger batches
- **Workflow Flexibility:** Decouples training and inference batch size configuration
- **Implementation Ease:** Phase C3 already wired `_build_inference_dataloader` to accept execution config (components.py:460-467)

**Implementation Notes:**
- If `None`, default to `config.batch_size` (maintains current behavior)
- If specified, override DataLoader batch_size for inference only
- Validate `inference_batch_size > 0`

**Override Precedence:**
- Level 2 (Execution Config) — overrides training batch_size for inference workflows

---

## Deferred Flags (Phase D)

The following execution config knobs are **intentionally out of scope** for Phase C4 due to higher implementation complexity:

### Checkpoint Management Flags (3 deferred)

| Flag | Deferred Reason |
|------|----------------|
| `--checkpoint-save-top-k` | Requires Lightning `ModelCheckpoint` callback configuration; not trivial argparse mapping |
| `--checkpoint-monitor-metric` | Requires validation against available metrics (val_loss, train_loss, custom metrics) |
| `--early-stop-patience` | Currently hardcoded to 100 in legacy code; requires `EarlyStopping` callback wiring |

**Rationale for Deferral:**
- Checkpoint flags require instantiating Lightning callbacks, not simple Trainer kwargs
- Governance decision needed on default checkpoint strategy (save all vs top-K)
- Early stopping requires metric monitoring setup (deferred until callback architecture finalized)

---

### Logger Backend Flags (1 deferred)

| Flag | Deferred Reason |
|------|----------------|
| `--logger-backend` | MLflow vs TensorBoard governance decision pending; current `--disable-mlflow` is inverse logic |

**Rationale for Deferral:**
- Legacy API uses MLflow autologging (enabled by default, disabled via `--disable-mlflow`)
- New workflow has no logger integration yet (TensorBoard vs MLflow vs WandB decision pending)
- Requires ADR or governance decision on default logger strategy

---

### Advanced Training Flags (2 deferred)

| Flag | Deferred Reason |
|------|----------------|
| `--scheduler` | LR scheduler selection (StepLR/ReduceLROnPlateau/CosineAnnealing) requires factory method design |
| `--gradient-clip-val` | Gradient clipping parameter; not yet implemented in workflows |

**Rationale for Deferral:**
- Scheduler selection requires mapping flag values to PyTorch scheduler classes
- Design decision needed on scheduler defaults and configuration surface
- Gradient clipping requires understanding of typical value ranges for this domain

---

### DataLoader Performance Flags (2 deferred)

| Flag | Deferred Reason |
|------|----------------|
| `--prefetch-factor` | DataLoader performance knob; not yet critical for current workflows |
| `--persistent-workers` | Requires `num_workers > 0`; dependent on Phase C4 `--num-workers` adoption |

**Rationale for Deferral:**
- Performance optimization flags are secondary to core functionality
- Persistent workers only apply when `num_workers > 0` (which is currently hardcoded to 0)
- Phase D can batch performance knobs together after `--num-workers` adoption

---

## Naming Harmonization with TensorFlow

### Consistent Naming Choices

| PyTorch Flag | TensorFlow Equivalent | Decision |
|--------------|----------------------|----------|
| `--accelerator` | (none) | **PT-specific:** Unique to Lightning; no TF equivalent |
| `--deterministic` | (none) | **PT-specific:** Matches Lightning API naming |
| `--num-workers` | (none) | **PT-specific:** DataLoader parallelism; TF uses different I/O model |
| `--learning-rate` | `--learning_rate` (if exists) | **Harmonized:** Use snake_case `--learning-rate` to match kebab-case convention |
| `--inference-batch-size` | (none) | **PT-specific:** TF workflows don't separate training/inference batch sizes |

**Convention Decision:**
- Use `--kebab-case` for all flags (matches existing `--train_data_file`, `--output_dir`)
- Underscores in destination fields (e.g., `learning_rate` in dataclass)
- Help text uses descriptive names ("Learning rate for optimizer")

---

## Override Precedence Clarification

Per `factory_design.md` §5, the override precedence is:

1. **Explicit Overrides** (user-provided dict parameter to factory)
2. **Execution Config** (PyTorchExecutionConfig fields) ← **Phase C4 FLAGS HERE**
3. **CLI Defaults** (argparse default values)
4. **PyTorch Config Defaults** (DataConfig, ModelConfig, TrainingConfig)
5. **TensorFlow Config Defaults** (canonical dataclasses)

**Critical Note:**
- Execution config flags apply at **Level 2** (between explicit overrides and CLI defaults)
- This means:
  - `create_training_payload(overrides={'learning_rate': 5e-4})` **beats** `--learning-rate 1e-3`
  - `--learning-rate 1e-3` **beats** hardcoded default `1e-3`
  - Execution config fields do NOT override canonical config fields (N, gridsize, batch_size)

**Audit Trail:**
- Factory functions log all execution config overrides to `payload.overrides_applied` dict
- Enables debugging precedence conflicts

---

## Validation Rules

| Flag | Validation Type | Error Condition | Action |
|------|----------------|-----------------|--------|
| `--accelerator` | Choice validation | Not in {auto, cpu, gpu, cuda, tpu, mps} | Raise `argparse.ArgumentError` |
| `--deterministic` | Boolean flag | (none) | N/A (store_true/store_false) |
| `--num-workers` | Range validation | `num_workers < 0` | Raise `ValueError("num_workers must be >= 0")` |
| `--learning-rate` | Range validation | `learning_rate <= 0` | Raise `ValueError("learning_rate must be > 0")` |
| `--inference-batch-size` | Optional range | `inference_batch_size <= 0` (if specified) | Raise `ValueError("inference_batch_size must be > 0")` |

**Factory Responsibility:**
- Factories validate execution config fields before constructing payloads
- Validation errors raised immediately (fail-fast principle)
- CLI argparse provides first layer of validation (type, choices)
- Factory provides second layer (range checks, cross-field constraints)

---

## Example CLI Usage (Post-C4)

### Training with Execution Config Flags

```bash
# CPU training with custom learning rate
python -m ptycho_torch.train \
  --train_data_file datasets/train.npz \
  --output_dir outputs/ \
  --n_images 512 \
  --max_epochs 10 \
  --accelerator cpu \
  --deterministic \
  --num-workers 4 \
  --learning-rate 5e-4

# GPU training with non-deterministic mode for speed
python -m ptycho_torch.train \
  --train_data_file datasets/train.npz \
  --output_dir outputs/ \
  --n_images 512 \
  --max_epochs 50 \
  --accelerator gpu \
  --no-deterministic \
  --num-workers 8 \
  --learning-rate 1e-3
```

### Inference with Execution Config Flags

```bash
# High-throughput inference with large batch size
python -m ptycho_torch.inference \
  --model_path training_outputs/ \
  --test_data datasets/test.npz \
  --output_dir inference_outputs/ \
  --n_images 128 \
  --accelerator gpu \
  --num-workers 4 \
  --inference-batch-size 32
```

---

## References

**Override Matrix:**
- `plans/active/ADR-003-BACKEND-API/reports/2025-10-19T232336Z/phase_b_factories/override_matrix.md` §5 (Execution Config Fields)

**Factory Design:**
- `plans/active/ADR-003-BACKEND-API/reports/2025-10-19T232336Z/phase_b_factories/factory_design.md` §5 (Override Precedence)

**Workflow Guide:**
- `docs/workflows/pytorch.md` §6 (Checkpoint Management and Reproducibility)

**Execution Config Definition:**
- `ptycho/config/config.py:72-90` (PyTorchExecutionConfig dataclass with 22 fields)

**Phase C3 Workflow Wiring:**
- `plans/active/ADR-003-BACKEND-API/reports/2025-10-20T025643Z/phase_c3_workflow_integration/summary.md` (Trainer/DataLoader kwargs wiring)

---

**Summary:**
- **5 flags selected:** 3 training (accelerator, deterministic, num-workers, learning-rate), 2 shared (accelerator, num-workers) + 1 inference (inference-batch-size)
- **Selection criteria:** High user impact, low complexity, runtime safety, TensorFlow parity where applicable
- **Deferred:** 10 execution config knobs requiring callback wiring or governance decisions
- **Naming:** Consistent `--kebab-case` CLI convention, underscores in dataclass fields
- **Validation:** Two-layer validation (argparse + factory range checks)
- **Override precedence:** Level 2 (Execution Config) beats CLI defaults, defers to explicit overrides
