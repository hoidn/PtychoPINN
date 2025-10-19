# ADR-003 Phase B1 — Factory Design Blueprint

**Date:** 2025-10-19
**Initiative:** ADR-003-BACKEND-API
**Phase:** B1 (Factory Design)
**Status:** Design documentation (no code changes)

---

## 1. Executive Summary

This document defines the architecture for centralized configuration factory functions that will eliminate duplicated config construction logic currently scattered across CLI and workflow entry points. The factories will serve as the single source of truth for translating canonical TensorFlow configurations plus PyTorch execution overrides into the objects consumed by the PyTorch backend.

**Key Design Principles:**
1. **Single Responsibility:** Each factory function handles one workflow (training vs inference)
2. **Bridge Delegation:** All TensorFlow dataclass translation delegated to existing `config_bridge.py` adapters
3. **CONFIG-001 Compliance:** Factories ensure `update_legacy_dict()` is called before data loading
4. **Override Transparency:** Explicit override dict parameter for all execution-specific knobs
5. **Test-Driven:** RED tests written before implementation (Phase B2)

---

## 2. Proposed Module Structure

### 2.1 Primary Module: `ptycho_torch/config_factory.py`

**Location:** `ptycho_torch/config_factory.py` (new file)
**Purpose:** Central factory functions for config object creation
**Dependencies:**
- `ptycho_torch.config_bridge` (existing translation adapters)
- `ptycho.config.config` (TensorFlow canonical dataclasses)
- `ptycho_torch.config_params` (PyTorch singleton configs)
- `ptycho.params` (legacy params.cfg)

**Exported Functions:**

```python
def create_training_payload(
    train_data_file: Path,
    output_dir: Path,
    overrides: Optional[Dict[str, Any]] = None,
    execution_config: Optional[PyTorchExecutionConfig] = None,
) -> TrainingPayload:
    """
    Create complete training configuration payload.

    Returns:
        TrainingPayload containing:
        - tf_training_config: TrainingConfig (canonical TensorFlow format)
        - pt_data_config: DataConfig (PyTorch singleton)
        - pt_model_config: ModelConfig (PyTorch singleton)
        - pt_training_config: TrainingConfig (PyTorch singleton)
        - execution_config: PyTorchExecutionConfig (runtime knobs)
        - overrides_applied: Dict[str, Any] (audit trail)
    """

def create_inference_payload(
    model_path: Path,
    test_data_file: Path,
    output_dir: Path,
    overrides: Optional[Dict[str, Any]] = None,
    execution_config: Optional[PyTorchExecutionConfig] = None,
) -> InferencePayload:
    """
    Create complete inference configuration payload.

    Returns:
        InferencePayload containing:
        - tf_inference_config: InferenceConfig (canonical TensorFlow format)
        - pt_data_config: DataConfig (PyTorch singleton)
        - pt_inference_config: InferenceConfig (PyTorch singleton)
        - execution_config: PyTorchExecutionConfig (runtime knobs)
        - overrides_applied: Dict[str, Any] (audit trail)
    """

def infer_probe_size(data_file: Path) -> int:
    """
    Extract probe size from NPZ metadata.

    Factored out from train.py:96-140 for reusability.
    Returns N value or raises ValueError with actionable message.
    """

def populate_legacy_params(
    tf_config: Union[TrainingConfig, InferenceConfig],
    force: bool = False,
) -> None:
    """
    Wrapper around update_legacy_dict with validation.

    Ensures CONFIG-001 compliance checkpoint is explicit.
    Logs params.cfg population for audit trails.
    """
```

### 2.2 Supporting Dataclasses

**Location:** `ptycho_torch/config_params.py` (extend existing) or `ptycho/config/config.py` (canonical location)

**New Dataclass: `PyTorchExecutionConfig`**

```python
@dataclass
class PyTorchExecutionConfig:
    """
    PyTorch-specific execution parameters (not in TensorFlow canonical configs).

    These knobs control runtime behavior: hardware selection, optimization,
    logging, checkpointing. They do NOT affect model topology or data pipeline.
    """

    # Hardware & Distributed Training
    accelerator: str = 'auto'  # cpu/gpu/tpu/mps/auto
    strategy: str = 'auto'  # auto/ddp/fsdp/deepspeed
    n_devices: int = 1
    deterministic: bool = True  # Reproducibility flag

    # Data Loading
    num_workers: int = 0
    pin_memory: bool = False
    persistent_workers: bool = False
    prefetch_factor: Optional[int] = None

    # Optimization
    learning_rate: float = 1e-3
    scheduler: str = 'Default'  # Default/Exponential/MultiStage/Adaptive
    gradient_clip_val: Optional[float] = None
    accum_steps: int = 1

    # Checkpointing & Early Stopping
    enable_checkpointing: bool = True
    checkpoint_save_top_k: int = 1
    checkpoint_monitor_metric: str = 'val_loss'
    early_stop_patience: int = 100

    # Logging & Experiment Tracking
    enable_progress_bar: bool = False  # Controlled by config.debug
    logger_backend: Optional[str] = None  # tensorboard/wandb/mlflow
    disable_mlflow: bool = False

    # Inference-Specific
    inference_batch_size: Optional[int] = None
    middle_trim: int = 0
    pad_eval: bool = False
```

**Rationale for Placement:**
- **Option A (Recommended):** `ptycho/config/config.py` alongside canonical configs → maintains single source of truth
- **Option B:** `ptycho_torch/config_params.py` → backend-specific but duplicates config pattern
- **Decision:** Deferred to Phase B1.c open questions (see Section 7)

### 2.3 Return Value Dataclasses

**Location:** `ptycho_torch/config_factory.py`

```python
@dataclass
class TrainingPayload:
    """Complete configuration bundle for training workflows."""
    tf_training_config: TrainingConfig  # Canonical TensorFlow format
    pt_data_config: DataConfig  # PyTorch singleton
    pt_model_config: ModelConfig  # PyTorch singleton
    pt_training_config: TrainingConfig  # PyTorch singleton
    execution_config: PyTorchExecutionConfig  # Runtime knobs
    overrides_applied: Dict[str, Any]  # Audit trail

@dataclass
class InferencePayload:
    """Complete configuration bundle for inference workflows."""
    tf_inference_config: InferenceConfig  # Canonical TensorFlow format
    pt_data_config: DataConfig  # PyTorch singleton
    pt_inference_config: InferenceConfig  # PyTorch singleton
    execution_config: PyTorchExecutionConfig  # Runtime knobs
    overrides_applied: Dict[str, Any]  # Audit trail
```

---

## 3. Integration Call Sites

### 3.1 CLI Entry Point: `ptycho_torch/train.py`

**Current Pattern (lines 464-535):**
- Manual construction of 3 PyTorch configs
- Inline probe size inference
- Scattered override application
- Manual config bridge calls

**After Factory Integration:**

```python
# OLD (lines 464-504): Manual construction
data_config = DataConfig(N=inferred_N, grid_size=(...), ...)
model_config = ModelConfig(mode='Unsupervised', ...)
training_config = TrainingConfig(epochs=args.max_epochs, ...)

# NEW: Single factory call
from ptycho_torch.config_factory import create_training_payload

payload = create_training_payload(
    train_data_file=args.train_data_file,
    output_dir=args.output_dir,
    overrides={
        'n_groups': args.n_groups,
        'batch_size': args.batch_size,
        'gridsize': args.gridsize,
        'max_epochs': args.max_epochs,
        # ... all CLI args
    },
    execution_config=PyTorchExecutionConfig(
        accelerator='cpu' if args.device == 'cpu' else 'auto',
        enable_progress_bar=args.debug,
    ),
)

# Factory handles:
# - Probe size inference
# - Config bridge translation
# - update_legacy_dict(params.cfg, payload.tf_training_config)
# - Validation (n_groups required, nphotons divergence warning)
```

**Benefits:**
- **58 lines → 15 lines** (73% reduction)
- Centralized validation logic
- Explicit audit trail via `payload.overrides_applied`
- Testable in isolation

### 3.2 Workflow Orchestration: `ptycho_torch/workflows/components.py`

**Current Pattern (line 150):**
- Receives pre-constructed TensorFlow `TrainingConfig`
- Calls `update_legacy_dict(params.cfg, config)` directly
- No PyTorch execution config management

**After Factory Integration:**

```python
# Workflows accept payload objects instead of raw configs
def run_cdi_example_torch(
    train_data: RawData,
    test_data: Optional[RawData],
    payload: TrainingPayload,  # NEW: Replaces 'config: TrainingConfig'
    do_stitching: bool = True,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Dict[str, Any]]:
    """
    Factory-driven workflow execution.

    Payload already contains:
    - All config objects (TF + PyTorch)
    - Execution parameters
    - params.cfg already populated (CONFIG-001 compliant)
    """
    # Use payload.execution_config for Lightning Trainer setup
    trainer = L.Trainer(
        max_epochs=payload.tf_training_config.nepochs,
        accelerator=payload.execution_config.accelerator,
        devices=payload.execution_config.n_devices,
        deterministic=payload.execution_config.deterministic,
        enable_progress_bar=payload.execution_config.enable_progress_bar,
        # ... other execution knobs
    )
```

**Benefits:**
- Decouples workflow logic from config construction
- Execution config centralized (no hardcoded `deterministic=True`)
- Workflows become config-agnostic (accept any valid payload)

### 3.3 Inference Entry Point: `ptycho_torch/inference.py`

**Current Pattern (lines 412-429, 494-495):**
- Similar manual construction as train.py
- Duplicate probe size inference logic
- Inline TensorFlow config creation

**After Factory Integration:**

```python
from ptycho_torch.config_factory import create_inference_payload

payload = create_inference_payload(
    model_path=args.model_path,
    test_data_file=args.test_data_file,
    output_dir=args.output_dir,
    overrides={
        'n_groups': args.n_groups,
        'gridsize': args.gridsize,
        # ... all CLI args
    },
    execution_config=PyTorchExecutionConfig(
        inference_batch_size=args.batch_size,
        middle_trim=args.middle_trim,
        pad_eval=args.pad_eval,
    ),
)

# Factory handles checkpoint loading + config restoration
```

---

## 4. Factory Implementation Strategy

### 4.1 Core Translation Flow

```
CLI Args
  ↓
[Factory Entry Point]
  ↓
[1] Validate Required Args (train_data_file, output_dir, etc.)
  ↓
[2] Infer Probe Size (if not in overrides)
  ↓
[3] Construct PyTorch Configs (DataConfig, ModelConfig, TrainingConfig)
  ↓
[4] Apply CLI Overrides (n_groups, batch_size, gridsize, etc.)
  ↓
[5] Translate to TensorFlow Canonical Configs
    └─> Delegate to config_bridge.to_training_config()
  ↓
[6] Populate params.cfg (CONFIG-001 checkpoint)
    └─> Call update_legacy_dict(params.cfg, tf_config)
  ↓
[7] Construct PyTorchExecutionConfig (runtime knobs)
  ↓
[8] Return Payload (TF config + PyTorch configs + execution config)
```

### 4.2 Override Precedence Rules

**Priority Order (highest to lowest):**
1. **Explicit overrides dict** (user-provided via factory call)
2. **Execution config fields** (PyTorchExecutionConfig instance)
3. **CLI argument defaults** (from argparse)
4. **PyTorch config defaults** (DataConfig, ModelConfig, TrainingConfig)
5. **TensorFlow config defaults** (TrainingConfig, ModelConfig, InferenceConfig)

**Example Conflict Resolution:**
```python
# Scenario: n_groups specified in both overrides and execution_config
overrides = {'n_groups': 512}
execution_config = PyTorchExecutionConfig(...)  # No n_groups field

# Resolution: overrides['n_groups'] = 512 wins (highest priority)
# If execution_config had n_groups field, it would still lose to overrides
```

### 4.3 Validation Requirements

**Critical Validations (must raise exceptions):**
- `train_data_file` / `test_data_file` path existence
- `n_groups` required in overrides (no default)
- Square grid check: `gridsize` must produce square (config_bridge.py:108-113)
- NPZ field validation: diffraction, probeGuess, xcoords, ycoords present

**Warnings (log but continue):**
- nphotons divergence: PyTorch default 1e5 vs TensorFlow 1e9 (config_bridge.py:259-269)
- Probe size inference failure → fallback to N=64
- test_data_file missing for training (optional validation data)

---

## 5. Testing Strategy (Phase B2)

### 5.1 RED Tests (Phase B2.b)

**Test Module:** `tests/torch/test_config_factory.py` (new file)

**Key Test Cases:**
1. **Factory Returns Correct Payload Structure**
   ```python
   def test_training_payload_structure():
       payload = create_training_payload(train_data, output_dir, overrides)
       assert isinstance(payload.tf_training_config, TrainingConfig)
       assert isinstance(payload.pt_data_config, DataConfig)
       assert 'n_groups' in payload.overrides_applied
   ```

2. **Config Bridge Integration**
   ```python
   def test_factory_delegates_to_bridge():
       payload = create_training_payload(...)
       # Verify grid_size tuple → gridsize int conversion happened
       assert payload.tf_training_config.model.gridsize == 2
       assert payload.pt_data_config.grid_size == (2, 2)
   ```

3. **params.cfg Population (CONFIG-001)**
   ```python
   def test_factory_populates_params_cfg():
       import ptycho.params as p
       p.cfg.clear()
       payload = create_training_payload(...)
       assert p.cfg['gridsize'] == 2
       assert p.cfg['N'] == 64
   ```

4. **Override Precedence**
   ```python
   def test_override_precedence():
       overrides = {'n_groups': 512, 'batch_size': 8}
       payload = create_training_payload(..., overrides=overrides)
       assert payload.tf_training_config.n_groups == 512
       assert payload.overrides_applied['batch_size'] == 8
   ```

5. **Validation Errors**
   ```python
   def test_missing_n_groups_raises_error():
       with pytest.raises(ValueError, match="n_groups required"):
           create_training_payload(..., overrides={})  # Missing n_groups
   ```

### 5.2 GREEN Tests (Phase B3.c)

After implementation, extend tests for:
- Inference payload creation
- Probe size inference helper
- PyTorchExecutionConfig integration
- End-to-end CLI → factory → workflow flow

**Runtime Budget:** Target <5s for factory tests (well under 90s integration budget)

---

## 6. Migration Plan (Phase B3)

### 6.1 Refactoring Sequence

1. **Phase B2:** Create factory module skeleton + RED tests
2. **Phase B3.a:** Implement factory logic
3. **Phase B3.b:** Refactor train.py CLI to use factories
4. **Phase B3.b:** Refactor workflows/components.py to accept payloads
5. **Phase B3.b:** Refactor inference.py CLI to use factories
6. **Phase B3.c:** Run full pytest suite, capture GREEN evidence

### 6.2 Backward Compatibility

**No Breaking Changes:**
- Existing `config_bridge.py` functions unchanged
- Legacy `update_legacy_dict()` API preserved
- Workflow function signatures accept payloads OR configs (overload during transition)

**Deprecation Path:**
- Phase B3: Factories coexist with manual construction
- Phase C: CLI refactored to use factories exclusively
- Phase D: Remove manual construction code, mark as deprecated

---

## 7. Open Questions & Design Decisions Deferred to B1.c

**See:** `open_questions.md` for full elaboration

**Summary of Unresolved Items:**
1. **PyTorchExecutionConfig Placement:** `ptycho/config/config.py` (canonical) vs `ptycho_torch/config_params.py` (backend-specific)?
2. **MLflow Positioning:** Execution config (runtime knob) vs canonical config (experiment metadata)?
3. **CLI Flag Naming Harmonization:** Should PyTorch CLI adopt `--nepochs` and `--n_groups` to match TensorFlow?
4. **Factory Ownership:** Should factories live in `ptycho_torch/` (backend-specific) or `ptycho/workflows/` (shared)?
5. **Probe Size Inference Fallback:** Hard error vs warning + fallback to N=64?
6. **specs/ptychodus_api_spec.md Update:** Does §4.8 need new subsection for factory contract?
7. **ADR-003 Governance Document:** Where should architectural decision record live?

---

## 8. Success Criteria (Phase B1 Exit)

- [x] Factory design documented with module structure, exported functions, and integration call sites
- [x] Override precedence rules defined with conflict resolution strategy
- [x] Testing strategy outlined (RED/GREEN phases)
- [x] Migration plan established with backward compatibility guarantees
- [x] Open questions captured for supervisor review
- [x] All file:line citations provided for existing code patterns
- [x] Design aligns with POLICY-001 (PyTorch mandatory), CONFIG-001 (params.cfg sync), DATA-001 (NPZ contracts)

**Next Phase:** B2 (Factory Module Skeleton + RED Tests)

---

## 9. References

**Phase A Inventories:**
- `plans/active/ADR-003-BACKEND-API/reports/2025-10-19T225905Z/phase_a_inventory/execution_knobs.md` — 54 PyTorch-only knobs
- `plans/active/ADR-003-BACKEND-API/reports/2025-10-19T225905Z/phase_a_inventory/cli_inventory.md` — CLI surface analysis

**Existing Code:**
- `ptycho_torch/config_bridge.py:79-380` — Translation adapters
- `ptycho_torch/train.py:464-535` — Manual config construction
- `ptycho_torch/workflows/components.py:150` — CONFIG-001 checkpoint
- `ptycho_torch/inference.py:412-495` — Inference config wiring

**Specifications:**
- `specs/ptychodus_api_spec.md` §4 — Reconstructor lifecycle
- `docs/workflows/pytorch.md` §§5–12 — PyTorch configuration guide
- `docs/findings.md` POLICY-001, CONFIG-001, DATA-001 — Critical constraints
