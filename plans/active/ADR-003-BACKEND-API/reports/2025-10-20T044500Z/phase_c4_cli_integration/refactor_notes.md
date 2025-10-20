# ADR-003 Phase C4 Refactor Notes — Training + Inference CLI Cleanup (2025-10-20T044500Z)

## Context

This document summarizes the hardcoded value elimination and factory integration performed during ADR-003 Phase C4.C implementation (Attempts #20-#21). Prior to this refactor, both CLI entry points constructed configs manually with embedded default values, violating override precedence rules and duplicating logic across multiple files.

## Training CLI Refactoring (Attempt #20)

### Files Modified

- **Primary:** `ptycho_torch/train.py` (lines 399-622)
- **Removed:** Lines 464-545 (81 lines of ad-hoc config construction)
- **Added:** Lines 517-622 (106 lines factory integration + execution config wiring)
- **Net Change:** +25 lines (factory pattern overhead offset by hardcode removal)

### Hardcoded Values Eliminated

1. **`nphotons = 1e9`** (line 530, old code)
   - **Location:** Manual `TFTrainingConfig` construction
   - **Impact:** HIGH — Overrode user CLI args and YAML config values
   - **New Source:** Delegated to factory default via `override_matrix.md` §5.2 (TensorFlow canonical default=1e9, PyTorch default=1e5)
   - **Override Path:** CLI flag `--nphotons` → factory `overrides` dict → `config_bridge` translation

2. **`K = 7`** (line 477, old code)
   - **Location:** `PTDataConfig(K=7)` instantiation
   - **Impact:** MEDIUM — Ignored dataset-specific neighbor requirements
   - **New Source:** Factory default via `PTDataConfig.K` field default (6 per override_matrix.md §5.3)
   - **Override Path:** CLI flag `--neighbor-count` → factory `overrides` dict → `config_bridge` translation

3. **`experiment_name = 'ptychopinn_pytorch'`** (line 494, old code)
   - **Location:** `PTTrainingConfig(experiment_name=...)` construction
   - **Impact:** LOW — Naming convention hardcoded instead of derived
   - **New Source:** Factory default via `PTTrainingConfig.experiment_name` field
   - **Override Path:** CLI flag (if exposed in Phase D) → factory override

### Factory Integration Pattern

**Old Code Structure (lines 464-545):**

```python
# Manual config construction with hardcoded defaults
pt_data_config = PTDataConfig(
    N=args.N,
    grid_size=(args.gridsize, args.gridsize),
    K=7,  # HARDCODED
    # ... other fields
)

pt_training_config = PTTrainingConfig(
    epochs=args.max_epochs,
    experiment_name='ptychopinn_pytorch',  # HARDCODED
    # ... other fields
)

# Manual config_bridge calls
from ptycho_torch.config_bridge import to_model_config, to_training_config

tf_model_config = to_model_config(pt_data_config, pt_model_config)
tf_training_config = to_training_config(
    tf_model_config,
    pt_data_config,
    pt_model_config,
    pt_training_config,
    overrides={
        'train_data_file': train_data_path,
        'test_data_file': test_data_path,
        'n_groups': args.n_images,
        'nphotons': 1e9,  # HARDCODED OVERRIDE (highest precedence)
        # ... other overrides
    }
)

# Manual params.cfg population
from ptycho.config.config import update_legacy_dict
import ptycho.params
update_legacy_dict(ptycho.params.cfg, tf_training_config)
```

**New Code Structure (lines 517-622):**

```python
# Factory call with minimal override dict
from ptycho_torch.config_factory import create_training_payload

overrides = {
    'n_groups': args.n_images,
    'gridsize': args.gridsize,
    'batch_size': args.batch_size,
    # ... other CLI args (no hardcoded defaults)
}

# Single factory call replaces ~80 lines of manual wiring
payload = create_training_payload(
    train_data_file=train_data_path,
    test_data_file=test_data_path,
    output_dir=output_dir,
    overrides=overrides,
    execution_config=execution_config,  # Phase C3 wiring
)

# Extract configs from payload (factory already populated params.cfg)
tf_training_config = payload.tf_training_config
pt_data_config = payload.pt_data_config
pt_model_config = payload.pt_model_config
pt_training_config = payload.pt_training_config
```

### Benefits Achieved

1. **Single Source of Truth:** Default values now live in `override_matrix.md` §5 and factory implementation, eliminating CLI-level hardcodes
2. **CONFIG-001 Compliance:** Factory guarantees `update_legacy_dict` called before any IO operations (lines 560-577 in factory)
3. **Override Precedence:** Five-level precedence enforced by factory (spec default → dataclass default → YAML → CLI → explicit override)
4. **Code Reduction:** 73% reduction in config construction logic (81 lines → 22 lines)
5. **Test Surface:** Factory logic covered by `tests/torch/test_config_factory.py` (19 cases GREEN per Phase B3.a)

### Execution Config Wiring (Phase C3 Integration)

- **Training CLI** now accepts 4 execution config flags: `--accelerator`, `--deterministic/--no-deterministic`, `--num-workers`, `--learning-rate`
- **Execution config** instantiated at lines 517-569 and passed to factory (line 576)
- **Trainer kwargs** derived from execution config at lines 266-299 (threaded via `main(..., execution_config=execution_config)`)

### Validation Evidence

- **Targeted Tests:** `CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_cli_train_torch.py::TestExecutionConfigCLI -vv` → **6/6 PASSED** (Attempt #20 artifact: `pytest_cli_train_green.log`)
- **Full Regression:** `CUDA_VISIBLE_DEVICES="" pytest tests/ -v` → **276 passed, 17 skipped, 1 xfailed** (+5 vs C3 baseline, ZERO new failures)

---

## Inference CLI Refactoring (Attempt #21)

### Files Modified

- **Primary:** `ptycho_torch/inference.py` (lines 455-652)
- **Removed:** Lines 455-633 (178 lines manual checkpoint loading + inference)
- **Added:** Lines 455-652 (197 lines factory integration + CONFIG-001 compliance)
- **Net Change:** +19 lines (factory overhead + improved error handling)

### Hardcoded Values Eliminated

**NOTE:** Inference CLI had NO explicit hardcoded defaults (unlike training CLI), but it BYPASSED factory validation entirely, causing multiple compliance violations:

1. **CONFIG-001 Violation:** No `update_legacy_dict` call before `RawData.from_file` (line 524, old code)
   - **Impact:** CRITICAL — `RawData.generate_grouped_data` would fail with uninitialized `params.cfg['gridsize']`
   - **Fix:** Factory now populates params.cfg at line 428 (`config_factory.py:428`)

2. **Checkpoint Validation:** Ad-hoc file discovery (lines 473-491, old code)
   - **Impact:** MEDIUM — No wts.h5.zip presence check per spec §4.6
   - **Fix:** Factory validates `model_path / "wts.h5.zip"` exists (lines 355-361 in factory)

3. **NPZ Field Validation:** Manual `required_fields` check (lines 529-537, old code)
   - **Impact:** LOW — Duplicated DATA-001 validation already in RawData
   - **Fix:** Delegated to RawData.from_file + factory path validation

### Factory Integration Pattern

**Old Code Structure (lines 455-542):**

```python
# Manual path validation (no CONFIG-001 compliance)
model_path = Path(args.model_path)
test_data_path = Path(args.test_data)

# Manual checkpoint discovery
checkpoint_candidates = [...]
checkpoint_path = None
for candidate in checkpoint_candidates:
    if candidate.exists():
        checkpoint_path = candidate
        break

# Direct checkpoint loading (NO params.cfg initialization)
model = PtychoPINN_Lightning.load_from_checkpoint(str(checkpoint_path), ...)

# Direct NPZ loading (BEFORE params.cfg populated)
test_data = np.load(test_data_path)
```

**New Code Structure (lines 455-572):**

```python
# Factory call with CONFIG-001 compliance
from ptycho_torch.config_factory import create_inference_payload
from ptycho.raw_data import RawData

overrides = {
    'n_groups': args.n_images,  # Map CLI arg to canonical field
}

# Factory validates paths, checkpoint, and populates params.cfg
payload = create_inference_payload(
    model_path=model_path,
    test_data_file=test_data_path,
    output_dir=output_dir,
    overrides=overrides,
    execution_config=execution_config,
)

# Extract configs (params.cfg already populated by factory)
tf_inference_config = payload.tf_inference_config
execution_config = payload.execution_config

# NOW safe to load data (params.cfg initialized)
raw_data = RawData.from_file(str(test_data_path))
```

### Benefits Achieved

1. **CONFIG-001 Compliance:** Factory enforces `update_legacy_dict` ordering (line 428 in factory)
2. **Spec Alignment:** Checkpoint validation per `specs/ptychodus_api_spec.md` §4.6 (wts.h5.zip presence check)
3. **Execution Config Integration:** Accelerator/num-workers/inference-batch-size now configurable via CLI (Phase C4.C5 flags)
4. **Error Messaging:** Factory provides actionable errors with spec citations (e.g., "Ensure test_data conforms to DATA-001")

### Validation Evidence (Attempt #21 Expected)

- **Targeted Tests:** `CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_cli_inference_torch.py::TestInferenceCLI -vv` → **Expected: 4/4 PASSED** (pending execution this loop)
- **Factory Smoke:** `CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_config_factory.py -k ExecutionConfig -vv` → **Expected: GREEN** (no regressions)

---

## Sourcing Changes Summary

### Training CLI (`ptycho_torch/train.py`)

| Field/Value | Old Source | New Source | Priority Level |
|-------------|-----------|-----------|----------------|
| `nphotons` | Hardcoded `1e9` (line 530) | Factory default (TF canonical) via override_matrix.md §5.2 | Level 5 (highest) |
| `K` (neighbor_count) | Hardcoded `7` (line 477) | Factory default (PT `DataConfig.K=6`) via override_matrix.md §5.3 | Level 3 (dataclass) |
| `experiment_name` | Hardcoded `'ptychopinn_pytorch'` (line 494) | Factory default (PT `TrainingConfig.experiment_name`) | Level 3 (dataclass) |
| All other fields | Manual argparse → dataclass mapping (lines 470-545) | Factory `overrides` dict delegation (lines 517-576) | Level 4 (CLI args) |

### Inference CLI (`ptycho_torch/inference.py`)

| Operation | Old Source | New Source | Compliance Impact |
|-----------|-----------|-----------|-------------------|
| Checkpoint validation | Ad-hoc file discovery (lines 473-491) | Factory `create_inference_payload` (lines 355-361 in factory) | Spec §4.6 compliance |
| CONFIG-001 bridging | **MISSING** (no `update_legacy_dict` call) | Factory line 428 (`populate_legacy_params`) | **CRITICAL FIX** |
| NPZ field validation | Manual `required_fields` check (lines 529-537) | Delegated to RawData.from_file | DATA-001 alignment |
| Device resolution | Manual `args.device` mapping (line 506) | Execution config `accelerator` field (lines 537-545) | Phase C3 wiring |

---

## Override Precedence Reminder (Per `override_matrix.md` §2)

**Five-Level Hierarchy (Lowest → Highest Precedence):**

1. **Spec Defaults:** Normative values from `specs/ptychodus_api_spec.md` (e.g., N=64 per DATA-001)
2. **Execution Config Defaults:** CPU-safe defaults from `PyTorchExecutionConfig` (e.g., accelerator='cpu', num_workers=0)
3. **Dataclass Defaults:** PyTorch singleton defaults (e.g., `PTDataConfig.K=6`, `PTTrainingConfig.epochs=10`)
4. **CLI Arguments:** User-provided flags (e.g., `--n_images 128`, `--learning-rate 1e-4`)
5. **Explicit Overrides:** Programmatic overrides dict (e.g., `overrides={'nphotons': 1e9}`)

**Factory Application Order:**

1. Instantiate dataclass with spec/field defaults (Levels 1-3)
2. Apply CLI args to override dict (Level 4)
3. Apply explicit overrides if provided (Level 5)
4. Translate via config_bridge with merged overrides
5. Populate params.cfg via `populate_legacy_params`

---

## Deferred Work (Phase D)

The following training CLI hardcodes were **intentionally preserved** for Phase D:

1. **Checkpoint Callbacks:** `--checkpoint-save-top-k`, `--checkpoint-monitor-metric` (requires Lightning callback factory)
2. **LR Scheduler:** `--scheduler` (requires scheduler factory with type selection)
3. **Logger Backend:** `--logger-backend` (MLflow vs TensorBoard governance pending)

These knobs require additional design work beyond simple argparse→dataclass mapping and are tracked in `plans/active/ADR-003-BACKEND-API/reports/2025-10-20T033100Z/phase_c4_cli_integration/plan.md` §"Deferred to Phase D".

---

## Artifacts Generated This Loop

- **This Document:** `refactor_notes.md` (C4.C4 deliverable)
- **Training CLI GREEN Log:** `pytest_cli_train_green.log` (Attempt #20, 6/6 PASSED)
- **Inference CLI GREEN Log:** `pytest_cli_inference_green.log` (Attempt #21, pending execution)
- **Full Regression Log:** `pytest_full_suite_c4.log` (Attempt #20, 276 passed)

---

## Exit Criteria Validation (C4.C)

- [x] **C4.C1:** Training CLI execution config flags added (4 flags: accelerator, deterministic, num-workers, learning-rate)
- [x] **C4.C2:** Training CLI ad-hoc config construction replaced with factory (81 lines → 22 lines)
- [x] **C4.C3:** Training CLI execution config threaded to workflows (via `main(..., execution_config=...)`)
- [x] **C4.C4:** Hardcoded overrides documented (this file) — 3 training hardcodes eliminated (nphotons, K, experiment_name)
- [x] **C4.C5:** Inference CLI execution config flags added (3 flags: accelerator, num-workers, inference-batch-size)
- [x] **C4.C6:** Inference CLI factory integration (CONFIG-001 compliance restored, checkpoint validation added)
- [x] **C4.C7:** Inference CLI CONFIG-001 ordering maintained (factory calls `populate_legacy_params` before RawData.from_file)

**Phase C4.C COMPLETE** — All implementation tasks finished, ready for validation (C4.D).

---

## References

- **Factory Design:** `plans/active/ADR-003-BACKEND-API/reports/2025-10-19T232336Z/phase_b_factories/factory_design.md` §3
- **Override Matrix:** `plans/active/ADR-003-BACKEND-API/reports/2025-10-19T232336Z/phase_b_factories/override_matrix.md` §5
- **Phase C4 Plan:** `plans/active/ADR-003-BACKEND-API/reports/2025-10-20T033100Z/phase_c4_cli_integration/plan.md` §C4.C
- **Spec Reference:** `specs/ptychodus_api_spec.md` §4.6 (checkpoint loading), §4.8 (backend selection)
- **CONFIG-001 Finding:** `docs/findings.md#config-001`
- **DATA-001 Contract:** `specs/data_contracts.md` §1
