# ADR-003 Phase B1 — Open Questions & Specification Impacts

**Date:** 2025-10-19
**Status:** Awaiting supervisor review and governance decisions

---

## 1. Critical Design Decisions (Blocking Phase B2)

### Q1: Where Should `PyTorchExecutionConfig` Live?

**Context:**
The factory design introduces a new dataclass `PyTorchExecutionConfig` containing 54 PyTorch-specific runtime knobs (see Phase A `execution_knobs.md`). These fields do NOT exist in TensorFlow canonical configs because they control backend-specific behavior (Lightning Trainer, DataLoader, distributed training).

**Options:**

**Option A: Canonical Location (`ptycho/config/config.py`)**
- **Pros:**
  - Single source of truth for ALL configuration dataclasses
  - Discoverable alongside TrainingConfig, ModelConfig, InferenceConfig
  - Follows existing pattern (all @dataclass definitions in one file)
  - Easier spec updates (single file to maintain)
- **Cons:**
  - Mixes TensorFlow-agnostic canonical configs with PyTorch-specific execution knobs
  - Violates separation of concerns (canonical vs backend-specific)
  - Increases cognitive load for TensorFlow-only users

**Option B: Backend-Specific Location (`ptycho_torch/config_params.py`)**
- **Pros:**
  - Clear separation: canonical configs (ptycho/config/) vs execution configs (ptycho_torch/)
  - PyTorch-only users don't pollute canonical namespace
  - Easier to maintain backend-specific knobs independently
- **Cons:**
  - Duplicates dataclass pattern across two locations
  - Harder to discover (split config definitions)
  - Potential import confusion (which config module to use?)

**Option C: New Module (`ptycho/config/execution.py`)**
- **Pros:**
  - Separates canonical (model/data) from execution (runtime) concerns
  - Keeps all config definitions under `ptycho/config/` namespace
  - Allows future TensorFlowExecutionConfig if needed
- **Cons:**
  - Introduces new module for single dataclass (may be premature)
  - Unclear ownership (ptycho/ vs ptycho_torch/)

**Recommendation:** **Option A** (canonical location) with clear docstring distinguishing canonical vs execution configs.

**Rationale:**
- Execution configs are still *configuration* (not runtime state)
- Centralizing in `ptycho/config/config.py` maintains single source of truth
- Spec updates (`specs/ptychodus_api_spec.md` §6) easier with single file
- Precedent: TensorFlow configs already mix model topology (ModelConfig) with training knobs (TrainingConfig)

**Decision (2025-10-19T234458Z, Supervisor):** Proceed with **Option A**. Implement `PyTorchExecutionConfig` in `ptycho/config/config.py` alongside the canonical dataclasses. Document the execution-only scope in the class docstring and reference POLICY-001 to reinforce PyTorch mandatory status.

**Follow-up Guidance:** Update Phase B2 plan items to treat this as an approved prerequisite. When authoring the dataclass, ensure imports are device/dtype neutral and avoid side effects during module import.

**Blocking:** Phase B2 skeleton creation depends on this decision. ✅ **Resolved**

---

### Q2: Should MLflow Knobs Be in Execution Config or Canonical Config?

**Context:**
Phase A identified 4 MLflow-related knobs:
1. `disable_mlflow` (CLI flag only, in legacy API)
2. `experiment_name` (already in TrainingConfig)
3. `notes` (already in TrainingConfig)
4. `model_name` (already in TrainingConfig)

**Current State:**
- TensorFlow `TrainingConfig` includes MLflow metadata fields (`experiment_name`, `notes`, `model_name`)
- PyTorch legacy API (`ptycho_torch/api/example_train.py:119`) has `--disable_mlflow` CLI flag
- Modern workflow (`ptycho_torch/workflows/components.py`) does NOT integrate MLflow (TODO comment exists)

**Question:** Should MLflow integration knobs be:
- **Option A:** Execution config (runtime logging/tracking behavior)
- **Option B:** Canonical config (experiment metadata)

**Recommendation:** **Option B** (canonical config) with execution config toggle.

**Rationale:**
- `experiment_name`, `notes`, `model_name` are experiment metadata (describe *what* is being trained)
- `disable_mlflow`, `logger_backend` are runtime toggles (control *how* tracking happens)
- Split responsibility:
  - **Canonical config:** experiment_name, notes, model_name (TensorFlow TrainingConfig)
  - **Execution config:** disable_mlflow, logger_backend (PyTorchExecutionConfig)

**Impact on Specs:**
- `specs/ptychodus_api_spec.md` §5 (Configuration Field Reference) should clarify canonical vs execution distinction
- New subsection needed: "§6. Backend Execution Configuration" documenting PyTorchExecutionConfig

**Blocking:** No (can proceed with split approach).

---

### Q3: Should PyTorch CLI Adopt TensorFlow Naming (`--nepochs`, `--n_groups`)?

**Context:**
Current PyTorch CLI uses different flag names than TensorFlow:
- PyTorch: `--max_epochs` → TensorFlow: `nepochs`
- PyTorch: `--n_images` → TensorFlow: `n_groups`
- PyTorch: `--activate` → TensorFlow: `amp_activation`

**User Impact:**
- Users switching between backends must remember different CLI syntax
- Documentation must maintain two parallel command references
- Breaks principle of backend transparency (user shouldn't care about implementation)

**Options:**

**Option A: Harmonize to TensorFlow Naming**
- **Pros:**
  - Single CLI syntax across backends
  - Easier documentation (one command reference)
  - Backend becomes true implementation detail
- **Cons:**
  - Breaking change for existing PyTorch users
  - Requires migration guide and deprecation warnings

**Option B: Accept Divergence**
- **Pros:**
  - No breaking changes
  - Each backend can optimize for its conventions (PyTorch uses `epochs` terminology)
- **Cons:**
  - Permanent user confusion
  - Violates backend transparency goal

**Option C: Alias Both (Accept Either Name)**
- **Pros:**
  - Backward compatible
  - Allows gradual migration to preferred naming
- **Cons:**
  - Increases CLI surface area
  - Harder to deprecate old names
  - Confusing help text

**Recommendation:** **Option A** (harmonize) with deprecation warnings in Phase D.

**Rationale:**
- TensorFlow CLI is older and more established (precedent)
- Backend selection (§4.8) should be transparent to users
- Breaking change acceptable if phased (deprecation warnings → error in 6 months → removal)

**Migration Plan:**
```python
# Phase C: Add deprecation warnings
if args.max_epochs:
    warnings.warn("--max_epochs deprecated; use --nepochs", DeprecationWarning)
    args.nepochs = args.max_epochs

# Phase D: Remove deprecated flags
# Phase E: Clean up legacy code
```

**Impact on Specs:**
- `specs/ptychodus_api_spec.md` should document CLI naming standards
- Recommendation: Add "§7. CLI Design Principles" subsection

**Blocking:** No (can proceed with current naming, refactor in Phase D).

---

## 2. Specification & Documentation Updates Required

### S1: Update `specs/ptychodus_api_spec.md`

**Required Changes:**

**New Section: §6. Backend Execution Configuration**
```markdown
## 6. Backend Execution Configuration

PyTorch workflows support additional execution parameters not present in TensorFlow
canonical configurations. These parameters control runtime behavior (hardware
selection, optimization, logging) without affecting model topology or data pipeline.

### 6.1. PyTorchExecutionConfig Fields

| Field | Type | Default | Purpose |
|-------|------|---------|---------|
| accelerator | str | 'auto' | Hardware backend (cpu/gpu/tpu/mps) |
| strategy | str | 'auto' | Distributed strategy (auto/ddp/fsdp) |
| deterministic | bool | True | Reproducibility flag |
| ... (54 fields total) ... |

### 6.2. Execution Config vs Canonical Config

**Canonical configs** (ModelConfig, TrainingConfig, InferenceConfig) define:
- Model architecture and topology
- Data pipeline parameters
- Physics simulation settings
- Cross-backend portable parameters

**Execution configs** (PyTorchExecutionConfig) define:
- Backend-specific runtime knobs
- Hardware and distributed training settings
- Logging and checkpointing behavior
- Optimization hyperparameters

Execution configs are OPTIONAL; reasonable defaults provided.
```

**Updates to Existing Sections:**

**§4.8. Backend Selection & Dispatch** (extend)
- Add factory contract: dispatchers SHOULD use `create_training_payload()` / `create_inference_payload()`
- Document override precedence rules (explicit overrides > execution config > CLI defaults)
- Clarify CONFIG-001 compliance requirement for factories

**§5. Configuration Field Reference** (extend)
- Add column: "Config Type" (Canonical vs Execution)
- Mark execution-only fields clearly
- Add notes about PyTorch-specific defaults (nphotons 1e5 vs 1e9)

**New Section: §7. CLI Design Principles** (recommended)
```markdown
## 7. CLI Design Principles

### 7.1. Naming Conventions
- Use snake_case for multi-word flags (--train_data_file, NOT --trainDataFile)
- Prefer TensorFlow canonical naming for cross-backend parameters
- Use full words over abbreviations (--nepochs, NOT --ne)

### 7.2. Backend Transparency
- Users should not need to know which backend is executing
- CLI syntax should be identical across backends (exceptions documented)
- Backend selection via --backend flag only (no duplicate CLIs)

### 7.3. Backward Compatibility
- Deprecated flags emit warnings for 2 releases before removal
- Aliases supported during transition periods
- Breaking changes documented in release notes
```

**Owner:** Spec maintainer (coordinate with supervisor)
**Timeline:** Phase B or C (before factory implementation stabilizes)

---

### S2: Create `docs/architecture/adr/ADR-003.md`

**Status:** **Directory `docs/architecture/adr/` does not exist.**

**Required Actions:**
1. Create ADR directory structure
2. Author ADR-003 documenting factory pattern architectural decision

**Proposed ADR-003 Outline:**
```markdown
# ADR-003: Centralized Configuration Factories for PyTorch Backend

## Status
Accepted (2025-10-19)

## Context
PyTorch backend configuration construction was scattered across CLI (train.py:464-535)
and workflow entry points (components.py:150), leading to:
- Duplicated probe inference logic
- Inconsistent override handling
- Hard-to-test configuration assembly
- CONFIG-001 compliance violations (missing update_legacy_dict calls)

## Decision
Introduce `ptycho_torch/config_factory.py` providing:
- `create_training_payload()` — Centralized training config assembly
- `create_inference_payload()` — Centralized inference config assembly
- `PyTorchExecutionConfig` dataclass — Backend-specific runtime knobs

Factories delegate to existing `config_bridge.py` adapters and enforce CONFIG-001
compliance checkpoint (params.cfg population before data loading).

## Consequences
**Positive:**
- Single source of truth for config construction
- Testable in isolation (RED/GREEN TDD cycle)
- Consistent override precedence enforcement
- Easier to maintain (58 lines → 15 lines in CLI)

**Negative:**
- Introduces new abstraction layer (learning curve)
- Requires test updates (extend test_config_bridge.py)
- Potential breaking change if CLI refactored

## Alternatives Considered
1. Keep scattered construction → rejected (tech debt accumulation)
2. Embed factories in workflows → rejected (violates single responsibility)
3. Use builder pattern → rejected (over-engineered for current needs)
```

**Owner:** ADR-003 initiative lead (current developer)
**Timeline:** Phase B or E2 (architectural decision documentation)

---

### S3: Update `docs/workflows/pytorch.md`

**Required Changes:**

**§12. Backend Selection in Ptychodus Integration** (extend)
- Add subsection: "12.1. Factory-Driven Configuration"
- Document `create_training_payload()` usage examples
- Explain override precedence rules
- Link to override_matrix.md for comprehensive field reference

**New Section: §13. Configuration Factory API** (new)
```markdown
## 13. Configuration Factory API

### 13.1. Training Configuration

```python
from ptycho_torch.config_factory import create_training_payload

payload = create_training_payload(
    train_data_file=Path('data.npz'),
    output_dir=Path('outputs/'),
    overrides={'n_groups': 512, 'batch_size': 8},
    execution_config=PyTorchExecutionConfig(accelerator='gpu'),
)

# Payload contains:
# - tf_training_config (canonical TensorFlow format)
# - pt_data_config, pt_model_config, pt_training_config (PyTorch singletons)
# - execution_config (runtime knobs)
# - overrides_applied (audit trail)
```

### 13.2. Override Precedence
1. Explicit overrides dict (highest priority)
2. Execution config fields
3. CLI argument defaults
4. PyTorch config defaults
5. TensorFlow config defaults (lowest priority)
```

**Owner:** PyTorch workflow documentation maintainer
**Timeline:** Phase B3 (after factory implementation)

---

## 3. Technical Debt & Known Limitations

### T1: Hardcoded Values Blocking CLI Flexibility

**Identified Hardcoded Values:**
- `learning_rate`: 1e-3 (`ptycho_torch/workflows/components.py:538`)
- `amp_activation`: 'silu' (`ptycho_torch/train.py:484`)
- `early_stop_patience`: 100 (`ptycho_torch/train.py:247`)
- `num_workers`: 0 (`ptycho_torch/workflows/components.py:361`)
- `pin_memory`: False (`ptycho_torch/workflows/components.py:362`)

**Impact:**
Users cannot tune these parameters without editing code.

**Recommendation:**
- Phase B: Document in PyTorchExecutionConfig
- Phase C: Add CLI flags (`--learning_rate`, `--early_stop_patience`)
- Phase D: Remove hardcoded values

**Owner:** ADR-003 Phase C/D implementer

---

### T2: Missing CLI Flags for Canonical Config Fields

**High-Priority Missing Flags:**
- `--n_subsample` (independent subsampling control)
- `--subsample_seed` (reproducible subsampling)
- `--learning_rate` (optimizer tuning)

**Medium-Priority Missing Flags:**
- `--sequential_sampling` (deterministic grouping)
- `--early_stop_patience` (training control)
- `--nll_weight`, `--mae_weight` (loss tuning)
- `--probe_trainable` (experimental feature)

**Recommendation:**
Add HIGH and MEDIUM priority flags in Phase D to achieve CLI parity with TensorFlow.

**Owner:** ADR-003 Phase D CLI harmonization

---

### T3: Probe Size Inference Fallback Behavior

**Current Behavior:**
`_infer_probe_size()` (`ptycho_torch/train.py:96-140`) silently falls back to N=64 if:
- NPZ file missing
- `probeGuess` key missing
- `probeGuess` not square
- Any other exception

**Question:** Should fallback be:
- **Option A:** Hard error (raise exception) → forces user to provide explicit N
- **Option B:** Warning + fallback to N=64 → permissive for quick experiments
- **Option C:** Configurable via execution config (`probe_size_inference_strict` bool)

**Current Recommendation:** **Option B** (warning + fallback)

**Rationale:**
- Matches TensorFlow permissive behavior
- Allows quick prototyping without complete dataset
- Warning alerts user to inferred vs default distinction

**Log Example:**
```
WARNING: Could not infer probe size from /path/to/data.npz (missing probeGuess key).
         Falling back to default N=64. For accurate results, provide explicit N
         via overrides={'N': 128} or ensure probeGuess exists in NPZ.
```

**Owner:** ADR-003 Phase B implementer (apply during factory implementation)

---

### T4: nphotons Divergence (PyTorch 1e5 vs TensorFlow 1e9)

**Context:**
- PyTorch `DataConfig` default: `nphotons = 1e5` (`ptycho_torch/config_params.py:32`)
- TensorFlow `TrainingConfig` default: `nphotons = 1e9` (`ptycho/config/config.py:109`)
- Config bridge warns but uses TensorFlow value (`config_bridge.py:259-269`)

**Question:** Should factory:
- **Option A:** Always use TensorFlow default (current behavior)
- **Option B:** Use PyTorch default if user doesn't override
- **Option C:** Raise error, force user to specify explicitly

**Recommendation:** **Option A** with enhanced warning.

**Rationale:**
- TensorFlow config is canonical source of truth
- Physics simulations more sensitive to nphotons than data loading
- Explicit override available if user wants PyTorch default

**Enhanced Warning:**
```
WARNING: nphotons divergence detected:
         - PyTorch DataConfig default: 1e5
         - TensorFlow TrainingConfig default: 1e9 (USING THIS)
         To suppress, explicitly set overrides={'nphotons': 1e9} (recommended)
         or overrides={'nphotons': 1e5} (PyTorch behavior).
```

**Owner:** ADR-003 Phase B implementer

---

## 4. Testing & Validation Concerns

### V1: Runtime Budget for Factory Tests

**Context:**
- Integration test budget: ≤90s (TEST-PYTORCH-001 Phase D)
- Current integration runtime: 14.53s (Phase D baseline)
- Factory tests should be unit tests (fast, isolated)

**Question:** What is acceptable runtime budget for factory test suite?

**Recommendation:** **≤5 seconds total** for all factory tests.

**Rationale:**
- Factory tests are unit tests (no model training)
- Should test config construction, validation, bridge integration only
- 5s allows ~50 test cases at 100ms each

**Monitoring:**
```bash
pytest tests/torch/test_config_factory.py -vv --durations=10
```

**Owner:** ADR-003 Phase B2 test author

---

### V2: Test Fixture Requirements

**Question:** Should factory tests use:
- **Option A:** Minimal synthetic configs (no real NPZ files)
- **Option B:** Existing integration fixture (`minimal_dataset_v1.npz`)
- **Option C:** New factory-specific fixture

**Recommendation:** **Option A** (synthetic configs) for unit tests, **Option B** for integration.

**Rationale:**
- Unit tests should not depend on file I/O (faster, more isolated)
- Integration tests (`test_integration_workflow_torch.py`) already use minimal_dataset_v1.npz
- Factory-specific fixture unnecessary (existing fixture sufficient)

**Example Synthetic Fixture:**
```python
@pytest.fixture
def mock_training_args():
    return {
        'train_data_file': Path('/mock/train.npz'),  # Won't be loaded in unit test
        'output_dir': Path('/tmp/test_output'),
        'overrides': {'n_groups': 512, 'gridsize': 2},
    }
```

**Owner:** ADR-003 Phase B2 test author

---

## 5. Governance & Ownership

### G1: Who Reviews Factory Design Artifacts?

**Question:** Who must approve Phase B1 documentation before proceeding to B2?

**Proposed Reviewers:**
1. **Supervisor agent** (primary reviewer via loop.sh)
2. **Spec maintainer** (if §6 addition required)
3. **PyTorch backend lead** (architectural alignment)

**Blocking:** Phase B2 (factory skeleton) depends on design approval.

---

### G2: Spec Update Approval Process

**Question:** Can ADR-003 initiative author spec updates directly, or required separate governance?

**Options:**
- **Option A:** Initiative owns spec updates (faster iteration)
- **Option B:** Spec updates require separate review (governance control)

**Recommendation:** **Option A** for minor additions (§6), **Option B** for structural changes.

**Rationale:**
- Adding §6 (PyTorchExecutionConfig) is clarification, not breaking change
- Structural changes (§7 CLI Principles) should have broader review

**Owner:** Project maintainer decision

---

## 6. Phase B2 Readiness Checklist

Before proceeding to Phase B2 (factory skeleton + RED tests), resolve:

- [ ] **Q1:** PyTorchExecutionConfig placement (ptycho/config/config.py vs ptycho_torch/)
- [ ] **Q2:** MLflow knob split (canonical vs execution) — **Can proceed without**
- [ ] **Q3:** CLI naming harmonization — **Can proceed, refactor in Phase D**
- [ ] **S1:** Spec §6 addition approval — **Document decision, implement in B or C**
- [ ] **S2:** ADR-003.md authoring — **Document decision, implement in B or E2**
- [ ] **T3:** Probe size fallback behavior (hard error vs warning) — **Apply warning approach**
- [ ] **T4:** nphotons divergence handling — **Apply enhanced warning**
- [ ] **V1:** Factory test runtime budget (5s target) — **Accepted**
- [ ] **V2:** Test fixture strategy (synthetic vs real) — **Accepted**
- [ ] **G1:** Design reviewer assignment — **Supervisor approval required**

**Status:** **BLOCKING:** Q1 (PyTorchExecutionConfig placement) must be resolved before B2.

---

## 7. Summary of Required Follow-Ups

| Item | Owner | Timeline | Blocking |
|------|-------|----------|----------|
| Resolve Q1 (PyTorchExecutionConfig placement) | Supervisor | Before B2 | **YES** |
| Draft specs §6 (Backend Execution Config) | Spec maintainer | Phase B or C | No |
| Create ADR-003.md | Initiative lead | Phase B or E2 | No |
| Add HIGH-priority CLI flags | Phase D implementer | Phase D | No |
| Harmonize CLI naming | Phase D implementer | Phase D | No |
| Remove hardcoded values | Phase C/D implementer | Phase C/D | No |
| Update docs/workflows/pytorch.md §13 | Workflow docs | Phase B3 | No |

**Immediate Blocker:** Q1 resolution (recommend Option A: canonical location).

**Deferrable Items:** S1, S2, T1, T2 can be addressed in parallel with Phase B2/B3 implementation.

---

## 8. References

**Design Documents:**
- `factory_design.md` — Factory architecture and integration strategy
- `override_matrix.md` — Comprehensive field mapping and precedence rules

**Phase A Inventories:**
- `execution_knobs.md` — 54 PyTorch-specific execution parameters
- `cli_inventory.md` — CLI surface analysis and parity gaps

**Specifications:**
- `specs/ptychodus_api_spec.md` — Current API contract (requires §6 addition)
- `docs/workflows/pytorch.md` — PyTorch workflow guide (requires §13 addition)

**Existing Code:**
- `ptycho_torch/config_bridge.py` — Translation adapter contract
- `ptycho_torch/train.py:464-535` — Manual config construction to be replaced
- `ptycho/config/config.py` — Canonical config dataclasses
