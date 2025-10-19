# TEST-PYTORCH-001 Phase D3 Handoff Brief

**Issued:** 2025-10-19
**Source Initiative:** INTEGRATE-PYTORCH-001 (Phase E3.D — Documentation Handoff)
**Target Initiative:** TEST-PYTORCH-001 (Phase D3 — CI Integration & Ongoing Monitoring)
**Purpose:** Equip TEST-PYTORCH-001 owners with authoritative guidance for operating, extending, and monitoring the PyTorch integration regression suite without re-deriving Phase E design decisions.

---

## 1. Backend Selection Contract (D1.A)

### 1.1. Configuration Literals

Per `specs/ptychodus_api_spec.md` §4.8, backend selection is controlled through the `backend` field in configuration dataclasses:

```python
from ptycho.config.config import TrainingConfig, InferenceConfig

# Training with PyTorch backend
config = TrainingConfig(
    model=model_config,
    train_data_file=Path('data.npz'),
    backend='pytorch',  # or 'tensorflow' (default)
    # ... other parameters
)

# Inference with PyTorch backend
infer_config = InferenceConfig(
    model=model_config,
    model_path=Path('trained_model/'),
    test_data_file=Path('test.npz'),
    backend='pytorch',  # or 'tensorflow' (default)
    # ... other parameters
)
```

**Valid Literals:**
- `'tensorflow'` — Default backend (backward compatible)
- `'pytorch'` — PyTorch Lightning backend (POLICY-001 compliant)

**Invalid values** (e.g., `'torch'`, `'tf'`, `None`) will raise `ValueError` with corrective messaging.

### 1.2. CONFIG-001 Requirement (CRITICAL)

**Before any data loading or model construction**, PyTorch workflows **MUST** call `update_legacy_dict` to synchronize `params.cfg`:

```python
from ptycho.config.config import update_legacy_dict
from ptycho import params

# 1. Create modern dataclass config
config = TrainingConfig(...)  # or InferenceConfig

# 2. Bridge to legacy system (MANDATORY)
update_legacy_dict(params.cfg, config)

# 3. NOW it is safe to import modules that depend on global state
from ptycho.raw_data import RawData
from ptycho_torch.workflows.components import run_cdi_example_torch
```

**Failure to call `update_legacy_dict` voids parity guarantees** and will cause shape mismatches, parameter drift, and silent failures in data grouping.

**Reference:** `docs/debugging/QUICK_REFERENCE_PARAMS.md` §Golden Rule, `CLAUDE.md` §4.1

### 1.3. Fail-Fast Error Messaging

When PyTorch backend is selected but the `torch` module cannot be imported, the dispatcher **MUST** raise an actionable `RuntimeError` per POLICY-001:

```
RuntimeError: PyTorch backend selected but torch module unavailable.
Install PyTorch: pip install torch>=2.2
See docs/workflows/pytorch.md for installation guidance.
```

**Silent fallbacks to TensorFlow are prohibited.** This ensures users are immediately aware of missing dependencies and prevents cryptic downstream failures.

**Test Coverage:** `tests/torch/test_backend_selection.py::test_pytorch_unavailable_raises_runtime_error`

### 1.4. Dispatcher Routing Guarantees

Per §4.8, the dispatcher provides these guarantees:

1. **TensorFlow Path** (`backend='tensorflow'`): Delegates to `ptycho.workflows.components` without attempting PyTorch imports
2. **PyTorch Path** (`backend='pytorch'`): Delegates to `ptycho_torch.workflows.components` and returns identical `(amplitude, phase, results_dict)` structure
3. **CONFIG-001 Compliance**: Dispatcher calls `update_legacy_dict` before inspecting `config.backend` or routing to backend-specific modules
4. **Result Metadata**: Returned `results_dict` includes `results['backend']` for downstream logging/auditing

**Reference Implementation:** `ptycho/workflows/backend_selector.py:121-165` (if present), `tests/torch/test_backend_selection.py:59-170`

---

## 2. Regression Selectors & Cadence (D1.B)

### 2.1. Mandatory Pytest Selectors

The following selectors provide comprehensive PyTorch backend coverage:

#### 2.1.1. Integration Workflow Test (Priority: HIGH)

```bash
CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_integration_workflow_torch.py::test_run_pytorch_train_save_load_infer -vv
```

**Coverage:** Complete train→save→load→infer cycle via subprocess CLI calls
**Runtime Baseline:** 35.92s ± 0.06s (CPU-only, Ryzen 9 5950X)
**CI Budget:** ≤90s maximum (2.5× baseline), 60s warning threshold
**Environment:** CPU-only execution enforced via `CUDA_VISIBLE_DEVICES=""` fixture
**Artifacts:** Checkpoint (`.ckpt`), reconstruction PNGs (amplitude/phase)

**Reference:** `docs/workflows/pytorch.md` §11, `plans/active/TEST-PYTORCH-001/reports/2025-10-19T193425Z/phase_d_hardening/runtime_profile.md`

#### 2.1.2. Backend Selection Suite (Priority: HIGH)

```bash
pytest tests/torch/test_backend_selection.py -vv
```

**Coverage:**
- Backend field defaults (`'tensorflow'`)
- PyTorch routing correctness
- TensorFlow routing correctness
- Invalid backend raises `ValueError`
- PyTorch unavailability raises `RuntimeError` with actionable message

**Runtime:** <5s (unit-level tests)
**Test Count:** 4-5 tests (lines 59-170)
**Reference:** `specs/ptychodus_api_spec.md` §4.8

#### 2.1.3. Parity Validation Suite (Priority: MEDIUM)

```bash
# Config bridge parity
pytest tests/torch/test_config_bridge.py -k parity -vv

# Lightning orchestration
pytest tests/torch/test_workflows_components.py::TestTrainWithLightningRed -vv

# Stitching implementation
pytest tests/torch/test_workflows_components.py -k ReassembleCdiImageTorch -vv

# Checkpoint serialization
pytest tests/torch/test_lightning_checkpoint.py -vv

# Decoder shape parity
pytest tests/torch/test_workflows_components.py::TestDecoderLastShapeParity -vv
```

**Rationale:** Validates TensorFlow→PyTorch dataclass translation, Lightning training contract, inference+stitching parity, checkpoint persistence, and decoder output dimensions.

**Combined Runtime:** ~20-30s
**Test Count:** ~20-25 tests

#### 2.1.4. Model Manager Cross-Backend Tests (Priority: MEDIUM)

```bash
pytest tests/torch/test_model_manager.py::test_load_tensorflow_checkpoint_with_pytorch_backend -vv
```

**Coverage:** Validates descriptive errors when attempting to load TensorFlow `.h5.zip` checkpoints with PyTorch backend (and vice versa).

**Reference:** `specs/ptychodus_api_spec.md` §4.8 (Persistence Parity)

### 2.2. Suggested Execution Cadence

| Selector Group | Frequency | Trigger | Environment |
|:---------------|:----------|:--------|:------------|
| **Integration Workflow** | Per PR (required) | Pre-merge gate | CPU-only CI worker |
| **Backend Selection Suite** | Per PR (required) | Pre-merge gate | Any CI worker |
| **Parity Validation Suite** | Nightly or per-commit to `ptycho_torch/` | Code changes in PyTorch stack | CPU-only CI worker |
| **Model Manager Cross-Backend** | Weekly or on-demand | Cross-backend changes | Any CI worker |

**Monitoring Frequency Detail:**

- **Per-PR (Pre-Merge):**
  - Mandatory for all PRs touching `ptycho_torch/`, `tests/torch/`, or backend selection code
  - Run Integration Workflow + Backend Selection Suite as blocking gate
  - Budget: ≤2 minutes total (90s integration + 5s backend suite + overhead)

- **Nightly Automated Runs:**
  - Execute full Parity Validation Suite (config bridge, Lightning, stitching, checkpoint, decoder)
  - Capture runtime trends; alert if integration workflow >60s (1.7× baseline)
  - Archive pytest logs under `plans/active/TEST-PYTORCH-001/reports/<timestamp>/nightly/`

- **Weekly Deep Validation:**
  - Full torch test suite: `pytest tests/torch/ -vv`
  - Cross-backend checkpoint compatibility validation
  - Environment refresh (update Python/PyTorch/Lightning, revalidate guardrails)

**CI Environment Notes:**
- Use `CUDA_VISIBLE_DEVICES=""` to enforce CPU-only execution (CUDA hardware optional for PyTorch tests)
- TensorFlow-only CI environments automatically skip `tests/torch/` via directory-based pytest collection rules (`tests/conftest.py`)
- Local development expects PyTorch to be present; missing PyTorch raises actionable `ImportError` per POLICY-001

### 2.3. Runtime Guardrails Summary

| Metric | Threshold | Action |
|:-------|:----------|:-------|
| Integration workflow runtime | ≤90s | Hard CI timeout (2.5× baseline) |
| Integration workflow runtime | >60s | Flag for investigation (1.7× baseline) |
| Integration workflow runtime | 36s ± 5s | Expected baseline (modern CPU) |
| Integration workflow runtime | <20s | Investigate incomplete execution |

**Variance Considerations:** CPU frequency scaling, I/O jitter, dataset size (35 MB canonical fixture). See runtime profile for full analysis.

---

## 3. Artifact Expectations & Ownership Matrix (D1.C)

### 3.1. Required Artifacts (Integration Workflow)

| Artifact Type | Location Pattern | Size | Validation |
|:--------------|:-----------------|:-----|:-----------|
| **Lightning Checkpoint** | `{output_dir}/checkpoints/last.ckpt` | ~1-2 MB | Contains `hyper_parameters` key with all four dataclass configs |
| **Reconstruction Amplitude** | `{output_dir}/reconstruction_amplitude.png` | >1 KB | Valid PNG, dimensions match probe size |
| **Reconstruction Phase** | `{output_dir}/reconstruction_phase.png` | >1 KB | Valid PNG, dimensions match probe size |
| **Training Debug Log** | `{output_dir}/train_debug.log` | ~50-100 KB | Optional (debug mode only) |

**Checkpoint Contract:** Per Phase D1c (`INTEGRATE-PYTORCH-001` Attempts #32-34), checkpoints MUST contain serialized hyperparameters enabling state-free reload via `PtychoPINN_Lightning.load_from_checkpoint(ckpt_path)` without manual config kwargs.

**Validation Command:**
```python
import torch
ckpt = torch.load('checkpoints/last.ckpt')
assert 'hyper_parameters' in ckpt, "Checkpoint missing hyperparameters"
assert ckpt['hyper_parameters'] is not None
```

### 3.2. Artifact Archival Policy

**Test Artifacts (Transient):**
- Integration test uses pytest `tmp_path` fixture; artifacts deleted after test completion
- Evidence logs stored under `plans/active/TEST-PYTORCH-001/reports/<timestamp>/`

**Development Artifacts (Persistent):**
- Phased implementation evidence under `plans/active/INTEGRATE-PYTORCH-001/reports/<timestamp>/`
- Runtime profiles, plan documents, and governance decisions archived for traceability

**Retention:** Keep regression evidence for ≥3 months; archive development artifacts indefinitely.

### 3.3. Ownership & Escalation Matrix

| Component | Owner Initiative | Escalation Triggers | Contact Point |
|:----------|:-----------------|:--------------------|:--------------|
| **Integration Test Harness** | TEST-PYTORCH-001 | Test collection errors, pytest fixture failures | `docs/fix_plan.md` [TEST-PYTORCH-001] |
| **PyTorch Backend Implementation** | INTEGRATE-PYTORCH-001 | Runtime >90s, checkpoint loading failures, decoder shape mismatches, POLICY-001 violations | `docs/fix_plan.md` [INTEGRATE-PYTORCH-001] |
| **Config Bridge Parity** | INTEGRATE-PYTORCH-001 | TensorFlow→PyTorch dataclass translation errors, missing field mappings | `ptycho_torch/config_bridge.py` maintainer |
| **Backend Dispatcher** | INTEGRATE-PYTORCH-001 | Routing failures, silent fallbacks, invalid backend handling | `ptycho/workflows/backend_selector.py` maintainer (if present) |
| **Checkpoint Persistence** | INTEGRATE-PYTORCH-001 | Hyperparameter serialization failures, cross-backend load errors | `docs/fix_plan.md` [INTEGRATE-PYTORCH-001-D1C] |

### 3.4. Escalation Triggers (Critical Thresholds)

Automated monitoring MUST flag the following conditions for immediate investigation:

| Trigger ID | Condition | Severity | Notification Target | Response SLA |
|:-----------|:----------|:---------|:--------------------|:-------------|
| **RT-001** | Integration workflow runtime >90s | CRITICAL | TEST-PYTORCH-001 owner | <4 hours |
| **RT-002** | Integration workflow runtime >60s (3 consecutive runs) | WARNING | TEST-PYTORCH-001 owner | <24 hours |
| **RT-003** | Integration workflow runtime <20s | WARNING | INTEGRATE-PYTORCH-001 owner | <24 hours (incomplete execution suspected) |
| **FAIL-001** | Integration workflow test FAILED status | CRITICAL | Both initiatives | <2 hours |
| **FAIL-002** | Backend selection suite any FAILED | CRITICAL | INTEGRATE-PYTORCH-001 owner | <2 hours |
| **FAIL-003** | Checkpoint loading failure (TypeError on config kwargs) | HIGH | INTEGRATE-PYTORCH-001 owner | <4 hours |
| **POLICY-001** | PyTorch ImportError in `tests/torch/` (local dev) | HIGH | Developer + TEST-PYTORCH-001 owner | Immediate (install torch>=2.2) |
| **CONFIG-001** | Shape mismatch errors (gridsize sync) | HIGH | INTEGRATE-PYTORCH-001 owner | <4 hours |
| **FORMAT-001** | NPZ transpose IndexError | MEDIUM | INTEGRATE-PYTORCH-001 owner | <24 hours |
| **PARITY-001** | Decoder shape mismatch (tensor dimension errors) | HIGH | INTEGRATE-PYTORCH-001 owner | <4 hours |
| **ARTIF-001** | Missing checkpoint artifacts (hyper_parameters key) | HIGH | INTEGRATE-PYTORCH-001 owner | <4 hours |
| **ARTIF-002** | Reconstruction PNG files <1KB or missing | MEDIUM | INTEGRATE-PYTORCH-001 owner | <24 hours |

**Automated Alert Logic (CI Integration):**

```python
# Pseudo-code for CI monitoring hook
if runtime > 90:
    alert("RT-001: CRITICAL - Integration test timeout", severity="critical")
elif runtime > 60:
    if consecutive_count(runtime > 60) >= 3:
        alert("RT-002: WARNING - Sustained runtime degradation", severity="warning")
elif runtime < 20:
    alert("RT-003: WARNING - Suspiciously fast execution", severity="warning")

if test_status == "FAILED":
    if "checkpoint" in error_message and "TypeError" in error_message:
        alert("FAIL-003: Checkpoint loading regression", severity="high", owner="INTEGRATE-PYTORCH-001")
    elif "shape" in error_message or "dimension" in error_message:
        alert("PARITY-001: Decoder shape mismatch", severity="high", owner="INTEGRATE-PYTORCH-001")
    else:
        alert("FAIL-001: Integration workflow failure", severity="critical", owner="both")
```

**Reference:** Runtime thresholds from `plans/active/TEST-PYTORCH-001/reports/2025-10-19T193425Z/phase_d_hardening/runtime_profile.md` §3.1.

### 3.5. Escalation Workflow

When regression failures occur:

1. **Reproduce Locally:**
   ```bash
   CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_integration_workflow_torch.py::test_run_pytorch_train_save_load_infer -vv | tee failure.log
   ```

2. **Capture Environment:**
   ```bash
   python -V > env.txt
   pip show torch lightning >> env.txt
   lscpu | grep -E "Model name|CPU\(s\)" >> env.txt
   ```

3. **Document Failure:**
   - Error message + stack trace
   - Runtime (if timeout)
   - Diff from known-good baseline (runtime, artifact sizes, checkpoint keys)
   - Match against escalation trigger ID from §3.4 table

4. **File Issue:**
   - Append to `docs/fix_plan.md` Attempts history for owning initiative (per §3.3 ownership matrix + §3.4 trigger target)
   - Create timestamped artifact directory under `plans/active/<initiative>/reports/<timestamp>/escalation/`
   - Include: `failure.log`, `env.txt`, pytest selector, reproduction command, trigger ID

5. **Reference Authorities:**
   - Runtime issues: `plans/active/TEST-PYTORCH-001/reports/2025-10-19T193425Z/phase_d_hardening/runtime_profile.md`
   - Backend config issues: `specs/ptychodus_api_spec.md` §4.8
   - Checkpoint issues: `docs/workflows/pytorch.md` §6 (checkpoint management)
   - Escalation trigger definitions: This document §3.4

---

## 4. Policy & Contract Reminders

### 4.1. POLICY-001 (PyTorch Requirement)

**Summary:** PyTorch `>=2.2` is **mandatory** for PtychoPINN as of Phase F (INTEGRATE-PYTORCH-001). All code in `ptycho_torch/` and `tests/torch/` assumes PyTorch is installed.

**Evidence:** `plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T184624Z/governance_decision.md`
**Reference:** `docs/findings.md#POLICY-001`

**Implications:**
- PyTorch listed in `setup.py` `install_requires`
- Tests in `tests/torch/` automatically skipped in TensorFlow-only CI via `tests/conftest.py` collection rules
- Local development expects PyTorch present; missing imports raise actionable `ImportError`

### 4.2. FORMAT-001 (NPZ Auto-Transpose Guard)

**Summary:** Legacy NPZ datasets use `(H,W,N)` diffraction format instead of canonical `(N,H,W)`. Auto-transpose heuristic detects and corrects at runtime.

**Evidence:** `plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T230724Z/callchain/summary.md`
**Reference:** `docs/findings.md#FORMAT-001`

**Affected Code:**
- `ptycho_torch/dataloader.py::_get_diffraction_stack` (lines 118-127)
- `ptycho_torch/dataloader.py::npz_headers` (lines 75-81)

**Test Coverage:** `tests/torch/test_dataloader.py::TestDataloaderFormatAutoTranspose`

### 4.3. CONFIG-001 (Legacy Sync Requirement)

**Summary:** Modern dataclasses MUST propagate to `params.cfg` via `update_legacy_dict` before any legacy module imports.

**Reference:** `docs/debugging/QUICK_REFERENCE_PARAMS.md` §Golden Rule, `CLAUDE.md` §4.1

**Failure Mode:** Silent shape mismatches, parameter drift in data grouping, physics layer inconsistencies

---

## 5. Open Questions & Future Work

### 5.1. CI Environment Matrix

**Question:** Should PyTorch integration tests run on both CPU-only and CUDA-enabled CI workers?

**Current State:** Test enforces CPU-only via `CUDA_VISIBLE_DEVICES=""` fixture. CUDA execution optional/untested.

**Recommendation:** Start with CPU-only CI gate; add CUDA smoke test (30s timeout) as nightly/weekly job once CPU regression stable.

### 5.2. Additional Pytest Markers

**Question:** Should integration workflow test be marked `@pytest.mark.integration` and `@pytest.mark.slow` for selective CI execution?

**Current State:** Test uses standard pytest fixtures (no custom markers).

**Recommendation:** Add markers in Phase D3:
```python
@pytest.mark.integration  # Integration-tier test
@pytest.mark.slow         # Runtime >30s
def test_run_pytorch_train_save_load_infer(...):
    ...
```

### 5.3. Native PyTorch Reassembly

**Question:** When should TensorFlow `tf_helper.reassemble_position` dependency be replaced with native PyTorch implementation?

**Current State:** Phase D2.C MVP uses TensorFlow reassembly for exact parity (lines 706-713 in `ptycho_torch/workflows/components.py::_reassemble_cdi_image_torch`).

**Deferred Rationale:** Parity priority over performance; native implementation could reduce inference runtime by ~30-40%.

**Tracking:** Future INTEGRATE-PYTORCH-001 phase or separate initiative.

---

## 6. References & Cross-Links

### 6.1. Normative Specifications

- `specs/ptychodus_api_spec.md` §4.8 — Backend Selection & Dispatch
- `specs/data_contracts.md` §1 — NPZ format requirements (diffraction=amplitude, float32)

### 6.2. Workflow Guides

- `docs/workflows/pytorch.md` §11 — Regression Test & Runtime Expectations
- `docs/workflows/pytorch.md` §12 — Backend Selection in Ptychodus Integration

### 6.3. Implementation Plans

- `plans/active/TEST-PYTORCH-001/implementation.md` — Phased regression plan (A→D)
- `plans/active/INTEGRATE-PYTORCH-001/phase_e_integration.md` — Backend integration phases

### 6.4. Evidence Archives

- `plans/active/TEST-PYTORCH-001/reports/2025-10-19T193425Z/phase_d_hardening/runtime_profile.md` — Comprehensive runtime analysis
- `plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-19T201500Z/phase_d2_completion/parity_update.md` — PyTorch parity achievement summary

### 6.5. Test Source Files

- `tests/torch/test_integration_workflow_torch.py` — Main integration regression (lines 1-200)
- `tests/torch/test_backend_selection.py` — Backend routing validation (lines 59-170)
- `tests/torch/test_config_bridge.py` — TensorFlow→PyTorch config translation
- `tests/torch/test_lightning_checkpoint.py` — Hyperparameter serialization (Phase D1c)

### 6.6. Ledger & Findings

- `docs/fix_plan.md` [INTEGRATE-PYTORCH-001-STUBS] — Backend implementation attempts history
- `docs/fix_plan.md` [TEST-PYTORCH-001] — Regression test development attempts history
- `docs/findings.md` — POLICY-001, FORMAT-001, CONFIG-001 authority

---

## 7. Handoff Checklist

When assuming ownership of TEST-PYTORCH-001 Phase D3, verify:

- [ ] PyTorch >=2.2 installed locally (`pip show torch`)
- [ ] Dataset available at `datasets/sim_data/Run1084_recon3_postPC_shrunk_3.npz` (35 MB)
- [ ] Integration workflow test passes locally (≤90s runtime)
- [ ] Backend selection suite passes (4-5 tests, <5s runtime)
- [ ] Familiar with CONFIG-001 requirement (`update_legacy_dict` before data loading)
- [ ] Reviewed runtime guardrails (§2.3) and escalation matrix (§3.3)
- [ ] Identified CI environment requirements (CPU-only, env vars)
- [ ] Read Phase D1 runtime profile for baseline expectations

**Ready State:** All checkboxes completed, first PR with PyTorch integration test merged and CI green.

---

**Document Revision:** 2025-10-19
**Authored By:** INTEGRATE-PYTORCH-001 (Phase E3.D handoff loop)
**Maintained By:** TEST-PYTORCH-001 (Phase D3 onwards)
