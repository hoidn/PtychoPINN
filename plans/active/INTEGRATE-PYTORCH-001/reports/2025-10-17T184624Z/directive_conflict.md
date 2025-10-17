# Phase F1.1 — Directive Conflict Analysis: Torch-Optional to Torch-Required Transition

## Executive Summary

This document catalogues the current torch-optional directive footprint across the PtychoPINN2 codebase and articulates the conflict between maintaining torch-optional execution pathways and the architectural decision to establish PyTorch as the required backend for Ptychodus integration.

**Date:** 2025-10-17
**Initiative:** INTEGRATE-PYTORCH-001 Phase F
**Conflict Status:** CRITICAL — Policy shift required before Phase F2+ implementation

---

## 1. Current Torch-Optional Directive Inventory

### 1.1 Authoritative Guidance (CLAUDE.md)

**File:** `CLAUDE.md:57-59`
**Level:** Critical directive
**Content:**
```xml
<directive level="critical" purpose="Keep PyTorch parity tests torch-optional">
  Parity and adapter tests must remain runnable when PyTorch is unavailable. Use documented shims or skip rules in `tests/conftest.py`, avoid hard `import torch` statements in modules that only need type aliases, and capture any fallback behavior in the loop artifacts.
</directive>
```

**Impact:** This directive mandates that all PyTorch-related tests and adapters maintain optional execution semantics. It explicitly prohibits hard `import torch` dependencies and requires conftest.py skip logic as the enforcement mechanism.

**Rationale (Historical):** Ensures CI/dev environments without PyTorch can still run core TensorFlow tests; reduces onboarding friction; maintains test suite portability.

---

### 1.2 Test Harness Infrastructure (tests/conftest.py)

**File:** `tests/conftest.py:21-22, 25-47`
**Level:** Implementation enforcement
**Content:**

```python
# Line 21-22: Custom marker registration
config.addinivalue_line("markers", "torch: mark test as requiring PyTorch")

# Line 25-47: Auto-skip logic with whitelist exception
def pytest_collection_modifyitems(config, items):
    """
    Modify test collection to handle optional dependencies.
    This runs after test collection and can modify the collected items.
    """

    # Check what optional dependencies are available
    torch_available = True
    try:
        import torch
    except ImportError:
        torch_available = False

    # Add skip markers for tests requiring unavailable dependencies
    for item in items:
        # Skip torch tests if torch is not available
        # EXCEPTIONS: Some torch/ tests can run without torch (use fallback/stub types)
        TORCH_OPTIONAL_MODULES = ["test_config_bridge", "test_data_pipeline", "test_workflows_components", "test_model_manager", "test_backend_selection"]
        is_torch_optional = any(module in str(item.fspath) for module in TORCH_OPTIONAL_MODULES)

        if ("torch" in str(item.fspath).lower() or item.get_closest_marker("torch")):
            if not torch_available and not is_torch_optional:
                item.add_marker(pytest.mark.skip(reason="PyTorch not available"))
```

**Impact:** Implements a whitelist-based mechanism where:
1. **Default behavior:** All tests in `tests/torch/` are auto-skipped when PyTorch is unavailable
2. **Exceptions:** 5 modules (`test_config_bridge`, `test_data_pipeline`, `test_workflows_components`, `test_model_manager`, `test_backend_selection`) are whitelisted to run without PyTorch via fallback implementations

**Enforcement Surface:** Currently protects 56 tests across the whitelisted modules (as of Attempt #63).

---

### 1.3 Implementation Pattern Examples

#### 1.3.1 Module-Level Guards (ptycho_torch/config_params.py)

**File:** `ptycho_torch/config_params.py:1-13` (from Attempt #15)
**Pattern:**
```python
try:
    import torch
    TORCH_AVAILABLE = True
    TensorType = torch.Tensor
except ImportError:
    TORCH_AVAILABLE = False
    TensorType = type(None)  # Fallback type alias
```

**Usage:** Enables type hints and optional tensor field declarations without hard PyTorch dependency.

#### 1.3.2 Adapter-Level Guards (ptycho_torch/config_bridge.py)

**File:** `ptycho_torch/config_bridge.py:70-78` (from Attempt #15)
**Pattern:**
```python
try:
    from ptycho_torch.config_params import TORCH_AVAILABLE, ...
except ImportError:
    TORCH_AVAILABLE = False

def to_model_config(...):
    # Function executes with or without torch runtime
    # Returns TensorFlow dataclasses regardless of torch availability
    ...
```

**Purpose:** Configuration bridge adapter remains importable and testable in torch-free environments.

#### 1.3.3 Backend Dispatcher Guards (ptycho/workflows/backend_selector.py)

**File:** `ptycho/workflows/backend_selector.py:180-201` (from Attempt #63)
**Pattern:**
```python
if config.backend == 'pytorch':
    try:
        from ptycho_torch.workflows import components as torch_components
    except ImportError as e:
        raise RuntimeError(
            f"PyTorch backend selected (backend='{config.backend}') but PyTorch workflows are unavailable.\n"
            f"Error: {e}\n"
            f"Install PyTorch support with: pip install torch torchvision\n"
            f"Or switch to TensorFlow backend (backend='tensorflow')."
        )
```

**Purpose:** Fail-fast with actionable guidance when user explicitly requests PyTorch but it's unavailable.

---

## 2. Evolution of Torch-Optional Through INTEGRATE-PYTORCH-001

### 2.1 Phase B (Config Bridge)

**Attempts:** #7-#17
**Torch-Optional Requirement:** Tests in `tests/torch/test_config_bridge.py` initially skipped (Attempt #7), later exempted from conftest auto-skip logic (Attempt #15) to enable torch-free test execution.
**Artifacts:** `config_bridge.py` uses `TORCH_AVAILABLE` flag; tests execute and validate adapter logic without torch runtime.

---

### 2.2 Phase C (Data Pipeline)

**Attempts:** #29-#35
**Torch-Optional Requirement:** Data adapters (`RawDataTorch`, `PtychoDataContainerTorch`) implement tensor conversion only when torch available, otherwise return NumPy arrays.
**Key Quote (Attempt #35):**
> "Torch-optional: uses `TORCH_AVAILABLE` flag with guarded imports; returns torch.Tensor when available, NumPy arrays otherwise"

**Rationale:** Ensures data pipeline tests can validate shapes/dtypes/parity without requiring PyTorch installation.

---

### 2.3 Phase D (Workflow Orchestration)

**Attempts:** #43-#49
**Torch-Optional Requirement:** `ptycho_torch/workflows/components.py` uses stub implementations; tests in `test_workflows_components.py` validate config bridge calls and orchestration contracts using monkeypatch spies rather than actual PyTorch execution.
**Artifacts:** All workflow tests added to conftest whitelist (Attempt #43, `test_workflows_components` in line 42).

---

### 2.4 Phase E (Backend Selection)

**Attempts:** #60-#63
**Torch-Optional Requirement:** Backend dispatcher raises `RuntimeError` when PyTorch unavailable but user sets `backend='pytorch'`. Tests in `test_backend_selection.py` validate dispatcher behavior without requiring torch runtime.
**Key Design (Attempt #63):**
> "PyTorch path guards import with try/except, raises RuntimeError when `ptycho_torch.workflows` unavailable"

---

## 3. The Conflict: Architectural Goals vs. Torch-Optional Mandate

### 3.1 Original Intent of Torch-Optional

**Hypothesis (from historical context):**
1. **Development Velocity:** Enable core PtychoPINN2 development to continue in environments without GPU/PyTorch during early prototyping
2. **CI Cost Management:** Avoid requiring PyTorch in every CI runner (TensorFlow-only pipelines remain lightweight)
3. **Progressive Enhancement:** Allow torch/ modules to be developed incrementally without blocking TensorFlow workflows

**Supporting Evidence:**
- Attempt #7 introduced first torch-optional test with XFAIL marker (expected skip)
- Attempt #15 explicitly noted "tests runnable without torch" as Phase A goal
- Conftest whitelist pattern grew organically from 1 module (test_config_bridge) to 5 modules by Attempt #63

---

### 3.2 New Reality: PyTorch as Production Backend

**Initiative Goal (from `plans/ptychodus_pytorch_integration_plan.md`):**
> "Enable Ptychodus to use PyTorch-based PtychoPINN backend as a production-ready alternative to TensorFlow"

**Architectural Shift:**
- **Attempt #63 Achievement:** Backend dispatcher functional with CONFIG-001 compliance
- **Phase D Completion:** All workflow orchestration adapters implemented
- **Phase C Completion:** Data pipeline parity established
- **Phase B Completion:** Configuration bridge fully operational

**Implication:** The PyTorch stack is no longer experimental infrastructure—it's a **production dependency** for the Ptychodus integration use case.

---

### 3.3 The Conflict Articulation

| Aspect | Torch-Optional Directive | Torch-Required Reality |
|:-------|:-------------------------|:-----------------------|
| **Test Execution** | Tests must skip gracefully when PyTorch unavailable | PyTorch is required for Ptychodus backend integration; skipping tests hides production failures |
| **Import Guards** | "Avoid hard `import torch` statements" (CLAUDE.md:58) | Production code in `ptycho_torch.workflows.components` must import `torch.nn.Module`, `lightning.Trainer` to function |
| **CI Philosophy** | Maintain torch-free CI runners for TensorFlow-only validation | Ptychodus integration CI must validate PyTorch workflows; torch-free runners cannot test the production backend |
| **Error Handling** | Graceful degradation with fallbacks | Ptychodus users expect hard failure when backend misconfigured, not silent fallback to TensorFlow |
| **Developer Workflow** | Optional torch installation reduces onboarding friction | PyTorch is now a **core dependency** for the Ptychodus use case; optional installation creates support burden |

---

## 4. Specific Directive Violations Under Torch-Required Policy

### 4.1 Import Statement Prohibition

**CLAUDE.md:58 states:**
> "avoid hard `import torch` statements in modules that only need type aliases"

**Phase F Requirement:**
- `ptycho_torch/model.py` (current implementation at lines 89-93) uses `import torch` unconditionally
- `ptycho_torch/workflows/components.py` (stub at lines 247-287, Attempt #43) will require `import lightning` when implemented
- Production inference path needs `torch.nn.Module.eval()`, `torch.no_grad()` context managers

**Conflict:** Torch-required policy would allow (and necessitate) these hard imports.

---

### 4.2 Conftest Whitelist Expansion

**Current State (tests/conftest.py:42):**
```python
TORCH_OPTIONAL_MODULES = ["test_config_bridge", "test_data_pipeline", "test_workflows_components", "test_model_manager", "test_backend_selection"]
```

**Phase F Requirement:**
- Remove whitelist exceptions
- Make all `tests/torch/` tests assume PyTorch is installed
- Auto-skip logic becomes obsolete for production backend testing

**Conflict:** Torch-optional directive mandates expanding whitelist; torch-required policy mandates removing it entirely.

---

### 4.3 Fallback Behavior in Data Pipeline

**Current Implementation (ptycho_torch/data_container_bridge.py:100-130, Attempt #35):**
```python
if TORCH_AVAILABLE:
    return torch.from_numpy(array).to(dtype=dtype)
else:
    return array.astype(dtype)
```

**Phase F Requirement:**
- Remove `TORCH_AVAILABLE` flag
- Assume `torch.from_numpy` always available
- Fail fast if torch unavailable rather than silently returning NumPy

**Conflict:** Torch-optional pattern enabled parity testing without torch; torch-required removes this capability.

---

## 5. Downstream Dependencies and Risks

### 5.1 TEST-PYTORCH-001 Initiative

**Status:** Pending (depends on INTEGRATE-PYTORCH-001 Phase D completion per docs/fix_plan.md:28-39)
**Goal:** Build minimal test suite for PyTorch backend mirroring `tests/test_integration_workflow.py`

**Impact of Torch-Required Shift:**
- **Positive:** TEST-PYTORCH-001 can assume PyTorch in CI, simplifying subprocess testing
- **Risk:** If TEST-PYTORCH-001 starts before torch-required policy finalized, may duplicate torch-optional patterns that must be refactored later

**Mitigation:** Phase F1 governance approval gates TEST-PYTORCH-001 activation (per `phase_f_torch_mandatory.md:6`).

---

### 5.2 CI/CD Environment Configuration

**Current Assumption:** CI runners may lack PyTorch (evidenced by conftest auto-skip logic)
**Torch-Required Requirement:** All CI runners executing Ptychodus integration tests MUST have PyTorch installed

**Action Required:**
- Update CI configuration (e.g., GitHub Actions, GitLab CI) to include `torch>=2.2` in test environment
- Document PyTorch installation as mandatory in developer setup guides

**Risk:** Breaks existing CI pipelines if torch not installed; increases CI runtime/resource usage.

---

### 5.3 Developer Onboarding

**Current State:** Developers can work on TensorFlow-only features without installing PyTorch
**Torch-Required State:** PyTorch becomes mandatory installation step in README/CLAUDE.md setup instructions

**Mitigation Strategy:**
- Clearly document in CLAUDE.md §5 that PyTorch is required for Ptychodus backend development
- Provide `pip install -e .[torch]` or equivalent extras syntax
- Update verification commands to test PyTorch availability

---

## 6. Precedent from Attempt #63 (Backend Dispatcher)

The Phase E1.C backend dispatcher (Attempt #63) established a **middle ground** that informs Phase F decision-making:

**Pattern Implemented:**
1. **Config-driven backend selection:** `config.backend='pytorch'` explicitly requests torch
2. **Fail-fast on unavailability:** Raises `RuntimeError` with installation guidance when torch unavailable but requested
3. **Default to TensorFlow:** Backwards compatibility preserved for existing workflows

**Key Insight:**
> The dispatcher already establishes that **PyTorch unavailability is an ERROR condition** when user explicitly selects it, not a graceful degradation scenario.

**Implication for Phase F:**
- Torch-required policy extends this pattern: make PyTorch absence an error at import time for `ptycho_torch.*` production modules
- Conftest skip logic becomes outdated—tests should FAIL (not skip) when production dependencies missing

---

## 7. Recommended Conflict Resolution Path

### 7.1 Immediate Actions (Phase F1.2 Governance)

1. **Stakeholder Confirmation:** Document consensus that PyTorch is now a production dependency for Ptychodus backend
2. **Directive Sunset:** Mark CLAUDE.md:57-59 torch-optional directive as superseded by torch-required policy
3. **Risk Acknowledgment:** Accept CI environment changes and developer workflow adjustments as necessary costs

---

### 7.2 Proposed Policy Statement (for Phase F1.3 Guidance Updates)

**New Directive:**
```xml
<directive level="critical" purpose="PyTorch is required for Ptychodus backend">
  PyTorch (torch>=2.2) is a mandatory dependency for the `ptycho_torch/` backend stack and Ptychodus integration workflows. Production modules in `ptycho_torch/` MUST use unconditional `import torch` statements. Test infrastructure in `tests/torch/` MUST assume PyTorch availability and FAIL (not skip) when unavailable. CI/CD pipelines validating Ptychodus integration MUST install PyTorch.
</directive>
```

**Rationale Reference:**
- `plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T184624Z/directive_conflict.md` (this document)
- `plans/active/INTEGRATE-PYTORCH-001/phase_f_torch_mandatory.md` (Phase F plan)

---

## 8. Open Questions for Governance (Phase F1.2)

1. **Q1:** Should TensorFlow-only CI runners continue to exist for legacy validation?
   - **If YES:** Keep conftest auto-skip for `tests/torch/` but remove whitelist exceptions
   - **If NO:** Consolidate to single PyTorch-required CI configuration

2. **Q2:** What is the timeline for deprecating torch-optional behavior in existing modules?
   - **Aggressive (Phase F3):** Remove all `TORCH_AVAILABLE` guards in single loop
   - **Conservative:** Maintain guards for 1-2 releases, deprecation warnings in logs

3. **Q3:** How should we handle environments that cannot install PyTorch (e.g., CPU-only ARM systems)?
   - **Option A:** Fail fast with clear error message directing to TensorFlow backend
   - **Option B:** Maintain minimal fallback for config/data bridges (but remove from workflows)

4. **Q4:** Should we version-gate the torch-required policy?
   - **Suggestion:** Document as breaking change in CHANGELOG, bump to v2.0.0 if following semver

---

## 9. Next Steps

1. **Phase F1.2:** Capture stakeholder decision in `governance_decision.md` addressing all open questions
2. **Phase F1.3:** Draft guidance updates for CLAUDE.md / docs/findings.md with exact wording changes
3. **Phase F2:** Execute comprehensive inventory of torch-optional code paths (modules, tests, config)
4. **Phase F3:** Implement torch-required migration per approved governance decision

---

**Conflict Summary:**
The torch-optional directive (CLAUDE.md:57-59) conflicts with the architectural decision to establish PyTorch as a production backend. Phase F1 resolves this by documenting the conflict, securing governance approval to sunset torch-optional requirements, and updating authoritative guidance before Phase F2 implementation begins.

**File References:**
- CLAUDE.md:57-59 (torch-optional directive)
- tests/conftest.py:21-47 (whitelist implementation)
- ptycho_torch/config_params.py:1-13 (guard pattern)
- ptycho_torch/config_bridge.py:70-78 (adapter guard)
- ptycho/workflows/backend_selector.py:180-201 (dispatcher guard)
- docs/fix_plan.md:80-137 (Phase B-E torch-optional evolution)
