# Phase D3 — CI Integration Strategy (TEST-PYTORCH-001)

**Date:** 2025-10-19
**Phase:** Phase D3 — CI Integration & Follow-up Gates
**Artifact Hub:** `plans/active/TEST-PYTORCH-001/reports/2025-10-19T232500Z/phase_d_hardening/`

---

## Executive Summary

This document defines the CI integration strategy for the PyTorch integration regression test (`test_run_pytorch_train_save_load_infer`) based on Phase D3 analysis. The project **currently has no GitHub workflows configured** (`.github/workflows/` directory does not exist), so this document provides **CI-ready guidance** for when automated testing infrastructure is implemented.

**Key Decision:** The test is **ready for CI integration** with pytest markers and documented execution requirements. Follow-up automation work is tracked separately in the fix plan.

---

## D3.A — CI Infrastructure Assessment

### Current State

**Finding:** No GitHub Actions workflows exist in this repository.

**Evidence:**
```bash
$ ls .github/workflows 2>/dev/null
=== No .github/workflows directory ===

$ find . -name "*.yml" -o -name "*.yaml" | grep -E "(github|workflows|ci)"
# No CI configuration files found
```

**Pytest Configuration Analysis:**

1. **Primary Test Config:** `pyproject.toml` lines 56-80
   - Configured markers: `torch`, `optional`, `slow`, `mvp`
   - Test collection: `testpaths = ["tests"]`
   - Existing addopts: `--strict-markers`, `--strict-config`, `-ra`

2. **Pytest Fixtures:** `tests/conftest.py`
   - Lines 21-23: Custom markers registered (`torch`, `optional`, `slow`)
   - Lines 25-47: PyTorch availability detection via `pytest_collection_modifyitems`
   - **Behavior:** Tests in `tests/torch/` automatically **skip** in TF-only CI environments
   - **Policy:** No GPU tests; CPU-only execution enforced via test fixtures

3. **Test Suite Organization:**
   - TensorFlow tests: `tests/test_*.py` (core suite)
   - PyTorch tests: `tests/torch/test_*.py` (isolated directory)
   - Integration tier: Documented in `docs/TESTING_GUIDE.md` lines 24-56

**Reference:** `docs/development/TEST_SUITE_INDEX.md` lines 69-76 (Torch Tests section)

### Existing Test Markers

The PyTorch integration test **should use** the following markers per existing project conventions:

- `@pytest.mark.torch` — Already implicit via directory-based collection
- `@pytest.mark.integration` — Recommended (mirrors TF integration test `test_integration_workflow.py:26`)
- `@pytest.mark.slow` — Recommended (35.92s runtime exceeds typical unit test duration)

**Rationale:** These markers enable selective execution in CI (e.g., `pytest -m "not slow"` for fast feedback loops).

---

## D3.B — Execution Strategy & Runtime Guardrails

### Recommended Test Selector

```bash
CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_integration_workflow_torch.py::test_run_pytorch_train_save_load_infer -vv
```

**Critical Environment Variables:**
- `CUDA_VISIBLE_DEVICES=""` — Enforces CPU-only execution (per `cuda_cpu_env` fixture contract)
- `PYTHONPATH=.` — Optional; project should be installed in editable mode (`pip install -e .`)

**Alternative Selectors (when CI implemented):**
```bash
# Run all PyTorch integration tests
pytest tests/torch/ -m integration -vv

# Run all integration tests (TensorFlow + PyTorch)
pytest -m integration -vv

# Skip slow tests for fast CI feedback
pytest -m "not slow" -vv
```

### Runtime Budget & Timeout

| Metric | Value | Source |
|:-------|:------|:-------|
| **Baseline Runtime (Observed)** | 35.92s ± 0.06s | Phase D1 runtime profile §1.2 |
| **Warning Threshold** | 60s | 1.7× baseline (investigate if exceeded) |
| **Maximum CI Timeout** | 90s | 2.5× baseline (allow for slower CI hardware) |
| **Conservative Timeout** | 120s | 3.3× baseline (recommended for initial CI setup) |

**Recommendation:** Start with **120s timeout + 1 retry** on first CI integration, then tune downward to 90s after observing stable performance.

### Skip Conditions & Dependencies

**Required Dependencies (enforced at test collection time):**
- PyTorch >= 2.2 (POLICY-001 per `docs/findings.md#POLICY-001`)
- Lightning >= 2.0
- Dataset: `datasets/Run1084_recon3_postPC_shrunk_3.npz` (35 MB)
  - **Path:** `project_root / "datasets" / "Run1084_recon3_postPC_shrunk_3.npz"`
  - **Fixture:** `data_file` in `test_integration_workflow_torch.py:51-58`

**Skip Behavior (existing `conftest.py` logic):**
- If PyTorch unavailable in **TF-only CI**: Skip entire `tests/torch/` directory (lines 25-47)
- If PyTorch unavailable in **local dev**: Fail with actionable `ImportError` message (POLICY-001)

**No additional skip guards needed** — existing pytest infrastructure handles dependency detection.

### Pytest Markers (Recommended Addition)

**Proposed modification to `tests/torch/test_integration_workflow_torch.py`:**

```python
@pytest.mark.torch       # Already implicit via directory, but explicit is clearer
@pytest.mark.integration # Aligns with TensorFlow integration test
@pytest.mark.slow        # Runtime >30s justifies marker
def test_run_pytorch_train_save_load_infer(tmp_path, data_file, cuda_cpu_env):
    """
    Phase C PyTorch Integration Workflow Test
    ...
    """
```

**Justification:**
1. `@pytest.mark.integration` — Enables CI to run integration tier separately from unit tests
2. `@pytest.mark.slow` — Allows fast feedback via `pytest -m "not slow"` in development
3. `@pytest.mark.torch` — Redundant with directory-based collection but improves discoverability

**Impact:** These markers enable flexible CI scheduling without modifying test logic.

---

## D3.C — Follow-Up Actions & Open Questions

### Immediate Actions (This Loop)

- [x] Document CI integration strategy (this file)
- [x] Define pytest selectors and timeout guardrails
- [x] Validate marker conventions against existing test suite

### Follow-Up Work (Future Loops / Tickets)

#### FU-001: Add Pytest Markers to Integration Test

**Description:** Add `@pytest.mark.integration` and `@pytest.mark.slow` decorators to `test_run_pytorch_train_save_load_infer` in `tests/torch/test_integration_workflow_torch.py`.

**Acceptance Criteria:**
- Test function annotated with markers (lines ~171-173 in test file)
- Markers execute correctly: `pytest -m integration -vv` selects the test
- Documentation updated in test docstring referencing marker purpose

**Priority:** Low (nice-to-have; test runs correctly without markers)

**Owner:** Future contributor or next TEST-PYTORCH-001 iteration

**Estimated Effort:** 5 minutes (single-line addition + commit)

---

#### FU-002: Implement GitHub Actions CI Workflow

**Description:** Create `.github/workflows/pytest-torch.yml` to run PyTorch integration tests on pull requests and main branch commits.

**Scope:**
- Configure Ubuntu 22.04 runner with Python 3.11
- Install dependencies: `pip install -e .` (includes torch>=2.2 per setup.py)
- Download/cache dataset fixture (35 MB)
- Execute: `CUDA_VISIBLE_DEVICES="" pytest tests/torch/ -m integration -vv --timeout=120`
- Upload pytest logs as artifacts on failure

**Acceptance Criteria:**
- CI job executes on PR open/sync and main branch pushes
- Test passes in ≤90s (nominal hardware)
- CI fails loudly if test exceeds 120s timeout
- Artifact upload configured for debugging failures

**Priority:** Medium (blocks CI-gated PyTorch regression enforcement)

**Owner:** DevOps / CI maintainer (requires repo admin access for workflow setup)

**Estimated Effort:** 2-4 hours (workflow authoring + dataset caching + debugging)

**References:**
- GitHub Actions pytest example: https://docs.pytest.org/en/stable/how-to/usage.html#ci-integration
- Dataset caching: GitHub Actions `actions/cache@v3`
- Timeout enforcement: `pytest-timeout` plugin (optional; runner timeout sufficient)

---

#### FU-003: Add Integration Test to TEST_SUITE_INDEX.md

**Description:** Update `docs/development/TEST_SUITE_INDEX.md` to document the PyTorch integration test selector and runtime expectations.

**Proposed Entry (Torch Tests table, line ~76):**

```markdown
| `test_integration_workflow_torch.py` | Validates the full PyTorch train → save → load → infer workflow using subprocesses, mirroring TensorFlow integration test. Exercises Lightning training, checkpoint persistence, and image reassembly. | `test_run_pytorch_train_save_load_infer` | `CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_integration_workflow_torch.py::test_run_pytorch_train_save_load_infer -vv` | Critical integration coverage for PyTorch backend. Runtime: ~36s CPU-only. |
```

**Acceptance Criteria:**
- Entry added to TEST_SUITE_INDEX.md
- Selector command tested and confirmed working
- Runtime guidance reflects Phase D1 profile (36s ± 5s)

**Priority:** Low (documentation hygiene)

**Owner:** Next documentation pass

**Estimated Effort:** 10 minutes

---

### Open Questions (None)

All D3 questions resolved during analysis. No blockers identified for CI integration when infrastructure is implemented.

---

## Recommendations Summary

### For Immediate CI Setup (When `.github/workflows/` Implemented)

1. **Test Selector:**
   ```bash
   CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_integration_workflow_torch.py::test_run_pytorch_train_save_load_infer -vv
   ```

2. **Timeout Configuration:** 120s (conservative), tune to 90s after observing stability

3. **Environment Requirements:**
   - Python 3.11
   - PyTorch >= 2.2 (installed via `pip install -e .`)
   - Dataset: `datasets/Run1084_recon3_postPC_shrunk_3.npz` (35 MB, cached)
   - CPU-only: `CUDA_VISIBLE_DEVICES=""`

4. **Retry Policy:** 1 retry on timeout (account for CI jitter)

5. **Pytest Markers (Optional):** Add `@pytest.mark.integration` + `@pytest.mark.slow` to test function (FU-001)

### For Test Suite Maintainers

- **No urgent action required** — Test is CI-ready with current implementation
- **Optional enhancement:** Add markers per FU-001 for flexible CI scheduling
- **Documentation update:** Add entry to TEST_SUITE_INDEX.md per FU-003

---

## Compliance Checklist

| Requirement | Status | Evidence |
|:------------|:-------|:---------|
| Existing CI infrastructure assessed | ✅ | No `.github/workflows/` directory; pytest markers configured in `pyproject.toml` |
| Pytest selector documented | ✅ | `CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_integration_workflow_torch.py::test_run_pytorch_train_save_load_infer -vv` |
| Runtime guardrails defined | ✅ | 90s max, 120s conservative, 60s warning threshold (Phase D1 profile) |
| Skip conditions clarified | ✅ | Auto-skip in TF-only CI via `conftest.py`; fail-fast in local dev per POLICY-001 |
| Markers recommended | ✅ | `@pytest.mark.integration` + `@pytest.mark.slow` (FU-001) |
| Follow-up tickets authored | ✅ | FU-001 (markers), FU-002 (CI workflow), FU-003 (docs index) |
| Open questions resolved | ✅ | Zero blockers; CI-ready when infrastructure implemented |

**Phase D3 Status:** **COMPLETE** — All D3.A, D3.B, D3.C deliverables satisfied.

---

## Artifact Inventory

| Artifact | Purpose |
|:---------|:--------|
| `ci_notes.md` | This file — CI integration strategy + follow-up tracking |
| `summary.md` | Phase D3 completion narrative (to be authored) |

**Artifact Hub:** `plans/active/TEST-PYTORCH-001/reports/2025-10-19T232500Z/phase_d_hardening/`

---

## References

- **Phase D3 Plan:** `plans/active/TEST-PYTORCH-001/reports/2025-10-19T193425Z/phase_d_hardening/plan.md` (D3.A–D3.C tasks)
- **Runtime Profile:** `plans/active/TEST-PYTORCH-001/reports/2025-10-19T193425Z/phase_d_hardening/runtime_profile.md` (guardrails)
- **Pytest Config:** `pyproject.toml` lines 56-80, `tests/conftest.py` lines 17-47
- **Test Suite Index:** `docs/development/TEST_SUITE_INDEX.md` lines 69-76
- **POLICY-001:** `docs/findings.md#POLICY-001` (PyTorch mandatory)
- **TensorFlow Integration Baseline:** `tests/test_integration_workflow.py` (marker reference)
- **Testing Guide:** `docs/TESTING_GUIDE.md` lines 24-56 (integration tier expectations)
