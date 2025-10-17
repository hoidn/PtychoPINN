# PtychoPINN Fix Plan Ledger

**Last Updated:** 2025-10-16
**Active Focus:** Restore pytest signal; PyTorch integration follows once green baseline returns.


---

## [TEST-SUITE-TRIAGE] Restore pytest signal and triage failures
- Spec/AT: `docs/TESTING_GUIDE.md`, `docs/debugging/debugging.md`, `specs/data_contracts.md`
- Plan: `plans/active/TEST-SUITE-TRIAGE/plan.md`
- Priority: Critical (Do Now)
- Status: done
- Owner/Date: Ralph/2025-10-16
- Reproduction: `pytest tests/ -vv`
- Attempts History:
  * [2025-10-16] Attempt #0 — Planning: Established multi-phase triage plan, awaiting execution. (Galph)
  * [2025-10-16T23:05:39Z] Attempt #1 — Phase A Execution: GREEN BASELINE ✅
    - **Result:** 153 passed, 12 skipped (intentional), 0 failed
    - **Exit Code:** 0 (success)
    - **Execution Time:** 201.42s (3m 21s)
    - **Artifacts:** `plans/active/TEST-SUITE-TRIAGE/reports/2025-10-16T230539Z/`
      - `pytest.log` (full verbose output)
      - `env.md` (environment metadata)
      - `requirements.txt` (package snapshot)
      - `summary.md` (comprehensive analysis)
    - **Key Finding:** Test suite is in excellent health with zero failures. All skips are intentional and documented (PyTorch env issue, missing optional datasets, TF Addons migration, documented deprecations).
    - **Phase B/C:** NOT REQUIRED — No failures to classify or remediate.
    - **First Divergence:** N/A (baseline is clean)
    - **Next Actions:** Mark initiative complete; proceed with PyTorch integration work.
- Exit Criteria: ✅ COMPLETE
  - ✅ Pytest sweep captured in timestamped report with failure ledger per Phase A.
  - ✅ Phase B/C not needed (zero failures)
  - ✅ docs/fix_plan.md updated with findings.


---

## [TEST-PYTORCH-001] Build Minimal Test Suite for PyTorch Backend
- Spec/AT: Corresponds to existing TensorFlow integration test `tests/test_integration_workflow.py` and guidance in `plans/pytorch_integration_test_plan.md`.
- Priority: Critical
- Status: pending
- Owner/Date: Codex Agent/2025-10-16
- Reproduction: N/A (new feature)
- Attempts History:
  * [2025-10-16] Attempt #0 — Planning: Initial task creation.
- Exit Criteria:
  - A new test file `tests/torch/test_integration_workflow.py` exists.
  - The test successfully runs a minimal train -> save -> load -> infer cycle using the PyTorch backend.
  - The test passes, confirming the basic viability of the PyTorch persistence layer.

## [INTEGRATE-PYTORCH-001] Prepare for PyTorch Backend Integration with Ptychodus
- Spec/AT: `specs/ptychodus_api_spec.md` and `plans/ptychodus_pytorch_integration_plan.md`.
- Priority: High
- Status: pending
- Owner/Date: Codex Agent/2025-10-16
- Reproduction: N/A (new feature)
- Attempts History:
  * [2025-10-16] Attempt #0 — Planning: Initial task creation.
- Exit Criteria:
  - All gaps identified in the "TensorFlow ↔ PyTorch Parity Map" within `plans/ptychodus_pytorch_integration_plan.md` are addressed with a concrete implementation plan.
  - A `RawDataTorch` shim and `PtychoDataContainerTorch` class are implemented.
  - Configuration parity (Phase 1 of the integration plan) is complete and tested.

---

## [VECTOR-TRICUBIC-001] Vectorize tricubic interpolation & detector absorption (nanoBragg PyTorch)
- Spec/AT: `../nanoBragg/specs/spec-a.md`, `../nanoBragg/tests/test_at_str_002.py`, `../nanoBragg/tests/test_at_abs_001.py`
- Plan: <doc-ref type="plan">plans/active/vectorization.md</doc-ref>
- Priority: High
- Status: pending
- Owner/Date: Galph/2025-10-17
- Reproduction:
  - `cd ../nanoBragg && pytest tests/test_at_str_002.py -vv`
  - `cd ../nanoBragg && pytest tests/test_at_abs_001.py -vv`
- Attempts History:
  * [2025-10-17T00:34:06Z] Attempt #0 — Planning: Authored phased vectorization plan; baseline characterization pending.
- Exit Criteria:
  - Baseline performance + acceptance artifacts recorded under `plans/active/VECTOR-TRICUBIC/reports/`.
  - New RED tests capture batched tricubic and absorption vectorization requirements.
  - Vectorized implementations pass GREEN tests (`pytest tests/test_tricubic_vectorized.py tests/test_at_str_002.py tests/test_at_abs_001.py -vv`).
  - Documented ≥2× speedup on CPU and GPU with knowledge base update.
