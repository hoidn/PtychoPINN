# Phase D2.B — PyTorch Training Path Implementation Summary

**Date:** 2025-10-17
**Initiative:** INTEGRATE-PYTORCH-001
**Phase:** D2.B (Implement training orchestration)
**Status:** ✅ COMPLETE (TDD green phase — stub implementation validated)

---

## Summary

Successfully implemented PyTorch training orchestration following TDD methodology. Delivered `_ensure_container` helper and `train_cdi_model_torch` function that normalizes input data via Phase C adapters and provides Lightning delegation hooks. Implementation is torch-optional and maintains API parity with TensorFlow baseline.

---

## Implementation Deliverables

### 1. `_ensure_container` Helper (`ptycho_torch/workflows/components.py:173-244`)
Factory function normalizing all input data types to `PtychoDataContainerTorch`:
- **RawData** → wrap with RawDataTorch → generate_grouped_data → PtychoDataContainerTorch
- **RawDataTorch** → generate_grouped_data → PtychoDataContainerTorch
- **PtychoDataContainerTorch** → return as-is (already normalized)

**Design Decision:** Mirrors TensorFlow `create_ptycho_data_container` pattern while delegating to Phase C adapters. Enforces CONFIG-001 compliance by passing config to RawDataTorch constructor.

### 2. `_train_with_lightning` Stub (`ptycho_torch/workflows/components.py:247-287`)
Orchestration stub for Lightning trainer:
- Accepts normalized containers + config
- Returns minimal results dict (history, train_container, test_container)
- **Stub Status:** Returns placeholder loss trajectory without actual training
- **Future Work:** Phase D2.B follow-up will implement full Lightning.Trainer orchestration

### 3. `train_cdi_model_torch` Entry Point (`ptycho_torch/workflows/components.py:290-347`)
Public API matching TensorFlow `train_cdi_model`:
- Normalizes train/test data via `_ensure_container`
- Delegates to `_train_with_lightning` stub
- Defers probe initialization (noted as TODO for full impl)
- Torch-optional (importable without PyTorch)

---

## Test Coverage

### Red Phase
- Authored `TestWorkflowsComponentsTraining::test_train_cdi_model_torch_invokes_lightning` (tests/torch/test_workflows_components.py:170-330)
- Used monkeypatch spies to validate orchestration contract
- Verified NotImplementedError before implementation (Phase D2.A baseline)

### Green Phase
- Updated test to assert on spy calls instead of expecting error
- **Validation Points:**
  1. `_ensure_container` called with train_data + config ✅
  2. `_train_with_lightning` invoked with containers + config ✅
  3. Results dict includes history, train_container, test_container ✅

**Test Selector:**
```bash
pytest tests/torch/test_workflows_components.py::TestWorkflowsComponentsTraining::test_train_cdi_model_torch_invokes_lightning -vv
```

**Test Result:** ✅ PASSED (1/1 in 3.64s)

---

## Regression Check

**Full Suite Run:**
```bash
pytest tests/ --ignore=tests/test_benchmark_throughput.py --ignore=tests/test_run_baseline.py -v
```

**Results:**
- 190 PASSED ✅
- 13 SKIPPED (expected: torch runtime absent, data files missing, deprecated APIs)
- 17 WARNINGS (UserWarning about test_data_file in config bridge, acceptable)
- 0 NEW FAILURES ✅

**Pre-existing broken modules** (not in scope for this loop, tracked in LEGACY-TESTS-001):
- `tests/test_benchmark_throughput.py` (ModuleNotFoundError: scripts.benchmark_inference_throughput)
- `tests/test_run_baseline.py` (ModuleNotFoundError: tests.test_utilities)

---

## Key Design Decisions

### 1. Stub vs Full Lightning Implementation
**Decision:** Implement orchestration skeleton with stub Lightning call for Phase D2.B TDD validation. Full Lightning.Trainer integration deferred to follow-up loop.

**Rationale:**
- Enables fast TDD cycle (unit test runs in <4s without GPU deps)
- Validates API contract and Phase C adapter integration
- Reduces scope to keep loop focused on orchestration structure
- Aligns with input.md guidance ("stub Lightning delegation")

### 2. Probe Initialization Deferred
**Decision:** Skip `probe.set_probe_guess()` equivalent in stub implementation.

**Rationale:**
- TensorFlow baseline uses global `ptycho.probe` module (Phase C.A TODO item per parity_map.md)
- PyTorch probe management strategy needs architectural decision (Open Question in phase_d2_training_analysis.md)
- Stub validation doesn't require probe state

**Follow-up:** Phase D.B.2 or Phase E should resolve probe handling approach (global vs module-local).

### 3. Torch-Optional Import Strategy
**Decision:** Guard Phase C adapter imports with try/except + TORCH_AVAILABLE flag; raise ImportError in `_ensure_container` if unavailable.

**Rationale:**
- Maintains module-level importability per tests/conftest.py whitelist
- Aligns with existing config_bridge + data_container_bridge patterns
- Defers torch requirement to actual usage site (function call)

---

## Artifacts Generated

| Artifact | Path | Purpose |
| --- | --- | --- |
| Green pytest log | `plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T094500Z/pytest_green.log` | Validates test passes after implementation |
| Full suite regression | `/tmp/pytest_full_suite.log` | Confirms no new test failures |
| Implementation summary | This file | Documents deliverables, decisions, and status |

---

## Next Steps (Phase D2.C)

1. **Inference Path** — Implement `load_inference_bundle_torch` and optional stitching delegation per `phase_d_workflow.md` D2.C.
2. **Full Lightning Integration** — Replace `_train_with_lightning` stub with actual Lightning.Trainer orchestration:
   - Import `ptycho_torch.train.PtychoPINN_Lightning` (torch-optional guarded)
   - Configure Trainer with `max_epochs=config.nepochs`, `devices`, `gradient_clip_val`
   - Execute `trainer.fit(model, train_dataloader)`
   - Extract history from `trainer.callback_metrics`
3. **Probe Management** — Resolve Open Question on probe initialization strategy for PyTorch.
4. **MLflow Toggle** — Add config-driven MLflow disable switch for CI (per phase_d2_training_analysis.md).

---

## Exit Criteria Status

**Phase D2.B Checklist (from phase_d_workflow.md):**
- [x] Scaffold orchestration module (D2.A) — **COMPLETE** (Attempt #43)
- [x] Implement training path (D2.B) — **COMPLETE** (This attempt #44)
  - [x] `_ensure_container` helper implemented
  - [x] Lightning delegation stub implemented
  - [x] TDD red→green cycle validated
  - [x] Torch-optional behavior preserved
  - [x] Full test suite regression passed
- [ ] Implement inference + stitching path (D2.C) — **PENDING** (Next loop)

**Phase D2.B Ready for Closure:** ✅ YES — Stub implementation validates orchestration API; full Lightning execution deferred to follow-up per input.md guidance.

---

## References

- **TensorFlow Baseline:** `ptycho/workflows/components.py:543-609` (create_ptycho_data_container, train_cdi_model)
- **Phase C Adapters:** RawDataTorch (`ptycho_torch/raw_data_bridge.py:1-324`), PtychoDataContainerTorch (`ptycho_torch/data_container_bridge.py:1-280`)
- **Config Bridge:** Phase B output (`ptycho_torch/config_bridge.py`) for TensorFlow dataclass compatibility
- **Training Analysis:** `plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T093500Z/phase_d2_training_analysis.md`
- **Phase D Plan:** `plans/active/INTEGRATE-PYTORCH-001/phase_d_workflow.md`
