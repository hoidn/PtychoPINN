# TEST-PYTORCH-001 — PyTorch Integration Regression Plan

## Context
- Initiative: TEST-PYTORCH-001 — Author PyTorch integration workflow regression
- Phase Goal: Deliver a CPU-friendly regression that exercises the PyTorch train→save→load→infer pipeline end-to-end and codifies evidence capture/log hygiene so CI guards the backend.
- Dependencies:
  - `plans/pytorch_integration_test_plan.md` (original charter and scope notes)
  - `plans/active/INTEGRATE-PYTORCH-001/phase_e2_implementation.md` (CLI contract, Lightning wiring, MLflow controls)
  - `specs/ptychodus_api_spec.md` §4.5–§4.6 (reconstructor lifecycle contract shared with Ptychodus)
  - `docs/workflows/pytorch.md` §§5–8 (operational workflow, dtype safeguards, artifact expectations)
  - Findings: POLICY-001 (PyTorch required), FORMAT-001 (NPZ transpose guard)
- Artifact Discipline: Store all new evidence under `plans/active/TEST-PYTORCH-001/reports/<ISO8601>/`. Each loop should at minimum add `summary.md`, targeted pytest logs, and fixture notes. Reference artifact paths in `docs/fix_plan.md` Attempts history.

---

### Phase A — Baseline Assessment & Scope Lock
Goal: Confirm prerequisites (CLI wiring, datasets, runtime requirements) and capture current-state observations before new work.
Prereqs: PyTorch extras installed (`pip install -e .[torch]`); datasets directory readable; Lightning integration from INTEGRATE-PYTORCH-001 green (Attempt #40 evidence).
Exit Criteria: Baseline summary documenting available fixtures, current test status, and runtime budget; artifact hub created under `reports/<TS>/baseline/`.

| ID | Task Description | State | How/Why & Guidance |
| --- | --- | --- | --- |
| A1 | Inventory existing coverage and blockers | [x] | Re-read `tests/torch/test_integration_workflow_torch.py` and charter to identify gaps (e.g., unittest style, heavy dataset). Document findings in `reports/<TS>/baseline/inventory.md`. **COMPLETE 2025-10-19:** Comprehensive inventory at `reports/2025-10-19T115303Z/baseline/inventory.md` catalogues unittest style, GREEN baseline status, zero blockers. |
| A2 | Validate fixture + CLI readiness | [x] | Dry-run `pytest tests/torch/test_integration_workflow_torch.py::TestPyTorchIntegrationWorkflow::test_pytorch_train_save_load_infer_cycle -vv` with `TEE_LOG=plans/active/TEST-PYTORCH-001/reports/<TS>/baseline/pytest_integration_current.log`. Note runtime, return code, and missing artifacts. **COMPLETE 2025-10-19:** Baseline PASSED in 32.54s (27% of 120s budget), full log captured at `reports/2025-10-19T115303Z/baseline/pytest_integration_current.log`. |
| A3 | Capture prerequisites checklist | [x] | Summarize required environment knobs (e.g., `CUDA_VISIBLE_DEVICES=""`, `--disable_mlflow`) and confirm dataset path size/time budget in `reports/<TS>/baseline/summary.md`. **COMPLETE 2025-10-19:** Prerequisites + runtime analysis at `reports/2025-10-19T115303Z/baseline/summary.md`. PyTorch 2.8.0+cu128, dataset 35 MB, CPU-only execution confirmed. |

---

### Phase B — Fixture Minimization & Deterministic Config
Goal: Produce a deterministic, ≤2 minute CPU fixture/config combo to keep regression lean and reproducible.
Prereqs: Phase A baseline complete; identify dataset bottlenecks.
Exit Criteria: Lightweight fixture committed (or documented sourcing), CLI overrides scripted, and guidance recorded for reuse.

| ID | Task Description | State | How/Why & Guidance |
| --- | --- | --- | --- |
| B1 | Design minimal NPZ/probe fixture | [ ] | If existing dataset exceeds runtime budget, use `scripts/tools/subsample_npz.py` (or new script) to create `tests/fixtures/pytorch_integration/minimal_train.npz`. Document provenance + checksum in `reports/<TS>/fixture/fixture_notes.md`. |
| B2 | Codify deterministic config overrides | [ ] | Define shared config snippet (epochs, batch size, seed, gridsize) in `reports/<TS>/fixture/config_profile.md`. Ensure aligns with `TrainingConfig` fields and POLICY-001 expectations. |
| B3 | Wire fixture loader helper | [ ] | Author helper (e.g., `tests/fixtures/pytorch_integration/__init__.py`) returning Paths, ensuring `update_legacy_dict` executed before data access. Capture usage example + ROI in `config_profile.md`. |

---

### Phase C — PyTorch Integration Test (TDD Cycle)
Goal: Implement a native pytest regression that orchestrates train→infer with the lightweight fixture, adhering to artifact hygiene and determinism.
Prereqs: Fixture + config profile ready; CLI confirmed functional.
Exit Criteria: New pytest-based test(s) passing locally with targeted selector, artifacts and hyperparameters verified.

| ID | Task Description | State | How/Why & Guidance |
| --- | --- | --- | --- |
| C1 | Author RED pytest test | [x] | Follow `plans/active/TEST-PYTORCH-001/reports/2025-10-19T120415Z/phase_c_modernization/plan.md` (Phase C1) to convert `tests/torch/test_integration_workflow_torch.py` to pytest style with `_run_pytorch_workflow` stub. Capture failure log at `reports/<TS>/phase_c_modernization/pytest_modernization_red.log`. **COMPLETE 2025-10-19:** Pytest conversion done with NotImplementedError stub. RED log captured at `reports/2025-10-19T120415Z/phase_c_modernization/pytest_modernization_red.log` (0.83s runtime, NotImplementedError as expected). Legacy unittest wrapped with `@pytest.mark.skip`. |
| C2 | Implement orchestration glue | [ ] | Implement helper + assertions per plan (Phase C2) so selector `pytest tests/torch/test_integration_workflow_torch.py::test_run_pytorch_train_save_load_infer -vv` passes. Maintain CPU determinism (`CUDA_VISIBLE_DEVICES=""`) and reuse subprocess commands. Capture GREEN log at `reports/<TS>/phase_c_modernization/pytest_modernization_green.log`. |
| C3 | Validate artifact set + metrics | [ ] | Perform artifact audit + documentation updates per plan (Phase C3). Record findings in `reports/<TS>/phase_c_modernization/artifact_audit.md` and update charter/docstrings before marking row complete. |

---

### Phase D — Regression Hardening & Documentation
Goal: Lock in regression within CI, document runtime/perf, and update ledgers.
Prereqs: Phase C GREEN run complete with artifacts.
Exit Criteria: CI-ready guidance, parity metrics, ledger updates, and follow-on risks captured.

| ID | Task Description | State | How/Why & Guidance |
| --- | --- | --- | --- |
| D1 | Record runtime + resource profile | [ ] | Parse GREEN log to confirm ≤120s CPU runtime; document hardware context in `reports/<TS>/summary.md`. |
| D2 | Update documentation + ledger | [ ] | Append Attempt entries in `docs/fix_plan.md`, update `docs/workflows/pytorch.md` (testing section) if necessary, and link artifact directory. |
| D3 | CI integration follow-up | [ ] | If additional markers/skip logic needed (e.g., `@pytest.mark.integration`), outline changes in `reports/<TS>/summary.md` and create follow-up fix plan entry if gating CI requires separate work. |

---

## Artifact Map Template
- `plans/active/TEST-PYTORCH-001/reports/<ISO>/baseline/` — Phase A inventory + logs
- `plans/active/TEST-PYTORCH-001/reports/<ISO>/fixture/` — Fixture notes, config profiles
- `plans/active/TEST-PYTORCH-001/reports/<ISO>/phase_c_modernization/` — Pytest migration plan, RED/GREEN logs, artifact audits
- `plans/active/TEST-PYTORCH-001/reports/<ISO>/summary.md` — Loop-level summary & decisions

## References
- `plans/pytorch_integration_test_plan.md`
- `docs/TESTING_GUIDE.md`
- `docs/workflows/pytorch.md`
- `specs/ptychodus_api_spec.md` §4.5–§4.6
- `plans/active/INTEGRATE-PYTORCH-001/phase_e2_implementation.md`
