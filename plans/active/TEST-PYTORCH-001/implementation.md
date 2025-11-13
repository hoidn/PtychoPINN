# TEST-PYTORCH-001 — PyTorch Integration Regression Plan

<plan_update version="1.0">
  <trigger>Baseline must shift from CPU-only determinism to CUDA GPU execution per new backend policy.</trigger>
  <focus_id>TEST-PYTORCH-001</focus_id>
  <documents_read>docs/index.md, docs/workflows/pytorch.md, plans/pytorch_integration_test_plan.md, plans/active/TEST-PYTORCH-001/implementation.md</documents_read>
  <current_plan_path>plans/active/TEST-PYTORCH-001/implementation.md</current_plan_path>
  <proposed_changes>Rewrite context/goals to describe GPU-default regression, update Phase A/B/D guidance, and replace CPU-only selectors with CUDA-pinned commands.</proposed_changes>
  <impacts>Requires CI and local developers to have access to at least one CUDA device; CPU evidence becomes fallback-only and must be labeled as such.</impacts>
  <ledger_updates>Log this GPU-baseline edit in docs/fix_plan.md under the TEST-PYTORCH-001 focus when recording the next attempt.</ledger_updates>
  <status>approved</status>
</plan_update>

## Context
- Initiative: TEST-PYTORCH-001 — Author PyTorch integration workflow regression
- Phase Goal: Deliver a GPU-default regression (CUDA device pinned via `CUDA_VISIBLE_DEVICES`) that exercises the PyTorch train→save→load→infer pipeline end-to-end and codifies evidence capture/log hygiene so CI guards the backend. Legacy CPU runs remain documented as fallback evidence only.
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
| A3 | Capture prerequisites checklist | [x] | Summarize required environment knobs (now `CUDA_VISIBLE_DEVICES="0"` + driver versions, `--disable_mlflow`) and confirm dataset path size/time budget in `reports/<TS>/baseline/summary.md`. **COMPLETE 2025-10-19:** Prerequisites + runtime analysis at `reports/2025-10-19T115303Z/baseline/summary.md`. PyTorch 2.8.0+cu128, dataset 35 MB, CPU-only execution confirmed (legacy). **Update:** Next refresh must restate these prerequisites for a CUDA host and capture GPU model/driver metadata. |

---

### Phase B — Fixture Minimization & Deterministic Config
Goal: Produce a deterministic, ≤2 minute GPU fixture/config combo to keep regression lean and reproducible (document CPU fallback behavior separately).
Prereqs: Phase A baseline complete; identify dataset bottlenecks.
Exit Criteria: Lightweight fixture committed (or documented sourcing), CLI overrides scripted, and guidance recorded for reuse.

| ID | Task Description | State | How/Why & Guidance |
| --- | --- | --- | --- |
| B1 | Design minimal NPZ/probe fixture | [x] | Evidence: `reports/2025-10-19T215300Z/phase_b_fixture/{dataset_probe.txt,logs/,fixture_scope.md}` captures dataset telemetry, runtime sweep (21.91s/17.11s), and nine acceptance criteria guiding fixture construction. |
| B2 | Codify deterministic config overrides | [x] | ✅ 2025-10-19 — Fixture generator implemented + documented (see `reports/2025-10-19T225900Z/phase_b_fixture/{fixture_generation.log,pytest_fixture_green.log,fixture_notes.md,summary.md}`). Core contract tests pass; remaining smoke failures (RawData attribute, PyTorch import path) deferred to Phase B3 integration work. |
| B3 | Wire fixture loader helper | [x] | ✅ B3.A/B3.B/B3.C COMPLETE — integration test now consumes `tests/fixtures/pytorch_integration/minimal_dataset_v1.npz` (runtime: 14.53s integration, 3.82s smoke, 33.7% faster vs B1 baseline). Workflow docs updated (docs/workflows/pytorch.md §11 refreshed with Phase B3 runtime/fixture evidence, CI thresholds adjusted to 90s timeout / 45s warning). Artifacts: `reports/2025-10-19T{215300Z,220500Z,225900Z,233500Z,224546Z}/phase_b_fixture/`. See `reports/2025-10-19T214052Z/phase_b_fixture/plan.md` for full Phase B1-B3 checklist. |

---

### Phase C — PyTorch Integration Test (TDD Cycle)
Goal: Implement a native pytest regression that orchestrates train→infer with the lightweight fixture, adhering to artifact hygiene and determinism.
Prereqs: Fixture + config profile ready; CLI confirmed functional.
Exit Criteria: New pytest-based test(s) passing locally with targeted selector, artifacts and hyperparameters verified.

| ID | Task Description | State | How/Why & Guidance |
| --- | --- | --- | --- |
| C1 | Author RED pytest test | [x] | Follow `plans/active/TEST-PYTORCH-001/reports/2025-10-19T120415Z/phase_c_modernization/plan.md` (Phase C1) to convert `tests/torch/test_integration_workflow_torch.py` to pytest style with `_run_pytorch_workflow` stub. Capture failure log at `reports/<TS>/phase_c_modernization/pytest_modernization_red.log`. **COMPLETE 2025-10-19:** Pytest conversion done with NotImplementedError stub. RED log captured at `reports/2025-10-19T120415Z/phase_c_modernization/pytest_modernization_red.log` (0.83s runtime, NotImplementedError as expected). Legacy unittest wrapped with `@pytest.mark.skip`. |
| C2 | Implement orchestration glue | [x] | ✅ 2025-10-19 — Helper implemented at `tests/torch/test_integration_workflow_torch.py:65-161` with subprocess train/infer commands. Targeted test PASSED in 35.86s. GREEN log captured at `reports/2025-10-19T122449Z/phase_c_modernization/pytest_modernization_green.log`. Full regression: 236 passed, 17 skipped, 1 xfailed, ZERO new failures. Artifacts + summary at `reports/2025-10-19T122449Z/phase_c_modernization/`. |
| C3 | Validate artifact set + metrics | [x] | ✅ 2025-10-19 — Artifact audit complete at `reports/2025-10-19T130900Z/phase_c_modernization/artifact_audit.md` (checkpoint format, reconstruction outputs, runtime validation). Rerun log captured at `pytest_modernization_rerun.log` (1 PASSED in 35.98s). Documentation updates: test comment (line 188), implementation.md C2 row, C2 summary.md. fix_plan.md Attempt #7 logged. Summary at `reports/2025-10-19T130900Z/phase_c_modernization/summary.md`. |

---

### Phase D — Regression Hardening & Documentation
Goal: Lock in regression within CI, document runtime/perf, and update ledgers.
Prereqs: Phase C GREEN run complete with artifacts.
Exit Criteria: CI-ready guidance, parity metrics, ledger updates, and follow-on risks captured.

Planning reference: `plans/active/TEST-PYTORCH-001/reports/2025-10-19T193425Z/phase_d_hardening/plan.md`

| ID | Task Description | State | How/Why & Guidance |
| --- | --- | --- | --- |
| D1 | Record runtime + resource profile | [x] | ✅ 2025-10-19 — Runtime profile complete at `reports/2025-10-19T193425Z/phase_d_hardening/runtime_profile.md`. Aggregated C2/C3/D1 runtimes (mean 35.92s, variance 0.17%). Environment documented: Python 3.11.13, PyTorch 2.8.0+cu128, Ryzen 9 5950X (32 CPUs), 128GB RAM. Guardrails defined: ≤90s CI max, 60s warning, 36s±5s baseline. **Update:** GPU baseline is now mandatory; capture next runtime profile on a reproducible CUDA host (record GPU model, driver, CUDA toolkit) and revise guardrails once data is collected. Artifacts: `env_snapshot.txt`, `pytest_modernization_phase_d.log`, `runtime_profile.md`. |
| D2 | Update documentation + ledger | [x] | ✅ 2025-10-19 — Documentation alignment complete (Attempt #11). Updated this implementation plan D1 row with artifact citations. Appended `docs/fix_plan.md` [TEST-PYTORCH-001] Attempt #11 with Phase D2 summary referencing runtime profile (2025-10-19T193425Z) and updated workflow doc. Refreshed `docs/workflows/pytorch.md` §11 (new "Regression Test & Runtime Expectations" subsection) with pytest selector, 36s±5s baseline, ≤90s CI guardrail, POLICY-001/FORMAT-001 reminders. Artifacts: `reports/2025-10-19T201900Z/phase_d_hardening/{doc_alignment_notes.md,summary.md}`. |
| D3 | CI integration follow-up | [x] | ✅ 2025-10-19 — CI integration strategy complete (Attempt #12). No `.github/workflows/` exists; documented CI-ready guidance in `reports/2025-10-19T232500Z/phase_d_hardening/ci_notes.md`. Pytest selector defined: `CUDA_VISIBLE_DEVICES="0" pytest tests/torch/test_integration_workflow_torch.py::test_run_pytorch_train_save_load_infer -vv`. Runtime guardrails: 120s timeout (conservative), 90s nominal, 60s warning (update once GPU runtime data is available). Recommended markers: `@pytest.mark.integration` + `@pytest.mark.slow` (optional). Follow-up tickets: FU-001 (add markers), FU-002 (implement GH workflow), FU-003 (update TEST_SUITE_INDEX.md). Artifacts: `ci_notes.md`, `summary.md`. |

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
