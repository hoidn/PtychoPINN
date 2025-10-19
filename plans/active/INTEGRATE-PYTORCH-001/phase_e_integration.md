# Phase E — Ptychodus Integration & Parity Validation

-## Context
- Initiative: INTEGRATE-PYTORCH-001
- Phase Goal: Enable Ptychodus to select and execute the PyTorch backend with parity safeguards that match the TensorFlow reconstructor contract defined in `specs/ptychodus_api_spec.md` §4.
- Dependencies: Phase D handoff (`reports/2025-10-17T121930Z/phase_d4c_summary.md`), canonical initiative plan (`plans/ptychodus_pytorch_integration_plan.md` Phase 6–8), PyTorch workflow guide (`docs/workflows/pytorch.md`), TEST-PYTORCH-001 test strategy (`plans/pytorch_integration_test_plan.md`), CONFIG-001 finding (docs/findings.md ID CONFIG-001).
- Environment Constraint: Headless CI/agent environment — GUI launch/validation is out of scope; all parity checks must rely on CLI workflows and logged artifacts.
- Artifact Storage: Capture all Phase E deliverables under `plans/active/INTEGRATE-PYTORCH-001/reports/<ISO8601>/phase_e_*` with descriptive filenames (e.g., `phase_e_callchain.md`, `phase_e_selector_map.md`, `phase_e_parity_summary.md`). Reference each artifact in docs/fix_plan.md.

---

### Phase E1 — Backend Selection & Orchestration Bridge
Goal: Design and implement the reconstructor-selection handshake so Ptychodus can delegate to PyTorch workflows without breaking the existing TensorFlow path.
Prereqs: Review spec §4.1–4.6, TensorFlow baseline call flow (`ptycho/workflows/components.py`, `ptycho/model_manager.py`), and Phase D4 handoff notes.
Exit Criteria: Documented callchain diff, TDD red tests in the Ptychodus repo covering backend selection, and implementation plan for wiring PtychoPINN reconstructor hooks to `ptycho_torch/workflows/components.py`.

| ID | Task Description | State | How/Why & Guidance |
| --- | --- | --- | --- |
| E1.A | Map current Ptychodus reconstructor callchain | [x] | ✅ Completed 2025-10-17. Callchain analysis captured at `plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T173826Z/phase_e_callchain/{static.md,summary.md,pytorch_workflow_comparison.md}`. Documents TensorFlow vs PyTorch workflow parity with CONFIG-001 gates mapped. |
| E1.B | Author backend-selection failing tests | [x] | ✅ Completed 2025-10-17. Created `tests/torch/test_backend_selection.py` with 6 red tests (all XFAIL). Red logs at `plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T173826Z/phase_e_red_backend_selection.log`. Tests document expected backend selection behavior per spec §4.1-4.6. |
| E1.C | Draft implementation blueprint | [x] | ✅ Completed 2025-10-17. See `plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T180500Z/phase_e_backend_design.md` for dispatcher design, task breakdown (E1.C1–E1.C4), and TDD guidance. |

---

### Phase E2 — Integration Regression & Parity Harness
Goal: Provide runnable parity tests that exercise the PyTorch backend end-to-end alongside TensorFlow.
Prereqs: E1 design complete, TEST-PYTORCH-001 coordination notes reviewed (`phase_d4_alignment.md`, `phase_d4_selector_map.md`).
Exit Criteria: Torch-optional failing tests documenting the integration gap, green implementation wiring PyTorch workflows, and parity report comparing TensorFlow vs PyTorch outputs on shared fixtures. Execution guidance is expanded in `reports/2025-10-17T212500Z/phase_e2_plan.md`.

| ID | Task Description | State | How/Why & Guidance |
| --- | --- | --- | --- |
| E2.A1 | Review TEST-PYTORCH-001 fixture inventory | [x] | ✅ Completed 2025-10-17T213500Z. Fixture inventory captured at `reports/2025-10-17T213500Z/phase_e_fixture_sync.md` documenting Run1084_recon3_postPC_shrunk_3.npz (1087 patterns, N=64, gridsize=1) as primary dataset with transposed diffraction shape noted. |
| E2.A2 | Define minimal reproducible dataset + env knobs | [x] | ✅ Completed 2025-10-17T213500Z. Minimal reproduction parameters documented in fixture_sync.md §2: nepochs=2, n_images=64, batch_size=4, device=cpu with CLI flag enumeration and CONFIG-001/DATA-001 compliance requirements. |
| E2.B1 | Author torch-optional integration test skeleton | [x] | ✅ Completed 2025-10-17T213500Z. Created `tests/torch/test_integration_workflow_torch.py` (179 lines) mirroring TensorFlow integration test with subprocess invocation structure and checkpoint validation. |
| E2.B2 | Capture red pytest evidence | [x] | ✅ Completed 2025-10-17T213500Z. Red test execution captured: 1 FAILED (ModuleNotFoundError: lightning), 1 SKIPPED (parity deferred). Logs at `reports/2025-10-17T213500Z/phase_e_red_integration.log` with failure analysis in `red_phase.md`. |
| E2.C1 | Wire backend dispatcher to PyTorch workflows | [x] | ✅ 2025-10-17T215500Z — CLI + dispatcher wiring complete (see `phase_e2_implementation.md` C1 row and `reports/2025-10-17T215500Z/phase_e2_green.md`). CONFIG-001 guard executes prior to delegating into `run_cdi_example_torch`; checkpoints stored under `<output_dir>/checkpoints/`. |
| E2.C2 | Verify fail-fast + params bridge behavior | [x] | ✅ 2025-10-17T215500Z — Targeted pytest selectors executed; logs archived at `reports/2025-10-17T215500Z/{phase_e_backend_green.log,phase_e_integration_green.log}`. Current environment skips due to torch absence, demonstrating fail-fast messaging. |
| E2.D1 | Execute TensorFlow baseline integration test | [x] | ✅ 2025-10-18T093500Z — Executed TensorFlow baseline test successfully (1 PASSED, 31.88s). Log archived at `reports/2025-10-18T093500Z/phase_e_tf_baseline.log`. Integration workflow validated: train→save→load→infer cycle complete. |
| E2.D2 | Execute PyTorch integration test | [x] | ✅ 2025-10-17T221500Z — Successfully installed PyTorch extras and executed integration test. Test failed due to DATA-001 violation in dataloader (expects `diff3d` key instead of canonical `diffraction`). Log archived at `reports/2025-10-17T221500Z/phase_e_torch_run.log`. CLI + CONFIG-001 validated; dataloader fix required for full parity. |
| E2.D3 | Compare outputs + publish parity summary | [x] | ✅ 2025-10-17T221500Z — Updated parity summary at `reports/2025-10-17T221500Z/phase_e_parity_summary.md`. Documented TensorFlow baseline success, PyTorch DATA-001 violation, dependency installation outcomes, and defined follow-up task [INTEGRATE-PYTORCH-001-DATALOADER]. Phase E2.D evidence capture complete. |

**Artifact Discipline:** Use ISO timestamps per loop (e.g., `reports/2025-10-17T212500Z/` for planning artifacts, `reports/<execution-ts>/` for red/green runs). Reference each generated file from docs/fix_plan.md Attempts history.

---

### Phase E3 — Documentation, Spec Sync, and Handoff
Goal: Ensure documentation, specs, and downstream initiatives reflect the new backend, and hand off to TEST-PYTORCH-001/production teams.
Prereqs: E1–E2 tasks completed with successful regression runs.
Exit Criteria: Updated documentation and spec excerpts, ledger entries recorded, and follow-up actions queued for TEST-PYTORCH-001.
Planning Reference: `plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-19T205832Z/phase_e3_docs_plan.md` outlines the phased checklist for this work.

| ID | Task Description | State | How/Why & Guidance |
| --- | --- | --- | --- |
| E3.A | Update workflow documentation | [x] | ✅ 2025-10-19 Attempt #17: Updated `docs/workflows/pytorch.md` §11–§12 with regression selector guidance and backend dispatcher instructions; refreshed `docs/architecture.md` dual-backend paragraph. Evidence archived at `reports/2025-10-19T210000Z/phase_e3_docs_update/{diff_notes.md,summary.md}`. |
| E3.B | Sync specs & findings | [x] | ✅ 2025-10-19 Attempt #16: Added spec §4.8 \"Backend Selection & Dispatch\" (`specs/ptychodus_api_spec.md:224-235`) and documented findings review (POLICY-001/CONFIG-001 sufficient). Evidence: `reports/2025-10-19T205832Z/{phase_e3_spec_patch.md,phase_e3_spec_update.md}` plus governance confirmation `reports/2025-10-19T202600Z/phase_e3_governance_review.md`. |
| E3.C | Prepare TEST-PYTORCH-001 handoff | [x] | ✅ 2025-10-19 Attempt #43: Authored `handoff_brief.md` under `reports/2025-10-19T215800Z/phase_e3_docs_handoff/` with backend contract, selector cadence, artifact expectations, ownership matrix, and open questions for TEST-PYTORCH-001 Phase D3. Summary recorded in `summary.md`. |

---

### Exit Checklist
- [x] Phase E1 artifacts stored and referenced; backend-selection tests failing as expected before implementation. (`reports/2025-10-17T173826Z/phase_e_callchain/{static.md,summary.md}`, `phase_e_red_backend_selection.log`, `phase_e_backend_design.md`)
- [x] Phase E2 integration tests capture PyTorch gaps then pass with wiring in place; parity report archived. (`reports/2025-10-19T111855Z/phase_d2_completion/pytest_integration_shape_green.log`, `reports/2025-10-19T201500Z/phase_d2_completion/parity_update.md`)
- [x] Documentation, spec updates, and TEST-PYTORCH-001 handoff complete with ledger entries and risk notes. (`reports/2025-10-19T210000Z/phase_e3_docs_update/summary.md`, `reports/2025-10-19T205832Z/phase_e3_spec_update.md`, `reports/2025-10-19T223500Z/phase_e3_docs_handoff/monitoring_update.md`, docs/fix_plan Attempt #44)

### Phase E Close-Out
Goal: Capture final exit narrative and prepare fix_plan closure recommendation for INTEGRATE-PYTORCH-001.

| ID | Task Description | State | How/Why & Guidance |
| --- | --- | --- | --- |
| CO1 | Author Phase E closure summary with exit checklist validation | [x] | ✅ 2025-10-19 Attempt #46: Authored `closure_summary.md` under `plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-19T225500Z/phase_e_closeout/` (11 sections, 650 lines) enumerating E1–E3 evidence, runtime guardrails (35.92s baseline, ≤90s CI budget, 60s warning threshold), monitoring handoff (12 escalation triggers), and governance closure checklist. References Attempts #32-45 across Phase E1/E2/E3 execution. |
| CO2 | Append fix_plan closure recommendation | [x] | ✅ 2025-10-19 Attempt #46: Added docs/fix_plan Attempt #46 summarizing closure readiness with closure_summary.md reference. Noted [INTEGRATE-PYTORCH-001-DATALOADER] complete (Attempts #0-1), [ADR-003-BACKEND-API] pending (separate governance initiative), TEST-PYTORCH-001 Phase D3 handoff complete. **RECOMMENDATION:** Close INTEGRATE-PYTORCH-001 with governance sign-off. |
