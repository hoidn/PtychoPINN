# INTEGRATE-PYTORCH-001 — PyTorch Backend Integration Plan

## Context
- Initiative: INTEGRATE-PYTORCH-001
- Phase Goal: Ship a PyTorch backend that satisfies the reconstructor contract in `specs/ptychodus_api_spec.md` and can be selected transparently from Ptychodus.
- Dependencies: specs/ptychodus_api_spec.md (contract for lifecycle + persistence); docs/workflows/pytorch.md (current backend behaviour); docs/architecture.md (module map for touchpoints); docs/DEVELOPER_GUIDE.md §1-3 (two-system guidance + params bridge); plans/pytorch_integration_test_plan.md (parallel guardrail effort).
- Artifact storage: capture evidence, callgraphs, and validation logs under `plans/active/INTEGRATE-PYTORCH-001/reports/` using ISO timestamps (e.g., `.../2025-10-17T013000Z/{summary.md,pytest.log}`). Link each run from docs/fix_plan.md Attempts History.

**CRITICAL UPDATE [2025-10-17]:** INTEGRATE-PYTORCH-000 Phase C.C2 stakeholder brief is now available. Review `plans/active/INTEGRATE-PYTORCH-000/reports/2025-10-17T031500Z/stakeholder_brief.md` for:
- 5 major architectural deltas from canonical plan updates
- Configuration bridge (Delta 1) confirmed as #1 blocker — field schema mismatches documented
- Immediate action items for Phase B (this plan) with specific task breakdowns
- 8 open governance questions requiring decisions (see `open_questions.md` in same directory)
- TEST-PYTORCH-001 coordination requirements for fixture design

---

### Phase A — Refresh Parity Baseline
Goal: Reconstruct a complete, current map of TensorFlow ↔ PyTorch responsibilities before touching code.
Prereqs: None; execute at loop start.
Exit Criteria: Consolidated parity document saved under `plans/active/INTEGRATE-PYTORCH-001/reports/<timestamp>/parity_map.md`, reviewed against spec sections noted below.

| ID | Task Description | State | How/Why & Guidance |
| --- | --- | --- | --- |
| A1 | Inventory current TensorFlow workflow touchpoints used by Ptychodus | [x] | Completed 2025-10-17 — see `reports/2025-10-17T020000Z/parity_map.md` for catalogued touchpoints. |
| A2 | Audit PyTorch equivalents and gaps | [x] | Completed 2025-10-17 — gaps recorded in parity_map.md and summarized in `reports/2025-10-17T020000Z/summary.md`. |
| A3 | Define initiative glossary + owner map | [x] | Completed 2025-10-17 — glossary captured in `plans/active/INTEGRATE-PYTORCH-001/glossary_and_ownership.md`. |

---

### Phase B — Configuration & Legacy Bridge Alignment
Goal: Replace singleton config usage with dataclass-driven state that updates `ptycho.params.cfg` per spec.
Prereqs: Completed parity map highlighting configuration deltas; reviewed stakeholder brief Delta 1.
Exit Criteria: PyTorch training/inference paths ingest `TrainingConfig`/`InferenceConfig`, call `update_legacy_dict`, and unit tests confirm params parity.

**Stakeholder Brief Guidance (Delta 1):** Configuration schema divergence is #1 blocker. Key mismatches:
- `grid_size: Tuple[int, int]` → spec requires `gridsize: int`
- `mode: 'Supervised' | 'Unsupervised'` → spec requires `model_type: 'pinn' | 'supervised'`
- Missing fields: `gaussian_smoothing_sigma`, `probe_scale`, `pad_object`
- No KEY_MAPPINGS translation layer for legacy dot-separated keys

**Open Question Q1:** Decide whether to refactor PyTorch config to shared dataclasses (recommended) or maintain dual schemas with translation (see `plans/active/INTEGRATE-PYTORCH-000/reports/2025-10-17T031500Z/open_questions.md` Q1).

| ID | Task Description | State | How/Why & Guidance |
| --- | --- | --- | --- |
| B1 | Complete field-by-field schema audit | [x] | ✅ 2025-10-17 — See `reports/2025-10-17T032218Z/{config_schema_map.md,scope_notes.md}` for 75+ field mapping plus MVP scope + open questions logged against stakeholder brief Delta 1. |
| B2 | Author minimal failing test (TDD) | [x] | ✅ 2025-10-17 — Red-phase contract captured in `tests/torch/test_config_bridge.py` with artifacts `reports/2025-10-17T033500Z/{failing_test.md,pytest.log}`. Test documents adapter API (`ptycho_torch.config_bridge.{to_model_config,to_training_config,to_inference_config}`) and asserts MVP params (`'N'`, `'gridsize'`, `'model_type'`, `'train_data_file_path'`, `'test_data_file_path'`, `'model_path'`, `'n_groups'`, `'neighbor_count'`, `'nphotons'`). Currently xfail+skip pending PyTorch runtime; ready for implementation loop. |
| B3 | Implement schema harmonization | [x] | ✅ 2025-10-17 — Adapter implemented in Attempts #9-#11/#17. See `reports/2025-10-17T045936Z/{adapter_diff.md,summary.md}` for final logic (probe_mask translation, nphotons override enforcement, path normalization). MVP test (`TestConfigBridgeMVP`) now passes; remaining parity work tracked under B5. |
| B4 | Extend parity tests (75+ fields) | [x] | Red-phase complete — see `reports/2025-10-17T041908Z/summary.md` (field matrix, pytest red log, baseline snapshot). Deferred tasks (`test_params_cfg_matches_baseline`, override matrix) now tracked under B5. |
| B5 | Make parity tests pass (Phase B.B5) | [x] | Harness refactor (Attempt #19) and P0 adapter fixes are complete; probe_mask + nphotons parity coverage shipped in Attempt #21 (`reports/2025-10-17T054009Z/notes.md`). Attempt #22 closed Phase C.C1-C2 (n_subsample parity) with evidence in `reports/2025-10-17T055335Z/summary.md`. Baseline comparison test (Attempt #24) is green — see `reports/2025-10-17T061500Z/summary.md`. Override matrix documented at `reports/2025-10-17T063613Z/override_matrix.md`; warning coverage for remaining gaps queued for Phase D3 (see `reports/2025-10-17T062820Z/plan_update.md`). |

---

### Phase C — Data Pipeline & Tensor Packaging Parity
Goal: Provide PyTorch-ready `RawDataTorch` and `PtychoDataContainerTorch` that satisfy grouping and tensor layout contracts.
Prereqs: Phase B completed; dataclasses drive configuration.
Exit Criteria: PyTorch data pipeline outputs match spec shapes/dtypes and pass targeted loader tests.
Detailed checklist: `plans/active/INTEGRATE-PYTORCH-001/phase_c_data_pipeline.md`.

| ID | Task Description | State | How/Why & Guidance |
| --- | --- | --- | --- |
| C1 | Specify data contract expectations | [ ] | Execute Phase C.A (tasks C.A1-C.A3) in `plans/active/INTEGRATE-PYTORCH-001/phase_c_data_pipeline.md`; publish `data_contract.md` + `torch_gap_matrix.md` under timestamped reports before writing tests. |
| C2 | Draft failing loader test | [ ] | Follow Phase C.B (C.B1-C.B3) blueprint in `phase_c_data_pipeline.md`; author torch-optional pytest red cases and archive logs under the same timestamp directory. |
| C3 | Implement `RawDataTorch` wrapper | [ ] | Deliverables defined in Phase C.C (C.C1) — wrap legacy `ptycho.raw_data.RawData`, enforce config bridge usage, document notes in `implementation_notes.md`. |
| C4 | Implement `PtychoDataContainerTorch` | [ ] | Complete Phase C.C tasks (C.C2-C.C3) ensuring tensor API parity; reference contract captured in C.A1. |
| C5 | Verify memmap + cache lifecycles | [ ] | Use Phase C.D (C.D1-C.D3) validation steps; capture pytest logs + cache evidence in reports and update ledger/documents per plan. |

---

### Phase D — Workflow Orchestration & Persistence
Goal: Expose PyTorch training/inference orchestration compatible with reconstructor expectations, including save/load semantics.
Prereqs: Phases B & C complete.
Exit Criteria: `ptychodus` can call into PyTorch backend for train/infer; artifacts saved in spec-compliant archives.

| ID | Task Description | State | How/Why & Guidance |
| --- | --- | --- | --- |
| D1 | Design PyTorch equivalents of `run_cdi_example` + `ModelManager` | [ ] | Author design note in `reports/<timestamp>/workflow_design.md` referencing `ptycho/workflows/components.py` and `ptycho/model_manager.py`. |
| D2 | Implement orchestration entry points | [ ] | Ensure new functions accept dataclass configs, avoid global state, and align with CLI semantics documented in `docs/workflows/pytorch.md`. |
| D3 | Implement persistence shim | [ ] | Save Lightning checkpoints + params bundle mirroring TensorFlow `.h5.zip` contents; record mapping decisions. |
| D4 | Add regression tests | [ ] | Extend test plan (existing `plans/pytorch_integration_test_plan.md`) with PyTorch backend path; log test commands + outputs. |

---

### Phase E — Ptychodus Integration & Parity Validation
Goal: Wire PyTorch backend into `ptychodus` reconstructor selection and prove parity through automated tests.
Prereqs: Phases B-D complete.
Exit Criteria: Ptychodus integration tests (existing + new for PyTorch) pass on CI; documentation updated.

| ID | Task Description | State | How/Why & Guidance |
| --- | --- | --- | --- |
| E1 | Update reconstructor selection logic | [ ] | Modify `ptychodus/src/ptychodus/model/ptychopinn/reconstructor.py` as guided by spec §4.3; ensure backend choice is configuration-driven. |
| E2 | Extend parity test suite | [ ] | Collaborate with TEST-PYTORCH-001 initiative to reuse fixtures; record test selection (likely `pytest tests/torch/test_integration_workflow.py`). |
| E3 | Document backend selection workflow | [ ] | Update `docs/workflows/pytorch.md` + Ptychodus docs to describe new flag; capture diffs in reports. |
| E4 | Perform final comparison run | [ ] | Execute both TensorFlow and PyTorch integration tests; store metrics comparison under `reports/<timestamp>/parity_summary.md`. |

---

### Phase Exit Reviews
- Each phase completion requires a short retrospective note stored alongside reports summarizing risks, open questions, and next steps.
- If a phase blocks (e.g., missing fixture, unresolved spec conflict), log the issue in docs/fix_plan.md Attempts History and flag in galph_memory.md.

### Open Risks & Mitigations
- Lightning + MLflow dependence may complicate CI runs → plan fallback configuration (`MLFLOW_TRACKING_URI= `, disable autolog) before Phase D.
- Data fixture availability for PyTorch tests unknown → coordinate with TEST-PYTORCH-001 to build shared minimal dataset.
- Persistence parity may require schema changes; ensure spec updates accompany any semantic shifts.
