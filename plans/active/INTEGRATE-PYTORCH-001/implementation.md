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
| B3 | Implement schema harmonization | [ ] | **Decision-dependent (Q1):** Implement minimal translation layer (`ptycho_torch.config_bridge`) that passes the B2 test while keeping Option 1 (full dataclass refactor) under evaluation. Adapter must translate MVP fields, call `update_legacy_dict` safely (respecting CONFIG-001), and live in a module location that can later delegate to shared dataclasses. Capture implementation notes + green test output under `reports/<timestamp>/{bridge_notes.md,pytest.log}`; once ready for refactor decision, document migration plan in same directory. **Update 2025-10-17:** Attempt `reports/2025-10-17T034800Z/` still fails because `to_model_config()` passes unsupported kwargs and leaves `amp_activation='silu'`. See `reports/2025-10-17T040158Z/config_bridge_debug.md` for reproduction + fix checklist; do not mark B3 complete until the targeted pytest passes without skip. |
| B4 | Extend parity tests (75+ fields) | [ ] | **Per stakeholder brief:** Implement parameterized tests validating all config fields propagate correctly through dataclass → `update_legacy_dict` → `params.cfg` → downstream consumers. Reference spec §5.1-5.3 for complete field inventory. Test strategy: compare `params.cfg` snapshots from TensorFlow and PyTorch backends for matrix of representative configurations. Store test results in `reports/<timestamp>/parity_test_results.md`. |

---

### Phase C — Data Pipeline & Tensor Packaging Parity
Goal: Provide PyTorch-ready `RawDataTorch` and `PtychoDataContainerTorch` that satisfy grouping and tensor layout contracts.
Prereqs: Phase B completed; dataclasses drive configuration.
Exit Criteria: PyTorch data pipeline outputs match spec shapes/dtypes and pass targeted loader tests.

| ID | Task Description | State | How/Why & Guidance |
| --- | --- | --- | --- |
| C1 | Specify data contract expectations | [ ] | Summarize required keys/shapes from `specs/data_contracts.md` and `specs/ptychodus_api_spec.md §3` in `reports/<timestamp>/data_contract.md`. |
| C2 | Draft failing loader test | [ ] | Add pytest case under `tests/torch/test_data_container.py` ensuring amplitude/complex dtype handling and neighbor grouping reuse cached outputs. |
| C3 | Implement `RawDataTorch` wrapper | [ ] | Mirror `ptycho/raw_data.py:120-380` behaviour; ensure caching semantics align with `docs/architecture.md` data flow figure. |
| C4 | Implement `PtychoDataContainerTorch` | [ ] | Provide API-compatible tensors for training/inference; document shapes vs. TensorFlow `PtychoDataContainer` in plan. |
| C5 | Verify memmap + cache lifecycles | [ ] | Capture test run output (expected command `pytest tests/torch/test_data_container.py -k cache`) into reports. |

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
