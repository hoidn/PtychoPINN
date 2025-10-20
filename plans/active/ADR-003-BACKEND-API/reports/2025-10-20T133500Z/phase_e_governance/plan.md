# Phase E Governance & Deprecation Plan

## Context
- Initiative: ADR-003-BACKEND-API (Standardize PyTorch backend API)
- Phase Goal: Secure governance acceptance for the new backend API, expose remaining execution knobs, and align documentation + test suite ahead of legacy API deprecation.
- Dependencies: Phase D smoke evidence (`reports/2025-10-20T125500Z/phase_d_cli_wrappers_smoke/`), execution design docs (`reports/2025-10-20T232336Z/phase_b_factories/`), spec §§4 & 7 (`specs/ptychodus_api_spec.md`), workflow guide §§11–13 (`docs/workflows/pytorch.md`), findings POLICY-001 / CONFIG-001 / FORMAT-001.

### Phase E.A — Governance Dossier
Goal: Produce the governance documentation required to accept ADR-003 and capture the authoritative contract updates.
Prereqs: Phase D artifacts ingested; smoke evidence and handoff summary reviewed.
Exit Criteria: ADR-003.md captures final state, spec + workflow guide reflect execution config fields, and ledger cross-links recorded.

| ID | Task Description | State | How/Why & Guidance (including API / document / artifact / source file references) |
| --- | --- | --- | --- |
| E.A1 | Draft ADR-003 acceptance addendum | [x] | Author `plans/active/ADR-003-BACKEND-API/reports/<TS>/phase_e_governance_adr_addendum/adr_addendum.md` summarising Phases A–D evidence, open issues, and acceptance rationale. Pull data from `factory_design.md`, `override_matrix.md`, and `phase_d_cli_wrappers_smoke/handoff_summary.md`. Reference `docs/architecture/adr/ADR-003.md` §Decision to note the acceptance timestamp. **COMPLETE:** Addendum authored (9 sections, 500+ lines) with comprehensive evidence compilation, acceptance criteria validation (37/37 tests GREEN), and Phase E backlog enumeration. Artifacts: `reports/2025-10-20T134500Z/phase_e_governance_adr_addendum/{adr_addendum.md,summary.md}`. |
| E.A2 | Update specs with execution config contract | [x] | Redline `specs/ptychodus_api_spec.md` §§4.7–4.9 to enumerate PyTorch execution config fields (accelerator, deterministic, num_workers, learning_rate, batch sizes). Capture before/after diff in `reports/<TS>/phase_e_governance/spec_redline.md`. Align wording with CONFIG-001 + POLICY-001 warns. **Reference prep notes:** `reports/2025-10-20T150020Z/phase_e_governance_spec_redline/spec_delta_notes.md`. **COMPLETE:** Spec updated with §4.7 backend-specific requirements (TensorFlow + PyTorch paths), new §4.9 PyTorchExecutionConfig contract (5 field categories, 17 fields, validation rules), and corrected §7 CLI tables (accelerator default `'auto'`, inference batch size default `None`, deprecated flags documented). Change log: `reports/2025-10-20T150020Z/phase_e_governance_spec_redline/spec_redline.md` (130 lines added, 0 breaking changes). |
| E.A3 | Refresh workflow guide and knowledge base | [ ] | Update `docs/workflows/pytorch.md` §§11–13 with Phase D runtime benchmarks, helper flow narrative, and upcoming deprecation schedule for `--device` / `--disable_mlflow`. Add summary of execution knobs to `docs/findings.md` (linking new finding if policy shifts). |

### Phase E.B — Execution Knob Hardening
Goal: Implement outstanding Lightning execution controls and ensure CLI/tests cover them.
Prereqs: Phase E.A ADR + spec updates drafted (so configuration intent is frozen before code changes).
Exit Criteria: CLI exposes agreed knobs, validation + tests cover new flags, and defaults documented.

| ID | Task Description | State | How/Why & Guidance |
| --- | --- | --- | --- |
| E.B1 | Expose checkpoint & early-stop controls | [ ] | Follow handoff §3 backlog: add CLI flags for `--checkpoint-save-top-k`, `--checkpoint-monitor`, `--checkpoint-mode`, `--early-stop-patience`. Update `ptycho_torch/workflows/components.py` and factories to consume overrides. Add RED test in `tests/torch/test_config_factory.py::TestExecutionConfigOverrides` then GREEN. |
| E.B2 | Wire scheduler / gradient accumulation knobs | [ ] | Add CLI + config support for `--scheduler` (enum) and `--accumulate-grad-batches`. Update Lightning module initialization. Tests: extend `tests/torch/test_cli_train_torch.py` to assert parsed overrides propagate to `PyTorchExecutionConfig`. |
| E.B3 | Logger backend / MLflow handling | [ ] | Decide whether to implement MLflow logger or formally deprecate flag. Capture decision in `reports/<TS>/phase_e_governance/logger_decision.md`. Add tests to confirm CLI warns or delegates correctly. |
| E.B4 | Runtime smoke extensions | [ ] | Add deterministic smoke covering `gridsize=3` and `--accelerator auto`. Store logs under `reports/<TS>/phase_e_governance/runtime_smoke/`. Update plan checklist once selectors executed. |

### Phase E.C — Deprecation & Closure
Goal: Finalise deprecation messaging, archive legacy API, and close out initiative bookkeeping.
Prereqs: Phase E.B implementation + tests GREEN; governance approvals captured.
Exit Criteria: Legacy API marked deprecated or removed, docs + ledger updated, work packaged for archive.

| ID | Task Description | State | How/Why & Guidance |
| --- | --- | --- | --- |
| E.C1 | Implement `ptycho_torch/api/` thin wrappers or deprecation | [ ] | Either wrap old API to new workflows or emit `DeprecationWarning` with migration guidance. Log decision + diff in `reports/<TS>/phase_e_governance/api_deprecation.md`. |
| E.C2 | Update docs/fix_plan + plan ledger | [ ] | When Phase E completes, mark `implementation.md` Phase E rows `[x]`, append Attempt summary to `docs/fix_plan.md`, and archive artifacts to `archive/` if initiative ends. |
| E.C3 | Archive initiative evidence | [ ] | Produce `summary.md` capturing final outcomes, test selectors, and references to ADR/spec commits. Move closed plan to `archive/` per workflow guide. |

## Reporting Discipline
- Store all Phase E artifacts beneath `plans/active/ADR-003-BACKEND-API/reports/<ISO8601>/phase_e_governance/` subdirectories (e.g., `adr_addendum.md`, `spec_redline.md`, `runtime_smoke/`).
- Every checklist update must link to selector commands sourced from `docs/TESTING_GUIDE.md` or existing tests; add new selectors to `docs/development/TEST_SUITE_INDEX.md` if they become canonical.
- Tests follow native pytest style; maintain deterministic CPU runs (`CUDA_VISIBLE_DEVICES=""`).
