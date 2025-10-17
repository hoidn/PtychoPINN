Summary: Map Ptychodus backend selection flow and capture TDD red tests for PyTorch reconstructor activation
Mode: Parity
Focus: INTEGRATE-PYTORCH-001 / Phase E1 backend selection groundwork
Branch: feature/torchapi
Mapped tests: none — evidence-only
Artifacts: plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T173826Z/{phase_e_callchain/static.md,phase_e_red_backend_selection.log}
Do Now:
1. INTEGRATE-PYTORCH-001 — Phase E1.A callchain map @ plans/active/INTEGRATE-PYTORCH-001/phase_e_integration.md:18 — tests: none — Run the callchain prompt to document how Ptychodus currently selects TensorFlow and capture the annotated static trace under `phase_e_callchain/`.
2. INTEGRATE-PYTORCH-001 — Phase E1.B backend red tests @ plans/active/INTEGRATE-PYTORCH-001/phase_e_integration.md:19 — tests: author + run `pytest tests/ptychodus/test_backend_selection.py::TestBackendSelection::test_selects_pytorch_backend -vv` (expected fail) — Add torch-optional pytest cases in the Ptychodus repo asserting backend flag switches to PyTorch while preserving TensorFlow default; tee the failing run to `phase_e_red_backend_selection.log`.
If Blocked: Note blocking gaps in docs/fix_plan.md (Attempt history) with artifact links, leave checklist rows `[P]`, and stash partial outputs under the same timestamped reports directory before exiting.
Priorities & Rationale:
- specs/ptychodus_api_spec.md:150 — Reconstructor selection (spec §4.1–4.3) defines the contract the new backend must satisfy; callchain evidence ensures parity scope is correct.
- docs/findings.md:9 — CONFIG-001 forces us to verify `update_legacy_dict` ordering inside the new tests.
- plans/active/INTEGRATE-PYTORCH-001/phase_e_integration.md:18 — Phase E1 checklist mandates callchain evidence before implementation.
- plans/ptychodus_pytorch_integration_plan.md:133 — Phase 6 tasks require backend registration and UI flag; our red tests encode these expectations.
- plans/pytorch_integration_test_plan.md:10 — Upcoming integration harness depends on backend flag semantics; documenting failures up front prevents rework.
How-To Map:
- Callchain: Follow the procedure in `prompts/callchain.md` using `analysis_question="How does Ptychodus select and invoke the TensorFlow backend?"`, `initiative_id="INTEGRATE-PYTORCH-001"`, and `scope_hints=["backend selection","CONFIG-001"]`; store outputs under `plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T173826Z/phase_e_callchain/` (`static.md`, `summary.md`).
- Scope callchain to `ptychodus/src/ptychodus/model/ptychopinn/reconstructor.py`, `ptychodus/src/ptychodus/model/ptychopinn/library.py`, and `ptycho/workflows/components.py`; annotate CONFIG-001 touch points explicitly.
- New tests: follow torch-optional pattern used in `tests/conftest.py` (skip when torch missing). Encode three assertions: default backend == TensorFlow, explicit PyTorch flag routes to torch orchestration, and `update_legacy_dict` is invoked before calling PyTorch workflows (use monkeypatch to assert ordering).
- After writing tests, run `pytest tests/ptychodus/test_backend_selection.py::TestBackendSelection -vv | tee plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T173826Z/phase_e_red_backend_selection.log`; ensure failure message is preserved.
- Update plan checkboxes (`phase_e_integration.md` E1.A/E1.B) to `[P]` with artifact references and add ledger Attempt details when done.
Pitfalls To Avoid:
- Do not modify production code yet; this loop is evidence-only.
- Keep new tests torch-optional (guard imports, use skip markers consistent with `tests/conftest.py`).
- Avoid assuming availability of PyTorch repo inside this workspace—reference interfaces abstractly and document gaps if repo absent.
- Ensure callchain artifacts include file:line anchors; incomplete traces slow downstream work.
- When authoring tests, avoid hardcoding MLflow dependencies; tests must pass without external services once implementation lands.
Pointers:
- plans/active/INTEGRATE-PYTORCH-001/phase_e_integration.md:16
- plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T121930Z/phase_d4c_summary.md:12
- specs/ptychodus_api_spec.md:150
- plans/ptychodus_pytorch_integration_plan.md:133
- plans/pytorch_integration_test_plan.md:1
- tests/conftest.py:1
Next Up: Phase E1.C design blueprint once red tests exist.
