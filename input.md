Summary: Define PyTorchExecutionConfig and capture RED→GREEN evidence (Phase C1)
Mode: TDD
Focus: [ADR-003-BACKEND-API] Standardize PyTorch backend API per ADR-003 — Phase C1 execution config dataclass
Branch: feature/torchapi
Mapped tests: CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_execution_config.py -vv
Artifacts: plans/active/ADR-003-BACKEND-API/reports/2025-10-20T004233Z/phase_c_execution/{summary.md,design_delta.md,pytest_execution_config_red.log,pytest_execution_config_green.log}

Do Now:
1. ADR-003-BACKEND-API C1.A1 @ plans/active/ADR-003-BACKEND-API/reports/2025-10-20T004233Z/phase_c_execution/plan.md — reconcile execution-config fields vs override_matrix; record decisions in design_delta.md; tests: none.
2. ADR-003-BACKEND-API C1.A3 @ plans/active/ADR-003-BACKEND-API/reports/2025-10-20T004233Z/phase_c_execution/plan.md — author failing pytest coverage (e.g., tests/torch/test_execution_config.py::TestDefaults) and capture RED log via CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_execution_config.py -vv | tee plans/active/ADR-003-BACKEND-API/reports/2025-10-20T004233Z/phase_c_execution/pytest_execution_config_red.log.
3. ADR-003-BACKEND-API C1.A2+C1.A4 @ plans/active/ADR-003-BACKEND-API/reports/2025-10-20T004233Z/phase_c_execution/plan.md — implement PyTorchExecutionConfig in ptycho/config/config.py, update exports/docs/specs, then rerun CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_execution_config.py -vv | tee plans/active/ADR-003-BACKEND-API/reports/2025-10-20T004233Z/phase_c_execution/pytest_execution_config_green.log.
4. ADR-003-BACKEND-API C1.A5 @ plans/active/ADR-003-BACKEND-API/reports/2025-10-20T004233Z/phase_c_execution/plan.md — update phase_c_execution/summary.md, add fix_plan Attempt, and relocate root train_debug.log into the Phase C reports directory; tests: none.

If Blocked: Document unresolved field-mapping questions in design_delta.md, keep C1 checklist items at [ ]/[P], and note blockers + evidence paths in docs/fix_plan.md Attempt.

Priorities & Rationale:
- plans/active/ADR-003-BACKEND-API/reports/2025-10-20T004233Z/phase_c_execution/plan.md — authoritative breakdown for Phase C1 tasks.
- plans/active/ADR-003-BACKEND-API/reports/2025-10-19T232336Z/phase_b_factories/override_matrix.md — canonical execution-knob precedence.
- specs/ptychodus_api_spec.md §4, §6 — defines CONFIG-001 and backend execution contract that spec updates must reflect.
- docs/workflows/pytorch.md §§12–13 — workflow + CLI guidance that must mention the new dataclass.
- docs/findings.md (CONFIG-001, POLICY-001) — non-negotiable requirements for initialization order and torch dependency.

How-To Map:
- Export AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md before running pytest selectors.
- Write RED tests in native pytest style (no unittest.TestCase) asserting dataclass defaults, optional fields, and repr behaviour. Store module under tests/torch/.
- Use field inventory from override_matrix.md §2 to populate design_delta.md (include table of additions/removals and rationale).
- When updating specs/ptychodus_api_spec.md, extend §4.8 Backend Selection + §6 KEY_MAPPINGS with execution-config description; cite new dataclass.
- Refresh docs/workflows/pytorch.md §12 with call-out showing how factories accept execution_config overrides.
- After GREEN run, move train_debug.log (if regenerated) into plans/active/ADR-003-BACKEND-API/reports/2025-10-20T004233Z/phase_c_execution/ and reference it in summary if relevant.

Pitfalls To Avoid:
- Do not skip TDD: capture RED log before implementing dataclass.
- Avoid mutating params.cfg outside populate_legacy_params(); execution config should remain orthogonal.
- Keep dataclass defaults aligned with override_matrix; no ad-hoc values.
- Maintain ASCII-only edits; respect existing doc formatting.
- Ensure pytest commands run with CUDA_VISIBLE_DEVICES="" for deterministic CPU execution.
- Do not leave logs at repo root; archive under plan directory.
- Keep spec/doc updates synchronized (spec vs workflow guide) in same loop.
- Do not touch protected TensorFlow core modules (`ptycho/model.py`, `ptycho/diffsim.py`, `ptycho/tf_helper.py`).
- Update __all__ exports when adding the dataclass.
- When tests fail unexpectedly, capture log and halt instead of blindly fixing.

Pointers:
- plans/active/ADR-003-BACKEND-API/reports/2025-10-20T004233Z/phase_c_execution/plan.md:1
- plans/active/ADR-003-BACKEND-API/reports/2025-10-19T232336Z/phase_b_factories/override_matrix.md:1
- plans/active/ADR-003-BACKEND-API/reports/2025-10-20T002041Z/phase_b3_implementation/summary.md:1
- specs/ptychodus_api_spec.md:40
- docs/workflows/pytorch.md:250

Next Up: C2 wiring — promote execution config through factory payloads and extend execution overrides once C1 is GREEN.
