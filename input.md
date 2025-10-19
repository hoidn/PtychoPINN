Summary: Capture PyTorch config factory RED scaffold and unblock implementation work
Mode: TDD
Focus: [ADR-003-BACKEND-API] Standardize PyTorch backend API per ADR-003 — Phase B2 factory skeleton
Branch: feature/torchapi
Mapped tests: CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_config_factory.py -vv (expected RED)
Artifacts: plans/active/ADR-003-BACKEND-API/reports/2025-10-19T234600Z/phase_b2_skeleton/{summary.md,pytest_factory_red.log}

Do Now:
1. ADR-003-BACKEND-API B2.a @ plans/active/ADR-003-BACKEND-API/reports/2025-10-19T232336Z/phase_b_factories/plan.md — add `ptycho_torch/config_factory.py` skeleton with Option-A import of `PyTorchExecutionConfig`; tests: none.
2. ADR-003-BACKEND-API B2.b @ plans/active/ADR-003-BACKEND-API/reports/2025-10-19T232336Z/phase_b_factories/plan.md — author failing pytest coverage under `tests/torch/test_config_factory.py` and capture RED log via CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_config_factory.py -vv | tee plans/active/ADR-003-BACKEND-API/reports/2025-10-19T234600Z/phase_b2_skeleton/pytest_factory_red.log; tests: CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_config_factory.py -vv.
3. ADR-003-BACKEND-API B2.c @ plans/active/ADR-003-BACKEND-API/reports/2025-10-19T232336Z/phase_b_factories/plan.md — update implementation plan, fix_plan Attempts, and draft `summary.md` under phase_b2_skeleton documenting RED state + outstanding gaps; tests: none.

If Blocked: Record the missing prerequisite (e.g., dataclass attribute absent) in `plans/active/ADR-003-BACKEND-API/reports/2025-10-19T234600Z/phase_b2_skeleton/summary.md`, leave B2 rows `[P]`, and append a fix_plan attempt describing the blocker before stopping.

Priorities & Rationale:
- specs/ptychodus_api_spec.md §4.8 — Backend selection requires CONFIG-001 sync before factories run.
- plans/active/ADR-003-BACKEND-API/implementation.md (B2 row) — Phase guidance now references Option-A decision and new artifact hub.
- plans/active/ADR-003-BACKEND-API/reports/2025-10-19T232336Z/phase_b_factories/factory_design.md §3 — Function signatures and override flow to mirror in skeleton/tests.
- plans/active/ADR-003-BACKEND-API/reports/2025-10-19T232336Z/phase_b_factories/override_matrix.md §5 — Field precedence to encode in test assertions.
- POLICY-001 (docs/findings.md#POLICY-001) — Ensure tests enforce PyTorch mandatory behavior (RuntimeError if torch missing).

How-To Map:
- export AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md before recording commands.
- Create `ptycho_torch/config_factory.py` with module docstring referencing factory_design.md; define public helpers (`create_training_payload`, `create_inference_payload`, `infer_probe_size`, `populate_legacy_params`) that raise NotImplementedError and import `PyTorchExecutionConfig` from `ptycho.config.config` (use type hints only; no logic yet).
- Author `tests/torch/test_config_factory.py` using pytest style: fixtures should instantiate canonical configs via existing helpers, assert factories currently raise NotImplementedError, and encode expected outputs (payload dataclasses, override dict). Keep runtime ≤5s; rely on minimal datasets from TEST-PYTORCH-001 when referencing file paths.
- Run `CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_config_factory.py -vv | tee plans/active/ADR-003-BACKEND-API/reports/2025-10-19T234600Z/phase_b2_skeleton/pytest_factory_red.log` to capture the RED failure (expect NotImplementedError).
- Summarize findings in `plans/active/ADR-003-BACKEND-API/reports/2025-10-19T234600Z/phase_b2_skeleton/summary.md`, noting Option-A dependency, test selector, failure message, and next GREEN objectives.
- Update `plans/active/ADR-003-BACKEND-API/implementation.md` (B2 row `[P]` with log reference) and append docs/fix_plan Attempt #6 reflecting RED completion plus artifact links.

Pitfalls To Avoid:
- Do not implement factory logic yet—stay RED with explicit NotImplementedError.
- Keep imports device/dtype neutral; no torch.cuda calls or implicit GPU selection.
- Avoid touching `ptycho/model.py`, `ptycho/diffsim.py`, `ptycho/tf_helper.py` (protected core logic).
- Ensure pytest file uses native pytest constructs only (no unittest.TestCase mix-in).
- Store all new artefacts under the 2025-10-19T234600Z directory; no logs at repo root.
- Do not commit generated datasets or large binaries—reuse minimal fixtures already in repo.
- Maintain CONFIG-001 order: tests should call `update_legacy_dict` before using legacy modules.
- Record failure message verbatim in summary; do not paraphrase without context.
- Leave TODOs in code only if tied to plan IDs (use comments sparingly).
- Keep runtime budgets in mind; skip slow selectors unless required by plan.

Pointers:
- plans/active/ADR-003-BACKEND-API/reports/2025-10-19T232336Z/phase_b_factories/factory_design.md:120
- plans/active/ADR-003-BACKEND-API/reports/2025-10-19T232336Z/phase_b_factories/override_matrix.md:210
- plans/active/ADR-003-BACKEND-API/reports/2025-10-19T232336Z/phase_b_factories/open_questions.md:1
- plans/active/ADR-003-BACKEND-API/implementation.md:24
- docs/findings.md#L8

Next Up: 1. Promote B2 to GREEN once factory helpers return payloads and tests pass (reuse same selector). 2. Begin C1 dataclass field implementation in `ptycho/config/config.py` after RED evidence is archived.
