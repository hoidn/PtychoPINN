Summary: Thread PyTorch execution config through workflow helpers and restore exports (Phase C3)
Mode: TDD
Focus: [ADR-003-BACKEND-API] Standardize PyTorch backend API per ADR-003 — Phase C3 workflow integration
Branch: feature/torchapi
Mapped tests: CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_workflows_components.py -k execution_config -vv
Artifacts: plans/active/ADR-003-BACKEND-API/reports/2025-10-20T025643Z/phase_c3_workflow_integration/{plan.md,summary.md,pytest_workflows_execution_red.log,pytest_workflows_execution_green.log}

Do Now:
1. ADR-003-BACKEND-API C3.A1+C3.C3 @ plans/active/ADR-003-BACKEND-API/reports/2025-10-20T025643Z/phase_c3_workflow_integration/plan.md — reintroduce `__all__` exports, author failing workflow tests, capture RED via CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_workflows_components.py -k execution_config -vv | tee plans/active/ADR-003-BACKEND-API/reports/2025-10-20T025643Z/phase_c3_workflow_integration/pytest_workflows_execution_red.log (tests: CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_workflows_components.py -k execution_config -vv).
2. ADR-003-BACKEND-API C3.A2+C3.A3+C3.B1+C3.B2 @ plans/active/ADR-003-BACKEND-API/reports/2025-10-20T025643Z/phase_c3_workflow_integration/plan.md — thread execution config through `_train_with_lightning` and inference helpers, then rerun CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_workflows_components.py -k execution_config -vv | tee plans/active/ADR-003-BACKEND-API/reports/2025-10-20T025643Z/phase_c3_workflow_integration/pytest_workflows_execution_green.log.
3. ADR-003-BACKEND-API C3.D1+C3.D3 @ plans/active/ADR-003-BACKEND-API/reports/2025-10-20T025643Z/phase_c3_workflow_integration/plan.md — document trainer/inference wiring in phase_c_execution/summary.md, relocate root-level train_debug.log into the C3 report folder (or delete if duplicate), and log Attempt #14 in docs/fix_plan.md (tests: none).

If Blocked: Capture the failing workflow selector output with notes in phase_c3_workflow_integration/summary.md, leave C3 tasks at [P], and record the blocker plus log path in docs/fix_plan.md Attempts.

Priorities & Rationale:
- plans/active/ADR-003-BACKEND-API/reports/2025-10-20T025643Z/phase_c3_workflow_integration/plan.md — authoritative C3 checklist (IDs C3.A1–C3.D3).
- plans/active/ADR-003-BACKEND-API/reports/2025-10-20T004233Z/phase_c_execution/summary.md — updated checkpoints highlighting `__all__` regression and log hygiene.
- ptycho_torch/workflows/components.py: `_train_with_lightning` / inference helpers accept new execution config wiring.
- tests/torch/test_workflows_components.py — target module for new TDD coverage.
- specs/ptychodus_api_spec.md §4.8 — backend selection contract; ensure CONFIG-001 order preserved.

How-To Map:
- Set AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md before running pytest.
- Implement workflow tests under `tests/torch/test_workflows_components.py` (use pytest fixtures already in module); add markers if runtime > 5s.
- For RED run: `CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_workflows_components.py -k execution_config -vv | tee plans/active/ADR-003-BACKEND-API/reports/2025-10-20T025643Z/phase_c3_workflow_integration/pytest_workflows_execution_red.log`.
- After wiring Trainer/inference updates, rerun the same selector and store output as `.../pytest_workflows_execution_green.log`.
- When relocating `train_debug.log`, move it to `plans/active/ADR-003-BACKEND-API/reports/2025-10-20T025643Z/phase_c3_workflow_integration/train_debug.log` (or remove if redundant) before committing.
- Update `phase_c_execution/summary.md` and `plans/active/ADR-003-BACKEND-API/implementation.md` to mark C3 tasks complete after GREEN proof; append Attempt entry in `docs/fix_plan.md` with artifact links.

Pitfalls To Avoid:
- Do not mutate factories or config bridge ordering—`update_legacy_dict` must remain before workflow imports.
- Keep Lightning Trainer kwargs CPU-safe (accelerator='cpu' unless GPU available); guard GPU-only knobs with skips.
- Preserve deterministic behaviour: ensure seed handling unaffected when passing execution config.
- Avoid dropping existing workflow tests; extend file rather than rewriting sections wholesale.
- Keep all logs in the C3 report directory; no artifacts at repo root.
- Maintain ASCII-only edits; respect protected modules (ptycho/model.py, ptycho/diffsim.py, ptycho/tf_helper.py).
- Capture RED log before implementation per TDD; do not skip RED phase.
- Ensure __all__ export restoration includes PyTorchExecutionConfig alongside existing names.
- Only run full pytest module once GREEN; rely on targeted selector for development.
- When editing tests, stay within pytest style (no unittest imports).

Pointers:
- plans/active/ADR-003-BACKEND-API/reports/2025-10-20T025643Z/phase_c3_workflow_integration/plan.md#L1
- plans/active/ADR-003-BACKEND-API/reports/2025-10-20T004233Z/phase_c_execution/summary.md#L40
- ptycho_torch/workflows/components.py#L120
- tests/torch/test_workflows_components.py#L20
- specs/ptychodus_api_spec.md#L220

Next Up:
1. ADR-003-BACKEND-API Phase C4 — expose execution knobs via CLI wrappers once workflows are green.
