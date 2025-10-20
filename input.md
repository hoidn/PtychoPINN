Summary: Wire PyTorch execution config into factory payloads (Phase C2)
Mode: TDD
Focus: [ADR-003-BACKEND-API] Standardize PyTorch backend API per ADR-003 — Phase C2 factory wiring
Branch: feature/torchapi
Mapped tests: CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_config_factory.py -k ExecutionConfig -vv
Artifacts: plans/active/ADR-003-BACKEND-API/reports/2025-10-20T010900Z/phase_c2_factory_wiring/{summary.md,pytest_factory_execution_red.log,pytest_factory_execution_green.log}

Do Now:
1. ADR-003-BACKEND-API C2.B3 @ plans/active/ADR-003-BACKEND-API/reports/2025-10-20T004233Z/phase_c_execution/plan.md — author failing override tests in tests/torch/test_config_factory.py (TestExecutionConfigOverrides) and capture RED log via CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_config_factory.py -k ExecutionConfig -vv | tee plans/active/ADR-003-BACKEND-API/reports/2025-10-20T010900Z/phase_c2_factory_wiring/pytest_factory_execution_red.log.
2. ADR-003-BACKEND-API C2.B1+C2.B2 @ plans/active/ADR-003-BACKEND-API/reports/2025-10-20T004233Z/phase_c_execution/plan.md — update TrainingPayload/InferencePayload to use PyTorchExecutionConfig, merge overrides into execution_config, then rerun CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_config_factory.py -k ExecutionConfig -vv | tee plans/active/ADR-003-BACKEND-API/reports/2025-10-20T010900Z/phase_c2_factory_wiring/pytest_factory_execution_green.log.
3. ADR-003-BACKEND-API C2.B4 @ plans/active/ADR-003-BACKEND-API/reports/2025-10-20T004233Z/phase_c_execution/plan.md — document override precedence + applied knobs in phase_c_execution/summary.md, mark implementation.md C2 row, and note evidence in docs/fix_plan.md; tests: none.

If Blocked: Capture the failing selector output plus notes about unresolved override precedence in phase_c_execution/summary.md and leave C2 checklist items at [P]; log blocker details + artifact path in docs/fix_plan.md Attempts.

Priorities & Rationale:
- plans/active/ADR-003-BACKEND-API/reports/2025-10-20T004233Z/phase_c_execution/plan.md — authoritative Phase C2 task list.
- ptycho_torch/config_factory.py:109 — factories already GREEN; wiring needs to stay CONFIG-001 compliant.
- tests/torch/test_config_factory.py:40 — existing factory tests to extend with ExecutionConfig overrides.
- plans/active/ADR-003-BACKEND-API/reports/2025-10-19T232336Z/phase_b_factories/override_matrix.md — precedence + default inventory for execution knobs.
- specs/ptychodus_api_spec.md §4.8 — execution config contract that factories must honour.

How-To Map:
- Export AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md before running pytest.
- Implement new pytest class `TestExecutionConfigOverrides` that asserts accelerator/deterministic/num_workers propagate through training/inference payloads; start by asserting NotImplementedError to drive RED.
- When wiring factories, instantiate PyTorchExecutionConfig when execution_config is None and merge overrides with priority order defined in override_matrix.md (explicit overrides > dataclass defaults).
- Record applied knobs in payload.overrides_applied; include accelerator, deterministic, num_workers, enable_progress_bar at minimum.
- Keep CONFIG-001 ordering: call update_legacy_dict before building execution config to avoid stale params.cfg.
- Store both RED and GREEN logs under plans/active/ADR-003-BACKEND-API/reports/2025-10-20T010900Z/phase_c2_factory_wiring/ and summarise decisions in summary.md.

Pitfalls To Avoid:
- Do not mutate execution_config in place after attaching to payload; use dataclasses.replace if tweaks are needed.
- Keep overrides deterministic — no random sampling when selecting defaults.
- Avoid touching workflow helpers yet (Phase C3 scope); limit edits to config_factory + tests.
- Do not drop existing validation tests in tests/torch/test_config_factory.py; extend rather than rewrite.
- Maintain ASCII formatting and keep docstrings concise.
- Ensure CUDA_VISIBLE_DEVICES="" is set for every pytest invocation to stay CPU-only.
- Preserve overrides_applied metadata — do not clear existing entries when adding execution knobs.
- Keep train_debug.log or other logs out of repo root; relocate to plan directory if generated.
- Respect protected modules (ptycho/model.py, ptycho/diffsim.py, ptycho/tf_helper.py).
- Fail fast if execution_config override precedence is ambiguous — document in summary.md.

Pointers:
- plans/active/ADR-003-BACKEND-API/reports/2025-10-20T004233Z/phase_c_execution/plan.md:32
- ptycho_torch/config_factory.py:109
- tests/torch/test_config_factory.py:40
- plans/active/ADR-003-BACKEND-API/reports/2025-10-19T232336Z/phase_b_factories/override_matrix.md:89
- specs/ptychodus_api_spec.md:220

Next Up: C3 execution config threading through workflows/components once factory payload wiring is green.
