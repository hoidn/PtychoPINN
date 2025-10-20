Summary: Implement PyTorch config factories and turn the RED test suite green
Mode: TDD
Focus: [ADR-003-BACKEND-API] Standardize PyTorch backend API per ADR-003 — Phase B3 factory implementation
Branch: feature/torchapi
Mapped tests: CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_config_factory.py -vv
Artifacts: plans/active/ADR-003-BACKEND-API/reports/2025-10-20T002041Z/phase_b3_implementation/{summary.md,pytest_factory_training_green.log,pytest_factory_green.log}

Do Now:
1. ADR-003-BACKEND-API B3.A1+B3.A2+B3.A3+B3.A4 @ plans/active/ADR-003-BACKEND-API/reports/2025-10-20T002041Z/phase_b3_implementation/plan.md — implement `create_training_payload()` override precedence + PyTorch dataclass instantiation and `populate_legacy_params()` helper per design docs; tests: none (code changes prepare for targeted run).
2. ADR-003-BACKEND-API B3.A5 @ plans/active/ADR-003-BACKEND-API/reports/2025-10-20T002041Z/phase_b3_implementation/plan.md — capture targeted training subset run via CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_config_factory.py -k "TrainingPayloadStructure or ConfigBridgeTranslation or LegacyParamsPopulation or OverridePrecedence" -vv | tee plans/active/ADR-003-BACKEND-API/reports/2025-10-20T002041Z/phase_b3_implementation/pytest_factory_training_green.log.
3. ADR-003-BACKEND-API B3.B1+B3.B2+B3.B3 @ plans/active/ADR-003-BACKEND-API/reports/2025-10-20T002041Z/phase_b3_implementation/plan.md — implement `infer_probe_size()` legacy handling and `create_inference_payload()` validations (n_groups required, checkpoint existence, overrides audit trail); tests: none.
4. ADR-003-BACKEND-API B3.B4+B3.C1+B3.C2+B3.C3 @ plans/active/ADR-003-BACKEND-API/reports/2025-10-20T002041Z/phase_b3_implementation/plan.md — run full suite CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_config_factory.py -vv | tee plans/active/ADR-003-BACKEND-API/reports/2025-10-20T002041Z/phase_b3_implementation/pytest_factory_green.log, update summary.md with runtime + CONFIG-001 evidence, flip implementation plan B3 row to [x], and append docs/fix_plan Attempt #9.

If Blocked: Document the blocker in plans/active/ADR-003-BACKEND-API/reports/2025-10-20T002041Z/phase_b3_implementation/summary.md, keep B3 rows at [ ]/[P], and note the issue plus log path in docs/fix_plan.md Attempt history.

Priorities & Rationale:
- plans/active/ADR-003-BACKEND-API/reports/2025-10-20T002041Z/phase_b3_implementation/plan.md — authoritative checklist for progressing RED→GREEN.
- plans/active/ADR-003-BACKEND-API/reports/2025-10-20T000736Z/phase_b2_redfix/summary.md — current failure signature to retire via implementation.
- specs/ptychodus_api_spec.md §4 — mandates CONFIG-001 legacy bridge population the factory must enforce.
- docs/workflows/pytorch.md §§5–7 — ensures factory output aligns with workflow expectations (training/inference/stitching).
- override_matrix.md §4 — override precedence rules to codify in implementation.

How-To Map:
- Export AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md for logging consistency.
- When implementing training payload, call update_legacy_dict(params.cfg, tf_config) before touching RawData/bridge helpers; preserve CONFIG-001 order to keep tests deterministic.
- Use helpers in ptycho_torch.config_bridge (e.g., create_training_configs/create_inference_configs) rather than duplicating CLI glue; consult factory_design.md §3 for sequence.
- For `infer_probe_size()`, load NPZ with numpy.load(..., allow_pickle=False); handle missing files with fallback N=64 and document TODO for legacy FORMAT-001 paths.
- Log outputs via tee into the plan directory; do not store artifacts at repo root. After final run, summarise runtime delta (RED 2.1s → GREEN Xs) and key assertions in summary.md.
- Update docs/fix_plan.md with Attempt #9 citing both training and full GREEN logs, and mark implementation plan B3 row `[x]`.

Pitfalls To Avoid:
- Do not bypass CONFIG-001 — update_legacy_dict must run before any loader/raw_data use.
- Leave PyTorchExecutionConfig handling optional (`None`) until Phase C1; add TODOs instead of ad-hoc dataclasses.
- Keep error messages aligned with tests (ValueError for missing n_groups, FileNotFoundError for NPZ path, ValueError for absent wts.h5.zip).
- Avoid reusing global params.cfg without clearing in tests; rely on fixtures that already clear state.
- No direct modifications to protected core modules (`ptycho/model.py`, `ptycho/diffsim.py`, `ptycho/tf_helper.py`).
- Maintain CPU execution by exporting CUDA_VISIBLE_DEVICES="" on every pytest invocation.
- Ensure override precedence honors explicit overrides before execution/default configs; reference override_matrix for ordering.
- Capture logs exactly where specified; no stray files under repo root.
- Adhere to native pytest style; do not reintroduce unittest.TestCase constructs.
- After implementation, re-run full suite only once; rely on targeted selectors during development.

Pointers:
- plans/active/ADR-003-BACKEND-API/reports/2025-10-20T002041Z/phase_b3_implementation/plan.md:1
- plans/active/ADR-003-BACKEND-API/reports/2025-10-19T234600Z/phase_b2_skeleton/summary.md:1
- tests/torch/test_config_factory.py:1
- ptycho_torch/config_factory.py:1
- specs/ptychodus_api_spec.md:40

Next Up:
- Integrate factories into workflows (`ptycho_torch/workflows/components.py`) and CLI wrappers per Phase C once B3 is green.
- Evaluate config bridge tests (`tests/torch/test_config_bridge.py`) to ensure factories provide parity coverage.
