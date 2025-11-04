Summary: Deprecate legacy ptycho_torch.api entry points by emitting explicit warnings and steering users to the new factory-driven workflows.
Mode: TDD
Focus: [ADR-003-BACKEND-API] Standardize PyTorch backend API per ADR-003
Branch: feature/torchapi
Mapped tests: pytest tests/torch/test_api_deprecation.py::test_example_train_import_emits_deprecation_warning -vv
Artifacts: plans/active/ADR-003-BACKEND-API/reports/2025-10-24T070500Z/phase_e_governance/api_deprecation/2025-10-24T070500Z/{red/pytest_api_deprecation_red.log,green/pytest_api_deprecation_green.log,collect/pytest_api_deprecation_collect.log,summary.md}
Do Now:
- [ADR-003-BACKEND-API] E.C1 @ plans/active/ADR-003-BACKEND-API/reports/2025-10-20T133500Z/phase_e_governance/plan.md — Implement: tests/torch/test_api_deprecation.py::test_example_train_import_emits_deprecation_warning (author RED test expecting DeprecationWarning on legacy API import); tests: pytest tests/torch/test_api_deprecation.py::test_example_train_import_emits_deprecation_warning -vv |& tee plans/active/ADR-003-BACKEND-API/reports/2025-10-24T070500Z/phase_e_governance/api_deprecation/2025-10-24T070500Z/red/pytest_api_deprecation_red.log
- [ADR-003-BACKEND-API] E.C1 @ plans/active/ADR-003-BACKEND-API/reports/2025-10-20T133500Z/phase_e_governance/plan.md — Implement: ptycho_torch/api/__init__.py::warn_legacy_api_import (plus call sites in example_train.py and trainer_api.py to emit DeprecationWarning with migration guidance); tests: pytest tests/torch/test_api_deprecation.py::test_example_train_import_emits_deprecation_warning -vv |& tee plans/active/ADR-003-BACKEND-API/reports/2025-10-24T070500Z/phase_e_governance/api_deprecation/2025-10-24T070500Z/green/pytest_api_deprecation_green.log
If Blocked: Capture stderr/stdout to plans/active/ADR-003-BACKEND-API/reports/2025-10-24T070500Z/phase_e_governance/api_deprecation/2025-10-24T070500Z/blockers.log, note failing command + stack trace in summary.md, and append blocker details to docs/fix_plan.md Attempt entry before stopping.
Priorities & Rationale:
- plans/active/ADR-003-BACKEND-API/reports/2025-10-20T133500Z/phase_e_governance/plan.md (E.C1) requires either thin wrappers or deprecation warnings before Phase E can close.
- plans/active/ADR-003-BACKEND-API/reports/2025-10-20T134500Z/phase_e_governance_adr_addendum/adr_addendum.md:295-334 defers the legacy API decision to Phase E.C1 with preference for user-facing guidance.
- docs/workflows/pytorch.md:188-196 flags ptycho_torch/api as deprecated surfaces needing migration instructions; this loop supplies actionable warnings.
- specs/ptychodus_api_spec.md:300-307 documents CLI logger deprecation semantics; aligning package-level warnings keeps backwards compatibility expectations consistent.
How-To Map:
- export AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md
- ARTIFACT_ROOT=plans/active/ADR-003-BACKEND-API/reports/2025-10-24T070500Z/phase_e_governance/api_deprecation/2025-10-24T070500Z
- mkdir -p "$ARTIFACT_ROOT/red" "$ARTIFACT_ROOT/green" "$ARTIFACT_ROOT/collect"
- CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_api_deprecation.py::test_example_train_import_emits_deprecation_warning -vv | tee "$ARTIFACT_ROOT/red/pytest_api_deprecation_red.log"  # expect fail until warning implemented
- After implementation, rerun the selector with same command teeing to "$ARTIFACT_ROOT/green/pytest_api_deprecation_green.log"
- After GREEN, run CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_api_deprecation.py::test_example_train_import_emits_deprecation_warning -vv --collect-only | tee "$ARTIFACT_ROOT/collect/pytest_api_deprecation_collect.log"
- Draft "$ARTIFACT_ROOT/summary.md" with warning message text, files touched, pytest results, and follow-up actions; update docs/fix_plan.md Attempt entry and flip plan row E.C1 guidance accordingly.
Pitfalls To Avoid:
- Do not delete legacy API modules; emit warnings and leave behavior unchanged aside from messaging.
- Use warnings.warn(..., DeprecationWarning, stacklevel=2) so callers see accurate stack origin.
- Ensure tests clear sys.modules or rely on importlib.reload so warnings fire reliably across reruns.
- Keep warning message consistent across modules; centralize text in ptycho_torch/api/__init__.py.
- Avoid hardcoding filesystem paths in warnings—reference CLI entry points instead.
- Do not run full pytest suite; stick to mapped selector to limit runtime.
- Clean up new artifact directories only after summary captured; no tmp/ leftovers.
- Leave MLflow behavior untouched; note any gaps in summary instead of patching here.
- Maintain ASCII in new files; no Unicode characters.
- Keep new tests pytest-native (no unittest mix-ins).
Findings Applied (Mandatory):
- CONFIG-002 — Maintain execution-config isolation; warning should not mutate params.cfg.
- POLICY-001 — PyTorch dependency remains mandatory; no optional imports introduced.
Pointers:
- plans/active/ADR-003-BACKEND-API/reports/2025-10-20T133500Z/phase_e_governance/plan.md#L60
- plans/active/ADR-003-BACKEND-API/reports/2025-10-20T134500Z/phase_e_governance_adr_addendum/adr_addendum.md#L295
- docs/workflows/pytorch.md:188
- specs/ptychodus_api_spec.md:300
Doc Sync Plan (Conditional): After GREEN, update docs/TESTING_GUIDE.md §2 selector table and docs/development/TEST_SUITE_INDEX.md (Torch unit tests) with the new api_deprecation selector; attach diff references in summary.md after verifying pytest --collect-only output stored at "$ARTIFACT_ROOT/collect/pytest_api_deprecation_collect.log".
Mapped Tests Guardrail: Confirm pytest --collect-only tests/torch/test_api_deprecation.py::test_example_train_import_emits_deprecation_warning -vv collects ≥1 test case; if collection fails, pause implementation and repair the test module first.
Next Up: E.C2 ledger + plan closure once deprecation warning lands.
