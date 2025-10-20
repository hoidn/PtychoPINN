Summary: Re-establish a genuine RED baseline for the config factory tests
Mode: TDD
Focus: [ADR-003-BACKEND-API] Standardize PyTorch backend API per ADR-003 — Phase B2 factory skeleton
Branch: feature/torchapi
Mapped tests: CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_config_factory.py -vv (expected RED)
Artifacts: plans/active/ADR-003-BACKEND-API/reports/2025-10-20T000736Z/phase_b2_redfix/{summary.md,pytest_factory_redfix.log}

Do Now:
1. ADR-003-BACKEND-API B2.a+B2.b @ plans/active/ADR-003-BACKEND-API/reports/2025-10-19T232336Z/phase_b_factories/plan.md — remove the `pytest.raises(NotImplementedError)` guards in `tests/torch/test_config_factory.py`, restore the commented assertions, keep the factory stubs raising `NotImplementedError`, then capture the RED failure via CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_config_factory.py -vv | tee plans/active/ADR-003-BACKEND-API/reports/2025-10-20T000736Z/phase_b2_redfix/pytest_factory_redfix.log.
2. ADR-003-BACKEND-API B2.c @ plans/active/ADR-003-BACKEND-API/reports/2025-10-19T232336Z/phase_b_factories/plan.md — update `plans/active/ADR-003-BACKEND-API/reports/2025-10-20T000736Z/phase_b2_redfix/summary.md` with the failing selector output and note that Phase B2 remains RED; refresh `plans/active/ADR-003-BACKEND-API/implementation.md` B2 guidance if additional clarifications are needed (state stays [P]). tests: none.
3. ADR-003-BACKEND-API ledger @ docs/fix_plan.md — append Attempt #7 documenting the RED failure, new artifact paths, and outstanding work before GREEN. tests: none.

If Blocked: Document the blocker (e.g., unexpected GREEN pass, import errors) in `plans/active/ADR-003-BACKEND-API/reports/2025-10-20T000736Z/phase_b2_redfix/summary.md`, keep B2 rows at [P], and add the blocker details to docs/fix_plan.md instead of proceeding.

Priorities & Rationale:
- plans/active/ADR-003-BACKEND-API/reports/2025-10-19T232336Z/phase_b_factories/plan.md §B2 — mandates a failing pytest baseline before GREEN work starts.
- plans/active/ADR-003-BACKEND-API/implementation.md:24 — supervisor note flags that the current selector reports 19 passed and must be converted to a true RED state.
- plans/active/ADR-003-BACKEND-API/reports/2025-10-19T234600Z/phase_b2_skeleton/summary.md — now carries supervisor warning to re-run without `pytest.raises` guards.
- docs/findings.md#POLICY-001 — PyTorch backend remains mandatory; ensure tests continue to honour torch availability checks during RED run.
- docs/TESTING_GUIDE.md — reinforces TDD requirement that failing tests precede implementation.

How-To Map:
- export AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md for logging.
- In `tests/torch/test_config_factory.py`, delete the `with pytest.raises(NotImplementedError, ...)` wrappers and uncomment/assert the expectations already sketched in comments; leave the factory stubs untouched so the selector fails on NotImplementedError.
- Capture the failing run with CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_config_factory.py -vv | tee plans/active/ADR-003-BACKEND-API/reports/2025-10-20T000736Z/phase_b2_redfix/pytest_factory_redfix.log (expect non-zero exit and NotImplementedError trace).
- Summarise failure signature, key stack trace lines, and next GREEN objectives in `plans/active/ADR-003-BACKEND-API/reports/2025-10-20T000736Z/phase_b2_redfix/summary.md`; reference plan §B2 and POLICY-001 in the narrative.
- Update docs/fix_plan.md Attempts History for ADR-003-BACKEND-API with the new log path and state that B2 remains RED pending implementation; keep Implementation Plan B2 row at [P].

Pitfalls To Avoid:
- Do not implement factory logic yet—this loop ends with failing tests.
- Keep runtime artefacts in the 2025-10-20T000736Z directory; do not reintroduce `train_debug.log` at repo root.
- Ensure pytest uses native style only; no unittest.TestCase reintroductions.
- Maintain CONFIG-001 ordering when the tests eventually go GREEN (document reminder in summary, no code changes now).
- Avoid mutating PyTorchExecutionConfig imports; Option A decision stands.
- Do not edit protected core files (`ptycho/model.py`, `ptycho/diffsim.py`, `ptycho/tf_helper.py`).
- Confirm the failing log captures the NotImplementedError message verbatim for traceability.
- Leave commented guidance in tests terse; rely on plan docs for extended rationale.
- Keep commit scope limited to tests/plan/ledger updates for this RED loop.
- Verify Git status is clean after updates; no stray binaries or caches.

Pointers:
- tests/torch/test_config_factory.py:1
- ptycho_torch/config_factory.py:1
- plans/active/ADR-003-BACKEND-API/reports/2025-10-20T000736Z/phase_b2_redfix/
- plans/active/ADR-003-BACKEND-API/reports/2025-10-19T232336Z/phase_b_factories/plan.md:47
- docs/TESTING_GUIDE.md:1

Next Up: 1. Once RED log captured, implement factory logic per plan B3.a and drive tests to GREEN. 2. After GREEN, refit config bridge tests (B3.b/B3.c) before touching CLI wiring.
