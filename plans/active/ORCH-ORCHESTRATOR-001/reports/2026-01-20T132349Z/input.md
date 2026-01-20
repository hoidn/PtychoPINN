Summary: Add combined-mode review cadence regression so orchestrator inherits the router `last_prompt_actor` fix and update the orchestration testing docs.
Focus: ORCH-ORCHESTRATOR-001 — Combined orchestrator entrypoint + shared runner refactor
Branch: paper
Mapped tests: pytest scripts/orchestration/tests/test_orchestrator.py::test_combined_review_last_prompt_actor -v; pytest scripts/orchestration/tests/test_sync_router_review.py -v
Artifacts: plans/active/ORCH-ORCHESTRATOR-001/reports/2026-01-20T132349Z/

Do Now (hard validity contract):
- Implement: scripts/orchestration/tests/test_orchestrator.py::test_combined_review_last_prompt_actor — add a regression that drives `run_combined_iteration` with router review cadence enabled, asserts reviewer runs only on the galph turn, and verifies `state.last_prompt_actor` toggles galph→ralph so ralph selects `main.md`. Update `docs/TESTING_GUIDE.md` and `docs/development/TEST_SUITE_INDEX.md` with the new selector plus wording that combined mode now mirrors the sync router cadence tests.
- Validate: pytest scripts/orchestration/tests/test_orchestrator.py::test_combined_review_last_prompt_actor -v; pytest scripts/orchestration/tests/test_sync_router_review.py -v (guard both entrypoints against regressions)

How-To Map:
1. Extend `scripts/orchestration/tests/test_orchestrator.py` with the new review-cadence test; reuse the tmp_path prompt helpers and ensure the test asserts both `executed` prompt order and `state.last_prompt_actor` transitions.
2. Document the selector in `docs/TESTING_GUIDE.md` §2 and `docs/development/TEST_SUITE_INDEX.md` under the orchestration submodule so reviewers know to run it when router cadence changes.
3. Run the targeted pytest commands with `AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md python -m pytest ...` so logs saved under the artifacts hub prove the selectors pass.

Pitfalls To Avoid:
- Do not touch router selection logic or `scripts/orchestration/router.py`; focus on tests + docs only.
- Keep the new test fully stubbed (tmp_path prompts, no real git or CLI execution).
- Ensure `RouterContext(use_router=True)` is in effect; otherwise `last_prompt_actor` never updates and the test gives false negatives.
- Do not skip CLI/test registry updates; selectors without documentation violate TEST-CLI-001.
- Avoid writing to `logs/` or `tmp/` in the repo when capturing pytest output; send logs to the artifacts hub instead.
- Keep pytest invocations CPU-only and deterministic; no environment-specific skips without justification.

If Blocked:
- Capture the failure signature + command output in `plans/active/ORCH-ORCHESTRATOR-001/reports/2026-01-20T132349Z/blocker.md`, note it in docs/fix_plan.md Attempts History, and ping Galph via galph_memory before stopping.

Findings Applied (Mandatory):
- PYTHON-ENV-001 — invoke python/pytest through PATH (`python -m pytest`) without repo-specific shims.
- TEST-CLI-001 — selector docs must stay in sync with any new orchestration tests; ensure red/green proof via saved pytest logs.

Pointers:
- docs/fix_plan.md:373 — initiative overview, dependencies, and attempts for ORCH-ORCHESTRATOR-001.
- plans/active/ORCH-ORCHESTRATOR-001/implementation.md:1 — Phase E checklist describing the new combined review cadence tests.
- scripts/orchestration/README.md:130 — authoritative router + review cadence behavior that the new regression must enforce.

Next Up (optional): If time remains, rerun the entire orchestrator test module (`pytest scripts/orchestration/tests/test_orchestrator.py -v`) to make sure earlier selectors still pass under the new test coverage.

Doc Sync Plan: After the new test passes, archive both pytest logs under the artifacts hub, then update docs/TESTING_GUIDE.md §2 and docs/development/TEST_SUITE_INDEX.md with the selector reference (already part of Do Now) to keep the registry consistent.
