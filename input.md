Summary: Fix sync supervisor/loop router review cadence so reviewer fires once per iteration by persisting `last_prompt_actor` and adding regression tests.
Focus: ORCH-ROUTER-001 — Router prompt + orchestration dispatch layer
Branch: paper
Mapped tests: pytest scripts/orchestration/tests/test_sync_router_review.py::TestSyncRouterReview::test_review_runs_once -v; pytest scripts/orchestration/tests/test_router.py -v
Artifacts: plans/active/ORCH-ROUTER-001/reports/2026-01-20T130941Z/
Do Now:
- Implement:
  - scripts/orchestration/supervisor.py::main — when `args.use_router` and sync-via-git are active, persist both `last_prompt` and `last_prompt_actor="galph"` before stamping state so deterministic routing can skip the reviewer on the next ralph turn.
  - scripts/orchestration/loop.py::main — mirror the supervisor change so the ralph turn records `last_prompt_actor="ralph"` whenever router selection runs, ensuring state.json parity across actors.
  - scripts/orchestration/tests/test_sync_router_review.py::TestSyncRouterReview::test_review_runs_once — add a new regression module that stubs the git/prompt executors, simulates a review cadence hit on galph, and asserts the second actor skips reviewer selection because `last_prompt_actor` is present; include an additional test that fails without the new state annotations to prove coverage.
- Validate: pytest scripts/orchestration/tests/test_sync_router_review.py::TestSyncRouterReview::test_review_runs_once -v; pytest scripts/orchestration/tests/test_router.py -v (after edits)
How-To Map:
- Edit both CLI entry points right after `_select_prompt` returns: reuse the existing `selected_prompt` variable and set `st.last_prompt_actor = "galph"`/`"ralph"` in the same branch where `last_prompt` is written so stamping captures the actor before pushing state.
- Keep deterministic routing unchanged; the skip logic already lives in scripts/orchestration/router.py:88-112 — only the state annotations are missing in sync mode.
- Author tests under scripts/orchestration/tests/test_sync_router_review.py using temporary directories and fake executors (see test_orchestrator.py for patterns). Seed OrchestrationState(iteration divisible by review_every_n) so galph hits reviewer and ralph confirms skip; assert `state.last_prompt_actor` toggles.
- Run `AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md pytest scripts/orchestration/tests/test_sync_router_review.py::TestSyncRouterReview::test_review_runs_once -v` to exercise the new module, then rerun the existing router suite via `AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md pytest scripts/orchestration/tests/test_router.py -v`.
- Archive logs for both selectors plus git diffs under plans/active/ORCH-ROUTER-001/reports/2026-01-20T130941Z/ (name them pytest_sync_router_review.log, pytest_router.log, git_diff.txt).
Pitfalls To Avoid:
- Do not mutate state when router is disabled; guard writes under `if args.use_router`.
- Avoid touching combined orchestrator files—this loop only fixes sync supervisor/loop; orchestrator stays blocked until this lands.
- Keep router prompt execution network-neutral (no real CLI calls) inside tests; stub executors to avoid accidental subprocess launches.
- Maintain deterministic prompt allowlist/paths; use tmp_path / "prompts" fixtures like the existing router/orchestrator tests.
- Ensure new tests do not rely on global git state—use temporary state.json paths and avoid touching real repos.
- Keep `AUTHORITATIVE_CMDS_DOC` set via env when running pytest selectors to honor PYTHON-ENV-001.
- Capture artifacts only under the provided reports hub; no stray files in repo root.
- Update docs/test indexes after tests pass; do not skip the registry work.
If Blocked:
- If router skip logic still triggers reviewer twice after the state writes, capture the exact state.json before/after plus pytest failure logs in the artifacts directory and log the blocker in docs/fix_plan.md Attempts History before stopping.
Findings Applied:
- No relevant findings in the knowledge base (router review cadence has no prior entry).
Pointers:
- scripts/orchestration/README.md:130 — authoritative description of review cadence behavior (reviewer should run only once per iteration).
- scripts/orchestration/router.py:88 — deterministic routing skip guard that needs `last_prompt_actor` populated.
- plans/active/ORCH-ROUTER-001/implementation.md:8 — Phase E checklist for this fix.
- docs/fix_plan.md:327 — current ledger entry describing the reopened focus and reporting requirements.
Next Up:
- Once sync supervisor/loop are fixed, unblock ORCH-ORCHESTRATOR-001 and rerun scripts/orchestration/tests/test_orchestrator.py to ensure combined mode inherits the new state annotations.
Doc Sync Plan:
- After the new test module lands, run `AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md pytest --collect-only scripts/orchestration/tests/test_sync_router_review.py -q` and archive the log.
- Append the selector to docs/TESTING_GUIDE.md §2 and docs/development/TEST_SUITE_INDEX.md under the orchestration section so the registry reflects the new coverage.
- Reference the collect-only log plus doc diffs in the artifacts hub before marking the loop complete.
