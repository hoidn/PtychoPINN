# Reviewer Findings — Action Required

## Summary
1. Router review cadence skips are broken in sync supervisor/loop runs because the state file never records which actor last executed the reviewer prompt, so the loop will run reviewer twice per iteration whenever `review_every_n` hits.
2. The DEBUG-SIM-LINES-DOSE-001 implementation plan still claims “NaN debugging complete” in the status block even though Phase D amplitude-bias work is active, so the plan’s status/exits no longer match reality.

## Evidence
- `scripts/orchestration/README.md:130-139` documents that review cadence should run the reviewer prompt only once per iteration (galph turn only).
- `scripts/orchestration/supervisor.py:654-657` and `scripts/orchestration/loop.py:357-363` only set `state.last_prompt` when router is enabled; they never write `state.last_prompt_actor`, so `router.deterministic_route` cannot detect that galph already ran the reviewer prompt (the skip guard checks `last_prompt_actor == "galph"`).
- `plans/active/DEBUG-SIM-LINES-DOSE-001/implementation.md:3-34` still advertises “Status: NaN DEBUGGING COMPLETE” and the exit criteria list recon success even though the plan summary + fix ledger reopened Phase D to chase the amplitude bias (underdamped reconstructions are still failing the exit criteria).

## Requested Plan Updates
- Reopen (or spin a quick follow-up to) ORCH-ROUTER-001 so the router state writes both `last_prompt` and `last_prompt_actor` (and add regression tests for sync supervisor/loop to prove reviewer cadence fires exactly once per iteration). Until then the reviewer cadence in the prompt cannot be trusted.
- Update `plans/active/DEBUG-SIM-LINES-DOSE-001/implementation.md` (and docs/fix_plan.md) so the status, exit criteria, and checklist reflect the reopened Phase D amplitude-bias work instead of claiming the initiative is complete.

## Next Steps for Supervisor
1. Assign an owner to land the router state fix (both Python runners + regression tests) and link it to an active plan so the reviewer cadence contract is reliable again.
2. Bring the DEBUG-SIM-LINES plan/fix ledger back in sync with the actual Phase D scope so downstream reviewers don’t assume the initiative was archived.
