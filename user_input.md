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

---

## 2026-01-20T134958Z Review Addendum

### Summary
1. `docs/fix_plan.md` marks ORCH-ORCHESTRATOR-001 as **done**, but the plan (`plans/active/ORCH-ORCHESTRATOR-001/implementation.md`) still lists the status as “in_progress (Phase E — sync router review cadence parity)” with open checklist boxes. The conflicting status signals make it impossible to tell whether more work is expected.
2. Phase D3 of DEBUG-SIM-LINES-DOSE-001 just uncovered a 12× training-length gap (60 epochs vs 5), yet no retrain hub or Do Now exists for the required gs2_ideal 60-epoch rerun, so the key hypothesis (H-NEPOCHS) remains untested.

### Evidence
- `docs/fix_plan.md:327-429` calls ORCH-ORCHESTRATOR-001 “Status: **done — Phase E sync review cadence parity complete 2026-01-20**”.
- `plans/active/ORCH-ORCHESTRATOR-001/implementation.md:5-20,141-153` still shows “Status: in_progress (Phase E …)” with unchecked tasks `E1–E3`.
- `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T133807Z/{analysis.md,hyperparam_diff.md}` logs the nepochs mismatch and recommends a 60-epoch gs2_ideal rerun, but no new artifacts hub/input.md exists for that work.

### Requested Plan Updates
1. Update whichever artifact is wrong for ORCH-ORCHESTRATOR-001 (plan or fix ledger) so status/exit criteria agree; if the work is done, check off the plan and archive it, otherwise re-open the fix ledger entry with the outstanding tasks.
2. Add a concrete D3b retrain step (input.md + artifacts hub) to DEBUG-SIM-LINES-DOSE-001 so the gs2_ideal 60-epoch verification is scheduled, logged, and tied to an explicit pytest guard before Phase D can close.

### Next Steps for Supervisor
1. Decide whether ORCH-ORCHESTRATOR-001 is officially complete; if yes, mark the plan finished (and archive). If not, re-open docs/fix_plan + summary so other agents don’t assume the orchestrator work is done.
2. Approve a follow-up loop for the D3 gs2_ideal retrain (nepochs=60) and insist on artifacts + analyzer updates so H-NEPOCHS has a definitive pass/fail.
