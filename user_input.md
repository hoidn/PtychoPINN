# Reviewer Findings — Action Required

## Summary
1. Phase D1 in `DEBUG-SIM-LINES-DOSE-001` is marked complete even though its corrective sub-tasks (D1a–D1c) remain undone, so the plan currently obscures the reopened loss-weight investigation.
2. The reviewer prompt requires `orchestration.yaml` (for `router.review_every_n`, `state_file`, and `logs_dir`) but that file does not exist anywhere in the repo, forcing reviewers to guess the cadence each time.

## Evidence
- `plans/active/DEBUG-SIM-LINES-DOSE-001/implementation.md:320-337` shows D1 checked off while simultaneously describing pending D1a–D1c work and a “pending guard”, so the plan’s status gating is inconsistent with reality.
- `prompts/reviewer.md:35-37` instructs reviewers to read `orchestration.yaml`, yet `find . -name 'orchestration.yaml'` returns nothing in the workspace, so there is no authoritative cadence/config file to consult.

## Requested Plan Updates
- Re-open Phase D1 in `plans/active/DEBUG-SIM-LINES-DOSE-001/implementation.md` (and docs/fix_plan.md) until the runtime loss-weight captures, CLI fix, and summary rewrite are actually delivered, with explicit selectors/logs for each D1a–D1c action.
- Add or document the missing orchestration configuration: either commit the expected `orchestration.yaml` (with router.review_every_n, state_file, logs_dir) or update all reviewer-facing prompts/docs to reference the real source of truth so future reviews don’t rely on a nonexistent file.

## Next Steps for Supervisor
1. Decide who owns the D1 remediation and ensure the plan + fix ledger reflect that ownership, completion criteria, and evidence paths.
2. Provide an authoritative orchestration config reference (file or doc) and update prompts to match, so reviewers can follow the cadence instructions without manual discovery.
