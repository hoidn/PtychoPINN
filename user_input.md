# Reviewer Findings — Action Required

## Summary
Phase D1 concluded that `dose_experiments` uses MAE loss while `sim_lines_4x` uses NLL, but that comparison is based on a parsing bug. The new `compare_sim_lines_params.py` records the branch inside `dose_experiments`’ `init()` that only fires when `loss_fn == 'mae'`. The default path (`loss_fn='nll'`) leaves the weights untouched, so the Markdown/JSON diff and docs/fix_plan.md now cite a MAE/NLL inversion that may not exist. We need to fix the tooling (or capture actual runtime configs) before reconfiguring loss weights.

## Evidence
- `notebooks/dose.py:3-34` — `cfg['mae_weight'] = 1.` and `cfg['nll_weight'] = 0.` live under `if loss_fn == 'mae'`; defaults remain NLL unless the caller overrides the argument.
- `plans/active/DEBUG-SIM-LINES-DOSE-001/bin/compare_sim_lines_params.py` — Phase D1 change blindly captures the first assignment per key.
- `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T110227Z/loss_config_diff.md` and `docs/fix_plan.md` now promote the incorrect MAE conclusion.

## Requested Plan Update
- Update the Phase D checklist to add a task that validates the actual loss weights used by `dose_experiments` (e.g., record the `loss_fn` argument during the legacy run or parse the `params.dill` artifacts) before touching sim_lines losses.
- Fix or replace `compare_sim_lines_params.py` so conditional assignments in `dose_experiments_param_scan.md` aren’t misinterpreted as defaults.

## Next Steps for Supervisor
1. Decide whether to modify the plan-local CLI (e.g., execute the legacy `init()` with both `loss_fn` options and capture resulting cfg) or to mine existing `params.dill` logs to learn the real loss weights.
2. Ensure docs/fix_plan.md and the plan summary stop claiming MAE/NLL inversion until that evidence exists.
3. Only after the loss mode is verified should we schedule any sim_lines reruns with MAE.
