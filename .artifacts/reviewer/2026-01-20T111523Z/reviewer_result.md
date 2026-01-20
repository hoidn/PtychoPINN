# Reviewer Report — 2026-01-20T111523Z

## Integration Test Result
- **Outcome:** PASS (first attempt)
- **Command:** `RUN_TS=$(date -u +%Y-%m-%dT%H%M%SZ) RUN_LONG_INTEGRATION=1 INTEGRATION_OUTPUT_DIR=.artifacts/integration_manual_1000_512/${RUN_TS}/output pytest tests/test_integration_manual_1000_512.py -v`
- **Output location:** `.artifacts/integration_manual_1000_512/2026-01-20T111045Z/output`
- **Key error excerpt:** n/a — run succeeded (1 passed in 96.39s)

## Review Scope
- **Baseline → Head:** e7f205d8 → 60cc6815
- **Window:** review_every_n not present; inspected fallback window of last 3 iterations (per instructions).
- **State/log paths:** defaulted to `sync/state.json` and `logs/` (orchestration.yaml absent).
- **Primary evidence reviewed:**
  - `docs/fix_plan.md`, `docs/findings.md`
  - `plans/active/DEBUG-SIM-LINES-DOSE-001/{implementation.md,summary.md}`
  - `plans/active/DEBUG-SIM-LINES-DOSE-001/bin/compare_sim_lines_params.py`
  - `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T110227Z/*`
  - `notebooks/dose.py`
  - `prompts/spec_reviewer.md`, `prompts/arch_reviewer.md`

## Issues Identified
1. **Phase D1 loss-config conclusion is unsupported.** `compare_sim_lines_params.py` now reads `cfg[...]` assignments by scanning `dose_experiments_param_scan.md`, but it blindly records the first assignment even if it sits under a conditional. `notebooks/dose.py:3-34` shows `mae_weight=1`/`nll_weight=0` are only set when `loss_fn == 'mae'`; the default path leaves those keys untouched (implying the same NLL weighting that sim_lines already uses). The new Markdown/JSON diff therefore reports a MAE/NLL inversion that never actually happens, and the plan/fix ledger now cite that faulty conclusion as the primary suspect. Without evaluating the actual `loss_fn` invocation, D1 provides no evidence toward the amplitude-bias root cause.
2. **Plan status contradicts current scope.** `plans/active/DEBUG-SIM-LINES-DOSE-001/implementation.md` still announces “Status: **NaN DEBUGGING COMPLETE** — amplitude bias (separate issue) remains open” at the very top even though docs/fix_plan.md and the summary clearly state Phase D is in progress (amplitude bias now in scope). That mismatch can easily misdirect future loops, especially because the exit criteria already include amplitude/intensity alignment.
3. **Deprecated spec paths linger in prompts.** `prompts/spec_reviewer.md:26-28` and `prompts/arch_reviewer.md:27` still direct agents to `docs/spec-shards/...` even though the authoritative spec root moved to `specs/`. This contradicts docs/index.md (which correctly lists `specs/…`) and risks sending future reviewers to missing files.

