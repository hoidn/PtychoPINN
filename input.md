## Summary
Regenerate the Phase D1 loss-configuration evidence with runtime snapshots + refreshed docs so H-LOSS-WEIGHT is provably addressed before touching normalization.

## Focus
DEBUG-SIM-LINES-DOSE-001 — Phase D1 runtime loss-weight capture

## Branch
paper

## Mapped tests
- pytest tests/scripts/test_synthetic_helpers_cli_smoke.py::test_sim_lines_pipeline_import_smoke -v

## Artifacts
plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T121449Z/

## Do Now — DEBUG-SIM-LINES-DOSE-001.D1a-D1c
- Implement: plans/active/DEBUG-SIM-LINES-DOSE-001/bin/compare_sim_lines_params.py::main — extend the `--output-dose-loss-weights` flow so it writes both JSON and Markdown loss-mode snapshots (or add an explicit `--output-dose-loss-weights-markdown` flag), and ensure conditional fields are clearly labeled in the diff/summary to match D1b requirements.
- Implement: rerun the comparison CLI using the existing sim_lines snapshot (`reports/2026-01-16T000353Z/sim_lines_4x_params_snapshot.json`) and legacy param scan (`reports/2026-01-16T000353Z/dose_experiments_param_scan.md`), writing the refreshed `loss_config_diff.{md,json}` plus `dose_loss_weights.{json,md}` and a concise summary under `reports/2026-01-20T121449Z/`, then update that summary to restate the corrected finding.
- Test: AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md pytest tests/scripts/test_synthetic_helpers_cli_smoke.py::test_sim_lines_pipeline_import_smoke -v | tee plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T121449Z/pytest_cli_smoke.log
- Artifacts: plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T121449Z/

## How-To Map
1. Modify `compare_sim_lines_params.py` so that when `--output-dose-loss-weights` is supplied it emits JSON **and** Markdown (either infer the `.md` path automatically or honor a new CLI flag); reuse `build_loss_modes_markdown()` and add explicit labels (e.g., `conditional (loss_fn=mae)`) in the generated text/JSON.
2. Regenerate the diff:
   ```bash
   AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md python plans/active/DEBUG-SIM-LINES-DOSE-001/bin/compare_sim_lines_params.py \
     --snapshot plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-16T000353Z/sim_lines_4x_params_snapshot.json \
     --dose-config plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-16T000353Z/dose_experiments_param_scan.md \
     --output-markdown plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T121449Z/loss_config_diff.md \
     --output-json plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T121449Z/loss_config_diff.json \
     --output-dose-loss-weights plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T121449Z/dose_loss_weights.json \
     --output-dose-loss-weights-markdown plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T121449Z/dose_loss_weights.md \
     --output-legacy-defaults plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T121449Z/legacy_params_cfg_defaults.json
   ```
3. Capture a short Markdown summary under the same hub (reference the updated tables, mention the selector, and restate that default NLL weights rely on `ptycho/params.cfg`).
4. Run the pytest selector above so the CLI import guard stays green.
5. Update docs/fix_plan.md Attempts History (D1 block) plus the initiative summary once the refreshed artifacts exist, citing the new timestamp.

## Pitfalls To Avoid
- Do not mutate the real `ptycho.params.cfg`; always stub and restore `sys.modules['ptycho.params']` exactly as the script currently does.
- Keep all new files under the specified report hub—no stray JSON/Markdown at repo root.
- Make sure the Markdown table clearly labels the MAE branch as conditional so reviewers stop assuming it is the default.
- Preserve deterministic ordering in JSON to keep future diffs stable; use `sort_keys=True` when dumping.
- Avoid touching production modules (`ptycho/*.py`); this work stays confined to plan-local scripts/docs.
- When editing the CLI, don’t introduce import-time side effects (per ANTIPATTERN-001); wrap work inside `main()`.
- Guard against missing files (e.g., snapshot path) with actionable error messages instead of stack traces.
- Keep runtime captures fast by reusing the existing snapshot; no new heavy sim_lines runs are required for this loop.
- Always prefix commands with `AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md` so they inherit the documented environment policy.

## If Blocked
If the legacy script fails to execute (e.g., syntax change), capture the stack trace into `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T121449Z/blocker.log`, note the exact command + loss mode used, and log the blocker in docs/fix_plan.md Attempts History so we can decide whether to fall back to archived cfg snapshots.

## Findings Applied (Mandatory)
- CONFIG-001 — Keep legacy modules isolated/stubbed so params.cfg isn’t corrupted while executing the old init(); never bypass the documented update pattern.
- ANTIPATTERN-001 — Avoid import-time side effects in the CLI (no top-level execution beyond `if __name__ == "__main__"`), ensuring reruns remain deterministic.

## Pointers
- plans/active/DEBUG-SIM-LINES-DOSE-001/implementation.md:330 — D1 checklist describing the runtime loss-mode capture requirements and evidence path.
- plans/active/DEBUG-SIM-LINES-DOSE-001/bin/compare_sim_lines_params.py:419 — Existing loss-mode Markdown helpers to extend when writing the standalone `dose_loss_weights.md` artifact.
- docs/fix_plan.md:200 — Phase D1 Attempts History noting the reviewer-required re-open and outstanding actions.

## Next Up (optional)
If D1 closes quickly, pivot to D2a/D2b normalization parity instrumentation per the plan’s next checklist items.
