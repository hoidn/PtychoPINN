# DEBUG-SIM-LINES-DOSE-001 — Phase D1 Loss Config Diff

## Summary
Compare loss-function configuration between sim_lines_4x and dose_experiments so we know whether MAE/NLL weighting differences explain the amplitude bias before touching normalization.

## Focus
DEBUG-SIM-LINES-DOSE-001 — Isolate sim_lines_4x vs dose_experiments discrepancy (Phase D amplitude bias investigation)

## Branch
paper

## Mapped Tests
`pytest tests/scripts/test_synthetic_helpers_cli_smoke.py::test_sim_lines_pipeline_import_smoke -v`

## Artifacts
`plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T110227Z/`

## Do Now (Phase D1)
- Implement: `plans/active/DEBUG-SIM-LINES-DOSE-001/bin/compare_sim_lines_params.py::main` — extend the parameter diff CLI so it instantiates each sim_lines scenario’s `TrainingConfig`, records `mae_weight`, `nll_weight`, `realspace_weight`, and `realspace_mae_weight`, and surfaces those values in both the Markdown table and JSON diff next to the legacy `dose_experiments_param_scan.md` defaults.
- Run: `AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md python plans/active/DEBUG-SIM-LINES-DOSE-001/bin/compare_sim_lines_params.py --snapshot plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-16T000353Z/sim_lines_4x_params_snapshot.json --dose-config plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-16T000353Z/dose_experiments_param_scan.md --output-markdown plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T110227Z/loss_config_diff.md --output-json plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T110227Z/loss_config_diff.json` and summarize the observed MAE/NLL deltas in `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T110227Z/summary.md`.
- Guard: `AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md pytest tests/scripts/test_synthetic_helpers_cli_smoke.py::test_sim_lines_pipeline_import_smoke -v 2>&1 | tee plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T110227Z/pytest_cli_smoke.log`

## How-To Map
1. Extend the CLI
   - Edit `plans/active/DEBUG-SIM-LINES-DOSE-001/bin/compare_sim_lines_params.py` to:
     - Import `RunParams`, `ScenarioSpec`, and `build_training_config` from `scripts/studies/sim_lines_4x/pipeline.py`.
     - Materialize scenario specs from the snapshot (name, gridsize, probe mode/scale, optional probe_big/mask) and instantiate a real `TrainingConfig` per scenario to read its loss weights.
     - Append `mae_weight`, `nll_weight`, `realspace_weight`, and `realspace_mae_weight` to both the diff JSON payload and Markdown table (keep existing formatting and columns order consistent).
2. Generate the diff artifacts
   - Run the provided `python ... --snapshot ... --dose-config ...` command; the CLI already writes Markdown/JSON — just point both outputs at the new hub paths.
   - Write a brief note (2–3 sentences) in `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T110227Z/summary.md` capturing whether the MAE/NLL weights match or diverge between pipelines.
3. Run the pytest guard
   - Execute the provided pytest selector and tee the output into `pytest_cli_smoke.log` inside the same hub.

## Pitfalls To Avoid
- No edits under `ptycho/` core modules or `scripts/studies/sim_lines_4x/pipeline.py` beyond safe imports; keep logic inside the plan-local script.
- Do not hard-code probe metadata; always derive from the snapshot to keep scenarios in sync when seeds/configs change.
- Ensure the CLI still works for previously generated parameters (backward compatible flags, JSON schema remains stable apart from the new keys).
- Keep Markdown tables human-readable; align new columns with existing heading rows rather than appending ad-hoc prose.
- Use the authoritative pytest selector and archive the log exactly at the mapped path; no ad-hoc tests.
- Respect `AUTHORITATIVE_CMDS_DOC` policy when executing commands.
- Capture stdout/stderr for the CLI command via tee if you need additional logging (store under the hub if saved).
- Do not delete or overwrite prior evidence in `reports/2026-01-20T110227Z/` — append beside existing files.

## If Blocked
- If the CLI import fails (e.g., missing dependency or attribute), capture the traceback in `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T110227Z/blocker.log`, mark Phase D1 blocked in docs/fix_plan.md Attempts History, and note the blocker plus reproduction steps in `plans/active/DEBUG-SIM-LINES-DOSE-001/summary.md`.

## Findings Applied (Mandatory)
- CONFIG-001 — keep legacy `params.cfg` synced before instantiating TrainingConfig/InferenceConfig helpers (already satisfied by importing from the pipeline; no additional singleton edits required).
- NORMALIZATION-001 — intensity normalization must remain symmetric; this comparison isolates whether loss weights (not normalization) are responsible before touching loader math.

## Pointers
- `docs/fix_plan.md:192` — Phase D goals, hypotheses, and course-correction notes.
- `plans/active/DEBUG-SIM-LINES-DOSE-001/implementation.md:320` — Phase D checklist (D1–D4 breakdown).
- `plans/active/DEBUG-SIM-LINES-DOSE-001/bin/compare_sim_lines_params.py:1` — existing snapshot vs legacy parameter diff CLI to extend.
- `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-16T003217Z/comparison_diff.json:1` — prior diff output (pre-loss-weight columns) for reference.

## Next Up (Optional)
1. Phase D2 — add normalization telemetry parity between sim_lines and dose_experiments once loss weights are reconciled.
