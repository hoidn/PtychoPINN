**Summary**: Execute B0f isolation test — run gs1_custom (gridsize=1 + custom probe) and compare metrics against gs1_ideal/gs2_ideal to determine if the amplitude bias is probe-type-specific or workflow-wide.
**Focus**: DEBUG-SIM-LINES-DOSE-001 — B0f isolation test (Phase B0 hypothesis verification)
**Branch**: paper
**Mapped tests**: `AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md pytest tests/scripts/test_synthetic_helpers_cli_smoke.py::test_sim_lines_pipeline_import_smoke -v`
**Artifacts**: `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T102300Z/`

**Do Now**
- Run: Execute the Phase C2 runner for `gs1_custom` scenario (gridsize=1 with the repository's custom probe instead of ideal probe):
  ```bash
  AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md python plans/active/DEBUG-SIM-LINES-DOSE-001/bin/run_phase_c2_scenario.py \
    --scenario gs1_custom \
    --output-dir plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T102300Z/gs1_custom \
    --prediction-scale-source least_squares \
    --group-limit 64 \
    2>&1 | tee plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T102300Z/gs1_custom_runner.log
  ```
- Run: Execute the analyzer to compare gs1_custom against the existing gs1_ideal and gs2_ideal baselines:
  ```bash
  AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md python plans/active/DEBUG-SIM-LINES-DOSE-001/bin/analyze_intensity_bias.py \
    --scenario gs1_custom=plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T102300Z/gs1_custom \
    --scenario gs1_ideal=plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T160000Z/gs1_ideal \
    --scenario gs2_ideal=plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T160000Z/gs2_ideal \
    --output-dir plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T102300Z/ \
    2>&1 | tee plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T102300Z/analyze_intensity_bias.log
  ```
- Run: Guard the imports with the synthetic helpers CLI smoke test:
  ```bash
  AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md pytest tests/scripts/test_synthetic_helpers_cli_smoke.py::test_sim_lines_pipeline_import_smoke -v \
    2>&1 | tee plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T102300Z/pytest_cli_smoke.log
  ```
- Analyze: Compare gs1_custom metrics against gs1_ideal/gs2_ideal in the analyzer output:
  - **If gs1_custom shows similar amplitude bias (~-2.3) and low pearson_r (~0.1)**: Problem is workflow/normalization-level, not probe-type-specific. Next step: investigate loss wiring or normalization math.
  - **If gs1_custom shows significantly better metrics (lower bias, higher pearson_r)**: Problem IS ideal-probe-specific. Next step: investigate H-PROBE-IDEAL-REGRESSION (diff dose_experiments vs sim_lines ideal probe code paths).
- Update: Record the findings in `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T102300Z/summary.md` with:
  1. gs1_custom metrics table (amplitude bias, pearson_r, least_squares scalar)
  2. Comparison against gs1_ideal and gs2_ideal baselines
  3. Decision: which hypothesis branch (probe-specific vs workflow-wide) is confirmed
  4. Recommended next action based on the decision tree
- Update: Mark B0f complete in `plans/active/DEBUG-SIM-LINES-DOSE-001/implementation.md` and record the finding.
- Update: Add a new Attempts entry to `docs/fix_plan.md` under DEBUG-SIM-LINES-DOSE-001 documenting the B0f results.

**How-To Map**
1. The `--scenario gs1_custom` flag selects gridsize=1 with the custom probe (as opposed to `gs1_ideal` which uses the ideal probe). The runner should already support this scenario via its stable profiles.
2. The analyzer compares multiple scenarios side-by-side; include all three (gs1_custom, gs1_ideal, gs2_ideal) so the comparison table is complete.
3. Decision criteria for the output analysis:
   - "Similar" means amplitude bias within ±0.3 of the ideal scenarios (~-2.3 baseline)
   - "Similar" pearson_r means within ±0.05 of ~0.1-0.14 baseline
   - "Significantly better" would be amplitude bias < -1.0 or pearson_r > 0.5

**Pitfalls To Avoid**
1. Do not modify the Phase C2 runner code — this is an evidence-only run using existing tooling.
2. Ensure the scenario name is exactly `gs1_custom` (not `gs1_custom_probe` or similar) to match the runner's stable profile lookup.
3. Keep `--group-limit 64` to avoid GPU OOM while still getting representative statistics.
4. Use `--prediction-scale-source least_squares` for consistency with the prior gs1_ideal/gs2_ideal runs.
5. Do not touch the 2026-01-20T160000Z hub artifacts — those are the baseline for comparison.
6. If the runner fails with an unknown scenario error, check `run_phase_c2_scenario.py::STABLE_PROFILES` for the supported scenario names and adapt accordingly.
7. Archive all logs even if the run fails — the error signature is valuable evidence.

**If Blocked**
- If `gs1_custom` is not a recognized scenario in the runner, check what custom-probe scenarios ARE available by inspecting the `STABLE_PROFILES` dict in `bin/run_phase_c2_scenario.py`. Run whichever custom-probe + gridsize=1 scenario exists.
- If the runner OOMs, reduce `--group-limit` to 32 and retry.
- Record the blocker in `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T102300Z/blocked.log` and update docs/fix_plan.md Attempts.

**Findings Applied (Mandatory)**
- CONFIG-001 — already enforced in the runner as of C4f; no additional bridging needed.
- NORMALIZATION-001 — preserve telemetry outputs for downstream comparison.
- H-PROBE-IDEAL-REGRESSION (B0) — this test determines whether to pursue this hypothesis.
- H-GRIDSIZE-NUMERIC (B0) — already deprioritized since both gs1_ideal and gs2_ideal show identical failure patterns after C4f.

**Pointers**
- `plans/active/DEBUG-SIM-LINES-DOSE-001/implementation.md:218-226` — B0f checklist item
- `plans/active/DEBUG-SIM-LINES-DOSE-001/implementation.md:193-214` — Decision tree for interpreting B0f results
- `plans/active/DEBUG-SIM-LINES-DOSE-001/bin/run_phase_c2_scenario.py` — Phase C2 runner with stable profiles
- `plans/active/DEBUG-SIM-LINES-DOSE-001/bin/analyze_intensity_bias.py` — Multi-scenario analyzer
- `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T160000Z/bias_summary.md` — Baseline gs1_ideal/gs2_ideal metrics

**Next Up (optional)**
1. If gs1_custom shows similar failure → scope H-LOSS-WIRING investigation (instrument loss function inputs)
2. If gs1_custom shows better metrics → scope H-PROBE-IDEAL-REGRESSION (diff ideal probe code paths)
