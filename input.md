## Summary
Instrument the sim_lines runner/analyzer so normalization-stage ratios are explicit and capture a dose-like scenario alongside gs1_ideal to validate H-NORMALIZATION against the spec.

## Focus
DEBUG-SIM-LINES-DOSE-001 — Isolate sim_lines_4x vs dose_experiments discrepancy (Phase D2 normalization parity)

## Branch
paper

## Mapped tests
- pytest tests/scripts/test_synthetic_helpers_cli_smoke.py::test_sim_lines_pipeline_import_smoke -v

## Artifacts
plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T114126Z/

## Do Now — DEBUG-SIM-LINES-DOSE-001.D2a-D2b
- Implement: plans/active/DEBUG-SIM-LINES-DOSE-001/bin/run_phase_c2_scenario.py::write_intensity_stats_outputs (plus its record_intensity_stage call sites) and plans/active/DEBUG-SIM-LINES-DOSE-001/bin/analyze_intensity_bias.py::build_normalization_summary — add stage-ratio metadata, capture the `normalize_data` gain per scenario, and surface the ratio table + "largest drop" marker in the Markdown summary so we can cite `specs/spec-ptycho-core.md §Normalization Invariants` directly.
- Implement: regenerate evidence by rerunning `run_phase_c2_scenario.py` twice (stable gs1_ideal profile + a dose_legacy_gs2 override using the custom probe with `--gridsize 2 --probe-big true --probe-mask false --probe-scale 4 --base-total-images 256 --group-count 128 --group-limit 128 --neighbor-count 4 --prediction-scale-source none`) and run `analyze_intensity_bias.py` against the new scenario hubs so normalization ratios for both pipelines live under `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T114126Z/`.
- Test: AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md pytest tests/scripts/test_synthetic_helpers_cli_smoke.py::test_sim_lines_pipeline_import_smoke -v | tee plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T114126Z/pytest_cli_smoke.log
- Artifacts: plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T114126Z/

## How-To Map
1. Edit the runner/analyzer files listed above so `intensity_stats.json` includes `ratios` + `normalize_gain` fields (raw→grouped→normalized→prediction) and the Markdown summary prints a table plus a note about the first stage that violates symmetry.
2. Re-run the sim_lines baseline (gs1 stable profile) with:
   ```bash
   AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md python plans/active/DEBUG-SIM-LINES-DOSE-001/bin/run_phase_c2_scenario.py \
     --scenario gs1_ideal \
     --output-dir plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T114126Z/gs1_ideal \
     --group-limit 64 --nepochs 5 --prediction-scale-source least_squares
   ```
3. Run the dose-like override (custom probe, gridsize=2) to mimic the legacy normalization path:
   ```bash
   AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md python plans/active/DEBUG-SIM-LINES-DOSE-001/bin/run_phase_c2_scenario.py \
     --scenario gs1_custom \
     --output-dir plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T114126Z/dose_legacy_gs2 \
     --gridsize 2 --probe-mode custom --probe-scale 4 --probe-big true --probe-mask false \
     --base-total-images 256 --group-count 128 --group-limit 128 --neighbor-count 4 \
     --image-multiplier 1 --group-multiplier 1 --prediction-scale-source none --nepochs 5
   ```
4. Summarize both scenarios via the analyzer so the Markdown/JSON comparison highlights the ratio deltas:
   ```bash
   AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md python plans/active/DEBUG-SIM-LINES-DOSE-001/bin/analyze_intensity_bias.py \
     --scenario gs1_ideal=plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T114126Z/gs1_ideal \
     --scenario dose_legacy_gs2=plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T114126Z/dose_legacy_gs2 \
     --output-dir plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T114126Z
   ```
5. Run the pytest selector noted above and keep all logs (runner, analyzer, pytest) under the same report hub alongside `bias_summary.{json,md}`.

## Pitfalls To Avoid
- Keep CONFIG-001 intact: every new runner path must still call `update_legacy_dict(params.cfg, config)` before touching loaders or inference helpers.
- Do not change production modules; all instrumentation stays under plans/active/DEBUG-SIM-LINES-DOSE-001/bin/.
- When overriding the dose-like scenario, ensure `group_count <= min(train_count, test_count)` so grouping doesn’t crash; adjust `base_total_images` instead of hacking RawData internals.
- Stage ratio math must treat zero/NaN means defensively (avoid divide-by-zero when gs2 NaNs appear).
- Reuse existing artifact helpers (write_intensity_stats_outputs, bias summary) rather than inventing new ad-hoc formats.
- Keep GPU usage modest (stable profiles + capped group_limit) so runs finish quickly and don’t resurrect past OOM/NaN regressions.
- Ensure analyzer Markdown cites `specs/spec-ptycho-core.md §Normalization Invariants`; no paraphrased math without references.
- Only touch plan-local directories; never dump intermediate `.npy` under repo root outside the reports hub.
- Record CLI stdout/stderr for both runner invocations; missing logs make it impossible to audit ratios later.
- Treat `plans/.../dose_legacy_gs2` as evidence only—do not wire those overrides into shipped scripts.

## If Blocked
Capture the failing command output in `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T114126Z/blocker.log`, update docs/fix_plan.md Attempts History with the blocker signature (e.g., grouping still unstable for the dose overrides), and ping me so we can either relax the overrides or fall back to archived legacy data before proceeding.

## Findings Applied (Mandatory)
- NORMALIZATION-001 — Stage-ratio instrumentation must demonstrate whether the RawData→normalized pipeline preserves the symmetry mandated by the spec; highlight where the ~2.5× drop occurs.
- CONFIG-001 / SIM-LINES-CONFIG-001 — All runner/analyzer edits must keep the `update_legacy_dict` contract so params.cfg mirrors the Training/InferenceConfig before loader/inference work; do not regress the NaN fix while instrumenting normalization.

## Pointers
- specs/spec-ptycho-core.md:86 — Normative normalization flow and symmetry requirements (cite this when adding ratio tables).
- plans/active/DEBUG-SIM-LINES-DOSE-001/implementation.md:338 — D2 checklist spelling out telemetry + dose-legacy capture expectations for this loop.

## Next Up (optional)
- If this finishes early, expand D2c by adding gs2_ideal to the new ratio tables or begin the D3 hyperparameter delta audit (neighbor_count / nepochs) per the implementation plan.
