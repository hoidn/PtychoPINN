# DEBUG-SIM-LINES-DOSE-001 — Phase B1 grouping instrumentation handoff

**Summary:** Capture grouping stats for both SIM-LINES-4X defaults and the legacy dose_experiments constraints so we can prove where the pipelines diverge before touching production code.

**Focus:** DEBUG-SIM-LINES-DOSE-001 — Isolate sim_lines_4x vs dose_experiments discrepancy

**Branch:** paper

**Mapped tests:**
- `pytest tests/scripts/test_synthetic_helpers_cli_smoke.py::test_sim_lines_pipeline_import_smoke -v`

**Artifacts:** `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-16T020000Z/`

---

## Do Now
- DEBUG-SIM-LINES-DOSE-001.B1 — Instrument grouping stats for sim_lines vs dose_experiments parameter regimes
  - Implement: `plans/active/DEBUG-SIM-LINES-DOSE-001/bin/grouping_summary.py::main` — new analysis CLI that loads the captured SIM-LINES snapshot plus `ScenarioSpec` defaults, constructs object/probe inputs via `scripts/studies/sim_lines_4x/pipeline.py`, and drives `RawData.generate_grouped_data` twice (train/test) with user-selectable `gridsize`, `group_count`, and `total_images`. The script should emit both JSON and Markdown summaries describing counts, coord/global-offset min/max, and whether grouping succeeded; if grouping fails (e.g., not enough points for `gridsize=2`), capture the exception message instead of raising so we can document the block.
  - Validate: `pytest tests/scripts/test_synthetic_helpers_cli_smoke.py::test_sim_lines_pipeline_import_smoke -v` (ensures the helper imports stayed stable while adding the plan-local CLI).
  - Artifacts: `grouping_sim_lines_default.{json,md}`, `grouping_dose_experiments_legacy.{json,md}`, and `pytest_sim_lines_pipeline_import.log` under `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-16T020000Z/`.

## How-To Map
1. Write `plans/active/DEBUG-SIM-LINES-DOSE-001/bin/grouping_summary.py` (mark executable) with arguments: `--scenario` (default `gs1_custom`), `--snapshot` (default to `reports/2026-01-16T000353Z/sim_lines_4x_params_snapshot.json`), `--label`, `--output-json`, `--output-markdown`, plus overrides `--gridsize`, `--group-count`, `--neighbor-count`, `--total-images`, `--split-fraction`, `--image-multiplier`, `--group-multiplier`, `--object-seed`, `--sim-seed`, `--probe-mode`, `--probe-scale`, `--probe-big`, `--probe-mask`.
   - Load scenario defaults from the snapshot, compute derived totals via `derive_counts()` (respecting overrides), and use `make_lines_object`/`make_probe`/`normalize_probe_guess` to build the inputs.
   - Use `simulate_nongrid_raw_data` + `split_raw_data_by_axis` to reproduce the pipeline up to grouping; call `RawData.generate_grouped_data` with explicit `gridsize`/`nsamples`/`K` and wrap the call in `try/except` to capture ValueError messages.
   - Summaries should contain run metadata (label, scenario, overrides), per-subset stats (point count, requested vs actual groups, coord/global-offset min/max, nn_indices shape), and an `error` field when grouping failed.
2. Run the CLI for the SIM-LINES baseline (defaults only):
   ```bash
   python plans/active/DEBUG-SIM-LINES-DOSE-001/bin/grouping_summary.py \
     --scenario gs1_custom \
     --snapshot plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-16T000353Z/sim_lines_4x_params_snapshot.json \
     --label sim_lines_default \
     --output-json plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-16T020000Z/grouping_sim_lines_default.json \
     --output-markdown plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-16T020000Z/grouping_sim_lines_default.md
   ```
3. Run the CLI for the legacy dose_experiments constraints (small gridsize=2 run):
   ```bash
   python plans/active/DEBUG-SIM-LINES-DOSE-001/bin/grouping_summary.py \
     --scenario gs1_custom \
     --snapshot plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-16T000353Z/sim_lines_4x_params_snapshot.json \
     --label dose_experiments_legacy \
     --gridsize 2 \
     --group-count 2 \
     --total-images 4 \
     --split-fraction 0.5 \
     --output-json plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-16T020000Z/grouping_dose_experiments_legacy.json \
     --output-markdown plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-16T020000Z/grouping_dose_experiments_legacy.md
   ```
   (Expect the train/test grouping attempt to fail due to insufficient neighbors; that’s acceptable as long as the error text is captured.)
4. Capture pytest evidence:
   ```bash
   AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md \
   pytest tests/scripts/test_synthetic_helpers_cli_smoke.py::test_sim_lines_pipeline_import_smoke -v \
     | tee plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-16T020000Z/pytest_sim_lines_pipeline_import.log
   ```
5. Update `plans/active/DEBUG-SIM-LINES-DOSE-001/summary.md` with a one-paragraph Turn Summary referencing the new grouping summaries and pytest evidence.

## Pitfalls To Avoid
1. Keep the new CLI under `plans/active/.../bin` so production modules remain untouched.
2. Always call `update_legacy_dict(params.cfg, config)` before `RawData.generate_grouped_data` (CONFIG-001) to prevent shape mismatches.
3. Use consistent seeds from `RunParams` (or the CLI overrides) for both modes so differences are attributable to gridsize/grouping, not randomness.
4. Guard the grouping call with try/except and record the exception text; do not crash or swallow the failure silently.
5. Do not emit large RawData dumps—summaries should contain only scalar stats/min/max to keep artifacts light.
6. Respect probe normalization (NORMALIZATION-001): reuse `normalize_probe_guess` rather than introducing ad-hoc scaling.
7. Ensure `group_count`/`total_images` overrides are integers; validate early and emit a helpful error if they are invalid.
8. When computing stats, convert tensors to NumPy on CPU (small slices) to avoid triggering GPU-only code paths.
9. Keep stdout quiet; rely on the Markdown/JSON files for reporting so the summary is reproducible.
10. Re-run the CLI smoke pytest even if the script only lives under `plans/active` so we still satisfy the mapped test requirement.

## If Blocked
- If the grouping script raises an unexpected exception (e.g., import errors, missing snapshot fields), log the stack trace to `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-16T020000Z/blocker.log`, note the scenario + parameters used, and stop; report the issue back to the supervisor.
- If both modes fail before reaching `generate_grouped_data` (e.g., `simulate_nongrid_raw_data` cache errors), capture the failure signature, stash any partial JSON with `"status": "blocked"`, and halt so we can reassess.

## Findings Applied
| Finding ID | Adherence |
|------------|-----------|
| CONFIG-001 | Script explicitly calls `update_legacy_dict` via TrainingConfig before grouping so gridsize/channel state remains synced. |
| MODULE-SINGLETON-001 | No model factories are invoked; analysis stays inside RawData + helper territory to avoid singleton misuse. |
| NORMALIZATION-001 | Probe normalization is delegated to `normalize_probe_guess`, keeping physics/statistical scaling separate. |
| BUG-TF-001 | The CLI records requested gridsize/channel counts and flags mismatches, preventing silent channel drift. |

## Pointers
- `plans/active/DEBUG-SIM-LINES-DOSE-001/implementation.md:83` — Phase B checklist + new B1 instrumentation notes.
- `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-16T003217Z/comparison_draft.md:1` — parameter deltas that informed which overrides to test.
- `scripts/studies/sim_lines_4x/pipeline.py:24` — `ScenarioSpec`, `RunParams`, and `run_scenario` helpers to mirror when constructing inputs.
- `scripts/simulation/synthetic_helpers.py:86` — `simulate_nongrid_raw_data`/`split_raw_data_by_axis` helpers reused by the new CLI.
- `docs/TESTING_GUIDE.md:1` — authoritative pytest usage + CLI evidence rules for archiving logs.

## Next Up (optional)
1. DEBUG-SIM-LINES-DOSE-001.B2 — Probe normalization A/B once the grouping evidence proves which parameters diverge.
