# DEBUG-SIM-LINES-DOSE-001 — Phase A: capture sim_lines_4x param snapshot

**Summary:** Capture a deterministic sim_lines_4x parameter snapshot and inventory dose_experiments defaults to start the discrepancy triage.

**Focus:** DEBUG-SIM-LINES-DOSE-001 — Isolate sim_lines_4x vs dose_experiments discrepancy

**Branch:** paper

**Mapped tests:**
- `pytest tests/scripts/test_synthetic_helpers_cli_smoke.py::test_sim_lines_pipeline_import_smoke -v`

**Artifacts:** `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-15T235900Z/`

---

## Do Now

- DEBUG-SIM-LINES-DOSE-001.A0 + A1 + A3
  - Implement: `scripts/tools/collect_sim_lines_4x_params.py` — new CLI that emits a JSON snapshot for all four sim_lines_4x scenarios (RunParams, ScenarioSpec, derived counts, probe_mask/probe_big defaults, nphotons, neighbor_count, reassemble_M, and CUSTOM_PROBE_PATH).
  - Evidence: capture dose_experiments defaults via `git ls-tree`/`git show` into `dose_experiments_param_scan.md` under the artifacts hub.
  - Validate: run the mapped pytest selector and archive logs under the artifacts hub.

## How-To Map
1. Create `scripts/tools/collect_sim_lines_4x_params.py` with argparse:
   - `--output <path>` (required) writes JSON.
   - Builds the four ScenarioSpec definitions (gs1/gs2 x ideal/custom) and RunParams defaults from `scripts.studies.sim_lines_4x.pipeline`.
   - Resolves probe defaults using the same rules as `run_scenario` (idealized => probe_big=False, probe_mask=True; custom => defaults from `ptycho.config.config.ModelConfig`).
   - Emits a single JSON payload with `scenarios` list + top-level `run_params` + `custom_probe_path`.
2. Run the snapshot tool:
   ```bash
   python scripts/tools/collect_sim_lines_4x_params.py \
     --output plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-15T235900Z/sim_lines_4x_params_snapshot.json
   ```
3. Inventory dose_experiments defaults (if the ref exists) and store notes:
   ```bash
   git ls-tree -r dose_experiments \
     > plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-15T235900Z/dose_experiments_tree.txt

   # For each relevant file, extract defaults and summarize them in a short table.
   git show dose_experiments:<path> \
     >> plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-15T235900Z/dose_experiments_param_scan.md
   ```
4. Run the mapped pytest selector:
   ```bash
   pytest tests/scripts/test_synthetic_helpers_cli_smoke.py::test_sim_lines_pipeline_import_smoke -v \
     | tee plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-15T235900Z/pytest_sim_lines_pipeline_import.log
   ```

## Pitfalls To Avoid
1. Do not run full training/inference loops in this phase; keep it metadata-only.
2. Avoid importing TensorFlow-heavy modules in the new snapshot tool.
3. Keep the snapshot tool deterministic (no RNG use, no filesystem side effects beyond JSON write).
4. Do not touch `params.cfg` or call legacy loaders from the tool.
5. Keep defaults aligned with `scripts/studies/sim_lines_4x/pipeline.py::run_scenario`.
6. If the `dose_experiments` ref is missing, record the error and stop rather than guessing defaults.

## If Blocked
- If `git ls-tree -r dose_experiments` fails, record the exact error in `dose_experiments_param_scan.md`, note the missing ref in `docs/fix_plan.md` Attempts History, and stop.
- If the snapshot tool import triggers a missing dependency, capture the minimal error signature in `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-15T235900Z/blocker.md` and halt.

## Findings Applied
| Finding ID | Adherence |
|------------|-----------|
| CONFIG-001 | Snapshot tool avoids touching `params.cfg`; it only reads config defaults. |
| MODULE-SINGLETON-001 | Tool does not import `ptycho.model` or other modules that trigger singleton creation. |
| NORMALIZATION-001 | Snapshot captures probe_scale/normalization inputs without applying scaling. |
| BUG-TF-001 | Snapshot records explicit gridsize to avoid silent mismatch. |

## Pointers
- `plans/active/DEBUG-SIM-LINES-DOSE-001/implementation.md` — Phase A checklist and evidence requirements.
- `scripts/studies/sim_lines_4x/pipeline.py` — authoritative sim_lines_4x defaults and scenario logic.
- `docs/DATA_GENERATION_GUIDE.md` — nongrid pipeline expectations.
- `docs/debugging/debugging.md` — standard debugging workflow.
