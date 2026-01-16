# DEBUG-SIM-LINES-DOSE-001 — Phase A: capture sim_lines_4x param snapshot

**Summary:** Build a deterministic sim_lines_4x parameter snapshot tool, record the historic dose_experiments defaults, and prove the sim_lines CLI import still succeeds.

**Focus:** DEBUG-SIM-LINES-DOSE-001 — Isolate sim_lines_4x vs dose_experiments discrepancy

**Branch:** paper

**Mapped tests:**
- `pytest tests/scripts/test_synthetic_helpers_cli_smoke.py::test_sim_lines_pipeline_import_smoke -v`

**Artifacts:** `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-16T000353Z/`

---

## Do Now
- DEBUG-SIM-LINES-DOSE-001.A0/A1/A3 (evidence capture kick-off)
  - Implement: `scripts/tools/collect_sim_lines_4x_params.py::main` — new argparse CLI with required `--output` that imports `ScenarioSpec`, `RunParams`, `derive_counts`, and `CUSTOM_PROBE_PATH` from `scripts.studies.sim_lines_4x.pipeline`, instantiates the four canonical scenarios (gs1/gs2 × ideal/custom), computes derived counts via `derive_counts`, resolves the `probe_big`/`probe_mask` defaults using the same logic as `run_scenario`, and emits JSON containing `collected_at`, `run_params` (dataclass to dict), `scenarios` (each with inputs, defaults, derived totals, `group_count`, `neighbor_count`, `reassemble_M`, and paths), plus the shared `custom_probe_path` string.
  - Validate: `pytest tests/scripts/test_synthetic_helpers_cli_smoke.py::test_sim_lines_pipeline_import_smoke -v` and tee output to the artifacts hub.
  - Artifacts: drop the JSON snapshot as `sim_lines_4x_params_snapshot.json`, save `dose_experiments_tree.txt` (from `git ls-tree`), and summarize defaults in `dose_experiments_param_scan.md` under the artifacts directory.

## How-To Map
1. Author the new tool under `scripts/tools/collect_sim_lines_4x_params.py`; keep imports limited to standard lib + `scripts.studies.sim_lines_4x.pipeline` + `ptycho.config.config.ModelConfig`. Persist JSON with `json.dump(payload, output, indent=2, sort_keys=True)`.
2. Run snapshot:
   ```bash
   python scripts/tools/collect_sim_lines_4x_params.py \
     --output plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-16T000353Z/sim_lines_4x_params_snapshot.json
   ```
3. Inventory historic configs:
   ```bash
   git ls-tree -r dose_experiments \
     > plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-16T000353Z/dose_experiments_tree.txt
   git show dose_experiments:notebooks/dose.py \
     > plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-16T000353Z/dose_experiments_param_scan.md
   ```
   Then append a short table to the markdown file summarizing the defaults (nphotons, gridsize, offsets, nimgs_train/test, loss selection) extracted from that script.
4. Capture the smoke-test evidence:
   ```bash
   AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md \
   pytest tests/scripts/test_synthetic_helpers_cli_smoke.py::test_sim_lines_pipeline_import_smoke -v \
     | tee plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-16T000353Z/pytest_sim_lines_pipeline_import.log
   ```

## Pitfalls To Avoid
1. Do **not** import TensorFlow-heavy modules (`train`, `loader`, etc.) inside the new tool; it must stay metadata-only.
2. Keep the snapshot deterministic: no RNG use, no filesystem writes besides the JSON destination.
3. Do not mutate `params.cfg` or run data-generation helpers—the tool should read constants only.
4. When pulling from `dose_experiments`, never modify the tree; use `git show` for read-only inspection and note the exact commit hash if errors occur.
5. Ensure the JSON structure mirrors the spec exactly (top-level run params, scenarios array, shared probe path) so Phase B diff scripts can consume it verbatim.
6. Store every artifact under `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-16T000353Z/`; no stray files in repo root.
7. Use `Path(...).parent.mkdir(parents=True, exist_ok=True)` before writing outputs so CI and local runs behave the same.
8. Keep imports relative to repo root but avoid modifying `sys.path` in the new tool (tests already do that for you).
9. Make the CLI fast—fail if `--output` is missing instead of entering an interactive mode.
10. Record any missing dataset or git ref errors into `dose_experiments_param_scan.md` instead of guessing defaults.

## If Blocked
- If `git ls-tree -r dose_experiments` or `git show dose_experiments:notebooks/dose.py` fails (e.g., missing submodule), capture the exact stderr in `dose_experiments_param_scan.md`, drop a short note into the artifacts summary, and stop—ping Galph so we can decide whether to vendor the historic defaults.
- If importing `scripts.studies.sim_lines_4x.pipeline` raises due to unmet dependencies, write the traceback to `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-16T000353Z/blocker.md` and notify the supervisor before attempting fallback implementations.

## Findings Applied
| Finding ID | Adherence |
|------------|-----------|
| CONFIG-001 | Tool only reads config dataclasses; no legacy state mutation occurs before data access. |
| MODULE-SINGLETON-001 | We avoid importing `ptycho.model` or other singleton creators when collecting metadata. |
| NORMALIZATION-001 | Snapshot captures probe scale/intensity inputs verbatim without reinterpreting scaling math. |
| BUG-TF-001 | Explicit gridsize entries are included in the metadata, preventing silent mismatch during later comparisons. |

## Pointers
- `plans/active/DEBUG-SIM-LINES-DOSE-001/implementation.md#L54` — checklist for Phase A evidence capture items (A0–A4).
- `scripts/studies/sim_lines_4x/pipeline.py#L20-L210` — authoritative ScenarioSpec/RunParams definitions and logic for defaults.
- `docs/DATA_GENERATION_GUIDE.md#nongrid-simulation` — nongrid workflow expectations driving this study.
- `docs/debugging/debugging.md` — required four-step workflow for new bug investigations.

## Next Up (if time remains)
1. Phase A4: compare the new snapshot against dose_experiments defaults and document divergences.
2. Phase B1 prep: design the grid vs nongrid A/B experiment matrix using the captured parameters.
