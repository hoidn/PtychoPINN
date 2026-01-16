# DEBUG-SIM-LINES-DOSE-001 — Phase A4 parameter comparison handoff

**Summary:** Generate an evidence-grade SIM-LINES-4X vs dose_experiments parameter diff so we know which knobs diverged before we set up the A/B experiments.

**Focus:** DEBUG-SIM-LINES-DOSE-001 — Isolate sim_lines_4x vs dose_experiments discrepancy

**Branch:** paper

**Mapped tests:**
- `pytest tests/scripts/test_synthetic_helpers_cli_smoke.py::test_sim_lines_pipeline_import_smoke -v`

**Artifacts:** `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-16T003217Z/`

---

## Do Now
- DEBUG-SIM-LINES-DOSE-001.A4 — Compare SIM-LINES-4X snapshot to dose_experiments defaults
  - Implement: `plans/active/DEBUG-SIM-LINES-DOSE-001/bin/compare_sim_lines_params.py::main` — new CLI script that reads the existing snapshot JSON (`sim_lines_4x_params_snapshot.json`) and the legacy reference file (`dose_experiments_param_scan.md`), extracts the canonical parameters (gridsize, probe flags, probe_scale, offset, outer_offset_train/test, nimgs_train/test, nphotons, group_count, neighbor_count, reassemble_M, intensity_scale.trainable), and emits a Markdown table + JSON diff covering gs1/gs2 ideal/custom scenarios with delta annotations.
  - Validate: `pytest tests/scripts/test_synthetic_helpers_cli_smoke.py::test_sim_lines_pipeline_import_smoke -v` (guard against accidental import regressions while touching the study helpers tree).
  - Artifacts: write `comparison_draft.md` (Markdown table + commentary) and `comparison_diff.json` (machine-readable diff payload) under `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-16T003217Z/`, and tee the pytest log to `pytest_sim_lines_pipeline_import.log` in the same directory.

## How-To Map
1. Create `plans/active/DEBUG-SIM-LINES-DOSE-001/bin/compare_sim_lines_params.py` (mark executable) with an argparse CLI: `--snapshot`, `--dose-config`, `--output-markdown`, `--output-json`.
2. Parsing logic:
   - Load the snapshot JSON and normalize each scenario into a dict capturing name, gridsize, probe_mode, probe_scale, probe_big, probe_mask, `group_count`, `neighbor_count`, `derived_counts`, and `reassemble_M`.
   - Read `dose_experiments_param_scan.md` as plain text; use regex to capture the canonical assignments (`cfg['offset'] = ...`, `cfg['outer_offset_train'] = ...`, etc.) plus `nphotons` from the `init()` function. Treat those as the legacy defaults.
   - Build a comparison structure keyed by parameter name with entries for `sim_lines` (per scenario) vs `dose_experiments`, including delta/notes highlighting mismatches (e.g., total images vs nimgs_{train,test}, probe scaling, offsets, gridsize).
3. Emit outputs:
   - Markdown: include a header summarizing methodology, then a table with columns `Parameter | dose_experiments | sim_lines_<scenario> | delta / note`. Add a short prose note for the most critical divergences.
   - JSON: dump the structured diff so we can feed it into future automation (`comparison_diff.json`).
4. Run the script:
   ```bash
   python plans/active/DEBUG-SIM-LINES-DOSE-001/bin/compare_sim_lines_params.py \
     --snapshot plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-16T000353Z/sim_lines_4x_params_snapshot.json \
     --dose-config plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-16T000353Z/dose_experiments_param_scan.md \
     --output-markdown plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-16T003217Z/comparison_draft.md \
     --output-json plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-16T003217Z/comparison_diff.json
   ```
5. Capture pytest evidence (environment variable already set by supervisor prompt):
   ```bash
   AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md \
   pytest tests/scripts/test_synthetic_helpers_cli_smoke.py::test_sim_lines_pipeline_import_smoke -v \
     | tee plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-16T003217Z/pytest_sim_lines_pipeline_import.log
   ```
6. Update `plans/active/DEBUG-SIM-LINES-DOSE-001/summary.md` with a one-liner referencing the new diff artifacts after the script runs.

## Pitfalls To Avoid
1. Keep the comparison script confined to `plans/active/.../bin` so production code remains untouched.
2. Do not execute the historic `dose_experiments` branch; operate on the captured markdown/text only.
3. Treat missing keys defensively—emit explicit "missing" markers instead of crashing when a field is absent.
4. Preserve float precision (probe_scale, offsets) when reporting deltas; no rounding beyond 2 decimals unless necessary.
5. Ensure regex parsing is case-sensitive to avoid accidental matches inside comments.
6. Avoid importing TensorFlow or running any heavy loaders—this loop is metadata-only.
7. Do not overwrite the existing snapshot JSON/markdown; the new outputs must live under the 2026-01-16T003217Z directory.
8. Always run the mapped pytest selector after editing the script (even though it’s outside tests/) to prove we didn’t break sim_lines imports.
9. When emitting Markdown, escape pipes (`|`) so the table renders correctly.
10. Record any parsing assumptions directly in the Markdown so future investigators know which defaults were inferred.

## If Blocked
- If the regex parser cannot find one of the required keys in `dose_experiments_param_scan.md`, log the failure (key + reason) inside `comparison_draft.md` and note it in `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-16T003217Z/blocker.md`, then stop and ping the supervisor.
- If `sim_lines_4x_params_snapshot.json` is missing or malformed, capture the traceback under the same blocker file and do not fabricate values.

## Findings Applied
| Finding ID | Adherence |
|------------|-----------|
| CONFIG-001 | Legacy vs new configs are compared without mutating `params.cfg`, so config sync guarantees remain intact. |
| MODULE-SINGLETON-001 | No model factories are touched; the diff script manipulates plain data, ensuring singleton-safe behavior. |
| NORMALIZATION-001 | Probe/intensity scaling values are reported verbatim without applying additional normalization. |
| BUG-TF-001 | Gridsize + grouping parameters are surfaced explicitly to prevent hidden channel mismatches. |

## Pointers
- `plans/active/DEBUG-SIM-LINES-DOSE-001/implementation.md#L80` — Phase A checklist describing A4 requirements.
- `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-16T000353Z/sim_lines_4x_params_snapshot.json` — authoritative SIM-LINES-4X metadata payload.
- `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-16T000353Z/dose_experiments_param_scan.md` — captured legacy defaults to parse.
- `docs/DATA_GENERATION_GUIDE.md#nongrid-simulation` — workflow context for interpreting grouping/offset parameters.
- `docs/specs/spec-ptycho-workflow.md#Reassembly Requirements` — normative reference for offsets/padding expectations.

## Next Up (optional)
1. Phase B1 grid vs nongrid A/B plan once the diff identifies the high-risk divergences.
