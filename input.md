Summary: Capture gs1_ideal vs gs2_ideal training telemetry by instrumenting the Phase C2 runner so we know exactly when gs1 starts emitting NaNs.
Focus: DEBUG-SIM-LINES-DOSE-001 — Isolate sim_lines_4x vs dose_experiments discrepancy
Branch: paper
Mapped tests: pytest tests/scripts/test_synthetic_helpers_cli_smoke.py::test_sim_lines_pipeline_import_smoke -v
Artifacts: plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T071800Z/

Do Now (hard validity contract)
- Implement: plans/active/DEBUG-SIM-LINES-DOSE-001/bin/run_phase_c2_scenario.py::main — extend the runner with helper utilities that persist the raw Lightning history (JSON) plus a derived Markdown/JSON summary that flags the first NaN per metric, warn in stdout when NaNs appear, reference the new artifacts inside `run_metadata.json`, and rerun the gs1_ideal + gs2_ideal profiles so the new telemetry (history.json, history_summary.{json,md}, updated notes) lands under the fresh hub. Preserve the baked profile behavior and keep inference/reassembly outputs identical aside from the added diagnostics.
- Pytest: pytest tests/scripts/test_synthetic_helpers_cli_smoke.py::test_sim_lines_pipeline_import_smoke -v
- Artifacts: plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T071800Z/{gs1_ideal_runner.log,gs2_ideal_runner.log,gs1_ideal/train_outputs/history.json,gs1_ideal/train_outputs/history_summary.json,gs1_ideal_training_summary.md,gs2_ideal/train_outputs/history.json,gs2_ideal/train_outputs/history_summary.json,gs2_ideal_training_summary.md,reassembly_cli.log,reassembly_gs1_ideal.json,reassembly_gs2_ideal.json,pytest_cli_smoke.log,pytest_collect_cli_smoke.log}

How-To Map
1. `export AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md; export ARTIFACT_DIR=plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T071800Z; mkdir -p "$ARTIFACT_DIR" "$ARTIFACT_DIR"/{gs1_ideal,gs2_ideal}` so all logs land in the new hub.
2. Edit `bin/run_phase_c2_scenario.py`:
   - Capture the dict returned by `run_training(...)` and add helpers such as `coerce_history_for_json(history)` and `summarize_history(history, epochs)` that convert numpy/tuple payloads into plain lists, compute per-metric stats (last value, min/max, first index containing NaN/inf), and emit warnings when any metric reports `nan` or `inf`.
   - Write `history.json` and `history_summary.json` into `<scenario>/train_outputs/` and create a short Markdown narrative (`<scenario>_training_summary.md`) that calls out the first failing epoch for each metric; stash the Markdown next to `run_metadata`.
   - Insert references to the new files inside `run_metadata.json` (e.g., `training_history_path`, `training_summary_path`) and include the NaN summary map so downstream consumers can read it without reopening the JSON file.
3. Rerun `gs1_ideal` with the baked profile (no manual overrides) while teeing stdout/stderr: `python plans/active/DEBUG-SIM-LINES-DOSE-001/bin/run_phase_c2_scenario.py --scenario gs1_ideal --snapshot plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-16T000353Z/sim_lines_4x_params_snapshot.json --output-dir "$ARTIFACT_DIR/gs1_ideal" --group-limit 64 |& tee "$ARTIFACT_DIR/gs1_ideal_runner.log"`.
4. Repeat for `gs2_ideal` to capture the healthy baseline: `python ... --scenario gs2_ideal --output-dir "$ARTIFACT_DIR/gs2_ideal" --group-limit 64 |& tee "$ARTIFACT_DIR/gs2_ideal_runner.log"`.
5. Summarize each run in `gs*_ideal_training_summary.md`, explicitly noting the first NaN epoch (gs1 is expected to fail, gs2 should remain finite) and referencing the saved history files by relative path.
6. Refresh the reassembly telemetry so padded-size evidence matches the new runs: `python plans/active/DEBUG-SIM-LINES-DOSE-001/bin/reassembly_limits_report.py --scenario gs1_ideal --group-limit 64 --output-json "$ARTIFACT_DIR/reassembly_gs1_ideal.json" --output-markdown "$ARTIFACT_DIR/reassembly_gs1_ideal.md" |& tee "$ARTIFACT_DIR/reassembly_cli.log"` and append the gs2 call (`--scenario gs2_ideal`) with `|& tee -a "$ARTIFACT_DIR/reassembly_cli.log"`.
7. Guard the CLI via testing: `pytest --collect-only tests/scripts/test_synthetic_helpers_cli_smoke.py -q | tee "$ARTIFACT_DIR/pytest_collect_cli_smoke.log"` followed by `pytest tests/scripts/test_synthetic_helpers_cli_smoke.py::test_sim_lines_pipeline_import_smoke -v | tee "$ARTIFACT_DIR/pytest_cli_smoke.log"`.
8. Verify the hub contains both scenarios’ training histories, summaries, PNGs, stats, runner logs, reassembly outputs, and pytest evidence before finishing.

Pitfalls To Avoid
- Don’t touch `ptycho/model.py`, `ptycho/diffsim.py`, or other protected modules—keep instrumentation confined to the plan-local runner.
- Preserve existing CLI behavior: manual overrides must still win over the baked profile and their effect should be logged rather than silently ignored.
- Ensure serialized histories only contain JSON-safe primitives (no numpy scalars or Path objects) or downstream tooling will break on load.
- Keep seeds, profile counts, and buffer identical to prior runs; otherwise comparisons with 2026-01-20T063500Z artifacts become meaningless.
- Avoid short-circuiting inference/reassembly even when NaNs appear—the goal is to capture both telemetry and downstream failures for gs1.
- Run commands from the repo root using `python` per PYTHON-ENV-001; no custom interpreter wrappers or relative `cd` gymnastics.
- Don’t overwrite the previous artifacts hub; all new evidence must live under `2026-01-20T071800Z`.
- When writing Markdown summaries, cite the exact epoch/metric names so future loops can script diffing if needed.

If Blocked
- If the runner still emits NaNs before writing `history.json`, capture whatever partial metrics are available, dump stdout/stderr to `$ARTIFACT_DIR/blocked_<timestamp>.log`, and log the failure plus exact command in docs/fix_plan.md Attempts History before marking the focus blocked.

Findings Applied (Mandatory)
- CONFIG-001 — runner still calls workflow helpers so `update_legacy_dict` syncs params before grouping/training (`docs/findings.md`).
- MODULE-SINGLETON-001 — we continue to rely on the factory flows (`train_cdi_model_with_backend`) so module-level singletons stay coherent.
- NORMALIZATION-001 — instrumentation must not mix probe/object normalization domains; stats are observational only.
- BUG-TF-REASSEMBLE-001 — keep padded-size calculations integer/consistent when logging stats so we don’t reintroduce the mixed-type crash.

Pointers
- plans/active/DEBUG-SIM-LINES-DOSE-001/implementation.md:148 — Phase C checklist (C2/C3) spells out the baked-profile scope and the new telemetry requirement.
- docs/specs/spec-ptycho-workflow.md:46 — Reassembly requirements mandate `M ≥ N + 2·max(|dx|,|dy|)`, which the reassembly telemetry continues to verify.
- docs/TESTING_GUIDE.md:1 — Testing policy and evidence logging conventions for the CLI smoke selector we are reusing.

Next Up (optional)
- Once telemetry shows when gs1 diverges, scope the actual gs1 stability fix (Phase C4) or escalate to a dedicated training-dynamics initiative.

Doc Sync Plan — N/A (existing selectors; no registry/test index updates expected).

Mapped Tests Guardrail — Run `pytest --collect-only tests/scripts/test_synthetic_helpers_cli_smoke.py -q` (Step 7) to prove the selector still collects before executing the test.

Normative Math/Physics — Reference `docs/specs/spec-ptycho-workflow.md §Reassembly Requirements` for any reasoning about padded-size vs offsets.
