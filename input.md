Summary: Capture Phase D3 hyperparameter evidence by extending the plan-local diff CLI so we can prove how sim_lines training knobs diverge from dose_experiments before planning retrains.
Focus: DEBUG-SIM-LINES-DOSE-001 — Hyperparameter delta audit (Phase D3)
Branch: paper
Mapped tests: pytest tests/scripts/test_synthetic_helpers_cli_smoke.py::test_sim_lines_pipeline_import_smoke -v
Artifacts: plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T133807Z/

Do Now (hard validity contract):
- Implement: plans/active/DEBUG-SIM-LINES-DOSE-001/bin/compare_sim_lines_params.py::main — extend the plan-local comparison CLI so the Markdown/JSON diff also surfaces training knobs (nepochs, batch_size, probe/intensity-scale trainability) with a `--default-sim-lines-nepochs` override, update the enrichment helper to build a TrainingConfig snapshot per scenario, and plumb the new fields through the Markdown/JSON builders.
- Validate: pytest tests/scripts/test_synthetic_helpers_cli_smoke.py::test_sim_lines_pipeline_import_smoke -v (AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md python -m pytest …)

How-To Map:
1. Edit `plans/active/DEBUG-SIM-LINES-DOSE-001/bin/compare_sim_lines_params.py`:
   • Add CLI flag `--default-sim-lines-nepochs` (int, default 5) so scenarios get the real training length.
   • Replace `get_loss_weights_from_training_config` with a helper that instantiates `TrainingConfig` using the scenario parameters + `nepochs` and returns `{mae_weight,nll_weight,realspace_weight,realspace_mae_weight,batch_size,nepochs,probe.trainable,intensity_scale.trainable}`.
   • Extend `PARAMETERS` to include `nepochs`, `batch_size`, and `probe.trainable`; ensure dose configs fall back to legacy params defaults when the captured script does not set those keys.
   • Update Markdown/JSON builders so the new fields appear in every per-scenario table and in the JSON diff.
2. Run the refreshed CLI (capture stdout with `tee`):
   ```bash
   AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md python plans/active/DEBUG-SIM-LINES-DOSE-001/bin/compare_sim_lines_params.py \
     --snapshot plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-16T000353Z/sim_lines_4x_params_snapshot.json \
     --dose-config plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-16T000353Z/dose_experiments_param_scan.md \
     --default-sim-lines-nepochs 5 \
     --output-markdown plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T133807Z/hyperparam_diff.md \
     --output-json plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T133807Z/hyperparam_diff.json \
     --output-dose-loss-weights plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T133807Z/dose_loss_weights.json \
     --output-dose-loss-weights-markdown plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T133807Z/dose_loss_weights.md \
     --output-legacy-defaults plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T133807Z/legacy_params_cfg_defaults.json \
     | tee plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T133807Z/compare_cli.log
   ```
3. Run the pytest guard with logs under the hub:
   ```bash
   AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md python -m pytest tests/scripts/test_synthetic_helpers_cli_smoke.py::test_sim_lines_pipeline_import_smoke -v \
     | tee plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T133807Z/pytest_cli_smoke.log
   ```
4. Drop a short `analysis.md` snippet in the hub summarizing the new hyperparameter deltas (nepochs jump from 5→60, batch_size parity, probe/intensity scale settings) so Phase D3 evidence is ready for the follow-up retrain planning.

Pitfalls To Avoid:
- Keep all code changes inside plan-local scripts; do not edit production modules (`ptycho/` or `scripts/studies/`).
- Preserve the existing loss-weight capture logic—do not regress the D1 evidence.
- When instantiating `TrainingConfig`, always call `update_legacy_dict` via existing helpers so CONFIG-001 stays satisfied before any legacy touch points.
- Treat unset dose_experiments fields carefully; pull defaults from `ptycho.params.cfg` rather than inventing values.
- Do not change CLI output paths; all artifacts must land under `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T133807Z/`.
- Re-run the pytest guard even if the CLI looks trivial; Phase D requires explicit proof of selector health per docs/TESTING_GUIDE.md.
- Keep Markdown tables tidy (pipe separators aligned) so the diff remains human-readable in future loops.
- Use `AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md` for both the CLI command (as shown) and pytest to honor PYTHON-ENV-001.
- Avoid deleting prior artifacts; new evidence must augment, not replace, earlier D1/D2 bundles.
- If additional helper data structures are introduced, scope them locally to avoid pollution of the repo namespace.

If Blocked:
- Capture the exact error (command + stderr) into `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T133807Z/blocker.md`, note the blocker in docs/fix_plan.md Attempts History, and ping Galph via galph_memory before stopping.

Findings Applied (Mandatory):
- CONFIG-001 — every code path that reads legacy modules must keep `update_legacy_dict(params.cfg, config)` intact; do not bypass it while instantiating TrainingConfig snapshots.
- SIM-LINES-CONFIG-001 — sim_lines runners and plan-local scripts must maintain the params.cfg bridge before training/inference helpers to prevent NaNs.

Pointers:
- docs/fix_plan.md:219 — Phase D attempts history + D3 scope reference.
- plans/active/DEBUG-SIM-LINES-DOSE-001/implementation.md:345 — checklist entry defining the D3 hyperparameter audit deliverables.
- plans/active/DEBUG-SIM-LINES-DOSE-001/bin/compare_sim_lines_params.py:1 — existing CLI that already captures parameter diffs and loss weights.
- docs/TESTING_GUIDE.md:51 — script helper test selectors (CLI smoke guard requirements).
- specs/spec-ptycho-workflow.md:67 — workflow spec stating training runs must emit histories + metrics, grounding the need for faithful hyperparameter capture.

Next Up (optional): If the hyperparameter diff lands quickly, queue a gs2_ideal retrain with `nepochs=60` using the plan-local runner to test whether training length alone closes the amplitude gap.
