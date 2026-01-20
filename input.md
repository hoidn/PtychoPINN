Summary: Extend the Phase D4 analyzer so we can see exactly how the gs2 baseline vs 60-epoch runs allocate loss weight and where the normalized→prediction pipeline drops amplitude.
Focus: DEBUG-SIM-LINES-DOSE-001 — Phase D4 architecture/loss diagnostics
Branch: paper
Mapped tests: pytest tests/scripts/test_synthetic_helpers_cli_smoke.py::test_sim_lines_pipeline_import_smoke -v
Artifacts: plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T162500Z/

Do Now (hard validity contract):
- Implement: plans/active/DEBUG-SIM-LINES-DOSE-001/bin/analyze_intensity_bias.py::main — teach the analyzer to parse each scenario’s `train_outputs/history_summary.json`, compute a loss-composition section (pred_intensity_loss vs intensity_scaler_inv_loss vs trimmed_obj_loss and learning-rate context per `specs/spec-ptycho-workflow.md §Training Outputs`), and expand the stage-ratio summary so normalized→prediction and prediction→truth deltas cite `specs/spec-ptycho-core.md §Normalization Invariants`. After updating the script, run `AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md python plans/active/DEBUG-SIM-LINES-DOSE-001/bin/analyze_intensity_bias.py --scenario gs2_base=plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T160000Z/gs2_ideal --scenario gs2_ne60=plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T140531Z/gs2_ideal_nepochs60 --output-dir plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T162500Z | tee plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T162500Z/analyze_loss_wiring.log` so we get refreshed `bias_summary.{json,md}` plus Markdown blocks highlighting the loss makeup and the stage where amplitude collapses.
- Validate: pytest tests/scripts/test_synthetic_helpers_cli_smoke.py::test_sim_lines_pipeline_import_smoke -v

How-To Map:
1. `export AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md HUB=plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T162500Z BASE=plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T160000Z/gs2_ideal LONG=plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T140531Z/gs2_ideal_nepochs60`
2. Implement analyzer updates (loss-composition parsing + stage-ratio Markdown). Keep changes scoped to `plans/active/DEBUG-SIM-LINES-DOSE-001/bin/analyze_intensity_bias.py` and add any helper functions within the same file.
3. Re-run the analyzer: `python plans/active/DEBUG-SIM-LINES-DOSE-001/bin/analyze_intensity_bias.py --scenario gs2_base="$BASE" --scenario gs2_ne60="$LONG" --output-dir "$HUB" | tee "$HUB"/analyze_loss_wiring.log`. Confirm the generated `bias_summary.json` now contains `loss_composition` and the Markdown highlights which stage ratio deviates most.
4. Copy the refreshed Markdown/JSON plus the tee’d log into `$HUB` and mention the findings in `summary.md` when the loop closes.
5. Guard selector: `python -m pytest tests/scripts/test_synthetic_helpers_cli_smoke.py::test_sim_lines_pipeline_import_smoke -v | tee "$HUB"/pytest_cli_smoke.log`.

Pitfalls To Avoid:
- Do not touch production modules under `ptycho/` or `scripts/studies/`; all diagnostics stay inside plan-local tooling.
- Reuse the existing scenario outputs; do **not** retrain — the analyzer can read the saved `train_outputs/` bundles directly.
- Keep ratios/metrics float-safe (guard against divide-by-zero) and cite the exact spec sections in Markdown so reviewers can trace requirements.
- Preserve prior analyzer outputs; write new files under the fresh hub instead of overwriting older evidence.
- Ensure CLI args remain backwards compatible (no behavioral change when new section disabled) so downstream automation keeps working.
- Remember SIM-LINES-CONFIG-001: never remove the CONFIG-001 bridge calls from the runner even though we’re only updating the analyzer.
- Avoid CPU training/inference — we’re only reading artifacts so no new GPU work is required.
- Keep JSON deterministic (sort keys) so diffs stay reviewable.
- Include scenario names in the new Markdown tables so gs2_base vs gs2_ne60 comparisons are obvious.
- Update docs/fix_plan.md only after analyzer artifacts exist; don’t pre-emptively mark D4 progress without evidence.

If Blocked:
- If either scenario directory is missing files, record the missing path + stack trace in `$HUB/blocker.md`, update docs/fix_plan.md Attempts History with the blocker, and stop so we can restage the evidence rather than guessing.

Findings Applied (Mandatory):
- CONFIG-001 — Analyzer must continue to respect the params.cfg bridge; do not remove the runner sync points.
- SIM-LINES-CONFIG-001 — All sim_lines tooling assumes CONFIG-001 bridging before loader/model usage; analyzer updates cannot change that contract.
- NORMALIZATION-001 — Stage-ratio reporting must keep normalization invariants separated from loss scaling to avoid mixing the three normalization domains.
- H-NEPOCHS-001 — Training length was ruled out as the root cause, so D4 instrumentation must pivot to loss/architecture diagnostics rather than proposing another retrain.

Pointers:
- plans/active/DEBUG-SIM-LINES-DOSE-001/implementation.md:359 — D4 checklist + architecture/loss verification scope.
- plans/active/DEBUG-SIM-LINES-DOSE-001/summary.md:1-20 — Current status + D3b outcome (H-NEPOCHS rejected).
- docs/fix_plan.md:3-20,256-268 — Active focus description + Phase D3 attempts history.
- specs/spec-ptycho-core.md#Normalization-Invariants — Stage ratio + invariant requirements.
- specs/spec-ptycho-workflow.md#training-outputs — Loss component logging obligations for training telemetry.

Next Up (optional):
1. If the loss-composition data shows normalized→prediction collapse, plan a plan-local probe that taps into `nbutils.reconstruct_image` to capture pre/post IntensityScaler tensors.
2. If analyzer proves only the intensity-scaler loss is active, prepare a follow-up loop to compare the sim_lines loss wiring vs the legacy `dose_experiments` script at the code level.
