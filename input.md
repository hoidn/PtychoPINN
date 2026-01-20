Summary: Rerun the plan-local gs2_ideal scenario for 60 epochs so we can validate the H-NEPOCHS hypothesis and capture whether extended training closes the amplitude gap.
Focus: DEBUG-SIM-LINES-DOSE-001 — Phase D3b 60-epoch gs2_ideal retrain
Branch: paper
Mapped tests: pytest tests/scripts/test_synthetic_helpers_cli_smoke.py::test_sim_lines_pipeline_import_smoke -v
Artifacts: plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T140531Z/

Do Now (hard validity contract):
- Implement: plans/active/DEBUG-SIM-LINES-DOSE-001/bin/run_phase_c2_scenario.py::main — rerun the `gs2_ideal` scenario with `--nepochs 60 --group-limit 64 --prediction-scale-source least_squares`, archive the refreshed training histories/PNGs/analyzer outputs under the new hub, and document the amplitude/pearson_r deltas so D3b evidence can confirm or refute H-NEPOCHS before adjusting sim_lines defaults.
- Validate: pytest tests/scripts/test_synthetic_helpers_cli_smoke.py::test_sim_lines_pipeline_import_smoke -v

How-To Map:
1. Prepare hub + env: `export AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md HUB=plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T140531Z`
2. Run 60-epoch retrain: `AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md python plans/active/DEBUG-SIM-LINES-DOSE-001/bin/run_phase_c2_scenario.py --scenario gs2_ideal --output-dir "$HUB"/gs2_ideal_nepochs60 --nepochs 60 --group-limit 64 --prediction-scale-source least_squares |& tee "$HUB"/gs2_ideal_nepochs60/runner.log`
   • Confirm `run_metadata.json` reflects nepochs=60 and that `history.json` / `history_summary.json` land under `train_outputs/`
   • Capture amplitude/phase PNGs + comparison metrics just like the 5-epoch baseline
3. Re-run analyzer for the retrain: `AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md python plans/active/DEBUG-SIM-LINES-DOSE-001/bin/analyze_intensity_bias.py --scenario gs2_ideal="$HUB"/gs2_ideal_nepochs60 --output-dir "$HUB" |& tee "$HUB"/analyze_intensity_bias.log`
   • Drop Markdown summary of MAE/pearson_r delta versus the 5-epoch run (reference `reports/2026-01-20T133807Z/analysis.md`)
4. Guard selector: `AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md python -m pytest tests/scripts/test_synthetic_helpers_cli_smoke.py::test_sim_lines_pipeline_import_smoke -v | tee "$HUB"/pytest_cli_smoke.log`
5. Update `$HUB/summary.md` with bullet comparisons (training length, loss curves, amplitude metrics) and state whether H-NEPOCHS is confirmed/denied

Pitfalls To Avoid:
- Do not edit production modules (`ptycho/` or `scripts/studies/`) for this retrain; stay within plan-local tooling.
- Always call `update_legacy_dict(params.cfg, config)` via the runner helpers so CONFIG-001 remains satisfied before training/inference.
- Keep GPU utilization bounded (use existing stable profile + group-limit=64) to avoid OOM; never switch to CPU to “save memory.”
- Leave prior evidence intact; append new logs/PNGs instead of overwriting the 5-epoch baseline hub.
- Ensure `--prediction-scale-source least_squares` is used consistently so amplitude comparisons are fair.
- Verify the runner actually ran 60 epochs (history JSON should show 60 entries) before starting analyzer work.
- Avoid running analyzer on stale directories—point it explicitly at the new hub path.
- Archive stdout/stderr for both runner and analyzer commands under the new hub for auditability.
- Keep CLI smoke pytest evidence under the same hub to satisfy TEST-CLI-001.
- If training diverges (NaNs/Infs), capture the failure log and stop rather than rerunning blindly.

If Blocked:
- Record command + stack trace in `$HUB/blocker.md`, note the issue in docs/fix_plan.md Attempts History and galph_memory, and pause so we can triage before wasting additional GPU cycles.

Findings Applied (Mandatory):
- CONFIG-001 — runner/analyzer must keep the params.cfg bridge in place before touching legacy modules.
- SIM-LINES-CONFIG-001 — sim_lines flows fail without CONFIG-001 bridging; maintain it for every rerun.
- NORMALIZATION-001 — analyzer comparisons must reference the normalization invariants when explaining any residual bias.

Pointers:
- docs/fix_plan.md:256 — Phase D3 attempts + retrain scope.
- plans/active/DEBUG-SIM-LINES-DOSE-001/implementation.md:345 — D3 checklist (D3b/D3c instructions).
- plans/active/DEBUG-SIM-LINES-DOSE-001/bin/run_phase_c2_scenario.py:1 — runner CLI with stable profiles and `--nepochs` override.
- plans/active/DEBUG-SIM-LINES-DOSE-001/bin/analyze_intensity_bias.py:1 — analyzer CLI + normalization invariant checks.
- specs/spec-ptycho-core.md:86 — normalization invariants (reference when interpreting analyzer output).

Next Up (optional): D3c documentation pass — compare 60-epoch vs 5-epoch results and, if H-NEPOCHS holds, decide whether to bump sim_lines defaults or continue to D4 architecture/loss wiring.
