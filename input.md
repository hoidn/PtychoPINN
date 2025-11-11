Summary: Reaffirm the dense Phase G rerun + `--post-verify-only` sweep after verifying the hub still lacks SSIM grid/verification/preview artifacts and ensure everything runs from `/home/ollie/Documents/PtychoPINN`.
Mode: Perf
Focus: STUDY-SYNTH-FLY64-DOSE-OVERLAP-001 — Phase G comparison & analysis (dense real evidence + automated report)
Branch: feature/torchapi-newprompt
Plan Document: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/implementation.md
Reports Hub: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/
Mapped tests:
- pytest --collect-only tests/study/test_phase_g_dense_orchestrator.py -k post_verify_only_executes_chain -vv
- pytest tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_post_verify_only_executes_chain -vv
Artifacts: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/

Do Now (hard validity contract)
- STUDY-SYNTH-FLY64-DOSE-OVERLAP-001
  - Implement: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py::main — rerun the dense Phase C→G pipeline with `--clobber` from `/home/ollie/Documents/PtychoPINN` so `{analysis,cli}` finally capture SSIM grid, verification, highlights, metrics, preview, and inventory artifacts, then immediately execute the `--post-verify-only` path to regenerate the shortened chain outputs.
  - Pytest: AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md HUB=$PWD/plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier pytest --collect-only tests/study/test_phase_g_dense_orchestrator.py -k post_verify_only_executes_chain -vv | tee "$HUB"/collect/pytest_collect_post_verify_only.log
  - Pytest: AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md HUB=$PWD/plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier pytest tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_post_verify_only_executes_chain -vv | tee "$HUB"/green/pytest_post_verify_only.log
  - CLI: AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md HUB=$PWD/plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py --hub "$HUB" --dose 1000 --view dense --splits train test --clobber |& tee "$HUB"/cli/run_phase_g_dense_stdout.log
  - CLI: AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md HUB=$PWD/plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py --hub "$HUB" --post-verify-only |& tee "$HUB"/cli/run_phase_g_dense_post_verify_only.log
  - Artifacts: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/

How-To Map
1. `test "$(pwd -P)" = "/home/ollie/Documents/PtychoPINN"` then export `AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md` and `HUB=$PWD/plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier`; `mkdir -p "$HUB"/{analysis,cli,collect,green,red,summary}` so tee never fails.
2. `pytest --collect-only tests/study/test_phase_g_dense_orchestrator.py -k post_verify_only_executes_chain -vv | tee "$HUB"/collect/pytest_collect_post_verify_only.log`; move failures to `$HUB`/red/ before rerunning.
3. `pytest tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_post_verify_only_executes_chain -vv | tee "$HUB"/green/pytest_post_verify_only.log` to confirm the success-banner guard stays GREEN before launching expensive CLI runs.
4. `python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py --hub "$HUB" --dose 1000 --view dense --splits train test --clobber |& tee "$HUB"/cli/run_phase_g_dense_stdout.log`; watch phase_c/phase_d, all Phase E/F logs, `phase_g_dense_compare.log`, `aggregate_report_cli.log`, `metrics_digest_cli.log`, `ssim_grid_cli.log`, `verify_dense_stdout.log`, and `check_dense_highlights.log` for SUCCESS sentinels.
5. Verify outputs: `ls "$HUB"/analysis` must show `metrics_delta_summary.json`, `metrics_delta_highlights_preview.txt`, `metrics_digest.md`, `aggregate_report.md`, `ssim_grid_summary.md`, `verification_report.json`, `verify_dense_stdout.log`, `check_dense_highlights.log`, `artifact_inventory.txt`. Run `rg -n "amplitude" "$HUB"/analysis/metrics_delta_highlights_preview.txt` (PREVIEW-PHASE-001) — no matches allowed.
6. `python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py --hub "$HUB" --post-verify-only |& tee "$HUB"/cli/run_phase_g_dense_post_verify_only.log`, confirming the shortened chain regenerates SSIM grid + verification outputs and refreshes `analysis/artifact_inventory.txt` with hub-relative success banner.
7. If verifier/highlights diverge, capture the RED artifacts under `$HUB`/red/, run `python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/check_dense_highlights_match.py --hub "$HUB" | tee "$HUB"/analysis/check_dense_highlights_manual.log`, and stop for supervisor guidance.
8. Update `$HUB/summary/summary.md` (mirror to `$HUB/summary.md`) with runtimes, MS-SSIM ±0.000 / MAE ±0.000000 deltas, preview verdict, SSIM grid table path, verification/highlights logs, artifact inventory + metrics digest references, pytest selectors, and doc/test updates; then refresh docs/fix_plan.md + galph_memory with the same evidence bundle.

Pitfalls To Avoid
- Launching any command from `/home/ollie/Documents/PtychoPINN2`; always check `pwd -P` before running tests or CLI.
- Forgetting the `mkdir -p "$HUB"/{...}` prep, which causes tee to fail and violates TEST-CLI-001 logging rules.
- Skipping the pytest selectors or failing to tee their output to `$HUB`/collect and `$HUB`/green.
- Accepting previews that contain “amplitude” or missing ± tokens (PREVIEW-PHASE-001) — grep for them immediately.
- Ignoring missing SSIM grid/verification artifacts; treat absence as a blocker and capture the failing logs under `$HUB`/red/.
- Running `--post-verify-only` before the fresh `--clobber` execution, which would only recycle stale artifacts.
- Failing to copy MS-SSIM/MAE deltas + preview/verifier references into docs/fix_plan.md and galph_memory before ending the loop.

If Blocked
- Pytest failures: archive the log under `$HUB`/red/, capture selector + failure snippet in docs/fix_plan.md and galph_memory, and pause for supervisor review.
- CLI non-zero exit: tee log already in place; move the failing per-phase log into `$HUB`/red/, note the command + error signature in docs/fix_plan.md and galph_memory, and stop.
- Missing SSIM grid or verification files: treat as blocker, archive `analysis/blocker.log`, and wait for new guidance before re-running.

Findings Applied (Mandatory)
- POLICY-001 (docs/findings.md:8) — Export `AUTHORITATIVE_CMDS_DOC` before touching PyTorch-dependent workflows so Torch availability stays within policy.
- CONFIG-001 (docs/findings.md:10) — Leave `update_legacy_dict` ordering intact inside `run_phase_g_dense.py` when rerunning the orchestrator.
- DATA-001 (docs/findings.md:14) — Require the SSIM grid + verification artifacts in `$HUB/analysis` to satisfy the NPZ/JSON data contract before closing the loop.
- TYPE-PATH-001 (docs/findings.md:21) — Success banners/logs must stay hub-relative; the pytest selectors enforce this.
- STUDY-001 (docs/findings.md:16) — Report MS-SSIM + MAE deltas with explicit ± precision in the hub summaries.
- TEST-CLI-001 (docs/findings.md:23) — Archive RED/GREEN pytest logs and both CLI stdout files for traceability.
- PREVIEW-PHASE-001 (docs/findings.md:24) — Ensure `metrics_delta_highlights_preview.txt` only includes the four phase deltas (no amplitude text).
- PHASEC-METADATA-001 (docs/findings.md:22) — Let the Phase C metadata guard run; capture its blocker log if it fires.

Pointers
- plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/implementation.md:204 — Phase G checklist tracks the remaining dense rerun/verification tasks and evidence targets.
- plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/plan/plan.md:1 — Execution sketch + acceptance criteria for the rerun and `--post-verify-only` sweep.
- plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/summary.md:1 — Turn summaries describing the outstanding evidence and required artifacts.
- docs/fix_plan.md:4 — Ledger status and Attempts History entries for this initiative.

Next Up (optional)
1. python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/report_phase_g_dense_metrics.py --metrics "$HUB"/analysis/metrics_summary.json --output "$HUB"/analysis/aggregate_report.md
2. python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/analyze_dense_metrics.py --hub "$HUB" --metrics "$HUB"/analysis/metrics_summary.json

Mapped Tests Guardrail
- pytest --collect-only tests/study/test_phase_g_dense_orchestrator.py -k post_verify_only_executes_chain -vv must collect >0 nodes before any CLI run; archive failures under `$HUB`/red/ and stop if collection breaks.
