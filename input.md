Summary: Reconfirm the dense Phase G hub still lacks analysis artifacts and hand Ralph a ready_for_implementation Do Now to rerun the counted Phase C→G pipeline plus the immediate `--post-verify-only` sweep from this repo with full SSIM grid, verifier, highlights, and metrics evidence.
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
  - Implement: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py::main — rerun the dense Phase C→G pipeline with `--clobber` from `/home/ollie/Documents/PtychoPINN` so `{analysis,cli}` finally capture SSIM grid, verification, highlights, metrics, preview, and inventory artifacts for the counted evidence hub, then immediately execute the `--post-verify-only` path to regenerate the shortened chain outputs.
  - Pytest: AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md HUB=$PWD/plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier pytest --collect-only tests/study/test_phase_g_dense_orchestrator.py -k post_verify_only_executes_chain -vv | tee "$HUB"/collect/pytest_collect_post_verify_only.log
  - Pytest: AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md HUB=$PWD/plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier pytest tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_post_verify_only_executes_chain -vv | tee "$HUB"/green/pytest_post_verify_only.log
  - CLI: AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md HUB=$PWD/plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py --hub "$HUB" --dose 1000 --view dense --splits train test --clobber |& tee "$HUB"/cli/run_phase_g_dense_stdout.log
  - CLI: AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md HUB=$PWD/plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py --hub "$HUB" --post-verify-only |& tee "$HUB"/cli/run_phase_g_dense_post_verify_only.log
  - Artifacts: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/

How-To Map
1. `test "$(pwd -P)" = "/home/ollie/Documents/PtychoPINN"`; export `AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md` and `HUB=$PWD/plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier`; `mkdir -p "$HUB"/{analysis,cli,collect,green,red,summary}` so rerun evidence cannot spill into the wrong clone again.
2. `pytest --collect-only tests/study/test_phase_g_dense_orchestrator.py -k post_verify_only_executes_chain -vv | tee "$HUB"/collect/pytest_collect_post_verify_only.log`; move failures to `$HUB`/red/ before retrying so the selector proves it still collects >0 tests.
3. `pytest tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_post_verify_only_executes_chain -vv | tee "$HUB"/green/pytest_post_verify_only.log` to keep the success-banner guard GREEN before launching the expensive CLI work.
4. `python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py --hub "$HUB" --dose 1000 --view dense --splits train test --clobber |& tee "$HUB"/cli/run_phase_g_dense_stdout.log`; monitor `phase_c_generation.log`, `phase_d_dense.log`, all Phase E/F logs, `phase_g_dense_compare.log`, `aggregate_report_cli.log`, `metrics_digest_cli.log`, `ssim_grid_cli.log`, `verify_dense_stdout.log`, and `check_dense_highlights.log` for SUCCESS sentinels.
5. Verify outputs: `ls "$HUB"/analysis` must show `metrics_delta_summary.json`, `metrics_delta_highlights_preview.txt`, `metrics_digest.md`, `aggregate_report.md`, `ssim_grid_summary.md`, `ssim_grid_cli.log`, `verification_report.json`, `verify_dense_stdout.log`, `check_dense_highlights.log`, and `artifact_inventory.txt`; run `rg -n "amplitude" "$HUB"/analysis/metrics_delta_highlights_preview.txt` to enforce PREVIEW-PHASE-001 (no matches allowed).
6. `python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py --hub "$HUB" --post-verify-only |& tee "$HUB"/cli/run_phase_g_dense_post_verify_only.log`, confirming the shortened chain regenerates SSIM grid + verification outputs and refreshes `analysis/artifact_inventory.txt`.
7. If verifier/highlights disagree, capture blocker logs under `$HUB`/red/, run `python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/check_dense_highlights_match.py --hub "$HUB" | tee "$HUB"/analysis/check_dense_highlights_manual.log`, and pause for supervisor guidance before re-running.
8. Summarize results in `$HUB/summary/summary.md` (mirror to `$HUB/summary.md`): note runtimes, MS-SSIM ±0.000 / MAE ±0.000000 deltas, preview verdict, SSIM grid table path, verification/highlights logs, artifact inventory status, pytest selectors, and doc/test updates; then update docs/fix_plan.md + galph_memory with the same evidence bundle.

Pitfalls To Avoid
- Running any command from `/home/ollie/Documents/PtychoPINN2`; confirm `pwd -P` first or the orchestrator will recreate the blocker.
- Skipping the `mkdir -p "$HUB"/{...}` prep, which causes CLI steps to fail when tee cannot open destination files.
- Forgetting to tee pytest/CLI stdout into `$HUB`, violating TEST-CLI-001 and forcing another rerun.
- Allowing `metrics_delta_highlights_preview.txt` to reference “amplitude” or drop ± tokens (PREVIEW-PHASE-001).
- Ignoring missing SSIM grid/verification outputs; treat absent JSON/log files as blockers and stop immediately.
- Rerunning `--post-verify-only` before the fresh `--clobber` execution; it would only recycle stale artifacts.
- Neglecting to copy the MS-SSIM/MAE deltas + preview/verifier evidence into docs/fix_plan.md and galph_memory before closing the loop.

If Blocked
- On pytest collection/execution failures, move the log to `$HUB`/red/, capture the selector + failure snippet in docs/fix_plan.md Attempts and galph_memory, then pause for supervisor updates.
- On CLI exit ≠ 0, archive the offending `cli/*` and per-phase log under `$HUB`/red/, note the command + failure signature in docs/fix_plan.md and galph_memory, and wait for revised instructions before restarting.

Findings Applied (Mandatory)
- POLICY-001 — Export AUTHORITATIVE_CMDS_DOC before tests/CLI so PyTorch-dependent helpers stay compliant.
- CONFIG-001 — `run_phase_g_dense.py::main` already calls `update_legacy_dict`; do not reorder legacy bridge steps when editing the script.
- DATA-001 — Treat missing SSIM grid/verification/metrics artifacts as fatal until they are archived into `$HUB`/analysis.
- TYPE-PATH-001 — Keep success banners hub-relative; the mapped pytest selectors verify those strings.
- STUDY-001 — Report MS-SSIM + MAE deltas with explicit ±0.000 / ±0.000000 precision inside the hub summaries.
- TEST-CLI-001 — Archive collect-only + execution pytest logs plus both CLI stdout logs under `$HUB` for RED/GREEN traceability.
- PREVIEW-PHASE-001 — Enforce phase-only preview text (grep for “amplitude” must stay silent) before accepting the run.
- PHASEC-METADATA-001 — Do not bypass the Phase C metadata guard; if it fires, capture blocker logs and stop.

Pointers
- plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/implementation.md:204 — Phase G checklist tracks the remaining dense rerun/post-verify tasks and evidence requirements.
- plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/plan/plan.md:1 — Reality check + execution sketch for the rerun workflow, including env setup and acceptance criteria.
- docs/fix_plan.md:3 — Ledger entry (s=264) capturing the 2025-11-11T120554Z attempt summary and current status.
- plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/summary.md:1 — Latest Turn Summary reiterating why the rerun evidence is still outstanding and where to log results.

Next Up (optional)
1. python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/report_phase_g_dense_metrics.py --metrics "$HUB"/analysis/metrics_summary.json --output "$HUB"/analysis/aggregate_report.md
2. python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/analyze_dense_metrics.py --hub "$HUB" --metrics "$HUB"/analysis/metrics_summary.json

Mapped Tests Guardrail
- `pytest --collect-only tests/study/test_phase_g_dense_orchestrator.py -k post_verify_only_executes_chain -vv` must collect the guarded node before any CLI run; if it ever reports 0 tests, archive the log under `$HUB`/red/ and stop for supervisor guidance before touching the pipeline.
