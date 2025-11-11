Summary: Re-run the dense Phase C→G pipeline from this repo (not the stale PtychoPINN2 clone) with `--clobber` plus the follow-on `--post-verify-only` sweep so the 2025-11-12 hub finally captures SSIM grid, verification, highlights, metrics, preview verdict, and MS-SSIM/MAE deltas with logged evidence.
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
  - Implement: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py::main — rerun the counted dense pipeline with `--clobber` plus the immediate `--post-verify-only` sweep from `/home/ollie/Documents/PtychoPINN` so `{analysis,cli}` capture Phase C→G logs, SSIM grid summary/log, verification report/log, highlights logs, metrics digest/delta files, preview text, and artifact inventory evidence for this hub.
  - Pytest: AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md HUB=$PWD/plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier pytest --collect-only tests/study/test_phase_g_dense_orchestrator.py -k post_verify_only_executes_chain -vv | tee "$HUB"/collect/pytest_collect_post_verify_only.log
  - Pytest: AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md HUB=$PWD/plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier pytest tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_post_verify_only_executes_chain -vv | tee "$HUB"/green/pytest_post_verify_only.log
  - CLI: AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md HUB=$PWD/plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py --hub "$HUB" --dose 1000 --view dense --splits train test --clobber |& tee "$HUB"/cli/run_phase_g_dense_stdout.log
  - CLI: AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md HUB=$PWD/plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py --hub "$HUB" --post-verify-only |& tee "$HUB"/cli/run_phase_g_dense_post_verify_only.log
  - Artifacts: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/

How-To Map
1. `test "$(pwd -P)" = "/home/ollie/Documents/PtychoPINN"` to ensure the orchestrator writes into this repo, then `export AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md; export HUB=$PWD/plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier; mkdir -p "$HUB"/{analysis,cli,collect,green,red,summary}` so evidence folders exist before any command runs.
2. Run the mapped collect-only selector (`pytest --collect-only tests/study/test_phase_g_dense_orchestrator.py -k post_verify_only_executes_chain -vv | tee "$HUB"/collect/pytest_collect_post_verify_only.log`) and move any failures into `$HUB`/red/ before retrying; do not run the CLI until collection stays >0 per TEST-CLI-001.
3. Execute `pytest tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_post_verify_only_executes_chain -vv | tee "$HUB"/green/pytest_post_verify_only.log` to confirm the success-banner guard remains GREEN before launching expensive CLI work.
4. Launch the counted dense pipeline: `python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py --hub "$HUB" --dose 1000 --view dense --splits train test --clobber |& tee "$HUB"/cli/run_phase_g_dense_stdout.log`. Monitor every phase log (`phase_c_generation.log`, `phase_d_dense.log`, `phase_e_*`, `phase_f_dense_train.log`, `phase_g_dense_compare.log`, `aggregate_report_cli.log`, `metrics_digest_cli.log`, `ssim_grid_cli.log`, `verify_dense_stdout.log`, `check_dense_highlights.log`) for SUCCESS sentinels and confirm `$HUB/analysis` gains `metrics_delta_summary.json`, `metrics_delta_highlights_preview.txt`, `ssim_grid_summary.md`, `verification_report.json`, `verify_dense_stdout.log`, `check_dense_highlights.log`, `metrics_digest.md`, and `artifact_inventory.txt`.
5. Guard PREVIEW-PHASE-001 immediately: `rg -n "amplitude" "$HUB"/analysis/metrics_delta_highlights_preview.txt` must print nothing; if amplitude slips back in, move the offending preview into `$HUB`/red/ and rerun after fixing highlights.
6. Rerun verification without Phase C: `python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py --hub "$HUB" --post-verify-only |& tee "$HUB"/cli/run_phase_g_dense_post_verify_only.log`, ensuring the banner repeats SSIM grid + verification lines and `analysis/artifact_inventory.txt` refreshes.
7. If either CLI exits non-zero, capture the failing log under `$HUB`/red/ and stop; otherwise, record runtimes, MS-SSIM ±0.000 / MAE ±0.000000 deltas (phase emphasis), preview verdict, SSIM grid table reference, verification/highlights log links, pytest selectors, and doc updates inside `$HUB`/summary/summary.md (mirror to summary.md), then refresh docs/fix_plan.md and galph_memory with the same evidence bundle.

Pitfalls To Avoid
- Running any command from `/home/ollie/Documents/PtychoPINN2` (evidence lands in the wrong repo and Phase D–G never start).
- Skipping the initial `mkdir -p` + env export step (breaks POLICY-001 and causes missing `{analysis,cli}` again).
- Forgetting to tee pytest and CLI outputs into `$HUB` (violates TEST-CLI-001 evidence requirements).
- Running `--post-verify-only` before the fresh `--clobber` execution (would reuse stale artifacts and leave inventory untouched).
- Allowing `metrics_delta_highlights_preview.txt` to include “amplitude” or missing ± formatting (violates PREVIEW-PHASE-001).
- Ignoring SSIM grid/verifier exit codes; capture blockers in `$HUB`/red/ and stop if anything fails.

If Blocked
- On pytest failures or zero collection, move the log to `$HUB`/red/, note the selector + failure snippet in docs/fix_plan.md Attempts History, and halt before invoking the CLI until supervisor guidance arrives.
- On CLI failure, archive the stdout/phase log under `$HUB`/red/, capture the phase + command in docs/fix_plan.md and galph_memory, and stop rerunning until the plan is updated.

Findings Applied (Mandatory)
- POLICY-001 — Export AUTHORITATIVE_CMDS_DOC before running verifiers so the PyTorch helpers stay compliant.
- CONFIG-001 — Keep `update_legacy_dict` ordering intact inside `run_phase_g_dense.py` so Phase C consumers see the updated params.
- DATA-001 — Treat missing SSIM grid / verification JSON/log / artifact inventory outputs as blockers; reruns only after evidence is archived.
- TYPE-PATH-001 — Preserve hub-relative success banners/log references; pytest guards will fail otherwise.
- STUDY-001 — Report MS-SSIM + MAE deltas with ± notation inside the hub summaries.
- TEST-CLI-001 — Archive collect-only + execution logs plus both CLI stdout files to maintain RED/GREEN evidence.
- PREVIEW-PHASE-001 — Use the grep guard to keep highlights preview phase-only (no amplitude text).
- PHASEC-METADATA-001 — Respect the refreshed Phase C metadata layout; capture blocker logs if the guard trips again.

Pointers
- plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/implementation.md:204 — Phase G checklist tracks the remaining dense rerun + verification-only tasks and evidence targets.
- plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/plan/plan.md:1 — Reality check + execution sketch for the rerun, including command ordering and acceptance criteria.
- docs/fix_plan.md:3 — Ledger entry marking this focus ready_for_implementation with the new 2025-11-11T115413Z attempt summary and artifact expectations.
- docs/findings.md:8 — POLICY-001 / CONFIG-001 / DATA-001 / TYPE-PATH-001 / STUDY-001 / TEST-CLI-001 / PREVIEW-PHASE-001 / PHASEC-METADATA-001 guardrails governing this rerun.

Next Up (optional)
1. python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/report_phase_g_dense_metrics.py --metrics "$HUB"/analysis/metrics_summary.json --output "$HUB"/analysis/aggregate_report.md
2. python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/analyze_dense_metrics.py --hub "$HUB" --metrics "$HUB"/analysis/metrics_summary.json

Mapped Tests Guardrail
- `pytest --collect-only tests/study/test_phase_g_dense_orchestrator.py -k post_verify_only_executes_chain -vv` must collect the `test_run_phase_g_dense_post_verify_only_executes_chain` node before any CLI run; if collection ever drops to zero, stop immediately, log the failure under `$HUB`/red/, and update docs/fix_plan.md before proceeding.
