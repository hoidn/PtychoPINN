Summary: After stashing/restoring the dirty hub evidence to satisfy `git pull --rebase`, we still need a counted dense Phase G run plus the immediate `--post-verify-only` sweep from `/home/ollie/Documents/PtychoPINN` so `{analysis,cli}` finally contain SSIM grid, verifier, preview, metrics, and artifact-inventory outputs.
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
  - Implement: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py::main — rerun `--dose 1000 --view dense --splits train test --clobber` from `/home/ollie/Documents/PtychoPINN` so `{analysis,cli}` capture SSIM grid, verifier, highlights, preview, metrics, and artifact-inventory files, then immediately run `--post-verify-only` against the fresh artifacts.
  - Pytest: AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md HUB=$PWD/plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier pytest --collect-only tests/study/test_phase_g_dense_orchestrator.py -k post_verify_only_executes_chain -vv | tee "$HUB"/collect/pytest_collect_post_verify_only.log
  - Pytest: AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md HUB=$PWD/plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier pytest tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_post_verify_only_executes_chain -vv | tee "$HUB"/green/pytest_post_verify_only.log
  - CLI: AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md HUB=$PWD/plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py --hub "$HUB" --dose 1000 --view dense --splits train test --clobber |& tee "$HUB"/cli/run_phase_g_dense_stdout.log
  - CLI: AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md HUB=$PWD/plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py --hub "$HUB" --post-verify-only |& tee "$HUB"/cli/run_phase_g_dense_post_verify_only.log
  - Artifacts: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/

How-To Map
1. `test "$(pwd -P)" = "/home/ollie/Documents/PtychoPINN"` then `export AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md` and `export HUB=$PWD/plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier`; `mkdir -p "$HUB"/{analysis,cli,collect,green,red,summary}` so tee never drops evidence (TEST-CLI-001).
2. `pytest --collect-only tests/study/test_phase_g_dense_orchestrator.py -k post_verify_only_executes_chain -vv | tee "$HUB"/collect/pytest_collect_post_verify_only.log`; if collection fails, move the log to `$HUB/red/` and stop for supervisor triage.
3. `pytest tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_post_verify_only_executes_chain -vv | tee "$HUB"/green/pytest_post_verify_only.log` to keep the banner/path assertions GREEN before touching the expensive pipeline.
4. `python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py --hub "$HUB" --dose 1000 --view dense --splits train test --clobber |& tee "$HUB"/cli/run_phase_g_dense_stdout.log`; monitor `phase_c_generation.log`, `phase_d_dense.log`, `phase_e_*`, `phase_f_dense_train.log`, `phase_g_dense_compare.log`, `ssim_grid_cli.log`, `verify_dense_stdout.log`, and `check_dense_highlights.log` for SUCCESS sentinels.
5. Confirm artifacts: `ls "$HUB"/analysis` must list metrics_delta_summary.json, metrics_delta_highlights_preview.txt, metrics_digest.md, aggregate_report.md, ssim_grid_summary.md, verification_report.json, verify_dense_stdout.log, check_dense_highlights.log, and artifact_inventory.txt; run `rg -n "amplitude" "$HUB"/analysis/metrics_delta_highlights_preview.txt` to enforce PREVIEW-PHASE-001.
6. `python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py --hub "$HUB" --post-verify-only |& tee "$HUB"/cli/run_phase_g_dense_post_verify_only.log` to prove the shortened chain regenerates SSIM grid + verification outputs and rewrites `analysis/artifact_inventory.txt` with hub-relative success banner lines.
7. If any verifier/highlights check fails, move the offending logs under `$HUB/red/`, run `python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/check_dense_highlights_match.py --hub "$HUB" | tee "$HUB"/analysis/check_dense_highlights_manual.log`, capture the failure signature, and stop for supervisor guidance.
8. When the run succeeds, summarize MS-SSIM ±0.000 / MAE ±0.000000 deltas, preview verdict, SSIM grid/verifier/highlights references, pytest selectors, and doc/test status in `$HUB/summary/summary.md`, mirror it to `$HUB/summary.md`, and update docs/fix_plan.md plus galph_memory before exiting.

Pitfalls To Avoid
- Running any command from `/home/ollie/Documents/PtychoPINN2`; the previous attempt there produced the blocker recorded in `analysis/blocker.log`.
- Skipping the `mkdir -p "$HUB"/...` prep, which causes tee to fail and breaks TEST-CLI-001 evidence logging.
- Forgetting to export `AUTHORITATIVE_CMDS_DOC` + `HUB`, leading to logs being written outside the hub.
- Accepting previews that mention “amplitude” or lack ± formatting; PREVIEW-PHASE-001 requires phase-only deltas.
- Proceeding when SSIM grid/verification/metrics files are missing; document under `$HUB/red/` instead.
- Running `--post-verify-only` before the counted `--clobber` execution finishes; it will only replay stale artifacts.
- Deleting or restoring `data/phase_c/run_manifest.json` manually; regenerate it through the counted run so git remains dirty only because of real evidence.
- Allowing pytest selectors to drift; collect/execution guards must stay GREEN before CLI work begins.

If Blocked
- Pytest failure: leave the log under `$HUB/red/`, capture the selector + failure snippet in docs/fix_plan.md and galph_memory, and halt.
- CLI non-zero exit: tee already wrote to `$HUB/cli`; move the failing per-phase log (and blocker log if present) into `$HUB/red/`, note the command + error, and stop for supervisor triage.
- Missing SSIM grid/verification/preview outputs after `--clobber`: archive the offending logs + `analysis/blocker.log`, update docs/fix_plan.md with the failure signature, and wait for new guidance before rerunning.

Findings Applied (Mandatory)
- POLICY-001 (docs/findings.md:8) — Keep PyTorch installed and export AUTHORITATIVE_CMDS_DOC before invoking the verifier/highlights helpers.
- CONFIG-001 (docs/findings.md:10) — Let `run_phase_g_dense.py` call `update_legacy_dict` before legacy consumers during Phase C generation.
- DATA-001 (docs/findings.md:14) — Treat SSIM grid + verification JSON/logs as contract artifacts; absence blocks completion.
- TYPE-PATH-001 (docs/findings.md:21) — Success banners, pytest assertions, and artifact inventory lines must remain hub-relative.
- STUDY-001 (docs/findings.md:16) — Report MS-SSIM + MAE deltas with explicit ± precision in the study summaries.
- TEST-CLI-001 (docs/findings.md:23) — Archive RED/GREEN pytest logs and CLI stdout for every selector/command.
- PREVIEW-PHASE-001 (docs/findings.md:24) — Reject previews containing amplitude text or malformed ± tokens.
- PHASEC-METADATA-001 (docs/findings.md:22) — Allow the Phase C metadata guard to run; capture its blocker log if the NPZ layout drifts.

Pointers
- plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/implementation.md:204 — Phase G checklist listing the remaining counted run + verification tasks.
- plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/plan/plan.md:1 — Reality check, objectives, and execution sketch for this hub.
- plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/summary.md:1 — Latest Turn Summary capturing the stashed git pull + evidence gaps.
- docs/fix_plan.md:3 — Ledger state and guardrails for the active focus (ready_for_implementation mandate).
- docs/TESTING_GUIDE.md:210 — Phase G orchestrator guidance and pytest selector references for the mapped tests.

Next Up (optional)
1. python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/report_phase_g_dense_metrics.py --metrics "$HUB"/analysis/metrics_summary.json --output "$HUB"/analysis/aggregate_report.md
2. python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/analyze_dense_metrics.py --hub "$HUB" --metrics "$HUB"/analysis/metrics_summary.json

Mapped Tests Guardrail
- pytest --collect-only tests/study/test_phase_g_dense_orchestrator.py -k post_verify_only_executes_chain -vv must collect (>0) before running any CLI commands; if it fails, log the selector under `$HUB/red/` and stop.
