Summary: Hub evidence still only contains `analysis/blocker.log` plus `cli/{phase_c_generation,phase_d_dense,run_phase_g_dense_stdout}.log`, so we must rerun the dense Phase C→G pipeline from `/home/ollie/Documents/PtychoPINN` with `--clobber` and follow with `--post-verify-only` to populate SSIM grid, verification, preview, metrics, and artifact inventory outputs.
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
  - Implement: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py::main — run `--dose 1000 --view dense --splits train test --clobber` from `/home/ollie/Documents/PtychoPINN` so `{analysis,cli}` capture SSIM grid, verification, highlights, preview, metrics, and artifact-inventory files, then immediately execute `--post-verify-only` on the fresh artifacts.
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
- Running any command outside `/home/ollie/Documents/PtychoPINN`; the prior `/home/ollie/Documents/PtychoPINN2` attempt produced the blocker logged in `analysis/blocker.log`.
- Skipping the `mkdir -p "$HUB"/...` prep, which causes tee to fail and breaks TEST-CLI-001 evidence logging.
- Restoring or editing `data/phase_c/run_manifest.json` manually; regenerate it through the counted run so git diffs remain tied to real artifacts.
- Accepting previews that mention “amplitude” or lack ± formatting; PREVIEW-PHASE-001 requires phase-only deltas.
- Forgetting to archive CLI/test logs under `$HUB/cli`, `$HUB/collect`, and `$HUB/green`, which would violate TEST-CLI-001.
- Running `--post-verify-only` before the `--clobber` execution finishes; it will replay stale artifacts and invalidate the rerun evidence.
- Ignoring SSIM grid/verification/metrics files after the run; missing outputs must be logged to `$HUB/red/` and reported immediately.
- Leaving pytest failures untriaged; move RED logs before reruns so the hub history stays auditable.

If Blocked
- Pytest failure: leave the log under `$HUB/red/`, capture the selector + failure snippet in docs/fix_plan.md and galph_memory, and halt for supervisor guidance.
- CLI non-zero exit: tee already wrote to `$HUB/cli`; move the failing per-phase log (and blocker log if present) into `$HUB/red/`, note the command + error text, and stop so we can triage before rerunning.
- Missing SSIM grid/verification/preview outputs after `--clobber`: archive the offending logs + `analysis/blocker.log`, update docs/fix_plan.md with the failure signature, and wait for new guidance before rerunning.

Findings Applied (Mandatory)
- POLICY-001 — docs/findings.md:8 — Keep PyTorch available for verifier/highlights helpers and export AUTHORITATIVE_CMDS_DOC ahead of CLI/test commands.
- CONFIG-001 — docs/findings.md:10 — Let `run_phase_g_dense.py` call `update_legacy_dict` before legacy consumers during Phase C generation.
- DATA-001 — docs/findings.md:14 — Treat SSIM grid and verification JSON/logs as contract artifacts; absence blocks completion.
- TYPE-PATH-001 — docs/findings.md:21 — Success banners, pytest assertions, and artifact inventory lines must stay hub-relative.
- STUDY-001 — docs/findings.md:16 — Report MS-SSIM + MAE deltas with explicit ± precision in summaries.
- TEST-CLI-001 — docs/findings.md:23 — Archive RED/GREEN pytest logs and CLI stdout for every selector/command.
- PREVIEW-PHASE-001 — docs/findings.md:24 — Reject previews containing amplitude text or malformed ± tokens.
- PHASEC-METADATA-001 — docs/findings.md:22 — Allow the Phase C metadata guard to run; capture its blocker log if the NPZ layout drifts.

Pointers
- plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/implementation.md:204 — Phase G checklist tracking the counted run + verification tasks.
- plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/plan/plan.md:1 — Execution sketch and acceptance criteria for this hub.
- docs/fix_plan.md:4 — Ledger status and guardrails for the active focus.
- docs/TESTING_GUIDE.md:210 — Selector references and CLI guidance for the Phase G orchestrator tests.

Next Up (optional)
1. python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/report_phase_g_dense_metrics.py --metrics "$HUB"/analysis/metrics_summary.json --output "$HUB"/analysis/aggregate_report.md
2. python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/analyze_dense_metrics.py --hub "$HUB" --metrics "$HUB"/analysis/metrics_summary.json

Mapped Tests Guardrail
- pytest --collect-only tests/study/test_phase_g_dense_orchestrator.py -k post_verify_only_executes_chain -vv must collect (>0) before running any CLI commands; if it fails, log the selector under `$HUB/red/` and stop.
