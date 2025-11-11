Summary: Ready-for-implementation rerun of the dense Phase G pipeline + `--post-verify-only` sweep after `git pull --rebase` was still blocked by the existing hub logs and `{analysis,cli}` only contain blocker evidence from the `/home/ollie/Documents/PtychoPINN2` attempt.
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
  - Implement: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py::main — rerun the dense Phase C→G pipeline with `--clobber` from `/home/ollie/Documents/PtychoPINN` so `{analysis,cli}` finally capture SSIM grid, verification, highlights, metrics, preview, and artifact-inventory files, then immediately execute the `--post-verify-only` mode to prove the shortened chain regenerates the verification artifacts.
  - Pytest: AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md HUB=$PWD/plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier pytest --collect-only tests/study/test_phase_g_dense_orchestrator.py -k post_verify_only_executes_chain -vv | tee "$HUB"/collect/pytest_collect_post_verify_only.log
  - Pytest: AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md HUB=$PWD/plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier pytest tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_post_verify_only_executes_chain -vv | tee "$HUB"/green/pytest_post_verify_only.log
  - CLI: AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md HUB=$PWD/plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py --hub "$HUB" --dose 1000 --view dense --splits train test --clobber |& tee "$HUB"/cli/run_phase_g_dense_stdout.log
  - CLI: AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md HUB=$PWD/plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py --hub "$HUB" --post-verify-only |& tee "$HUB"/cli/run_phase_g_dense_post_verify_only.log
  - Artifacts: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/

How-To Map
1. `test "$(pwd -P)" = "/home/ollie/Documents/PtychoPINN"` then `export AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md` and `export HUB=$PWD/plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier`; run `mkdir -p "$HUB"/{analysis,cli,collect,green,red,summary}` before any tee commands so logging meets TEST-CLI-001.
2. `pytest --collect-only tests/study/test_phase_g_dense_orchestrator.py -k post_verify_only_executes_chain -vv | tee "$HUB"/collect/pytest_collect_post_verify_only.log`; if collection fails, move the log to `$HUB/red/` and stop for supervisor triage.
3. `pytest tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_post_verify_only_executes_chain -vv | tee "$HUB"/green/pytest_post_verify_only.log` to ensure the success banner + hub-relative references remain GREEN before touching the expensive pipeline.
4. `python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py --hub "$HUB" --dose 1000 --view dense --splits train test --clobber |& tee "$HUB"/cli/run_phase_g_dense_stdout.log`; watch per-phase logs plus `ssim_grid_cli.log`, `verify_dense_stdout.log`, and `check_dense_highlights.log` for SUCCESS sentinels.
5. Verify outputs: `ls "$HUB"/analysis` should list `metrics_delta_summary.json`, `metrics_delta_highlights_preview.txt`, `metrics_digest.md`, `aggregate_report.md`, `ssim_grid_summary.md`, `verification_report.json`, `verify_dense_stdout.log`, `check_dense_highlights.log`, and `artifact_inventory.txt`; run `rg -n "amplitude" "$HUB"/analysis/metrics_delta_highlights_preview.txt` (PREVIEW-PHASE-001) to ensure the preview stays phase-only.
6. `python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py --hub "$HUB" --post-verify-only |& tee "$HUB"/cli/run_phase_g_dense_post_verify_only.log` so the shortened chain regenerates SSIM grid + verification outputs and refreshes `analysis/artifact_inventory.txt` with hub-relative success banner lines.
7. If the verifier flags mismatches, move the failing logs under `$HUB/red/`, run `python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/check_dense_highlights_match.py --hub "$HUB" | tee "$HUB"/analysis/check_dense_highlights_manual.log`, and halt for supervisor input; otherwise update `$HUB/summary/summary.md` (mirror to `$HUB/summary.md`), docs/fix_plan.md, and galph_memory with MS-SSIM ±0.000 / MAE ±0.000000 deltas, preview verdict, SSIM grid/verifier/highlights references, pytest selectors, and doc/test notes.

Pitfalls To Avoid
- Launching commands from `/home/ollie/Documents/PtychoPINN2`; that recreates the `ValueError: Object arrays cannot be loaded when allow_pickle=False` seen in `cli/phase_d_dense.log`.
- Forgetting the `mkdir -p "$HUB"/{...}` prep, which causes tee to fail and breaks TEST-CLI-001 evidence requirements.
- Skipping the pytest selectors; they enforce the hub-relative success banners for both the full and post-verify-only paths.
- Accepting previews that mention “amplitude” or lack ± markers; run `rg` immediately (PREVIEW-PHASE-001).
- Ignoring missing SSIM grid/verification files or artifact_inventory; absence remains a blocker that must be logged under `$HUB/red/` with failure signatures.
- Running `--post-verify-only` before completing the fresh `--clobber` execution; it only replays whatever artifacts exist.
- Deleting or stashing the current hub logs just to run git pull; they are required evidence for this loop.

If Blocked
- Pytest failure: leave the failing log under `$HUB/red/`, capture the selector + exception snippet in docs/fix_plan.md and galph_memory, and wait for supervisor guidance.
- CLI non-zero exit: tee already writes to `$HUB/cli`; move the failing per-phase log under `$HUB/red/`, note the command + error signature in docs/fix_plan.md and galph_memory, and stop.
- Missing SSIM grid/verification artifacts after the rerun: archive `analysis/blocker.log` + CLI log under `$HUB/red/`, log the failure details, and halt for replan.

Findings Applied (Mandatory)
- POLICY-001 (docs/findings.md:8) — Export `AUTHORITATIVE_CMDS_DOC` and keep PyTorch available for verifier/highlights helpers.
- CONFIG-001 (docs/findings.md:10) — Let `run_phase_g_dense.py` continue calling `update_legacy_dict` before legacy consumers during Phase C generation.
- DATA-001 (docs/findings.md:14) — Treat the SSIM grid + verification outputs as contract-enforced artifacts; runs without them stay blocked.
- TYPE-PATH-001 (docs/findings.md:21) — Success banners, pytest guards, and artifact inventory lines must stay hub-relative.
- STUDY-001 (docs/findings.md:16) — Summaries must report MS-SSIM + MAE deltas with explicit ± precision for the study’s phase focus.
- TEST-CLI-001 (docs/findings.md:23) — Archive RED/GREEN pytest logs plus CLI stdout per selector/command.
- PREVIEW-PHASE-001 (docs/findings.md:24) — Reject previews that include amplitude content or malformed ± formatting.
- PHASEC-METADATA-001 (docs/findings.md:22) — Allow the Phase C metadata guard to run and capture its blocker log if the NPZ layout drifts.

Pointers
- plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/implementation.md:204 — Active Phase G checklist showing the remaining dense rerun + verification tasks.
- plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/plan/plan.md:1 — Reality check + execution sketch for the counted rerun and post-verify-only sweep.
- plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/summary.md:1 — Turn summaries recording blockers and expected evidence bundle.
- docs/fix_plan.md:4 — Ledger status, guardrails, and the ready_for_implementation decision for this initiative.

Next Up (optional)
1. python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/report_phase_g_dense_metrics.py --metrics "$HUB"/analysis/metrics_summary.json --output "$HUB"/analysis/aggregate_report.md
2. python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/analyze_dense_metrics.py --hub "$HUB" --metrics "$HUB"/analysis/metrics_summary.json

Mapped Tests Guardrail
- pytest --collect-only tests/study/test_phase_g_dense_orchestrator.py -k post_verify_only_executes_chain -vv must collect >0 nodes before any CLI run; if collection fails, log it as a blocker and stop.
