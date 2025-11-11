Summary: Execute the dense Phase C→G pipeline with --clobber and immediately rerun --post-verify-only so the 2025-11-12 hub finally captures SSIM grid, verification, highlights, metrics, preview verdict, and MS-SSIM/MAE deltas with documented evidence.
Mode: Perf
Focus: STUDY-SYNTH-FLY64-DOSE-OVERLAP-001 — Phase G comparison & analysis (dense real evidence + automated report)
Branch: feature/torchapi-newprompt
Plan Document: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/implementation.md
Reports Hub: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/
Mapped tests:
- AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md HUB=$PWD/plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier pytest --collect-only tests/study/test_phase_g_dense_orchestrator.py -k post_verify_only_executes_chain -vv
- AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md HUB=$PWD/plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier pytest tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_post_verify_only_executes_chain -vv
Artifacts: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/

Do Now (hard validity contract)
- STUDY-SYNTH-FLY64-DOSE-OVERLAP-001
  - Implement: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py::main — run the counted dense pipeline with `--clobber` and the follow-on `--post-verify-only` sweep so `{analysis,cli}` capture Phase C→G logs, SSIM grid summary/log, verification report/log, highlights logs, metrics digest, metrics delta JSON/preview, and artifact inventory evidence for this hub.
  - Pytest: AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md HUB=$PWD/plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier pytest --collect-only tests/study/test_phase_g_dense_orchestrator.py -k post_verify_only_executes_chain -vv | tee "$HUB"/collect/pytest_collect_post_verify_only.log
  - Pytest: AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md HUB=$PWD/plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier pytest tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_post_verify_only_executes_chain -vv | tee "$HUB"/green/pytest_post_verify_only.log
  - CLI: AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md HUB=$PWD/plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py --hub "$HUB" --dose 1000 --view dense --splits train test --clobber |& tee "$HUB"/cli/run_phase_g_dense_stdout.log
  - CLI: AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md HUB=$PWD/plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py --hub "$HUB" --post-verify-only |& tee "$HUB"/cli/run_phase_g_dense_post_verify_only.log
  - Artifacts: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/

How-To Map
1. `export AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md; export HUB=$PWD/plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier; mkdir -p "$HUB"/{analysis,cli,collect,green,red,summary}` so every command honors POLICY-001 and logs into the active hub.
2. Run the mapped collect-only selector (`pytest --collect-only ... -k post_verify_only_executes_chain -vv`) and tee to `$HUB`/collect/pytest_collect_post_verify_only.log; stash failures in red/ then green-run before expensive CLI work.
3. Execute `pytest tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_post_verify_only_executes_chain -vv` (tee to `$HUB`/green/pytest_post_verify_only.log) to prove the banner/log assertions still pass before launching runtime-heavy commands.
4. Launch the counted dense pipeline: `python plans/active/.../bin/run_phase_g_dense.py --hub "$HUB" --dose 1000 --view dense --splits train test --clobber |& tee "$HUB"/cli/run_phase_g_dense_stdout.log`; monitor every phase log for SUCCESS and confirm `$HUB/analysis` now holds `metrics_delta_summary.json`, `metrics_delta_highlights_preview.txt`, `metrics_delta_highlights.txt`, `metrics_digest.md`, `aggregate_report.md`, `ssim_grid_summary.md`, `verification_report.json`, `verify_dense_stdout.log`, `check_dense_highlights.log`, and `artifact_inventory.txt`.
5. Guard PREVIEW-PHASE-001: `rg -n "amplitude" "$HUB"/analysis/metrics_delta_highlights_preview.txt` must return zero matches; if not, archive the offending artifacts under red/ and rerun after fixing preview content.
6. Immediately rerun verification with `python plans/active/.../bin/run_phase_g_dense.py --hub "$HUB" --post-verify-only |& tee "$HUB"/cli/run_phase_g_dense_post_verify_only.log`; ensure the banner repeats the SSIM grid + verification/highlights lines and that `analysis/artifact_inventory.txt` gains a fresh timestamp.
7. If the orchestrator reports preview/highlights mismatches, run `python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/check_dense_highlights_match.py --hub "$HUB" --debug | tee "$HUB"/analysis/check_dense_highlights_manual.log` before attempting any reruns; keep the log under analysis/ for evidence.
8. Summarize results: capture runtimes, MS-SSIM ±0.000 / MAE ±0.000000 deltas (phase emphasis), preview verdict, SSIM grid table reference, verification/highlights log links, CLI/test selectors, and doc updates in `$HUB`/summary/summary.md (mirror to summary.md) plus docs/fix_plan.md and galph_memory.
9. Leave the hub tidy—if any step fails, stop immediately, move logs to red/, and record the failure signature in docs/fix_plan.md + summary.md instead of clobbering evidence.

Pitfalls To Avoid
- Skipping the AUTHORITATIVE_CMDS_DOC/HUB exports (breaks orchestrator guard + env policy).
- Forgetting to tee pytest/CLI output into `$HUB` (violates TEST-CLI-001 evidence contract).
- Running `--post-verify-only` before the counted `--clobber` command (would reuse stale artifacts).
- Allowing `metrics_delta_highlights_preview.txt` to contain amplitude tokens or missing ± values (breaches PREVIEW-PHASE-001).
- Ignoring Phase C metadata failures; per PHASEC-METADATA-001, stop and capture blocker logs if validation trips.
- Dropping hub-relative success-banner references—stdout must still mention `analysis/...` and `cli/...` per TYPE-PATH-001.
- Leaving verification/highlights logs outside `analysis/`; keep filenames identical to orchestrator defaults for future guards.
- Ending the loop without updating summary.md + docs/fix_plan.md with MS-SSIM/MAE deltas, preview verdict, and verifier references.

If Blocked
- If any mapped pytest selector fails or collects 0, move the log to `$HUB`/red/, capture the selector + failure snippet in docs/fix_plan.md Attempts History, then pause for guidance instead of proceeding to CLI commands.
- If run_phase_g_dense.py exits non-zero, archive the offending CLI/phase logs under `$HUB`/red/, log the phase + error signature in docs/fix_plan.md and galph_memory, and stop (no reruns until a fresh supervisor plan arrives).

Findings Applied (Mandatory)
- POLICY-001 — Export AUTHORITATIVE_CMDS_DOC so PyTorch-dependent verifiers run under the sanctioned environment.
- CONFIG-001 — Leave `update_legacy_dict(params.cfg, config)` untouched inside the orchestrator so legacy consumers observe correct params.
- DATA-001 — Treat missing SSIM grid, verification JSON/log, or artifact inventory outputs as blockers and archive failures.
- TYPE-PATH-001 — Keep banner/log references hub-relative (tests assert `analysis/...` and `cli/...`).
- STUDY-001 — Report MS-SSIM + MAE deltas with ± precision inside summary docs.
- TEST-CLI-001 — Archive collect + exec pytest logs plus CLI stdout per command in `$HUB`.
- PREVIEW-PHASE-001 — Verify the preview file is phase-only; rerun or file blocker if amplitude text appears.
- PHASEC-METADATA-001 — Respect the refreshed Phase C metadata layout when the orchestrator validates hub inputs.

Pointers
- plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/implementation.md:204 — Phase G checklist items that remain unchecked (dense rerun + verification-only sweep + ledger updates).
- plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/plan/plan.md:1 — Latest execution sketch outlining regression tests + both CLI commands required this loop.
- docs/TESTING_GUIDE.md:389 — Prescribed run_phase_g_dense.py commands and Phase G evidence expectations (ms-ssim-focused).
- docs/fix_plan.md:4 — Active ledger entry describing the missing evidence and artifact requirements.
- docs/findings.md:8 — POLICY-001 / CONFIG-001 / DATA-001 / TYPE-PATH-001 / STUDY-001 / TEST-CLI-001 / PREVIEW-PHASE-001 / PHASEC-METADATA-001 guardrails that govern this run.

Next Up (optional)
1. Run `python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/report_phase_g_dense_metrics.py --metrics "$HUB"/analysis/metrics_summary.json --output "$HUB"/analysis/aggregate_report.md --highlights "$HUB"/analysis/aggregate_highlights.txt` to regenerate the MS-SSIM sanity table once evidence exists.
2. Execute `python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/analyze_dense_metrics.py --hub "$HUB" --metrics "$HUB"/analysis/metrics_summary.json` to refresh `metrics_digest.md` and embed the phase-only preview verdict inside the digest.
