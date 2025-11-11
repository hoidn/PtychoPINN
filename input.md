Summary: Execute the dense Phase C→G pipeline (--clobber) plus the immediate --post-verify-only sweep so the 2025-11-12 hub finally contains SSIM grid, verification, highlights, metrics, and MS-SSIM/MAE evidence with updated summaries.
Mode: Perf
Focus: STUDY-SYNTH-FLY64-DOSE-OVERLAP-001 — Phase G comparison & analysis (dense real evidence + automated report)
Branch: feature/torchapi-newprompt
Plan Document: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/implementation.md
Reports Hub (active): plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/
Mapped tests:
- AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md HUB=$PWD/plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier pytest --collect-only tests/study/test_phase_g_dense_orchestrator.py -k post_verify_only_executes_chain -vv
- AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md HUB=$PWD/plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier pytest tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_post_verify_only_executes_chain -vv
Artifacts: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/

Do Now (hard validity contract)
- STUDY-SYNTH-FLY64-DOSE-OVERLAP-001
  - Implement: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/summary/summary.md::dense_run_evidence_entry — document the counted dense run + post-verify-only sweep (MS-SSIM ±0.000 / MAE ±0.000000 deltas, preview verdict, SSIM grid + verification/highlights links, artifact inventory + metrics digest references, mapped pytest selectors) and mirror the block into summary.md per hub hygiene.
  - Pytest: AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md HUB=$PWD/plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier pytest --collect-only tests/study/test_phase_g_dense_orchestrator.py -k post_verify_only_executes_chain -vv | tee "$HUB"/collect/pytest_collect_post_verify_only.log
  - Pytest: AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md HUB=$PWD/plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier pytest tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_post_verify_only_executes_chain -vv | tee "$HUB"/green/pytest_post_verify_only.log
  - CLI: AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md HUB=$PWD/plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py --hub "$HUB" --dose 1000 --view dense --splits train test --clobber |& tee "$HUB"/cli/run_phase_g_dense_stdout.log
  - CLI: AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md HUB=$PWD/plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py --hub "$HUB" --post-verify-only |& tee "$HUB"/cli/run_phase_g_dense_post_verify_only.log
  - Artifacts: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/

How-To Map
1. export AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md; export HUB=$PWD/plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier; mkdir -p "$HUB"/{analysis,cli,collect,green,red,summary} so every command honors POLICY-001 and writes evidence into the active hub.
2. Run the mapped collect-only selector (`pytest --collect-only ... -k post_verify_only_executes_chain -vv`) and tee to "$HUB"/collect/pytest_collect_post_verify_only.log; if it flakes, stash the log under red/ then rerun before proceeding.
3. Execute the targeted test (`pytest ...::test_run_phase_g_dense_post_verify_only_executes_chain -vv`) and tee to "$HUB"/green/pytest_post_verify_only.log; ensure stdout still prints the SSIM Grid summary/log plus verification/highlights banners before launching long CLI commands.
4. Launch the counted dense pipeline: python plans/active/.../bin/run_phase_g_dense.py --hub "$HUB" --dose 1000 --view dense --splits train test --clobber |& tee "$HUB"/cli/run_phase_g_dense_stdout.log; watch every phase log for SUCCESS and confirm `$HUB/analysis` now contains metrics_delta_summary.json, metrics_delta_highlights_preview.txt, ssim_grid_summary.md, verification_report.json, verify_dense_stdout.log, check_dense_highlights.log, metrics_digest.md, and artifact_inventory.txt.
5. Validate PREVIEW-PHASE-001 by running `rg -n "amplitude" "$HUB"/analysis/metrics_delta_highlights_preview.txt`; if any hits appear, treat as failure, capture logs to red/, and re-run after fixing preview/validator output.
6. Run the verification-only sweep: python plans/active/.../bin/run_phase_g_dense.py --hub "$HUB" --post-verify-only |& tee "$HUB"/cli/run_phase_g_dense_post_verify_only.log; confirm the banner repeats all SSIM grid/verification/highlights lines and that analysis/artifact_inventory.txt gets a fresh timestamp.
7. If the orchestrator spots preview/highlights mismatches, immediately run `python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/check_dense_highlights_match.py --hub "$HUB" --debug | tee "$HUB"/analysis/check_dense_highlights_manual.log` and keep the log under analysis/ for reference before any reruns.
8. Summarize the run: record runtimes, MS-SSIM ±0.000 / MAE ±0.000000 deltas (phase-only focus), preview verdict, SSIM grid table path, verification/highlights log references, artifact inventory + metrics digest links, and the exact pytest/CLI commands inside "$HUB"/summary/summary.md (then copy to "$HUB"/summary.md) and update docs/fix_plan.md plus galph_memory accordingly.
9. Leave the hub cleanly populated: if any command fails, archive the failing log under red/, capture the failure signature in docs/fix_plan.md and summary.md, and stop rather than clobbering evidence.

Pitfalls To Avoid
- Skipping `AUTHORITATIVE_CMDS_DOC`/`HUB` exports (breaks orchestrator guard + env policy).
- Forgetting to `tee` pytest/CLI output into `$HUB` (violates TEST-CLI-001 evidence contract).
- Running `--post-verify-only` before the counted `--clobber` run (analysis/verification artifacts would still be stale).
- Leaving `analysis/metrics_delta_highlights_preview.txt` with amplitude text or missing ± tokens (breaks PREVIEW-PHASE-001).
- Ignoring Phase C metadata errors—stop immediately if run_phase_g_dense raises PHASEC-METADATA-001 blockers and capture logs under red/.
- Allowing run logs to stay outside the hub or renaming them (stick to orchestrator filenames so scripts/tests can find them).
- Editing production modules while chasing runtime issues—this loop is evidence-only; revert any incidental code edits before committing.
- Forgetting to refresh summary.md/docs/fix_plan.md with MS-SSIM/MAE deltas and verifier references before ending the loop.

If Blocked
- If any pytest selector fails or collects 0, move the log to `$HUB/red/`, note the failure in docs/fix_plan.md (Attempts History) with the selector/output snippet, and stop so we can fix the guard before reattempting.
- If run_phase_g_dense.py exits non-zero, archive the offending CLI log + phase-specific logs into `$HUB/red/`, document the failure signature + phase in docs/fix_plan.md and galph_memory, and await guidance before rerunning (do **not** delete artifacts).

Findings Applied (Mandatory)
- POLICY-001 — PyTorch must remain available for Phase F + verifier steps.
- CONFIG-001 — Keep `update_legacy_dict` ordering intact inside run_phase_g_dense.
- DATA-001 — SSIM grid/verifier artifacts must match the data contract; missing JSON/NPZ files block completion.
- TYPE-PATH-001 — Success banners/log references must stay hub-relative when we inspect stdout/logs.
- STUDY-001 — Report MS-SSIM + MAE deltas with explicit ± formatting in summary docs.
- TEST-CLI-001 — Archive collect + exec pytest logs plus CLI stdout under `$HUB` every loop.
- PREVIEW-PHASE-001 — Ensure `metrics_delta_highlights_preview.txt` lists only phase deltas with ± tokens (no amplitude noise).
- PHASEC-METADATA-001 — Respect the refreshed Phase C metadata layout; failures here must halt the run.

Pointers
- plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/implementation.md:204 — Active Phase G checklist + remaining evidence tasks.
- plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/plan/plan.md:1 — Detailed objectives + execution sketch for this rerun.
- docs/TESTING_GUIDE.md:389 — Phase G full-pipeline orchestrator commands + evidence requirements.
- docs/fix_plan.md:4 — Initiative ledger (update Attempts History with run evidence + summaries).

Next Up (optional)
1. Run `python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/report_phase_g_dense_metrics.py --metrics "$HUB"/analysis/metrics_delta_summary.json --ms-ssim-threshold 0.80` to generate the MS-SSIM sanity table once evidence exists.
2. Execute `python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/analyze_dense_metrics.py --hub "$HUB"` to refresh `analysis/metrics_digest.md` and embed the sanity table inside the digest.
