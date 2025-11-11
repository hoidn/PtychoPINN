Summary: Add the `Metrics digest log:` uniqueness guard and then rerun the dense Phase C→G pipeline (`--clobber` + `--post-verify-only`) so `{analysis,verification,metrics}` finally populate the 2025-11-12 hub with MS-SSIM/MAE + preview/verifier evidence.
Mode: Perf
Focus: STUDY-SYNTH-FLY64-DOSE-OVERLAP-001 — Phase G comparison & analysis (dense real evidence + automated report)
Branch: feature/torchapi-newprompt
Plan Document: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/implementation.md
Reports Hub (active): plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/
Mapped tests:
- pytest --collect-only tests/study/test_phase_g_dense_orchestrator.py -k exec_runs_analyze_digest -vv
- pytest tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_exec_runs_analyze_digest -vv
Artifacts: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/

Do Now (hard validity contract)
- STUDY-SYNTH-FLY64-DOSE-OVERLAP-001
  - Implement: tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_exec_runs_analyze_digest — add `stdout.count("Metrics digest log: ") == 1` (or equivalent) so the test fails if the CLI log reference appears twice while preserving the existing Markdown banner/delta assertions (TYPE-PATH-001, TEST-CLI-001).
  - Pytest: AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md HUB=$PWD/plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier pytest --collect-only tests/study/test_phase_g_dense_orchestrator.py -k exec_runs_analyze_digest -vv | tee "$HUB"/collect/pytest_collect_exec_digest.log
  - Pytest: AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md HUB=$PWD/plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier pytest tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_exec_runs_analyze_digest -vv | tee "$HUB"/green/pytest_exec_digest.log
  - CLI: AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md HUB=$PWD/plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py --hub "$HUB" --dose 1000 --view dense --splits train test --clobber |& tee "$HUB"/cli/run_phase_g_dense_stdout.log
  - CLI: AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md HUB=$PWD/plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py --hub "$HUB" --post-verify-only |& tee "$HUB"/cli/run_phase_g_dense_post_verify_only.log
  - Artifacts: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/

How-To Map
1. `export AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md; export HUB=$PWD/plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier` before *every* pytest/CLI command; orchestrator helpers refuse to run if the env var is missing (POLICY-001).
2. In `tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_exec_runs_analyze_digest`, insert the new `stdout.count("Metrics digest log: ") == 1` assertion directly after the existing `Metrics digest:` guard so both banner lines stay unique; keep the existing failure messaging that cites TYPE-PATH-001.
3. `pytest --collect-only tests/study/test_phase_g_dense_orchestrator.py -k exec_runs_analyze_digest -vv | tee "$HUB"/collect/pytest_collect_exec_digest.log`; if collection fails, move the log to `$HUB/red/pytest_collect_exec_digest.log` and stop for supervisor triage.
4. `pytest tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_exec_runs_analyze_digest -vv | tee "$HUB"/green/pytest_exec_digest.log`; on failure, archive the RED log under `$HUB/red/pytest_exec_digest.log` with the stack trace before fixing the assertion.
5. `python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py --hub "$HUB" --dose 1000 --view dense --splits train test --clobber |& tee "$HUB"/cli/run_phase_g_dense_stdout.log`; confirm the CLI banner enumerates hub-relative paths, then verify `{analysis,cli}` now contain SSIM grid summaries/logs, verification_report.json, verify_dense_stdout.log, check_dense_highlights.log, metrics_delta_summary.json, metrics_delta_highlights_preview.txt, metrics_digest.md, aggregate_highlights.txt, and artifact_inventory.txt.
6. `python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py --hub "$HUB" --post-verify-only |& tee "$HUB"/cli/run_phase_g_dense_post_verify_only.log`; ensure the short chain regenerates SSIM grid + verification artifacts and rewrites `analysis/artifact_inventory.txt` without touching Phase C (PHASEC-METADATA-001).
7. If verifier/highlights fail, run `python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/check_dense_highlights_match.py --hub "$HUB" --debug | tee "$HUB"/red/check_dense_highlights_manual.log` to capture the mismatch signature before retrying.
8. Update `$HUB/summary/summary.md` (and copy to `$HUB/summary.md`) with runtimes, CLI/test commands, MS-SSIM ±0.000 / MAE ±0.000000 deltas, preview verdict (phase-only), SSIM grid + verifier/highlights links, artifact inventory path, and doc/test guard status; mirror the evidence in docs/fix_plan.md and galph_memory.

Pitfalls To Avoid
- Do not start the `--post-verify-only` sweep until the counted `--clobber` run finishes successfully; it reuses artifacts in place.
- Keep AUTHORITATIVE_CMDS_DOC and HUB exported for every subprocess or the orchestrator exits early (POLICY-001).
- Always tee pytest/CLI output into `$HUB` (`collect/`, `green/`, `cli/`, `red/`) so TEST-CLI-001 evidence stays complete.
- Treat any preview text that mentions “amplitude” or drops ± signs as a blocker; archive the preview + verifier logs before rerunning (PREVIEW-PHASE-001).
- Use `--clobber` instead of manually deleting hub contents so `prepare_hub` can clean stale artifacts safely (DATA-001).
- Stop immediately if `verify_dense_pipeline_artifacts.py` or `check_dense_highlights_match.py` fails; capture the RED log and surface the blocker in docs/fix_plan.md rather than guessing.
- Make sure success-banner lines remain hub-relative; `/home/...` paths violate TYPE-PATH-001.
- Preserve Phase C metadata outputs when rerunning; never delete patched NPZ files outside the orchestrator (PHASEC-METADATA-001).

If Blocked
- Archive failing pytest/CLI logs under `$HUB/red/` with a short README describing the command, phase, and exception text, then record the same signature in docs/fix_plan.md + galph_memory before stopping.
- For pipeline blockers, capture the tail of `cli/run_phase_g_dense_stdout.log` plus the relevant phase log (e.g., `phase_e_dense.log`), mark the attempt blocked in docs/fix_plan.md, and wait for supervisor guidance.

Findings Applied (Mandatory)
- POLICY-001 (docs/findings.md:8) — PyTorch + AUTHORITATIVE_CMDS_DOC must stay exported for Phase F/G helpers.
- CONFIG-001 (docs/findings.md:10) — Keep `update_legacy_dict` ordering intact when the orchestrator rebuilds Phase C state.
- DATA-001 (docs/findings.md:14) — The rerun must emit the full metrics/verifier/SSIM inventory bundle before being marked complete.
- TYPE-PATH-001 (docs/findings.md:21) — Success banners/log lines stay hub-relative; new test guard enforces uniqueness.
- STUDY-001 (docs/findings.md:16) — Report MS-SSIM ±0.000 / MAE ±0.000000 deltas in the summaries.
- TEST-CLI-001 (docs/findings.md:23) — Maintain RED/GREEN/collect pytest logs plus full CLI bundle with SUCCESS sentinels.
- PREVIEW-PHASE-001 (docs/findings.md:24) — Highlights preview must remain phase-only; treat amplitude text as failure evidence.
- PHASEC-METADATA-001 (docs/findings.md:22) — Respect the refreshed Phase C directory layout; verification-only sweeps cannot bypass metadata guards.

Pointers
- docs/fix_plan.md:4 — Active focus metadata + latest attempt requirements for Phase G evidence.
- plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/implementation.md:204 — Checklist tracking digest guards and dense rerun tasks.
- plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/plan/plan.md:1 — Execution sketch + acceptance criteria for this hub.
- plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py:1280 — Success-banner + artifact inventory logic referenced by the rerun.
- docs/TESTING_GUIDE.md:1 — Authoritative commands + pytest collection guidance referenced via AUTHORITATIVE_CMDS_DOC.

Next Up (optional)
1. Repeat the same dense evidence workflow for the sparse view once this hub contains Phase C→G artifacts.
2. Extend `verify_dense_pipeline_artifacts.py` to emit a structured JSON verdict for CI ingestion after the rerun evidence is green.
