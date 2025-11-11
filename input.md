Summary: Add a digest-banner regression test and then rerun the dense Phase C→G pipeline (`--clobber` + `--post-verify-only`) so `{analysis,verification,metrics}` finally populate the 2025-11-12 hub with MS-SSIM/MAE + preview/verifier evidence.
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
  - Implement: tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_exec_runs_analyze_digest — add `stdout.count("Metrics digest: ") == 1` (or equivalent) so the test fails if the Markdown line reappears twice while still asserting the `Metrics digest log:` stdout line (TYPE-PATH-001, TEST-CLI-001).
  - Pytest: AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md HUB=$PWD/plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier pytest --collect-only tests/study/test_phase_g_dense_orchestrator.py -k exec_runs_analyze_digest -vv | tee "$HUB"/collect/pytest_collect_exec_digest.log
  - Pytest: AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md HUB=$PWD/plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier pytest tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_exec_runs_analyze_digest -vv | tee "$HUB"/green/pytest_exec_digest.log
  - CLI: AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md HUB=$PWD/plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py --hub "$HUB" --dose 1000 --view dense --splits train test --clobber |& tee "$HUB"/cli/run_phase_g_dense_stdout.log
  - CLI: AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md HUB=$PWD/plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py --hub "$HUB" --post-verify-only |& tee "$HUB"/cli/run_phase_g_dense_post_verify_only.log
  - Artifacts: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/

How-To Map
1. `export AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md; export HUB=$PWD/plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier` before *every* pytest/CLI command; orchestrator helpers exit early if the env var is missing (POLICY-001).
2. In `tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_exec_runs_analyze_digest`, use `stdout.count("Metrics digest: ")` (or a regex) to assert the Markdown path appears exactly once while keeping the existing `Metrics digest log:` assertion; update the failure message to mention TYPE-PATH-001.
3. `pytest --collect-only tests/study/test_phase_g_dense_orchestrator.py -k exec_runs_analyze_digest -vv | tee "$HUB"/collect/pytest_collect_exec_digest.log`; move any RED output to `$HUB`/red/pytest_collect_exec_digest.log before rerunning.
4. `pytest tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_exec_runs_analyze_digest -vv | tee "$HUB"/green/pytest_exec_digest.log`; if it fails, capture the raw stdout in `$HUB`/red/pytest_exec_digest.log` with the stack trace before fixing.
5. `python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py --hub "$HUB" --dose 1000 --view dense --splits train test --clobber |& tee "$HUB"/cli/run_phase_g_dense_stdout.log`; verify `{analysis,cli}` now contain SSIM grid summary/log, verification_report.json, verify_dense_stdout.log, check_dense_highlights.log, metrics_delta_summary.json, metrics_delta_highlights_preview.txt, artifact_inventory.txt, and updated aggregate/metrics/preview files.
6. `python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py --hub "$HUB" --post-verify-only |& tee "$HUB"/cli/run_phase_g_dense_post_verify_only.log`; confirm SSIM grid + verifier artifacts refresh and the success banner reprints the hub-relative paths.
7. If verifier/highlights fail, run `python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/check_dense_highlights_match.py --hub "$HUB" --debug | tee "$HUB"/red/check_dense_highlights_manual.log` before rerunning; log the blocker signature in docs/fix_plan.md and galph_memory.
8. Update `$HUB/summary/summary.md` (and copy to `$HUB/summary.md`) with runtimes, CLI/test commands, MS-SSIM ±0.000 / MAE ±0.000000 deltas, preview verdict (phase-only), SSIM grid path, verification/highlights logs, artifact inventory + metrics digest references, and doc/test updates; mirror the evidence in docs/fix_plan.md and galph_memory.

Pitfalls To Avoid
- Never start `--post-verify-only` until the counted `--clobber` run finishes successfully; it reuses artifacts in place.
- Keep AUTHORITATIVE_CMDS_DOC exported in every shell; missing env vars abort orchestrator helpers (POLICY-001).
- Always tee pytest/CLI output into `$HUB` (`collect/`, `green/`, `cli/`, `red/`); missing logs break TEST-CLI-001 evidence requirements.
- Treat any preview text that mentions “amplitude” or lacks explicit ± signs as a blocker; archive the failing preview + verifier logs before rerunning (PREVIEW-PHASE-001).
- Do not delete hub directories manually—use `--clobber` so `prepare_hub` handles stale outputs safely (DATA-001).
- Stop immediately if `verify_dense_pipeline_artifacts.py` or `check_dense_highlights_match.py` fails; capture the log under `$HUB`/red/ and update docs/fix_plan.md instead of retrying blindly.
- Ensure all success-banner lines remain hub-relative; no `/home/...` strings should appear after your edits (TYPE-PATH-001).
- Copy the refreshed summary block to both `$HUB/summary/summary.md` and `$HUB/summary.md` before ending the loop.

If Blocked
- Archive failing pytest/CLI logs under `$HUB`/red/ with a short README describing the phase, command, and exception text; include the same signature in docs/fix_plan.md and galph_memory, then stop.
- For pipeline blockers, capture the tail of `cli/run_phase_g_dense_stdout.log` plus the specific phase log (e.g., `phase_e_dense.log`), mark the attempt blocked in docs/fix_plan.md, and await supervisor direction before rerunning.

Findings Applied (Mandatory)
- POLICY-001 (docs/findings.md:8) — PyTorch + AUTHORITATIVE_CMDS_DOC must be available for Phase F verification helpers.
- CONFIG-001 (docs/findings.md:10) — Preserve the `update_legacy_dict` bridge ordering while touching the orchestrator.
- DATA-001 (docs/findings.md:14) — Artifact inventory + SSIM grid/verifier files are required deliverables; missing files mean failure.
- TYPE-PATH-001 (docs/findings.md:21) — Success banners/logs must use hub-relative paths; the new digest test enforces this.
- STUDY-001 (docs/findings.md:16) — Record MS-SSIM ±0.000 and MAE ±0.000000 deltas in summary.md + docs/fix_plan.md.
- TEST-CLI-001 (docs/findings.md:23) — Maintain RED/GREEN/collect pytest logs and full CLI bundles with SUCCESS sentinels.
- PREVIEW-PHASE-001 (docs/findings.md:24) — Highlights preview must stay phase-only; treat amplitude contamination as a hard stop.
- PHASEC-METADATA-001 (docs/findings.md:22) — Do not bypass the Phase C metadata guard; failures are blocking until resolved.

Pointers
- docs/fix_plan.md:4 — Active focus metadata + latest attempt requirements.
- plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/implementation.md:204 — Phase G checklist tracking the dense run + digest guard.
- plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/plan/plan.md:1 — Execution sketch and acceptance criteria for this hub.
- plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/summary.md:1 — Accumulated turn summaries and artifact expectations.

Next Up (optional)
1. After dense evidence lands, repeat the same pipeline + verifier workflow for the sparse view (dose 1000) using the existing hub scaffold.
2. Extend `verify_dense_pipeline_artifacts.py` to emit a structured JSON verdict for CI ingestion once the dense rerun artifacts exist.
