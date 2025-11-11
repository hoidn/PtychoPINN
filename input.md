Summary: Guard the verification/SSIM grid success-banner lines in test_run_phase_g_dense_exec_runs_analyze_digest, then rerun the dense Phase C→G pipeline (--clobber + --post-verify-only) to populate the 2025-11-12 hub with full analysis/verifier evidence.
Mode: Perf
Focus: STUDY-SYNTH-FLY64-DOSE-OVERLAP-001 — Phase G comparison & analysis (dense real evidence + automated report)
Branch: feature/torchapi-newprompt
Plan Document: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/implementation.md
Reports Hub (active): plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/
Mapped tests:
- AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md HUB=$PWD/plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier pytest --collect-only tests/study/test_phase_g_dense_orchestrator.py -k exec_runs_analyze_digest -vv
- AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md HUB=$PWD/plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier pytest tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_exec_runs_analyze_digest -vv
Artifacts: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/

Do Now (hard validity contract)
- STUDY-SYNTH-FLY64-DOSE-OVERLAP-001
  - Implement: tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_exec_runs_analyze_digest — add assertions that stdout prints the SSIM grid summary/log plus the verification report/log and highlights check log lines, and update the test’s stub_run_command helper to write each CLI log so `.exists()` checks succeed (TEST-CLI-001, TYPE-PATH-001).
  - Pytest: AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md HUB=$PWD/plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier pytest --collect-only tests/study/test_phase_g_dense_orchestrator.py -k exec_runs_analyze_digest -vv | tee "$HUB"/collect/pytest_collect_exec_digest.log
  - Pytest: AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md HUB=$PWD/plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier pytest tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_exec_runs_analyze_digest -vv | tee "$HUB"/green/pytest_exec_digest.log
  - CLI: AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md HUB=$PWD/plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py --hub "$HUB" --dose 1000 --view dense --splits train test --clobber |& tee "$HUB"/cli/run_phase_g_dense_stdout.log
  - CLI: AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md HUB=$PWD/plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py --hub "$HUB" --post-verify-only |& tee "$HUB"/cli/run_phase_g_dense_post_verify_only.log
  - Artifacts: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/

How-To Map
1. `export AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md; export HUB=$PWD/plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier` before editing or running anything; the orchestrator refuses to run without both vars (POLICY-001).
2. In `tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_exec_runs_analyze_digest`, append assertions that stdout contains `SSIM Grid Summary (phase-only): analysis/ssim_grid_summary.md`, `SSIM Grid log: cli/ssim_grid_cli.log`, `Verification report: analysis/verification_report.json`, `Verification log: analysis/verify_dense_stdout.log`, and `Highlights check log: analysis/check_dense_highlights.log`. Within the same test, update `stub_run_command` to call `log_path.parent.mkdir(...); log_path.write_text(...)` for every invocation so those banner checks see existing files.
3. `AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md HUB=$PWD/plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier pytest --collect-only tests/study/test_phase_g_dense_orchestrator.py -k exec_runs_analyze_digest -vv | tee "$HUB"/collect/pytest_collect_exec_digest.log`; move failures to `$HUB`/red/ and stop if collection breaks.
4. `AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md HUB=$PWD/plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier pytest tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_exec_runs_analyze_digest -vv | tee "$HUB"/green/pytest_exec_digest.log`; archive RED logs under `$HUB`/red/ on failure before retrying.
5. `AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md HUB=$PWD/plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py --hub "$HUB" --dose 1000 --view dense --splits train test --clobber |& tee "$HUB"/cli/run_phase_g_dense_stdout.log`; verify SUCCESS banners mention the SSIM grid + verification artifacts and that `{analysis,cli}` now contain the listed files (metrics deltas, preview, digest, verification_report.json, verify_dense_stdout.log, check_dense_highlights.log, ssim_grid_summary.md, artifact_inventory.txt).
6. `AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md HUB=$PWD/plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py --hub "$HUB" --post-verify-only |& tee "$HUB"/cli/run_phase_g_dense_post_verify_only.log`; confirm the banner reports regenerated SSIM grid + verification artifacts and refreshed `analysis/artifact_inventory.txt`.
7. If SSIM grid or highlights mismatch, run `python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/check_dense_highlights_match.py --hub "$HUB" --debug | tee "$HUB"/red/check_dense_highlights_manual.log` before attempting another rerun; archive the failing log and surface the blocker.
8. Update `$HUB/summary/summary.md` (and copy to `$HUB/summary.md`) with runtimes, CLI/test selectors, MS-SSIM ±0.000 / MAE ±0.000000 deltas, preview verdict, SSIM grid path, verification/highlights log links, artifact inventory path, and doc/ledger statuses; then refresh docs/fix_plan.md + galph_memory with the same evidence references.

Pitfalls To Avoid
- Do not skip exporting AUTHORITATIVE_CMDS_DOC and HUB for every pytest/CLI command (POLICY-001 guard will abort the run).
- Keep the test assertions device/dtype-agnostic; only validate stdout text and run_command side effects (no filesystem hardcodes outside the hub) (TEST-CLI-001).
- When updating stub_run_command, ensure each generated log uses UTF-8 and mirrors the orchestrator paths so `.exists()` checks behave exactly like the real CLI logs.
- Treat any preview file that mentions “amplitude” or lacks ± signs as a blocker; archive preview + log evidence before rerunning (PREVIEW-PHASE-001).
- Use `--clobber` instead of manual deletions; let `prepare_hub` handle stale artifacts (DATA-001).
- Stop immediately if verify/highlights commands fail; move the failing log to `$HUB`/red/ with a short README and document the blocker in docs/fix_plan.md.
- Never touch Phase C patched NPZs outside the orchestrator; rely on `--post-verify-only` for verification sweeps (PHASEC-METADATA-001).
- Keep success-banner strings hub-relative; any `/home/...` output violates TYPE-PATH-001 and must be fixed before rerunning.

If Blocked
- For pytest failures, archive the log under `$HUB`/red/ (e.g., `pytest_exec_digest.log`) with the stack trace, note the failure signature in docs/fix_plan.md + galph_memory, and halt for supervisor guidance.
- For pipeline blockers, capture the tail of `cli/run_phase_g_dense_stdout.log` plus the implicated phase log (e.g., `cli/phase_e_dense.log`), stash them under `$HUB`/red/, and record the blocker in docs/fix_plan.md + galph_memory before stopping.

Findings Applied (Mandatory)
- POLICY-001 — PyTorch + AUTHORITATIVE_CMDS_DOC export are required for the orchestrator helpers; commands fail fast without them.
- CONFIG-001 — Respect `update_legacy_dict` ordering inside the orchestrator so Phase C metadata stays valid before running comparisons.
- DATA-001 — Dense reruns must emit the complete metrics/verifier/SSIM bundle and regenerate `analysis/artifact_inventory.txt`.
- TYPE-PATH-001 — Success banners/log references stay hub-relative; new assertions lock the verification + SSIM grid lines.
- STUDY-001 — Summaries must report MS-SSIM ±0.000 and MAE ±0.000000 deltas for the dense view.
- TEST-CLI-001 — Archive collect + exec pytest logs and keep CLI outputs with SUCCESS sentinels + helper filenames per dose/view.
- PREVIEW-PHASE-001 — Highlights preview must remain phase-only with ± formatting; failures block completion.
- PHASEC-METADATA-001 — Never bypass the refreshed Phase C layout; verification-only sweeps operate on existing outputs without touching Phase C files.

Pointers
- docs/fix_plan.md:1 — Active focus metadata + latest Attempt summary drive the Do Now requirements.
- plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/implementation.md:204 — Phase G checklist showing the new verification-banner guard plus dense rerun tasks.
- plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/plan/plan.md:1 — Execution sketch + acceptance criteria for this hub.
- plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py:1030 — Command graph + success-banner printing for SSIM grid, verification, highlights, and inventory paths.
- plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/verify_dense_pipeline_artifacts.py:553 — SSIM grid + verification validation rules referenced by the rerun evidence.

Next Up (optional)
1. Once dense evidence is archived, repeat the same workflow for the sparse view hub.
2. Promote the verifier/test updates into CI by wiring the highlights check logs into a JSON verdict for future automation.
