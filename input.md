Summary: Add hub-relative assertions for the full Phase G execution test, clean up the duplicate metrics-digest banner line, then run the dense `--clobber` + `--post-verify-only` commands into the 2025-11-12 hub and publish MS-SSIM/MAE + preview evidence.
Mode: Perf
Focus: STUDY-SYNTH-FLY64-DOSE-OVERLAP-001 — Phase G comparison & analysis (dense real evidence + automated report)
Branch: feature/torchapi-newprompt
Plan Document: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/implementation.md
Reports Hub (active): plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/
Mapped tests:
- pytest --collect-only tests/study/test_phase_g_dense_orchestrator.py -k exec_prints_highlights_preview -vv
- pytest tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_exec_prints_highlights_preview -vv
Artifacts: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/

Do Now (hard validity contract)
- Implement: tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_exec_prints_highlights_preview — extend the capsys assertions so the counted execution banner must include `CLI logs: cli`, `Analysis outputs: analysis`, and `analysis/artifact_inventory.txt` (TYPE-PATH-001, TEST-CLI-001) before we trust the dense rerun evidence.
- Implement: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py::main — remove the duplicated “Metrics digest” stdout line so the success banner prints the Markdown path once and keeps the CLI log reference distinct, matching what the engineer will archive inside `$HUB/cli/`.
- Validate: AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md HUB=$PWD/plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier pytest --collect-only tests/study/test_phase_g_dense_orchestrator.py -k exec_prints_highlights_preview -vv | tee "$HUB"/collect/pytest_collect_exec_highlights.log (move failures to `$HUB`/red/ before rerun).
- Validate: AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md HUB=$PWD/plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier pytest tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_exec_prints_highlights_preview -vv | tee "$HUB"/green/pytest_exec_highlights.log (archive failing output under `$HUB`/red/pytest_exec_highlights.log` before fixing).
- Execute: AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md HUB=$PWD/plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py --hub "$HUB" --dose 1000 --view dense --splits train test --clobber |& tee "$HUB"/cli/run_phase_g_dense_stdout.log, ensuring `{analysis,cli}` now contain the complete Phase C→G bundle (phase logs, SSIM grid summary/log, verification_report.json, verify_dense_stdout.log, check_dense_highlights.log, metrics_delta_summary.json, metrics_delta_highlights_preview.txt, artifact_inventory.txt).
- Execute: AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md HUB=$PWD/plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py --hub "$HUB" --post-verify-only |& tee "$HUB"/cli/run_phase_g_dense_post_verify_only.log so the shortened chain regenerates SSIM grid + verification artifacts, refreshes `analysis/artifact_inventory.txt`, and prints the hub-relative banner lines you just guarded.
- Document: Update `$HUB/summary/summary.md`, `$HUB/summary.md`, docs/fix_plan.md, and galph_memory with runtimes, MS-SSIM ±0.000 / MAE ±0.000000 deltas, preview verdict (phase-only), SSIM grid table path, verification/highlights log references, CLI/test selectors, and any RED log context (PREVIEW-PHASE-001, TEST-CLI-001).

How-To Map
1. `export AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md; export HUB=$PWD/plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier` — run this in every shell before invoking pytest or the orchestrator so POLICY-001 stays satisfied and all logs land under the correct hub.
2. In `plans/active/.../bin/run_phase_g_dense.py::main`, replace the second `print(f"Metrics digest: {metrics_digest_md.relative_to(hub)}")` with nothing (we already print the Markdown path once) and keep `print(f"Metrics digest log: {analyze_digest_log.relative_to(hub)}")` so the banner now lists one Markdown + one CLI log reference.
3. In `tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_exec_prints_highlights_preview`, keep the existing highlights assertions but append checks that `stdout` contains `CLI logs: cli`, `Analysis outputs: analysis`, and `analysis/artifact_inventory.txt`.
4. `pytest --collect-only tests/study/test_phase_g_dense_orchestrator.py -k exec_prints_highlights_preview -vv | tee "$HUB"/collect/pytest_collect_exec_highlights.log` (copy any RED output to `$HUB`/red/pytest_collect_exec_highlights.log before rerun).
5. `pytest tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_exec_prints_highlights_preview -vv | tee "$HUB"/green/pytest_exec_highlights.log` (append `-rerunN` suffixes if multiple attempts are required).
6. `python plans/active/.../bin/run_phase_g_dense.py --hub "$HUB" --dose 1000 --view dense --splits train test --clobber |& tee "$HUB"/cli/run_phase_g_dense_stdout.log`; after completion, confirm `{analysis,cli}` hold the SSIM grid summary/log, verification_report.json, verify_dense_stdout.log, check_dense_highlights.log, metrics_delta_summary.json, metrics_delta_highlights_preview.txt, and artifact_inventory.txt.
7. `python plans/active/.../bin/run_phase_g_dense.py --hub "$HUB" --post-verify-only |& tee "$HUB"/cli/run_phase_g_dense_post_verify_only.log`; verify that timestamps on SSIM grid + verification artifacts refresh and the banner still shows hub-relative paths.
8. If verifier/highlights fail, run `python plans/active/.../bin/check_dense_highlights_match.py --hub "$HUB" --debug | tee "$HUB"/red/check_dense_highlights_manual.log`, capture the blocker signature inside `$HUB`/red/, update docs/fix_plan.md + galph_memory, then rerun once resolved.
9. Update `$HUB/summary/summary.md` and `$HUB/summary.md` with runtimes, CLI/test commands, MS-SSIM ±0.000 / MAE ±0.000000 deltas, preview verdict, SSIM grid table link, verification/highlights log references, and doc/test notes; mirror the same evidence in docs/fix_plan.md and galph_memory before finishing the loop.

Pitfalls To Avoid
- Do not start `--post-verify-only` until the full `--clobber` run succeeds; the short mode assumes Phase C→F artifacts exist.
- Keep `AUTHORITATIVE_CMDS_DOC` exported for every subprocess; missing env causes orchestrator exits (POLICY-001).
- Tee every pytest/CLI command into `$HUB` (`collect/`, `green/`, `cli/`, `red/`); missing logs invalidate evidence (TEST-CLI-001).
- Treat any preview text containing amplitude tokens or missing `±` signs as a blocker (PREVIEW-PHASE-001) and capture the failing log before rerun.
- Use `--clobber` instead of manual deletion so `prepare_hub` handles stale artifacts safely (DATA-001).
- Keep success-banner paths hub-relative; never reintroduce `/home/...` output when touching the orchestrator (TYPE-PATH-001).
- If `verify_dense_pipeline_artifacts.py` raises, stop immediately, drop the failing CLI log under `$HUB`/red/, and record the failure in docs/fix_plan.md before attempting another run.

If Blocked
- Archive failing CLI/test output under `$HUB`/red/ with a short blocker note (phase, log path, exception text), update docs/fix_plan.md Attempts History + galph_memory with the failure signature, and halt retries until the supervisor provides guidance.
- If the dense pipeline fails twice in a row, capture the tail of `cli/run_phase_g_dense_stdout.log` plus the specific phase log, mark the attempt blocked in docs/fix_plan.md, and stop to avoid burning more time.

Findings Applied (Mandatory)
- POLICY-001 — PyTorch + AUTHORITATIVE_CMDS_DOC are mandatory for the Phase F/verification helpers.
- CONFIG-001 — Do not disturb `update_legacy_dict` ordering when editing the orchestrator; dense runs rely on the current bridge.
- DATA-001 — Artifact inventory + verifier outputs must exist; treat missing files as fatal and capture logs before rerunning.
- TYPE-PATH-001 — Success banners/logs must emit hub-relative paths; new assertions + banner fix enforce this.
- STUDY-001 — Record MS-SSIM ±0.000 / MAE ±0.000000 in summary.md + docs/fix_plan.md for comparability.
- TEST-CLI-001 — Maintain RED/GREEN/collect pytest logs plus CLI bundles with SUCCESS sentinels for each phase.
- PREVIEW-PHASE-001 — Highlights preview must stay phase-only; rerun the checker + archive diagnostics if amplitude strings leak in.
- PHASEC-METADATA-001 — Leave the Phase C metadata guard intact; failures from `validate_phase_c_metadata` are hard blocks.

Pointers
- docs/fix_plan.md:18 — Active focus entry + Latest Attempt summary for this initiative.
- plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/implementation.md:204 — Checklist items covering the new test requirement and the dense run/remediation steps.
- plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/plan/plan.md:1 — Execution sketch with the exact command sequence and acceptance criteria for this hub.

Next Up (optional)
1. Repeat the dense evidence workflow for the sparse view (dose 1000) once Phase G dense artifacts are archived.
2. Extend the verifier to emit a summarized JSON verdict for CI ingestion after the dense evidence + banner guard land.
