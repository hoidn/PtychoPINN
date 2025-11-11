Summary: Normalize the Phase G success-banner paths to hub-relative strings, then run the dense Phase C→G pipeline (plus --post-verify-only) into the 2025-11-12 hub and publish MS-SSIM/MAE + preview evidence.
Mode: Perf
Focus: STUDY-SYNTH-FLY64-DOSE-OVERLAP-001 — Phase G comparison & analysis (dense real evidence + automated report)
Branch: feature/torchapi-newprompt
Plan Document: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/implementation.md
Reports Hub (active): plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/
Mapped tests:
- pytest --collect-only tests/study/test_phase_g_dense_orchestrator.py -k post_verify_only -vv
- pytest tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_post_verify_only_executes_chain -vv
Artifacts: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/

Do Now (hard validity contract)
- Implement: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py::main — convert every success-banner artifact reference (CLI logs, analysis outputs, aggregate report, highlights, metrics digest/log, metrics delta JSON/TXT/preview, SSIM grid summary/log, verification report/log, highlights log) to `Path(...).relative_to(hub)` in both the full pipeline and `--post-verify-only` flows; keep artifact inventory validation intact (TYPE-PATH-001, DATA-001).
- Implement: tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_post_verify_only_executes_chain — extend stdout assertions so the success banner must contain `CLI logs: cli` and `Analysis outputs: analysis` (hub-relative), in addition to the existing artifact-inventory check and command-order coverage (TEST-CLI-001).
- Validate: AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md HUB=$PWD/plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier pytest --collect-only tests/study/test_phase_g_dense_orchestrator.py -k post_verify_only -vv | tee "$HUB"/collect/pytest_collect_orchestrator_post_verify_only.log (RED logs → `$HUB`/red/ on failure, rerun after fixes).
- Validate: AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md HUB=$PWD/plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier pytest tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_post_verify_only_executes_chain -vv | tee "$HUB"/green/pytest_post_verify_only.log (archive failures under `$HUB`/red/ with failure text before rerun).
- Execute: AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md HUB=$PWD/plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py --hub "$HUB" --dose 1000 --view dense --splits train test --clobber |& tee "$HUB"/cli/run_phase_g_dense_stdout.log, ensuring `{analysis,cli}` gain the full Phase C→G artifact bundle (SSIM grid summary/log, verification_report.json, verify_dense_stdout.log, check_dense_highlights.log, metrics_delta_summary.json, metrics_delta_highlights_preview.txt, artifact_inventory.txt).
- Execute: AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md HUB=$PWD/plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py --hub "$HUB" --post-verify-only |& tee "$HUB"/cli/run_phase_g_dense_post_verify_only.log so the shortened chain regenerates SSIM grid + verification artifacts, refreshes `analysis/artifact_inventory.txt`, and prints hub-relative success-banner paths.
- Document: Update `$HUB/summary/summary.md`, `$HUB/summary.md`, docs/fix_plan.md, and galph_memory with runtimes, MS-SSIM ±0.000 / MAE ±0.000000 deltas, preview verdict (phase-only), SSIM grid table reference, verification/highlights log paths, hub-relative success-banner status, CLI/test selectors, and rerun commands (PREVIEW-PHASE-001, TEST-CLI-001).

How-To Map
1. `export AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md; export HUB=$PWD/plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier` (POLICY-001; keep both exports for every subsequent command).
2. In `plans/active/.../bin/run_phase_g_dense.py::main`, wrap every success-banner path with `.relative_to(hub)` (full run section: CLI logs, Analysis outputs, Aggregate report, Highlights, Metrics digest/log, delta JSON/TXT/preview, SSIM grid summary/log, verification report/log, highlights log; `--post-verify-only` section: CLI logs, Analysis outputs, SSIM grid summary/log, verification report/log, highlights log). Leave `Artifacts saved to: {hub}` and runtime prints untouched.
3. Update `tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_post_verify_only_executes_chain` to keep capsys capture but add assertions that stdout contains `CLI logs: cli` and `Analysis outputs: analysis` (hub-relative). Preserve existing command-order, inventory-call, and artifact-inventory string checks.
4. `pytest --collect-only tests/study/test_phase_g_dense_orchestrator.py -k post_verify_only -vv | tee "$HUB"/collect/pytest_collect_orchestrator_post_verify_only.log` (move failing logs to `$HUB`/red/ if needed, rerun after fixes).
5. `pytest tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_post_verify_only_executes_chain -vv | tee "$HUB"/green/pytest_post_verify_only.log` (record reruns with suffix `-rerunN.log`).
6. `python plans/active/.../bin/run_phase_g_dense.py --hub "$HUB" --dose 1000 --view dense --splits train test --clobber |& tee "$HUB"/cli/run_phase_g_dense_stdout.log`; watch for SUCCESS sentinels per phase, then confirm `{analysis,cli}` contain SSIM grid summary/log, verification_report.json, verify_dense_stdout.log, check_dense_highlights.log, metrics_delta_summary.json, metrics_delta_highlights_preview.txt, artifact_inventory.txt.
7. `python plans/active/.../bin/run_phase_g_dense.py --hub "$HUB" --post-verify-only |& tee "$HUB"/cli/run_phase_g_dense_post_verify_only.log`; verify that hub-relative banner lines print and that SSIM grid + verification artifacts have refreshed timestamps.
8. On preview/highlights discrepancies run `python plans/active/.../bin/check_dense_highlights_match.py --hub "$HUB" --debug | tee "$HUB"/analysis/check_dense_highlights_manual.log`, archive blockers under `$HUB`/red/, and only rerun after addressing root cause.
9. Update `$HUB/summary/summary.md` + `$HUB/summary.md` with runtimes, MS-SSIM/MAE ± deltas, preview verdict, SSIM grid table link, verification/highlights log paths, collect/green log references, CLI command strings, and mention that hub-relative banners + post-verify-only rerun succeeded; mirror the same info in docs/fix_plan.md Attempts History and galph_memory.

Pitfalls To Avoid
- Do not leave any absolute `/home/...` paths in success banners; every artifact reference must be hub-relative (TYPE-PATH-001).
- Do not start `--post-verify-only` until the full `--clobber` run succeeds; the shortened mode assumes Phase C→F artifacts exist.
- Never combine `--post-verify-only` with `--clobber` or `--skip-post-verify`; mutual exclusion guards exit non-zero.
- Keep `AUTHORITATIVE_CMDS_DOC` exported for every subprocess so verifiers inherit the command reference (POLICY-001).
- Tee every CLI/test command into `$HUB` (`cli/`, `collect/`, `green/`, `red/`); missing logs invalidate evidence (TEST-CLI-001).
- Use `--clobber` instead of manual deletion so `prepare_hub` handles stale artifacts safely (DATA-001).
- Treat preview regressions (amplitude tokens or missing ±) as blockers; rerun checker and capture diagnostics (PREVIEW-PHASE-001).
- If a phase fails, stop immediately, drop the failing log in `$HUB`/red/, and record the failure in docs/fix_plan.md before retrying.

If Blocked
- Archive failing CLI/test output under `$HUB`/red/ with a short blocker note (phase name, log path, exception text), update docs/fix_plan.md Attempts History + galph_memory with the failure signature, and leave the hub intact for debugging before reruns.
- If the dense pipeline aborts repeatedly, capture `cli/run_phase_g_dense_stdout.log` tail plus the specific phase log causing the abort, mark the attempt blocked in docs/fix_plan.md, and stop after the second failed rerun pending supervisor guidance.

Findings Applied (Mandatory)
- POLICY-001 — PyTorch remains required for Phase F + verification; keep AUTHORITATIVE_CMDS_DOC exported for every command.
- CONFIG-001 — Do not reorder `update_legacy_dict` or other legacy bridges while editing the orchestrator; dense run relies on the existing sequence.
- DATA-001 — Artifact inventory + verifier outputs must exist; fail fast if files are missing and capture logs before reruns.
- TYPE-PATH-001 — Success banners/logs must use hub-relative paths, motivating the code/test change in this Do Now.
- STUDY-001 — Report MS-SSIM ±0.000 / MAE ±0.000000 in summary.md and docs/fix_plan.md for study comparability.
- TEST-CLI-001 — Maintain RED/GREEN/collect pytest logs plus CLI log bundles with dose/view-aware filenames and SUCCESS sentinels.
- PREVIEW-PHASE-001 — Highlights preview must stay phase-only; rerun checker if amplitude strings leak in.
- PHASEC-METADATA-001 — Leave the Phase C metadata guard intact; treat any failure from `validate_phase_c_metadata` as a hard block.

Pointers
- docs/fix_plan.md:18 — Active focus entry, guardrail, and Latest Attempt summary for this initiative.
- plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/implementation.md:199 — Phase G checklist detailing the relative-path task and evidence runs.
- plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/plan/plan.md:1 — Execution sketch + objectives for the dense run + verification sweep.

Next Up (optional)
1. After dense evidence lands, repeat the same workflow for the sparse view (dose 1000) so both overlap conditions have Phase C→G artifacts.
2. Extend the verifier to emit a summarized JSON verdict for CI ingestion once the dense evidence + hub-relative guard passes.
