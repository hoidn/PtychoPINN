Summary: Run the counted dense Phase C→G pipeline (dose 1000, view dense) into the 2025-11-12 hub, prove the new `--post-verify-only` workflow on real artifacts, and publish MS-SSIM/MAE deltas + artifact inventory evidence.
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
- Implement: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py::main — after `generate_artifact_inventory(hub)` completes in both the full pipeline path and the `--post-verify-only` path, emit the hub-relative `analysis/artifact_inventory.txt` location in the success banner and fail fast if the file is missing (TYPE-PATH-001, TEST-CLI-001, DATA-001).
- Implement: tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_post_verify_only_executes_chain — capture stdout (capsys) and assert the new success-banner line mentions `analysis/artifact_inventory.txt`, keeping the existing command-order + inventory-call assertions for regression coverage.
- Execute: AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md HUB=$PWD/plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py --hub "$HUB" --dose 1000 --view dense --splits train test --clobber |& tee "$HUB"/cli/run_phase_g_dense_stdout.log to generate fresh Phase C→G artifacts plus SSIM grid/verification/highlights outputs.
- Execute: AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md HUB=$PWD/plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py --hub "$HUB" --post-verify-only |& tee "$HUB"/cli/run_phase_g_dense_post_verify_only.log so the shortened chain regenerates SSIM grid + verification artifacts and refreshes `analysis/artifact_inventory.txt`.
- Validate: pytest --collect-only tests/study/test_phase_g_dense_orchestrator.py -k post_verify_only -vv | tee "$HUB"/collect/pytest_collect_orchestrator_post_verify_only.log (RED logs → `$HUB`/red/ on failure, rerun after fixes).
- Validate: pytest tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_post_verify_only_executes_chain -vv | tee "$HUB"/green/pytest_post_verify_only.log (archive any failure under `$HUB`/red/ and rerun after fixes).
- Document: Update `$HUB/summary/summary.md`, `$HUB/summary.md`, docs/fix_plan.md, and galph_memory with runtimes, MS-SSIM ±0.000 / MAE ±0.000000 deltas, preview verdict, SSIM grid table path, verification/highlights logs, pytest selectors, and CLI command references (PREVIEW-PHASE-001, TEST-CLI-001).

How-To Map
1. `export AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md; export HUB=$PWD/plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier` (keep exports active for every CLI call; POLICY-001).
2. Update `plans/active/.../bin/run_phase_g_dense.py`:
   - In the main pipeline completion banner (after `generate_artifact_inventory`), add `artifact_inventory_path = Path(phase_g_root) / "artifact_inventory.txt"`; if it does not exist raise `RuntimeError` with actionable text, else print `Artifact inventory: {artifact_inventory_path.relative_to(hub)}`.
   - In the `--post-verify-only` success banner, add the same guard/print so verification-only sweeps prove the inventory refresh (TYPE-PATH-001, DATA-001).
3. Extend `tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_post_verify_only_executes_chain` to accept `capsys`, run `main()` once, call `captured = capsys.readouterr()`, and assert `'analysis/artifact_inventory.txt' in captured.out`. Keep the existing run_command-order assertions + `generate_artifact_inventory` call count (TEST-CLI-001).
4. `pytest --collect-only tests/study/test_phase_g_dense_orchestrator.py -k post_verify_only -vv | tee "$HUB"/collect/pytest_collect_orchestrator_post_verify_only.log` (log failure under `$HUB`/red/ before re-running; ensure selector collects >0 tests).
5. `pytest tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_post_verify_only_executes_chain -vv | tee "$HUB"/green/pytest_post_verify_only.log` (expect 1 test run; include rerun logs in `$HUB`/green/).
6. `python plans/active/.../bin/run_phase_g_dense.py --hub "$HUB" --dose 1000 --view dense --splits train test --clobber |& tee "$HUB"/cli/run_phase_g_dense_stdout.log`
   - Monitor CLI to ensure each phase log (`phase_c_generation.log`, `phase_d_dense.log`, `phase_e_{baseline,dense}_gs*.log`, `phase_f_dense_train.log`, `phase_g_dense_compare.log`, helpers, SSIM grid, verify, highlights) lands under `$HUB/cli/`.
   - After completion, verify `$HUB/analysis/metrics_delta_summary.json`, `metrics_delta_highlights_preview.txt`, `ssim_grid_summary.md`, `verification_report.json`, `verify_dense_stdout.log`, `check_dense_highlights.log`, and `artifact_inventory.txt` all exist.
7. `python plans/active/.../bin/run_phase_g_dense.py --hub "$HUB" --post-verify-only |& tee "$HUB"/cli/run_phase_g_dense_post_verify_only.log` (confirm banner prints the artifact inventory path and the log references for SSIM grid + verification).
8. If previews fail, run `python plans/active/.../bin/check_dense_highlights_match.py --hub "$HUB" --debug | tee "$HUB"/analysis/check_dense_highlights_manual.log` before re-running the orchestrator; keep blocker logs under `$HUB`/red/.
9. Update `$HUB/summary/summary.md` + `$HUB/summary.md` with runtime, MS-SSIM/MAE deltas (±0.000/±0.000000), preview verdict, SSIM grid table link, verification/highlights log refs, pytest selectors + log paths, and CLI commands; then refresh docs/fix_plan.md Attempts History + galph_memory with the evidence pointers.

Pitfalls To Avoid
- Do not start `--post-verify-only` before the counted `--clobber` run completes; the flag expects fresh Phase C→G artifacts and will just rerun validators.
- Never combine `--post-verify-only` with `--clobber` or `--skip-post-verify`; the guard exits 1 (TEST-CLI-001).
- Keep `AUTHORITATIVE_CMDS_DOC` exported for every subprocess so verifiers inherit the correct command reference (POLICY-001).
- Always tee CLI/test output into `$HUB` subdirs; missing logs invalidate evidence requirements.
- If any phase fails, stop immediately, archive the failing log under `$HUB`/red/, and document the failure signature in docs/fix_plan.md before rerunning.
- Preview text must contain only phase deltas with explicit ± signs; amplitude tokens signal PREVIEW-PHASE-001 violations.
- Do not delete hub contents manually—use `--clobber` so `prepare_hub` archives stale Phase C outputs safely (DATA-001).
- Ensure MS-SSIM/MAE deltas (±0.000 / ±0.000000) are copied verbatim into summary.md; rounding differently breaks study comparability (STUDY-001).

If Blocked
- Capture the failing CLI/test output under `$HUB`/red/ with a short blocker note, update docs/fix_plan.md Attempts History + galph_memory with the failure signature (phase + log path), and leave the hub intact for debugging.
- If the dense pipeline aborts mid-phase, do not rerun blindly—inspect the blocking log, record the exception in docs/fix_plan.md, and only retry once the root cause is known.

Findings Applied (Mandatory)
- POLICY-001 — PyTorch dependency for Phase F + verification; export AUTHORITATIVE_CMDS_DOC before every command.
- CONFIG-001 — Preserve `update_legacy_dict` ordering inside the orchestrator; do not reorder Phase C setup while editing success banners.
- DATA-001 — Artifact inventory + verifier outputs prove dataset contract compliance; treat missing files as blockers.
- TYPE-PATH-001 — Success banners/logs must use hub-relative paths for every artifact reference.
- STUDY-001 — Report MS-SSIM/MAE deltas with ± precision inside summary.md and docs/fix_plan.md.
- TEST-CLI-001 — Maintain RED/GREEN/collect logs for orchestrator tests + CLI commands with correct filenames/dose suffixes.
- PREVIEW-PHASE-001 — Highlights preview must stay phase-only; rerun checker if amplitude text appears.
- PHASEC-METADATA-001 — Dense rerun cannot tamper with Phase C metadata; rely on existing guard inside `run_phase_g_dense.py`.

Pointers
- docs/fix_plan.md:18 — Current focus metadata, guardrails, and Do Now summary for STUDY-SYNTH-FLY64-DOSE-OVERLAP-001.
- plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/implementation.md:199 — Phase G context + active checklist outlining the dense rerun + verification sweep deliverables.
- plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/plan/plan.md:1 — Detailed execution sketch for the counted dense run and post-verify-only workflow.

Next Up (optional)
1. Once dense evidence is archived, repeat the workflow for the sparse view (dose 1000) so both overlap conditions have fresh Phase C→G runs.
2. Extend `verify_dense_pipeline_artifacts.py` to emit a short JSON verdict summary for CI ingestion (follows after post-verify-only evidence).
