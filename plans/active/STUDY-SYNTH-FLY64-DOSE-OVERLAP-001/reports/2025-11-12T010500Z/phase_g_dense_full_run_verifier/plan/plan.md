# Dense Phase G Evidence Run + Post-Verify Sweep (2025-11-12T153500Z)

## Reality Check
- Commit `a65bda9c` already extended `test_run_phase_g_dense_exec_prints_highlights_preview` with the hub-relative assertions and removed the duplicate “Metrics digest” line inside `run_phase_g_dense.py::main`, so the counted path now matches TYPE-PATH-001/TDD expectations.
- `test_run_phase_g_dense_exec_runs_analyze_digest` still only checks that the “Metrics digest” and “Metrics digest log” strings appear somewhere in stdout; it does **not** fail if the Markdown line is emitted twice, so a future regression could silently reintroduce duplicates.
- The active hub still only has `cli/run_phase_g_dense_stdout.log` and `cli/phase_c_generation.log`; `{analysis,verification,metrics}` remain empty because no dense Phase C→G rerun executed after the success-banner guard landed, so MS-SSIM/MAE/preview verdicts are still unverified.
- Until we rerun the counted pipeline plus `--post-verify-only`, we lack SSIM grid summaries, verification/highlights logs, metrics deltas, and artifact inventory proof (PREVIEW-PHASE-001, TEST-CLI-001).

## Objectives (single Ralph loop)
1. **Banner regression guard** — Update `tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_exec_runs_analyze_digest` so it asserts **exactly one** `Metrics digest:` line is emitted (e.g., via `stdout.count("Metrics digest: ") == 1`) while still checking for `Metrics digest log:`. This locks the deduplicated banner behavior before we rerun the pipeline (TYPE-PATH-001, TEST-CLI-001).
2. **Counted dense run (Phase C→G)** — Run `plans/.../bin/run_phase_g_dense.py --hub "$HUB" --dose 1000 --view dense --splits train test --clobber` so Phase C→G artifacts, SSIM grid summary/log, verification report/log, highlights logs, metrics delta files, and artifact inventory populate `{analysis,cli}` with fresh timestamps.
3. **Verification-only sweep** — Immediately rerun `run_phase_g_dense.py --hub "$HUB" --post-verify-only` against the refreshed artifacts to prove the shortened chain regenerates SSIM grid + verification outputs and rewrites `analysis/artifact_inventory.txt`. Capture CLI logs for both runs.
4. **Publish metrics + ledger updates** — Record MS-SSIM ±0.000 / MAE ±0.000000 deltas, preview verdict (phase-only), SSIM grid table reference, CLI/test selectors, and verification/highlights links inside `summary/summary.md`, summary.md, docs/fix_plan.md, and galph_memory.

## Execution Sketch
1. `export AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md` and `export HUB=$PWD/plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier`.
2. Extend `tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_exec_runs_analyze_digest` so it asserts `stdout.count("Metrics digest: ") == 1` (or equivalent) and still checks for `Metrics digest log:` plus the MS-SSIM/MAE delta block.
3. `pytest --collect-only tests/study/test_phase_g_dense_orchestrator.py -k exec_runs_analyze_digest -vv | tee "$HUB"/collect/pytest_collect_exec_digest.log` (move failures to `$HUB`/red/ before rerun).
4. `pytest tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_exec_runs_analyze_digest -vv | tee "$HUB"/green/pytest_exec_digest.log`.
5. `python plans/active/.../bin/run_phase_g_dense.py --hub "$HUB" --dose 1000 --view dense --splits train test --clobber |& tee "$HUB"/cli/run_phase_g_dense_stdout.log`
   - Monitor `phase_c_generation.log`, `phase_d_dense.log`, `phase_e_{baseline,dense}_*.log`, `phase_f_dense_train.log`, `phase_g_dense_compare.log`, `aggregate_report_cli.log`, `metrics_digest_cli.log`, `ssim_grid_cli.log`, `verify_dense_stdout.log`, and `check_dense_highlights.log` for SUCCESS sentinels.
   - After completion ensure `analysis/metrics_delta_summary.json`, `analysis/metrics_delta_highlights_preview.txt`, `analysis/ssim_grid_summary.md`, `analysis/verification_report.json`, `analysis/verify_dense_stdout.log`, `analysis/check_dense_highlights.log`, and `analysis/artifact_inventory.txt` exist.
7. `python plans/active/.../bin/run_phase_g_dense.py --hub "$HUB" --post-verify-only |& tee "$HUB"/cli/run_phase_g_dense_post_verify_only.log`
   - Confirm the success banner reuses hub-relative path strings and announces SSIM grid + verification regeneration along with refreshed `analysis/artifact_inventory.txt`.
8. If the orchestrator reports preview/highlights discrepancies, run `python plans/.../bin/check_dense_highlights_match.py --hub "$HUB" | tee "$HUB"/analysis/check_dense_highlights_manual.log` before attempting another rerun.
9. Update `$HUB/summary/summary.md` (and copy to `$HUB/summary.md`) with runtimes, MS-SSIM/MAE deltas (phase-only), preview verdict, SSIM grid table path, verification/highlights log references, CLI/test selectors, artifact inventory + metrics digest references, and doc ledger updates.
10. Refresh docs/fix_plan.md Attempts History and galph_memory with the execution evidence (metrics, logs, preview verdict, verifier output hashes).

## Acceptance Criteria
- `tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_exec_runs_analyze_digest` fails if stdout contains more than one `Metrics digest:` line and passes locally (collect + execution logs archived under `$HUB`).
- `{analysis,cli}` exist and contain Phase C→G logs plus `verification_report.json`, `verify_dense_stdout.log`, `check_dense_highlights.log`, `ssim_grid_summary.md`, `metrics_delta_summary.json`, `metrics_delta_highlights_preview.txt`, and `artifact_inventory.txt`.
- `run_phase_g_dense.py --post-verify-only` succeeds without touching Phase C data, regenerates SSIM grid + verification artifacts, and updates the artifact inventory + success banner.
- CLI logs for both runs sit under `$HUB` (`cli/run_phase_g_dense_stdout.log`, `cli/run_phase_g_dense_post_verify_only.log`, phase-specific logs) with SUCCESS sentinels.
- `$HUB/summary/summary.md` captures run parameters, MS-SSIM/MAE deltas (±0.000 / ±0.000000), preview verdict, SSIM grid path, verifier/highlights logs, pytest selectors, and doc ledger updates; docs/fix_plan.md references the same evidence.

## Evidence & Artifacts
- Hub: `plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/`
- Required subdirs: `cli/`, `analysis/`, `collect/`, `green/`, `red/`, `plan/`, `summary/`.

## Findings Applied
- **POLICY-001** — Torch must remain available for Phase F + verifier steps.
- **CONFIG-001** — Keep legacy bridge order intact (Phase C generation still calls `update_legacy_dict`).
- **DATA-001** — SSIM grid + verifier enforce NPZ/JSON schemas; failures block completion.
- **TYPE-PATH-001** — Continue emitting hub-relative paths in success banners + logs.
- **STUDY-001** — Report MS-SSIM + MAE deltas with ± precision.
- **TEST-CLI-001** — Archive collect + exec pytest logs and ensure CLI filenames retain dose/view suffixes and SUCCESS sentinels.
- **PREVIEW-PHASE-001** — Reject previews that include amplitude or missing ± tokens; rely on `check_dense_highlights_match.py`.
- **PHASEC-METADATA-001** — Phase C metadata guard remains untouched by verification-only sweeps.

## Risks & Mitigations
- **Runtime (hours)** — Stream CLI output to `$HUB`/cli and keep `tail -f` running to catch early failures.
- **Partial reruns** — If any phase fails, stop immediately, archive the failing log under `red/`, and note blocker in docs/fix_plan.md.
- **Preview drift** — If preview text mismatches, run the checker manually with `--debug` and keep the log before re-running.
- **Doc drift** — Update summary + ledger before ending the loop to avoid stale exit criteria.
