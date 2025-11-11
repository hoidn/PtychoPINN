# Dense Phase G Evidence Run + Post-Verify Sweep (2025-11-12T133500Z)

## Reality Check
- Commit `7dcb2297` normalized every success-banner artifact reference in `run_phase_g_dense.py`, and the updated `test_run_phase_g_dense_post_verify_only_executes_chain` now asserts the hub-relative `CLI logs: cli` and `Analysis outputs: analysis` strings for the `--post-verify-only` workflow.
- The active hub still contains only `cli/run_phase_g_dense_stdout.log` and `cli/phase_c_generation.log`; `{analysis,verification,metrics}` remain empty because no dense Phase C→G rerun executed after the guard merged, so MS-SSIM/MAE/preview verdicts are still unverified.
- The full execution test (`test_run_phase_g_dense_exec_prints_highlights_preview`) does not assert the new banner content, so the counted run path could regress before we capture real evidence.
- The success banner prints duplicate “Metrics digest” lines, which is confusing when summarizing CLI artifacts; we need a single Markdown reference plus the CLI log path.

## Objectives (single Ralph loop)
1. **Full-run guardrails** — Extend `tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_exec_prints_highlights_preview` to assert the counted execution banner prints `CLI logs: cli`, `Analysis outputs: analysis`, and `analysis/artifact_inventory.txt`. While touching `run_phase_g_dense.py::main`, drop the duplicate “Metrics digest” line so only one Markdown path and one CLI log line remain (TYPE-PATH-001, TEST-CLI-001).
2. **Counted dense run (Phase C→G)** — Run `plans/.../bin/run_phase_g_dense.py --hub "$HUB" --dose 1000 --view dense --splits train test --clobber` into this hub so Phase C→G artifacts, SSIM grid summary/log, verification report/log, highlights logs, metrics delta files, and artifact inventory populate `{analysis,cli}`.
3. **Verification-only sweep** — Immediately rerun `run_phase_g_dense.py --hub "$HUB" --post-verify-only` against the fresh artifacts, proving the shortened chain regenerates SSIM grid + verification outputs and refreshes `analysis/artifact_inventory.txt`. Capture CLI logs for both runs.
4. **Publish metrics + ledger updates** — Record MS-SSIM ±0.000 / MAE ±0.000000 deltas, preview verdict (phase-only), SSIM grid table reference, and CLI/test artifacts inside `summary/summary.md` and docs/fix_plan.md, then mirror the same context in galph_memory.

## Execution Sketch
1. `export AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md` and `export HUB=$PWD/plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier`.
2. Update `plans/active/.../bin/run_phase_g_dense.py::main` to print `Metrics digest (Markdown)` once and keep `Metrics digest log: {analyze_digest_log.relative_to(hub)}` as the CLI reference (remove the duplicate Markdown line).
3. Extend `tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_exec_prints_highlights_preview` (capsys already wired) so stdout assertions cover `CLI logs: cli`, `Analysis outputs: analysis`, and `analysis/artifact_inventory.txt` in addition to the highlights preview text.
4. `pytest --collect-only tests/study/test_phase_g_dense_orchestrator.py -k exec_prints_highlights_preview -vv | tee "$HUB"/collect/pytest_collect_exec_highlights.log` (move failures to `$HUB`/red/ before rerun).
5. `pytest tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_exec_prints_highlights_preview -vv | tee "$HUB"/green/pytest_exec_highlights.log`.
6. `python plans/active/.../bin/run_phase_g_dense.py --hub "$HUB" --dose 1000 --view dense --splits train test --clobber |& tee "$HUB"/cli/run_phase_g_dense_stdout.log`
   - Monitor `phase_c_generation.log`, `phase_d_dense.log`, `phase_e_{baseline,dense}_*.log`, `phase_f_dense_train.log`, `phase_g_dense_compare.log`, `aggregate_report_cli.log`, `metrics_digest_cli.log`, `ssim_grid_cli.log`, `verify_dense_stdout.log`, and `check_dense_highlights.log` for SUCCESS sentinels.
   - After completion ensure `analysis/metrics_delta_summary.json`, `analysis/metrics_delta_highlights_preview.txt`, `analysis/ssim_grid_summary.md`, `analysis/verification_report.json`, `analysis/verify_dense_stdout.log`, `analysis/check_dense_highlights.log`, and `analysis/artifact_inventory.txt` exist.
7. `python plans/active/.../bin/run_phase_g_dense.py --hub "$HUB" --post-verify-only |& tee "$HUB"/cli/run_phase_g_dense_post_verify_only.log`
   - Confirm the success banner reuses hub-relative path strings and announces SSIM grid + verification regeneration along with refreshed `analysis/artifact_inventory.txt`.
8. If the orchestrator reports preview/highlights discrepancies, run `python plans/.../bin/check_dense_highlights_match.py --hub "$HUB" | tee "$HUB"/analysis/check_dense_highlights_manual.log` before attempting another rerun.
9. Update `$HUB/summary/summary.md` (and copy to `$HUB/summary.md`) with runtimes, MS-SSIM/MAE deltas (phase-only), preview verdict, SSIM grid table path, verification/highlights log references, CLI/test selectors, and doc ledger updates.
10. Refresh docs/fix_plan.md Attempts History and galph_memory with the execution evidence (metrics, logs, preview verdict, verifier output hashes).

## Acceptance Criteria
- `tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_exec_prints_highlights_preview` asserts the hub-relative banner strings and passes locally (collect + execution logs archived under `$HUB`).
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
