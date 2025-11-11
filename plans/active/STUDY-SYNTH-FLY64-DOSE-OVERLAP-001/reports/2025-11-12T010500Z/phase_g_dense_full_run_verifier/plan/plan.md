# Dense Phase G Evidence Run + Post-Verify Sweep (2025-11-12T113500Z)

## Reality Check
- Commit `24f2a1af` (post-verify-only automation) added artifact-inventory validation but the success banner still prints absolute hub paths (`CLI logs: /home/.../cli`, `Analysis outputs: /home/.../analysis`), violating TYPE-PATH-001.
- The current hub contains only `cli/run_phase_g_dense_stdout.log` and `cli/phase_c_generation.log`, both originating from `/home/ollie/Documents/PtychoPINN2`; there is no `{analysis,verification,metrics}` payload because the dense run aborted after Phase C.
- We must (a) normalize banner path strings before rerunning, (b) execute one counted dense run with `--clobber` to populate `{analysis,cli}`, and (c) immediately exercise `--post-verify-only` so verification-only sweeps are proven on real artifacts with refreshed inventories.

## Objectives (single Ralph loop)
1. **Success-banner compliance** — Update `run_phase_g_dense.py::main` (full pipeline + `--post-verify-only`) so every success-banner artifact reference uses `relative_to(hub)` (CLI logs, analysis outputs, aggregate report, highlights, metrics digests, SSIM grid summary/log, verification report/log, highlights log) and extend `tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_post_verify_only_executes_chain` to assert the relative `cli/` + `analysis/` lines.
2. **Counted dense run (Phase C→G)** — Run `plans/.../bin/run_phase_g_dense.py --hub "$HUB" --dose 1000 --view dense --splits train test --clobber` into this hub so Phase C–G artifacts, SSIM grid summary, verification report, highlights logs, and metrics delta files materialize under `{analysis,cli}`.
3. **Verification-only sweep** — Re-run `run_phase_g_dense.py --hub "$HUB" --post-verify-only` against the freshly produced artifacts, proving the shortened chain regenerates SSIM grid + verification outputs and refreshes `analysis/artifact_inventory.txt`. Capture the CLI logs for both runs.
4. **Publish metrics + ledger updates** — Record MS-SSIM ±0.000 / MAE ±0.000000 deltas, preview verdict, SSIM grid table reference, and CLI/test artifacts inside `summary/summary.md`, then sync docs/fix_plan.md + galph_memory with the evidence.

## Execution Sketch
1. `export AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md` and `export HUB=$PWD/plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier`.
2. Update `plans/active/.../bin/run_phase_g_dense.py`:
   - After the full-run success banner prints `Artifacts saved to`, ensure every subsequent path output uses `.relative_to(hub)` (CLI logs, Analysis outputs, Aggregate report, Highlights, Metrics digest/log, delta JSON/TXT/preview, SSIM grid summary/log, verification report/log, highlights log).
   - Mirror the same relative-path formatting inside the `--post-verify-only` branch for its `CLI logs`, `Analysis outputs`, SSIM grid summary/log, verification report/log, and highlights log.
3. Extend `tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_post_verify_only_executes_chain` to continue asserting `analysis/artifact_inventory.txt` and add new stdout checks for the relative `CLI logs: cli` and `Analysis outputs: analysis` lines (capsys already wired).
4. `pytest --collect-only tests/study/test_phase_g_dense_orchestrator.py -k post_verify_only -vv | tee \"$HUB\"/collect/pytest_collect_orchestrator_post_verify_only.log` (archive failures under `$HUB`/red/ before rerunning).
5. `pytest tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_post_verify_only_executes_chain -vv | tee \"$HUB\"/green/pytest_post_verify_only.log`.
6. `python plans/active/.../bin/run_phase_g_dense.py --hub \"$HUB\" --dose 1000 --view dense --splits train test --clobber |& tee \"$HUB\"/cli/run_phase_g_dense_stdout.log`
   - Monitor `phase_c_generation.log`, `phase_d_dense.log`, `phase_e_{baseline,dense}_*.log`, `phase_f_dense_train.log`, `phase_g_dense_compare.log`, `aggregate_report_cli.log`, `metrics_digest_cli.log`, `ssim_grid_cli.log`, `verify_dense_stdout.log`, and `check_dense_highlights.log` for SUCCESS sentinels.
   - Post-run verify that `analysis/metrics_delta_summary.json`, `analysis/metrics_delta_highlights_preview.txt`, `analysis/ssim_grid_summary.md`, `analysis/verification_report.json`, `analysis/verify_dense_stdout.log`, `analysis/check_dense_highlights.log`, and `analysis/artifact_inventory.txt` exist.
7. `python plans/active/.../bin/run_phase_g_dense.py --hub \"$HUB\" --post-verify-only |& tee \"$HUB\"/cli/run_phase_g_dense_post_verify_only.log`
   - Confirm the success banner reuses hub-relative path strings and announces SSIM grid + verification regeneration along with refreshed `analysis/artifact_inventory.txt`.
8. If the orchestrator reports preview/highlights discrepancies, run `python plans/.../bin/check_dense_highlights_match.py --hub \"$HUB\" | tee \"$HUB\"/analysis/check_dense_highlights_manual.log` for diagnostics before re-running.
9. Update `$HUB/summary/summary.md` with runtime, MS-SSIM/MAE deltas (phase-only), preview verdict, CLI/test/log pointers (collect + exec), rerun commands, and mention that the post-verify-only workflow is proven on real artifacts. Append the same Turn Summary to `summary.md`.
10. Refresh docs/fix_plan.md Attempts History and galph_memory with the execution evidence (metrics, logs, preview verdict, verifier output hashes).

## Acceptance Criteria
- `{analysis,cli}` exist and contain Phase C→G logs plus `verification_report.json`, `verify_dense_stdout.log`, `check_dense_highlights.log`, `ssim_grid_summary.md`, `metrics_delta_summary.json`, `metrics_delta_highlights_preview.txt`, and `artifact_inventory.txt`.
- `run_phase_g_dense.py --post-verify-only` succeeds without touching Phase C data, regenerates SSIM grid + verification artifacts, and updates the artifact inventory + success banner.
- CLI logs for both runs are archived under `$HUB` (`cli/run_phase_g_dense_stdout.log`, `cli/run_phase_g_dense_post_verify_only.log`, plus individual phase logs) with SUCCESS sentinels.
- `$HUB/summary/summary.md` records the run parameters, MS-SSIM/MAE deltas (±0.000 / ±0.000000), preview verdict, SSIM grid table path, verifier/log references, pytest selectors, and doc ledger updates. `docs/fix_plan.md` references the hub and metrics.

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
