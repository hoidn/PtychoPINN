# Dense Phase G Evidence Run + Post-Verify Sweep (2025-11-11T120554Z)

## Reality Check
- `git log -5 --oneline` shows `962bffba` (Galph sync) as the latest commit; no engineer-side change has landed since the workspace-mismatch blocker (`c1f907bc`), so no new CLI/test evidence exists.
- The active hub inspection confirms `analysis/` still contains only `blocker.log`, while `cli/` holds `phase_c_generation.log`, `phase_d_dense.log`, and `run_phase_g_dense_stdout.log`; there are no SSIM grid summaries, verification logs, metrics deltas, preview artifacts, or artifact inventory files from a counted rerun.
- `summary/blocker_2025-11-11T115954Z.md` (untracked) reiterates that the previous attempt ran from `/home/ollie/Documents/PtychoPINN2`; until Ralph works from `/home/ollie/Documents/PtychoPINN` the orchestrator will continue writing to the wrong tree and Phase D–G will never execute.
- Latest sync `32954c41` is another supervisor bookkeeping commit; Ralph hasn’t shipped any dense rerun or verification evidence since the prior blocker logs.
- The hub still contains only `cli/run_phase_g_dense_stdout.log`, `cli/phase_c_generation.log`, `cli/phase_d_dense.log`, and `analysis/blocker.log`. There is **still no populated `analysis/` directory** (no SSIM grid summaries, verification report/log, metrics deltas, preview artifacts, or artifact inventory evidence), so Phase G exit criteria remain unmet.
- `analysis/blocker.log` confirms the last counted attempt died during Phase C generation while running from `/home/ollie/Documents/PtychoPINN2` (`--output-root .../data/phase_c`), meaning Phase D–G never started and the ledger guardrail (TEST-CLI-001 + PREVIEW-PHASE-001) is still unsatisfied.
- Until a dense Phase C→G rerun immediately followed by `--post-verify-only` completes from **this** repo (`/home/ollie/Documents/PtychoPINN`), we cannot demonstrate SSIM grid/verifier/highlights success with hub-relative paths, prove `analysis/artifact_inventory.txt` regeneration, or capture MS-SSIM/MAE + preview evidence.

## Objectives (single Ralph loop)
1. **Regression check for the guard** — Re-run the collect-only + execution selectors for `test_run_phase_g_dense_post_verify_only_executes_chain` so the new assertions stay GREEN before launching expensive CLI work (TEST-CLI-001).
2. **Counted dense run (Phase C→G)** — Run `plans/.../bin/run_phase_g_dense.py --hub "$HUB" --dose 1000 --view dense --splits train test --clobber` so Phase C→G artifacts, SSIM grid summary/log, verification report/log, highlights logs, metrics delta files, and artifact inventory populate `{analysis,cli}` with fresh timestamps.
3. **Verification-only sweep** — Immediately rerun `run_phase_g_dense.py --hub "$HUB" --post-verify-only` against the refreshed artifacts to prove the shortened chain regenerates SSIM grid + verification outputs and rewrites `analysis/artifact_inventory.txt`. Capture CLI logs for both runs.
4. **Publish metrics + ledger updates** — Record MS-SSIM ±0.000 / MAE ±0.000000 deltas, preview verdict (phase-only), SSIM grid table reference, CLI/test selectors, and verification/highlights links inside `summary/summary.md`, summary.md, docs/fix_plan.md, and galph_memory.

## Execution Sketch
1. Confirm you are in `/home/ollie/Documents/PtychoPINN` (`pwd -P`) so `$HUB` resolves inside this repo, then `export AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md` and `export HUB=$PWD/plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier`; `mkdir -p "$HUB"/{analysis,cli,collect,green,red,summary}` before running anything else.
2. `pytest --collect-only tests/study/test_phase_g_dense_orchestrator.py -k post_verify_only_executes_chain -vv | tee "$HUB"/collect/pytest_collect_post_verify_only.log` (move failures to `$HUB`/red/ before rerun).
3. `pytest tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_post_verify_only_executes_chain -vv | tee "$HUB"/green/pytest_post_verify_only.log`.
4. `python plans/active/.../bin/run_phase_g_dense.py --hub "$HUB" --dose 1000 --view dense --splits train test --clobber |& tee "$HUB"/cli/run_phase_g_dense_stdout.log`
   - Monitor `phase_c_generation.log`, `phase_d_dense.log`, `phase_e_{baseline,dense}_*.log`, `phase_f_dense_train.log`, `phase_g_dense_compare.log`, `aggregate_report_cli.log`, `metrics_digest_cli.log`, `ssim_grid_cli.log`, `verify_dense_stdout.log`, and `check_dense_highlights.log` for SUCCESS sentinels.
   - After completion ensure `analysis/metrics_delta_summary.json`, `analysis/metrics_delta_highlights_preview.txt`, `analysis/ssim_grid_summary.md`, `analysis/verification_report.json`, `analysis/verify_dense_stdout.log`, `analysis/check_dense_highlights.log`, and `analysis/artifact_inventory.txt` exist.
5. `python plans/active/.../bin/run_phase_g_dense.py --hub "$HUB" --post-verify-only |& tee "$HUB"/cli/run_phase_g_dense_post_verify_only.log`
   - Confirm the success banner reuses hub-relative path strings and announces SSIM grid + verification regeneration along with refreshed `analysis/artifact_inventory.txt`.
6. If the orchestrator reports preview/highlights discrepancies, run `python plans/.../bin/check_dense_highlights_match.py --hub "$HUB" | tee "$HUB"/analysis/check_dense_highlights_manual.log` before attempting another rerun.
7. Update `$HUB/summary/summary.md` (and copy to `$HUB/summary.md`) with runtimes, MS-SSIM/MAE deltas (phase-only), preview verdict, SSIM grid table path, verification/highlights log references, CLI/test selectors, artifact inventory + metrics digest references, and doc ledger updates.
8. Refresh docs/fix_plan.md Attempts History and galph_memory with the execution evidence (metrics, logs, preview verdict, verifier output hashes).

## Acceptance Criteria
- `tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_post_verify_only_executes_chain` fails if stdout drops the SSIM grid lines, verification report/log lines, highlights check line, or omits the artifact inventory references for the post-verify-only path; collect + execution logs archived under `$HUB`.
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
