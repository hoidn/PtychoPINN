# Dense Phase G Post-Verify Automation Follow-up (2025-11-12T010500Z)

## Reality Check
- Commit `74a97db5` landed the default-on post-verify automation plus pytest coverage for the happy-path hooks (`collect_only` + monkeypatched execution). Evidence lives under `.../green/{pytest_collect_only.log,pytest_post_verify_hooks.log}`.
- The 2025-11-12 hub still lacks `analysis/` and `cli/` payloads because no dense Phase C→G rerun has completed since the automation shipped. We therefore have zero counted MS-SSIM/MAE deltas, no preview verdict, and no verification report in this hub.
- Manual verifier invocations remain necessary whenever we touch `verify_dense_pipeline_artifacts.py` or `check_dense_highlights_match.py`. Re-running two standalone scripts by hand is slow/error-prone, so the orchestrator needs a light-weight “post-verify-only” mode to revalidate existing hubs without burning hours on Phase C→F.

## Objectives (single Ralph loop)
1. **Add a `--post-verify-only` mode** — Teach `plans/.../bin/run_phase_g_dense.py` to skip the Phase C→F commands and execute only the SSIM grid helper + verifier + highlights checker (still default-on) when this flag is set. The mode should refuse `--skip-post-verify`, reuse the same log/report paths, and refresh `artifact_inventory.txt` so verification artifacts are listed even when we bypass heavy phases.
2. **TDD for the new mode** — Extend `tests/study/test_phase_g_dense_orchestrator.py` with:
   - `test_run_phase_g_dense_collect_only_post_verify_only` that asserts `--collect-only --post-verify-only` emits the trimmed command list (ssim_grid → verify → highlights) and logs.
   - `test_run_phase_g_dense_post_verify_only_executes_chain` that monkeypatches `run_command` to capture the command order when the flag is enabled, proving it skips Phase C→F but still re-generates the artifact inventory.
3. **Counted dense run + verification evidence** — With the new mode landed, export `AUTHORITATIVE_CMDS_DOC` and execute `run_phase_g_dense.py --hub "$HUB" --dose 1000 --view dense --splits train test --clobber` so Phase C→G outputs, SSIM grid summary, verification report, and highlights logs populate this hub. Afterwards, re-run `run_phase_g_dense.py --hub "$HUB" --post-verify-only` to exercise the new mode on the freshly produced artifacts, then update `summary/summary.md` + docs/fix_plan.md with MS-SSIM/MAE deltas, preview verdict, and CLI/log/test pointers.

## Execution Sketch
1. `export AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md`; `export HUB=$PWD/plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier`.
2. Update `plans/active/.../bin/run_phase_g_dense.py`:
   - Add `--post-verify-only` argparse flag (default False). In this mode skip `prepare_hub` + Phase C→F command execution, but still announce hub info, regenerate SSIM grid, run the post-verify chain, and call `generate_artifact_inventory`.
   - Disallow `--post-verify-only` together with `--skip-post-verify` or `--collect-only` mismatches, and ensure `--collect-only --post-verify-only` prints only the SSIM grid + verifier + highlights entries (with hub-relative logs).
   - When executing the new mode, write a short banner noting the reuse of existing phase outputs so reviewers can tell it was a verification-only sweep.
3. Tests (`tests/study/test_phase_g_dense_orchestrator.py`):
   - Extend the collect-only test suite to cover the new flag and assert the command count shrinks to three entries with the expected log/report names.
   - Add `test_run_phase_g_dense_post_verify_only_executes_chain(monkeypatch, tmp_path)` that seeds fake hub paths, records the command invocations, and asserts the order `[ssim_grid, verify_dense_pipeline_artifacts, check_dense_highlights]` plus an artifact-inventory regeneration call.
4. Evidence commands (archive logs under `$HUB`):
   - `pytest --collect-only tests/study/test_phase_g_dense_orchestrator.py -k post_verify_only -vv | tee "$HUB"/collect/pytest_collect_orchestrator_post_verify_only.log`
   - `pytest tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_post_verify_only_executes_chain -vv | tee "$HUB"/green/pytest_post_verify_only.log`
   - `python plans/.../bin/run_phase_g_dense.py --hub "$HUB" --dose 1000 --view dense --splits train test --clobber |& tee "$HUB"/cli/run_phase_g_dense_stdout.log`
   - `python plans/.../bin/run_phase_g_dense.py --hub "$HUB" --post-verify-only |& tee "$HUB"/cli/run_phase_g_dense_post_verify_only.log`
   - `python plans/.../bin/check_dense_highlights_match.py --hub "$HUB" | tee "$HUB"/analysis/check_dense_highlights_manual.log` (only if the automated pass surfaces issues; otherwise rely on orchestrator logs)
5. Update `$HUB/summary/summary.md` with run duration, MS-SSIM/MAE deltas (±0.000 / ±0.000000), preview verdict, verifier/highlights log locations, new pytest selectors, and the rerun command lines. Sync docs/fix_plan.md + galph_memory.

## Acceptance Criteria
- `run_phase_g_dense.py --collect-only --post-verify-only` emits exactly three commands (SSIM grid, verifier, highlights checker) with hub-relative log/report paths and an explicit note that Phase C→F are skipped.
- Executing `run_phase_g_dense.py --post-verify-only` skips hub preparation, does **not** touch Phase C outputs, but regenerates SSIM grid + verification artifacts and refreshes `analysis/artifact_inventory.txt`.
- New pytest coverage proves both collect-only output and execution order for the post-verify-only flag (RED/green logs archived under `{collect,green}/` with selectors recorded in summary.md).
- The counted dense run fills `{analysis,cli}` with Phase C→G payloads plus `verification_report.json`, `verify_dense_stdout.log`, `check_dense_highlights.log`, `metrics_delta_summary.json`, `metrics_delta_highlights_preview.txt`, `ssim_grid_summary.md`, and `artifact_inventory.txt`.
- `$HUB/summary/summary.md` logs MS-SSIM/MAE deltas, highlights preview verdict, CLI/log/test references, and documents the new `--post-verify-only` workflow; docs/fix_plan.md references this run.

## Evidence & Artifacts
- Hub: `plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/`
- Subdirs to populate: `cli/` (phase logs + run_phase_g_dense_stdout.log + post_verify_only log), `analysis/` (metrics/preview summaries, verification_report.json, verify_dense/check_highlights logs, artifact_inventory.txt), `collect/`, `red/`, `green/`, `plan/`, `summary/`.

## Findings Applied
- **POLICY-001** — Keep PyTorch available for Phase F recon + verifiers; export `AUTHORITATIVE_CMDS_DOC` before every subprocess.
- **CONFIG-001** — Preserve legacy bridge ordering in generation/training; new flag must not reorder env setup.
- **DATA-001** — Verification-only runs still enforce NPZ/JSON layout without regenerating data.
- **TYPE-PATH-001** — All printed/logged paths stay hub-relative; artifact inventory refreshed after post-verify sweeps.
- **STUDY-001** — Report MS-SSIM (±0.000) and MAE (±0.000000) deltas in summary.
- **TEST-CLI-001** — Record RED/GREEN/collect logs for the new flag selectors; ensure CLI filenames match contract.
- **PREVIEW-PHASE-001** — Post-verify-only mode must still enforce phase-only previews via SSIM grid + highlights checker.
- **PHASEC-METADATA-001** — Verification-only runs do **not** mutate Phase C data; guard remains intact.

## Risks & Mitigations
- **Flag misuse:** Disallow `--post-verify-only` with `--clobber` or `--skip-post-verify` to prevent accidental data deletion or missing guards.
- **Evidence gaps:** If the dense run fails mid-phase, retain blocker log, document the failure, and rerun with `--clobber` before attempting post-verify-only.
- **Time pressure:** Dense Phase C→G still takes hours; capture incremental CLI logs so failures are diagnosable.
- **Doc drift:** Update docs/fix_plan.md + summary.md immediately after the run to keep ledger + artifacts aligned with the new workflow.
