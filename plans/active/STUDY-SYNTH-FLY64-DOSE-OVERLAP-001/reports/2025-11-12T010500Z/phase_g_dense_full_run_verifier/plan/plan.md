# Dense Phase G Full Run + Auto Verification (2025-11-12T010500Z)

## Reality Check
- `plans/.../bin/check_dense_highlights_match.py` plus tests/study/test_check_dense_highlights_match.py already landed (commit 3496351) and now parse `analysis/ssim_grid_summary.md`, so the prior Do Now’s “extend the checker” task is satisfied.
- `verify_dense_pipeline_artifacts.py` and its pytest module were hardened yesterday to require `ssim_grid_cli.log` + `analysis/ssim_grid_summary.md`, with docs/TESTING_GUIDE.md and TEST_SUITE_INDEX.md already reflecting the new helper.
- Despite the guard work, the active hub `reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/` still lacks `{analysis,cli}` payloads—no counted Phase C→G rerun has been captured since the helper/checker changes.
- Manual verifier + highlights invocations live outside `run_phase_g_dense.py`, which means `--collect-only` output and CLI logs do not yet show the post-run verification chain; every dense rerun has to remember a long checklist manually, which keeps regressing.

## Objectives (single Ralph loop)
1. **Post-verify automation** — Extend `plans/.../bin/run_phase_g_dense.py` so successful runs automatically invoke `verify_dense_pipeline_artifacts.py` and `check_dense_highlights_match.py`, teeing logs to `analysis/verify_dense_stdout.log` and `analysis/check_dense_highlights.log`, emitting the JSON report, and surfacing the locations in the success banner. Add a `--skip-post-verify` flag for debugging but default to running the verifiers.
2. **Orchestrator TDD** — Update `tests/study/test_phase_g_dense_orchestrator.py` to prove the new automation: (a) `--collect-only` output lists the verifier/highlights commands in order, and (b) a new unit test monkeypatches `run_command` to assert the post-verify commands fire with hub-relative paths and log destinations.
3. **Counted dense run + evidence** — Export `AUTHORITATIVE_CMDS_DOC` and rerun `run_phase_g_dense.py --hub "$HUB" --dose 1000 --view dense --splits train test --clobber` (post-verify on) into this hub. Publish the resulting CLI/analysis artifacts, run `pytest --collect-only/-k` evidence for the orchestrator tests, and refresh `summary/summary.md` with MS-SSIM/MAE deltas, preview verdict, and verifier/highlights log paths. Update docs/fix_plan.md + galph_memory.

## Execution Sketch
1. `export AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md`; `export HUB=$PWD/plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier`.
2. Update `run_phase_g_dense.py`:
   - Introduce `--skip-post-verify` (default False) and guard for `--collect-only` so the planned command list now shows verifier and checker invocations.
   - After the existing SSIM grid command succeeds, call `verify_dense_pipeline_artifacts.py --hub ... --report analysis/verification_report.json --dose 1000 --view dense` and `check_dense_highlights_match.py --hub ...`, both routed through `run_command` so logs land at `analysis/verify_dense_stdout.log` and `analysis/check_dense_highlights.log`.
   - Add hub-relative success banner lines for the new logs + JSON report and ensure `generate_artifact_inventory()` runs after the post-verify commands so the new artifacts are listed.
3. Extend `tests/study/test_phase_g_dense_orchestrator.py`:
   - Update `test_run_phase_g_dense_collect_only_generates_commands` expectations to include the two new commands/log names.
   - Add `test_run_phase_g_dense_post_verify_hooks(monkeypatch,tmp_path)` that injects a sentinel `run_command` to capture command tuples, runs `main()` with `--splits train test --post-verify` into a tmp hub, and asserts the verifier/checker invocations appear after SSIM grid with the intended CLI/log/report paths.
4. `pytest --collect-only tests/study/test_phase_g_dense_orchestrator.py -k post_verify -vv | tee "$HUB"/collect/pytest_collect_orchestrator_post_verify.log`.
5. `pytest tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_post_verify_hooks -vv | tee "$HUB"/green/pytest_phase_g_dense_post_verify.log` (capture RED if it fails before fix).
6. Execute the dense pipeline: `python plans/.../bin/run_phase_g_dense.py --hub "$HUB" --dose 1000 --view dense --splits train test --clobber |& tee "$HUB"/cli/run_phase_g_dense_stdout.log`. Post-verify automation should emit the JSON report + logs under `analysis/`.
7. If either post-verify command fails, keep the blocker evidence, fix the issue, rerun with `--clobber`, and archive red/green logs accordingly. After a GREEN run, snapshot MS-SSIM/MAE deltas from `metrics_delta_summary.json` and the preview verdict from `analysis/check_dense_highlights.log`.
8. Update `$HUB/summary/summary.md` with run statistics, MS-SSIM/MAE deltas, preview guard status, CLI/verifier log pointers, pytest selectors, and doc/test references; sync docs/fix_plan.md + galph_memory.

## Acceptance Criteria
- `run_phase_g_dense.py --collect-only` now prints verifier/highlights commands (with log/report paths) after the SSIM grid line.
- Default runs produce `analysis/verification_report.json`, `analysis/verify_dense_stdout.log`, and `analysis/check_dense_highlights.log` without manual intervention; `--skip-post-verify` disables them for debugging.
- New pytest coverage exercises the post-verify automation (collect-only expectation + monkeypatched execution) with RED/GREEN evidence in `{collect,red,green}/`.
- The counted dense run populates `{analysis,cli}` with real Phase C→G outputs, SSIM grid summary, verification report, highlights logs, and `artifact_inventory.txt` listing the new files.
- `$HUB/summary/summary.md` reports MS-SSIM ±0.000 / MAE ±0.000000 deltas, preview guard outcome, CLI/log references, and pytest selectors; docs/fix_plan.md references this hub.

## Evidence & Artifacts
- Hub: `plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/`
- Expected subdirs: `cli/` (phase logs + run_phase_g_dense_stdout.log), `analysis/` (metrics summaries, verification_report.json, ssim_grid_summary.md, check_dense_highlights.log, verify_dense_stdout.log, artifact_inventory.txt), `collect/`, `red/`, `green/`, `plan/`, `summary/`.

## Findings Applied
- **POLICY-001** — PyTorch dependency must remain available for Phase E/F helpers invoked inside the orchestrator.
- **CONFIG-001** — Preserve `update_legacy_dict` ordering inside generation/training modules by reusing the AUTHORITATIVE_CMDS_DOC export before each subprocess.
- **DATA-001** — Verifier enforces the canonical NPZ/JSON layout; post-verify automation must not skip it.
- **TYPE-PATH-001** — All success banners, CLI logs, and inventory entries stay hub-relative.
- **STUDY-001** — Report MS-SSIM (±0.000) and MAE (±0.000000) deltas per model/view in the summary.
- **TEST-CLI-001** — Post-verify commands capture RED/GREEN logs under the hub to prove CLI coverage.
- **PREVIEW-PHASE-001** — Highlights checker + SSIM grid summary continue to enforce phase-only previews.
- **PHASEC-METADATA-001** — Phase C guard remains active; reruns must not bypass metadata validation.

## Risks & Mitigations
- **Runtime pressure:** Dense Phase C→G runs can take hours; capture intermediate logs and plan for retries with `--clobber`.
- **Post-verify flakiness:** If automation masks verifier failures, keep blocker logs and rerun manually with `--skip-post-verify` for debugging.
- **Storage churn:** A failed run leaves partially populated `data/` trees; ensure `prepare_hub(..., clobber=True)` runs before retrying.
- **Docs drift:** Update ledger + summary immediately after the run so the automation change and counted evidence stay in sync.
