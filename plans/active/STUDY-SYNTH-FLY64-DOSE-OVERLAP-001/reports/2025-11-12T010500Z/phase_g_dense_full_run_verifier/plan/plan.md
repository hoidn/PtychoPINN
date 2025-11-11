# Dense Phase G Full Run + Verifier Hardening (2025-11-12T010500Z)

## Reality Check
- `run_phase_g_dense.py` already invokes `ssim_grid.py` (commit 979cd2b3) and tests under `reports/2025-11-11T235500Z/.../green/` prove the helper wiring, but that hub still has empty `analysis/` and `cli/` directories—no counted Phase C→G execution exists post-preview guard.
- `plans/active/.../bin/verify_dense_pipeline_artifacts.py` never checks for `analysis/ssim_grid_summary.md` or `cli/ssim_grid_cli.log`, so PREVIEW-PHASE-001 regressions can slip in even if the orchestrator succeeds.
- `tests/study/test_phase_g_dense_artifacts_verifier.py` lacks coverage for the new helper/log, and docs (`docs/TESTING_GUIDE.md` §Phase G Delta Metrics Persistence + `docs/development/TEST_SUITE_INDEX.md` Phase G row) still describe the pre-helper workflow.
- Guardrails in `docs/fix_plan.md:17-33` still claim the helper isn't integrated; need to update ledger + galph memory to mirror reality and point Ralph at a real run.

## Objectives (single Ralph loop)
1. **Verifier & tests** — Extend `verify_dense_pipeline_artifacts.py` so `validate_cli_logs()` and the artifact inventory check require `cli/ssim_grid_cli.log` + `analysis/ssim_grid_summary.md` and surface preview metadata in the JSON report. Add RED/GREEN pytest coverage in `tests/study/test_phase_g_dense_artifacts_verifier.py` to lock the new guard (missing summary/log vs happy path).
2. **Counted dense run** — Export `AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md`, point `HUB` at `plans/active/.../reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/`, and execute `python plans/.../bin/run_phase_g_dense.py --hub "$HUB" --dose 1000 --view dense --splits train test --clobber`. Archive all stdout under `cli/` and ensure `analysis/ssim_grid_summary.md` + preview files exist.
3. **Verification & digests** — Run `verify_dense_pipeline_artifacts.py` and `check_dense_highlights_match.py`, keeping JSON/markdown outputs in `analysis/`. Rerun `ssim_grid.py --hub "$HUB"` manually if the orchestrator stops early.
4. **Doc/test registry sync** — Update `docs/TESTING_GUIDE.md` Phase G section + `docs/development/TEST_SUITE_INDEX.md` (Phase G row) to describe the helper, preview-only guard, and the new pytest selectors. Capture diffs + MS-SSIM/MAE table + verifier verdict in `summary/summary.md` alongside pointers to `cli/` logs.

## Execution Sketch
1. `export AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md` and `export HUB=$PWD/plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier`.
2. Modify `plans/active/.../bin/verify_dense_pipeline_artifacts.py`:
   - Teach `validate_cli_logs()` to demand `ssim_grid_cli.log` (helper log list) and record it in metadata.
   - Add a new `validate_ssim_grid_summary()` that checks `analysis/ssim_grid_summary.md` exists, is non-empty, and that preview guard metadata came from the verifier. Wire it into `main()` validations list right after the delta highlight checks.
3. Grow `tests/study/test_phase_g_dense_artifacts_verifier.py` with:
   - RED test: missing `ssim_grid_cli.log` or summary triggers verifier failure with actionable error fields.
   - GREEN test: complete hub (include helper log + markdown) passes and report JSON notes the helper artifacts.
   Capture logs under `$HUB/collect` / `$HUB/red` / `$HUB/green`.
4. `pytest tests/study/test_phase_g_dense_artifacts_verifier.py::test_verify_dense_pipeline_cli_logs_require_ssim_grid_log -vv` (expect RED→GREEN cycle) and `pytest tests/study/test_phase_g_dense_artifacts_verifier.py::test_verify_dense_pipeline_requires_ssim_grid_summary -vv`.
5. `pytest --collect-only tests/study/test_phase_g_dense_artifacts_verifier.py -k ssim_grid -vv | tee "$HUB"/collect/pytest_collect_verifier.log` (Mapped Tests Guardrail).
6. Run the dense pipeline command (step 2) with `--clobber`; tee stdout to `cli/run_phase_g_dense_stdout.log`. Ensure orchestrator prints success banner referencing ssim_grid summary/log.
7. `python plans/.../bin/verify_dense_pipeline_artifacts.py --hub "$HUB" --report "$HUB"/analysis/verification_report.json --dose 1000 --view dense | tee "$HUB"/analysis/verify_dense_stdout.log`.
8. `python plans/.../bin/check_dense_highlights_match.py --hub "$HUB" | tee "$HUB"/analysis/check_dense_highlights.log`.
9. If needed, `python plans/.../bin/ssim_grid.py --hub "$HUB" --output "$HUB"/analysis/ssim_grid_summary.md` to regenerate summary; document whether rerun was necessary.
10. Update docs/TESTING_GUIDE.md + docs/development/TEST_SUITE_INDEX.md to reflect helper + preview guard + precision rules. Save diffs under `summary/`.
11. Write `$HUB/summary/summary.md` capturing MS-SSIM/MAE deltas, verifier outcome, preview guard status, doc/test updates, and CLI log links (TYPE-PATH-001 relative paths).

## Acceptance Criteria
- New verifier logic fails fast if `ssim_grid_cli.log` or `analysis/ssim_grid_summary.md` missing; report JSON exposes `missing_helper_logs` / `missing_summary` metadata.
- Added pytest RED/GREEN logs recorded under `{red,green}/` for both new selectors and collect-only evidence stored under `collect/`.
- Dense `run_phase_g_dense.py --clobber` populates `$HUB/analysis` with the complete artifact bundle (metrics summary, deltas, preview, digest, ssim_grid summary) and `$HUB/cli` with orchestrator + helper logs.
- `verify_dense_pipeline_artifacts.py` + `check_dense_highlights_match.py` succeed, with report JSON + stdout archived.
- docs/TESTING_GUIDE.md + docs/development/TEST_SUITE_INDEX.md mention ssim_grid helper, preview-only guard, MAE ±0.000000 precision, and mapped pytest selectors.
- docs/fix_plan.md + galph_memory updated with this loop’s attempt + artifact hub.

## Evidence & Artifacts
- Hub: `plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/`
- Expect populated subdirs: `plan/`, `summary/`, `analysis/` (verification_report.json, metrics*, ssim_grid_summary.md, doc diffs), `cli/` (orchestrator + helper logs), `collect/`, `red/`, `green/`.

## Findings Applied
- POLICY-001 — PyTorch dependency must be available for Phase E/F steps invoked by the orchestrator.
- CONFIG-001 — Export `AUTHORITATIVE_CMDS_DOC` before running orchestration helpers to keep params.cfg bridge intact.
- DATA-001 — `verify_dense_pipeline_artifacts.py` enforces canonical NPZ/JSON layout before declaring success.
- TYPE-PATH-001 — Keep success banners, doc references, and artifact inventory entries relative to the hub.
- STUDY-001 — Report MS-SSIM/MAE deltas with ± precision and phase emphasis across helper + summary.
- TEST-CLI-001 — Preserve full CLI logs + pytest evidence in hub to prove orchestrator/test coverage.
- PREVIEW-PHASE-001 — Fail if preview/highlights/summary surface amplitude terms; ssim_grid helper + new verifier checks guard this.

## Risks & Mitigations
- **Runtime duration:** Dense pipeline can take hours; capture incremental logs + partial outputs if interrupted, then resume with `--clobber` after addressing blockers.
- **Storage pressure:** Phase G artifacts are large; ensure `$HUB` lives under repo (tracked in .gitignore) and avoid scattering duplicates.
- **Verifier brittleness:** TDD the new guard so helper log/summary expectations don’t regress when file names change.
- **Doc drift:** Update TESTING_GUIDE + TEST_SUITE_INDEX in the same loop to avoid another docs-only turn.
