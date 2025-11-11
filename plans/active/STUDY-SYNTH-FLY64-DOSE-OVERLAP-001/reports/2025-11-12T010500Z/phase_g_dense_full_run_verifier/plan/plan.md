# Dense Phase G Full Run + Verifier Hardening (2025-11-12T010500Z)

## Reality Check
- `run_phase_g_dense.py` already invokes `ssim_grid.py` (commit 979cd2b3) and the helper + verifier/tests/docs landed on feature/torchapi-newprompt last loop. No further wiring is needed before a real run.
- Evidence gap: the 2025-11-11T235500Z and 2025-11-12T010500Z hubs still have empty `{analysis,cli}` folders. We still do not have a counted dense Phase C→G execution after the preview guard + helper shipped.
- Checker gap: `plans/.../bin/check_dense_highlights_match.py` only cross-checks JSON ↔ highlights/preview. It never inspects the new `analysis/ssim_grid_summary.md`, so a drift between the helper table and JSON would go unnoticed.
- docs/fix_plan.md still lists the helper verification tasks as pending; need to update ledger + galph memory to reflect the new checker requirement + run focus.

## Objectives (single Ralph loop)
1. **Highlights checker upgrade** — Extend `plans/.../bin/check_dense_highlights_match.py` so it parses the SSIM grid summary Markdown table, confirms the preview metadata (`phase-only: ✓`) is present, and asserts the MS-SSIM ±0.000 / MAE ±0.000000 values match `metrics_delta_summary.json` and the preview/highlights text. Add pytest coverage (new module) with RED (summary drift) and GREEN fixtures.
2. **Counted dense run** — Export `AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md`, set `HUB=plans/active/.../reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/`, and execute `python plans/.../bin/run_phase_g_dense.py --hub "$HUB" --dose 1000 --view dense --splits train test --clobber`. Tee stdout to `cli/run_phase_g_dense_stdout.log` and ensure `analysis/` + `cli/` populate with real artifacts (metrics, previews, digest, ssim_grid summary/log).
3. **Verification & digests** — Run `verify_dense_pipeline_artifacts.py` plus the upgraded highlights checker (now SSIM-grid aware), keeping JSON + logs under `analysis/`. Re-run `ssim_grid.py` manually if needed.
4. **Evidence roll-up** — Update `$HUB/summary/summary.md` with MS-SSIM/MAE deltas, preview guard verdict, SSIM grid snippet, pytest evidence, and doc references. Update docs/fix_plan.md + galph_memory with this attempt.

## Execution Sketch
1. `export AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md`; `export HUB=$PWD/plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier`.
2. Update `plans/.../bin/check_dense_highlights_match.py` to:
   - Read `analysis/ssim_grid_summary.md`, parse the Markdown table, and capture MS-SSIM/MAE deltas + preview metadata.
   - Compare parsed values to `metrics_delta_summary.json` (phase deltas) and the preview/highlights text; raise if mismatched or if preview metadata lacks `phase-only: ✓`.
   - Print a concise summary of aligned values + preview verdict to stdout (relative paths only per TYPE-PATH-001).
3. Add `tests/study/test_check_dense_highlights_match.py` with:
   - RED fixture: tamper with the summary table so it drifts from JSON (expect SystemExit with actionable error).
   - GREEN fixture: consistent JSON/highlights/preview/summary (expect pass). Capture logs under `$HUB/red/` and `$HUB/green/`; `pytest --collect-only ...` evidence goes under `$HUB/collect/`.
4. `pytest --collect-only tests/study/test_check_dense_highlights_match.py -vv | tee "$HUB"/collect/pytest_collect_highlights.log`.
5. `pytest tests/study/test_check_dense_highlights_match.py -vv`; save first failing run (if any) under `$HUB/red/pytest_highlights_checker.log` and the passing run under `$HUB/green/pytest_highlights_checker.log`.
6. Execute dense pipeline: `python plans/.../bin/run_phase_g_dense.py --hub "$HUB" --dose 1000 --view dense --splits train test --clobber |& tee "$HUB"/cli/run_phase_g_dense_stdout.log`.
7. `python plans/.../bin/verify_dense_pipeline_artifacts.py --hub "$HUB" --report "$HUB"/analysis/verification_report.json --dose 1000 --view dense | tee "$HUB"/analysis/verify_dense_stdout.log`.
8. `python plans/.../bin/check_dense_highlights_match.py --hub "$HUB" | tee "$HUB"/analysis/check_dense_highlights.log`; if it fails because summary missing/out-of-sync, re-run `python plans/.../bin/ssim_grid.py --hub "$HUB"` and re-check.
9. Update `$HUB/summary/summary.md` with MS-SSIM/MAE deltas, preview guard outcome, pytest selectors, and CLI log pointers; refresh docs/fix_plan.md + galph memory.

## Acceptance Criteria
- Upgraded `check_dense_highlights_match.py` exits non-zero when `ssim_grid_summary.md` is missing, lacks preview metadata, or disagrees with JSON/preview values; stdout reports parsed deltas + preview verdict.
- New pytest module exercises RED/GREEN flows and produces collect-only proof under `$HUB/collect/`.
- Dense `run_phase_g_dense.py --clobber` populates `$HUB/analysis` with metrics summary, deltas, previews, digest, SSIM grid summary, verification report, checker log, and `$HUB/cli` with orchestrator/helper logs (including `ssim_grid_cli.log`).
- `verify_dense_pipeline_artifacts.py` and the enhanced highlights checker succeed on the counted hub, with logs saved in `$HUB/analysis/`.
- `$HUB/summary/summary.md` documents MS-SSIM/MAE deltas, preview guard outcome, checker findings, and log/test references; docs/fix_plan.md + galph memory capture this Attempt.

## Evidence & Artifacts
- Hub: `plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/`
- Expect populated subdirs: `plan/`, `summary/`, `analysis/` (metrics*, ssim_grid_summary.md, verification_report.json, check_dense_highlights.log), `cli/` (run_phase_g_dense_stdout.log, per-phase logs, helper logs), `collect/`, `red/`, `green/`.

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
