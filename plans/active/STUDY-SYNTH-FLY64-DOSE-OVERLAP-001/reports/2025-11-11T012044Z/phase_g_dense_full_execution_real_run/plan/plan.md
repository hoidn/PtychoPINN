# Phase G Dense Preview Guard Follow-Through (2025-11-11T012044Z)

## Reality Check
- Commit 783c32aa shipped the preview-phase-only guard: `plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/verify_dense_pipeline_artifacts.py::validate_metrics_delta_highlights` (lines 309-481) now rejects preview lines containing `amplitude`, enforces the single-value `<prefix>: <delta>` pattern, and surfaces `preview_phase_only` / `preview_format_errors` metadata for reports.
- New RED fixture `tests/study/test_phase_g_dense_artifacts_verifier.py::test_verify_dense_pipeline_highlights_preview_contains_amplitude` (≈line 1921) fails on amplitude-contaminated previews; GREEN coverage (`...highlights_complete`) asserts the metadata stays clean. Tests passed locally but logs were not archived under the prior 2025-11-11T005802Z hub.
- Documentation is stale: `docs/TESTING_GUIDE.md` §"Phase G Delta Metrics Persistence" still states “Signed 3-decimal formatting for all deltas” and never mentions the preview artifact. `docs/development/TEST_SUITE_INDEX.md` lacks an entry for the new preview guard selector, so other engineers cannot discover/run it.
- The previous evidence hub `plans/active/.../reports/2025-11-11T005802Z/phase_g_dense_full_execution_real_run/` still contains empty `{analysis,cli,collect,green,red}` directories (no Phase D–G CLI logs, highlights, verifier outputs, or pytest logs). No dense run has occurred since the preview helper landed.
- Provisioned a fresh hub for this loop: `HUB=$PWD/plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-11T012044Z/phase_g_dense_full_execution_real_run`. Subdirectories `{analysis,cli,collect,green,red,summary}` are ready for logs, so the stale 005802Z hub can be left as a planning artifact.

## Objectives for Ralph (single loop)
1. **Doc + Registry Sync (Implement)**
   - Update `docs/TESTING_GUIDE.md::Phase G Delta Metrics Persistence` (around lines 330-370) to describe the JSON/highlights/preview trio: MS-SSIM keeps ±0.000 precision, MAE uses ±0.000000, and the preview file must contain only four phase deltas (no amplitude keyword). Reference PREVIEW-PHASE-001 and TYPE-PATH-001, and document where the preview lives in the hub.
   - Update `docs/development/TEST_SUITE_INDEX.md` Phase G section to list the new preview guard selector (`pytest tests/study/test_phase_g_dense_artifacts_verifier.py::test_verify_dense_pipeline_highlights_preview_contains_amplitude -vv`), explaining the RED fixture (amplitude contamination) and GREEN expectations (`preview_phase_only=True`, empty `preview_format_errors`).
2. **Test Confirmation (logs archived under $HUB/)**
   - Export `AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md` and `HUB=$PWD/plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-11T012044Z/phase_g_dense_full_execution_real_run` before running anything.
   - Re-run `pytest tests/study/test_phase_g_dense_artifacts_verifier.py::test_verify_dense_pipeline_highlights_preview_contains_amplitude -vv | tee "$HUB"/green/pytest_preview_guard_green.log` to capture the guard evidence (test stays GREEN while the validator rejects amplitude contamination inside the fixture).
   - Re-run `pytest tests/study/test_phase_g_dense_artifacts_verifier.py::test_verify_dense_pipeline_highlights_complete -vv | tee "$HUB"/green/pytest_highlights_complete.log` to confirm metadata assertions.
   - Re-run `pytest tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_exec_runs_analyze_digest -vv | tee "$HUB"/green/pytest_orchestrator_digest.log` to ensure orchestrator digest step still passes with the helper changes.
   - `pytest --collect-only tests/study/test_phase_g_dense_artifacts_verifier.py -vv | tee "$HUB"/collect/pytest_collect_highlights.log` for guardrail evidence.
3. **Dense Pipeline Execution (real artifacts)**
   - Ensure `$HUB/cli` is empty, then run `python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py --hub "$HUB" --dose 1000 --view dense --splits train test --clobber |& tee "$HUB"/cli/run_phase_g_dense.log`. Monitor `[1/8]`→`[8/8]` progression; if a phase fails, stop, archive the log under `$HUB/cli/` and update summary + fix_plan with the failure signature.
   - After SUCCESS, copy the per-phase CLI logs from orchestrator into `$HUB/cli/` (baseline/dense Phase E, Phase F train/test, Phase G comparisons, helper logs). Confirm `analysis/metrics_delta_highlights.txt`, `analysis/metrics_delta_highlights_preview.txt`, and `analysis/metrics_delta_summary.json` exist.
4. **Verification + Highlights Parity**
   - `python plans/active/.../bin/verify_dense_pipeline_artifacts.py --hub "$HUB" --report "$HUB"/analysis/pipeline_verification.json |& tee "$HUB"/analysis/verifier_cli.log` — expect `preview_phase_only=True` and empty `preview_format_errors`.
   - `python plans/active/.../bin/check_dense_highlights_match.py --hub "$HUB" |& tee "$HUB"/analysis/highlights_check.log` to confirm highlights vs preview parity.
   - `pytest --collect-only tests/study/test_phase_g_dense_artifacts_verifier.py -vv | tee "$HUB"/collect/pytest_collect_highlights_post_run.log` (post-run evidence for guardrail per policy).
5. **Evidence Bundle + Ledger Updates**
   - Summarize MS-SSIM/MAE deltas (phase emphasis), preview guard status, verifier outcome, highlight parity, and CLI log inventory in `$HUB/summary/summary.md`. Note any runs skipped/skipped tests with rationale. Reference PREVIEW-PHASE-001 resolution intent.
   - Update `docs/fix_plan.md` Attempts History with the execution attempt (include `$HUB` path and pytest selectors). If preview guard passes end-to-end, mark PREVIEW-PHASE-001 as mitigated in docs/findings.md with cross-link to the new artifacts.

## Required Findings to Observe
- **POLICY-001:** PyTorch backend must remain installed/enabled for Phase F (PtyChi LSQML) during dense run.
- **CONFIG-001:** Keep `AUTHORITATIVE_CMDS_DOC` exported before Phase C jobs or verifiers to avoid params.cfg drift.
- **DATA-001:** Never mutate Phase C NPZ outputs; rerun generators instead.
- **TYPE-PATH-001:** Maintain POSIX-relative paths (`analysis/...`) inside JSON + summary metadata.
- **STUDY-001:** Report MS-SSIM/MAE deltas with explicit ± signs and phase emphasis.
- **TEST-CLI-001:** Capture real CLI filenames (dose/view suffixes, helper logs) in `$HUB/cli/` and ensure verifier/tests assert on them.
- **PREVIEW-PHASE-001:** Preview files must remain phase-only; failure evidence must keep metadata actionable.

## Risks & Mitigations
- **Long-running orchestrator:** Dense pipeline can take hours; if it fails mid-run, stop and document the phase-specific log instead of rerunning blindly.
- **Evidence completeness:** Tests already pass, but logs were never archived; ensure each selector's stdout/stderr is captured under `$HUB`/{green,collect} so PREVIEW-PHASE-001 has reproducible proof.
- **Doc drift:** Ensure doc updates exactly match script behavior (MS-SSIM ±0.000, MAE ±0.000000, preview path `analysis/metrics_delta_highlights_preview.txt`).
