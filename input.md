Summary: Rerun `run_phase_g_dense.py --clobber` then `--post-verify-only` from `/home/ollie/Documents/PtychoPINN` so the active hub captures real SSIM grid, verification, preview, metrics, and artifact-inventory evidence for the dense Phase G study.
Mode: Perf
Focus: STUDY-SYNTH-FLY64-DOSE-OVERLAP-001 - Phase G comparison & analysis (dense real evidence + automated report)
Branch: feature/torchapi-newprompt
Plan Document: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/implementation.md
Reports Hub: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/
Mapped tests:
- pytest --collect-only tests/study/test_phase_g_dense_orchestrator.py -k post_verify_only_executes_chain -vv
- pytest tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_post_verify_only_executes_chain -vv
Artifacts: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/

Do Now (hard validity contract)
- STUDY-SYNTH-FLY64-DOSE-OVERLAP-001
  - Implement: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py::main — execute the counted dense Phase C→G pipeline with `--dose 1000 --view dense --splits train test --clobber`, then immediately run `--post-verify-only` so `{analysis,cli}` contain SSIM grid, verification, preview, metrics, and artifact inventory outputs.
  - Pytest: AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md HUB=$PWD/plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier pytest --collect-only tests/study/test_phase_g_dense_orchestrator.py -k post_verify_only_executes_chain -vv | tee "$HUB"/collect/pytest_collect_post_verify_only.log
  - Pytest: AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md HUB=$PWD/plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier pytest tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_post_verify_only_executes_chain -vv | tee "$HUB"/green/pytest_post_verify_only.log
  - CLI: AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md HUB=$PWD/plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py --hub "$HUB" --dose 1000 --view dense --splits train test --clobber |& tee "$HUB"/cli/run_phase_g_dense_stdout.log
  - CLI: AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md HUB=$PWD/plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py --hub "$HUB" --post-verify-only |& tee "$HUB"/cli/run_phase_g_dense_post_verify_only.log
  - Artifacts: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/

How-To Map
1. `test "$(pwd -P)" = "/home/ollie/Documents/PtychoPINN"` then `export AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md` and `export HUB=$PWD/plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier`; `mkdir -p "$HUB"/{analysis,archive,cli,collect,data,green,red,summary}` before running anything so every `tee` persists (TEST-CLI-001).
2. `pytest --collect-only tests/study/test_phase_g_dense_orchestrator.py -k post_verify_only_executes_chain -vv | tee "$HUB"/collect/pytest_collect_post_verify_only.log`; stop if collection fails (move the log to `$HUB/red/` and capture the traceback in docs/fix_plan.md + galph_memory instead of continuing).
3. `pytest tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_post_verify_only_executes_chain -vv | tee "$HUB"/green/pytest_post_verify_only.log` to keep the SSIM grid + verification banner guards GREEN before launching the expensive CLI run.
4. `python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py --hub "$HUB" --dose 1000 --view dense --splits train test --clobber |& tee "$HUB"/cli/run_phase_g_dense_stdout.log`; monitor `cli/phase_*` logs until success sentinels appear, then confirm `analysis/metrics_delta_summary.json`, `analysis/metrics_delta_highlights_preview.txt`, `analysis/ssim_grid_summary.md`, `analysis/verification_report.json`, `analysis/verify_dense_stdout.log`, `analysis/check_dense_highlights.log`, `analysis/artifact_inventory.txt`, and `analysis/metrics_digest.md` all exist.
5. Immediately run `python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py --hub "$HUB" --post-verify-only |& tee "$HUB"/cli/run_phase_g_dense_post_verify_only.log`; ensure stdout repeats the SSIM grid + verification references and that `analysis/verify_dense_stdout.log`, `analysis/check_dense_highlights.log`, and `analysis/artifact_inventory.txt` timestamps refresh. If preview validation fails (PREVIEW-PHASE-001) archive the log under `$HUB/red/` and stop for supervisor review.
6. Post-run validation: (a) `rg -n "amplitude" "$HUB"/analysis/metrics_delta_highlights_preview.txt` must print nothing, (b) `python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/report_phase_g_dense_metrics.py --metrics "$HUB"/analysis/metrics_summary.json --hub "$HUB"` if the sanity table needs refresh, (c) record MS-SSIM ±0.000 / MAE ±0.000000 deltas, preview verdict, SSIM grid summary path, verification/highlights log names, and both pytest outcomes inside `$HUB/summary/summary.md`, copy the block to `$HUB/summary.md`, and update docs/fix_plan.md + galph_memory with the same evidence pointers before ending the loop.

Pitfalls To Avoid
- Do not run commands from `/home/ollie/Documents/PtychoPINN2`; the stale logs showing `ValueError: Object arrays cannot be loaded when allow_pickle=False` came from there.
- Never restore `data/phase_c/run_manifest.json` manually—Phase C regeneration during the counted run must recreate it for provenance (PHASEC-METADATA-001).
- Keep every long CLI piped through `tee` into `$HUB/cli/...` so we have stdout/stderr even if a phase fails mid-run (TEST-CLI-001).
- Abort immediately if either pytest selector fails; rerunning the CLI without green guards will hide regressions (TEST-CLI-001).
- Preview files must contain only four phase-delta lines; any "amplitude" text is a hard failure that must be captured and reported (PREVIEW-PHASE-001).
- Leave blocker evidence in place; new failures go under `$HUB/red/` with timestamps instead of deleting or overwriting logs.
- Ensure PyTorch ≥2.2 remains available; do not modify the environment mid-run (POLICY-001).

If Blocked
- Capture the failing command, exit code, and log path inside `$HUB/red/blocked_$(date -u +%FT%H%M%SZ).md`, include excerpts from the relevant CLI/analysis log, and stop.
- Update docs/fix_plan.md Attempts History and galph_memory with the failure signature plus artifact pointers.
- Leave the hub dirty (no cleanup) so the next loop can inspect the exact evidence.

Findings Applied (Mandatory)
- POLICY-001 — PyTorch verifier + CLI helpers require torch≥2.2; keep the environment intact.
- CONFIG-001 — Keep `update_legacy_dict` ordering during Phase C regeneration so patched NPZs stay valid.
- DATA-001 — Treat the previous ValueError as a schema violation; rerun until NPZ loads succeed and SSIM grid/verifier outputs are produced.
- TYPE-PATH-001 — Success banners and log references must remain hub-relative; the mapped pytest guards enforce this.
- STUDY-001 — Report MS-SSIM/MAE deltas with ± formatting when updating summaries and docs.
- TEST-CLI-001 — Archive collect/exec pytest logs plus all CLI outputs to `$HUB` for reproducibility.
- PREVIEW-PHASE-001 — Reject previews containing amplitude text; use the highlights checker if anything looks off.
- PHASEC-METADATA-001 — Let Phase C reruns rebuild the `_metadata` files instead of manual edits.

Pointers
- docs/fix_plan.md:18 — Active ledger entry for STUDY-SYNTH-FLY64-DOSE-OVERLAP-001 detailing status/attempts.
- plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/implementation.md:204 — Phase G checklist + outstanding counted-run tasks.
- plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/plan/plan.md:1 — Reality check + execution sketch for the dense rerun hub.
- docs/TESTING_GUIDE.md:389 — Phase G orchestrator run/validation commands and evidence expectations (AUTHORITATIVE_CMDS_DOC).
- tests/study/test_phase_g_dense_orchestrator.py:1945 — `test_run_phase_g_dense_post_verify_only_executes_chain` selector guarding SSIM grid + verification references.

Next Up (optional)
1. After the counted run succeeds, execute `plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/analyze_dense_metrics.py --hub "$HUB"` to refresh the MS-SSIM digest artifacts.
2. Re-run `plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/report_phase_g_dense_metrics.py` to capture the MS-SSIM sanity table under `analysis/metrics_digest.md` if deltas shift.

Mapped Tests Guardrail
- `pytest --collect-only tests/study/test_phase_g_dense_orchestrator.py -k post_verify_only_executes_chain -vv` must report 1 collected test before running the long CLI; if not, stop and escalate.

Hard Gate
- Do not mark this focus done until both CLI runs succeed, `analysis/metrics_delta_highlights_preview.txt` stays phase-only, SSIM grid + verification artifacts exist in `$HUB/analysis`, and docs/fix_plan.md + galph_memory record MS-SSIM ±0.000 / MAE ±0.000000 deltas with preview/verifier references; otherwise log the blocker under `$HUB/red/` and halt.
