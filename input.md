Summary: Dense Phase G hub still only has `analysis/blocker.log` plus the short trio of CLI logs, so rerun `run_phase_g_dense.py --clobber` and then `--post-verify-only` from `/home/ollie/Documents/PtychoPINN` to regenerate SSIM grid, verification, highlights, preview, metrics, and artifact-inventory artifacts in the active hub.
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
  - Implement: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py::main - execute the counted dense Phase C->G run with `--dose 1000 --view dense --splits train test --clobber` from `/home/ollie/Documents/PtychoPINN`, then immediately invoke `--post-verify-only` so `{analysis,cli}` capture SSIM grid, verification, highlights, preview, metrics, and artifact inventory evidence.
  - Pytest: AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md HUB=$PWD/plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier pytest --collect-only tests/study/test_phase_g_dense_orchestrator.py -k post_verify_only_executes_chain -vv | tee "$HUB"/collect/pytest_collect_post_verify_only.log
  - Pytest: AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md HUB=$PWD/plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier pytest tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_post_verify_only_executes_chain -vv | tee "$HUB"/green/pytest_post_verify_only.log
  - CLI: AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md HUB=$PWD/plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py --hub "$HUB" --dose 1000 --view dense --splits train test --clobber |& tee "$HUB"/cli/run_phase_g_dense_stdout.log
  - CLI: AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md HUB=$PWD/plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py --hub "$HUB" --post-verify-only |& tee "$HUB"/cli/run_phase_g_dense_post_verify_only.log
  - Artifacts: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/

How-To Map
1. `test "$(pwd -P)" = "/home/ollie/Documents/PtychoPINN"`; then `export AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md` and `export HUB=$PWD/plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier`. `mkdir -p "$HUB"/{analysis,archive,cli,collect,data,green,red,summary}` before running anything else so all `tee` commands keep evidence (TEST-CLI-001).
2. `pytest --collect-only tests/study/test_phase_g_dense_orchestrator.py -k post_verify_only_executes_chain -vv | tee "$HUB"/collect/pytest_collect_post_verify_only.log`; if collection fails, stop immediately, move the log to `$HUB/red/`, and capture the traceback in docs/fix_plan.md + galph_memory instead of proceeding.
3. `pytest tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_post_verify_only_executes_chain -vv | tee "$HUB"/green/pytest_post_verify_only.log` to keep the SSIM grid + verification banner guards GREEN before launching the long CLI run.
4. `python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py --hub "$HUB" --dose 1000 --view dense --splits train test --clobber |& tee "$HUB"/cli/run_phase_g_dense_stdout.log`; leave the terminal attached so Phase C->G progress stays visible. Confirm `cli/phase_*` logs, `analysis/metrics_delta_summary.json`, `analysis/metrics_delta_highlights_preview.txt`, `analysis/ssim_grid_summary.md`, `analysis/verification_report.json`, `analysis/verify_dense_stdout.log`, `analysis/check_dense_highlights.log`, `analysis/artifact_inventory.txt`, and `analysis/metrics_digest.md` all exist at completion.
5. Immediately run the shortened verification sweep: `python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py --hub "$HUB" --post-verify-only |& tee "$HUB"/cli/run_phase_g_dense_post_verify_only.log`; ensure stdout repeats the SSIM grid + verification references (no missing lines) and that `analysis/verify_dense_stdout.log` plus `analysis/check_dense_highlights.log` timestamps refresh. If preview validation fails (PREVIEW-PHASE-001), archive the failing log under `$HUB/red/` and stop for supervisor review.
6. Post-run validation: (a) `rg -n "amplitude" "$HUB"/analysis/metrics_delta_highlights_preview.txt` must return nothing; (b) `python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/report_phase_g_dense_metrics.py --metrics "$HUB"/analysis/metrics_summary.json --hub "$HUB"` to regenerate the MS-SSIM sanity table if needed; (c) record MS-SSIM +/-0.000 / MAE +/-0.000000 deltas, preview verdict, SSIM grid summary path, verification/highlights log names, and both pytest outcomes inside `$HUB/summary/summary.md`, copy the block to `$HUB/summary.md`, and update docs/fix_plan.md + galph_memory with the same evidence pointers before ending the loop.

Pitfalls To Avoid
- Do not run any command from `/home/ollie/Documents/PtychoPINN2`; the old attempt failed there with the same ValueError.
- Do not restore `data/phase_c/run_manifest.json` manually-Phase C regeneration must recreate it for provenance.
- Keep every long-running CLI piped through `tee "$HUB"/cli/...` so we retain stdout/stderr even if the job fails.
- Never skip the pytest guards; if they fail, stop and log instead of attempting the expensive dense run.
- Preview artifacts must remain phase-only; if `metrics_delta_highlights_preview.txt` contains "amplitude" do not suppress the failure (PREVIEW-PHASE-001).
- Avoid deleting or moving blocker evidence; place any new failures under `$HUB/red/` with clear filenames.

If Blocked
- Capture the failing command, exit code, and log path (e.g., `$HUB/cli/phase_d_dense.log`) inside `$HUB/red/blocked_$(date -u +%FT%H%M%SZ).md`.
- Update docs/fix_plan.md Attempts History with the failure signature and add the same note to galph_memory.
- Leave the hub dirty (do not clean up artifacts) so the next supervisor loop can inspect the evidence.

Findings Applied (Mandatory)
- POLICY-001 - PyTorch-backed verifier/log parsing still runs under torch>=2.2; do not disable or downgrade dependencies mid-run.
- CONFIG-001 - Ensure `run_phase_g_dense.py` keeps calling `update_legacy_dict` before legacy consumers during Phase C regeneration.
- DATA-001 - Phase C NPZs and Phase D overlap outputs must stay schema-compliant; treat the current ValueError as a schema violation until rerun succeeds.
- TYPE-PATH-001 - Keep success-banner and log references hub-relative; the pytest guard ensures regressions are caught.
- STUDY-001 - Record MS-SSIM/MAE deltas with +/- formatting so the fly64 comparison remains auditable.
- TEST-CLI-001 - Archive collect/exec pytest logs plus CLI outputs for every long-running command.
- PREVIEW-PHASE-001 - Preview artifacts must exclude amplitude text; rerun the checker if the guard trips.
- PHASEC-METADATA-001 - Let Phase C regeneration rebuild `_metadata` files; do not hand-edit the manifest.

Pointers
- plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/implementation.md:199 - Active Phase G checklist and outstanding tasks.
- plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/plan/plan.md:1 - Reality check + execution sketch for the dense rerun.
- docs/TESTING_GUIDE.md:331 - Defines the metrics delta artifacts that must exist after the counted run.
- tests/study/test_phase_g_dense_orchestrator.py:1945 - Guarded `test_run_phase_g_dense_post_verify_only_executes_chain` selector referenced above.
- docs/findings.md:16 - STUDY-001 context for MS-SSIM/MAE reporting discipline.

Next Up (optional)
1. If the dense run succeeds quickly, invoke `plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/analyze_dense_metrics.py` + `report_phase_g_dense_metrics.py` to refresh the MS-SSIM digest artifacts.
2. Generate comparison plots via `scripts/compare_models.py` for dose 1000 dense view to visualize the new deltas.

Mapped Tests Guardrail
- `pytest --collect-only tests/study/test_phase_g_dense_orchestrator.py -k post_verify_only_executes_chain -vv` must report 1 collected test before proceeding; if collection drops to 0, stop and mark the loop blocked.

Hard Gate
- Do not mark the focus done unless both CLI runs finish successfully, the preview file stays phase-only, and artifact inventory/verification logs exist inside `$HUB/analysis`; otherwise capture the failure under `$HUB/red/` and request supervisor guidance.
