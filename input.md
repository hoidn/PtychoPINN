Summary: Execute the dense Phase C→G pipeline from this repo (not PtychoPINN2) with --clobber plus the follow-on --post-verify-only sweep so the 2025-11-12 hub finally captures SSIM grid, verification, highlights, metrics, preview verdict, and MS-SSIM/MAE deltas with logged evidence.
Mode: Perf
Focus: STUDY-SYNTH-FLY64-DOSE-OVERLAP-001 — Phase G comparison & analysis (dense real evidence + automated report)
Branch: feature/torchapi-newprompt
Plan Document: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/implementation.md
Reports Hub: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/
Mapped tests:
- pytest --collect-only tests/study/test_phase_g_dense_orchestrator.py -k post_verify_only_executes_chain -vv
- pytest tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_post_verify_only_executes_chain -vv
Artifacts: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/

Do Now (hard validity contract)
- STUDY-SYNTH-FLY64-DOSE-OVERLAP-001
  - Implement: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py::main — rerun the counted dense pipeline with `--clobber` plus the immediate `--post-verify-only` sweep from `/home/ollie/Documents/PtychoPINN` so `{analysis,cli}` capture Phase C→G logs, SSIM grid summary/log, verification report/log, highlights logs, metrics digest, metrics delta JSON/preview, and artifact inventory evidence for this hub.
  - Pytest: AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md HUB=$PWD/plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier pytest --collect-only tests/study/test_phase_g_dense_orchestrator.py -k post_verify_only_executes_chain -vv | tee "$HUB"/collect/pytest_collect_post_verify_only.log
  - Pytest: AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md HUB=$PWD/plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier pytest tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_post_verify_only_executes_chain -vv | tee "$HUB"/green/pytest_post_verify_only.log
  - CLI: AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md HUB=$PWD/plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py --hub "$HUB" --dose 1000 --view dense --splits train test --clobber |& tee "$HUB"/cli/run_phase_g_dense_stdout.log
  - CLI: AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md HUB=$PWD/plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py --hub "$HUB" --post-verify-only |& tee "$HUB"/cli/run_phase_g_dense_post_verify_only.log
  - Artifacts: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/

How-To Map
1. `test "$(pwd -P)" = "/home/ollie/Documents/PtychoPINN"` to confirm you are in the canonical repo, then `export AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md; export HUB=$PWD/plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier; mkdir -p "$HUB"/{analysis,cli,collect,green,red,summary}` so evidence lands in the right hub.
2. Run the mapped collect-only selector (`pytest --collect-only tests/study/test_phase_g_dense_orchestrator.py -k post_verify_only_executes_chain -vv | tee "$HUB"/collect/pytest_collect_post_verify_only.log`) and stop if it fails or collects zero tests; stash failing logs under `red/` before retrying.
3. Execute `pytest tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_post_verify_only_executes_chain -vv | tee "$HUB"/green/pytest_post_verify_only.log` to keep TEST-CLI-001 guards GREEN before expensive CLI work.
4. Launch the counted dense pipeline from this repo path: `python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py --hub "$HUB" --dose 1000 --view dense --splits train test --clobber |& tee "$HUB"/cli/run_phase_g_dense_stdout.log`. Monitor every phase log for SUCCESS and ensure `$HUB/analysis` gains `metrics_delta_summary.json`, `metrics_delta_highlights_preview.txt`, `metrics_delta_highlights.txt`, `metrics_digest.md`, `aggregate_report.md`, `ssim_grid_summary.md`, `verification_report.json`, `verify_dense_stdout.log`, `check_dense_highlights.log`, and `artifact_inventory.txt`.
5. Guard PREVIEW-PHASE-001 immediately after the run: `rg -n "amplitude" "$HUB"/analysis/metrics_delta_highlights_preview.txt` must print nothing; if it does, move the offending files into `red/` and rerun after fixing the preview content.
6. Rerun verification without Phase C via `python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py --hub "$HUB" --post-verify-only |& tee "$HUB"/cli/run_phase_g_dense_post_verify_only.log`, confirming the banner repeats the SSIM grid / verification lines and `analysis/artifact_inventory.txt` receives a fresh timestamp.
7. If either CLI run reports preview/highlights mismatches, collect diagnostics with `python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/check_dense_highlights_match.py --hub "$HUB" --debug | tee "$HUB"/analysis/check_dense_highlights_manual.log` before attempting another rerun; keep the log under `analysis/` for evidence.
8. Summarize results: add runtimes, MS-SSIM ±0.000 / MAE ±0.000000 deltas (phase emphasis), preview verdict, SSIM grid table reference, verification/highlights log links, pytest selectors, and doc updates to `$HUB`/summary/summary.md (mirror in summary.md), then refresh docs/fix_plan.md and galph_memory with the same evidence bundle.

Pitfalls To Avoid
- Running commands from `/home/ollie/Documents/PtychoPINN2` (the hub paths written there never populate this repo’s evidence tree).
- Skipping the AUTHORITATIVE_CMDS_DOC/HUB exports or the initial `mkdir -p` (breaks POLICY-001 + leaves `analysis/` missing again).
- Forgetting to tee pytest/CLI output into `$HUB` (violates TEST-CLI-001 evidence contract).
- Running `--post-verify-only` before the fresh `--clobber` execution (would reuse stale Phase C artifacts and leave inventory untouched).
- Allowing `metrics_delta_highlights_preview.txt` to include "amplitude" or drop ± formatting (breaches PREVIEW-PHASE-001).
- Ignoring Phase C metadata failures or SSIM grid/verifier exits ≠ 0; capture blocker logs under `red/` and stop.
- Ending the loop without updating summary.md + docs/fix_plan.md with MS-SSIM/MAE deltas, preview verdict, and verifier/highlights references.

If Blocked
- If either mapped pytest selector fails or collects zero, move the log to `$HUB`/red/, capture the selector + failure snippet in docs/fix_plan.md Attempts History, and request new supervisor guidance before touching the CLI.
- If `run_phase_g_dense.py` exits non-zero, archive the CLI and offending phase log under `$HUB`/red/, log the phase + command in docs/fix_plan.md and galph_memory, and stop rerunning until a plan update arrives.

Findings Applied (Mandatory)
- POLICY-001 — Export AUTHORITATIVE_CMDS_DOC so PyTorch-dependent verifiers run under the sanctioned environment.
- CONFIG-001 — Preserve the orchestrator’s `update_legacy_dict(params.cfg, config)` ordering so Phase C consumers observe the right params.
- DATA-001 — Treat missing SSIM grid, verification JSON/log, or artifact inventory outputs as blockers; archive evidence before reruns.
- TYPE-PATH-001 — Keep success-banner and summary references hub-relative (`analysis/...`, `cli/...`) as guarded by the pytest suite.
- STUDY-001 — Report MS-SSIM + MAE deltas with ± precision inside summary docs.
- TEST-CLI-001 — Tee both pytest selectors and both CLI commands into the hub to maintain auditable RED/GREEN evidence.
- PREVIEW-PHASE-001 — Verify the preview file is phase-only via the grep guard; rerun if amplitude tokens reappear.
- PHASEC-METADATA-001 — Respect the refreshed Phase C metadata layout; if validation fails, capture the blocker log and halt.

Pointers
- plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/implementation.md:204 — Phase G checklist tracks the unchecked dense rerun + verification-only sweep requirements.
- plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/plan/plan.md:1 — Reality check + execution sketch (repo-path guard, commands, acceptance criteria).
- docs/TESTING_GUIDE.md:389 — Canonical run_phase_g_dense command block and artifact expectations for Phase G.
- docs/fix_plan.md:18 — Ledger entry describing the missing dense rerun evidence and artifact paths to populate.
- docs/findings.md:24 — PREVIEW-PHASE-001 plus the other active guardrails governing preview/verifier content.

Next Up (optional)
1. `python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/report_phase_g_dense_metrics.py --metrics "$HUB"/analysis/metrics_summary.json --output "$HUB"/analysis/aggregate_report.md --highlights "$HUB"/analysis/aggregate_highlights.txt`
2. `python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/analyze_dense_metrics.py --hub "$HUB" --metrics "$HUB"/analysis/metrics_summary.json`

Mapped Tests Guardrail
- `pytest --collect-only tests/study/test_phase_g_dense_orchestrator.py -k post_verify_only_executes_chain -vv` must collect the `test_run_phase_g_dense_post_verify_only_executes_chain` node before any CLI command; if collection ever drops to 0, stop immediately, file the blocker, and do not proceed to the expensive pipeline.
