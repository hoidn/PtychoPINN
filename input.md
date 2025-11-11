Summary: Dense Phase C→G hub evidence still tops out at `analysis/blocker.log` plus `cli/{phase_c_generation,phase_d_dense,run_phase_g_dense_stdout}.log`, so rerun `run_phase_g_dense.py --clobber` (then `--post-verify-only`) from `/home/ollie/Documents/PtychoPINN` to regenerate Phase C outputs and capture SSIM grid, verification, preview, metrics, and artifact-inventory artifacts.
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
  - Implement: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py::main — run `--dose 1000 --view dense --splits train test --clobber` from `/home/ollie/Documents/PtychoPINN`, then immediately execute `--post-verify-only` so `{analysis,cli}` collect SSIM grid, verification, highlights, preview, metrics, and artifact inventory evidence.
  - Pytest: AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md HUB=$PWD/plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier pytest --collect-only tests/study/test_phase_g_dense_orchestrator.py -k post_verify_only_executes_chain -vv | tee "$HUB"/collect/pytest_collect_post_verify_only.log
  - Pytest: AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md HUB=$PWD/plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier pytest tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_post_verify_only_executes_chain -vv | tee "$HUB"/green/pytest_post_verify_only.log
  - CLI: AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md HUB=$PWD/plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py --hub "$HUB" --dose 1000 --view dense --splits train test --clobber |& tee "$HUB"/cli/run_phase_g_dense_stdout.log
  - CLI: AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md HUB=$PWD/plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py --hub "$HUB" --post-verify-only |& tee "$HUB"/cli/run_phase_g_dense_post_verify_only.log
  - Artifacts: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/

How-To Map
1. `test "$(pwd -P)" = "/home/ollie/Documents/PtychoPINN"`; export `AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md` and `HUB=$PWD/plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier`, then `mkdir -p "$HUB"/{analysis,cli,collect,green,red,summary}` so tee never drops evidence (TEST-CLI-001). Do **not** restore `data/phase_c/run_manifest.json` manually—the counted run must regenerate it.
2. `pytest --collect-only tests/study/test_phase_g_dense_orchestrator.py -k post_verify_only_executes_chain -vv | tee "$HUB"/collect/pytest_collect_post_verify_only.log`; if collection fails, move the log to `$HUB/red/` and stop for supervisor triage.
3. `pytest tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_post_verify_only_executes_chain -vv | tee "$HUB"/green/pytest_post_verify_only.log` to keep the hub-relative banner/path guards GREEN before touching the expensive pipeline.
4. `python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py --hub "$HUB" --dose 1000 --view dense --splits train test --clobber |& tee "$HUB"/cli/run_phase_g_dense_stdout.log`; monitor `phase_c_generation.log`, `phase_d_dense.log`, `phase_e_*`, `phase_f_dense_train.log`, `phase_g_dense_compare.log`, `ssim_grid_cli.log`, `verify_dense_stdout.log`, and `check_dense_highlights.log` for SUCCESS sentinels and stash any RED log under `$HUB/red/`.
5. Validate artifacts: `ls "$HUB"/analysis` must now list metrics_delta_summary.json, metrics_delta_highlights_preview.txt, metrics_digest.md, aggregate_report.md, ssim_grid_summary.md, verification_report.json, verify_dense_stdout.log, check_dense_highlights.log, and artifact_inventory.txt; `rg -n "amplitude" "$HUB"/analysis/metrics_delta_highlights_preview.txt` must return nothing (PREVIEW-PHASE-001).
6. `python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py --hub "$HUB" --post-verify-only |& tee "$HUB"/cli/run_phase_g_dense_post_verify_only.log` to prove the shortened chain regenerates SSIM grid + verification outputs and rewrites `analysis/artifact_inventory.txt` with hub-relative success banners.
7. If verification/highlights disagree, move the failing logs under `$HUB/red/`, run `python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/check_dense_highlights_match.py --hub "$HUB" | tee "$HUB"/analysis/check_dense_highlights_manual.log`, capture the failure signature, and stop for supervisor guidance.
8. When both runs succeed, prepend a new block to `$HUB/summary/summary.md` (mirror to `$HUB/summary.md`) covering run parameters, MS-SSIM ±0.000 / MAE ±0.000000 deltas, preview verdict, SSIM grid/verifier/highlights references, pytest selectors, and doc/test status; update docs/fix_plan.md Attempts History plus galph_memory with the same evidence.

Pitfalls To Avoid
- Launching any command outside `/home/ollie/Documents/PtychoPINN`; the previous `/home/ollie/Documents/PtychoPINN2` attempt produced the ValueError logged in `cli/phase_d_dense.log` and yielded zero artifacts.
- Skipping the `mkdir -p "$HUB"/...` prep, which causes tee to fail and violates TEST-CLI-001 evidence rules.
- Manually recreating `data/phase_c/run_manifest.json`; it must be regenerated by the counted `--clobber` run so git diffs reflect real evidence.
- Accepting previews that include “amplitude” or missing ± formatting; PREVIEW-PHASE-001 requires phase-only deltas.
- Forgetting to archive CLI/test logs under `$HUB/{collect,green,cli}`; missing logs break TEST-CLI-001 and force a rerun.
- Starting `--post-verify-only` before the `--clobber` pipeline finishes; it will replay stale artifacts and invalidate the evidence.
- Ignoring SSIM grid/verification/metrics outputs; if any are missing, move logs to `$HUB/red/`, capture the blocker, and stop.
- Leaving pytest failures untriaged; move RED logs before reruns so the hub history stays auditable.

If Blocked
- Pytest failure: leave the log under `$HUB/red/`, capture the selector + failure snippet in docs/fix_plan.md and galph_memory, then halt for supervisor guidance.
- CLI non-zero exit: tee already wrote to `$HUB/cli`; move the offending phase log (and blocker log) into `$HUB/red/`, note the command + error text, and stop so we can triage before rerunning.
- Missing SSIM grid/verification/preview outputs after `--clobber`: archive the incomplete logs + `analysis/blocker.log`, update docs/fix_plan.md with the failure signature, and halt for new guidance.

Findings Applied (Mandatory)
- POLICY-001 — docs/findings.md:8 — PyTorch must remain available for verifier/highlights helpers; export AUTHORITATIVE_CMDS_DOC before running commands.
- CONFIG-001 — docs/findings.md:10 — Keep `update_legacy_dict(params.cfg, config)` in the Phase C path by using the orchestrator rather than ad-hoc scripts.
- DATA-001 — docs/findings.md:14 — SSIM grid and verification JSON/logs are contract artifacts; if missing, treat as a blocking defect.
- TYPE-PATH-001 — docs/findings.md:21 — Success banners, logs, and pytest assertions must keep hub-relative paths; the guard test enforces this.
- STUDY-001 — docs/findings.md:16 — Summaries must report MS-SSIM + MAE deltas with explicit ± precision.
- TEST-CLI-001 — docs/findings.md:23 — Archive RED/GREEN pytest logs and CLI stdout for every command/selector.
- PREVIEW-PHASE-001 — docs/findings.md:24 — Reject preview files containing amplitude text or malformed ± tokens; rely on the highlights checker.
- PHASEC-METADATA-001 — docs/findings.md:22 — Allow the Phase C metadata guard to run via the orchestrator; log blockers if the NPZ layout regresses.

Pointers
- plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/implementation.md:204 — Active Phase G checklist enumerating the counted rerun + reporting deliverables.
- plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/plan/plan.md:1 — Hub-specific execution sketch and acceptance criteria (pwd guard, artifact list).
- docs/fix_plan.md:4 — Ledger status + guardrails for this focus, plus the latest Attempts History entry.
- docs/TESTING_GUIDE.md:210 — Phase G orchestrator test references and selector descriptions for the mapped pytest nodes.

Next Up (optional)
1. python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/report_phase_g_dense_metrics.py --metrics "$HUB"/analysis/metrics_summary.json --output "$HUB"/analysis/aggregate_report.md
2. python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/analyze_dense_metrics.py --hub "$HUB" --metrics "$HUB"/analysis/metrics_summary.json

Mapped Tests Guardrail
- pytest --collect-only tests/study/test_phase_g_dense_orchestrator.py -k post_verify_only_executes_chain -vv must collect (>0) before any CLI command. Hard Gate: if the selector fails to collect due to changes made this loop, stop immediately, log the RED output under `$HUB/red/`, and request supervisor guidance before proceeding.
