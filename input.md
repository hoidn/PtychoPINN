Summary: Guard the --post-verify-only success banner, then execute the dense Phase C→G pipeline (--clobber plus --post-verify-only) to populate the 2025-11-12 hub with SSIM grid, verification, highlights, metrics, and inventory evidence.
Mode: Perf
Focus: STUDY-SYNTH-FLY64-DOSE-OVERLAP-001 — Phase G comparison & analysis (dense real evidence + automated report)
Branch: feature/torchapi-newprompt
Plan Document: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/implementation.md
Reports Hub (active): plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/
Mapped tests:
- AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md HUB=$PWD/plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier pytest --collect-only tests/study/test_phase_g_dense_orchestrator.py -k post_verify_only_executes_chain -vv
- AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md HUB=$PWD/plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier pytest tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_post_verify_only_executes_chain -vv
Artifacts: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/

Do Now (hard validity contract)
- STUDY-SYNTH-FLY64-DOSE-OVERLAP-001
  - Implement: tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_post_verify_only_executes_chain — extend the `--post-verify-only` pytest so stdout must include the SSIM Grid summary/log lines plus `Verification report/log` and `Highlights check log`, and update its stubs to create `cli/ssim_grid_cli.log`, `analysis/verification_report.json`, `analysis/verify_dense_stdout.log`, and `analysis/check_dense_highlights.log` (TEST-CLI-001, TYPE-PATH-001).
  - Pytest: AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md HUB=$PWD/plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier pytest --collect-only tests/study/test_phase_g_dense_orchestrator.py -k post_verify_only_executes_chain -vv | tee "$HUB"/collect/pytest_collect_post_verify_only.log
  - Pytest: AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md HUB=$PWD/plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier pytest tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_post_verify_only_executes_chain -vv | tee "$HUB"/green/pytest_post_verify_only.log
  - CLI: AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md HUB=$PWD/plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py --hub "$HUB" --dose 1000 --view dense --splits train test --clobber |& tee "$HUB"/cli/run_phase_g_dense_stdout.log
  - CLI: AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md HUB=$PWD/plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py --hub "$HUB" --post-verify-only |& tee "$HUB"/cli/run_phase_g_dense_post_verify_only.log
  - Artifacts: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/

How-To Map
1. `export AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md; export HUB=$PWD/plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier; mkdir -p "$HUB"/{collect,green,red,cli,analysis,summary}` so every command honors POLICY-001 and writes logs into the active hub.
2. In `tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_post_verify_only_executes_chain`, assert stdout contains `SSIM Grid Summary (phase-only): analysis/ssim_grid_summary.md`, `SSIM Grid log: cli/ssim_grid_cli.log`, `Verification report: analysis/verification_report.json`, `Verification log: analysis/verify_dense_stdout.log`, and `Highlights check log: analysis/check_dense_highlights.log`; update `stub_run_command` + `stub_generate_artifact_inventory` to create the matching files so `.exists()` checks pass.
3. Run the mapped collect command (`pytest --collect-only ... -k post_verify_only_executes_chain -vv`) and tee the output to `$HUB/collect/pytest_collect_post_verify_only.log`; move failures to `$HUB/red/` before re-running.
4. Execute the targeted pytest (`pytest ...::test_run_phase_g_dense_post_verify_only_executes_chain -vv`) and capture `$HUB/green/pytest_post_verify_only.log`; copy failing stdout/stderr into `$HUB/red/` if it flakes.
5. Launch the counted dense pipeline: `python plans/active/.../bin/run_phase_g_dense.py --hub "$HUB" --dose 1000 --view dense --splits train test --clobber |& tee "$HUB"/cli/run_phase_g_dense_stdout.log`; monitor `cli/phase_*` logs for SUCCESS banners and confirm `analysis/{metrics_delta_summary.json,metrics_delta_highlights_preview.txt,ssim_grid_summary.md,verification_report.json,verify_dense_stdout.log,check_dense_highlights.log,artifact_inventory.txt}` exist afterward.
6. Extract MS-SSIM/MAE deltas from `analysis/metrics_delta_summary.json` with `python - <<'PY' ...` (dump phase-only values vs Baseline/PtyChi) and note them for summary/fix_plan updates; also double-check `analysis/metrics_delta_highlights_preview.txt` contains only phase rows via `rg -n 'amplitude'` (PREVIEW-PHASE-001).
7. Run the verification-only sweep: `python plans/active/.../bin/run_phase_g_dense.py --hub "$HUB" --post-verify-only |& tee "$HUB"/cli/run_phase_g_dense_post_verify_only.log`; ensure banner prints the SSIM grid + verification lines once the guard is in place and that `analysis/artifact_inventory.txt` timestamp refreshes.
8. If either CLI run reports preview/highlights mismatches, execute `python plans/active/.../bin/check_dense_highlights_match.py --hub "$HUB" --debug |& tee "$HUB"/analysis/check_dense_highlights_manual.log` and record the blocker under `$HUB/red/` + docs/fix_plan.md before retrying.
9. Update `$HUB/summary/summary.md` and `$HUB/summary.md` with: command flags, runtimes, MS-SSIM ±0.000 / MAE ±0.000000 deltas, preview verdict, SSIM grid summary path, verifier/highlights logs, pytest selectors/log filenames, and note that docs/fix_plan.md + galph_memory were refreshed; link to the exact CLI logs.
10. Append a new Attempt entry in docs/fix_plan.md plus a galph_memory note with the metrics, preview verdict, verification references, pytest logs, and CLI commands; attach any blocking signatures under `$HUB/red/` if something fails.

Pitfalls To Avoid
- Do not clear or move the active hub; reuse `plans/.../2025-11-12T010500Z/phase_g_dense_full_run_verifier` per docs/INITIATIVE_WORKFLOW_GUIDE.md.
- Never run the orchestrator without `AUTHORITATIVE_CMDS_DOC` set (POLICY-001) and the correct `HUB`, or the CLI will abort early.
- Keep stdout hub-relative: success banners must stay path-normalized (TYPE-PATH-001); avoid absolute paths in new assertions/logs.
- Always tee pytest/CLI output into `$HUB/{collect,green,cli}` before retrying; populate `$HUB/red/` on failures (TEST-CLI-001 evidence requirement).
- Do not skip `--clobber`; stale Phase C artifacts will trip PHASEC-METADATA-001 guard and waste the loop.
- Preview files must stay phase-only; fail fast if `analysis/metrics_delta_highlights_preview.txt` includes “amplitude” (PREVIEW-PHASE-001).
- If any subprocess exits non-zero, stop immediately, capture blocker.log, and update docs/fix_plan.md instead of rerunning blindly.

If Blocked
- Capture the failing CLI or pytest log under `$HUB/red/` (e.g., `cli/phase_e_dense_train.log`), summarize the error signature in docs/fix_plan.md and galph_memory, and mark the Attempt blocked.
- Note whether the failure is upstream (Phase C metadata, training convergence) or downstream (verifier/highlights) and list the exact command + return code so we can spin a dependency item if needed.

Findings Applied (Mandatory)
- POLICY-001 — PyTorch + Torch-based verifiers must stay available; export AUTHORITATIVE_CMDS_DOC before any command (docs/findings.md:8).
- CONFIG-001 — Ensure `run_phase_g_dense.py` still calls `update_legacy_dict` through Phase C helpers before touching legacy code (docs/findings.md:10).
- DATA-001 — Validate all regenerated NPZ/JSON artifacts follow the contract; SSIM grid + verifier will fail otherwise (docs/findings.md:14).
- TYPE-PATH-001 — Success banners/log references must stay hub-relative; the new pytest assertions enforce this (docs/findings.md:21).
- STUDY-001 — Report MS-SSIM/MAE deltas with explicit ± precision for the fly64 study (docs/findings.md:16).
- TEST-CLI-001 — Archive collect + exec pytest logs and ensure CLI filenames include dose/view context (docs/findings.md:23).
- PREVIEW-PHASE-001 — Reject previews containing amplitude terms; rely on `check_dense_highlights_match.py` + manual `rg` check (docs/findings.md:24).
- PHASEC-METADATA-001 — Keep the Phase C guard happy by clobbering the hub before running the dense pipeline (docs/findings.md:22).

Pointers
- plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/implementation.md:204 — Phase G checklist tracks the required pytest guard plus the two CLI runs.
- plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/plan/plan.md:1 — Current objectives, execution sketch, and acceptance criteria.
- docs/fix_plan.md:31 — Ledger entry describing this focus, required evidence, and the latest Do Now state.
- docs/TESTING_GUIDE.md:75 — Study test selectors and pytest usage expectations for CLI guards.

Next Up (optional)
1. If dense `--clobber` and `--post-verify-only` succeed, capture a baseline gs1/batch Phase E training run under the Phase E hub to satisfy the standing guardrail.
2. After dense evidence is green, run `python -m studies.fly64_dose_overlap.comparison --dry-run=false ...` to gather MS-SSIM parity for the baseline view and log the metrics in comparison summaries.

Mapped Tests Guardrail: The `post_verify_only_executes_chain` selector must collect at least one test (verified via the mapped `--collect-only` command); rerun the collect step before editing if pytest reports 0 items.
