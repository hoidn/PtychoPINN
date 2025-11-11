Summary: Add `--post-verify-only` to the dense orchestrator, cover it with new pytest selectors, then run the counted dense Phase C→G pipeline (plus a post-verify-only rerun) into the 2025-11-12 hub with full verification evidence.
Mode: Perf
Focus: STUDY-SYNTH-FLY64-DOSE-OVERLAP-001 — Phase G comparison & analysis (dense real evidence + automated report)
Branch: feature/torchapi-newprompt
Plan Document: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/implementation.md
Reports Hub (active): plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/
Mapped tests:
- pytest --collect-only tests/study/test_phase_g_dense_orchestrator.py -k post_verify_only -vv
- pytest tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_post_verify_only_executes_chain -vv
Artifacts: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/

Do Now (hard validity contract)
- Implement: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py::main — add a `--post-verify-only` mode (mutually exclusive with `--skip-post-verify`) that skips Phase C→F execution, reuses the existing hub outputs, runs only the SSIM grid helper + verifier + highlights checker, refreshes `analysis/artifact_inventory.txt`, prints hub-relative log/report paths, and supports `--collect-only` output for the trimmed command list (POLICY-001, TYPE-PATH-001, TEST-CLI-001, PREVIEW-PHASE-001, PHASEC-METADATA-001).
- Implement: tests/study/test_phase_g_dense_orchestrator.py::{test_run_phase_g_dense_collect_only_post_verify_only,test_run_phase_g_dense_post_verify_only_executes_chain} — extend the collect-only assertions for the new flag and add a monkeypatched execution test that captures the SSIM grid → verifier → highlights chain and proves artifact inventory regeneration is always invoked (DATA-001, TEST-CLI-001).
- Execute: AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md HUB=$PWD/plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py --hub "$HUB" --dose 1000 --view dense --splits train test --clobber |& tee "$HUB"/cli/run_phase_g_dense_stdout.log to produce the counted Phase C→G artifacts, SSIM grid summary, verification report, and highlights logs under this hub (STUDY-001, DATA-001).
- Validate: pytest --collect-only tests/study/test_phase_g_dense_orchestrator.py -k post_verify_only -vv | tee "$HUB"/collect/pytest_collect_orchestrator_post_verify_only.log (RED logs go to `$HUB`/red/ if the selector fails to collect).
- Validate: pytest tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_post_verify_only_executes_chain -vv | tee "$HUB"/green/pytest_post_verify_only.log (archive any initial failure under `$HUB`/red/ and rerun after fixes).
- Validate: python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py --hub "$HUB" --post-verify-only |& tee "$HUB"/cli/run_phase_g_dense_post_verify_only.log to exercise the new mode against the freshly produced artifacts; confirm `analysis/verification_report.json`, `analysis/verify_dense_stdout.log`, `analysis/check_dense_highlights.log`, and `analysis/artifact_inventory.txt` refresh.
- Document: Update `$HUB/summary/summary.md` with runtime, MS-SSIM ±0.000 / MAE ±0.000000 deltas, preview verdict, CLI/test/log references, and rerun commands; refresh docs/fix_plan.md + galph_memory (TEST-CLI-001, TYPE-PATH-001).

How-To Map
1. export AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md; export HUB=$PWD/plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier
2. In `plans/active/.../bin/run_phase_g_dense.py`, add `--post-verify-only` to argparse, guard against incompatible flags (`--skip-post-verify`, `--clobber`), and branch early: when enabled, skip `prepare_hub` + the Phase C→F command loop, print a banner explaining reused outputs, run `run_command` for the SSIM grid helper, verifier, and highlights checker, then call `generate_artifact_inventory(hub)` and exit. Ensure `--collect-only --post-verify-only` prints only these three commands with hub-relative log/report paths.
3. Update `tests/study/test_phase_g_dense_orchestrator.py`:
   - Extend the existing collect-only assertions (or add a new test) to run `main()` with `--collect-only --post-verify-only` and assert only SSIM grid + verifier + highlights entries are printed with the correct log/report strings.
   - Add a pytest (`test_run_phase_g_dense_post_verify_only_executes_chain`) that monkeypatches `run_command` to capture command tuples, points the hub at a tmp directory, runs `main()` with `--post-verify-only`, and asserts the order `[ssim_grid_cmd, verify_cmd, check_cmd]` plus a single artifact-inventory regeneration.
4. `pytest --collect-only tests/study/test_phase_g_dense_orchestrator.py -k post_verify_only -vv | tee "$HUB"/collect/pytest_collect_orchestrator_post_verify_only.log`; copy any initial failure into `$HUB`/red/.
5. `pytest tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_post_verify_only_executes_chain -vv | tee "$HUB"/green/pytest_post_verify_only.log`; if it fails, archive the RED log before rerunning.
6. `python plans/active/.../bin/run_phase_g_dense.py --hub "$HUB" --dose 1000 --view dense --splits train test --clobber |& tee "$HUB"/cli/run_phase_g_dense_stdout.log`; monitor `analysis/` for `metrics_delta_summary.json`, `metrics_delta_highlights.txt`, `metrics_delta_highlights_preview.txt`, `ssim_grid_summary.md`, `verification_report.json`, `verify_dense_stdout.log`, `check_dense_highlights.log`, `artifact_inventory.txt`.
7. `python plans/active/.../bin/run_phase_g_dense.py --hub "$HUB" --post-verify-only |& tee "$HUB"/cli/run_phase_g_dense_post_verify_only.log` to verify the new mode; if any command fails, capture blocker evidence, fix, and rerun with the same hub (no `--clobber`).
8. (Only if preview drift appears) `python plans/active/.../bin/check_dense_highlights_match.py --hub "$HUB" | tee "$HUB"/analysis/check_dense_highlights_manual.log` to record the guard output separately.
9. Update `$HUB/summary/summary.md` with MS-SSIM/MAE deltas (±0.000 / ±0.000000), preview verdict, CLI/test/log locations, and rerun commands; log the same Turn Summary in `summary.md`.
10. Refresh docs/fix_plan.md Attempts History + galph_memory with artifact pointers; if new tests are added, run doc sync steps from the Doc Sync Plan.

Pitfalls To Avoid
- Do not pair `--post-verify-only` with `--clobber` or `--skip-post-verify`; guard and exit early with actionable messaging.
- Keep all printed/logged paths hub-relative (TYPE-PATH-001) and ensure `generate_artifact_inventory` runs after every verifying pass.
- Never mutate Phase C data during verification-only runs—no calls to `prepare_hub` or data deletion (PHASEC-METADATA-001).
- Capture RED/GREEN/collect logs under `$HUB`/{red,green,collect}/ per TEST-CLI-001 before rerunning any selector.
- Dense pipeline reruns require `--clobber`; forgetting it leaves stale Phase C outputs and blocks `prepare_hub`.
- Export `AUTHORITATIVE_CMDS_DOC` before every orchestrator invocation to keep legacy consumers configured (CONFIG-001).
- If `run_phase_g_dense.py --post-verify-only` surfaces errors, keep the blocker logs and do not delete them before documenting the failure.
- Avoid manual `rm` inside the hub; rely on orchestrator helpers to manage directories (TYPE-PATH-001, DATA-001).

If Blocked
- If post-verify-only implementation hits an unexpected dependency (e.g., missing SSIM grid outputs), capture the stack trace and log path in `$HUB`/analysis/blocker.log, note the failing command in docs/fix_plan.md Attempts History, and roll the block reason + evidence into galph_memory.
- If the counted dense run crashes mid-phase, keep the partial `cli/` logs, document the failure under `$HUB`/analysis/blocker.log, and stop—the next loop should resolve the specific phase blocker before re-running.

Findings Applied (Mandatory)
- POLICY-001 — PyTorch dependencies (Phase F recon + verifier helpers) remain available; keep env intact when adding new flags.
- CONFIG-001 — `update_legacy_dict` ordering preserved by exporting AUTHORITATIVE_CMDS_DOC ahead of every subprocess call.
- DATA-001 — Verification-only runs still enforce canonical NPZ/JSON layout via `verify_dense_pipeline_artifacts.py`.
- TYPE-PATH-001 — Success banners, logs, and inventories must stay hub-relative; artifact inventory refreshed after post-verify sweeps.
- STUDY-001 — Summaries must report MS-SSIM ±0.000 and MAE ±0.000000 deltas per view/dose.
- TEST-CLI-001 — Archive RED/GREEN/collect logs for every new selector and helper command.
- PREVIEW-PHASE-001 — SSIM grid + highlights checker must continue rejecting amplitude pollution even in post-verify-only mode.
- PHASEC-METADATA-001 — Phase C metadata guard stays active; verification-only runs cannot alter Phase C outputs.

Pointers
- docs/findings.md:22-24 — Active guardrails (PHASEC-METADATA-001, TEST-CLI-001, PREVIEW-PHASE-001) that define required evidence.
- docs/TESTING_GUIDE.md:358-370, 418-470 — Phase G verification workflow, CLI log expectations, and AUTHORITATIVE_CMDS_DOC export requirements.
- plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/plan/plan.md — Current execution sketch + acceptance criteria for this hub.
- plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py:840-1370 — Orchestrator implementation that needs the new flag and verification-only branch.
- tests/study/test_phase_g_dense_orchestrator.py:140-520 — Existing collect-only + post-verify hook tests to extend for the new mode.

Next Up (optional)
1. If time remains, run `check_dense_highlights_match.py --hub "$HUB"` independently and capture a GREEN log for the preview guard.
2. Draft the summary updates for docs/TESTING_GUIDE.md + TEST_SUITE_INDEX.md once the new pytest selectors are GREEN.

Doc Sync Plan (Conditional)
- After the new selectors pass, run `pytest --collect-only tests/study/test_phase_g_dense_orchestrator.py -k post_verify_only -vv | tee "$HUB"/collect/pytest_collect_post_verify_only_for_docs.log`, copy the selector details into docs/TESTING_GUIDE.md §Phase G Delta Metrics Persistence and `docs/development/TEST_SUITE_INDEX.md` (add the two new test names + selector usage), then attach the collect-only log under `$HUB`/collect/. Apply updates only after the implementation is GREEN to keep registries authoritative.
