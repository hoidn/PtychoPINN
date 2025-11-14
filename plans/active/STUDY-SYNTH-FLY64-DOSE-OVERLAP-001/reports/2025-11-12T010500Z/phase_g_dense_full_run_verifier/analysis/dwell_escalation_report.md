# Dwell Escalation Report — STUDY-SYNTH-FLY64-DOSE-OVERLAP-001

## Trigger
- Dwell counter hit Tier 3 (6 consecutive supervisor planning loops) without any new `{analysis}` deliverables or Ralph execution evidence for the Phase G dense rerun hub (`plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/`).
- Input briefs on 2025-11-11 through 2025-11-13 repeatedly handed off the same counted rerun + post-verify instructions (pytest guard + `run_phase_g_dense.py --clobber` + fully parameterized `--post-verify-only` + metrics bundle publication), but no CLI logs or metrics files have been produced since the failure recorded on 2025-11-11.

## Evidence Snapshot (2025-11-13T00:05Z audit)
- `{analysis}` only contains `blocker.log`; no SSIM grid, verification, highlights, metrics, preview, or artifact inventory artifacts exist anywhere under this hub.
- `cli/phase_d_dense.log` still ends with `TypeError: len() of unsized object` thrown by `studies/fly64_dose_overlap/overlap.py:194` when `filter_dataset_by_mask` receives scalar metadata (`len()` on unsized object) while running **inside `/home/ollie/Documents/PtychoPINN`**.
- `cli/run_phase_g_dense_post_verify_only.log` is only the argparse usage banner because the helper was invoked without the mandatory `--dose/--view/--splits` arguments.
- `data/phase_c/run_manifest.json` remains deleted (tracked by git), so even the Phase C staging artifacts were never regenerated locally after the original run failed.
- Git log shows no `RALPH` commits since `54e60337` (geometry acceptance test fix), confirming the engineer has not executed any new evidence loops for this focus since the last successful attempt.

## Attempts Summary
1. **2025-11-11T19:55Z** — Supervisor restated counted rerun instructions after verifying zero `{analysis}` artifacts and the lingering `allow_pickle=False` failure inside `overlap.py`.
2. **2025-11-11T20:27Z** — Reissued same Do Now, clarified Phase C manifest deletion and metrics expectations; no new outputs appeared.
3. **2025-11-12T00:48Z** — Verified regression test existed/passed, launched dense pipeline (PID f6af92) but no completion evidence or post-verify logs landed.
4. **2025-11-12T01:29Z & 05:12Z** — Multiple audits confirmed `{analysis}` still empty, `run_phase_g_dense_post_verify_only.log` equals usage error, and the counted rerun never finished.
5. **2025-11-13T00:05Z** — Sixth planning loop: situation unchanged; dwell now Tier 3.

## Blocking Condition
- The dense pipeline cannot progress until Ralph executes the ready-for-implementation Do Now: fix `filter_dataset_by_mask` TypeError (already reconfirmed as pending locally), add/green `pytest tests/study/test_dose_overlap_overlap.py::test_filter_dataset_by_mask_handles_scalar_metadata`, rerun `test_generate_overlap_views_dense_acceptance_floor`, execute `run_phase_g_dense.py --clobber` followed by the fully parameterized `--post-verify-only` command, and publish the SSIM/verification/highlights/preview/metrics/inventory bundle with MS-SSIM ±0.000 / MAE ±0.000000 deltas.

## Required Actions Before Unblock
1. Engineering owner must acknowledge the dwell escalation and run the counted Phase C→G pipeline locally from `/home/ollie/Documents/PtychoPINN` with the guard pytest selectors.
2. Capture fresh CLI logs under `$HUB/cli/`, populate `{analysis}` with the required artifacts, and record MS-SSIM/MAE deltas + preview verdict in `summary.md`, `summary/summary.md`, docs/fix_plan.md, and galph_memory.
3. If the pipeline fails again, append the command + exit code and minimal stack trace under `$HUB/red/blocked_<timestamp>.md` before handing off.
4. Only after new evidence lands should this focus exit the `blocked_escalation` state and re-enter the Supervisor/Engineer loop.

## Escalation Owner
- Supervisor: Galph (planning agent)
- Engineering owner: Ralph (implementation agent)
- Hub: `plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/`

## Update — 2025-11-13T091500Z
- Dwell climbed back to Tier 3 (8 consecutive supervisor loops) because no new `{analysis}` artifacts or Ralph commits have landed since the 2025-11-13T051200Z audit; `analysis/` still contains only `blocker.log` and this report.
- `git status --porcelain` lists only hub evidence files, and `git log --all --oneline --grep 'RALPH' -n 5` still tops out at `b6cd7e4f` (2025-11-11), confirming engineering has not rerun the counted dense pipeline.
- `cli/phase_d_dense.log` continues to show the argparse usage error (`overlap.py: error: the following arguments are required: --gridsize, --s-img, --n-groups`), meaning the helper never forwarded the sampling parameters that Phase D requires. `cli/run_phase_g_dense_post_verify_only.log` remains the same usage banner, so the verifier chain has still never executed under this repo.
- Until Ralph executes the ready_for_implementation brief (pytest scalar-mask + acceptance selectors, counted `run_phase_g_dense.py --clobber`, immediately followed by the fully parameterized `--post-verify-only`, then `report_phase_g_dense_metrics.py`/`analyze_dense_metrics.py` and publication of the SSIM/verification/highlights/preview/metrics bundle), this focus stays in `blocked_escalation`. Any further failures must be logged under `$HUB/red/blocked_<timestamp>.md` with the exact command and exit code before another supervisor hand-off occurs.

## Update — 2025-11-16T110500Z
- Dwell hit Tier 3 again (six consecutive supervisor planning loops) after the background compare_models job completed without the required chunk flags; `analysis/dose_1000/dense/train_debug_v3/logs/logs/debug.log:347-365` still shows “Using single-shot PINN inference: 5088 groups (batch_size=32)” so the refactored chunk helper never ran.
- Baseline metrics remain blank (`analysis/dose_1000/dense/test/comparison_metrics.csv:2-13` only lists PtychoPINN rows, `analysis/metrics_summary.json` is still missing) and the verification bundle is unchanged (`analysis/verification_report.json` reports `n_valid=0/10` and every check is “File not found”).
- `cli/run_phase_g_dense_post_verify_only.log:1-22` continues to fail PREVIEW-PHASE-001 because `analysis/metrics_delta_highlights_preview.txt` was never generated, so the SSIM grid step aborts immediately.
- `git log --all --oneline -n 1 --grep 'actor=ralph'` still points to `[SYNC i=355] actor=ralph … ralph_commit=cdfaf857` (2025-11-14), confirming no new Ralph evidence has landed inside this repo since the translation guard log + blocker upload.
- Required unblock actions are unchanged: rerun the guarded translation selector, execute the chunked debug + full compare_models commands with the explicit `--n-test-groups/--pinn-chunk-size/--baseline-chunk-size` flags until Baseline DIAGNOSTIC rows are non-zero, then immediately drive the Phase D selectors, counted `run_phase_g_dense.py --dose 1000 --view dense --splits train test --clobber`, metrics helpers, and fully parameterized `--post-verify-only` command so `{analysis}` contains the SSIM grid, metrics summaries (JSON/MD/delta/highlights/preview), verification_report.json=10/10, and artifact_inventory.txt. Any further failures must produce `$HUB/red/blocked_<timestamp>.md` entries before another supervisor loop engages.
