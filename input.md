# FNO-STABILITY-OVERHAUL-001 — Phase 9 Crash Hunt (control depth sweep)

Plan: `plans/active/FNO-STABILITY-OVERHAUL-001/plan_crash_hunt.md`

References:
- `docs/strategy/mainstrategy.md` (§3 Crash Hunt & Shootout protocol + Status note)
- `plans/active/FNO-STABILITY-OVERHAUL-001/implementation.md` (Phase 9 tasks)
- `docs/plans/2026-01-30-stable-hybrid-crash-hunt.md`
- `docs/findings.md` (STABLE-LS-001, FNO-DEPTH-001/002)
- `docs/TESTING_GUIDE.md`
- `docs/specs/spec-ptycho-core.md §Forward Model` (normative amplitude/phase + MAE definitions)

**Summary:** Free ≥5 GB, run the 3-depth × 3-seed hybrid Crash Hunt with the canonical Stage A dataset (`--set-phi`, capped channels), log stats, and sync docs/findings with the measured crash depth.
**Summary (goal):** Produce `crash_hunt_summary.{json,md}` plus per-run logs/stats proving the shallowest depth where P_crash > 0.

**Focus:** FNO-STABILITY-OVERHAUL-001 — Phase 9 Crash Hunt (control depth sweep)
**Branch:** fno2-phase8-optimizers
**Mapped tests:** `pytest tests/test_grid_lines_compare_wrapper.py::test_wrapper_handles_stable_hybrid -v`, `pytest tests/test_grid_lines_compare_wrapper.py::test_wrapper_passes_max_hidden_channels -v`
**Artifacts:** `plans/active/FNO-STABILITY-OVERHAUL-001/reports/2026-01-29T085840Z/`
**Next Up (optional):** Stage Shootout arm directories once crash depth is documented.

---

## Do Now — plan_crash_hunt.md Task 1 (Crash Hunt depth sweep)

1. **Disk headroom:** Run `df -h .` to confirm `/` usage. Capture `du -sh outputs/fno_hyperparam_study_e10` (≈18 GB), then delete it (`rm -rf outputs/fno_hyperparam_study_e10`) or an equivalent-sized archive so at least 5 GB is free. Re-run `df -h .` afterward and drop both before/after outputs + rationale into the artifacts hub.
2. **Warm-up selectors:** In repo root, run `pytest tests/test_grid_lines_compare_wrapper.py::test_wrapper_handles_stable_hybrid -v` and `pytest tests/test_grid_lines_compare_wrapper.py::test_wrapper_passes_max_hidden_channels -v`; store the combined log as `plans/active/FNO-STABILITY-OVERHAUL-001/reports/2026-02-01T000000Z/pytest_warmup.log` (append if reusing the file). Stop here if either test fails.
3. **Dataset sanity check:** Verify each `outputs/grid_lines_crash_hunt/depth{4,6,8}_seed{A,B,C}/datasets` symlink still points to `outputs/grid_lines_stage_a/arm_control/datasets` and that the Stage A cache contains the `--set-phi` metadata (spot-check `metadata.json`). Update the artifacts README if anything changed.
4. **Command template:** For every `{depth ∈ {4,6,8}, seed ∈ {20260128,20260129,20260130}}`, run:
   ```bash
   python scripts/studies/grid_lines_compare_wrapper.py \
     --N 64 --gridsize 1 --set-phi \
     --output-dir outputs/grid_lines_crash_hunt/depth${depth}_seed${seed} \
     --architectures hybrid --seed ${seed} \
     --nimgs-train 1 --nimgs-test 1 --nphotons 1e9 \
     --nepochs 30 --torch-epochs 30 \
     --fno-blocks ${depth} --torch-max-hidden-channels 512 \
     --torch-grad-clip 1.0 --torch-grad-clip-algorithm norm \
     --torch-log-grad-norm --torch-grad-norm-log-freq 1 \
     --torch-loss-mode mae --torch-infer-batch-size 8 \
     2>&1 | tee plans/active/FNO-STABILITY-OVERHAUL-001/reports/2026-02-01T000000Z/depth${depth}_seed${seed}.log
   ```
   - Depth 8 may still OOM; capture the full traceback and note it in the README.
   - If a run crashes early, keep the log + partial artifacts for diagnosis.
5. **Stats capture:** After each successful run (or controlled failure), run `python scripts/internal/stage_a_dump_stats.py --run-dir outputs/grid_lines_crash_hunt/depth${depth}_seed${seed}/runs/pinn_hybrid --out-json plans/active/FNO-STABILITY-OVERHAUL-001/reports/2026-02-01T000000Z/depth${depth}_seed${seed}_stats.json`. For OOM/NaN runs that never produced metrics, write a stub JSON with `"crashed": true` and the reason.
6. **Crash summary:** Execute the Python aggregation snippet from `plan_crash_hunt.md` to build `crash_hunt_summary.json` and companion `crash_hunt_summary.md`, computing `crashed = (amp_ssim < 0.5) or (final_val_loss > 0.15) or log indicates NaN/OOM`. Highlight the first depth where any seed flagged as crashed, and tabulate P_crash per depth (3 seeds each).
7. **Docs & plans:**
   - Update `plans/active/FNO-STABILITY-OVERHAUL-001/implementation.md` Phase 9 with Crash Hunt findings.
   - Add a findings ledger entry (e.g., STABLE-CRASH-DEPTH-001) to `docs/findings.md` summarizing P_crash + heuristics.
   - Refresh `docs/strategy/mainstrategy.md §3` with the observed crash depth and note whether Shootout should run at depth 6 or 8.
   - Append an attempts-history bullet and FSM line update to `docs/fix_plan.md` referencing the artifacts path.
8. **Hub upkeep:** Ensure `plans/active/FNO-STABILITY-OVERHAUL-001/reports/2026-02-01T000000Z/README.md` lists the free-space remediation, depth/seed matrix completion status, and any OOM annotations. Place stats, logs, and summaries there; keep raw model checkpoints in `outputs/...` (git-ignored).
9. **Shootout staging:** If Crash Hunt identifies `DEPTH_CRASH`, mkdir `outputs/grid_lines_shootout/{control,stable_layerscale,optimizer}/seed{A,B,C}` with Stage A dataset symlinks so Task 2 can start immediately next loop. Note this readiness in the README even if runs are not started.
10. **Git hygiene:** Do not commit large logs or NPZs. Stage only doc/plan/test edits needed to record Crash Hunt evidence. Follow `docs/TESTING_GUIDE.md` for log archival naming when running additional selectors.
