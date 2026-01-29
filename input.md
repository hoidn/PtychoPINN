# FNO-STABILITY-OVERHAUL-001 — Phase 9 Crash Hunt (Control Depth Sweep)

Plan: `plans/active/FNO-STABILITY-OVERHAUL-001/plan_crash_hunt.md`

References:
- `docs/strategy/mainstrategy.md` (§3 Crash Hunt & Shootout protocol)
- `plans/active/FNO-STABILITY-OVERHAUL-001/implementation.md` (Phase 9 section)
- `docs/plans/2026-01-30-stable-hybrid-crash-hunt.md`
- `docs/findings.md` (STABLE-LS-001, FNO-DEPTH-001/002)
- `docs/TESTING_GUIDE.md`
- `docs/specs/spec-ptycho-core.md §Forward Model` (normative loss/metric contract)

**Summary:** Execute the Crash Hunt depth sweep (control hybrid arm) to measure stochastic stability across depths 4/6/8 using the canonical Stage A dataset with `--set-phi`, capped channels, and multi-seed runs, then aggregate crash statistics for Phase 9.
**Summary (goal):** Produce crash_hunt_summary.{json,md} + per-run stats/logs proving the shallowest depth with instability, per docs/strategy/mainstrategy.md §3.

**Focus:** FNO-STABILITY-OVERHAUL-001 — Phase 9 Crash Hunt (control depth sweep)
**Branch:** fno2-phase8-optimizers
**Mapped tests:** `pytest tests/test_grid_lines_compare_wrapper.py::test_wrapper_handles_hybrid -q`
**Artifacts:** `plans/active/FNO-STABILITY-OVERHAUL-001/reports/2026-02-01T000000Z/`
**Next Up (optional):** Prep Shootout arm configs once crash depth is confirmed.

---

## Do Now — plan_crash_hunt.md Task 1 (Crash Hunt depth sweep)

1. **Warm-up test:** Run `pytest tests/test_grid_lines_compare_wrapper.py::test_wrapper_handles_hybrid -q` to ensure CLI plumbing is green before launching long jobs. Archive the pytest log under the artifacts hub. If it fails, stop and investigate before running any training.
2. **Artifacts hub prep:** Inside `plans/active/FNO-STABILITY-OVERHAUL-001/reports/2026-02-01T000000Z/`, add the README scaffold from Step 0 of the plan (stage dataset provenance, depth/seed list, logging conventions). This README must mention the enforced `--set-phi` requirement plus the crash/SSIM heuristics you will use later.
3. **Dataset sync:** For each depth ∈ {4,6,8} and seed slot ∈ {seedA, seedB, seedC}, `rsync -a outputs/grid_lines_stage_a/arm_control/datasets/` into `outputs/grid_lines_crash_hunt/depth${depth}_${slot}/datasets/`. This guarantees every run consumes the identical Stage A cache (N=64, gridsize=1, nimgs=1, nphotons=1e9, MAE loss).
4. **Seed mapping:** Use seeds 20260128, 20260129, 20260130 (map to seedA/B/C respectively) and record the mapping in the artifacts README before launching runs.
5. **Run commands:** For each depth/seed combination, execute the control-hybrid command from the plan with:
   - `--set-phi` (non-negotiable for phase metrics)
   - `--fno-blocks {depth}`
   - `--torch-max-hidden-channels 512` (guards VRAM per FNO-DEPTH-001/002)
   - `--torch-grad-clip 1.0 --torch-grad-clip-algorithm norm`
   - `--torch-log-grad-norm --torch-grad-norm-log-freq 1`
   - `--nepochs 30 --torch-epochs 30`
   - Redirect stdout/stderr to `plans/active/FNO-STABILITY-OVERHAUL-001/reports/2026-02-01T000000Z/depth${depth}_seed${seed}.log`.
   Note any OOMs or NaNs inline in the log (and resumable instructions if needed). Depth 8 may still OOM; capture the failure reason verbatim if so.
6. **Stats capture:** After each run, call `scripts/internal/stage_a_dump_stats.py --run-dir .../runs/pinn_hybrid --out-json .../depth${depth}_seed${seed}_stats.json`. This records best/final losses, amp/phase SSIM, MAE, grad norms, etc.
7. **Crash annotation:** Once all stats exist, run the provided Python snippet from the plan to assemble `crash_hunt_summary.json` (and companion `crash_hunt_summary.md`). Use the crash heuristic `(amp_ssim < 0.5) or (final_val_loss > 0.15) or log shows NaN/OOM`. Explicitly call out the first depth showing any crash and the observed failure signatures.
8. **Docs & plan sync:**
   - Update `plans/active/FNO-STABILITY-OVERHAUL-001/implementation.md` Phase 9 status with the Crash Hunt findings.
   - Append a new turn entry to `plans/active/FNO-STABILITY-OVERHAUL-001/summary.md` summarizing the Crash Hunt outcomes.
   - Update `docs/strategy/mainstrategy.md §3` with the measured crash depth/P_crash.
   - Extend `docs/findings.md` with a new finding (e.g., `STABLE-CRASH-DEPTH-001`) if the crash depth result is durable.
   - Expand `docs/fix_plan.md` attempts history + FSM block with the Crash Hunt evidence (reference the artifacts hub path).
9. **Next step setup:** Once Crash Hunt is documented, leave the Shootout directories unrun but created (per plan Task 2 Step 1) so the next pass can start immediately. Note this readiness in the artifacts README.
10. **Git hygiene:** Do not commit training logs or outputs; keep them under `plans/.../reports/` or `outputs/` with `.gitignore`. Stage only source/docs/test updates when ready (no commit in this supervisor loop unless requested).

> Metrics/physics reminder: When interpreting amp/phase SSIM or MAE, defer to `docs/specs/spec-ptycho-core.md §Forward Model` for the authoritative definition of amplitude/phase tensors and loss calculations.
