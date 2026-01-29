# FNO-STABILITY-OVERHAUL-001 — Phase 3: Stage A Shootout

**Summary:** Run the grid-lines Stage A shootout (control vs stable_hybrid vs AGC) using the shared dataset workflow from `plans/active/FNO-STABILITY-OVERHAUL-001/implementation.md` Phase 3.

**Focus:** FNO-STABILITY-OVERHAUL-001 — Phase 3 (Stage A Validation)

**Branch:** fno2

**Mapped tests:**
- `pytest tests/torch/test_fno_generators.py -k stable -v`
- `pytest tests/test_grid_lines_compare_wrapper.py::test_wrapper_handles_stable_hybrid -v`
- `pytest tests/torch/test_grid_lines_torch_runner.py::TestChannelGridsizeAlignment::test_runner_accepts_stable_hybrid -v`
- `pytest tests/test_grid_lines_compare_wrapper.py::test_wrapper_passes_grad_clip_algorithm -v`

**Artifacts:** `plans/active/FNO-STABILITY-OVERHAUL-001/reports/2026-01-29T010000Z/`

**Next Up (optional):** Stage B deep-depth validation after Stage A metrics confirm the winning arm.

---

## Do Now

Execute Phase 3 Tasks 3.1–3.5:
1. Prepare the shared dataset directories under `outputs/grid_lines_stage_a/{arm_control,arm_stable,arm_agc}` and copy the `datasets/` tree from control to the other arms once generated.
2. Run the three CLI commands (control hybrid, stable_hybrid, hybrid+AGC) exactly as listed in the plan, capturing each stdout/stderr log into the artifacts hub.
3. After each run, archive `metrics.json` plus `runs/pinn_<arch>/{history.json,metrics.json[,model.pt]}` under the appropriate arm directory.
4. Use the provided Python snippet to aggregate `val_loss` and `ssim_phase` across arms, writing `stage_a_metrics.json` + a narrative `stage_a_summary.md` (table + Stage B recommendation) under the artifacts hub.
5. Update `plans/active/FNO-STABILITY-OVERHAUL-001/implementation.md` + `docs/fix_plan.md` with the Stage A outcome if time permits.
