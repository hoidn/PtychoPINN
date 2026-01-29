# FNO-STABILITY-OVERHAUL-001 — Phase 10 ReduceLROnPlateau multi-seed

Plan: `docs/plans/2026-01-29-reduce-lr-plateau.md`

References:
- `plans/active/FNO-STABILITY-OVERHAUL-001/implementation.md` (Phase 10 status + reporting hooks)
- `docs/strategy/mainstrategy.md` (§High-Priority Next Actions — scheduler experiments)
- `docs/findings.md` (STABLE-LS-001 context for documenting outcomes)
- `docs/TESTING_GUIDE.md`
- `docs/specs/spec-ptycho-core.md §Forward Model` (normative physics reference for interpreting val_loss/SSIM)

Summary: Execute Task 5 of the ReduceLROnPlateau plan — extend the Stage A A/B comparison to three seeds per arm. Reuse the canonical Stage A dataset, prep `arm_control_seed{20260128,29,30}` and `arm_plateau_seed{...}` directories, clear stale runs, then loop the compare_wrapper commands for Default vs ReduceLROnPlateau, capturing logs under the new artifacts hub. After each run, dump stats with `scripts/internal/stage_a_dump_stats.py` and aggregate them into `stage_a_plateau_multiseed_summary.json`, highlighting val_loss_best and amp/phase SSIM deltas. Once the runs are archived, update the implementation plan, strategy doc, fix_plan attempts history, and findings (extend STABLE-LS-001 unless plateau succeeds). Keep the mapped pytest selectors green before/after the runs per the plan.
**Summary (goal):** Quantify whether ReduceLROnPlateau improves loss spikes across three seeds relative to the Default scheduler.
**Focus:** FNO-STABILITY-OVERHAUL-001 — Phase 10 ReduceLROnPlateau multi-seed comparison
**Branch:** fno2-phase8-optimizers
**Mapped tests:** `pytest tests/test_grid_lines_compare_wrapper.py::test_wrapper_accepts_plateau_scheduler -v`, `pytest tests/torch/test_model_training.py::test_configure_optimizers_supports_plateau -v`, `pytest tests/scripts/test_training_backend_selector.py::test_torch_scheduler_plateau_roundtrip -v`
**Artifacts:** `plans/active/FNO-STABILITY-OVERHAUL-001/reports/2026-01-29T220000Z/`
**Next Up (optional):** Kick off Phase 9 Crash Hunt depth sweep once plateau verdict is documented.
