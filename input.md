# FNO-STABILITY-OVERHAUL-001 — Phase 10 ReduceLROnPlateau A/B

Plan: `docs/plans/2026-01-29-reduce-lr-plateau.md`

References:
- `plans/active/FNO-STABILITY-OVERHAUL-001/implementation.md`
- `docs/strategy/mainstrategy.md` (§High-Priority Next Actions)
- `docs/findings.md` (STABLE-LS-001, FNO-DEPTH-001/002)
- `docs/TESTING_GUIDE.md`
- `docs/specs/spec-ptycho-core.md §Forward Model`

Summary: Validate the ReduceLROnPlateau plumbing (config literals, CLI choices, Lightning branch) with the targeted pytest selectors, tighten `tests/torch/test_model_training.py::test_configure_optimizers_supports_plateau` so it exercises `PtychoPINN_Lightning.configure_optimizers`, then run the Stage A A/B comparison (Default vs ReduceLROnPlateau, identical dataset/seed, 20 epochs, `--set-phi`, fno_blocks=4, nimgs_train/test=1) and archive logs/stats under the artifacts hub. Capture best val_loss + amp/phase SSIM via the plan’s helper script, and update the implementation plan + docs/strategy with whether plateau reduces the late-epoch spikes; add a `docs/findings.md` entry if the outcome is durable.
**Summary (goal):** Determine if ReduceLROnPlateau suppresses Stage A loss spikes without degrading final metrics.
**Focus:** FNO-STABILITY-OVERHAUL-001 — Phase 10 ReduceLROnPlateau scheduler test
**Branch:** fno2-phase8-optimizers
**Mapped tests:** `pytest tests/test_grid_lines_compare_wrapper.py::test_wrapper_accepts_plateau_scheduler -v`, `pytest tests/torch/test_model_training.py::test_configure_optimizers_supports_plateau -v`, `pytest tests/scripts/test_training_backend_selector.py::test_torch_scheduler_plateau_roundtrip -v`
**Artifacts:** `plans/active/FNO-STABILITY-OVERHAUL-001/reports/2026-01-29T163000Z/`
**Next Up (optional):** Resume Phase 9 Crash Hunt once the scheduler experiment is documented.
