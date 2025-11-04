# Phase C Regression — n_images missing in simulation plan (2025-11-07T09:05:00Z)

## Objective
Restore Phase C dataset generation by ensuring the simulation plan populates the legacy `TrainingConfig.n_images` attribute (in addition to `n_groups`) so the simulation backend knows how many diffraction patterns to synthesize.

## Checklist
- [ ] Author failing regression test in `tests/study/test_dose_overlap_generation.py` that asserts `TrainingConfig.n_images` is set to the base dataset length before running the pipeline.
- [ ] Run targeted pytest selector to confirm the new test fails (expect `AttributeError`/`None` usage leading to scalar length bug message).
- [ ] Update `build_simulation_plan` in `studies/fly64_dose_overlap/generation.py` to assign `n_images` alongside `n_groups` (consider explicit `object.__setattr__` to avoid dataclass warnings) and normalize via int.
- [ ] Rerun targeted pytest selector to confirm GREEN.
- [ ] Rerun initiative orchestration script in `--collect-only` first to ensure command list intact, then full execution to hit Phase C simulation and verify no blocker.
- [ ] Capture updated CLI log + blocker resolution summary under `reports/2025-11-07T090500Z/phase_c_generation_fix/` and update ledger/memory.

## Command Guard (AUTHORITATIVE_CMDS_DOC)
Set `AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md` before running pytest or orchestration commands.

## Notes
- Regression observed while executing `bin/run_phase_g_dense.py` (Phase C command failed with `TypeError: object of type 'float' has no len()` at `ptycho/raw_data.py:227`).
- Root cause: `TrainingConfig` constructed in `build_simulation_plan` never sets `n_images`, so the legacy simulator sees `config.n_images is None` and synthesizes a scalar coordinate instead of vector.
- Fix scope limited to Phase C helpers + tests; do not touch stable core modules outside this pathway.
- Ensure updated tests document the regression message so future refactors keep `n_images` compatibility until simulator is modernized.
