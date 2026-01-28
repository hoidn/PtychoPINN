# FNO-STABILITY-OVERHAUL-001 — Phase 2: Stable Hybrid Generator

**Summary:** Implement the Phase 2 architectural fix from `docs/strategy/mainstrategy.md §1.A` by adding the StablePtychoBlock + `stable_hybrid` generator path and exposing it through the Torch runner + compare harness.

**Focus:** FNO-STABILITY-OVERHAUL-001 — Phase 2 (Stable hybrid generator)

**Branch:** fno2

**Mapped tests:**
- `pytest tests/torch/test_fno_generators.py -k stable -v`
- `pytest tests/test_grid_lines_compare_wrapper.py::test_wrapper_handles_stable_hybrid -v`
- `pytest tests/torch/test_grid_lines_torch_runner.py::TestChannelGridsizeAlignment::test_runner_accepts_stable_hybrid -v`

**Artifacts:** `plans/active/FNO-STABILITY-OVERHAUL-001/reports/2026-01-28T050000Z/`

**Next Up (optional):** Phase 3 Stage A shootout once `stable_hybrid` + AGC paths are wired.

---

## Do Now

Work through Phase 2 Tasks 2.1–2.3 in `plans/active/FNO-STABILITY-OVERHAUL-001/implementation.md`, keeping the new instructions from the writing-plans update. Capture the pytest logs for each mapped selector under the artifacts path above (see `docs/TESTING_GUIDE.md` on naming). Reference `docs/strategy/mainstrategy.md §1.A` + `docs/workflows/pytorch.md` for architectural contracts.

### Task 1 — Implement StablePtychoBlock + parametrized Hybrid
1. In `ptycho_torch/generators/fno.py` add `StablePtychoBlock` next to `PtychoBlock` (same spectral + local branches) but wrap the GELU sum with `nn.InstanceNorm2d(channels, affine=True, eps≈1e-5)` and zero-init gamma/beta so the residual is identity before training. This block must retain the `(B,C,H,W)` contract.
2. Update `HybridUNOGenerator.__init__` to accept `block_cls=PtychoBlock` and instantiate encoder/bottleneck blocks via that callable so subclasses can inject the stable block without copy/paste. Default behaviour must stay identical for `'hybrid'`.
3. Add `TestStablePtychoBlock` to `tests/torch/test_fno_generators.py`:
   - `test_identity_init` (block(x) ≈ x with zero gamma).
   - `test_zero_mean_update` (set `block.norm.weight.data.fill_(1.0)` and assert `(block(x)-x).mean(dim=(2,3))` ≈ 0).
   - Shapes consistent with existing tests.

### Task 2 — Expose `stable_hybrid` generator
1. Add `StableHybridUNOGenerator` (calls `super().__init__(..., block_cls=StablePtychoBlock)`) and `StableHybridGenerator` to `ptycho_torch/generators/fno.py`.
2. Register `'stable_hybrid'` in `ptycho_torch/generators/registry.py`, mirroring the existing `HybridGenerator`.
3. Extend the `architecture` `Literal[...]` in both `ptycho/config/config.py::ModelConfig` and `ptycho_torch/config_params.py::ModelConfig` to include `'stable_hybrid'`. Update any helper casting (e.g., `scripts/studies/grid_lines_torch_runner.py::setup_torch_configs`) so type hints accept the new value.
4. Update `docs/workflows/pytorch.md` (architecture section + CLI recap) to mention the stable hybrid option (cite `docs/strategy/mainstrategy.md §1.A`).
5. Extend `tests/torch/test_fno_generators.py` to cover the new generator: add `test_stable_hybrid_generator_output_shape` and update registry tests so `resolve_generator` accepts `'stable_hybrid'`.

### Task 3 — Wire `stable_hybrid` through CLI + compare harness
1. Allow `--architecture stable_hybrid` via `scripts/studies/grid_lines_torch_runner.py` (argparse choices, `TorchRunnerConfig` comments, and the literal cast inside `setup_torch_configs`). Ensure metrics key naming uses `pinn_stable_hybrid`.
2. Update `scripts/studies/grid_lines_compare_wrapper.py` so `architectures` containing `'stable_hybrid'` invoke the Torch runner and append `'pinn_stable_hybrid'` to the visuals ordering/metrics dict.
3. Extend `tests/test_grid_lines_compare_wrapper.py` with `test_wrapper_handles_stable_hybrid` (fake runners + asserts merged metrics include the new key and parse_args accepts the string). Add a quick `tests/torch/test_grid_lines_torch_runner.py::TestChannelGridsizeAlignment::test_runner_accepts_stable_hybrid` (or similar) that exercises `setup_torch_configs`.

### Task 4 — Verification
Run the mapped selectors (in order) and archive each log under `plans/active/FNO-STABILITY-OVERHAUL-001/reports/2026-01-28T050000Z/` per `docs/TESTING_GUIDE.md`.
