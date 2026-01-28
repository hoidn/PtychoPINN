# FNO-STABILITY-OVERHAUL-001: Implementation Plan

**Strategy:** `docs/strategy/mainstrategy.md`
**Created:** 2026-01-28

---

## Phase 1: Foundation (Config & Utilities)

### Task 1.1: Add `gradient_clip_algorithm` config field

**Files:**
- `ptycho/config/config.py` — TF `TrainingConfig`: add `gradient_clip_algorithm: Literal['norm', 'value', 'agc'] = 'norm'` after `gradient_clip_val` (line ~232)
- `ptycho_torch/config_params.py` — Torch `TrainingConfig`: add same field after `gradient_clip_val` (line ~131)
- `ptycho_torch/config_bridge.py` — Bridge the field if needed (check if auto-bridged by name match)

**Contract:** `gradient_clip_algorithm` selects the clipping method. Default `'norm'` preserves current behavior.

**Status 2026-01-28:** COMPLETE — TF `TrainingConfig` now exposes the field, `config_bridge.to_training_config()` threads it through to the TF dataclass + `params.cfg`, and `tests/torch/test_config_bridge.py::TestConfigBridgeParity::test_training_config_gradient_clip_algorithm_roundtrip` proves the bridge + `update_legacy_dict` path.

### Task 1.2: Implement AGC utility

**File:** `ptycho_torch/train_utils.py`

Add `adaptive_gradient_clip_(parameters, clip_factor=0.01, eps=1e-3)`:
- Compute unit-wise gradient-to-parameter norm ratio: `||G_i|| / max(||W_i||, eps)`
- Clip gradients where ratio exceeds `clip_factor`: scale `G_i` down to `clip_factor * ||W_i|| / ||G_i||`
- Operate in-place on `.grad` tensors
- Reference: Brock et al., "High-Performance Large-Scale Image Recognition Without Normalization" (2021), Algorithm 2

### Task 1.3: Update training_step dispatch

**File:** `ptycho_torch/model.py` — `PtychoPINN_Lightning.training_step` (line 1334-1340)

Replace the hardcoded `clip_grad_norm_` block with dispatch:
```python
algo = self.training_config.gradient_clip_algorithm
if self.gradient_clip_val is not None and self.gradient_clip_val > 0:
    if algo == 'norm':
        torch.nn.utils.clip_grad_norm_(self.parameters(), self.gradient_clip_val)
    elif algo == 'value':
        torch.nn.utils.clip_grad_value_(self.parameters(), self.gradient_clip_val)
    elif algo == 'agc':
        adaptive_gradient_clip_(self.parameters(), clip_factor=self.gradient_clip_val)
```

### Task 1.4: Update CLI flags

**File:** `scripts/studies/grid_lines_torch_runner.py`
- Add `--gradient-clip-algorithm` argument (choices: norm, value, agc; default: norm)
- Pass to `TorchRunnerConfig` and through to `TrainingConfig`

**File:** `scripts/studies/grid_lines_compare_wrapper.py`
- Forward the flag to torch runner invocations

**Status 2026-01-28:** COMPLETE — `scripts/studies/grid_lines_compare_wrapper.py` now threads `--torch-grad-clip-algorithm` through to `TorchRunnerConfig`, and `tests/test_grid_lines_compare_wrapper.py::test_wrapper_passes_grad_clip_algorithm` exercises the end-to-end flag propagation.

---

## Phase 2: Generator Engineering

### Task 2.1: Implement StablePtychoBlock

**Files:**
- Modify `ptycho_torch/generators/fno.py` right after `PtychoBlock`.
- Extend `HybridUNOGenerator` ctor to accept a `block_cls` (default `PtychoBlock`) so downstream subclasses can swap the block without copying code.
- Tests live in `tests/torch/test_fno_generators.py`.

**Steps:**
1. **Add block implementation.** Introduce `StablePtychoBlock(channels, modes=12)` with the residual form `x + InstanceNorm(GELU(SpectralConv(x) + Conv3x3(x)))`. Use the same spectral + local conv branches as `PtychoBlock`, add `nn.InstanceNorm2d(channels, affine=True, eps=1e-5)` and zero-initialize `weight`/`bias` so the block is identity pre-training.
2. **Parameterize HybridUNOGenerator.** Update the Hybrid constructor so encoder blocks and bottleneck are instantiated via the injected `block_cls`. Default behaviour must stay unchanged for `'hybrid'`.
3. **Add TDD coverage.** Extend `tests/torch/test_fno_generators.py` with `TestStablePtychoBlock`:
   - `test_identity_init` — feed random tensor and assert `torch.allclose(block(x), x, atol=1e-6)`.
   - `test_zero_mean_update` — set `block.norm.weight.data.fill_(1.0)` and verify `(block(x) - x).mean(dim=(2,3))` is ≈0.
   - Keep shapes consistent with existing tests.

### Task 2.2: Implement StableHybridGenerator

**Files:**
- `ptycho_torch/generators/fno.py`
- `ptycho_torch/generators/registry.py`
- Config dataclasses: `ptycho/config/config.py` (`ModelConfig.architecture`) + `ptycho_torch/config_params.py`.
- Docs: `docs/workflows/pytorch.md` (architecture list).
- Tests: `tests/torch/test_fno_generators.py`.

**Steps:**
1. **Generator subclass.** Add `StableHybridUNOGenerator` that simply calls `super().__init__(..., block_cls=StablePtychoBlock)` so it reuses the parametrized Hybrid base.
2. **Registry + adapter.** Create `StableHybridGenerator` (mirrors `HybridGenerator` but instantiates `StableHybridUNOGenerator`) and register it under `'stable_hybrid'` in `ptycho_torch/generators/registry.py`.
3. **Config surface.** Allow the new architecture everywhere:
   - Extend the `Literal[...]` list for `ModelConfig.architecture` (both TF + Torch dataclasses) to include `'stable_hybrid'`.
   - Update any validation/usage sites (e.g., `scripts/studies/grid_lines_torch_runner.py` casting) so the literal type accepts the new string.
4. **Docs.** Update `docs/workflows/pytorch.md` §3 to mention `'stable_hybrid'` (Norm-Last residual with zero-mean updates) referencing `docs/strategy/mainstrategy.md §1.A`.
5. **TDD.** Expand `tests/torch/test_fno_generators.py`:
   - Add `test_stable_hybrid_generator_output_shape` (instantiates `StableHybridUNOGenerator`, asserts `(B, H, W, C, 2)` output).
   - Update registry tests to cover `'stable_hybrid'`.

### Task 2.3: Wire `stable_hybrid` through CLI + compare harness

**Files:**
- `scripts/studies/grid_lines_torch_runner.py`
- `scripts/studies/grid_lines_compare_wrapper.py`
- `tests/test_grid_lines_compare_wrapper.py`
- `tests/torch/test_grid_lines_torch_runner.py`

**Steps:**
1. **Runner CLI + config.** Allow `--architecture stable_hybrid` by updating argparse choices, `TorchRunnerConfig` docstring, and the literal cast inside `setup_torch_configs`. Ensure metrics/reporting use `pinn_stable_hybrid` naming consistently.
2. **Compare wrapper.** Update `run_grid_lines_compare` to treat `'stable_hybrid'` exactly like `'hybrid'` when invoking the Torch runner, and append `'pinn_stable_hybrid'` to the `order` tuple so visuals land in the merge.
3. **Tests.** Extend `tests/test_grid_lines_compare_wrapper.py` with `test_wrapper_handles_stable_hybrid` that injects a fake torch runner and ensures the merged metrics include the new key + parse_args accepts the value. Add a simple `tests/torch/test_grid_lines_torch_runner.py` assertion proving `setup_torch_configs` propagates `'stable_hybrid'` into the training config.
4. **Docs.** Mention the new CLI option in `docs/workflows/pytorch.md` (Torch runner recap) when you touch the doc for Task 2.2.

**Status 2026-01-28:** Pending Phase 2 implementation.

---

## Phase 3: Validation (Stage A Shootout)

Deferred to post-implementation. Uses `grid_lines_compare_wrapper.py` with 3 arms:
1. Control: `hybrid` + `norm` clip (1.0)
2. Arch Fix: `stable_hybrid` + no clip (0.0)
3. Opt Fix: `hybrid` + `agc` (0.01)

---

## Test Strategy

### Unit Tests (Phase 1)
- `tests/torch/test_agc.py`:
  - `test_agc_clips_large_gradients` — verify gradients are scaled down when ratio exceeds threshold
  - `test_agc_preserves_small_gradients` — verify well-behaved gradients are untouched
  - `test_agc_handles_zero_params` — verify eps guard works
- `tests/torch/test_grid_lines_torch_runner.py`:
  - Add test for `--gradient-clip-algorithm` CLI argument parsing

### Unit Tests (Phase 2)
- `tests/torch/test_stable_block.py`:
  - `test_identity_init` — at step 0, output == input (zero-gamma)
  - `test_zero_mean_update` — the norm layer output has mean ~0
  - `test_forward_shape` — output shape matches input
- `tests/torch/test_stable_hybrid_registry.py`:
  - `test_stable_hybrid_resolves` — registry returns correct class

---

## Exit Criteria

- [ ] `gradient_clip_algorithm` field exists in both TF and Torch TrainingConfig
- [ ] AGC utility function passes unit tests
- [ ] training_step dispatches clipping based on algorithm selection
- [ ] `stable_hybrid` resolves from registry and produces correct output shapes
- [ ] StablePtychoBlock passes identity-init and zero-mean tests
- [ ] All existing tests continue to pass (no regressions)
