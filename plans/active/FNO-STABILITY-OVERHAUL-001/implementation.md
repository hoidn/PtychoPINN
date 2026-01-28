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

**Status 2026-01-28:** Torch `TrainingConfig` already exposes the field, but TF `TrainingConfig` and the bridge adapter still lack it. Next action: add the field to `ptycho/config/config.py` and thread it through `to_training_config` + parity tests.

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

**Status 2026-01-28:** Runner CLI flag and `TorchRunnerConfig` wiring are in place. Compare wrapper/CLI still lack the `--torch-grad-clip-algorithm` flag, so the compare harness cannot request AGC yet. Tests covering this flow are also missing.

---

## Phase 2: Generator Engineering

### Task 2.1: Implement StablePtychoBlock

**File:** `ptycho_torch/generators/fno.py`

After `PtychoBlock` (line ~120), add `StablePtychoBlock`:
```
y = x + InstanceNorm(GELU(SpectralConv(x) + Conv3x3(x)))
```

Key differences from `PtychoBlock`:
- Add `nn.InstanceNorm2d(channels, affine=True)` after GELU
- **Zero-Gamma init:** In `__init__`, set `self.norm.weight.data.zero_()` and `self.norm.bias.data.zero_()`
- This ensures the block acts as Identity at initialization (zero output from norm → residual = x)

### Task 2.2: Implement StableHybridGenerator

**File:** `ptycho_torch/generators/fno.py`

Subclass `HybridUNOGenerator`:
- Override the block construction to use `StablePtychoBlock` instead of `PtychoBlock`
- All other architecture (U-Net skip connections, decoder) remains identical

### Task 2.3: Register `stable_hybrid`

**File:** `ptycho_torch/generators/registry.py`
- Import `StableHybridGenerator` (or the wrapper class)
- Add `'stable_hybrid': StableHybridGenerator` to `_REGISTRY`

**Status 2026-01-28:** Pending Phase 2. No changes yet.

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
