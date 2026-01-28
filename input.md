# FNO-STABILITY-OVERHAUL-001 — Phase 1: Foundation (Config + AGC + Dispatch)

**Summary:** Implement the configuration, AGC utility, and training_step dispatch needed for the FNO stability shootout.

**Focus:** FNO-STABILITY-OVERHAUL-001 — Phase 1 (Foundation)

**Branch:** fno2

**Mapped tests:**
- `tests/torch/test_agc.py` — new (author as part of this task)
- `tests/torch/test_grid_lines_torch_runner.py` — regression (23/23 must stay green)

**Artifacts:** `plans/active/FNO-STABILITY-OVERHAUL-001/reports/2026-01-28T010000Z/`

---

## Do Now

Implement Phase 1 from `plans/active/FNO-STABILITY-OVERHAUL-001/implementation.md`. Four tasks:

### Task 1.1: Config field `gradient_clip_algorithm`

Add `gradient_clip_algorithm: Literal['norm', 'value', 'agc'] = 'norm'` to:
1. `ptycho/config/config.py` — TF `TrainingConfig`, after `gradient_clip_val` (line ~232)
2. `ptycho_torch/config_params.py` — Torch `TrainingConfig`, after `gradient_clip_val` (line ~131)
3. Check `ptycho_torch/config_bridge.py` — if auto-bridged by name match, no change needed; otherwise add mapping

### Task 1.2: AGC utility

Add to `ptycho_torch/train_utils.py`:

```python
def adaptive_gradient_clip_(parameters, clip_factor: float = 0.01, eps: float = 1e-3):
    """Adaptive Gradient Clipping (AGC).

    Clips gradients based on the unit-wise ratio of gradient norm to parameter norm.
    See Brock et al., 2021 (NFNet), Algorithm 2.

    Operates in-place on parameter .grad tensors.
    """
    for p in parameters:
        if p.grad is None:
            continue
        p_norm = p.data.norm(2).clamp(min=eps)
        g_norm = p.grad.data.norm(2)
        max_norm = p_norm * clip_factor
        if g_norm > max_norm:
            p.grad.data.mul_(max_norm / g_norm)
```

### Task 1.3: Training step dispatch

In `ptycho_torch/model.py`, `PtychoPINN_Lightning.training_step` (lines 1334-1340), replace the hardcoded `clip_grad_norm_` with:

```python
from ptycho_torch.train_utils import adaptive_gradient_clip_

algo = getattr(self.training_config, 'gradient_clip_algorithm', 'norm')
if self.gradient_clip_val is not None and self.gradient_clip_val > 0:
    if algo == 'norm':
        torch.nn.utils.clip_grad_norm_(self.parameters(), self.gradient_clip_val)
    elif algo == 'value':
        torch.nn.utils.clip_grad_value_(self.parameters(), self.gradient_clip_val)
    elif algo == 'agc':
        adaptive_gradient_clip_(self.parameters(), clip_factor=self.gradient_clip_val)
```

### Task 1.4: CLI flag

In `scripts/studies/grid_lines_torch_runner.py`, add after `--grad-clip` (line ~578):
```python
parser.add_argument('--gradient-clip-algorithm', choices=['norm', 'value', 'agc'],
                    default='norm', help='Gradient clipping algorithm')
```
Pass to `TorchRunnerConfig` and ensure it reaches `TrainingConfig.gradient_clip_algorithm`.

### Tests

Create `tests/torch/test_agc.py`:
- `test_agc_clips_large_gradients`: create param with small norm, assign large grad, verify clipped
- `test_agc_preserves_small_gradients`: create param with large norm, assign small grad, verify unchanged
- `test_agc_handles_zero_params`: param with all zeros, verify no crash (eps guard)

### Verification

```bash
pytest tests/torch/test_agc.py -v 2>&1 | tee plans/active/FNO-STABILITY-OVERHAUL-001/reports/2026-01-28T010000Z/pytest_agc.log
pytest tests/torch/test_grid_lines_torch_runner.py -v 2>&1 | tee plans/active/FNO-STABILITY-OVERHAUL-001/reports/2026-01-28T010000Z/pytest_torch_runner_regression.log
```

---

## Next Up

Phase 2: StablePtychoBlock + StableHybridGenerator + registry (see implementation.md Tasks 2.1–2.3).
