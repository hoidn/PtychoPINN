# PyTorch Refactor: Block Consolidation + Single-Output Complex Generators Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Consolidate FNO building blocks, enforce a single complex output contract across generators, and extract scheduler logic from `PtychoPINN_Lightning` without breaking existing tests or stable_hybrid checkpoints.

**Architecture:** Create `base_blocks.py` for shared FNO primitives, keep `StablePtychoBlock` as a deprecated alias to preserve checkpoint compatibility, and make all generators return complex tensors `(B, C, H, W)` while `_predict_complex` derives amp/phase only when needed.

**Tech Stack:** PyTorch, Lightning, pytest, existing `ptycho_torch` generators/model stack.

---

### Task 1: Consolidate FNO Blocks into `base_blocks.py`

**Files:**
- Create: `ptycho_torch/generators/base_blocks.py`
- Modify: `ptycho_torch/generators/fno.py`
- Modify: `ptycho_torch/generators/hybrid_resnet.py`
- Test: `tests/torch/test_fno_generators.py`

**Step 1: Write the failing test**

Add a small unit test to assert the unified block supports LayerScale and preserves shape:

```python
# tests/torch/test_fno_generators.py

def test_unified_ptychoblock_layerscale_shape():
    from ptycho_torch.generators.base_blocks import PtychoBlock
    x = torch.randn(2, 8, 16, 16)
    block = PtychoBlock(channels=8, modes=4, use_layerscale=True, norm_layer=torch.nn.InstanceNorm2d)
    y = block(x)
    assert y.shape == x.shape
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/torch/test_fno_generators.py::test_unified_ptychoblock_layerscale_shape -v`
Expected: FAIL (import or class missing).

**Step 3: Write minimal implementation**

Create `ptycho_torch/generators/base_blocks.py` and move `InputTransform`, `SpatialLifter`, and a unified `PtychoBlock` with optional LayerScale + optional normalization. Keep the spectral/local conv structure and add a small LayerScale param (init 1e-3). Update `fno.py` and `hybrid_resnet.py` to import from `base_blocks.py`.

Also keep backward compatibility:
- Keep `StablePtychoBlock` in `fno.py` as a **deprecated alias** that wraps `PtychoBlock(use_layerscale=True, norm_layer=nn.InstanceNorm2d)`.
- Preserve attribute names in stable_hybrid so checkpoint keys stay stable.

**Step 4: Run test to verify it passes**

Run: `pytest tests/torch/test_fno_generators.py::test_unified_ptychoblock_layerscale_shape -v`
Expected: PASS.

**Step 5: Commit**

```bash
git add ptycho_torch/generators/base_blocks.py ptycho_torch/generators/fno.py ptycho_torch/generators/hybrid_resnet.py tests/torch/test_fno_generators.py
git commit -m "refactor(fno): unify blocks in base_blocks"
```

---

### Task 2: Enforce Single-Output Complex Generator Contract

**Files:**
- Modify: `ptycho_torch/model.py`
- Modify: `ptycho_torch/generators/fno.py`
- Modify: `ptycho_torch/generators/fno_vanilla.py`
- Modify: `ptycho_torch/generators/hybrid_resnet.py`
- Modify: `ptycho_torch/generators/cnn.py` (wrapper) **or** `ptycho_torch/model.py` (autoencoder output)
- Test: `tests/torch/test_model_output_modes.py`
- Test: `tests/torch/test_grid_lines_torch_runner.py`

**Step 1: Write the failing test**

Update dummy generators in `tests/torch/test_model_output_modes.py` to return **complex tensors** while keeping assertions unchanged:

```python
class DummyTwoChannelGenerator(torch.nn.Module):
    def forward(self, x):
        b, c, h, w = x.shape
        amp = torch.sigmoid(torch.randn(b, c, h, w, device=x.device, dtype=x.dtype))
        phase = math.pi * torch.tanh(torch.randn(b, c, h, w, device=x.device, dtype=x.dtype))
        return (amp.to(torch.complex64) * torch.exp(1j * phase.to(torch.complex64)))

class DummyAmpPhaseGenerator(torch.nn.Module):
    def forward(self, x):
        b, c, h, w = x.shape
        amp = torch.sigmoid(torch.randn(b, c, h, w, device=x.device, dtype=x.dtype))
        phase = math.pi * torch.tanh(torch.randn(b, c, h, w, device=x.device, dtype=x.dtype))
        return (amp.to(torch.complex64) * torch.exp(1j * phase.to(torch.complex64)))
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/torch/test_model_output_modes.py -v`
Expected: FAIL (because `_predict_complex` assumes tuple/real_imag).

**Step 3: Write minimal implementation**

- Update all generators to **return complex `(B, C, H, W)`**.
  - For `real_imag` mode: project to real/imag, then convert to complex.
  - For `amp_phase` and `amp_phase_logits`: apply sigmoid/tanh inside generator, then combine to complex.
- Update `Autoencoder.forward()` (or wrap in `CnnGenerator`) to return complex instead of `(amp, phase)`.
- Simplify `_predict_complex` to:
  1) call `self.autoencoder(x)` to get complex
  2) derive `amp = torch.abs(x_complex)` and `phase = torch.angle(x_complex)`
  3) return `(x_complex, amp, phase)`
- Keep `generator_output_mode` in the generator constructor and handle logits inside the generator.

**Step 4: Run tests to verify they pass**

Run:
- `pytest tests/torch/test_model_output_modes.py -v`
- `pytest tests/torch/test_grid_lines_torch_runner.py -v`

Expected: PASS.

**Step 5: Commit**

```bash
git add ptycho_torch/model.py ptycho_torch/generators/fno.py ptycho_torch/generators/fno_vanilla.py         ptycho_torch/generators/hybrid_resnet.py ptycho_torch/generators/cnn.py         tests/torch/test_model_output_modes.py

git commit -m "refactor(torch): single-output complex generators"
```

---

### Task 3: Extract Scheduler Factory

**Files:**
- Modify: `ptycho_torch/schedulers.py`
- Modify: `ptycho_torch/model.py`
- Test: `tests/torch/test_lr_scheduler.py`

**Step 1: Write the failing test**

Add a test to ensure the factory returns a Lightning-compatible scheduler config (plateau + non-plateau):

```python
# tests/torch/test_lr_scheduler.py

def test_get_lr_scheduler_config_plateau():
    from ptycho_torch.schedulers import get_lr_scheduler_config
    optimizer = torch.optim.Adam([torch.zeros(1, requires_grad=True)], lr=1e-3)
    config = SimpleNamespace(
        scheduler='ReduceLROnPlateau',
        plateau_factor=0.5,
        plateau_patience=2,
        plateau_min_lr=1e-4,
        plateau_threshold=0.0,
        lr_warmup_epochs=0,
        lr_min_ratio=0.1,
        nepochs=5,
    )
    sched = get_lr_scheduler_config(optimizer, config, monitor_name='val_loss')
    assert 'scheduler' in sched and sched['monitor'] == 'val_loss'
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/torch/test_lr_scheduler.py::test_get_lr_scheduler_config_plateau -v`
Expected: FAIL (factory missing).

**Step 3: Write minimal implementation**

- Add `get_lr_scheduler_config(...)` to `ptycho_torch/schedulers.py` and move scheduler construction logic from `PtychoPINN_Lightning.configure_optimizers` into it.
- Update `configure_optimizers` to call the factory and return `{optimizer, lr_scheduler}`.
- Keep behavior identical, including monitor name and `ReduceLROnPlateau` dict format.

**Step 4: Run test to verify it passes**

Run: `pytest tests/torch/test_lr_scheduler.py -v`
Expected: PASS.

**Step 5: Commit**

```bash
git add ptycho_torch/schedulers.py ptycho_torch/model.py tests/torch/test_lr_scheduler.py
git commit -m "refactor(torch): extract scheduler factory"
```

---

### Task 4: Integration Gate + Evidence Capture

**Files:**
- Evidence: `.artifacts/ptycho-torch-refactor/logs/`

**Step 1: Run integration marker**

Run: `pytest -v -m integration | tee .artifacts/ptycho-torch-refactor/logs/pytest_integration.log`
Expected: PASS.

**Step 2: Run targeted regression tests**

Run:
- `pytest tests/torch/test_fno_generators.py -v | tee .artifacts/ptycho-torch-refactor/logs/test_fno_generators.log`
- `pytest tests/torch/test_model_output_modes.py -v | tee .artifacts/ptycho-torch-refactor/logs/test_model_output_modes.log`
- `pytest tests/torch/test_grid_lines_torch_runner.py -v | tee .artifacts/ptycho-torch-refactor/logs/test_grid_lines_torch_runner.log`
- `pytest tests/torch/test_lr_scheduler.py -v | tee .artifacts/ptycho-torch-refactor/logs/test_lr_scheduler.log`

Expected: PASS.

**Step 3: Commit evidence pointers (no artifacts in git)**

If a plan summary or ledger is required by workflow, add links to `.artifacts/ptycho-torch-refactor/logs/`.

---

## Notes / Guardrails

- Preserve module attribute names for stable_hybrid so checkpoint keys remain stable.
- Keep `amp_phase_logits` semantics: logits are converted inside generators.
- Use `python` via PATH for all commands (PYTHON-ENV-001).
- Avoid touching `ptycho/model.py`, `ptycho/diffsim.py`, or `ptycho/tf_helper.py`.

---

## Definition of Done

1. All generators return `torch.complex64` tensors in `(B, C, H, W)`.
2. Stable_hybrid checkpoints load without key or shape mismatches.
3. `PtychoPINN_Lightning.configure_optimizers` contains no direct scheduler branching.
4. `tests/torch/test_model_output_modes.py` and `tests/torch/test_grid_lines_torch_runner.py` pass with unchanged assertions.
5. `pytest -m integration` passes and logs are archived under `.artifacts/ptycho-torch-refactor/logs/`.
