# LayerScale Stable Hybrid Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Allow `stable_hybrid` to learn meaningful residuals by adding a LayerScale gate to `StablePtychoBlock` and validating it via Stage A reruns.

**Architecture:** Extend the PyTorch generator stack so `StablePtychoBlock` uses InstanceNorm + learnable LayerScale instead of hard zero-gamma, then prove the change via focused unit tests and a rerun of the Stage A stable arm with shared datasets.

**Tech Stack:** Python 3.11, PyTorch/Lightning, pytest, grid_lines CLI harness.

---

### Task 1: LayerScale-Enhanced StablePtychoBlock

**Files:**
- Modify: `ptycho_torch/generators/fno.py:120-220, 380-520`

**Step 1: Write the failing test**
- Edit `tests/torch/test_fno_generators.py::TestStablePtychoBlock`:
  ```python
  def test_layerscale_grad_flow(self):
      block = StablePtychoBlock(channels=8, modes=4, layerscale_init=1e-3)
      x = torch.randn(2, 8, 16, 16, requires_grad=True)
      loss = (block(x) ** 2).mean()
      loss.backward()
      assert block.layerscale.grad is not None
      assert block.layerscale.grad.norm().item() > 0
  ```
- Relax `test_identity_init` tolerance to `atol=5e-3` and assert `torch.max(torch.abs(out - x)) > 1e-6`.
- Remove the manual `norm.weight` override in `test_zero_mean_update` (defaults now 1.0) and keep the zero-mean assertion.

**Step 2: Run test to verify it fails**
- Command: `pytest tests/torch/test_fno_generators.py::TestStablePtychoBlock -v`
- Expected: Fails because `StablePtychoBlock` lacks `layerscale` parameter / gradients.

**Step 3: Write minimal implementation**
- In `StablePtychoBlock.__init__`:
  - Add `layerscale_init: float = 1e-3` argument.
  - Initialize InstanceNorm weights to ones (remove zero init) and biases to zeros.
  - Add `self.layerscale = nn.Parameter(torch.full((channels,), layerscale_init))` and helper `self._layerscale_shape = (1, channels, 1, 1)`.
- In `forward`:
  ```python
  update = self.act(self.spectral(x) + self.local_conv(x))
  update = self.norm(update)
  ls = self.layerscale.view(*self._layerscale_shape)
  return x + ls * update
  ```
- Update docstrings/comments for LayerScale behavior.

**Step 4: Run test to verify it passes**
- `pytest tests/torch/test_fno_generators.py::TestStablePtychoBlock -v`
- Expected: PASS.

**Step 5: Commit**
- `git add ptycho_torch/generators/fno.py tests/torch/test_fno_generators.py`
- `git commit -m "feat: add layerscale gate to StablePtychoBlock"`

---

### Task 2: Stage A Stable Arm Rerun with LayerScale

**Files:**
- CLI: `scripts/studies/grid_lines_compare_wrapper.py`
- Outputs: `outputs/grid_lines_stage_a/arm_stable_layerscale`
- Artifacts: `plans/active/FNO-STABILITY-OVERHAUL-001/reports/<timestamp>/`

**Step 1: Prep datasets**
- Command: `rsync -a outputs/grid_lines_stage_a/arm_control/datasets/ outputs/grid_lines_stage_a/arm_stable_layerscale/datasets/`
- `rm -rf outputs/grid_lines_stage_a/arm_stable_layerscale/runs`

**Step 2: Run Stage A stable arm**
- Command:
  ```bash
  AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md \
  python scripts/studies/grid_lines_compare_wrapper.py \
    --N 64 --gridsize 1 \
    --output-dir outputs/grid_lines_stage_a/arm_stable_layerscale \
    --architectures stable_hybrid \
    --seed 20260128 \
    --nimgs-train 1 --nimgs-test 1 --nphotons 1e9 \
    --nepochs 50 --torch-epochs 50 \
    --torch-grad-clip 0.0 --torch-grad-clip-algorithm norm \
    --torch-loss-mode mae --fno-blocks 4 --torch-infer-batch-size 8 \
    2>&1 | tee plans/active/FNO-STABILITY-OVERHAUL-001/reports/<timestamp>/stage_a_arm_stable_layerscale.log
  ```
- Expected: Training completes without NaNs; log archived.

**Step 3: Archive metrics**
- Copy `outputs/.../runs/pinn_stable_hybrid/{history.json,metrics.json,model.pt}` into the same `<timestamp>` hub.
- Use helper snippet:
  ```bash
  python scripts/internal/stage_a_dump_stats.py \
    --run-dir outputs/grid_lines_stage_a/arm_stable_layerscale/runs/pinn_stable_hybrid \
    --out-json plans/active/FNO-STABILITY-OVERHAUL-001/reports/<timestamp>/stage_a_arm_stable_layerscale_stats.json
  ```

**Step 4: Update metrics table**
- Append new row to `plans/active/FNO-STABILITY-OVERHAUL-001/reports/2026-01-29T010000Z/stage_a_metrics.json` (or new file) and refresh `stage_a_summary.md` describing whether LayerScale improved val_loss / SSIM.

**Step 5: Commit**
- `git add plans/active/FNO-STABILITY-OVERHAUL-001/reports/<timestamp>/*.json`
- `git commit -m "chore: stage A stable arm rerun with layerscale"`

---

### Task 3: Docs & Findings Sync

**Files:**
- `docs/strategy/mainstrategy.md`
- `docs/fix_plan.md`
- `docs/findings.md`
- `plans/active/FNO-STABILITY-OVERHAUL-001/implementation.md`

**Step 1: Update implementation plan**
- Add status notes under Phase 5 referencing the LayerScale change and rerun outcome.

**Step 2: Strategy doc**
- In §Stage A table, include the LayerScale arm metrics plus decision (e.g., whether it beats control or remains stalled).

**Step 3: Fix plan ledger**
- Add Attempts History bullet summarizing the rerun (dataset reuse, metrics, stability) and update supervisor state/dwell line per FSM.

**Step 4: Findings**
- If LayerScale unlocks progress (or fails), add/modify a Finding (e.g., update `STABLE-GAMMA-001` or introduce `STABLE-LS-001`) citing the new evidence path.

**Step 5: Commit**
- `git add docs/strategy/mainstrategy.md docs/fix_plan.md docs/findings.md plans/active/FNO-STABILITY-OVERHAUL-001/implementation.md`
- `git commit -m "docs: record layerscale stable_hybrid results"`

---

**Execution Options:**
1. **Subagent-Driven (this session):** Use superpowers:subagent-driven-development, tackle each task sequentially with reviews after every commit.
2. **Parallel Session:** Start a new session/worktree, invoke superpowers:executing-plans, and implement tasks with checkpoint reviews between groups of steps.

Which approach? (Respond `1` or `2`).

