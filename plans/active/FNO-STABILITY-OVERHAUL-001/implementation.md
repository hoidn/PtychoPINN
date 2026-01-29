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

**Status 2026-01-28:** COMPLETE — All three tasks (2.1–2.3) implemented. `StablePtychoBlock` with zero-init InstanceNorm, `StableHybridUNOGenerator` via `block_cls` injection, registry entry `'stable_hybrid'`, config Literal extensions (TF + Torch), CLI wiring (`--architecture stable_hybrid`), compare wrapper routing, and docs updated. All mapped test selectors pass (7 tests total).

---

## Phase 3: Validation (Stage A Shootout)

Stage A validates the architectural fix (`stable_hybrid`) against the optimization fix (Hybrid + AGC) using the grid-lines harness described in `docs/strategy/mainstrategy.md §2.Stage A`. All three arms must share the exact cached dataset and probe so that loss/SSIM comparisons are meaningful. Use the compare wrapper to run the TF workflow + Torch runner in lockstep, and archive every CLI log under `plans/active/FNO-STABILITY-OVERHAUL-001/reports/<timestamp>/` per `docs/TESTING_GUIDE.md`.

### Task 3.1: Prepare shared dataset + run directories

**Files / Paths:** `scripts/studies/grid_lines_compare_wrapper.py`; output root `outputs/grid_lines_stage_a/`; artifacts hub `plans/active/FNO-STABILITY-OVERHAUL-001/reports/2026-01-29T010000Z/`.

1. Create the root folders:
   ```bash
   mkdir -p outputs/grid_lines_stage_a/{arm_control,arm_stable,arm_agc}
   mkdir -p plans/active/FNO-STABILITY-OVERHAUL-001/reports/2026-01-29T010000Z
   ```
2. The first arm (`arm_control`) generates the canonical dataset under `arm_control/datasets/N64/gs1/`. After it finishes, copy that `datasets` tree to the other arms **before** running them so all NPZs/metadata match:
   ```bash
   rsync -a outputs/grid_lines_stage_a/arm_control/datasets/ outputs/grid_lines_stage_a/arm_stable/datasets/
   rsync -a outputs/grid_lines_stage_a/arm_control/datasets/ outputs/grid_lines_stage_a/arm_agc/datasets/
   ```
   (Re-run the copy whenever you regenerate the control arm.)
3. Record the shared seed (`20260128`) and hyperparameters (N=64, gridsize=1, nimgs_train/test=2, nphotons=1e9, nepochs=50, fno_blocks=4) in a short README inside the artifacts hub for traceability.
   - For quick test runs, use `nimgs_train=1` and `nimgs_test=1` to reduce runtime/GPU memory before running the full settings.

### Task 3.2: Arm 1 — Control (`hybrid`, norm clip 1.0)

Goal: Establish the failure baseline and produce the shared dataset.

Command:
```bash
python scripts/studies/grid_lines_compare_wrapper.py \
  --N 64 --gridsize 1 \
  --output-dir outputs/grid_lines_stage_a/arm_control \
  --architectures hybrid \
  --seed 20260128 \
  --nimgs-train 2 --nimgs-test 2 --nphotons 1e9 \
  --nepochs 50 --torch-epochs 50 \
  --torch-grad-clip 1.0 --torch-grad-clip-algorithm norm \
  --torch-loss-mode mae --fno-blocks 4 --torch-infer-batch-size 8
```

Artifacts: capture stdout/stderr to `.../stage_a_arm_control.log` and copy:
- `outputs/grid_lines_stage_a/arm_control/metrics.json`
- `outputs/grid_lines_stage_a/arm_control/runs/pinn_hybrid/{history.json,metrics.json}`

### Task 3.3: Arm 2 — Architectural Fix (`stable_hybrid`, no clip)

Prep: ensure `outputs/grid_lines_stage_a/arm_stable/datasets` exists (copied from control) and delete any stale `runs/pinn_stable_hybrid` files.

Command:
```bash
python scripts/studies/grid_lines_compare_wrapper.py \
  --N 64 --gridsize 1 \
  --output-dir outputs/grid_lines_stage_a/arm_stable \
  --architectures stable_hybrid \
  --seed 20260128 \
  --nimgs-train 2 --nimgs-test 2 --nphotons 1e9 \
  --nepochs 50 --torch-epochs 50 \
  --torch-grad-clip 0.0 --torch-grad-clip-algorithm norm \
  --torch-loss-mode mae --fno-blocks 4 --torch-infer-batch-size 8
```

Artifacts: log to `stage_a_arm_stable.log` and archive `runs/pinn_stable_hybrid/{history.json,metrics.json,model.pt}` plus the top-level `metrics.json`.

### Task 3.4: Arm 3 — Optimization Fix (`hybrid`, AGC 0.01)

Prep: copy the dataset into `outputs/grid_lines_stage_a/arm_agc/datasets` and remove any previous `runs/pinn_hybrid` inside `arm_agc`.

Command:
```bash
python scripts/studies/grid_lines_compare_wrapper.py \
  --N 64 --gridsize 1 \
  --output-dir outputs/grid_lines_stage_a/arm_agc \
  --architectures hybrid \
  --seed 20260128 \
  --nimgs-train 2 --nimgs-test 2 --nphotons 1e9 \
  --nepochs 50 --torch-epochs 50 \
  --torch-grad-clip 0.01 --torch-grad-clip-algorithm agc \
  --torch-loss-mode mae --fno-blocks 4 --torch-infer-batch-size 8
```

Artifacts: log to `stage_a_arm_agc.log` and archive the `runs/pinn_hybrid` metrics/history for reference.

### Task 3.5: Summarize metrics + pick Stage B candidate

1. Use a short Python snippet to pull the key numbers (best `val_loss` + phase SSIM) from each arm. `val_loss` lives in `history.json`, while the phase SSIM is the second element of the `ssim` tuple returned by `eval_reconstruction`:
   ```bash
   python - <<'PY'
   import json, pathlib
   base = pathlib.Path('outputs/grid_lines_stage_a')
   arms = {
       'control': ('pinn_hybrid', base/'arm_control'),
       'stable': ('pinn_stable_hybrid', base/'arm_stable'),
       'agc': ('pinn_hybrid', base/'arm_agc'),
   }
   rows = []
   for name, (arch_key, arm_dir) in arms.items():
       run_dir = arm_dir/'runs'/arch_key
       history_path = run_dir/'history.json'
       metrics_path = run_dir/'metrics.json'
       if not history_path.exists() or not metrics_path.exists():
           continue
       history = json.loads(history_path.read_text())
       val_losses = history.get('val_loss', [])
       best_val_loss = min(val_losses) if val_losses else None
       metrics = json.loads(metrics_path.read_text())
       ssim = metrics.get('ssim', [None, None])
       rows.append({
           'arm': name,
           'arch': arch_key,
           'best_val_loss': best_val_loss,
           'phase_ssim': ssim[1] if isinstance(ssim, (list, tuple)) else None,
       })
   rows.sort(key=lambda r: (r['best_val_loss'] is None, r['best_val_loss']))
   out = pathlib.Path('plans/active/FNO-STABILITY-OVERHAUL-001/reports/2026-01-29T010000Z/stage_a_metrics.json')
   out.write_text(json.dumps(rows, indent=2))
   PY
   ```
2. Write `stage_a_summary.md` with a table ranking the arms, note any NaNs/instability, and recommend the Stage B candidate per `docs/strategy/mainstrategy.md §2.Stage B` (if `stable_hybrid` wins, Stage B uses it; otherwise AGC).
3. Update this plan + `docs/fix_plan.md` with the outcome and Stage B next tasks.

**Status 2026-01-29:** COMPLETE — All Stage A arms executed. Results:
- Control (hybrid, norm 1.0): best val_loss=0.0138, amp_ssim=0.925, phase_ssim=0.997
- AGC (hybrid, agc 0.01): best val_loss=0.0243, amp_ssim=0.811, phase_ssim=0.989
- Stable (stable_hybrid, no clip): best val_loss=0.178, amp_ssim=0.277, phase_ssim=1.0 (vacuous)

**Outcome:** Control won. Neither fix outperformed baseline. stable_hybrid stagnated in near-identity regime due to zero-gamma initialization preventing gradient flow through branches. See `reports/2026-01-29T010000Z/stage_a_summary.md` for full analysis.

**Stage B recommendation:** Run control at fno_blocks=8 to test depth scaling. If control survives, revise the instability hypothesis.

---

## Phase 4: Stage B Stress Test (Deep Control Run)

Stage B validates whether the Stage A winner (control arm: `hybrid` + norm clip 1.0) remains stable when we double the model depth to `fno_blocks=8` per `docs/strategy/mainstrategy.md §Stage B`. The deep arm must reuse the exact Stage A dataset/probe so the only changed factor is depth.

### Task 4.1: Prep Stage B workspace

**Paths:**
- Output root: `outputs/grid_lines_stage_b/deep_control`
- Shared datasets: copy from `outputs/grid_lines_stage_a/arm_control/datasets/`
- Artifacts hub: `plans/active/FNO-STABILITY-OVERHAUL-001/reports/2026-01-29T180000Z/`

**Steps:**
1. Ensure the artifacts hub exists: `mkdir -p plans/active/FNO-STABILITY-OVERHAUL-001/reports/2026-01-29T180000Z`.
2. Create the Stage B output tree and wipe any stale runs:
   ```bash
   mkdir -p outputs/grid_lines_stage_b/deep_control
   rsync -a --delete outputs/grid_lines_stage_a/arm_control/datasets/ \
     outputs/grid_lines_stage_b/deep_control/datasets/
   rm -rf outputs/grid_lines_stage_b/deep_control/runs
   ```
3. Drop a short README under the artifacts hub documenting the shared hyperparameters (N=64, gridsize=1, `fno_blocks=8`, seed=20260128, nimgs_train/test=2, nphotons=1e9, loss=MAE, clip=1.0 norm). This mirrors the Stage A README for traceability.
   - For quick test runs, start with `nimgs_train=1` and `nimgs_test=1` to confirm stability before re-running the full settings.

### Task 4.2: Execute Stage B deep control run (`fno_blocks=8`)

**Command:**
```bash
AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md \
python scripts/studies/grid_lines_compare_wrapper.py \
  --N 64 --gridsize 1 \
  --output-dir outputs/grid_lines_stage_b/deep_control \
  --architectures hybrid \
  --seed 20260128 \
  --nimgs-train 2 --nimgs-test 2 --nphotons 1e9 \
  --nepochs 50 --torch-epochs 50 \
  --fno-blocks 8 \
  --torch-grad-clip 1.0 --torch-grad-clip-algorithm norm \
  --torch-loss-mode mae --torch-infer-batch-size 8 \
  --torch-log-grad-norm --torch-grad-norm-log-freq 1 \
  2>&1 | tee plans/active/FNO-STABILITY-OVERHAUL-001/reports/2026-01-29T180000Z/stage_b_deep_control.log
```

**Notes:**
- Keep epochs at 50 so `fno_blocks` is the only variable relative to Stage A. If gradients spike, the log will show per-epoch norms via `--torch-log-grad-norm`.
- After the run, copy `metrics.json`, `history.json`, and `model.pt` from `outputs/grid_lines_stage_b/deep_control/runs/pinn_hybrid/` into the artifacts hub.
- Capture quick stats by reusing the Stage A helper:
  ```bash
  python scripts/internal/stage_a_dump_stats.py \
    --run-dir outputs/grid_lines_stage_b/deep_control/runs/pinn_hybrid \
    --out-json plans/active/FNO-STABILITY-OVERHAUL-001/reports/2026-01-29T180000Z/stage_b_deep_control_stats.json
  ```

### Task 4.3: Summarize Stage B metrics + decide next move

1. Extend the existing metrics snippet to produce `stage_b_metrics.json` under the new artifacts hub (same format as Stage A, but single-row). Include `best_val_loss`, amp/phase SSIM, amp MAE, and gradient norm extrema.
2. Author `plans/active/FNO-STABILITY-OVERHAUL-001/reports/2026-01-29T180000Z/stage_b_summary.md` with:
   - Table comparing Stage A control vs Stage B deep control (val_loss, SSIMs, MAEs, grad norm max/median).
   - Narrative on whether instability emerged at depth; cite `docs/strategy/mainstrategy.md §Stage B` for the success criteria.
   - Decision tree: if the deep run fails, outline remediation (e.g., rerun with AGC or stable_hybrid); if it survives, explicitly state the hypothesis revision and propose next experiments (e.g., higher epochs or multi-seed sweep).
3. Update this implementation plan (§Phase 4 status), `docs/strategy/mainstrategy.md` (Stage A outcome + Stage B status), and `docs/fix_plan.md` attempts history with the Stage B verdict.
4. Append any durable lessons (e.g., zero-gamma stagnation evidence, deep-control behavior) to `docs/findings.md`.

### Task 4.4: Regression guard (unchanged selectors)

Re-run the Phase 3 regression selectors to prove the CLI plumbing and runner flags still work with `fno_blocks=8`:
- `pytest tests/torch/test_fno_generators.py -k stable -v`
- `pytest tests/test_grid_lines_compare_wrapper.py::test_wrapper_handles_stable_hybrid -v`
- `pytest tests/torch/test_grid_lines_torch_runner.py::TestChannelGridsizeAlignment::test_runner_accepts_stable_hybrid -v`
- `pytest tests/test_grid_lines_compare_wrapper.py::test_wrapper_passes_grad_clip_algorithm -v`

Archive the pytest logs plus Stage B CLI log, stats JSON, and metrics JSON under `plans/active/FNO-STABILITY-OVERHAUL-001/reports/2026-01-29T180000Z/` per `docs/TESTING_GUIDE.md`.

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
