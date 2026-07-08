# VarPro / Probe-Scaling Ablation Demo (main) + Merge into fno-stable — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Demonstrate, visually and numerically, the effect of VarPro-style intensity scaling and probe-weighted reassembly on `main`'s PyTorch stack under gridsize 1 and 2; then merge those physics into `fno-stable` (full CNN parity, backlog B1–B8) preserving the generator abstraction, verified by fixtures and the same demo harness.

**Architecture:** Phase 0 builds branch-independent low-photon fly001 datasets. Phase 1 adds three small ablation gates on a branch off `main` (training probe weighting, s1/s2 freeze, inference VarPro bypass), captures frozen parity fixtures, and runs a 5-arm training matrix with cheap per-checkpoint inference sweeps. Phase 2 merges into `fno-stable`: verbatim extraction of `reassemble_patches_position_real_probe` from main, reuse of the already-in-tree `beta_modules` `RectangularScaledDiffraction`, config-gated wiring at the two `ForwardModel.forward` plug points, CNN rectangular mode + shared decoder (B1/B2), with **replace** semantics for scaling and fixture-parity acceptance.

**Tech Stack:** PyTorch/Lightning (`ptycho_torch`), pytest, matplotlib, tmux (ptycho311 env), git file-level merges.

**Branch context:** `main` = `5bd07399` (ancestor of `fno-stable`; the 13978cf1 merge already consumed it — main's physics survives verbatim in `ptycho_torch/beta_modules/model.py`, bit-identical on both branches). Backlog: `docs/plans/2026-06-29-main-fno-stable-physics-reconciliation-backlog.md` (this plan executes B1–B8; B6 fixtures are captured in Phase 1).

**Documents read:** `docs/index.md`, `docs/findings.md` (NORMALIZATION-001, TORCH-REASSEMBLY-NORM-001, PROBE-MASK-DEFAULT-001, DATA-001), `docs/workflows/pytorch.md`, `docs/DATA_GENERATION_GUIDE.md`, `docs/TESTING_GUIDE.md`, `specs/data_contracts.md`, the 2026-06-29 backlog, `ptycho_torch/{model,helper,reassembly,config_params,config_factory}.py` (both branches), `ptycho_torch/beta_modules/model.py`, `scripts/studies/{grid_lines_torch_runner,grid_study_dataset_builder,demo_varpro_probe_weighted_reassembly}.py`.

## Global Constraints

- **No worktrees** (CLAUDE.md §2.12). All branch switches happen in this tree; run `git submodule update --init --recursive` after every switch (§2.13).
- **Knob names must be identical on both branches:** `training_patch_weighting`, `rect_s1s2_trainable`, `physics_forward_mode`, `cnn_output_mode`, `use_shared_decoder`, `patch_weighting`, `varpro_scaling`.
- **Merge over rewrite:** code that exists on `main` or in `beta_modules` is extracted via git (`git show main:<path>`), never retyped.
- **Defaults never change behavior:** every new knob defaults to the branch's current behavior (`'probe'`/trainable on the ablation branch = main's hardcoded behavior; `'central_mask'`/`'amplitude'` on fno-stable).
- **Scaling semantics = replace:** when `physics_forward_mode='rectangular_scaled'`, `RectangularScaledDiffraction` replaces `ProbeIllumination`→`pad_and_diffract`→`inv_scale`→`alpha/beta` AND the loss-time `physics_scaling_constant` multiply. Main routes physics scale into the forward instead: `modified_output_scale = sqrt(1/(probe_scaling²·physics_scale + 1e-9))` passed as `output_scale_factor` (main model.py L1538→L1542) — the rectangular mode must reproduce that routing (per the Task 1.1 Step 2 finding), not merely skip fno-stable's multiply. Acceptance is fixture parity at the loss-input tensor.
- **Artifacts:** datasets, checkpoints, PNGs, metrics → `.artifacts/varpro_ablation/` (git-ignored). Never committed. Fixtures ≤ ~2 MB may live under `tests/fixtures/varpro_parity/`. Note: `.artifacts/` is only in fno-stable's `.gitignore` (main's lacks it), so it is additionally ignored via `.git/info/exclude` (branch-independent; already applied by the controller).
- **Long runs:** tmux + ptycho311 env; track the launched PID (`cmd & pid=$!; wait "$pid"`), never `pgrep -f` loops; a run is complete only on exit 0 + fresh artifacts.
- **Do not touch** `ptycho/model.py`, `ptycho/diffsim.py`, `ptycho/tf_helper.py` (TF core; out of scope).
- Python via PATH `python` (PYTHON-ENV-001). Test evidence archived per `docs/TESTING_GUIDE.md`.

## Condition Matrix (approved)

Training arms (each = one training run; gs1 forces nothing extra — main CLI sets `object_big=False` at gs1, so training reassembly is skipped and `training_patch_weighting` is irrelevant there):

| Arm | N | gridsize | training_patch_weighting | rect_s1s2_trainable |
|---|---|---|---|---|
| `gs1_frozen` | 64 | 1 | (n/a, object_big=False) | False |
| `gs1_trainable` | 64 | 1 | (n/a) | True |
| `gs2_neither` | 64 | 2 | uniform | False |
| `gs2_probe_frozen` | 64 | 2 | probe | False |
| `gs2_probe_trainable` | 64 | 2 | probe | True |
| `gs2_neither_n128` | 128 | 2 | uniform | False |
| `gs2_probe_trainable_n128` | 128 | 2 | probe | True |

The N=64 vs N=128 dyad is the paired comparison `gs2_{neither,probe_trainable}` at both N — it shows whether the varpro/probe benefit grows with patch size (larger patches ⇒ more intra-group overlap area and stronger probe-envelope variation across the patch). Two arms, not a full replication, to control runtime (N=128 is ~4× the pixels per image).

Inference sweeps per checkpoint (deterministic, no retraining): gs1 checkpoints × {uniform, probe} × {varpro on, off} = 4 variants; gs2 checkpoints (both N) × {probe+varpro, uniform+no-varpro} = 2 variants.

---

## Phase 0 — Datasets (on `fno-stable`, before any branch switch)

### Task 0.1: Generate low-photon fly001 train/test NPZs

**Files:**
- Create: `.artifacts/varpro_ablation/datasets/fly64_p1e9_gs{1,2}_{train,test}.npz` (git-ignored)
- Create: `.artifacts/varpro_ablation/datasets/provenance.json`

**Interfaces:**
- Produces: raw nongrid RawData NPZs (keys `diff3d`/`diffraction`, `xcoords`, `ycoords`, `probeGuess (64,64) c64`, `objectGuess`, `scan_index`) consumed by main's `python -m ptycho_torch.train --train_data_file …` (Phase 1) and by `grid_study_dataset_builder` (Phase 2 verification).

- [ ] **Step 1: Verify the source dataset and simulator CLI**

```bash
python - <<'EOF'
import numpy as np
d = np.load('datasets/fly/fly001.npz')
print({k: (v.shape, v.dtype) for k, v in d.items()})
EOF
python scripts/simulation/simulate_and_save.py --help
```
Expected: `objectGuess (232,232) complex`, `probeGuess (64,64) complex`, `diff3d (10304,64,64)`; help lists `--input-file --output-file --n-images --n-photons --gridsize --seed` (and `--buffer`). If flag names differ, use the actual names in Step 2 and record them in provenance.json.

- [ ] **Step 2: Simulate low-photon splits (subsampled for runtime)**

```bash
mkdir -p .artifacts/varpro_ablation/datasets
for gs in 1 2; do
  python scripts/simulation/simulate_and_save.py \
    --input-file datasets/fly/fly001.npz \
    --output-file .artifacts/varpro_ablation/datasets/fly64_p1e9_gs${gs}_train.npz \
    --n-images 512 --n-photons 1e9 --gridsize ${gs} --seed 7
  python scripts/simulation/simulate_and_save.py \
    --input-file datasets/fly/fly001.npz \
    --output-file .artifacts/varpro_ablation/datasets/fly64_p1e9_gs${gs}_test.npz \
    --n-images 128 --n-photons 1e9 --gridsize ${gs} --seed 8
done
```
(Two gs variants because DATA_GENERATION_GUIDE requires simulation gridsize == training gridsize; if inspection shows the nongrid output is gridsize-independent — ungrouped `diff3d` identical across gs — keep only gs1 files and note it.)

- [ ] **Step 2b: Build the N=128 variant (probe upsample + simulate)**

N is inferred from the probe (`_infer_probe_size` in main's `train.py`), and fly001's `probeGuess` is 64×64, so N=128 needs a derived probe. First check for existing probe-rescaling machinery (the N=128 grid-lines path had one): `grep -rn "probe" ptycho/workflows/grid_lines_workflow.py | grep -in "resize\|zoom\|interp\|128"`. If a utility exists, use it; otherwise Fourier-pad (zero-pad the probe's Fourier transform to 128×128, inverse transform, renormalize power):

```bash
python - <<'EOF'
import numpy as np
d = dict(np.load('datasets/fly/fly001.npz'))
p = d['probeGuess'].astype(np.complex64)
P = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(p)))
Ppad = np.zeros((128, 128), dtype=np.complex64)
Ppad[32:96, 32:96] = P
p128 = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(Ppad))).astype(np.complex64)
p128 *= np.sqrt(np.sum(np.abs(p)**2) / np.sum(np.abs(p128)**2))
d['probeGuess'] = p128
np.savez_compressed('.artifacts/varpro_ablation/datasets/fly001_probe128.npz', **d)
print('probe128:', p128.shape, 'power ratio:', np.sum(np.abs(p128)**2)/np.sum(np.abs(p)**2))
EOF
for split_seed in "train 7" "test 8"; do
  set -- $split_seed
  python scripts/simulation/simulate_and_save.py \
    --input-file .artifacts/varpro_ablation/datasets/fly001_probe128.npz \
    --output-file .artifacts/varpro_ablation/datasets/fly128_p1e9_gs2_$1.npz \
    --n-images $([ "$1" = train ] && echo 512 || echo 128) --n-photons 1e9 --gridsize 2 --seed $2
done
```
Expected: power ratio ≈ 1.0; two N=128 NPZs with `diff3d (·,128,128)`. Note: at N=128 the 232×232 fly001 object gives patches covering ~55% of the object per side — verify the simulator accepts this geometry (it may need `--buffer` reduced); if it rejects, upsample `objectGuess` 2× (bicubic on real/imag separately) into the same NPZ and record that in provenance.

- [ ] **Step 3: Validate contract (DATA-001) and record provenance**

```bash
python - <<'EOF'
import json, numpy as np, subprocess
out = {}
for gs in (1, 2):
    for split in ('train', 'test'):
        p = f'.artifacts/varpro_ablation/datasets/fly64_p1e9_gs{gs}_{split}.npz'
        d = np.load(p)
        keys = set(d.keys())
        assert {'xcoords', 'ycoords', 'probeGuess', 'objectGuess'} <= keys, (p, keys)
        assert ('diff3d' in keys) or ('diffraction' in keys), (p, keys)
        out[p] = {k: str(getattr(v, 'shape', v)) for k, v in d.items()}
out['git_commit'] = subprocess.run(['git','rev-parse','HEAD'],capture_output=True,text=True).stdout.strip()
out['source'] = 'datasets/fly/fly001.npz'; out['nphotons'] = 1e4; out['seeds'] = {'train': 7, 'test': 8}
json.dump(out, open('.artifacts/varpro_ablation/datasets/provenance.json','w'), indent=2)
print('OK')
EOF
```
Expected: `OK`.

- [ ] **Step 4: Commit the plan document only**

```bash
git add docs/plans/2026-07-01-varpro-probe-scaling-ablation-and-merge.md
git commit -m "docs: plan varpro/probe-scaling ablation demo and main merge"
```

---

## Phase 1 — Ablation branch off `main`

### Task 1.1: Create branch and establish baseline

**Files:** none (branch setup)

- [ ] **Step 1: Switch (uncommitted submodule pointer drift is expected; do not commit it)**

```bash
git switch -c varpro-ablation 5bd07399
git submodule update --init --recursive
python -c "import ptycho_torch.model, ptycho_torch.train, ptycho_torch.reassembly; print('imports OK')"
```
Expected: `imports OK`.

- [ ] **Step 2: Record main's loss-units convention AND scale routing (decision input for Task 2.6)**

```bash
sed -n '1500,1580p' ptycho_torch/model.py   # compute_loss is at L1522 — confirm what consumes pred_scaled_intensity
grep -n "class PoissonLoss\|class MAELoss\|modified_output_scale" ptycho_torch/model.py
```
Record TWO facts in a NEW findings file `docs/plans/2026-07-01-varpro-ablation-phase1-findings.md` (the main plan doc does not exist on this branch — do not recreate it here) and commit:
1. Loss-input units: Poisson/MAE input = intensity or amplitude; any internal squaring.
2. Scale routing: main does NOT multiply `physics_scale` at loss time — it folds it into the forward via `modified_output_scale = torch.sqrt(1/(probe_scaling**2 * physics_scale + 1e-9))` (L1538) passed as `output_scale_factor` (L1542). fno-stable instead multiplies `pred * physics_scale` at the loss (its model.py L1815-1816). Task 2.6 must reproduce main's routing, not merely skip fno-stable's multiply.
This file rides to fno-stable via cherry-pick (Phase 2 intro); the fixtures freeze exactly these conventions.

### Task 1.2: Training ablation knobs (`training_patch_weighting`, `rect_s1s2_trainable`)

**Files:**
- Modify: `ptycho_torch/config_params.py` (ModelConfig)
- Modify: `ptycho_torch/model.py` (`ForwardModel.forward` ~L675-702; `RectangularScaledDiffraction.__init__` ~L705+)
- Modify: `ptycho_torch/train.py` (argparse ~L526+, overrides dict ~L806-815)
- Test: `tests/torch/test_training_forward_ablation_knobs.py` (new)

**Interfaces:**
- Produces: `ModelConfig.training_patch_weighting: Literal['probe','uniform'] = 'probe'`; `ModelConfig.rect_s1s2_trainable: bool = True`; CLI `--training-patch-weighting {probe,uniform}`, `--freeze-s1s2` (sets `rect_s1s2_trainable=False`). These exact names recur in Phase 2.

- [ ] **Step 1: Write failing tests**

```python
# tests/torch/test_training_forward_ablation_knobs.py
import dataclasses
import torch
from ptycho_torch.config_params import ModelConfig
from ptycho_torch.model import RectangularScaledDiffraction


def test_model_config_exposes_ablation_knobs_with_main_defaults():
    cfg = ModelConfig()
    assert cfg.training_patch_weighting == "probe"
    assert cfg.rect_s1s2_trainable is True


def test_frozen_s1s2_reduces_to_plain_scaled_diffraction():
    cfg = dataclasses.replace(ModelConfig(), rect_s1s2_trainable=False, num_datasets=1)
    rect = RectangularScaledDiffraction(cfg)
    assert not rect.s1.requires_grad and not rect.s2.requires_grad
    torch.manual_seed(0)
    B, C, N = 2, 1, 32
    x = torch.randn(B, C, N, N, dtype=torch.complex64)
    probe = torch.randn(N, N, dtype=torch.complex64)  # single mode
    scale = torch.ones(B, 1, 1, 1)
    ids = torch.zeros(B, dtype=torch.long)
    out = rect(x=x, I_raw=None, probe=probe, scale=scale, experiment_ids=ids, autograd=True)
    # s1 = s2 = 1 ⇒ exit_wave == scale * probe * x exactly
    exit_wave = scale * (probe * x.real + 1j * (probe * x.imag))
    expected = torch.abs(
        torch.fft.fftshift(torch.fft.fft2(exit_wave, norm="ortho"), dim=(-2, -1))) ** 2
    torch.testing.assert_close(out, expected)


def test_trainable_s1s2_are_parameters_with_grad():
    cfg = dataclasses.replace(ModelConfig(), rect_s1s2_trainable=True, num_datasets=1)
    rect = RectangularScaledDiffraction(cfg)
    assert rect.s1.requires_grad and rect.s2.requires_grad
```
Before running, adapt the `rect(...)` call signature, probe/mode axes, and any mode-summation in `test_frozen_s1s2_reduces_to_plain_scaled_diffraction` to the actual `RectangularScaledDiffraction.forward` read from the source (main ~L705-870; note its s1/s2 indexing goes through a 5-D `view(-1,1,1,1,1)` with a mode axis, so the oracle's broadcasting will need matching). The assertion of substance is fixed: with s1=s2=1 frozen, the output must equal the hand-computed `sum_modes |fftshift(fft2(scale·probe·x, norm='ortho'))|²`.

- [ ] **Step 2: Run to verify failure**

```bash
python -m pytest -q tests/torch/test_training_forward_ablation_knobs.py 2>&1 | tail -5
```
Expected: FAIL (`training_patch_weighting` not a field).

- [ ] **Step 3: Implement**

In `config_params.py` `ModelConfig`, following the existing field style:
```python
training_patch_weighting: Literal['probe', 'uniform'] = 'probe'
rect_s1s2_trainable: bool = True
```
In `model.py` `RectangularScaledDiffraction.__init__`, replace the two Parameter lines:
```python
self.s1 = nn.Parameter(torch.ones(model_config.num_datasets),
                       requires_grad=model_config.rect_s1s2_trainable)
self.s2 = nn.Parameter(torch.ones(model_config.num_datasets),
                       requires_grad=model_config.rect_s1s2_trainable)
```
In `ForwardModel.forward` (~L685), replace the hardcoded kwarg:
```python
use_probe_weights=(self.model_config.training_patch_weighting == 'probe'),
```
(stash `self.model_config = model_config` in `__init__` if not already held). In `train.py`: add `--training-patch-weighting` (choices probe/uniform, default probe) and `--freeze-s1s2` (store_true) to argparse; map into the overrides dict next to the existing `gridsize` entry (`'training_patch_weighting': args.training_patch_weighting`, `'rect_s1s2_trainable': not args.freeze_s1s2`). If main's factory drops unknown override keys, thread them explicitly where `ModelConfig` is constructed/updated.

- [ ] **Step 4: Run tests + a knob-threading smoke**

```bash
python -m pytest -q tests/torch/test_training_forward_ablation_knobs.py
```
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add ptycho_torch/config_params.py ptycho_torch/model.py ptycho_torch/train.py tests/torch/test_training_forward_ablation_knobs.py
git commit -m "feat: config-gate training probe weighting and s1/s2 trainability"
```

### Task 1.3: Inference VarPro bypass (reverse mini-merge from fno-stable)

**Files:**
- Modify: `ptycho_torch/config_params.py` (`InferenceConfig`: add `varpro_scaling: bool = True`)
- Modify: `ptycho_torch/reassembly.py` (`reconstruct_image_barycentric` ~L1238-1247)
- Modify: `ptycho_torch/inference.py` (CLI: `--patch-weighting`, `--no-varpro-scaling`)
- Test: `tests/torch/test_varpro_canvas_scaling_bypass.py` (new)

**Interfaces:**
- Consumes: fno-stable's helper — extract with `git show fno-stable:ptycho_torch/reassembly.py` (function `apply_varpro_canvas_scaling`, ~L986-1001 there) and insert verbatim.
- Produces: `apply_varpro_canvas_scaling(canvas, scaler, *, enabled, verbose) -> (scaled, s1, s2)`; `InferenceConfig.varpro_scaling`.

- [ ] **Step 1: Write failing test** — copy the two bypass tests verbatim from fno-stable (they are branch-portable):

```bash
git show fno-stable:tests/torch/test_varpro_probe_weighted_reassembly.py \
  | sed -n '1,53p' > tests/torch/test_varpro_canvas_scaling_bypass.py   # bypass tests end at L52; L55 starts an unrelated def
python -m pytest -q tests/torch/test_varpro_canvas_scaling_bypass.py 2>&1 | tail -3
```
Expected: FAIL (`apply_varpro_canvas_scaling` not defined).

- [ ] **Step 2: Extract the helper from fno-stable and wire it**

```bash
git show fno-stable:ptycho_torch/reassembly.py | sed -n '980,1005p'   # locate exact bounds first
```
Insert the function verbatim into `reassembly.py`; in `reconstruct_image_barycentric` replace the inline solve/apply block (~L1238-1247) with a call to it, reading `enabled=getattr(inference_config, 'varpro_scaling', True)`. Add `varpro_scaling: bool = True` to `InferenceConfig` beside `patch_weighting`. Add `--patch-weighting {uniform,probe}` and `--no-varpro-scaling` flags to `inference.py`, threading into the `InferenceConfig` it constructs.

- [ ] **Step 3: Run tests**

```bash
python -m pytest -q tests/torch/test_varpro_canvas_scaling_bypass.py
```
Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add ptycho_torch/config_params.py ptycho_torch/reassembly.py ptycho_torch/inference.py tests/torch/test_varpro_canvas_scaling_bypass.py
git commit -m "feat: backport inference varpro_scaling bypass from fno-stable"
```

### Task 1.4: Freeze parity fixtures (backlog B6, captured at the source)

**Files:**
- Create: `scripts/studies/dump_forward_parity_fixtures.py`
- Create: `tests/fixtures/varpro_parity/` (small NPZs, committed)
- Test: `tests/torch/test_forward_parity_fixtures.py` (new)

**Interfaces:**
- Produces: one NPZ per case `{c1_bigF, c1_bigT, c4}` × `{probe, uniform}` × `{frozen, trainable_init}` containing inputs (`x` complex `(B,C,N,N)`, `positions`, `probe`, `scale`, `experiment_ids`, all seeded) and `expected` = `ForwardModel.forward` output at initialization. B=1, C∈{1,4}, N=64 → ≤ ~2 MB total, committable. The **same test file** must pass on fno-stable after Task 2.6 with the mode knobs on.

- [ ] **Step 1: Write the dump script** — deterministic construction only (`torch.manual_seed(0)` per case; probe = the fly001 `probeGuess` loaded from `datasets/fly/fly001.npz`, cast complex64); instantiate `ForwardModel` with the case's `ModelConfig` (`object_big` True/False, `C_forward`/`C_model` per case, `training_patch_weighting`, `rect_s1s2_trainable`); save inputs + output with `np.savez_compressed`.
- [ ] **Step 2: Write the fixture test** — loads each NPZ, rebuilds the config from stored metadata, runs `ForwardModel.forward`, `torch.testing.assert_close(out, expected, rtol=1e-5, atol=1e-6)`.
- [ ] **Step 3: Generate + verify + commit**

```bash
python scripts/studies/dump_forward_parity_fixtures.py --output tests/fixtures/varpro_parity
python -m pytest -q tests/torch/test_forward_parity_fixtures.py
du -sh tests/fixtures/varpro_parity
git add scripts/studies/dump_forward_parity_fixtures.py tests/fixtures/varpro_parity tests/torch/test_forward_parity_fixtures.py
git commit -m "test: freeze forward-model parity fixtures for varpro/probe modes"
```
Expected: PASS; size ≤ ~2 MB (if larger, drop to N=32 and regenerate).

### Task 1.5: Ablation harness (train → sweep → panels)

**Files:**
- Create: `scripts/studies/varpro_probe_ablation_runner.py`
- Test: `tests/torch/test_varpro_probe_ablation_runner.py` (fast unit tests of metrics/plot helpers only)

**Interfaces:**
- Produces: `run_arm(arm: dict, datasets: dict, output_root: Path)` → trains via `subprocess python -m ptycho_torch.train …`, then for each inference variant runs `python -m ptycho_torch.inference … --patch-weighting X [--no-varpro-scaling]`, computes metrics, writes `<output_root>/<arm>/<variant>/metrics.json` + shared `<output_root>/summary.json`, `reconstruction_grid.png` (amp+phase rows: truth + each variant), `error_grid.png`, `invocation.json` (argv, seeds, git commit, dataset provenance path).
- Metrics (global-phase-aligned before MAE): `complex_mae`, `amp_mae`, `phase_mae`, plus `s1`, `s2` actually used. Alignment: `recon *= exp(-1j*angle(vdot(recon_crop, truth_crop)))` on the overlapping crop.
- The harness asserts nothing about orderings — orderings are reported in `summary.json` for the write-up; pytest asserts artifact completeness, finite metrics, and that probe-vs-uniform canvases differ (non-triviality).

- [ ] **Step 1: Write metric/plot helpers with unit tests** (phase-alignment correctness on a synthetic case: `recon = truth * exp(1j*0.7)` → `complex_mae ≈ 0` after alignment). Run: `python -m pytest -q tests/torch/test_varpro_probe_ablation_runner.py` → PASS.
- [ ] **Step 2: Smoke the full loop on one tiny arm** (1 epoch, 64 images):

```bash
python scripts/studies/varpro_probe_ablation_runner.py \
  --arm gs1_frozen --smoke \
  --train-npz .artifacts/varpro_ablation/datasets/fly64_p1e9_gs1_train.npz \
  --test-npz  .artifacts/varpro_ablation/datasets/fly64_p1e9_gs1_test.npz \
  --output-root .artifacts/varpro_ablation/smoke
ls .artifacts/varpro_ablation/smoke/gs1_frozen
```
Expected: checkpoint + 4 variant dirs each with metrics.json + PNGs.

- [ ] **Step 3: Commit** (`git add scripts/studies/varpro_probe_ablation_runner.py tests/torch/test_varpro_probe_ablation_runner.py && git commit -m "feat: varpro/probe ablation harness"`).

### Task 1.6: Execute the matrix

- [ ] **Step 1: Launch the 7 arms sequentially in tmux** (ptycho311 env; ~20-30 epochs, n_images 512, batch 16, cuda; one arm at a time — same `--output-root`, no duplicate runs; the two N=128 arms use the `fly128_*` datasets and run last since they're the slowest):

```bash
# inside tmux, per arm:
python scripts/studies/varpro_probe_ablation_runner.py --arm <ARM> \
  --train-npz .artifacts/varpro_ablation/datasets/fly64_p1e9_gs1_train.npz \
  --test-npz  .artifacts/varpro_ablation/datasets/fly64_p1e9_gs1_test.npz \
  --max-epochs 25 --output-root .artifacts/varpro_ablation/matrix & pid=$!; wait "$pid"
```
Dataset note (Phase-0 outcome): `RawData.from_simulation` is gridsize-1-only, so all raw NPZs are ungrouped gs1 simulations — main's `train.py --gridsize` does the grouping at load time. ALL N=64 arms (gs1 and gs2) use the `fly64_p1e9_gs1_*` pair above; the two N=128 arms use `fly128_p1e9_{train,test}.npz`. See `provenance.json`.
Arms: `gs1_frozen`, `gs1_trainable`, `gs2_neither`, `gs2_probe_frozen`, `gs2_probe_trainable`, `gs2_neither_n128`, `gs2_probe_trainable_n128`.

- [ ] **Step 2: Verify completeness + write the evidence summary** into `docs/plans/2026-07-01-varpro-ablation-phase1-findings.md` (same file as Task 1.1 Step 2 — the main plan doc does not exist on this branch): the metric table from `summary.json`, links to PNGs under `.artifacts/`, the N=64 vs N=128 dyad comparison — does the probe/varpro benefit grow with N — and 2-3 sentences on where varpro/probe effects are visible. Commit the doc update.

---

## Phase 2 — Merge into `fno-stable`

Switch back first: `git switch fno-stable && git submodule update --init --recursive`. Then cherry-pick the branch-portable commits from `varpro-ablation` rather than re-authoring: the fixture dump script/test/fixtures (Task 1.4), the harness (Task 1.5), AND the Phase-1 findings doc commits (`docs/plans/2026-07-01-varpro-ablation-phase1-findings.md`, Tasks 1.1/1.6) should arrive via `git cherry-pick <sha>` (resolve trivial conflicts; knob names are identical by constraint). Without the findings cherry-pick, the Task 2.6 decision record is stranded on the ablation branch.

**Cherry-pick SHAs (updated 2026-07-02 — these post-date the list above and MUST be included or this session's Task 1.6 closeout is stranded on `varpro-ablation`):**
- **`a40676b5`** (`studies: lines + flux-sweep ablation tooling, hardened recon gate, Phase-1 findings`) — carries the REFRESHED `2026-07-01-varpro-ablation-phase1-findings.md` (corrected s1/s2 physics, lines/dead_leaves validation, flux-sweep + s2-sign-flip results) **and** the new tooling: `make_lines_datasets.py`, `make_flux_sweep.py`, `flux_sweep_eval.py`, `plot_flux_sweep.py`, `recon_quality_gate.py`, `diagnose_*.py`, and the `make_synthetic_truth_datasets.py` PHASE_MAX edit. This is the must-cherry-pick (it subsumes the "Tasks 1.1/1.6 findings" bullet above).
- **`67493082`** (`docs: hybrid-resnet representation/scaling comparison extension plan`) — the Phase-2+ extension roadmap `docs/plans/2026-07-01-hybrid-resnet-varpro-probe-extension.md`. Cherry-pick now to pre-place it, OR let it land via its own Task E1 Step 3 commit (do one, not both, to avoid a duplicate).
- Note: `flux_sweep_eval.py` / `recon_quality_gate.py` import the Task 1.5 harness + `diagnose_placement.py`, so cherry-pick them in the same batch as Task 1.5. The SDD ledger (`.superpowers/sdd/plan-amendments-pending.md`, esp. amendment #14) is git-ignored and does NOT travel with any branch — fold its still-pending items into this doc before relying on them.

### Task 2.1: Pre-merge C=4 regression (prerequisite — fno-stable has zero end-to-end C>1 forward coverage)

**Files:** Test: `tests/torch/test_compute_loss_c4_regression.py` (new)

- [ ] Write a test that runs `PtychoPINN_Lightning.compute_loss` on a seeded synthetic batch with `object_big=True`, `C=C_model=C_forward=4`, default knobs, asserting loss is finite and `expected` output hash/tensor matches a value frozen on first run (store as fixture NPZ beside the varpro fixtures). Run twice to confirm determinism, commit. This pins current behavior so every later task can prove "defaults unchanged".

### Task 2.2: Verbatim merge of `reassemble_patches_position_real_probe` (unblocks B3)

**Files:**
- Modify: `ptycho_torch/helper.py`
- Test: `tests/torch/test_training_forward_probe_weighted_reassembly.py` (new — the module name the backlog's verification bundle already cites)

- [ ] **Step 1:** Extract main's function body exactly: `git show main:ptycho_torch/helper.py | sed -n '148,231p'`. Insert into fno-stable `helper.py` after `reassemble_patches_position_real`; resolve callee names against fno-stable's helper (same file layout lineage — expected near-clean). Signature must match the dangling call in `beta_modules/model.py:654` (`probe=`, `use_probe_weights=`, 3-tuple return).
- [ ] **Step 2:** Tests: (a) two overlapping patches, corrupted edge, probe weight low there → probe-weighted seam MAE < uniform (adapt the deterministic two-patch pattern from `test_varpro_probe_weighted_reassembly.py::_assemble_two_patch_overlap`, but through this helper); (b) C=1 single patch: no NaN/div-by-zero, support preserved; (c) return contract `(imgs_merged, boolean_mask, M)` shapes. Run → PASS. Commit (`git commit -m "feat: merge probe-weighted reassembly helper from main"`).

### Task 2.3: B1 — CNN rectangular output mode (`cnn_output_mode`)

**Files:**
- Modify: `ptycho_torch/config_params.py` (`ModelConfig.cnn_output_mode: Literal['amp_phase','real_imag'] = 'amp_phase'`)
- Modify: `ptycho_torch/model.py` (`_resolve_generator_from_config` L529-532 guard; `_predict_complex_patches` L84+ to accept tuple `(real, imag)` `(B,C,H,W)` alongside tensor `(B,H,W,C,2)`; CNN `Autoencoder` head)
- Modify: `ptycho_torch/generators/cnn.py`
- Test: extend `tests/torch/test_model_output_modes.py`, `tests/torch/test_generator_adapter.py`

- [ ] TDD per backlog B1 acceptance: default CNN still amp/phase (existing tests untouched and passing); opt-in real/imag returns `torch.complex(real, imag)`; FNO/hybrid `real_imag` tensor path byte-identical (assert against Task 2.1-style frozen output). Do **not** reuse `generator_output_mode` for CNN (its default `real_imag` would silently flip CNN). Commit.

### Task 2.4: B2 — shared decoder port (opt-in)

**Files:** Modify `config_params.py` (`use_shared_decoder: bool = False`), `ptycho_torch/model.py`; Test: `tests/torch/test_model_output_modes.py`

- [ ] Extract `FeatureRefinementBlock` + `Decoder_shared` verbatim: `git show main:ptycho_torch/model.py` (locate with `grep -n "class FeatureRefinementBlock\|class Decoder_shared"`). Wire `Autoencoder` to select shared vs separate decoders; shared output (2·C channels) split per `cnn_output_mode`. Tests: shape contracts at C=1 and C=4 for both modes; default False leaves current architecture untouched. Commit.

### Task 2.5: B3 — training probe weighting behind config

**Files:** Modify `config_params.py` (`training_patch_weighting: Literal['central_mask','probe','uniform'] = 'central_mask'`), `ptycho_torch/model.py:1224`; Test: extend `test_training_forward_probe_weighted_reassembly.py`

- [ ] Dispatch at the single plug point: `'central_mask'` → existing `reassemble_patches_position_real` (unchanged, default); `'probe'`/`'uniform'` → merged `reassemble_patches_position_real_probe(..., use_probe_weights=(mode=='probe'))`. Preserve the 3-tuple contract for `extract_channels_from_region` and `reassembly.py:143`. Tests: default bit-stable (Task 2.1 fixture passes); probe mode inside `ForwardModel.forward` matches the Phase-1 fixture for the C=4 probe case. Commit.

### Task 2.6: B5 — rectangular scaled forward (`physics_forward_mode`, replace semantics)

**Files:**
- Modify: `config_params.py` (`physics_forward_mode: Literal['amplitude','rectangular_scaled'] = 'amplitude'`; reuse `rect_s1s2_trainable: bool = True`)
- Modify: `ptycho_torch/model.py` (plug point L1237-1260; loss-time physics-scale multiply L1815-1816)
- Test: `tests/torch/test_rectangular_scaled_forward.py` (new — the backlog's cited module name)

- [ ] **Step 1:** Reuse the in-tree implementation: `from ptycho_torch.beta_modules.model import RectangularScaledDiffraction` if the module imports cleanly (`python -c "from ptycho_torch.beta_modules.model import RectangularScaledDiffraction"`); otherwise move the class verbatim into `model.py` (it is identical to main's). Apply the Task 1.2 `requires_grad` patch to whichever copy is used.
- [ ] **Step 2:** Gate: when `'rectangular_scaled'`, `ForwardModel.forward` routes `extracted_patch_objs` through `RectangularScaledDiffraction` and **skips** `ProbeIllumination`, `pad_and_diffract`, `inv_scale`, `alpha/beta`; `compute_loss` for this mode reproduces main's scale routing — fold physics scale into the forward's `output_scale_factor` as `modified_output_scale = sqrt(1/(probe_scaling²·physics_scale + 1e-9))` (main model.py L1538→L1542) instead of the loss-time `pred * physics_scale` multiply. Units and exact routing follow the Task 1.1 Phase-1 finding — the acceptance oracle is the frozen fixture, not a convention argument. Restrict the mode to real/imag-derived objects (FNO/hybrid `real_imag`, or CNN with `cnn_output_mode='real_imag'`); raise `ValueError` with a descriptive message otherwise (fail fast).
- [ ] **Step 3:** Tests: default `'amplitude'` bit-stable (2.1 fixture); rectangular output matches Phase-1 fixtures (frozen + trainable-init cases) within `rtol=1e-5`; ValueError on amp/phase CNN. Commit.

### Task 2.7: B7 — factory/runner plumbing

**Files:** `ptycho_torch/config_factory.py`, `scripts/studies/grid_lines_torch_runner.py`; Test: `tests/torch/test_config_factory.py`, `tests/torch/test_grid_lines_torch_runner.py`

- [ ] `ModelConfig` fields flow via `update_existing_config` automatically — add explicit tests proving `training_patch_weighting`/`physics_forward_mode`/`cnn_output_mode`/`use_shared_decoder`/`rect_s1s2_trainable` survive `create_training_payload` overrides (remember: unknown keys are dropped *silently*). `InferenceConfig` kwargs are explicit — no new inference fields needed (patch_weighting/varpro_scaling already wired). Runner: add `--training-patch-weighting`, `--physics-forward-mode`, `--cnn-output-mode`, `--freeze-s1s2` to `TorchRunnerConfig` + argparse + overrides (three places, mirroring `--output-mode`). Exclude `neuralop_uno` from gridsize-2 assumptions (existing guard at runner L1127). Commit.

### Task 2.8: Cross-branch verification

- [ ] **Step 1:** Cherry-picked fixture test green: `python -m pytest -q tests/torch/test_forward_parity_fixtures.py` with fno-stable configured as `training_patch_weighting='probe'`, `physics_forward_mode='rectangular_scaled'` → all cases match main-frozen tensors.
- [ ] **Step 2:** Re-run one full harness arm on fno-stable (`gs2_probe_trainable`, same datasets, cherry-picked runner + adapter dict mapping conditions→fno-stable overrides) → metrics within tolerance of the Phase-1 run (document the tolerance actually observed; training is stochastic, fixture parity is the hard gate).
- [ ] **Step 3:** Full verification bundle (backlog list): `python -m pytest -q tests/torch/test_generator_adapter.py tests/torch/test_model_output_modes.py tests/torch/test_training_forward_probe_weighted_reassembly.py tests/torch/test_rectangular_scaled_forward.py tests/torch/test_inference_reassembly_parity.py tests/torch/test_inference_reassembly_aggregation.py tests/torch/test_reassembly_multi_patch_parity.py tests/torch/test_reassembly_sign_parity.py tests/torch/test_compute_loss_c4_regression.py` — archive logs per TESTING_GUIDE.

### Task 2.9: B8 — docs, findings, backlog closure

- [ ] Update `docs/workflows/pytorch.md` (mode matrix: current defaults / main-compatible rectangular CNN / inference-only knobs; state explicitly that inference `patch_weighting`/`varpro_scaling` are not training-forward controls), `ptycho_torch/generators/README.md` (output-mode contract incl. CNN tuple), `docs/findings.md` (new finding: rectangular mode replaces the scaling stack; C=1-vs-main residual differences), and mark B1–B8 statuses in the 2026-06-29 backlog with links to evidence. B4 (denominator semantics): record as intentionally deferred — both helpers now coexist behind `training_patch_weighting`, each with pinned tests. Commit.

## Deferred / Non-goals

- B4 default-denominator decision (both modes gated + tested; decision deferred until a consumer needs a single default).
- No FNO/hybrid default changes anywhere; no committed bulky artifacts; no worktrees.
