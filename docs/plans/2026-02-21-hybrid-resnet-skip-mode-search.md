# Hybrid ResNet Skip Connections + Mode Search Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add optional encoder-decoder skip connections to `hybrid_resnet`, add a reproducible mode×skip×width benchmark workflow for `N=128` and `N=256`, and define a staged structural-search extension for depth/downsampling/capacity/skip-design axes.

**Architecture:** Keep default behavior unchanged (`skip_connections=False`) to preserve current baselines/integration expectations. Implement additive skip fusion with lightweight `1x1` projection layers at decoder resolutions (`N/2`, `N`). Expose one boolean knob end-to-end (`hybrid_skip_connections`) through config + CLI, then run a deterministic sweep over `fno_modes × hybrid_skip_connections × fno_width` with fixed probe-mask/loss-normalization controls. Execute Stage A in two steps (full grid on `N=128`, then top-K promotion to `N=256`), then add structural axes one stage at a time (B→E) with bounded per-stage run budgets.

**Tech Stack:** PyTorch/Lightning (`ptycho_torch`), existing grid-lines + NERSC study scripts, `pytest`, runbook-style orchestration, JSON/CSV/Markdown artifacts.

**Companion Design:** `docs/plans/2026-02-21-hybrid-resnet-skip-mode-search-design.md` (normative knob semantics, stage gates, ranking/promotion policy, and artifact contract).

---

### Task 0: Preflight Environment (No Code)

**Files:**
- Modify: none

**Step 1: Verify clean execution context**

Run:
```bash
pwd
git status --short
```
Expected: working directory is repo root; you understand unrelated dirty files and will not revert them.

**Step 2: Start tmux + ptycho311 shell for long commands**

Run:
```bash
tmux new-session -d -s skip_sweep "source ~/miniconda3/etc/profile.d/conda.sh && conda activate ptycho311 && bash"
tmux capture-pane -pt skip_sweep:0
```
Expected: pane output includes activated `ptycho311` prompt.

**Step 3: Commit**

No commit.

---

### Task 1: RED Test for HybridResnet Skip Forward Contract

**Files:**
- Modify: `tests/torch/test_fno_generators.py`
- Test: `tests/torch/test_fno_generators.py`

**Step 1: Write the failing test**

Add:
```python
def test_output_shape_real_imag_with_skip_connections(self):
    model = HybridResnetGeneratorModule(
        in_channels=1,
        out_channels=2,
        hidden_channels=16,
        n_blocks=3,
        modes=4,
        C=4,
        skip_connections=True,
    )
    x = torch.randn(2, 4, 32, 32)
    out = model(x)
    assert out.shape == (2, 32, 32, 4, 2)
```

**Step 2: Run test to verify it fails**

Run:
```bash
pytest tests/torch/test_fno_generators.py::TestHybridResnetGenerator::test_output_shape_real_imag_with_skip_connections -v
```
Expected: FAIL with unexpected kwarg `skip_connections`.

**Step 3: Commit**

No commit (RED only).

---

### Task 2: GREEN HybridResnet Skip Implementation

**Files:**
- Modify: `ptycho_torch/generators/hybrid_resnet.py`
- Test: `tests/torch/test_fno_generators.py`

**Step 1: Write minimal implementation**

Implement:
```python
# __init__ arg
skip_connections: bool = False

self.skip_connections = bool(skip_connections)
if self.skip_connections:
    self.skip_proj_n2 = nn.Conv2d(skip_n2_channels, target_width // 2, kernel_size=1)
    self.skip_proj_n = nn.Conv2d(skip_n_channels, target_width // 4, kernel_size=1)
```

Forward path (minimal):
```python
skip_n = None
skip_n2 = None
...
if self.skip_connections and i == 0:
    skip_n = x
if self.skip_connections and i == 1:
    skip_n2 = x
...
x = self.up1(x)
if self.skip_connections and skip_n2 is not None:
    x = x + self.skip_proj_n2(skip_n2)
x = self.up2(x)
if self.skip_connections and skip_n is not None:
    x = x + self.skip_proj_n(skip_n)
```

**Step 2: Run tests to verify pass**

Run:
```bash
pytest tests/torch/test_fno_generators.py::TestHybridResnetGenerator::test_output_shape_real_imag_with_skip_connections -v
pytest tests/torch/test_fno_generators.py::TestHybridResnetGenerator::test_output_shape_real_imag -v
```
Expected: PASS.

**Step 3: Commit**

```bash
git add ptycho_torch/generators/hybrid_resnet.py tests/torch/test_fno_generators.py
git commit -m "feat(torch): add optional skip connections to hybrid_resnet generator"
```

---

### Task 3: RED Tests for Config/CLI Plumbing of Skip Toggle

**Files:**
- Modify: `tests/torch/test_grid_lines_torch_runner.py`
- Modify: `tests/torch/test_config_bridge.py`
- Test: `tests/torch/test_grid_lines_torch_runner.py`
- Test: `tests/torch/test_config_bridge.py`

**Step 1: Write failing tests**

Add in runner tests:
```python
def test_runner_passes_hybrid_skip_connections(self, tmp_path):
    cfg = TorchRunnerConfig(
        train_npz=tmp_path / "train.npz",
        test_npz=tmp_path / "test.npz",
        output_dir=tmp_path / "output",
        architecture="hybrid_resnet",
        hybrid_skip_connections=True,
    )
    training_config, _ = setup_torch_configs(cfg)
    assert training_config.model.hybrid_skip_connections is True
```

Add in config bridge tests:
```python
def test_model_config_passes_hybrid_skip_connections(params_cfg_snapshot):
    from ptycho_torch.config_params import DataConfig, ModelConfig
    from ptycho_torch import config_bridge
    tf_model = config_bridge.to_model_config(DataConfig(), ModelConfig(hybrid_skip_connections=True))
    assert tf_model.hybrid_skip_connections is True
```

**Step 2: Run tests to verify fail**

Run:
```bash
pytest tests/torch/test_grid_lines_torch_runner.py::TestSetupTorchConfigs::test_runner_passes_hybrid_skip_connections -v
pytest tests/torch/test_config_bridge.py::TestConfigBridgeParity::test_model_config_direct_fields -v
```
Expected: FAIL (missing field in configs/bridge).

**Step 3: Commit**

No commit (RED only).

---

### Task 4: GREEN Config + CLI + Generator Wrapper Plumbing

**Files:**
- Modify: `ptycho/config/config.py`
- Modify: `ptycho_torch/config_params.py`
- Modify: `ptycho_torch/config_bridge.py`
- Modify: `scripts/studies/grid_lines_torch_runner.py`
- Modify: `ptycho_torch/generators/hybrid_resnet.py`
- Test: `tests/torch/test_grid_lines_torch_runner.py`
- Test: `tests/torch/test_config_bridge.py`

**Step 1: Add field to canonical + torch config dataclasses**

Add to both model configs:
```python
hybrid_skip_connections: bool = False
```

**Step 2: Thread field through bridge + runner setup + CLI**

Bridge:
```python
'hybrid_skip_connections': getattr(model, 'hybrid_skip_connections', False),
```

Runner dataclass + setup:
```python
hybrid_skip_connections: bool = False
...
model_config = ModelConfig(..., hybrid_skip_connections=cfg.hybrid_skip_connections)
```

Runner CLI:
```python
parser.add_argument("--hybrid-skip-connections", dest="hybrid_skip_connections", action="store_true", default=False)
parser.add_argument("--no-hybrid-skip-connections", dest="hybrid_skip_connections", action="store_false")
```

Generator wrapper:
```python
hybrid_skip_connections = getattr(model_config, "hybrid_skip_connections", False)
...
skip_connections=hybrid_skip_connections,
```

**Step 3: Run tests to verify pass**

Run:
```bash
pytest tests/torch/test_grid_lines_torch_runner.py::TestSetupTorchConfigs::test_runner_passes_hybrid_skip_connections -v
pytest tests/torch/test_grid_lines_torch_runner.py::TestSetupTorchConfigs::test_runner_accepts_hybrid_resnet -v
pytest tests/torch/test_config_bridge.py -k hybrid_skip_connections -v
```
Expected: PASS.

**Step 4: Commit**

```bash
git add ptycho/config/config.py ptycho_torch/config_params.py ptycho_torch/config_bridge.py scripts/studies/grid_lines_torch_runner.py ptycho_torch/generators/hybrid_resnet.py tests/torch/test_grid_lines_torch_runner.py tests/torch/test_config_bridge.py
git commit -m "feat(torch): plumb hybrid_skip_connections through configs and CLI"
```

---

### Task 5: RED Tests for Mode×Skip×Width Sweep Runbook Helpers

**Files:**
- Create: `tests/studies/test_hybrid_resnet_mode_skip_sweep.py`
- Test: `tests/studies/test_hybrid_resnet_mode_skip_sweep.py`

**Step 1: Write failing helper tests**

Add tests for:
- matrix expansion
- run id formatting
- summary row extraction from `metrics.json`
- N=256 top-train/bottom-test file selection behavior
- top-K selection from `N=128` summary for `N=256` promotion

Example:
```python
def test_build_matrix_modes_and_skips():
    from scripts.studies.runbooks.run_hybrid_resnet_mode_skip_sweep import build_matrix
    rows = build_matrix(
        modes=(12, 16, 24),
        skip_values=(False, True),
        widths=(32, 48, 64),
        ns=(128, 256),
    )
    assert len(rows) == 36
```

**Step 2: Run test to verify fail**

Run:
```bash
pytest tests/studies/test_hybrid_resnet_mode_skip_sweep.py -v
```
Expected: FAIL (module/file not found).

**Step 3: Commit**

No commit (RED only).

---

### Task 6: GREEN Sweep Runbook Implementation (N=128 + N=256)

**Files:**
- Create: `scripts/studies/runbooks/run_hybrid_resnet_mode_skip_sweep.py`
- Test: `tests/studies/test_hybrid_resnet_mode_skip_sweep.py`

**Step 1: Implement runbook with deterministic matrix + artifacts**

Use design constraints from:
- `docs/plans/2026-02-21-hybrid-resnet-skip-mode-search-design.md` (Sections 5-7)

Required behavior:
- Parse:
  - `--modes 12,16,24`
  - `--skip-values off,on`
  - `--widths 32,48,64`
  - `--ns 128,256`
  - `--epochs-n128`, `--epochs-n256`
  - `--top-k-n256` (default 6)
  - `--seed`
  - `--output-root`
  - `--probe-mask/--no-probe-mask` (default off)
  - `--torch-mae-pred-l2-match-target/--no-...` (default off)
- `N=128`: generate integration-style dataset once, then run `grid_lines_torch_runner.py` for each combo in full grid.
- Rank `N=128` runs by primary score (phase-aware composite documented in script) and select top-K for `N=256`.
- `N=256`: call `prepare_hybrid_dataset(..., half="top")` for train NPZ and `prepare_hybrid_dataset(..., half="bottom")` and use its `train_npz` as bottom-half test NPZ, but only for promoted top-K configs.
- Persist invocation artifacts + per-run stdout/stderr logs + `sweep_manifest.json` + `summary.csv` + `summary.md`.

**Step 2: Run tests to verify pass**

Run:
```bash
pytest tests/studies/test_hybrid_resnet_mode_skip_sweep.py -v
```
Expected: PASS.

**Step 3: Commit**

```bash
git add scripts/studies/runbooks/run_hybrid_resnet_mode_skip_sweep.py tests/studies/test_hybrid_resnet_mode_skip_sweep.py
git commit -m "feat(studies): add hybrid_resnet mode-skip sweep runbook"
```

---

### Task 7: Documentation for New Toggle + Sweep Workflow

**Files:**
- Modify: `docs/CONFIGURATION.md`
- Modify: `ptycho_torch/generators/README.md`
- Modify: `docs/studies/index.md`

**Step 1: Add docs entries**

- `docs/CONFIGURATION.md`: add `hybrid_skip_connections` row (default `False`, PyTorch/hybrid_resnet only).
- `ptycho_torch/generators/README.md`: document skip toggle and interaction with `fno_modes`.
- `docs/studies/index.md`: add command recipe for new sweep runbook.

**Step 2: Verify docs are discoverable**

Run:
```bash
rg -n "hybrid_skip_connections|run_hybrid_resnet_mode_skip_sweep" docs ptycho_torch/generators/README.md
```
Expected: references present in all three docs.

**Step 3: Commit**

```bash
git add docs/CONFIGURATION.md ptycho_torch/generators/README.md docs/studies/index.md
git commit -m "docs: add hybrid_resnet skip toggle and mode-skip sweep workflow"
```

---

### Task 8: Verification and Smoke Runs (tmux + ptycho311)

**Files:**
- Modify: none

**Step 1: Run targeted regression tests**

Run in tmux pane:
```bash
pytest tests/torch/test_fno_generators.py -k hybrid_resnet -v
pytest tests/torch/test_grid_lines_torch_runner.py -k "hybrid_resnet or hybrid_skip_connections" -v
pytest tests/studies/test_hybrid_resnet_mode_skip_sweep.py -v
```
Expected: PASS.

**Step 2: Execute small sweep smoke**

Run:
```bash
python scripts/studies/runbooks/run_hybrid_resnet_mode_skip_sweep.py \
  --modes 12,16 \
  --skip-values off,on \
  --widths 32,48 \
  --ns 128 \
  --epochs-n128 5 \
  --output-root outputs/hybrid_resnet_mode_skip_sweep_smoke_n128 \
  --seed 3 \
  --no-probe-mask \
  --no-torch-mae-pred-l2-match-target
```
Expected: `sweep_manifest.json`, `summary.csv`, `summary.md` exist.

**Step 3: Optional N=256 smoke**

Run:
```bash
python scripts/studies/runbooks/run_hybrid_resnet_mode_skip_sweep.py \
  --modes 12 \
  --skip-values off,on \
  --widths 32,48 \
  --ns 256 \
  --epochs-n256 5 \
  --top-k-n256 2 \
  --cameraman-dp /home/ollie/Downloads/nersc/testdata/cameraman256_dp.hdf5 \
  --cameraman-para /home/ollie/Downloads/nersc/testdata/cameraman256_para.hdf5 \
  --output-root outputs/hybrid_resnet_mode_skip_sweep_smoke_n256 \
  --seed 3 \
  --no-probe-mask \
  --no-torch-mae-pred-l2-match-target
```
Expected: per-run metrics + aggregate summary artifacts created.

**Step 4: Final commit (if needed)**

```bash
git add -A
git commit -m "test+feat+docs: hybrid_resnet skip connections and mode-skip sweep workflow"
```

---

### Task 9: Final Full Sweep Command (Hand-off)

**Files:**
- Modify: none

**Step 1: Record full production command in plan handoff**

```bash
python scripts/studies/runbooks/run_hybrid_resnet_mode_skip_sweep.py \
  --modes 12,16,24 \
  --skip-values off,on \
  --widths 32,48,64 \
  --ns 128,256 \
  --epochs-n128 20 \
  --epochs-n256 40 \
  --top-k-n256 6 \
  --cameraman-dp /home/ollie/Downloads/nersc/data/cameraman256_dp.hdf5 \
  --cameraman-para /home/ollie/Downloads/nersc/data/cameraman256_para.hdf5 \
  --output-root outputs/hybrid_resnet_mode_skip_sweep_full_20260221 \
  --seed 3 \
  --no-probe-mask \
  --no-torch-mae-pred-l2-match-target
```

**Step 2: Confirm artifacts**

Run:
```bash
find outputs/hybrid_resnet_mode_skip_sweep_full_20260221 -maxdepth 2 -type f | rg "sweep_manifest.json|summary.csv|summary.md"
```
Expected: all aggregate files present.

Run:
```bash
cat outputs/hybrid_resnet_mode_skip_sweep_full_20260221/summary.md | rg "N=128|N=256|top-k"
```
Expected: report contains full-grid `N=128` section and promoted top-K `N=256` section.

**Step 3: Commit**

No commit (execution-only handoff).

---

### Task 10: Add Structural-Axis Hooks to Sweep Runbook (No Cartesian Explosion)

**Files:**
- Modify: `scripts/studies/runbooks/run_hybrid_resnet_mode_skip_sweep.py`
- Modify: `tests/studies/test_hybrid_resnet_mode_skip_sweep.py`
- Modify: `docs/studies/index.md`

**Step 1: Add CLI hooks for staged axes with safe defaults**

Follow knob semantics in:
- `docs/plans/2026-02-21-hybrid-resnet-skip-mode-search-design.md` (Section 4)

Add optional arguments (defaults preserve Stage A behavior):
- `--fno-blocks-values` (default: `4`)
- `--downsample-schedule-values` (default: `2`)  # number of encoder downsample steps
- `--max-hidden-values` (default: `none`)  # maps to `max_hidden_channels`
- `--resnet-width-values` (default: `none`)
- `--resnet-blocks-values` (default: `6`)
- `--skip-style-values` (default: `add`)  # `add|concat|gated_add`

Add `--stage-id` metadata label (`A|B|C|D|E`) to manifest/summary.

**Step 2: Add matrix builder constraints**

Implement guardrails:
- exactly one structural axis may vary per stage B-E
- Stage A varies only `{modes, widths, skip on/off}`
- raise actionable error if multiple structural axes contain >1 value in one stage

**Step 3: Add/adjust tests**

Run:
```bash
pytest tests/studies/test_hybrid_resnet_mode_skip_sweep.py -k "matrix or guardrail or stage_id" -v
```
Expected: PASS.

**Step 4: Commit**

```bash
git add scripts/studies/runbooks/run_hybrid_resnet_mode_skip_sweep.py tests/studies/test_hybrid_resnet_mode_skip_sweep.py docs/studies/index.md
git commit -m "feat(studies): add staged structural-axis hooks to hybrid_resnet sweep runbook"
```

---

### Task 11: Stage B Search (Axis 1: `fno_blocks`)

**Files:**
- Modify: none (execution + artifacts)

**Step 1: Run Stage B at N=128 on promoted Stage A configs**

Run:
```bash
python scripts/studies/runbooks/run_hybrid_resnet_mode_skip_sweep.py \
  --stage-id B \
  --modes 12,16,24 \
  --skip-values off,on \
  --widths 32,48,64 \
  --fno-blocks-values 4,5,6 \
  --ns 128 \
  --epochs-n128 20 \
  --top-k-n256 0 \
  --output-root outputs/hybrid_resnet_stageB_fno_blocks_n128_20260221 \
  --seed 3 \
  --no-probe-mask \
  --no-torch-mae-pred-l2-match-target
```
Expected: `summary.md` includes `stage_id=B` and `fno_blocks` column.

**Step 2: Promote top-K and run N=256**

Run:
```bash
python scripts/studies/runbooks/run_hybrid_resnet_mode_skip_sweep.py \
  --stage-id B \
  --modes 12,16,24 \
  --skip-values off,on \
  --widths 32,48,64 \
  --fno-blocks-values 4,5,6 \
  --ns 256 \
  --epochs-n256 40 \
  --top-k-n256 6 \
  --cameraman-dp /home/ollie/Downloads/nersc/data/cameraman256_dp.hdf5 \
  --cameraman-para /home/ollie/Downloads/nersc/data/cameraman256_para.hdf5 \
  --output-root outputs/hybrid_resnet_stageB_fno_blocks_n256_20260221 \
  --seed 3 \
  --no-probe-mask \
  --no-torch-mae-pred-l2-match-target
```

**Step 3: Commit**

No commit (execution-only stage).

---

### Task 12: Stage C Search (Axis 2: Downsampling Schedule / Bottleneck Resolution + Downsampling Operator)

**Files:**
- Modify: `ptycho_torch/generators/hybrid_resnet.py`
- Modify: `ptycho/config/config.py`
- Modify: `ptycho_torch/config_params.py`
- Modify: `ptycho_torch/config_bridge.py`
- Modify: `scripts/studies/grid_lines_torch_runner.py`
- Modify: `tests/torch/test_fno_generators.py`
- Modify: `tests/torch/test_grid_lines_torch_runner.py`
- Modify: `tests/torch/test_config_bridge.py`

**Step 1: Add `hybrid_downsample_steps` config/CLI plumbing**

Add model field:
```python
hybrid_downsample_steps: int = 2
```
Generator uses it to choose `1` (`N->N/2`) vs `2` (`N->N/4`) downsample steps.

Also add:
```python
hybrid_downsample_op: Literal["stride_conv", "avgpool_conv", "blurpool_conv"] = "stride_conv"
```
Generator uses this to select operator family for each downsample stage.

**Step 2: Add RED/GREEN tests**

Cover:
- valid range `[1,2]` for current implementation
- output shape invariance
- runner/config bridge propagation
- invalid `hybrid_downsample_op` rejected
- output shape invariance for each `hybrid_downsample_op`

Run:
```bash
pytest tests/torch/test_fno_generators.py -k "hybrid_downsample_steps" -v
pytest tests/torch/test_fno_generators.py -k "hybrid_downsample_op" -v
pytest tests/torch/test_grid_lines_torch_runner.py -k "downsample_steps or downsample_op" -v
pytest tests/torch/test_config_bridge.py -k "hybrid_downsample_steps or hybrid_downsample_op" -v
```
Expected: PASS.

**Step 3: Execute Stage C runs**

Sub-stage C1 (schedule): use `--downsample-schedule-values 1,2` while fixing all other structural axes to best Stage B config.

Sub-stage C2 (operator): lock best schedule from C1, then run:
- `--downsample-op-values stride_conv,avgpool_conv,blurpool_conv`

Stage budget:
- N=128: max 12 runs for C1 + max 12 runs for C2
- N=256: top 4 from C2 only

**Step 4: Commit**

```bash
git add ptycho_torch/generators/hybrid_resnet.py ptycho/config/config.py ptycho_torch/config_params.py ptycho_torch/config_bridge.py scripts/studies/grid_lines_torch_runner.py tests/torch/test_fno_generators.py tests/torch/test_grid_lines_torch_runner.py tests/torch/test_config_bridge.py
git commit -m "feat(torch): add hybrid_resnet downsample-step schedule control"
```

---

### Task 13: Stage D Search (Axes 3 + 4: Capacity and Decoder Depth)

**Files:**
- Modify: `ptycho_torch/generators/hybrid_resnet.py`
- Modify: `ptycho/config/config.py`
- Modify: `ptycho_torch/config_params.py`
- Modify: `ptycho_torch/config_bridge.py`
- Modify: `scripts/studies/grid_lines_torch_runner.py`
- Modify: `tests/torch/test_fno_generators.py`
- Modify: `tests/torch/test_grid_lines_torch_runner.py`
- Modify: `tests/torch/test_config_bridge.py`

**Step 1: Axis 3 (capacity) sub-stage**

Evaluate one capacity knob at a time:
- `max_hidden_channels` values: `none,256,512`, or
- `resnet_width` values: `none,192,256` (only if divisibility constraints pass).

Keep `resnet_blocks` fixed for this sub-stage.

**Step 2: Axis 4 (decoder depth) sub-stage**

Add/plumb:
```python
hybrid_resnet_blocks: int = 6
```
Sweep `4,6,8` using best capacity setting from Step 1.

**Step 3: Run bounded stage budgets**

Budget rule:
- N=128: max 18 runs per sub-stage
- N=256: top 4 only

**Step 4: Commit**

```bash
git add ptycho_torch/generators/hybrid_resnet.py ptycho/config/config.py ptycho_torch/config_params.py ptycho_torch/config_bridge.py scripts/studies/grid_lines_torch_runner.py tests/torch/test_fno_generators.py tests/torch/test_grid_lines_torch_runner.py tests/torch/test_config_bridge.py
git commit -m "feat(torch): add hybrid_resnet capacity and decoder-depth controls"
```

---

### Task 14: Stage E Search (Axis 5: Skip-Connection Design)

**Files:**
- Modify: `ptycho_torch/generators/hybrid_resnet.py`
- Modify: `ptycho/config/config.py`
- Modify: `ptycho_torch/config_params.py`
- Modify: `ptycho_torch/config_bridge.py`
- Modify: `scripts/studies/grid_lines_torch_runner.py`
- Modify: `tests/torch/test_fno_generators.py`
- Modify: `tests/torch/test_grid_lines_torch_runner.py`
- Modify: `tests/torch/test_config_bridge.py`

**Step 1: Add `hybrid_skip_style` enum**

Add model field:
```python
hybrid_skip_style: Literal["add", "concat", "gated_add"] = "add"
```

Implementation guidance:
- `add`: current behavior
- `concat`: concat then `1x1` projection
- `gated_add`: learnable scalar/channel gate initialized near zero

**Step 2: Add RED/GREEN tests**

Cover:
- shape contract unchanged for each style
- invalid style rejected
- propagation through runner/config bridge

Run:
```bash
pytest tests/torch/test_fno_generators.py -k "skip_style" -v
pytest tests/torch/test_grid_lines_torch_runner.py -k "skip_style" -v
pytest tests/torch/test_config_bridge.py -k "skip_style" -v
```

**Step 3: Execute Stage E**

Run skip styles on best config from Stage D, budget:
- N=128: all 3 styles
- N=256: top 2 styles only

**Step 4: Commit**

```bash
git add ptycho_torch/generators/hybrid_resnet.py ptycho/config/config.py ptycho_torch/config_params.py ptycho_torch/config_bridge.py scripts/studies/grid_lines_torch_runner.py tests/torch/test_fno_generators.py tests/torch/test_grid_lines_torch_runner.py tests/torch/test_config_bridge.py
git commit -m "feat(torch): add hybrid_resnet skip-style variants for staged search"
```

---

### Task 15: Stage Governance and Stop/Go Criteria

**Files:**
- Modify: `docs/studies/index.md`
- Modify: `docs/plans/2026-02-21-hybrid-resnet-skip-mode-search.md`

**Step 1: Define promotion criteria between stages**

Use the canonical policy in:
- `docs/plans/2026-02-21-hybrid-resnet-skip-mode-search-design.md` (Section 6)

Use explicit gates:
- must beat previous stage baseline on amplitude `MAE` and `MSE`
- no catastrophic phase regression (phase SSIM drop > 0.03 vs stage baseline)
- runtime/params within budget envelope recorded in summary

**Step 2: Define hard stop conditions**

Stop expansion if:
- two consecutive stages show <1% relative gain on primary metric, or
- all candidate configs at new stage regress on both amplitude MAE and MSE at `N=256`.

**Step 3: Commit**

```bash
git add docs/studies/index.md docs/plans/2026-02-21-hybrid-resnet-skip-mode-search.md
git commit -m "docs(studies): add staged structural-search governance for hybrid_resnet sweep"
```
