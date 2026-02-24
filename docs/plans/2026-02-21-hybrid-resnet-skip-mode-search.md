# Hybrid ResNet Skip Connections + Mode Search Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add optional encoder-decoder skip connections to `hybrid_resnet`, add a reproducible mode×skip×width benchmark workflow for `N=128` and `N=256`, and define a staged structural-search extension for depth/downsampling/capacity/skip-design axes.

**Architecture:** Keep default behavior unchanged (`skip_connections=False`) to preserve current baselines/integration expectations. Implement additive skip fusion with lightweight `1x1` projection layers at decoder resolutions (`N/2`, `N`). Expose one boolean knob end-to-end (`hybrid_skip_connections`) through config + CLI, then run a deterministic sweep over `fno_modes × hybrid_skip_connections × fno_width` with fixed probe-mask/loss-normalization controls. Make dataset choice explicit via named dataset profiles so the same sweep can run on multiple failure-mode regimes. Execute Stage A in two steps (full grid on `N=128`, then top-K promotion to `N=256`), then add structural axes one stage at a time (B→E) with bounded per-stage run budgets.

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

**Step 2: Start tmux shell and bootstrap the runtime env once**

Run:
```bash
tmux new-session -d -s skip_sweep "bash -lc 'source ~/miniconda3/etc/profile.d/conda.sh && conda activate ptycho311 && python -V; exec bash'"
tmux capture-pane -pt skip_sweep:0
```
Expected: pane output shows `python -V` from the `ptycho311` session PATH interpreter.
All subsequent commands in this plan should still use plain `python ...` (no interpreter wrappers).

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
pytest tests/torch/test_fno_generators.py::TestHybridResnetGenerator::test_skip_connections_default_false_parity -v
```
Expected: PASS.

Add parity regression assertion for default behavior:
```python
def test_skip_connections_default_false_parity(self):
    torch.manual_seed(7)
    model_default = HybridResnetGeneratorModule(
        in_channels=1, out_channels=2, hidden_channels=16, n_blocks=3, modes=4, C=4
    )
    torch.manual_seed(7)
    model_explicit = HybridResnetGeneratorModule(
        in_channels=1, out_channels=2, hidden_channels=16, n_blocks=3, modes=4, C=4,
        skip_connections=False,
    )
    x = torch.randn(2, 4, 32, 32)
    y_default = model_default(x)
    y_explicit = model_explicit(x)
    assert torch.allclose(y_default, y_explicit, atol=1e-6, rtol=1e-6)
```

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
pytest tests/torch/test_config_bridge.py::TestConfigBridgeParity::test_model_config_passes_hybrid_skip_connections -v
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
- dataset-profile expansion and per-profile summary aggregation

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
  - `--modes 12,16,24` (must also support higher-frequency probes such as `32,48` at `N=256`)
  - `--skip-values off,on`
  - `--widths 32,48,64`
  - `--ns 128,256`
  - `--dataset-profiles-n128` (default: `integration_grid_lines_n128_v1`)
  - `--dataset-profiles-n256` (default: `cameraman256_halfsplit_v1`)
  - `--epochs-n128`, `--epochs-n256`
  - `--top-k-n256` (default 6)
  - `--seed`
  - `--output-root`
  - `--probe-mask/--no-probe-mask` (default off)
  - `--torch-mae-pred-l2-match-target/--no-...` (default off)
- Dataset profile resolution contract:
  - profile ids are orchestration-level aliases only (runbook/sweep layer).
  - leaf CLIs remain path-first and unchanged:
    - `grid_lines_compare_wrapper.py`: `--dataset-source` plus `--train-data/--test-data` where required.
    - `grid_lines_torch_runner.py`: `--train-npz/--test-npz`.
  - explicit caller-provided dataset paths override profile defaults.
- `N=128`: resolve dataset profile(s), then run each combo against each profile.
  - Default profile `integration_grid_lines_n128_v1` MUST use the exact integration-fixture generation recipe:
    - `N=128`, `gridsize=1`
    - `probe_npz=datasets/Run1084_recon3_postPC_shrunk_3.npz`
    - `nimgs_train=2`, `nimgs_test=1`, `nphotons=1e9`
    - `probe_source=custom`, `probe_smoothing_sigma=0.5`, `probe_scale_mode=pad_extrapolate`, `set_phi=True`
    - deterministic dataset seed (recorded in manifest; default `3`)
  - Additional N=128 profiles:
    - `fly001_external_n128_top_bottom_v1` (external NPZ split; caller supplies paths)
    - `custom_npz_pair_n128` (caller-supplied `train.npz` / `test.npz`)
- Rank `N=128` runs using the Section-6 policy from the companion design
  (lexicographic amplitude ranking with phase-SSIM guardrail), then select top-K for `N=256`.
- `N=256`: resolve dataset profile(s), then run promoted top-K configs for each profile.
  - Default profile `cameraman256_halfsplit_v1`:
    - call `prepare_hybrid_dataset(..., half="top")` for train NPZ
    - call `prepare_hybrid_dataset(..., half="bottom")` and use `train_npz` as bottom-half test NPZ
  - Additional N=256 profiles:
    - `custom_npz_pair_n256` (caller-supplied `train.npz` / `test.npz`)
- Stage progression contract:
  - Stage `A` seeds the first promoted set directly from its own `N=128` results.
  - Stages `B-E` MUST load their baseline/promoted config set from a prior-stage `N=128` summary via `--promotion-source-summary`.
  - For stages `B-E`, `--ns 256`-only runs are invalid without `--promotion-source-summary`.
- Persist invocation artifacts + per-run stdout/stderr logs + `sweep_manifest.json` + `summary.csv` + `summary.md`.
- Persist visual-evidence collation artifacts under:
  - `<output-root>/comparison_bundle/shared_pngs/`
  - descriptive filenames containing at least `{stage_id, N, dataset_profile, run_id}`.
- Include per-profile and aggregate ranking views:
  - per-profile metrics/ranks
  - macro aggregate rank across profiles (median rank; tie-break by mean primary score).

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

### Task 8: Verification and Smoke Runs

**Files:**
- Modify: none

**Step 0: GPU fail-fast preflight**

Run in tmux pane:
```bash
python - <<'PY'
import torch
assert torch.cuda.is_available(), "CUDA GPU is required for this study plan"
print("CUDA available:", torch.cuda.get_device_name(0))
PY
```
Expected: PASS; abort study execution immediately if CUDA is unavailable.

**Step 1: Run targeted regression tests**

Run in tmux pane:
```bash
pytest tests/torch/test_fno_generators.py -k hybrid_resnet -v
pytest tests/torch/test_grid_lines_torch_runner.py -k "hybrid_resnet or hybrid_skip_connections" -v
pytest tests/studies/test_hybrid_resnet_mode_skip_sweep.py -v
pytest -m integration -v
```
Expected: PASS.
Archive the integration-marker stdout/stderr log under the active plan artifact location.

**Step 2: Execute small sweep smoke**

Run:
```bash
python scripts/studies/runbooks/run_hybrid_resnet_mode_skip_sweep.py \
  --modes 12,16 \
  --skip-values off,on \
  --widths 32,48 \
  --ns 128 \
  --dataset-profiles-n128 integration_grid_lines_n128_v1 \
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
  --ns 256 \
  --promotion-source-summary outputs/hybrid_resnet_mode_skip_sweep_smoke_n128/summary.csv \
  --dataset-profiles-n256 cameraman256_halfsplit_v1 \
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
git status --short
git add \
  ptycho_torch/generators/hybrid_resnet.py \
  ptycho/config/config.py \
  ptycho_torch/config_params.py \
  ptycho_torch/config_bridge.py \
  scripts/studies/grid_lines_torch_runner.py \
  scripts/studies/runbooks/run_hybrid_resnet_mode_skip_sweep.py \
  tests/torch/test_fno_generators.py \
  tests/torch/test_grid_lines_torch_runner.py \
  tests/torch/test_config_bridge.py \
  tests/studies/test_hybrid_resnet_mode_skip_sweep.py \
  docs/CONFIGURATION.md \
  ptycho_torch/generators/README.md \
  docs/studies/index.md \
  docs/plans/2026-02-21-hybrid-resnet-skip-mode-search.md \
  docs/plans/2026-02-21-hybrid-resnet-skip-mode-search-design.md
git status --short
git commit -m "test+feat+docs: hybrid_resnet skip connections and mode-skip sweep workflow"
```

---

### Task 9: Final Full Sweep Command (Hand-off)

**Files:**
- Modify: none

**Step 1: Record full production command in plan handoff**

```bash
python scripts/studies/runbooks/run_hybrid_resnet_mode_skip_sweep.py \
  --modes 12,16,24,32,48 \
  --skip-values off,on \
  --widths 32,48,64 \
  --ns 128,256 \
  --dataset-profiles-n128 integration_grid_lines_n128_v1,fly001_external_n128_top_bottom_v1 \
  --dataset-profiles-n256 cameraman256_halfsplit_v1 \
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

**Step 2: Add targeted N=256 high-mode probe (diagnostic)**

Run:
```bash
python scripts/studies/runbooks/run_hybrid_resnet_mode_skip_sweep.py \
  --modes 32,48 \
  --skip-values off,on \
  --widths 48 \
  --ns 256 \
  --dataset-profiles-n256 cameraman256_halfsplit_v1 \
  --epochs-n256 20 \
  --top-k-n256 0 \
  --cameraman-dp /home/ollie/Downloads/nersc/data/cameraman256_dp.hdf5 \
  --cameraman-para /home/ollie/Downloads/nersc/data/cameraman256_para.hdf5 \
  --output-root outputs/hybrid_resnet_mode_skip_sweep_n256_highmode_probe_20260221 \
  --seed 3 \
  --no-probe-mask \
  --no-torch-mae-pred-l2-match-target
```
Expected: explicit N=256-only high-frequency sensitivity evidence.

**Step 3: Confirm artifacts**

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

Run:
```bash
find outputs/hybrid_resnet_mode_skip_sweep_full_20260221/comparison_bundle/shared_pngs -maxdepth 1 -type f | head
```
Expected: shared visual-comparison PNGs are present with descriptive filenames.

**Step 4: Finalist seed-robustness check (must-run before stage promotion)**

From `summary.csv`, pick top-3 `N=256` finalists and rerun each finalist config with seeds `11` and `17` (same epochs and confounder settings) using single-value `--modes/--skip-values/--widths` commands.
Expected:
- ranking is stable (no major order inversion among finalists),
- no catastrophic regressions in amplitude MAE/MSE or phase SSIM guardrail.

Archive these reruns under:
- `outputs/hybrid_resnet_mode_skip_sweep_seed_robustness_<date>/`

**Step 5: Commit**

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
- `--downsample-op-values` (default: `stride_conv`)  # `stride_conv|avgpool_conv|blurpool_conv`
- `--max-hidden-values` (default: `none`)  # maps to `max_hidden_channels`
- `--resnet-width-values` (default: `none`)
- `--resnet-blocks-values` (default: `6`)
- `--skip-style-values` (default: `add`)  # `add|concat|gated_add`
- `--dataset-profiles-n128` (default: `integration_grid_lines_n128_v1`)
- `--dataset-profiles-n256` (default: `cameraman256_halfsplit_v1`)
- `--promotion-source-summary` (default: empty; required for stages `B-E`, and always required when stage `B-E` runs with `--ns 256` only)

Add `--stage-id` metadata label (`A|B|C|D|E`) to manifest/summary.
Persist dataset provenance in manifest:
- profile ids
- resolved input paths and which path-based args were emitted (`--train-data/--test-data` vs `--train-npz/--test-npz`)
- SHA256 for train/test NPZ (or source HDF5 pair where applicable)

**Step 2: Add matrix builder constraints**

Implement guardrails:
- exactly one structural axis may vary per stage B-E
- Stage A varies only `{modes, widths, skip on/off}`
- For stages B-E, non-active structural axes are inherited from `--promotion-source-summary`; multi-value lists on non-active axes are rejected.
- For stages B-E, reject `--ns 256`-only invocations when `--promotion-source-summary` is missing.
- raise actionable error if multiple structural axes contain >1 value in one stage

**Step 3: Add/adjust tests**

Add explicit invocation-provenance assertions in
`tests/studies/test_hybrid_resnet_mode_skip_sweep.py`:
- `invocation.json` is created at the run root with expected script path/argv/parsed args.
- `invocation.sh` is created at the run root and contains a reconstructible command.

Run:
```bash
pytest tests/studies/test_hybrid_resnet_mode_skip_sweep.py -k "invocation or matrix or guardrail or stage_id" -v
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
  --promotion-source-summary outputs/hybrid_resnet_mode_skip_sweep_full_20260221/summary.csv \
  --fno-blocks-values 4,5,6 \
  --ns 128 \
  --dataset-profiles-n128 integration_grid_lines_n128_v1,fly001_external_n128_top_bottom_v1 \
  --epochs-n128 20 \
  --top-k-n256 0 \
  --output-root outputs/hybrid_resnet_stageB_fno_blocks_n128_20260221 \
  --seed 3 \
  --no-probe-mask \
  --no-torch-mae-pred-l2-match-target
```
Expected: `summary.md` includes `stage_id=B` and `fno_blocks` column.
Non-axis knobs (`modes`, `skip`, `width`) come from `--promotion-source-summary`; do not re-sweep them in Stage B.

**Step 2: Promote top-K and run N=256**

Run:
```bash
python scripts/studies/runbooks/run_hybrid_resnet_mode_skip_sweep.py \
  --stage-id B \
  --promotion-source-summary outputs/hybrid_resnet_stageB_fno_blocks_n128_20260221/summary.csv \
  --ns 256 \
  --dataset-profiles-n256 cameraman256_halfsplit_v1 \
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

Sub-stage C1 (schedule): use `--downsample-schedule-values 1,2` while fixing all other structural axes to promoted Stage B configs loaded via `--promotion-source-summary <stageB_n128_summary.csv>`.

Sub-stage C2 (operator): lock best schedule from C1 (from `--promotion-source-summary <stageC1_n128_summary.csv>`), then run:
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
