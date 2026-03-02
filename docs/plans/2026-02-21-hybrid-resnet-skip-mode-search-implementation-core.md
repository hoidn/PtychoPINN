# Hybrid ResNet Skip Connections + Mode Search - Implementation Core Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add optional encoder-decoder skip connections to `hybrid_resnet`, add a reproducible mode×skip×width benchmark workflow for `N=128` and `N=256`, and define a staged structural-search extension for depth/downsampling/capacity/skip-design axes.

**Architecture:** Keep default behavior unchanged (`skip_connections=False`) to preserve current baselines/integration expectations. Implement additive skip fusion with lightweight `1x1` projection layers at decoder resolutions (`N/2`, `N`). Expose one boolean knob end-to-end (`hybrid_skip_connections`) through Torch-only runner/execution config + CLI (do not bridge this knob into TensorFlow/canonical model contracts), then run a deterministic sweep over `fno_modes × hybrid_skip_connections × fno_width` with fixed probe-mask/loss-normalization controls. Make dataset choice explicit via named dataset profiles so the same sweep can run on multiple failure-mode regimes. Execute Stage A in two steps (full grid on `N=128`, then top-K promotion to `N=256`), then add structural axes one stage at a time (B→E) with bounded per-stage run budgets. Promotion governance: keep broad sweeps single-seed (`seed=3` default), then run boundary seed reranks (`top-K + next 2`, seeds `11` and `17`) before every promotion, and promote by median rank across seeds. Governance decision for this initiative: new Stage C-E knobs stay Torch-only (runner/execution/model paths) unless a follow-up plan explicitly approves cross-backend bridge expansion.

**Tech Stack:** PyTorch/Lightning (`ptycho_torch`), existing grid-lines + NERSC study scripts, `pytest`, runbook-style orchestration, JSON/CSV/Markdown artifacts.

**Companion Design:** `docs/plans/2026-02-21-hybrid-resnet-skip-mode-search-design.md` (normative knob semantics, stage gates, ranking/promotion policy, and artifact contract).

## Scope

This split document owns Tasks 0-8 (preflight, RED/GREEN implementation, plumbing, runbook bring-up, documentation sync, and smoke verification).

## Shared Contracts

- Follow the Test Evidence Contract below for every `pytest` command in this document.
- Normative ranking/promotion rules live in `docs/plans/2026-02-21-hybrid-resnet-skip-mode-search-design.md`.
- Hub document: `docs/plans/2026-02-21-hybrid-resnet-skip-mode-search.md`.


- For every `pytest` selector in this plan, run a matching `--collect-only` command before execution.
- Save both collect and execution logs under:
  - `docs/plans/2026-02-21-hybrid-resnet-skip-mode-search/reports/<timestamp>/`
- Required naming:
  - `<selector_slug>_collect.log`
  - `<selector_slug>_run.log`
- Do not mark a selector complete if its collect-only log shows `collected 0 items`.
- Regenerate `docs/development/TEST_SUITE_INDEX.md` via `python scripts/tools/generate_test_index.py > docs/development/TEST_SUITE_INDEX.md` before every commit that adds/renames/removes tests.
- Command-block convention:
  - `pytest ...` lines shown in tasks are execution commands.
  - For each such line, run the implied collect pair first:
    - `pytest <same selector> --collect-only -v | tee "${REPORT_DIR}/<selector_slug>_collect.log"`
    - `pytest <same selector> -v | tee "${REPORT_DIR}/<selector_slug>_run.log"`

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

**Step 2: Initialize submodules only when needed (new worktree/clone or missing content)**

Run:
```bash
# Run this block only in a new worktree/clone or when required submodule paths are missing.
test -e ptycho/FRC || git submodule update --init --recursive
test -e ptycho/FRC || { echo "Missing ptycho/FRC submodule content"; exit 1; }
```
Expected: required paths (for example `ptycho/FRC`) exist; existing initialized worktrees can skip submodule initialization.

**Step 3: Start tmux shell and bootstrap the runtime env once**

Run:
```bash
# Use a dedicated plan session and reset pane command each run for deterministic env bootstrap.
tmux has-session -t skip_sweep 2>/dev/null || tmux new-session -d -s skip_sweep
tmux respawn-pane -k -t skip_sweep:0 "bash -lc 'source ~/miniconda3/etc/profile.d/conda.sh && conda activate ptycho311 && python -V; exec bash'"
sleep 1
tmux capture-pane -pt skip_sweep:0
```
Expected: pane output shows `python -V` from the `ptycho311` session PATH interpreter for both fresh and reused `skip_sweep` sessions.
All subsequent commands in this plan should still use plain `python ...` (no interpreter wrappers).

**Step 4: Initialize timestamped report directory for test evidence**

Run:
```bash
TS="$(date -u +%Y%m%dT%H%M%SZ)"
REPORT_DIR="docs/plans/2026-02-21-hybrid-resnet-skip-mode-search/reports/${TS}"
mkdir -p "${REPORT_DIR}"
echo "${REPORT_DIR}"
```
Expected: printed path is used for all `*_collect.log` and `*_run.log` artifacts in this session.

**Step 5: Commit**

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

def test_skip_connections_true_changes_output(self):
    torch.manual_seed(11)
    model_off = HybridResnetGeneratorModule(
        in_channels=1, out_channels=2, hidden_channels=16, n_blocks=3, modes=4, C=4,
        skip_connections=False,
    )
    torch.manual_seed(11)
    model_on = HybridResnetGeneratorModule(
        in_channels=1, out_channels=2, hidden_channels=16, n_blocks=3, modes=4, C=4,
        skip_connections=True,
    )
    x = torch.randn(2, 4, 32, 32)
    y_off = model_off(x)
    y_on = model_on(x)
    assert not torch.allclose(y_off, y_on, atol=1e-6, rtol=1e-6)
```

**Step 2: Run test to verify it fails**

Run:
```bash
pytest tests/torch/test_fno_generators.py::TestHybridResnetGenerator::test_output_shape_real_imag_with_skip_connections -v
pytest tests/torch/test_fno_generators.py::TestHybridResnetGenerator::test_skip_connections_true_changes_output -v
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
    # Use lazy 1x1 projections so channel counts come from actual tapped tensors.
    self.skip_proj_n2 = nn.LazyConv2d(target_width // 2, kernel_size=1)
    self.skip_proj_n = nn.LazyConv2d(target_width // 4, kernel_size=1)
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
pytest tests/torch/test_fno_generators.py::TestHybridResnetGenerator::test_skip_connections_true_changes_output -v
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
python scripts/tools/generate_test_index.py > docs/development/TEST_SUITE_INDEX.md
git add ptycho_torch/generators/hybrid_resnet.py tests/torch/test_fno_generators.py docs/development/TEST_SUITE_INDEX.md
git commit -m "feat(torch): add optional skip connections to hybrid_resnet generator"
```

---

### Task 3: RED Tests for Config/CLI Plumbing of Skip Toggle

**Files:**
- Modify: `tests/torch/test_grid_lines_torch_runner.py`
- Modify: `tests/test_grid_lines_compare_wrapper.py`
- Test: `tests/torch/test_grid_lines_torch_runner.py`
- Test: `tests/test_grid_lines_compare_wrapper.py`

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
    training_config, execution_config = setup_torch_configs(cfg)
    assert getattr(execution_config, "hybrid_skip_connections", False) is True
    assert not hasattr(training_config.model, "hybrid_skip_connections")

def test_workflow_forwards_hybrid_skip_connections_to_factory(monkeypatch, tmp_path):
    # _train_with_lightning should pass hybrid_skip_connections from execution config
    # into create_training_payload(..., overrides=...) so generator behavior can change.
    ...
```

Add in compare-wrapper tests:
```python
def test_compare_wrapper_passes_torch_hybrid_skip_connections(monkeypatch, tmp_path):
    # parse_args should accept toggle and run_grid_lines_compare should pass
    # hybrid_skip_connections=True into TorchRunnerConfig for torch model IDs.
    ...
```

**Step 2: Run tests to verify fail**

Run:
```bash
pytest tests/torch/test_grid_lines_torch_runner.py::TestSetupTorchConfigs::test_runner_passes_hybrid_skip_connections -v
pytest tests/torch/test_grid_lines_torch_runner.py -k "workflow and hybrid_skip_connections and factory" -v
pytest tests/test_grid_lines_compare_wrapper.py -k "hybrid_skip_connections" -v
```
Expected: FAIL (missing Torch-only runner/execution/CLI plumbing).

**Step 3: Commit**

No commit (RED only).

---

### Task 4: GREEN Config + CLI + Generator Wrapper Plumbing

**Files:**
- Modify: `ptycho/config/config.py` (PyTorchExecutionConfig only)
- Modify: `ptycho_torch/config_params.py`
- Modify: `ptycho_torch/workflows/components.py`
- Modify: `scripts/studies/grid_lines_torch_runner.py`
- Modify: `scripts/studies/grid_lines_compare_wrapper.py`
- Modify: `ptycho_torch/generators/hybrid_resnet.py`
- Test: `tests/torch/test_grid_lines_torch_runner.py`
- Test: `tests/test_grid_lines_compare_wrapper.py`

**Step 1: Add field to Torch-only configs (no canonical ModelConfig / no config-bridge mapping)**

Add to Torch model + execution configs:
```python
# ptycho_torch/config_params.py
hybrid_skip_connections: bool = False

# ptycho/config/config.py -> PyTorchExecutionConfig
hybrid_skip_connections: bool = False
```

Scope guard:
- Keep this knob Torch-only: do not add it to canonical `ModelConfig`, and do not add config-bridge/spec mappings for TensorFlow.
- Do not emit this knob into `params.cfg`; it must travel only through Torch execution/model payloads.

**Step 2: Thread field through runner setup + Torch execution overrides + CLI**

Runner dataclass + setup:
```python
hybrid_skip_connections: bool = False
...
execution_config = PyTorchExecutionConfig(
    ...,
    hybrid_skip_connections=cfg.hybrid_skip_connections,
)
```

Runner CLI:
```python
parser.add_argument("--hybrid-skip-connections", dest="hybrid_skip_connections", action="store_true", default=False)
parser.add_argument("--no-hybrid-skip-connections", dest="hybrid_skip_connections", action="store_false")
```

Torch workflow propagation:
```python
# ptycho_torch/workflows/components.py
factory_overrides["hybrid_skip_connections"] = getattr(
    execution_config, "hybrid_skip_connections", False
)
```

Compare-wrapper pass-through:
```python
parser.add_argument("--torch-hybrid-skip-connections", dest="torch_hybrid_skip_connections", action="store_true", default=False)
parser.add_argument("--torch-no-hybrid-skip-connections", dest="torch_hybrid_skip_connections", action="store_false")
parser.add_argument("--torch-mae-pred-l2-match-target", dest="torch_mae_pred_l2_match_target", action="store_true", default=False)
parser.add_argument("--no-torch-mae-pred-l2-match-target", dest="torch_mae_pred_l2_match_target", action="store_false")
# Backward-compatible alias
parser.add_argument("--torch-no-mae-pred-l2-match-target", dest="torch_mae_pred_l2_match_target", action="store_false")
...
TorchRunnerConfig(
    ...,
    hybrid_skip_connections=torch_hybrid_skip_connections,
    torch_mae_pred_l2_match_target=torch_mae_pred_l2_match_target,
)
```

Generator wrapper:
```python
hybrid_skip_connections = getattr(model_config, "hybrid_skip_connections", False)
...
skip_connections=hybrid_skip_connections,
```

Torch-only enforcement checks:
- Add regression coverage proving the runner path sets `execution_config.hybrid_skip_connections` and the canonical `training_config.model` contract remains unchanged for this knob.
- Add workflow-level regression coverage proving `_train_with_lightning` forwards `hybrid_skip_connections` into `create_training_payload(..., overrides=...)`.
- Add regression coverage proving wrapper parse+pass-through sets `TorchRunnerConfig.hybrid_skip_connections` for Torch model IDs.
- Add wrapper-CLI regression coverage proving both `--no-torch-mae-pred-l2-match-target` and `--torch-no-mae-pred-l2-match-target` map to the same destination.

**Step 3: Run tests to verify pass**

Run:
```bash
pytest tests/torch/test_grid_lines_torch_runner.py::TestSetupTorchConfigs::test_runner_passes_hybrid_skip_connections -v
pytest tests/torch/test_grid_lines_torch_runner.py::TestSetupTorchConfigs::test_runner_accepts_hybrid_resnet -v
pytest tests/torch/test_grid_lines_torch_runner.py -k "workflow and hybrid_skip_connections and factory" -v
pytest tests/test_grid_lines_compare_wrapper.py -k "hybrid_skip_connections" -v
pytest tests/test_grid_lines_compare_wrapper.py -k "mae_pred_l2_match_target and alias" -v
```
Expected: PASS.

**Step 4: Commit**

```bash
python scripts/tools/generate_test_index.py > docs/development/TEST_SUITE_INDEX.md
git add ptycho/config/config.py ptycho_torch/config_params.py ptycho_torch/workflows/components.py scripts/studies/grid_lines_torch_runner.py scripts/studies/grid_lines_compare_wrapper.py ptycho_torch/generators/hybrid_resnet.py tests/torch/test_grid_lines_torch_runner.py tests/test_grid_lines_compare_wrapper.py docs/development/TEST_SUITE_INDEX.md
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
- custom profile path validation for `custom_npz_pair_n128`/`custom_npz_pair_n256` (required args + actionable error paths)
- seed-robust promotion helper behavior:
  - boundary candidate set computation (`top-K + next 2`, capped by eligible rows),
  - median-rank aggregation across seeds `{3,11,17}`,
  - promotion decisions sourced from robustness ranking (not raw seed-3 rank).

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

### Task 5A: Study-Index Conformance Gate (Pre-Implementation)

**Files:**
- Modify: none

**Step 1: Pre-read study-index conventions**

Read:
- `docs/studies/index.md`

Extract and record the conventions this runbook must follow:
- study key naming style (slug in header),
- script path placement under `scripts/studies/runbooks/`,
- command block style and required args,
- output/artifact contract language (manifest/summary/invocation artifacts),
- smoke/full run guidance style when applicable.

**Step 2: Define a conformance checklist for this initiative**

Before Task 6 coding, write a short checklist in this plan (or task notes) that the implementation must satisfy:
- unique study key for this sweep (`hybrid-resnet-mode-skip-sweep`),
- canonical script path and output root conventions,
- invocation artifact expectations (`invocation.json`, `invocation.sh`),
- explicit mention of collation/visual artifact locations.

**Step 3: Commit**

No commit (planning/conformance gate only).

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
  - `--cameraman-dp` (required when `N=256` is selected with `cameraman256_halfsplit_v1`)
  - `--cameraman-para` (required when `N=256` is selected with `cameraman256_halfsplit_v1`)
  - `--fly001-external-train-npz` (required when `fly001_external_n128_top_bottom_v1` is selected)
  - `--fly001-external-test-npz` (required when `fly001_external_n128_top_bottom_v1` is selected)
  - `--custom-n128-train-npz` (required when `custom_npz_pair_n128` is selected)
  - `--custom-n128-test-npz` (required when `custom_npz_pair_n128` is selected)
  - `--custom-n256-train-npz` (required when `custom_npz_pair_n256` is selected)
  - `--custom-n256-test-npz` (required when `custom_npz_pair_n256` is selected)
  - `--stage-id` (default: `A`; allowed: `A|B|C|D|E`)
  - `--epochs-n128`, `--epochs-n256`
  - `--top-k-n256` (default 6)
  - `--promotion-source-summary` (required for stage-id `B-E` and other promotion-based `N=256` runs)
  - `--allow-n256-direct-diagnostic` (default off; explicit exception mode, see stage progression contract)
  - `--aggregate-seed-rerank-root` (optional; enables robustness-summary collation mode)
  - `--source-summary` (required with `--aggregate-seed-rerank-root`)
  - `--emit-robust-promotion-summary` (required with `--aggregate-seed-rerank-root`)
  - `--seed`
  - `--output-root`
  - `--prune-heavy-artifacts/--no-prune-heavy-artifacts` (default: prune on)
  - `--probe-mask/--no-probe-mask` (default off)
  - `--torch-mae-pred-l2-match-target/--no-torch-mae-pred-l2-match-target` (default off)
- Dataset profile resolution contract:
  - profile ids are orchestration-level aliases only (runbook/sweep layer).
  - leaf CLIs remain path-first and unchanged:
    - `grid_lines_compare_wrapper.py`: `--dataset-source` plus `--train-data/--test-data` where required.
    - `grid_lines_torch_runner.py`: `--train-npz/--test-npz`.
  - runbook execution path is runner-only for all stages/N values in this initiative; wrapper is parity-test coverage only.
  - wrapper/runner parity rule: any Torch sweep knob shared by both leaf CLIs must be accepted and passed through with aligned defaults, covered by wrapper+runner tests.
  - N=256 sweeps in this initiative are runner-led; wrapper parity assertions apply within the wrapper-supported `N` domain.
  - wrapper parity for MAE normalization must accept both `--no-torch-mae-pred-l2-match-target` (canonical) and `--torch-no-mae-pred-l2-match-target` (legacy alias), mapping to the same destination.
  - explicit caller-provided dataset paths override profile defaults.
  - custom profile ids (`custom_npz_pair_n128`, `custom_npz_pair_n256`) must resolve from explicit custom path args; no implicit defaults.
  - custom profile path mapping:
    - `custom_npz_pair_n128` -> `--custom-n128-train-npz` + `--custom-n128-test-npz`
    - `custom_npz_pair_n256` -> `--custom-n256-train-npz` + `--custom-n256-test-npz`
  - fail fast with an actionable error when `cameraman256_halfsplit_v1` is selected for an active `N=256` run and either `--cameraman-dp` or `--cameraman-para` is missing.
- Study-index conformance contract:
  - runbook metadata/outputs must align with the conventions extracted in Task 5A,
  - manifest should include a stable study key (`hybrid-resnet-mode-skip-sweep`) and script path,
  - output layout and invocation artifacts must match study-index expectations.
- `N=128`: resolve dataset profile(s), then run each combo against each profile.
  - Default profile `integration_grid_lines_n128_v1` MUST use the exact integration-fixture generation recipe:
    - `N=128`, `gridsize=1`
    - `probe_npz=datasets/Run1084_recon3_postPC_shrunk_3.npz`
    - `nimgs_train=2`, `nimgs_test=1`, `nphotons=1e9`
    - `probe_source=custom`, `probe_smoothing_sigma=0.5`, `probe_scale_mode=pad_extrapolate`, `set_phi=True`
    - deterministic dataset seed (recorded in manifest; default `3`)
  - Additional N=128 profiles:
    - `fly001_external_n128_top_bottom_v1` (external NPZ split; requires `--fly001-external-train-npz` and `--fly001-external-test-npz`)
    - `custom_npz_pair_n128` (caller-supplied `train.npz` / `test.npz`)
- Rank `N=128` runs using the Section-6 policy from the companion design
  (`amp_ssim`-driven promotion ranking with phase-SSIM guardrail and runtime feasibility), then select top-K for `N=256`.
- Promotion seed-robustness rule (mandatory before each `N=128 -> N=256` promotion event):
  - keep broad sweep single-seed (`seed=3` default),
  - build boundary candidate set from source summary: `top-K + next 2` after guardrails (or all eligible candidates if fewer than `K+2`),
  - rerun boundary candidates with seeds `11` and `17` at the same source `N`,
  - recompute candidate ranking using median rank across seeds `{3,11,17}` and use that robustness-validated ranking for promotion.
  - persist a robustness artifact (for example `promotion_seed_robustness.csv`) containing per-seed ranks and `median_rank` used for promotion.
- `N=256`: resolve dataset profile(s), then run promoted top-K configs for each profile.
  - Default profile `cameraman256_halfsplit_v1`:
    - requires `--cameraman-dp` and `--cameraman-para`
    - call `prepare_hybrid_dataset(..., half="top")` for train NPZ
    - call `prepare_hybrid_dataset(..., half="bottom")` and use `train_npz` as bottom-half test NPZ
  - Additional N=256 profiles:
    - `custom_npz_pair_n256` (caller-supplied `train.npz` / `test.npz`)
- Stage progression contract:
  - `--stage-id A` seeds the first promoted set directly from its own `N=128` results.
  - Stages `B-E` MUST load canonical transition config from a prior-stage single-row champion anchor summary (derived from prior-stage `promotion/summary_seed_robust.csv`) via `--promotion-source-summary`.
  - True-default control baselines must be retained as a separate comparison lane and must not be used as canonical transition anchors.
  - Promotion to `N=256` MUST use a robustness-validated promotion summary (median-rank across seeds `{3,11,17}` over the boundary candidate set), not a raw single-seed ranking summary.
  - For stages `B-E`, `--ns 256`-only runs are invalid without `--promotion-source-summary`.
  - Exception: a direct `N=256` diagnostic run is allowed only when `--allow-n256-direct-diagnostic` is set and `--top-k-n256 0`; these runs are excluded from promotion/ranking state.
- Promotion-state API contract (`summary.csv`):
  - treat prior-stage summary as a versioned API; include `summary_schema_version` in each summary row and manifest.
  - validate promotion-source summaries strictly before use (version + required columns + non-empty candidate set), failing fast with actionable errors.
  - for promotion-enabled `N=256` runs, require robustness-ranking fields in source summary/artifacts (seed set `{3,11,17}` and median-rank promotion columns); reject raw single-seed summaries.
  - in seed-rerank aggregation mode (`--aggregate-seed-rerank-root`), require `--source-summary` and `--emit-robust-promotion-summary`, then emit a consolidated robustness summary with per-seed + median-rank fields.
- Persist invocation artifacts + per-run stdout/stderr logs + `sweep_manifest.json` + `summary.csv` + `summary.md`.
- Persist visual-evidence collation artifacts under:
  - `<output-root>/comparison_bundle/shared_pngs/`
  - descriptive filenames containing at least `{stage_id, N, dataset_profile, run_id}`.
- Run-level disk hygiene (default behavior):
  - After each run row is appended to summary/manifest, apply retention-tier policy before pruning.
  - Keep one full-artifact anchor run for each `(stage_id, N, dataset_profile)` tuple (`retention_tier=full_anchor`) to preserve forensic evidence.
  - Prune-heavy policy applies to subsequent successful runs in the same tuple (`retention_tier=pruned`).
  - Prune candidates include large NPZ datasets/recons and checkpoints (for example `datasets/**/*.npz`, `recons/**/*.npz`, `*.ckpt`, `*.pt`, `*.pth`, large transient caches/log blobs) generated by that run.
  - Do not delete external source inputs outside run output root.
  - Always retain minimal reproducibility artifacts:
    - `invocation.json`, `invocation.sh`
    - per-run metrics JSON + summary row fields needed for promotion/ranking
    - stage `sweep_manifest.json`, `summary.csv`, `summary.md`
    - curated visual evidence under `comparison_bundle/shared_pngs/`.
  - Emit `cleanup_report.json` per run with deleted paths, bytes reclaimed, and chosen `retention_tier`.
- Confounder provenance contract (hard-required):
  - persist `probe_mask_enabled` and `torch_mae_pred_l2_match_target` in both manifest and summary rows for every run.
  - fail run finalization if either field is missing from persisted row/manifest payload.
- Include per-profile and aggregate ranking views:
  - per-profile metrics/ranks
  - macro aggregate rank across profiles (median rank; deterministic tie-break by higher `amp_ssim`, then lower `train_wall_time_sec`, then lower `model_params`).

**Step 2: Run tests to verify pass**

Run:
```bash
pytest tests/studies/test_hybrid_resnet_mode_skip_sweep.py -v
```
Expected: PASS.
Also capture matching `--collect-only` and execution logs under `${REPORT_DIR}` per the Test Evidence Contract.

**Step 3: Commit**

```bash
python scripts/tools/generate_test_index.py > docs/development/TEST_SUITE_INDEX.md
git add scripts/studies/runbooks/run_hybrid_resnet_mode_skip_sweep.py tests/studies/test_hybrid_resnet_mode_skip_sweep.py docs/development/TEST_SUITE_INDEX.md
git commit -m "feat(studies): add hybrid_resnet mode-skip sweep runbook"
```

---

### Task 7: Documentation for Stage-A API Surface + Sweep Workflow

**Files:**
- Modify: `docs/CONFIGURATION.md`
- Modify: `docs/workflows/pytorch.md`
- Modify: `ptycho_torch/generators/README.md`
- Modify: `docs/studies/index.md`
- Regenerate output: `docs/development/TEST_SUITE_INDEX.md` via `scripts/tools/generate_test_index.py`

**Step 1: Add docs entries**

- `docs/CONFIGURATION.md`: add `hybrid_skip_connections` under `PyTorchExecutionConfig` (default `False`, Torch-only execution knob).
- `ptycho_torch/generators/README.md`: document skip toggle and interaction with `fno_modes`.
- `docs/studies/index.md`: add the new study entry under a unique key:
  - `hybrid-resnet-mode-skip-sweep`
  - include purpose, runbook script path, canonical smoke/full command blocks, and artifact contract.
- `docs/workflows/pytorch.md`: document `--hybrid-skip-connections/--no-hybrid-skip-connections`, expected behavior (`default False` parity), and explicit Torch-only scope (not bridged into TensorFlow config/spec contracts).

**Step 2: Confirm Torch-only scope in docs**

- No `docs/specs/spec-ptycho-config-bridge.md` changes for `hybrid_skip_connections`.
- Ensure wording in workflow/config docs states this knob is consumed only by Torch runner/workflow paths.

**Step 3: Regenerate test-index documentation**

Run:
```bash
python scripts/tools/generate_test_index.py > docs/development/TEST_SUITE_INDEX.md
```
Expected: regenerated index includes selectors for `tests/torch/test_fno_generators.py`,
`tests/torch/test_grid_lines_torch_runner.py`, and
`tests/studies/test_hybrid_resnet_mode_skip_sweep.py`.

**Step 4: Verify docs are discoverable**

Run:
```bash
rg -n "hybrid_skip_connections|run_hybrid_resnet_mode_skip_sweep|Torch-only|torch-only" docs ptycho_torch/generators/README.md
rg -n "hybrid-resnet-mode-skip-sweep|scripts/studies/runbooks/run_hybrid_resnet_mode_skip_sweep.py" docs/studies/index.md
```
Expected: references present across config guide, workflow guide, generators README, studies index, and test-suite index.

**Step 5: Commit**

```bash
python scripts/tools/generate_test_index.py > docs/development/TEST_SUITE_INDEX.md
git add docs/CONFIGURATION.md docs/workflows/pytorch.md ptycho_torch/generators/README.md docs/studies/index.md docs/development/TEST_SUITE_INDEX.md
git commit -m "docs: sync stage-a hybrid_resnet skip API across torch config/workflow/studies"
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
pytest tests/torch/test_fno_generators.py::TestHybridResnetGenerator -v
pytest tests/torch/test_grid_lines_torch_runner.py -k "hybrid_resnet or hybrid_skip_connections" -v
pytest tests/studies/test_hybrid_resnet_mode_skip_sweep.py -v
pytest -m integration -v
```
Expected: PASS.
Archive the integration-marker stdout/stderr log under the active plan artifact location.
For each selector above, store both collect-only and execution logs under `${REPORT_DIR}`.

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
  --allow-n256-direct-diagnostic \
  --dataset-profiles-n256 cameraman256_halfsplit_v1 \
  --epochs-n256 5 \
  --top-k-n256 0 \
  --cameraman-dp /home/ollie/Downloads/nersc/data/cameraman256_dp.hdf5 \
  --cameraman-para /home/ollie/Downloads/nersc/data/cameraman256_para.hdf5 \
  --output-root outputs/hybrid_resnet_mode_skip_sweep_smoke_n256 \
  --seed 3 \
  --no-probe-mask \
  --no-torch-mae-pred-l2-match-target
```
Expected: diagnostic-only N=256 smoke artifacts created (non-promotion path; no robustness summary dependency).

**Step 4: Final commit (if needed)**

```bash
git status --short
python scripts/tools/generate_test_index.py > docs/development/TEST_SUITE_INDEX.md
git add \
  ptycho_torch/generators/hybrid_resnet.py \
  ptycho/config/config.py \
  ptycho_torch/config_params.py \
  ptycho_torch/workflows/components.py \
  scripts/studies/grid_lines_torch_runner.py \
  scripts/studies/runbooks/run_hybrid_resnet_mode_skip_sweep.py \
  tests/torch/test_fno_generators.py \
  tests/torch/test_grid_lines_torch_runner.py \
  tests/studies/test_hybrid_resnet_mode_skip_sweep.py \
  docs/CONFIGURATION.md \
  docs/workflows/pytorch.md \
  ptycho_torch/generators/README.md \
  docs/studies/index.md \
  docs/development/TEST_SUITE_INDEX.md \
  docs/plans/2026-02-21-hybrid-resnet-skip-mode-search.md \
  docs/plans/2026-02-21-hybrid-resnet-skip-mode-search-design.md
git status --short
git commit -m "test+feat+docs: hybrid_resnet skip connections and mode-skip sweep workflow"
```

---
