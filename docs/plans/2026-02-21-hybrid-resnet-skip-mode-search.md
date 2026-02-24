# Hybrid ResNet Skip Connections + Mode Search Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add optional encoder-decoder skip connections to `hybrid_resnet`, add a reproducible mode×skip×width benchmark workflow for `N=128` and `N=256`, and define a staged structural-search extension for depth/downsampling/capacity/skip-design axes.

**Architecture:** Keep default behavior unchanged (`skip_connections=False`) to preserve current baselines/integration expectations. Implement additive skip fusion with lightweight `1x1` projection layers at decoder resolutions (`N/2`, `N`). Expose one boolean knob end-to-end (`hybrid_skip_connections`) through Torch-only runner/execution config + CLI (do not bridge this knob into TensorFlow/canonical model contracts), then run a deterministic sweep over `fno_modes × hybrid_skip_connections × fno_width` with fixed probe-mask/loss-normalization controls. Make dataset choice explicit via named dataset profiles so the same sweep can run on multiple failure-mode regimes. Execute Stage A in two steps (full grid on `N=128`, then top-K promotion to `N=256`), then add structural axes one stage at a time (B→E) with bounded per-stage run budgets. Promotion governance: keep broad sweeps single-seed (`seed=3` default), then run boundary seed reranks (`top-K + next 2`, seeds `11` and `17`) before every promotion, and promote by median rank across seeds. Governance decision for this initiative: new Stage C-E knobs stay Torch-only (runner/execution/model paths) unless a follow-up plan explicitly approves cross-backend bridge expansion.

**Tech Stack:** PyTorch/Lightning (`ptycho_torch`), existing grid-lines + NERSC study scripts, `pytest`, runbook-style orchestration, JSON/CSV/Markdown artifacts.

**Companion Design:** `docs/plans/2026-02-21-hybrid-resnet-skip-mode-search-design.md` (normative knob semantics, stage gates, ranking/promotion policy, and artifact contract).

## Progress Checklist

- [ ] Task 0: Preflight Environment (No Code)
- [ ] Task 1: RED Test for HybridResnet Skip Forward Contract
- [ ] Task 2: GREEN HybridResnet Skip Implementation
- [ ] Task 3: RED Tests for Config/CLI Plumbing of Skip Toggle
- [ ] Task 4: GREEN Config + CLI + Generator Wrapper Plumbing
- [ ] Task 5: RED Tests for Mode-Skip Sweep Runbook
- [ ] Task 6: GREEN Sweep Runbook Implementation (N=128 + N=256)
- [ ] Task 7: Documentation for New Toggle + Sweep Workflow
- [ ] Task 8: Verification and Smoke Runs
- [ ] Task 9: Final Full Sweep Command (Hand-off)
- [ ] Task 10: Add Structural-Axis Hooks to Sweep Runbook (No Cartesian Explosion)
- [ ] Task 11: Stage B Search (Axis 1: `fno_blocks`)
- [ ] Task 12: Stage C Search (Axis 2: Downsampling Schedule / Bottleneck Resolution + Downsampling Operator)
- [ ] Task 13: Stage D Search (Axes 3 + 4: Capacity and Decoder Depth)
- [ ] Task 14: Stage E Search (Axis 5: Skip-Connection Design)
- [ ] Task 15: Stage Governance and Stop/Go Criteria

## Session Log

| Date (UTC) | Task IDs | Status | Evidence Paths | Commit(s) |
| --- | --- | --- | --- | --- |
| 2026-02-24 | Plan hygiene + contracts | Completed | `docs/plans/2026-02-21-hybrid-resnet-skip-mode-search.md`, `docs/plans/2026-02-21-hybrid-resnet-skip-mode-search-design.md` | `9c477c4e`, `cf2dee67` |

## Do Next (Update Every Session)

- Mark completed tasks in the checklist and append one row to Session Log before ending the session.
- Keep `Evidence Paths` concrete (artifacts, test logs, or output summaries), not prose-only.
- Replace this block with the next 1-3 concrete actions for the following session.

## Test Evidence Contract (Mandatory)

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
...
TorchRunnerConfig(..., hybrid_skip_connections=torch_hybrid_skip_connections)
```

Generator wrapper:
```python
hybrid_skip_connections = getattr(model_config, "hybrid_skip_connections", False)
...
skip_connections=hybrid_skip_connections,
```

Torch-only enforcement checks:
- Add regression coverage proving the runner path sets `execution_config.hybrid_skip_connections` and the canonical `training_config.model` contract remains unchanged for this knob.
- Add regression coverage proving wrapper parse+pass-through sets `TorchRunnerConfig.hybrid_skip_connections` for Torch model IDs.

**Step 3: Run tests to verify pass**

Run:
```bash
pytest tests/torch/test_grid_lines_torch_runner.py::TestSetupTorchConfigs::test_runner_passes_hybrid_skip_connections -v
pytest tests/torch/test_grid_lines_torch_runner.py::TestSetupTorchConfigs::test_runner_accepts_hybrid_resnet -v
pytest tests/test_grid_lines_compare_wrapper.py -k "hybrid_skip_connections" -v
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
  - `--custom-train-npz-n128` (required when `custom_npz_pair_n128` is selected)
  - `--custom-test-npz-n128` (required when `custom_npz_pair_n128` is selected)
  - `--custom-train-npz-n256` (required when `custom_npz_pair_n256` is selected)
  - `--custom-test-npz-n256` (required when `custom_npz_pair_n256` is selected)
  - `--epochs-n128`, `--epochs-n256`
  - `--top-k-n256` (default 6)
  - `--promotion-source-summary` (required for stages `B-E` and other promotion-based `N=256` runs)
  - `--allow-n256-direct-diagnostic` (default off; explicit exception mode, see stage progression contract)
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
  - wrapper/runner parity rule: any Torch sweep knob shared by both leaf CLIs must be accepted and passed through with aligned defaults, covered by wrapper+runner tests.
  - N=256 sweeps in this initiative are runner-led; wrapper parity assertions apply within the wrapper-supported `N` domain.
  - explicit caller-provided dataset paths override profile defaults.
  - custom profile ids (`custom_npz_pair_n128`, `custom_npz_pair_n256`) must resolve from explicit custom path args; no implicit defaults.
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
  (lexicographic amplitude ranking with phase-SSIM guardrail), then select top-K for `N=256`.
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
  - Stage `A` seeds the first promoted set directly from its own `N=128` results.
  - Stages `B-E` MUST load their baseline/promoted config set from a prior-stage `N=128` summary via `--promotion-source-summary`.
  - Promotion to `N=256` MUST use a robustness-validated promotion summary (median-rank across seeds `{3,11,17}` over the boundary candidate set), not a raw single-seed ranking summary.
  - For stages `B-E`, `--ns 256`-only runs are invalid without `--promotion-source-summary`.
  - Exception: a direct `N=256` diagnostic run is allowed only when `--allow-n256-direct-diagnostic` is set and `--top-k-n256 0`; these runs are excluded from promotion/ranking state.
- Promotion-state API contract (`summary.csv`):
  - treat prior-stage summary as a versioned API; include `summary_schema_version` in each summary row and manifest.
  - validate promotion-source summaries strictly before use (version + required columns + non-empty candidate set), failing fast with actionable errors.
  - for promotion-enabled `N=256` runs, require robustness-ranking fields in source summary/artifacts (seed set `{3,11,17}` and median-rank promotion columns); reject raw single-seed summaries.
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
  - macro aggregate rank across profiles (median rank; tie-break by mean primary score).

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
  --promotion-source-summary outputs/hybrid_resnet_mode_skip_sweep_smoke_n128/summary.csv \
  --dataset-profiles-n256 cameraman256_halfsplit_v1 \
  --epochs-n256 5 \
  --top-k-n256 2 \
  --cameraman-dp /home/ollie/Downloads/nersc/data/cameraman256_dp.hdf5 \
  --cameraman-para /home/ollie/Downloads/nersc/data/cameraman256_para.hdf5 \
  --output-root outputs/hybrid_resnet_mode_skip_sweep_smoke_n256 \
  --seed 3 \
  --no-probe-mask \
  --no-torch-mae-pred-l2-match-target
```
Expected: per-run metrics + aggregate summary artifacts created.

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

### Task 9: Final Full Sweep Command (Hand-off)

**Files:**
- Modify: none

**Step 1: Run full Stage-A grid at N=128 (single-seed exploration)**

```bash
python scripts/studies/runbooks/run_hybrid_resnet_mode_skip_sweep.py \
  --modes 12,16,24,32,48 \
  --skip-values off,on \
  --widths 32,48,64 \
  --ns 128 \
  --dataset-profiles-n128 integration_grid_lines_n128_v1,fly001_external_n128_top_bottom_v1 \
  --fly001-external-train-npz <path/to/fly001_n128_train_top_half.npz> \
  --fly001-external-test-npz <path/to/fly001_n128_test_bottom_half.npz> \
  --epochs-n128 20 \
  --top-k-n256 6 \
  --output-root outputs/hybrid_resnet_mode_skip_sweep_full_n128_20260221 \
  --seed 3 \
  --no-probe-mask \
  --no-torch-mae-pred-l2-match-target
```

**Step 2: Boundary seed-rerank before promotion (mandatory)**

From `outputs/hybrid_resnet_mode_skip_sweep_full_n128_20260221/summary.csv`, build the boundary candidate set:
- `top-K + next 2` (for this command, `K=6`, so rerank top 8),
- if fewer than 8 eligible candidates remain after guardrails, rerank all eligible candidates.

Rerun each boundary candidate at `N=128` with seeds `11` and `17` (same confounder settings, single-value `--modes/--skip-values/--widths` per candidate).

Template:
```bash
python scripts/studies/runbooks/run_hybrid_resnet_mode_skip_sweep.py \
  --modes <mode> \
  --skip-values <off_or_on> \
  --widths <width> \
  --ns 128 \
  --dataset-profiles-n128 integration_grid_lines_n128_v1,fly001_external_n128_top_bottom_v1 \
  --fly001-external-train-npz <path/to/fly001_n128_train_top_half.npz> \
  --fly001-external-test-npz <path/to/fly001_n128_test_bottom_half.npz> \
  --epochs-n128 20 \
  --top-k-n256 0 \
  --output-root outputs/hybrid_resnet_mode_skip_sweep_full_n128_20260221/seed_rerank/<candidate_id>_seed<seed> \
  --seed <11_or_17> \
  --no-probe-mask \
  --no-torch-mae-pred-l2-match-target
```
Expected:
- promotion summary `outputs/hybrid_resnet_mode_skip_sweep_full_n128_20260221/promotion/summary_seed_robust.csv` exists,
- ranking in that summary uses median rank across seeds `{3,11,17}`.

**Step 3: Promote robust top-K and run N=256**

Run:
```bash
python scripts/studies/runbooks/run_hybrid_resnet_mode_skip_sweep.py \
  --ns 256 \
  --promotion-source-summary outputs/hybrid_resnet_mode_skip_sweep_full_n128_20260221/promotion/summary_seed_robust.csv \
  --dataset-profiles-n256 cameraman256_halfsplit_v1 \
  --epochs-n256 40 \
  --top-k-n256 6 \
  --cameraman-dp /home/ollie/Downloads/nersc/data/cameraman256_dp.hdf5 \
  --cameraman-para /home/ollie/Downloads/nersc/data/cameraman256_para.hdf5 \
  --output-root outputs/hybrid_resnet_mode_skip_sweep_full_n256_20260221 \
  --seed 3 \
  --no-probe-mask \
  --no-torch-mae-pred-l2-match-target
```
Expected: promoted `N=256` run set is driven by `summary_seed_robust.csv` (not raw single-seed summary).

**Step 4: Add targeted N=256 high-mode probe (diagnostic)**

Run:
```bash
python scripts/studies/runbooks/run_hybrid_resnet_mode_skip_sweep.py \
  --modes 32,48 \
  --skip-values off,on \
  --widths 48 \
  --ns 256 \
  --allow-n256-direct-diagnostic \
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
Expected: explicit N=256-only high-frequency sensitivity evidence (diagnostic-only; not used for stage promotion state).

**Step 5: Confirm artifacts**

Run:
```bash
find outputs/hybrid_resnet_mode_skip_sweep_full_n128_20260221 -maxdepth 3 -type f | rg "summary.csv|summary.md|summary_seed_robust.csv"
```
Expected: Stage-A `N=128` summary and robustness summary are present.

Run:
```bash
find outputs/hybrid_resnet_mode_skip_sweep_full_n256_20260221 -maxdepth 2 -type f | rg "sweep_manifest.json|summary.csv|summary.md"
```
Expected: promoted `N=256` aggregate artifacts are present.

Run:
```bash
cat outputs/hybrid_resnet_mode_skip_sweep_full_n128_20260221/promotion/summary_seed_robust.csv | head
```
Expected: robust summary header/rows include seeded promotion ranking fields.

**Step 6: Commit**

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
- `--cameraman-dp` (required when `N=256` is selected with `cameraman256_halfsplit_v1`)
- `--cameraman-para` (required when `N=256` is selected with `cameraman256_halfsplit_v1`)
- `--fly001-external-train-npz`
- `--fly001-external-test-npz`
- `--promotion-source-summary` (default: empty; required for stages `B-E`, and always required when stage `B-E` runs with `--ns 256` only)
- `--allow-n256-direct-diagnostic` (default off; only valid with `--ns 256 --top-k-n256 0`)

Execution-path note:
- do not widen wrapper `--N` in this initiative; keep wrapper at `N in {64,128}` and route all `N=256` sweeps through `grid_lines_torch_runner.py` pathing.

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
- Reject `cameraman256_halfsplit_v1` profile usage for active `N=256` runs unless both `--cameraman-dp` and `--cameraman-para` are provided.
- Reject `--allow-n256-direct-diagnostic` unless `--ns 256` and `--top-k-n256 0`.
- raise actionable error if multiple structural axes contain >1 value in one stage

**Step 3: Add/adjust tests**

Add explicit invocation-provenance assertions in
`tests/studies/test_hybrid_resnet_mode_skip_sweep.py`:
- `invocation.json` is created at the run root with expected script path/argv/parsed args.
- `invocation.sh` is created at the run root and contains a reconstructible command.
- cleanup behavior is validated:
  - first successful run per `(stage_id, N, dataset_profile)` is retained as `full_anchor`.
  - subsequent successful runs in same tuple are `pruned`.
  - required retained artifacts and `cleanup_report.json` still exist with `retention_tier`.
- promotion-source validation is strict:
  - missing/unknown `summary_schema_version` fails with actionable error.
  - required summary columns missing fails before any stage execution.
- confounder provenance is enforced:
  - persisted summary rows and manifest always include `probe_mask_enabled` and `torch_mae_pred_l2_match_target`.
- wrapper/runner parity:
  - new sweep knobs shared across both leaf CLIs are validated to parse and pass through on both wrapper and runner code paths.
  - MAE normalization toggle naming parity is enforced:
    - canonical shared flags: `--torch-mae-pred-l2-match-target` and `--no-torch-mae-pred-l2-match-target`,
    - wrapper keeps backward-compatible alias `--torch-no-mae-pred-l2-match-target` mapped to the same destination.
  - runner-only knobs and N=256 execution paths are validated in runner/runbook tests.
- seed-robust promotion tests are explicit:
  - verify boundary candidate construction (`top-K + next 2`) against fixture summaries,
  - verify median-rank aggregation across seeds `{3,11,17}`,
  - verify promoted set is selected from robustness ranking, not raw seed-3 ranking.

Run:
```bash
pytest tests/studies/test_hybrid_resnet_mode_skip_sweep.py -k "invocation or cleanup or matrix or guardrail or stage_id" -v
```
Expected: PASS.

**Step 4: Commit**

```bash
python scripts/tools/generate_test_index.py > docs/development/TEST_SUITE_INDEX.md
git add scripts/studies/runbooks/run_hybrid_resnet_mode_skip_sweep.py tests/studies/test_hybrid_resnet_mode_skip_sweep.py docs/studies/index.md docs/development/TEST_SUITE_INDEX.md
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
  --promotion-source-summary outputs/hybrid_resnet_mode_skip_sweep_full_n128_20260221/promotion/summary_seed_robust.csv \
  --fno-blocks-values 4,5,6 \
  --ns 128 \
  --dataset-profiles-n128 integration_grid_lines_n128_v1,fly001_external_n128_top_bottom_v1 \
  --fly001-external-train-npz <path/to/fly001_n128_train_top_half.npz> \
  --fly001-external-test-npz <path/to/fly001_n128_test_bottom_half.npz> \
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

Before this step, run the boundary seed-rerank policy on `outputs/hybrid_resnet_stageB_fno_blocks_n128_20260221/summary.csv`:
- rerank `top-K + next 2` at `N=128` with seeds `11` and `17`,
- produce `outputs/hybrid_resnet_stageB_fno_blocks_n128_20260221/promotion/summary_seed_robust.csv`,
- use that robustness summary as promotion source.

Run:
```bash
python scripts/studies/runbooks/run_hybrid_resnet_mode_skip_sweep.py \
  --stage-id B \
  --promotion-source-summary outputs/hybrid_resnet_stageB_fno_blocks_n128_20260221/promotion/summary_seed_robust.csv \
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
- Modify: `ptycho/config/config.py` (PyTorchExecutionConfig only; no canonical `ModelConfig` edits)
- Modify: `ptycho_torch/config_params.py`
- Modify: `scripts/studies/grid_lines_torch_runner.py`
- Modify: `tests/torch/test_fno_generators.py`
- Modify: `tests/torch/test_grid_lines_torch_runner.py`
- Modify: `docs/CONFIGURATION.md`
- Modify: `docs/workflows/pytorch.md`
- Modify: `ptycho_torch/generators/README.md`
- Regenerate output: `docs/development/TEST_SUITE_INDEX.md` via `scripts/tools/generate_test_index.py`

**Step 0: Topology Refactor Checkpoint (required before C/E axis expansion)**

Refactor `HybridResnetGeneratorModule` skip-fusion wiring to be topology-driven:
- derive encoder/decoder stage metadata from configured downsample schedule,
- resolve skip fusion points from metadata (resolution/stage mapping), not hard-coded layer indices,
- keep behavior identical to current two-step path when `hybrid_downsample_steps=2`.

Add focused tests before Stage C sweep execution:
- `hybrid_skip_connections=True` forward pass works for `hybrid_downsample_steps=1` and `2`,
- fusion-point selection is metadata-driven (no brittle assumptions tied to exactly two downsamples),
- output shape contract remains unchanged across schedules.

**Step 1: Add `hybrid_downsample_steps` config/CLI plumbing**

Scope guard for this stage:
- add Stage-C knobs to Torch execution/model paths only.
- in `ptycho/config/config.py`, touch `PyTorchExecutionConfig` only.
- do not add new canonical `ModelConfig` keys, config-bridge mappings, or `params.cfg` emissions for these knobs in this initiative.

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
- runner Torch execution/model propagation
- invalid `hybrid_downsample_op` rejected
- output shape invariance for each `hybrid_downsample_op`
- branch-distinctness: `stride_conv`, `avgpool_conv`, and `blurpool_conv` must exercise distinct operator code paths (for example module-type assertions plus non-identical forward outputs under fixed seed/input).

Run:
```bash
pytest tests/torch/test_fno_generators.py -k "hybrid_downsample_steps" -v
pytest tests/torch/test_fno_generators.py -k "hybrid_downsample_op" -v
pytest tests/torch/test_grid_lines_torch_runner.py -k "downsample_steps or downsample_op or torch_only" -v
```
Expected: PASS.

**Step 3: Execute Stage C runs**

Sub-stage C1 (schedule): use `--downsample-schedule-values 1,2` while fixing all other structural axes to promoted Stage B configs loaded via `--promotion-source-summary <stageB_n128_summary.csv>`.

Sub-stage C2 (operator): lock best schedule from C1 (from `--promotion-source-summary <stageC1_n128_summary.csv>`), then run:
- `--downsample-op-values stride_conv,avgpool_conv,blurpool_conv`

Stage budget:
- N=128: max 12 runs for C1 + max 12 runs for C2
- N=256: top 4 from C2 only
- Before selecting top-4 for `N=256`, apply the boundary seed-rerank policy on the C2 `N=128` source summary (`top-K + next 2`, seeds `11` and `17`) and promote from the resulting robustness summary.

**Step 4: Documentation sync for Stage-C knobs**

- `docs/CONFIGURATION.md`: add `hybrid_downsample_steps` and `hybrid_downsample_op` rows (defaults, valid values, constraints).
- `docs/workflows/pytorch.md`: add CLI examples and guardrail notes for `--downsample-schedule-values` / `--downsample-op-values`, including explicit Torch-only scope.
- `ptycho_torch/generators/README.md`: describe downsampling semantics and operator differences.
- no `docs/specs/spec-ptycho-config-bridge.md` changes for Stage-C knobs.
- regenerate `docs/development/TEST_SUITE_INDEX.md` via:
  - `python scripts/tools/generate_test_index.py > docs/development/TEST_SUITE_INDEX.md`

**Step 5: Commit**

```bash
python scripts/tools/generate_test_index.py > docs/development/TEST_SUITE_INDEX.md
git add ptycho_torch/generators/hybrid_resnet.py ptycho/config/config.py ptycho_torch/config_params.py scripts/studies/grid_lines_torch_runner.py tests/torch/test_fno_generators.py tests/torch/test_grid_lines_torch_runner.py docs/CONFIGURATION.md docs/workflows/pytorch.md ptycho_torch/generators/README.md docs/development/TEST_SUITE_INDEX.md
git commit -m "feat+docs(torch): add hybrid_resnet downsample controls with torch-only scope"
```

---

### Task 13: Stage D Search (Axes 3 + 4: Capacity and Decoder Depth)

**Files:**
- Modify: `ptycho_torch/generators/hybrid_resnet.py`
- Modify: `ptycho/config/config.py` (PyTorchExecutionConfig only; no canonical `ModelConfig` edits)
- Modify: `ptycho_torch/config_params.py`
- Modify: `scripts/studies/grid_lines_torch_runner.py`
- Modify: `tests/torch/test_fno_generators.py`
- Modify: `tests/torch/test_grid_lines_torch_runner.py`
- Modify: `docs/CONFIGURATION.md`
- Modify: `docs/workflows/pytorch.md`
- Modify: `ptycho_torch/generators/README.md`
- Regenerate output: `docs/development/TEST_SUITE_INDEX.md` via `scripts/tools/generate_test_index.py`

**Step 1: Axis 3 (capacity) sub-stage**

Scope guard for this stage:
- keep new Stage-D knob plumbing Torch-only in execution/model paths.
- in `ptycho/config/config.py`, touch `PyTorchExecutionConfig` only.
- do not add new config-bridge/spec mappings for newly introduced Stage-D knobs in this initiative.

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

**Step 3: Add/execute Task-13 tests (required by test-evidence contract)**

Run:
```bash
pytest tests/torch/test_fno_generators.py -k "hybrid_resnet_blocks or max_hidden_channels or resnet_width" -v
pytest tests/torch/test_grid_lines_torch_runner.py -k "hybrid_resnet_blocks or max_hidden_channels or resnet_width or torch_only" -v
```
Expected: PASS.
Also capture matching `--collect-only` and execution logs under `${REPORT_DIR}`.

**Step 4: Run bounded stage budgets**

Budget rule:
- N=128: max 18 runs per sub-stage
- N=256: top 4 only
- Before selecting top-4 for `N=256`, apply the boundary seed-rerank policy on the D2 `N=128` source summary (`top-K + next 2`, seeds `11` and `17`) and promote from the resulting robustness summary.

**Step 5: Documentation sync for Stage-D knobs**

- `docs/CONFIGURATION.md`: document `hybrid_resnet_blocks` (new) and any capacity-option constraints used in this stage.
- `docs/workflows/pytorch.md` and `ptycho_torch/generators/README.md`: add usage guidance and constraints with explicit Torch-only scope.
- no `docs/specs/spec-ptycho-config-bridge.md` changes for Stage-D knobs in this initiative.
- regenerate `docs/development/TEST_SUITE_INDEX.md` via:
  - `python scripts/tools/generate_test_index.py > docs/development/TEST_SUITE_INDEX.md`

**Step 6: Commit**

```bash
python scripts/tools/generate_test_index.py > docs/development/TEST_SUITE_INDEX.md
git add ptycho_torch/generators/hybrid_resnet.py ptycho/config/config.py ptycho_torch/config_params.py scripts/studies/grid_lines_torch_runner.py tests/torch/test_fno_generators.py tests/torch/test_grid_lines_torch_runner.py docs/CONFIGURATION.md docs/workflows/pytorch.md ptycho_torch/generators/README.md docs/development/TEST_SUITE_INDEX.md
git commit -m "feat+docs(torch): add hybrid_resnet capacity/depth controls with torch-only scope"
```

---

### Task 14: Stage E Search (Axis 5: Skip-Connection Design)

**Files:**
- Modify: `ptycho_torch/generators/hybrid_resnet.py`
- Modify: `ptycho/config/config.py` (PyTorchExecutionConfig only; no canonical `ModelConfig` edits)
- Modify: `ptycho_torch/config_params.py`
- Modify: `scripts/studies/grid_lines_torch_runner.py`
- Modify: `tests/torch/test_fno_generators.py`
- Modify: `tests/torch/test_grid_lines_torch_runner.py`
- Modify: `docs/CONFIGURATION.md`
- Modify: `docs/workflows/pytorch.md`
- Modify: `ptycho_torch/generators/README.md`
- Regenerate output: `docs/development/TEST_SUITE_INDEX.md` via `scripts/tools/generate_test_index.py`

**Step 1: Add `hybrid_skip_style` enum**

Scope guard for this stage:
- keep new Stage-E knob plumbing Torch-only in execution/model paths.
- in `ptycho/config/config.py`, touch `PyTorchExecutionConfig` only.
- do not add new config-bridge/spec mappings for newly introduced Stage-E knobs in this initiative.

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
- propagation through runner Torch execution/model paths with explicit no-bridge assertions
- branch-distinctness: `add`, `concat`, and `gated_add` each execute distinct fusion logic and produce non-identical outputs for a fixed seed/input.

Run:
```bash
pytest tests/torch/test_fno_generators.py -k "skip_style" -v
pytest tests/torch/test_grid_lines_torch_runner.py -k "skip_style or torch_only" -v
```

**Step 3: Execute Stage E**

Run skip styles on best config from Stage D, budget:
- N=128: all 3 styles
- N=256: top 2 styles only
- Before selecting top-2 for `N=256`, apply the boundary seed-rerank policy on the Stage-E `N=128` source summary (`top-K + next 2`, seeds `11` and `17`) and promote from the resulting robustness summary.

Constraint:
- Stage E must reuse the topology-driven fusion-point mechanism introduced in Task 12 Step 0; do not introduce new hard-coded skip tap indices.

**Step 4: Documentation sync for Stage-E knob**

- `docs/CONFIGURATION.md`: add `hybrid_skip_style` row with allowed enum values and default.
- `docs/workflows/pytorch.md` and `ptycho_torch/generators/README.md`: document behavior/constraints of `add|concat|gated_add` with explicit Torch-only scope.
- no `docs/specs/spec-ptycho-config-bridge.md` changes for Stage-E knobs in this initiative.
- regenerate `docs/development/TEST_SUITE_INDEX.md` via:
  - `python scripts/tools/generate_test_index.py > docs/development/TEST_SUITE_INDEX.md`

**Step 5: Commit**

```bash
python scripts/tools/generate_test_index.py > docs/development/TEST_SUITE_INDEX.md
git add ptycho_torch/generators/hybrid_resnet.py ptycho/config/config.py ptycho_torch/config_params.py scripts/studies/grid_lines_torch_runner.py tests/torch/test_fno_generators.py tests/torch/test_grid_lines_torch_runner.py docs/CONFIGURATION.md docs/workflows/pytorch.md ptycho_torch/generators/README.md docs/development/TEST_SUITE_INDEX.md
git commit -m "feat+docs(torch): add hybrid_resnet skip-style variants with torch-only scope"
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
- promotion source must be robustness-validated (`top-K + next 2` reranked with seeds `{3,11,17}` and promoted by median rank across seeds)
- stage-promotion governance: Stage A is not complete until `docs/studies/index.md` contains the verified `hybrid-resnet-mode-skip-sweep` entry (Task 7 checks pass).

**Step 2: Define hard stop conditions**

Stop expansion if:
- two consecutive stages show <1% relative gain on primary metric, or
- all candidate configs at new stage regress on both amplitude MAE and MSE at `N=256`.

**Step 3: Commit**

```bash
git add docs/studies/index.md docs/plans/2026-02-21-hybrid-resnet-skip-mode-search.md
git commit -m "docs(studies): add staged structural-search governance for hybrid_resnet sweep"
```
