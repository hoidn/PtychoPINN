# Merge Plan: `main` into `fno-stable`

## Initiative
- ID: MERGE-MAIN-INTO-FNO-STABLE-2026-06-29
- Title: Merge `main` infrastructure fixes into `fno-stable` without regressing FNO/FFNO generator support
- Status: executed; merge staged pending commit
- Branch target: `fno-stable`
- Source branch: `main`

## Context

`fno-stable` is currently checked out at `e5387d46` and is up to date with `origin/fno-stable`.
`main` is at `5bd07399` and is up to date with `origin/main`.

Branch relationship from local refs:
- `fno-stable...main`: `fno-stable` is 4124 commits ahead, `main` has 46 commits not in `fno-stable`.
- `git merge-tree --write-tree fno-stable main` reports 41 conflicts.

Main-side commits mostly add or fix PyTorch backend/API/DDP infrastructure, multi-mode probe handling, dataloader and reassembly behavior, config factories, and tests.
`fno-stable` carries the richer FNO/FFNO/hybrid/spectral generator registry and study harness work. The merge must preserve that architecture surface.

## Documents Read
- `AGENTS.md` / `CLAUDE.md` planning and artifact hygiene guidance.
- `plans/templates/implementation_plan.md` for plan structure.
- `plans/active/FNO-STABILITY-OVERHAUL-001/implementation.md` for FNO branch architecture intent.
- `docs/plans/2026-02-05-bisect-recent-hybrid-resnet.md` for prior Torch/grid-lines regression context.

## Compliance Matrix
- [ ] Preserve `fno-stable` architecture registry entries and checkpoint reload support for `fno`, `ffno`, `stable_hybrid`, `fno_vanilla`, `neuralop_uno`, `hybrid_resnet`, `hybrid_resnet_ffno_ptychoblock_encoder`, `hybrid_resnet_ptychoblock_ffno_encoder`, `spectral_resnet_bottleneck_net`, and related variants.
- [ ] Preserve `main` DDP/DDP-spawn fixes, run-directory synchronization, Lightning dataloader training_config propagation, and API path crash fixes.
- [ ] Preserve `main` multi-mode probe, physics scale, weighted reassembly, and dataloader filtering fixes unless they demonstrably conflict with FNO stability behavior.
- [ ] Do not overwrite branch-local FNO study scripts and tests with the smaller `main` Torch surface.
- [ ] Keep bulky artifacts out of git; store any merge logs under `.artifacts/merge-main-into-fno-stable/`.

## Preflight

1. Confirm branch and cleanliness.
   ```bash
   git switch fno-stable
   git status --short --branch
   git fetch origin
   git status --short --branch
   git rev-list --left-right --count fno-stable...main
   ```

2. Resolve or intentionally ignore local noise before merge.
   - Current known local noise: `? notebooks/archive/ePIE_recon_simulation`.
   - Do not delete or revert it unless explicitly requested.
   - If any tracked files are dirty, stop and preserve them before merging.

3. Capture baseline references.
   ```bash
   mkdir -p .artifacts/merge-main-into-fno-stable
   git rev-parse HEAD main origin/main origin/fno-stable > .artifacts/merge-main-into-fno-stable/refs_before.txt
   git merge-tree --write-tree fno-stable main > .artifacts/merge-main-into-fno-stable/merge_tree_before.txt || true
   ```

## Merge Execution

1. Start a no-commit merge.
   ```bash
   git merge --no-commit main
   ```

2. Resolve conflicts by authority class.

### Class A: Prefer `fno-stable`, then port selected `main` changes

These files define the branch's FNO architecture surface. Start from `fno-stable` and manually port infrastructure fixes from `main`.

- `ptycho_torch/model.py`
  - Preserve FNO-compatible `PtychoPINN`, Lightning checkpoint reload, generator output modes, and architecture dispatch.
  - Port `main` fixes for multi-mode probe handling, DDP-safe logging assumptions, loss scaling, `AmplitudeVarianceLoss` if absent, and any forward/inference bug fixes.

- `ptycho_torch/config_params.py`
  - Preserve all FNO/FFNO/hybrid architecture literals and fields.
  - Port `main` fields for DDP strategies, `patch_weighting`, multi-mode probe, scheduler/optimizer, and any API/runtime flags not already present.

- `ptycho_torch/generators/registry.py`
  - Preserve `fno-stable` registry as authoritative.
  - If `main` adds generic CNN registry behavior, integrate it without dropping FNO entries.

- `tests/torch/test_lightning_checkpoint.py`
  - Preserve FNO checkpoint matrix from `fno-stable`.
  - Add any `main` DDP/checkpoint regressions as additional cases rather than replacing the file.

### Class B: Prefer `main`, then restore FNO hooks

These files are primarily infrastructure in `main`, but must recognize FNO-specific config and generators.

- `ptycho_torch/config_bridge.py`
- `ptycho_torch/config_factory.py`
- `ptycho_torch/train.py`
- `ptycho_torch/train_utils.py`
- `ptycho_torch/workflows/components.py`
- `ptycho_torch/api/base_api.py`
- `ptycho_torch/api/api_helper.py`
- `ptycho_torch/api/trainer_api.py`
- `scripts/inference/inference.py`

Resolution rule:
- Start from `main` for DDP, API, backend dispatch, dataloader, and run-name fixes.
- Re-add all `fno-stable` architecture fields and pass-throughs:
  - `architecture`
  - `fno_modes`
  - `fno_width`
  - `fno_blocks`
  - `fno_cnn_blocks`
  - `fno_input_transform`
  - `generator_output_mode`
  - hybrid-resnet and spectral-resnet-specific fields.
- Confirm `create_training_payload` and workflow training overrides pass those fields into PyTorch configs.

### Class C: Resolve by semantic merge

These files have meaningful changes on both sides and need line-by-line review.

- `ptycho/config/config.py`
  - Preserve FNO architecture literals and Torch execution knobs.
  - Port `main` DDP/DDP-spawn strategy fields, auto device detection fields, and backend config additions.

- `ptycho_torch/dataloader.py`
  - Preserve `fno-stable` dataset contracts if present.
  - Port `main` fixes:
    - `training_config` propagation to `PtychoDataset` in Lightning data module.
    - group filtering / memory map size mismatch fixes.
    - multi-mode `probeGuess` loading and normalization behavior.

- `ptycho_torch/helper.py`
  - Preserve FNO branch reassembly behavior if tests depend on it.
  - Port `main` physics-scale, probe normalization, and weighted reassembly fixes.

- `ptycho_torch/reassembly.py`
  - Preserve `fno-stable` inference-time stitching behavior and grid-lines runner expectations.
  - Port `main` consolidated weighted reassembly and `InferenceConfig.patch_weighting` support.
  - Ensure probe-weighted stitching remains default and `uniform` remains available.

- `ptycho_torch/patch_generator.py`
  - Port `main` duplicate-center quadrant group handling and filtering fixes.
  - Preserve any FNO branch coordinate assumptions used by grid-lines runners.

- `ptycho/workflows/grid_lines_workflow.py`
  - Both branches added this file.
  - If `fno-stable` has the richer study workflow, keep it and port `main` fixes only where they are not already represented.
  - Confirm custom probe scaling/masking and Torch runner compatibility remain intact.

- `pyproject.toml`
  - Include `main` optional dependency changes, especially torch ecosystem and optional OpenCV handling.
  - Preserve FNO branch dependencies needed for NeuralOp/U-NO/FFNO tests.

### Class D: Usually take union

- `ptycho_torch/__init__.py`
- `ptycho_torch/datagen/datagen.py`
- `ptycho_torch/datagen/objects.py`
- `ptycho_torch/eval/eval_metrics.py`
- `ptycho_torch/eval/frc.py`
- `ptycho_torch/train_full.py`
- tests with add/add conflicts:
  - `tests/test_grid_lines_workflow.py`
  - `tests/tf_helper/test_translation_shape_guard.py`
  - `tests/torch/test_loss_modes.py`
  - `tests/torch/test_loss_units.py`
  - `tests/torch/test_parity_probe_normalization.py`
  - `tests/torch/test_physics_scale_loss.py`
  - `tests/torch/test_probe_normalization_parity.py`
  - `tests/torch/test_workflows_components.py`

Resolution rule:
- Keep both branches' assertions unless they encode contradictory expectations.
- If contradictory, prefer current intended behavior:
  - multi-mode probes supported;
  - probe normalization parity preserved;
  - FNO architecture choices preserved;
  - DDP/spawn-safe behavior preserved.

### Class E: Documentation and low-risk files

- `.gitignore`
- `CLAUDE.md`
- `README.md`
- `docs/DEVELOPER_GUIDE.md`
- `scripts/training/README.md`

Resolution rule:
- Prefer current branch project guidance unless `main` added operationally necessary DDP/API usage.
- Avoid resurrecting obsolete process text.

## Conflict List From Virtual Merge

The virtual merge reported conflicts in:

```text
.gitignore
CLAUDE.md
README.md
docs/DEVELOPER_GUIDE.md
ptycho/config/config.py
ptycho/workflows/grid_lines_workflow.py
ptycho_torch/__init__.py
ptycho_torch/api/api_helper.py
ptycho_torch/api/base_api.py
ptycho_torch/api/example_predict_lightning.py
ptycho_torch/api/example_train_lightning.py
ptycho_torch/api/trainer_api.py
ptycho_torch/config_bridge.py
ptycho_torch/config_factory.py
ptycho_torch/config_params.py
ptycho_torch/datagen/datagen.py
ptycho_torch/datagen/objects.py
ptycho_torch/dataloader.py
ptycho_torch/eval/eval_metrics.py
ptycho_torch/eval/frc.py
ptycho_torch/helper.py
ptycho_torch/inference.py
ptycho_torch/model.py
ptycho_torch/patch_generator.py
ptycho_torch/reassembly.py
ptycho_torch/train.py
ptycho_torch/train_full.py
ptycho_torch/train_utils.py
ptycho_torch/workflows/components.py
pyproject.toml
scripts/inference/inference.py
scripts/training/README.md
tests/test_grid_lines_workflow.py
tests/tf_helper/test_translation_shape_guard.py
tests/torch/test_lightning_checkpoint.py
tests/torch/test_loss_modes.py
tests/torch/test_loss_units.py
tests/torch/test_parity_probe_normalization.py
tests/torch/test_physics_scale_loss.py
tests/torch/test_probe_normalization_parity.py
tests/torch/test_workflows_components.py
```

## Post-Resolution Static Checks

Run these before tests:

```bash
git diff --check
python -m compileall ptycho ptycho_torch scripts
python - <<'PY'
from ptycho_torch.config_params import ModelConfig
from ptycho_torch.generators.registry import resolve_generator
for arch in [
    "cnn",
    "fno",
    "ffno",
    "stable_hybrid",
    "fno_vanilla",
    "neuralop_uno",
    "hybrid_resnet",
    "hybrid_resnet_ffno_ptychoblock_encoder",
    "hybrid_resnet_ptychoblock_ffno_encoder",
    "spectral_resnet_bottleneck_net",
]:
    cfg = ModelConfig(architecture=arch)
    resolve_generator(cfg)
print("generator registry OK")
PY
```

## Verification Commands

Run in this order. Save logs under `.artifacts/merge-main-into-fno-stable/`.

```bash
python -m pytest -q \
  tests/torch/test_config_factory.py \
  tests/torch/test_config_bridge.py \
  tests/torch/test_generator_registry.py \
  tests/torch/test_fno_generators.py \
  tests/torch/test_fno_lightning_integration.py \
  | tee .artifacts/merge-main-into-fno-stable/pytest_config_generator_fno.log
```

```bash
python -m pytest -q \
  tests/torch/test_lightning_checkpoint.py \
  tests/torch/test_loss_modes.py \
  tests/torch/test_loss_units.py \
  tests/torch/test_workflows_components.py \
  | tee .artifacts/merge-main-into-fno-stable/pytest_lightning_workflows.log
```

```bash
python -m pytest -q \
  tests/torch/test_parity_probe_normalization.py \
  tests/torch/test_probe_normalization_parity.py \
  tests/torch/test_physics_scale_loss.py \
  tests/torch/test_inference_reassembly_parity.py \
  tests/torch/test_inference_reassembly_aggregation.py \
  tests/torch/test_reassembly_multi_patch_parity.py \
  tests/torch/test_reassembly_sign_parity.py \
  | tee .artifacts/merge-main-into-fno-stable/pytest_physics_reassembly.log
```

```bash
python -m pytest -q \
  tests/torch/test_grid_lines_torch_runner.py \
  tests/torch/test_grid_lines_torch_runner_grad_norm_flag.py \
  tests/torch/test_grid_lines_position_reassembly_strategy.py \
  tests/torch/test_grid_lines_hybrid_resnet_integration.py \
  | tee .artifacts/merge-main-into-fno-stable/pytest_grid_lines_torch.log
```

Optional DDP-focused smoke, if GPUs/runtime allow:

```bash
python -m pytest -q \
  tests/torch/test_execution_config_defaults.py \
  tests/torch/test_lightning_dataloader_coords_guard.py \
  tests/torch/test_cli_train_torch.py \
  tests/torch/test_cli_inference_torch.py \
  | tee .artifacts/merge-main-into-fno-stable/pytest_ddp_cli.log
```

## Completion Criteria

- [ ] Merge commit exists on `fno-stable` with no unresolved conflicts.
- [x] FNO/FFNO/hybrid generator registry accepts all branch-local architecture names.
- [x] `main` DDP/DDP-spawn/run-dir fixes are present in the merged code.
- [x] `main` multi-mode probe, dataloader filtering, and weighted reassembly fixes are present or explicitly documented as superseded by `fno-stable`.
- [x] Required verification commands pass, or each failure has a documented owner and reason.
- [x] Merge logs are stored under `.artifacts/merge-main-into-fno-stable/`.

## Execution Results

Executed on 2026-06-29 from `fno-stable` after `git merge --no-commit main`.

Resolution notes:
- Preserved the `fno-stable` FNO/FFNO/hybrid generator registry and branch-local architecture fields.
- Added `InferenceConfig.patch_weighting` propagation through `ptycho_torch/config_factory.py`.
- Kept `main` reassembly changes for probe-weighted stitching while preserving `probe` as the default and `uniform` as an available option.
- Made `get_training_strategy()` backward compatible with both the legacy one-argument call style and the new `(strategy, n_devices)` style.
- Restored `adaptive_gradient_clip_()` and `compute_grad_norm()` in `ptycho_torch/train_utils.py` for existing FNO-stable model imports.
- Filtered the shared generator payload in `ptycho_torch/generators/cnn.py` so the auxiliary `execution_config` key does not get forwarded to `PtychoPINN_Lightning`.

Verification:
- `rg -n "^<<<<<<<|^=======$|^>>>>>>>" . --glob '!notebooks/**' --glob '!archive/**' --glob '!*.ipynb'`: no conflict markers.
- `git diff --check`: pass.
- `python -m compileall -q ptycho ptycho_torch scripts`: pass.
- Interface smoke: `InferenceConfig().patch_weighting == "probe"`, `TrainingConfig(n_devices="auto")` accepted, old and new `get_training_strategy()` call forms accepted, AGC/grad-norm helpers import.
- Generator registry smoke using `ptycho.config.config.TrainingConfig`: `cnn`, `fno`, `ffno`, `stable_hybrid`, `fno_vanilla`, `neuralop_uno`, `hybrid_resnet`, `hybrid_resnet_ffno_ptychoblock_encoder`, `hybrid_resnet_ptychoblock_ffno_encoder`, and `spectral_resnet_bottleneck_net` resolve.
- `python -m pytest -q tests/torch/test_config_factory.py tests/torch/test_config_bridge.py tests/torch/test_generator_registry.py tests/torch/test_fno_generators.py tests/torch/test_fno_lightning_integration.py`: 239 passed, 102 warnings.
- `python -m pytest -q tests/torch/test_lightning_checkpoint.py tests/torch/test_loss_modes.py tests/torch/test_loss_units.py tests/torch/test_workflows_components.py`: 60 passed, 47 warnings.
- `python -m pytest -q tests/torch/test_parity_probe_normalization.py tests/torch/test_probe_normalization_parity.py tests/torch/test_physics_scale_loss.py tests/torch/test_inference_reassembly_parity.py tests/torch/test_inference_reassembly_aggregation.py tests/torch/test_reassembly_multi_patch_parity.py tests/torch/test_reassembly_sign_parity.py`: 10 passed.
- `python -m pytest -q tests/torch/test_grid_lines_torch_runner.py tests/torch/test_grid_lines_torch_runner_grad_norm_flag.py tests/torch/test_grid_lines_position_reassembly_strategy.py tests/torch/test_grid_lines_hybrid_resnet_integration.py`: initial run produced 172 passed, 3 failed, 32 warnings because `datasets/Run1084_recon3_postPC_shrunk_3.npz` was absent.
- Follow-up grid-lines fixture check found compatible local sources:
  - `tmp/Run1084_recon3_postPC_shrunk_3_torch.npz`: `diff3d` `(1087, 64, 64)` float32, `probeGuess` `(64, 64)` complex128.
  - `.artifacts/pytorch_integration_workflow/canonical/Run1084_recon3_postPC_shrunk_3_canonical.npz`: `diff3d` `(1087, 64, 64)` float32, `probeGuess` `(64, 64)` complex64, `scan_index` int32.
- Created a temporary untracked symlink `datasets/Run1084_recon3_postPC_shrunk_3.npz -> ../tmp/Run1084_recon3_postPC_shrunk_3_torch.npz`, reran the three formerly failing tests, then removed the symlink. Result: `python -m pytest -q tests/torch/test_grid_lines_hybrid_resnet_integration.py::test_grid_lines_dataset_stats tests/torch/test_grid_lines_hybrid_resnet_integration.py::test_grid_lines_hybrid_resnet_metrics tests/torch/test_grid_lines_hybrid_resnet_integration.py::test_grid_lines_spectral_resnet_bottleneck_smoke`: 3 passed in 274.93s.
- Optional DDP/CLI smoke `python -m pytest -q tests/torch/test_execution_config_defaults.py tests/torch/test_lightning_dataloader_coords_guard.py tests/torch/test_cli_train_torch.py tests/torch/test_cli_inference_torch.py`: 37 passed, 1 skipped, 9 warnings.

## Rollback / Abort Criteria

Abort the merge with `git merge --abort` if any of the following occur before committing:
- FNO architecture literals or registry entries are lost.
- Checkpoint reload cannot reconstruct FNO-family generators.
- DDP fixes require removing FNO-specific training config pass-throughs.
- Reassembly changes make grid-lines Torch inference incompatible with probe-weighted stitching.

After committing, rollback is a normal revert of the merge commit:

```bash
git revert -m 1 <merge_commit_sha>
```

## Notes for Executor

- Do not create a worktree for this merge.
- Do not run destructive checkout/reset commands against user changes.
- Treat `fno-stable` as authoritative for FNO generator implementations and study harnesses.
- Treat `main` as authoritative for recently fixed DDP/API/backend infrastructure unless it conflicts with preserved FNO architecture behavior.
