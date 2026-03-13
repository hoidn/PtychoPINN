# Studies Index

## Grid-Lines Studies

### `lines_256` dataset note

- Purpose: Document the repo-local `N=256` lines dataset alias used for single-dataset architecture experiments.
- Document: `docs/studies/lines_256_dataset.md`
- Runbook profile name: `custom_npz_pair_n256`
- Probe scaling contract: `pad_preserve` for the working `lines_256` pair; `pad_extrapolate` remains available as an explicit alternative mode.
- Preferred use: `scripts/studies/run_lines_256_arch_experiment.py` for fixed-budget experiments or explicit diagnostic-mode runbook invocations when you want lines-only `N=256` runs.

### `lines_256` architecture-improvement loop

- Purpose: Fix the exact autonomous loop for `lines_256` architecture experiments, including fresh baseline generation at session start, the untracked TSV ledger path, the session-local champion rule, and the keep/discard reset behavior.
- Document: `docs/studies/lines_256_arch_improvement_loop.md`
- Workflow: `workflows/agent_orchestration/lines_256_arch_improvement_session_loop.yaml`
- Provider prompts:
  - `prompts/workflows/lines_256_arch_improvement/experiment_step.md`
  - `prompts/workflows/lines_256_arch_improvement/debug_crash.md`
- Ledger path: `state/lines_256_arch_improvement/results.tsv`
- Baseline rule: regenerate from the current `HEAD` at the beginning of each session using the default control and fixed budget in the loop document
- Thin wrapper: `scripts/studies/run_lines_256_arch_experiment.py`
- Comparison gallery: `outputs/lines_256_arch_improvement/comparison_pngs/<session_id>/`

### `hybrid-resnet-mode-skip-sweep`

- Purpose: Run staged `hybrid_resnet` search loops over `mode x skip x width` (Stage A) and later structural axes (Stages B-E) with strict stage/substage guardrails, promotion-source validation, seed-rerank aggregation, and retention-tier cleanup.
- Script: `scripts/studies/runbooks/run_hybrid_resnet_mode_skip_sweep.py`
- Stage IDs: `A|B|C|D|E` with `substage_id=none` for `A/B/E`, `C1|C2` for stage `C`, and `D1|D2|D3|D4` for stage `D`.
- Stage-D branch-capacity axes:
  - canonical scale knobs: `--encoder-conv-hidden-scale-values` and `--encoder-spectral-hidden-scale-values`
  - legacy aliases: `--encoder-conv-hidden-values` / `--encoder-spectral-hidden-values` (diagnostic compatibility only)
  - summary/manifest provenance includes configured scales plus deterministic resolved-width metadata (`encoder_*_resolved_*` fields).
- Seed-rerank aggregation mode:
  - inputs: `--aggregate-seed-rerank-root` + `--source-summary`
  - outputs: `--emit-robust-promotion-summary` and `--emit-stage-anchor-summary`
  - coverage gate: boundary candidates (`top-K + next 2`) must include seeds `{3,11,17}`.
- Output/artifact contract:
  - run root: `invocation.json`, `invocation.sh`, `sweep_manifest.json`, `summary.csv`, `summary.md`
  - per-run: `runs/<run_id>/metrics.json` and `runs/<run_id>/cleanup_report.json`
  - summary rows persist confounder controls (`probe_mask_enabled`, `torch_mae_pred_l2_match_target`) and stage identity (`stage_id`, `substage_id`).
- Promotion governance gates:
  - rank feasible candidates with amplitude SSIM as the primary objective and `train_wall_time_sec` as the efficiency objective.
  - enforce feasibility before promotion: `phase_ssim_drop_vs_baseline <= max_phase_ssim_drop` (default `0.03`), train-time and model-parameter limits, and inference SLA (`<=60s` at `N=128`, `<=240s` at `N=256`).
  - require robustness validation before every promotion event: boundary candidate set (`top-K + next 2`) reranked across seeds `{3,11,17}` and promoted by median Pareto rank.
  - Stage A is not complete until this `hybrid-resnet-mode-skip-sweep` index entry is present and verified.
- Stop/go diagnostics:
  - pause-and-diagnose when two consecutive stages deliver `<1%` median relative gain and the rerank confidence interval overlaps zero.
  - pause-and-diagnose when all new-stage candidates regress on amplitude SSIM at `N=256` and the same direction appears in `N=128` robustness summaries.
  - before halting an axis, run one bounded rescue mini-sweep; if still failing, pause expansion on that axis and carry at least one hedge candidate forward under low budget.

Runbook CLI (Stage A full N=128 example):

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
  --promotion-objectives amp_ssim,train_wall_time_sec \
  --max-train-seconds-n128 2700 \
  --max-inference-seconds-n128 60 \
  --max-phase-ssim-drop 0.03 \
  --max-model-params 300000000 \
  --output-root outputs/hybrid_resnet_mode_skip_sweep_full_n128_20260221 \
  --seed 3 \
  --no-probe-mask \
  --no-torch-mae-pred-l2-match-target
```

Runbook CLI (seed-rerank aggregation example):

```bash
python scripts/studies/runbooks/run_hybrid_resnet_mode_skip_sweep.py \
  --stage-id A \
  --aggregate-seed-rerank-root outputs/hybrid_resnet_mode_skip_sweep_full_n128_20260221/seed_rerank \
  --source-summary outputs/hybrid_resnet_mode_skip_sweep_full_n128_20260221/summary.csv \
  --promotion-objectives amp_ssim,train_wall_time_sec \
  --top-k-n256 6 \
  --emit-stage-anchor-summary outputs/hybrid_resnet_mode_skip_sweep_full_n128_20260221/promotion/stage_anchor_summary.csv \
  --emit-robust-promotion-summary outputs/hybrid_resnet_mode_skip_sweep_full_n128_20260221/promotion/summary_seed_robust.csv
```

### `nersc-scan807-cameraman-ptychovit-hybrid-orchestration`

- Purpose: Run checkpoint-restored `pinn_ptychovit` inference on `scan807` and `cameraman256`, train `pinn_hybrid_resnet` on cameraman top/bottom-half (`N=128`, 40 epochs), run checkpoint-reuse hybrid inference across both full datasets, and aggregate per-dataset metrics/visuals.
- Script: `scripts/studies/runbooks/run_nersc_scan807_cameraman_study.py`
- Canonical PtychoViT checkpoint for this study: `datasets/run145/best_model.pth` (required; do not substitute ad-hoc `tmp/ptychovit_initial_*` checkpoints).
- Position reassembly backend for external hybrid inference is pinned to `shift_sum` (`--position-reassembly-backend shift_sum` only).
- N=128 prep semantics are configurable via `--downsample-policy`:
  - `bin-crop` (default): diffraction is block-binned; `objectGuess`/`probeGuess` are center-cropped; coordinates remain in the same pixel frame.
  - `crop-bin`: diffraction is center-cropped; `objectGuess`/`probeGuess` are block-binned; coordinates are scaled by `1/factor`.
- Multimode probe collapse policy is configurable via `--probe-mode-policy`:
  - `incoherent_aggregate` (default): single-probe collapse with incoherent amplitude aggregation.
  - `first_mode`: compatibility fallback that uses only mode 0.
- Core helpers:
  - `scripts/studies/nersc_pair_adapter.py`
  - `scripts/studies/prepare_nersc_hybrid_dataset.py`
  - `scripts/studies/nersc_orchestration.py`
  - `scripts/studies/hybrid_checkpoint_inference.py`

CLI entry point (full command):

```bash
python scripts/studies/runbooks/run_nersc_scan807_cameraman_study.py \
  --scan807-dp /home/ollie/Downloads/nersc/testdata/scan807_dp.hdf5 \
  --scan807-para /home/ollie/Downloads/nersc/testdata/scan807_para.hdf5 \
  --cameraman-dp /home/ollie/Downloads/nersc/data/cameraman256_dp.hdf5 \
  --cameraman-para /home/ollie/Downloads/nersc/data/cameraman256_para.hdf5 \
  --ptychovit-checkpoint datasets/run145/best_model.pth \
  --half top \
  --downsample-policy bin-crop \
  --probe-mode-policy incoherent_aggregate \
  --position-reassembly-backend shift_sum \
  --output-dir outputs/nersc_scan807_cameraman_study \
  --seed 3
```

### `nersc-scan807-cameraman-ptychovit-hybrid-orchestration-n256-no-downsample`

- Purpose: Companion to the `N=128` orchestration that keeps both model arms on `N=256` end-to-end (no 256->128 conversion), while preserving the same staged workflow and strict `shift_sum` reassembly policy.
- Script: `scripts/studies/runbooks/run_nersc_scan807_cameraman_study_n256.py`
- Smoke/full rule: use the same command, changing only `--epochs`:
  - smoke: `--epochs 5`
  - full: `--epochs 40`
- Core helpers:
  - `scripts/studies/prepare_nersc_hybrid_dataset.py` (explicit no-downsample path when `target_n == source_n`)
  - `scripts/studies/nersc_orchestration.py` (`target_n` + `epochs` threaded through training/inference)

CLI entry point (smoke example):

```bash
python scripts/studies/runbooks/run_nersc_scan807_cameraman_study_n256.py \
  --scan807-dp /home/ollie/Downloads/nersc/testdata/scan807_dp.hdf5 \
  --scan807-para /home/ollie/Downloads/nersc/testdata/scan807_para.hdf5 \
  --cameraman-dp /home/ollie/Downloads/nersc/data/cameraman256_dp.hdf5 \
  --cameraman-para /home/ollie/Downloads/nersc/data/cameraman256_para.hdf5 \
  --ptychovit-checkpoint datasets/run145/best_model.pth \
  --half top \
  --probe-mode-policy incoherent_aggregate \
  --epochs 5 \
  --output-dir outputs/nersc_scan807_cameraman_study_n256_no_downsample_smoke \
  --seed 3
```

### `nersc-scan807-cameraman-n128-factorial-probe-mask-mae-downsample`

- Purpose: Run a sequential `N=128` factorial sweep over:
  - probe mask mode: `off`, `on_soft`, `on_hard`
  - Torch MAE prediction-L2 matching: `off`, `on`
  - downsample policy: `bin-crop`, `crop-bin`
- Script: `scripts/studies/runbooks/run_nersc_scan807_cameraman_study_n128_factorial.py`
- Matrix size: `3 * 2 * 2 = 12` runs per sweep.
- Epoch policy: fixed for the whole sweep (`--epochs 20` or `--epochs 40`), not a matrix axis.
- Output contract:
  - `factorial_manifest.json` at study root
  - per-run outputs under `runs/<run_id>/`

Runbook CLI (full example):

```bash
python scripts/studies/runbooks/run_nersc_scan807_cameraman_study_n128_factorial.py \
  --scan807-dp /home/ollie/Downloads/nersc/testdata/scan807_dp.hdf5 \
  --scan807-para /home/ollie/Downloads/nersc/testdata/scan807_para.hdf5 \
  --cameraman-dp /home/ollie/Downloads/nersc/data/cameraman256_dp.hdf5 \
  --cameraman-para /home/ollie/Downloads/nersc/data/cameraman256_para.hdf5 \
  --ptychovit-checkpoint datasets/run145/best_model.pth \
  --half top \
  --epochs 40 \
  --soft-mask-sigma 1.0 \
  --output-root outputs/nersc_scan807_cameraman_study_n128_factorial_$(date +%Y%m%d_%H%M%S) \
  --seed 3
```

Collation script:

```bash
python scripts/studies/collate_nersc_n128_factorial_results.py \
  --factorial-root outputs/<factorial_run_dir> \
  --shared-dir outputs/<factorial_run_dir>/comparison_bundle
```

Collation outputs:
- `comparison_bundle/metrics_summary.csv`
- `comparison_bundle/metrics_summary.md`
- `comparison_bundle/shared_pngs/{run_id}__dataset-{dataset}__compare_amp_phase.png`

### `grid-lines-external-fly001-n128-top-train-full-test-e40`

- Purpose: Run external-raw `fly001` study at `N=128` with top-half train and full-object test (no additional subsampling), comparing Torch `cnn` and `hybrid_resnet`.
- Script: `scripts/studies/runbooks/grid_lines_external_fly001_n128_top_train_full_test_e40.sh`
- Output directory: `outputs/grid_lines_external_fly001_n128_top_train_full_test_e40_seed3_cnn_hybrid_resnet`
  - Rerun output: `outputs/grid_lines_external_fly001_n128_top_train_full_test_e40_seed3_cnn_hybrid_resnet_rerun_20260216_213242_pty`
- Dataset inputs:
  - `datasets/fly001_128/fly001_128_top_half_converted.npz`
  - `datasets/fly001_128/fly001_128_full_test_converted.npz`
  - `datasets/fly001_128/manifest.json`
- Position reassembly strategy (external mode):
  - Default is `auto` (`--torch-position-reassembly-backend auto`)
  - `auto` prefers `shift_sum` and falls back to `batched` on TF OOM.
  - Use explicit batched mode only as an opt-in override:
    - `--torch-position-reassembly-backend batched`
    - `--torch-position-reassembly-batch-size 32`

CLI entry point (full command):

```bash
bash scripts/studies/runbooks/grid_lines_external_fly001_n128_top_train_full_test_e40.sh
```

### `grid-lines-n64-pinn-hybrid-resnet-e20`

- Purpose: Run `N=64` grid-lines with `pinn` (TF) and `pinn_hybrid_resnet` (Torch) at `20` epochs, then render combined visuals.
- Script: `.artifacts/studies/grid_lines_n64_pinn_hybrid_resnet_e20/run_study.sh`
- Output directory: `outputs/grid_lines_n64_compare_padex_lr2e4_plateau_e20_seed3_pinn_hybrid_resnet`
- Invocation artifacts:
  - `outputs/grid_lines_n64_compare_padex_lr2e4_plateau_e20_seed3_pinn_hybrid_resnet/invocation.json`
  - `outputs/grid_lines_n64_compare_padex_lr2e4_plateau_e20_seed3_pinn_hybrid_resnet/invocation.sh`
  - `outputs/grid_lines_n64_compare_padex_lr2e4_plateau_e20_seed3_pinn_hybrid_resnet/runs/pinn_hybrid_resnet/invocation.json`
  - `outputs/grid_lines_n64_compare_padex_lr2e4_plateau_e20_seed3_pinn_hybrid_resnet/runs/pinn_hybrid_resnet/invocation.sh`

CLI entry points (full commands):

```bash
python scripts/studies/grid_lines_compare_wrapper.py \
  --N 64 \
  --gridsize 1 \
  --output-dir outputs/grid_lines_n64_compare_padex_lr2e4_plateau_e20_seed3_pinn_hybrid_resnet \
  --models pinn \
  --nimgs-train 2 \
  --nimgs-test 1 \
  --nphotons 1e9 \
  --nepochs 20 \
  --batch-size 16 \
  --seed 3 \
  --probe-source custom \
  --probe-scale-mode pad_extrapolate \
  --set-phi
```

```bash
python scripts/studies/grid_lines_torch_runner.py \
  --output-dir outputs/grid_lines_n64_compare_padex_lr2e4_plateau_e20_seed3_pinn_hybrid_resnet \
  --architecture hybrid_resnet \
  --train-npz outputs/grid_lines_n64_compare_padex_lr2e4_plateau_e20_seed3_pinn_hybrid_resnet/datasets/N64/gs1/train.npz \
  --test-npz outputs/grid_lines_n64_compare_padex_lr2e4_plateau_e20_seed3_pinn_hybrid_resnet/datasets/N64/gs1/test.npz \
  --N 64 \
  --gridsize 1 \
  --epochs 20 \
  --batch-size 16 \
  --infer-batch-size 16 \
  --learning-rate 2e-4 \
  --scheduler ReduceLROnPlateau \
  --plateau-factor 0.5 \
  --plateau-patience 2 \
  --plateau-min-lr 1e-4 \
  --plateau-threshold 0.0 \
  --seed 3 \
  --optimizer adam \
  --weight-decay 0.0 \
  --beta1 0.9 \
  --beta2 0.999 \
  --torch-loss-mode mae \
  --output-mode real_imag \
  --probe-source custom \
  --fno-modes 12 \
  --fno-width 32 \
  --fno-blocks 4 \
  --fno-cnn-blocks 2 \
  --torch-logger mlflow
```

```bash
python - <<'PY'
from pathlib import Path
from ptycho.workflows.grid_lines_workflow import render_grid_lines_visuals

out = Path("outputs/grid_lines_n64_compare_padex_lr2e4_plateau_e20_seed3_pinn_hybrid_resnet")
render_grid_lines_visuals(out, order=("gt", "pinn", "pinn_hybrid_resnet"))
print("Rendered visuals under", out / "visuals")
PY
```

### `grid-lines-n64-pinn-only-retry1-e20-seed13`

- Purpose: Rebuild `pinn` recon for `N=64` (`seed=13`, `20` epochs), reuse an existing `pinn_hybrid_resnet` recon, then regenerate merged comparison metrics/visuals in one output directory.
- Script: `.artifacts/studies/grid_lines_n64_pinn_hybrid_resnet_e20_retry1/finish_from_completed_pinn.sh`
- Output directory: `outputs/grid_lines_n64_compare_padex_lr2e4_plateau_e20_seed13_pinn_only_retry1`
- Invocation artifacts:
  - `outputs/grid_lines_n64_compare_padex_lr2e4_plateau_e20_seed13_pinn_only_retry1/invocation.json`
  - `outputs/grid_lines_n64_compare_padex_lr2e4_plateau_e20_seed13_pinn_only_retry1/invocation.sh`

CLI entry points (full commands):

```bash
python scripts/studies/grid_lines_compare_wrapper.py \
  --N 64 \
  --gridsize 1 \
  --output-dir /home/ollie/Documents/tmp/PtychoPINN/outputs/grid_lines_n64_compare_padex_lr2e4_plateau_e20_seed13_pinn_only_retry1 \
  --probe-npz datasets/Run1084_recon3_postPC_shrunk_3.npz \
  --models pinn,pinn_hybrid_resnet \
  --reuse-existing-recons \
  --seed 13 \
  --nimgs-train 2 \
  --nimgs-test 1 \
  --nphotons 1e9 \
  --nepochs 20 \
  --batch-size 16 \
  --nll-weight 0 \
  --mae-weight 1 \
  --realspace-weight 0 \
  --probe-source custom \
  --probe-scale-mode pad_extrapolate \
  --set-phi \
  --torch-epochs 20 \
  --torch-batch-size 16 \
  --torch-learning-rate 2e-4 \
  --torch-scheduler ReduceLROnPlateau \
  --torch-plateau-patience 2 \
  --torch-plateau-factor 0.5 \
  --torch-plateau-min-lr 1e-4 \
  --torch-plateau-threshold 0 \
  --torch-output-mode amp_phase \
  --torch-loss-mode poisson \
  --torch-grad-clip 0 \
  --torch-grad-clip-algorithm norm \
  --torch-resnet-width 64
```
