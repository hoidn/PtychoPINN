# FNO Hyperparameter Study Design

**Goal:** Systematically evaluate FNO/Hybrid hyperparameters to map the trade‑off between reconstruction quality (phase SSIM/PSNR) and computational cost (parameter count/inference time), and identify a “sweet spot” configuration.

**Scope:** Training configuration only (input transform, modes, width, architecture), using the existing grid‑lines dataset (N=64, gridsize=1). No changes to core physics/model code; use the existing torch runner and grid‑lines workflow.

## Architecture

**Core idea:** Add minimal statistics to the Torch runner (parameter count + inference timing), then create a standalone sweep orchestrator that reuses cached grid‑lines data and aggregates results.

Components:
1. **Grid‑lines data harness**: Reuse cached train/test NPZ from `grid_lines_workflow` (N=64, gridsize=1). If missing, generate once.
2. **Torch runner enhancements**: `grid_lines_torch_runner.py` will report:
   - `model_params` (trainable parameter count)
   - `inference_time_s` (wall time for inference; synchronize CUDA if available)
3. **Sweep orchestrator**: New `scripts/studies/fno_hyperparam_study.py` that:
   - Iterates a fixed grid (architecture ∈ {fno, hybrid}; input_transform ∈ {none, log1p, sqrt}; modes ∈ {12, 16, 24}; width ∈ {32, 48, 64}).
   - Calls `run_grid_lines_torch` for each config.
   - Aggregates metrics + stats into `study_results.csv`.
   - Generates a simple Pareto scatter plot (param count vs phase SSIM/PSNR) with series keyed by input_transform or modes.

## Data Flow

1. **Prepare data**: Ensure `output_dir/datasets/N64/gs1/train.npz` and `test.npz` exist.
2. **Run sweep**: For each config, call `run_grid_lines_torch` with cached NPZs.
3. **Collect metrics**:
   - Extract phase metrics explicitly (index 1 of `[amp, phase]`).
   - Record `model_params`, `inference_time_s`, and config values.
4. **Write results**: Append to CSV and save a simple plot.

## Error Handling

- If a run errors, log the error in the CSV row and continue (do not abort the sweep).
- If metrics are missing, write `NaN` and include the error message.
- Timing uses `time.perf_counter()`; if CUDA is available, call `torch.cuda.synchronize()` before/after inference to avoid under‑reporting.

## Testing Strategy

- **Torch runner test**: Assert `run_grid_lines_torch` returns `model_params` and `inference_time_s` using mocked training/inference.
- **Sweep orchestrator test**: Monkeypatch `run_grid_lines_torch` to return fixed metrics; verify CSV columns and row count.
- **Phase metric extraction test**: Ensure `[amp, phase]` arrays are parsed to phase metrics.

## Outputs

- `outputs/fno_hyperparam_study/<run_id>/study_results.csv`
- `outputs/fno_hyperparam_study/<run_id>/pareto_plot.png`

## Non‑Goals

- No training‑time hyperparameter sweeps (learning rate, batch size, etc.).
- No modifications to core physics or model internals.
