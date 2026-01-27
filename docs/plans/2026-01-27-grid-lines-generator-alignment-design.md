# Grid-Lines ⇄ Modular Generator Alignment Design

Date: 2026-01-27

## Summary
Align the modular generator registry plan with the grid-lines workflow so grid-lines is the canonical harness for generator comparisons on a simple gridsize=1 dataset. The workflow must support multiple generator architectures per run, reuse cached datasets via manifest + seed, and emit standardized run artifacts. A separate comparison script aggregates per-run metrics into a single JSON report. The supervised Baseline is included as a comparator and must be labeled distinctly from generator CNNs.

## Goals
- One shared dataset for parallel generator comparisons (gridsize=1, lines simulation).
- Deterministic reuse with caching (manifest + seed) to avoid regeneration.
- Consistent outputs for PINN generators and supervised Baseline.
- Clear naming that separates `pinn_cnn` from `baseline`.
- Simple JSON aggregation for metrics vs ground truth.

## Key Decisions
- **Harness:** `grid_lines_workflow` is the canonical test harness.
- **Architecture selection:** CLI supports `--architectures cnn,fno,hybrid`.
- **Dataset reuse:** Cache under `output_dir/datasets/N{N}/gs{g}/` with manifest; reuse when matching.
- **Artifacts:** Per-run outputs under `output_dir/runs/<model_id>/` with predictions NPZ, stitched PNGs, and metrics JSON.
- **Aggregation:** `scripts/studies/grid_lines_compare.py` produces a single JSON report across runs.
- **Naming:** Generator runs labeled `pinn_<arch>`; supervised baseline labeled `baseline`.

## Data Flow
1. Prepare or reuse dataset (seed + manifest).
2. For each generator architecture, run PINN training/inference and emit run artifacts.
3. Run supervised Baseline once and emit run artifacts.
4. Compare script aggregates metrics into JSON.

## Testing Focus
- Dataset reuse/manifest matching.
- Run naming and artifact layout.
- JSON aggregation across runs.
- Unknown architecture handling (fail fast).

## Documentation
- Add generator README guidance in `ptycho/generators/README.md` and `ptycho_torch/generators/README.md` covering registration, output contract, naming, and tests.
