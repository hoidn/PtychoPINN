# D0 Parity Logger Requirements

## Objective
Promote the ad-hoc `dose_baseline_snapshot` probe/dataset summary into a reusable CLI under `scripts/tools/` that the maintainer can run for any dose_experiments scenario. The CLI must emit machine-readable JSON plus a Markdown digest capturing dataset parity evidence, probe stats, and baseline inference metrics for the selected photon dose sweep.

## Inputs
- `--dataset-root`: directory containing `data_p1e*.npz` files (at least one). Default: `photon_grid_study_20250826_152459`.
- `--baseline-params`: path to `params.dill` captured from the authoritative baseline run.
- `--scenario-id`: string identifier (e.g., `PGRID-20250826-P1E5-T1024`).
- `--output`: directory for generated artifacts (JSON, Markdown, raw tables).
- `--limit-datasets` *(optional)*: comma-separated list of dataset filenames; when omitted, auto-detect via glob.

## Outputs
1. `dose_parity_log.json` — Structured summary with:
   - Metadata block (scenario_id, dataset_root, baseline_params, timestamp, git SHA via `subprocess.check_output(['git','rev-parse','HEAD'])`).
   - Dataset entries (`filename`, `photon_dose`, `sha256`, `size_bytes`, `arrays` shape/dtype map, stage stats, grouped counts).
   - Probe stats (magnitude + phase percentiles, min/max, L2 norm).
   - Baseline params excerpt (`N`, `gridsize`, train/test counts, batch_size, nepochs, loss weights, probe/intensity flags, intensity scale value, timestamp, label, ms_ssim/psnr/mae/mse/frc50 tuples).
   - Stage-level stats: see below.
2. `dose_parity_log.md` — Human-readable table quoting the JSON values plus problem statement for maintainers.
3. `probe_stats.csv` *(optional but desired)* — CSV with amplitude/phase percentiles for quick diffing.

## Stage-Level Stats Definition
For every dataset NPZ:
- **Raw diffraction**: operate on `diff3d` values as stored. Report `min`, `max`, `mean`, `std`, `median`, `p01`, `p10`, `p90`, `p99`, `nonzero_fraction`.
- **Grouped intensity**: treat `scan_index` as grouping key. Compute per-scan average intensity (mean over each frame's pixels, then `np.bincount` to aggregate). Report stats over those per-scan averages + number of unique scans.
- **Normalized diffraction**: divide `diff3d` by `(diff3d.max() + 1e-12)` and compute the same summary stats as raw. This mirrors the normalized tensor that feeds training after amplitude scaling, covering the "normalized" portion of the maintainer request.

## Probe Logging
- Extract `probeGuess` array.
- Compute amplitude (`np.abs`) and phase (`np.angle`). Record `min`, `max`, `mean`, `std`, and percentiles (1, 5, 50, 95, 99) for each.
- Include boolean flags derived from params (`probe.trainable`, `probe.mask`).

## Intensity Scale + Metrics
- Convert `params.get('intensity_scale')` into a float; log alongside `intensity_scale.trainable`.
- Metrics to copy directly: `mae`, `ms_ssim`, `psnr`, `mse`, `frc50`. Format as `[train, test]` pairs.

## Implementation Notes
- Lift helper functions from `plans/active/seed/bin/dose_baseline_snapshot.py` where possible, but move the code into `scripts/tools/d0_parity_logger.py` to make it part of the shipping tools.
- Structure the module with pure helpers so we can unit-test them:
  - `sha256_file(path)`
  - `summarize_array(arr)` → returns stats dictionary given a NumPy array.
  - `summarize_grouped(diff3d, scan_index)` → returns stats dict for grouped intensities.
  - `summarize_probe(probe_array)`.
  - `load_params(path)` → wraps dill loading + conversion to plain Python types.
- Provide a `main()` entry point guarded by `if __name__ == "__main__"` so maintainers can run `python scripts/tools/d0_parity_logger.py --dataset-root ...`.
- Keep outputs ASCII/UTF-8; avoid Pandas dependencies.

## Testing Expectations
- Add `tests/tools/test_d0_parity_logger.py` with fixtures that build a tiny synthetic NPZ + fake `params.dill` to validate:
  - `summarize_array` percentiles and `nonzero_fraction` logic.
  - `summarize_grouped` correctly handles repeated scan indices.
  - CLI `main` produces JSON + MD files given the synthetic dataset (use tmp_path, 2×2 arrays to keep runtime low).
- Selector to run: `pytest tests/tools/test_d0_parity_logger.py::test_cli_emits_outputs -q`.

## Reporting + Artifacts
- Place generated JSON/MD into `$ARTIFACT_DIR` (timestamped directory under `plans/active/seed/reports/`).
- Update `docs/fix_plan.md` attempts history with new evidence.
- Summaries should cite `inbox/README_prepare_d0_response.md` for parity requirements and explain any interpretations (e.g., normalized stats derived from max scaling).
