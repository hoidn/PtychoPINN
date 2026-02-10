# PtychoViT Workflow

**Status:** implementation contract for `pinn_ptychovit` backend integration in grid-lines studies.

This document defines the source-pinned interop and checkpoint contracts used by PtychoPINN when dispatching the `pinn_ptychovit` model arm via subprocess execution.

## Scope

- Add `pinn_ptychovit` as a selectable model arm in studies.
- Keep data adaptation isolated in a dedicated interop layer (`NPZ -> paired HDF5`).
- Compare explicitly selected models only.
- Evaluate cross-model metrics in object space on one canonical GT object grid.

## Interop Contract Source

Contract fields:

```yaml
source_repo: /home/ollie/Documents/ptycho-vit
source_commit: 2316b378006ef330e18af343d10dc8a7b821b0a8
source_paths:
  - data.py
  - scripts/make_normalization_dict.py
  - tests/test_utils.py
validated_on: 2026-02-10
```

Source-derived HDF5 pairing and required datasets:

- Data pair naming:
  - `{object_name}_dp.hdf5`
  - `{object_name}_para.hdf5`
- Required in `*_dp.hdf5`:
  - dataset `dp` (scan-major diffraction tensor; rank-3 `[Nscan, H, W]`)
- Required in `*_para.hdf5`:
  - dataset `object`
  - dataset `probe`
  - dataset `probe_position_x_m`
  - dataset `probe_position_y_m`
- Required attrs in `object` and `probe` datasets for PtychoPINN compatibility validation:
  - `pixel_height_m`
  - `pixel_width_m`

Adapter mapping from grid-lines NPZ to paired HDF5:

- `npz["diffraction"]` amplitude is converted to intensity for `dp` via `dp = diffraction**2`.
- `npz["YY_full"]` maps to `object` (primary, scan-consistent object geometry).
  Fallback to `npz["YY_ground_truth"]` only when `YY_full` is absent.
- `npz["probeGuess"]` maps to `probe`.
- Absolute scan positions from `npz["coords_offsets"]` (or `coords_start_offsets`) map to
  `probe_position_y_m`/`probe_position_x_m` after conversion into the centered coordinate frame
  expected by upstream `PtychographyDataset`.

## Checkpoint Contract Source

Contract fields:

```yaml
source_repo: /home/ollie/Documents/ptycho-vit
source_commit: 2316b378006ef330e18af343d10dc8a7b821b0a8
source_paths:
  - main.py
  - training.py
  - inference.py
  - config.yaml
validated_on: 2026-02-10
```

Source-derived checkpoint files and semantics:

- `best_model.pth`:
  - used by inference load path (`inference.py`) for prediction.
  - default initialization for inference-only/fine-tune bootstrap.
- `checkpoint_model.pth`:
  - rolling model-weight checkpoint during training (`training.py`).
  - used by resume path in `main.py`.
- `checkpoint.state`:
  - optimizer/scheduler state and epoch/loss trackers (`training.py`).
  - consumed with `checkpoint_model.pth` on resume.
- `config.yaml`:
  - persisted into the run directory for reproducibility (`training.py`).

Resume and restore policy:

- Resume interrupted run:
  - require `checkpoint_model.pth` and `checkpoint.state`
  - set `training.resume_from_checkpoint: true`
- Fine-tune via bridge:
  - currently resume-only (same upstream contract as interrupted-run resume)
  - bridge requires `--resume-from-checkpoint=true` and rejects scratch-style finetune calls
- Inference-only:
  - load `best_model.pth` unless an explicit override is provided

## Backend Contract

- Model ID: `pinn_ptychovit`
- Supported resolution in v1: `N=256` only
- Adapter-owned artifacts:
  - paired HDF5 files
  - adapter manifest/provenance summary
- Runner handoff artifact:
  - `recons/pinn_ptychovit/recon.npz`

## Fine-Tuning

Current policy:

- use the upstream resume contract (`checkpoint_model.pth` + `checkpoint.state`)
- bridge-enforced guardrail: `--mode finetune` requires `--resume-from-checkpoint=true`
- keep learning-rate partitioning as an upstream concern (no local override policy yet)

## Inference

Inputs:

- model run directory containing `best_model.pth` and `config.yaml`
- paired test HDF5 files (`*_dp.hdf5`, `*_para.hdf5`)

Outputs:

- subprocess logs (`runs/pinn_ptychovit/stdout.log`, `runs/pinn_ptychovit/stderr.log`)
- reconstruction artifact (`recons/pinn_ptychovit/recon.npz`)
- per-model metrics entry in `metrics_by_model.json`

## Fresh Initial Baseline (Checkpoint-Restored, Lines Synthetic)

Use this runbook when you need trustworthy initial metrics and recon PNGs from a **fresh** execution path (no stale artifact reuse).

Prerequisites:

- A checkpoint to restore from (`best_model.pth`).
- Clean output directory (or pass `--force-clean` in the helper script).
- Local ptycho-vit checkout path.

Recommended command (helper script):

```bash
python scripts/studies/run_fresh_ptychovit_initial_metrics.py \
  --checkpoint <absolute-path-to-best_model.pth> \
  --output-dir tmp/ptychovit_initial_fresh \
  --ptychovit-repo /home/ollie/Documents/ptycho-vit \
  --N 128 --gridsize 1 --nimgs-train 8 --nimgs-test 8 --set-phi
```

Equivalent direct wrapper command:

```bash
python scripts/studies/grid_lines_compare_wrapper.py \
  --N 128 --gridsize 1 \
  --output-dir tmp/ptychovit_initial_fresh \
  --architectures hybrid \
  --models pinn_ptychovit \
  --model-n pinn_ptychovit=256 \
  --ptychovit-repo /home/ollie/Documents/ptycho-vit \
  --nimgs-train 8 --nimgs-test 8 --set-phi
```

Reuse policy:

- `--reuse-existing-recons` is optional and **disabled by default**.
- Do not pass `--reuse-existing-recons` when generating initial baseline metrics.

Verification command:

```bash
python scripts/studies/verify_fresh_ptychovit_initial_metrics.py \
  --output-dir tmp/ptychovit_initial_fresh
```

Expected artifacts:

- `tmp/ptychovit_initial_fresh/metrics_by_model.json`
- `tmp/ptychovit_initial_fresh/recons/pinn_ptychovit/recon.npz`
- `tmp/ptychovit_initial_fresh/recons/gt/recon.npz`
- `tmp/ptychovit_initial_fresh/visuals/amp_phase_pinn_ptychovit.png`
- `tmp/ptychovit_initial_fresh/visuals/compare_amp_phase.png`
- `tmp/ptychovit_initial_fresh/runs/pinn_ptychovit/manifest.json`

## Evaluation Policy

- Native backends may use different diffraction patch sizes.
- Cross-model metrics run after harmonizing each reconstructed object to the canonical GT object grid.
- Harmonization target is GT object space, not diffraction patch size `N`.
- No physical-unit harmonization in v1 (pixel-space canonicalization only).

## Runtime Assumptions

- `probe_position_x_m/probe_position_y_m must be non-constant` for lines interop runs.
- Position vectors in HDF5 must be in the centered frame expected by upstream loader
  (`data.py` adds origin internally).
- `object` geometry in `*_para.hdf5` must be scan-consistent with `dp` and probe positions.
- The bridge must generate a runtime normalization dictionary and set both
  `data.normalization_dict_path` and `data.test_normalization` to that file.
- If stdout contains `Normalization file not found`, treat the run as invalid and
  regenerate bridge outputs.

## Reconstruction Assembly Contract

- Bridge inference must reconstruct object-space output using scan-position-aware stitching
  (equivalent to upstream `place_patches_fourier_shift` + occupancy normalization).
- scan-wise mean aggregation (simple patch averaging via `mean` across scan predictions) is not contract-compliant for
  PtychoViT object reconstruction and can produce flat/low-information reconstructions.

## Troubleshooting

- Missing dataset keys in HDF5 pair:
  - check required key set in this document and regenerate adapter outputs.
- Probe-position length mismatch:
  - ensure `len(probe_position_x_m) == len(probe_position_y_m) == dp.shape[0]`.
- Probe positions degenerate (constant zeros):
  - inspect `runs/pinn_ptychovit/bridge_work/data/test_para.hdf5`; regenerate NPZ/HDF5 interop
    if `probe_position_x_m` or `probe_position_y_m` is constant.
- Frame mismatch or out-of-bounds patch extraction:
  - if positions are unique but reconstruction remains poor, verify centered-frame semantics
    and object-size consistency (`YY_full` should drive `object` when available).
- Normalization fallback warning:
  - if stdout includes `Normalization file not found`, runtime config did not point to a valid
    normalization dictionary; regenerate via bridge entrypoint.
- Flat reconstruction despite valid contract files:
  - check bridge assembly logic; if it averages predicted patches instead of using scan-position
    stitching, results can remain poor even with correct normalization/probe inputs.
- Resume failure:
  - confirm both `checkpoint_model.pth` and `checkpoint.state` are present in the run directory.
- Inference load failure:
  - confirm `best_model.pth` exists and matches the expected model config.
