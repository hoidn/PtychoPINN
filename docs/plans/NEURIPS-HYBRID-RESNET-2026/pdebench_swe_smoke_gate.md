# PDEBench SWE Smoke Gate

## Scope

This is the Roadmap Phase 2 smoke/data-contract gate for the selected PDEBench 2D Shallow Water Equations primary benchmark. It validates official data access, HDF5 layout, deterministic split behavior, local one-step metric computation, and tiny supervised Hybrid ResNet, FNO, and U-Net train/eval feasibility.

Non-goals confirmed: no full PDE training, no rollout evaluation, no ablation, no CDI regeneration, no `256x256` scaling, and no paper-facing artifact assembly.

## Dataset Identity

- Source: PDEBench repository, `https://github.com/pdebench/PDEBench`.
- Accessed datafile: DaRUS datafile `133021`, `https://darus.uni-stuttgart.de/api/access/datafile/133021`.
- Official file: `2D_rdb_NA_NA.h5`.
- Local path: `/home/ollie/Documents/pdebench-data/swe/2D_rdb_NA_NA.h5`.
- Size: `6626098972` bytes.
- SHA256: `28f0c33723d70eebb420fc170e94b675c18e032fb697dcef080e114ca9645e3a`.
- File mtime: `2026-04-20T10:50:24.183332+00:00`.
- Disk feasibility at verification: `/` reported `25G` available and `95%` used after materializing the 6.2 GiB HDF5 file, so longer runs should keep bulky outputs under ignored or external storage.
- License/access note carried forward: record the PDEBench repository/data license and DaRUS access terms before longer execution.

## HDF5 Layout

- Selected state dataset: virtual grouped dataset `*/data`, with pattern `{trajectory_id:04d}/data`.
- Trajectories: `1000` numeric groups, `0000` through `0999`.
- Per-trajectory state shape: `[101, 128, 128, 1]`.
- Virtual state shape: `[1000, 101, 128, 128, 1]`.
- Axis order: `NTHWC`.
- Dtype: `float32`.
- Grid fields per trajectory: `grid/t` shape `[101]`, `grid/x` shape `[128]`, and `grid/y` shape `[128]`.
- Channel contract: one real SWE state channel, adapted to supervised PyTorch tensors as `(C,H,W)` and emitted directly as real state, not CDI complex output.

## Split Manifest

- Seed: `20260420`.
- Ratio contract: `0.8 / 0.1 / 0.1` train/val/test trajectory split before smoke caps.
- Official smoke caps: `4` train trajectories, `2` val trajectories, `2` test trajectories, and `1` one-step pair per trajectory.
- Smoke train IDs: `[744, 785, 712, 648]`.
- Smoke val IDs: `[645, 412]`.
- Smoke test IDs: `[114, 515]`.
- Pair counts: train `4`, val `2`, test `2`.
- Horizon: one-step next-state prediction.

## Normalization And Metrics

- Normalization source: train split inputs only.
- Normalization sample count: `2` batches/samples as bounded by the smoke settings.
- Mean: `[1.0167236328125]`.
- Std: `[0.1282339772371296]`.
- Metrics: local `err_RMSE` and `err_nRMSE`.
- Metric units: denormalized physical units.
- Primary smoke metric: aggregate `err_nRMSE` over the documented one-step evaluation axes, with per-channel values emitted in each `metrics.json`.

## Smoke Command

The official run used run ID `20260420T105459.340303Z`, tracked PID `3037364`, CUDA device execution, and this bounded scope:

```bash
python scripts/studies/run_pdebench_swe_smoke.py \
  --data-file /home/ollie/Documents/pdebench-data/swe/2D_rdb_NA_NA.h5 \
  --output-root .artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-swe-primary-smoke-gate \
  --dataset-source PDEBench \
  --dataset-source-url https://github.com/pdebench/PDEBench \
  --dataset-darus-id 133021 \
  --split-seed 20260420 \
  --train-fraction 0.8 \
  --val-fraction 0.1 \
  --test-fraction 0.1 \
  --models hybrid_resnet,fno,unet \
  --epochs 1 \
  --batch-size 2 \
  --learning-rate 1e-3 \
  --max-train-trajectories 4 \
  --max-val-trajectories 2 \
  --max-test-trajectories 2 \
  --max-pairs-per-trajectory 1 \
  --max-train-batches 2 \
  --max-eval-batches 2 \
  --device cuda \
  --num-workers 0 \
  --run-id 20260420T105459.340303Z \
  --allow-existing-output-root
```

## Model Results

| Model | Status | Parameters | Train batches | Eval pairs | err_RMSE | err_nRMSE | Runtime sec | Peak CUDA bytes |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Hybrid ResNet | metrics written | 22034 | 2 | 2 | 0.1624556035 | 0.1509005725 | 0.4910554886 | 69120512 |
| FNO | metrics written | 3133 | 2 | 2 | 0.2270323634 | 0.2108841538 | 0.7164065838 | 69120512 |
| U-Net | metrics written | 4689 | 2 | 2 | 0.2158706486 | 0.2005163431 | 0.0134313107 | 69120512 |

No model wrote a blocker artifact. FNO was importable and runnable in the environment, though the package-version probe recorded `neuralop: null`, so longer-run provenance should preserve the installed package source if that matters for the paper record.

## Runtime And Provenance

- Python executable: `/home/ollie/miniconda3/envs/ptycho311/bin/python`.
- Python version: `3.11.13`.
- PyTorch: `2.9.1+cu128`.
- CUDA version: `12.8`.
- GPU: `NVIDIA GeForce RTX 3090`.
- h5py: `3.14.0`.
- Git commit at run time: `2f8e13cec1484f3ac2a109106b022efae06fc876`.
- Git dirty state: recorded in per-model `provenance.json`; unrelated pre-existing workspace changes were present and were not reverted.
- Root smoke exit code: `0`.
- Freshness marker: `logs/smoke.started_at_ns` recorded `1776682499361062641`.

## Gate Checks

- Official HDF5 was materialized outside the repository.
- Inspect-only mode wrote current dataset and HDF5 metadata for `2D_rdb_NA_NA.h5`.
- The selected grouped SWE state layout is unambiguous after applying the virtual `*/data` adapter.
- The bounded smoke loader reads one-step pairs lazily instead of loading the full HDF5 file into memory.
- Hybrid ResNet-compatible one-step training/evaluation succeeded.
- Both baseline smoke runs succeeded, so the OpenFWI FlatVel-A fallback pivot is not triggered by this gate.
- Freshness verification passed: root manifests and per-model provenance/results were written after the tracked start marker, record run ID `20260420T105459.340303Z`, and match tracked PID `3037364`.
- No human-scope block remains from data access, split construction, metric calculation, or quick local baseline feasibility.

## Raw Artifact Links

- Dataset manifest: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-swe-primary-smoke-gate/dataset_manifest.json`
- HDF5 metadata: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-swe-primary-smoke-gate/hdf5_metadata.json`
- Split manifest: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-swe-primary-smoke-gate/split_manifest.json`
- Normalization stats: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-swe-primary-smoke-gate/normalization_stats.json`
- Model metrics/provenance: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-swe-primary-smoke-gate/runs/`
- Logs: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-swe-primary-smoke-gate/logs/`

## Carry-Forward Notes

- Longer execution should use a fresh output root or the same run-id/freshness guard; existence-only artifacts are not acceptable evidence.
- Preserve the grouped `*/data` layout handling in any full-run loader or dataset cache.
- Decide before longer execution whether the smoke-capped split manifest should be paired with a second full-ID manifest for paper provenance.
- Monitor disk pressure because `/` was already `95%` used after data materialization.

## Decision

Decision: proceed with longer SWE execution
