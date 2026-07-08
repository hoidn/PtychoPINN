# SRU-Net

SRU-Net is the paper-facing name for the Hybrid ResNet family used in the
NeurIPS benchmark work. In the codebase, the main CDI implementation is
registered as `hybrid_resnet`; `sru_net` is only used as a visible row label in
some study adapters.

This page is a usage map. For architecture details, see the PyTorch generator
README and the study summaries linked below.

## Name Mapping

| Surface | Name |
|---|---|
| Paper label | `SRU-Net` |
| CDI architecture id | `hybrid_resnet` |
| CDI wrapper row id | `pinn_hybrid_resnet` |
| CDI supervised control row | `supervised_hybrid_resnet` |
| PDEBench CNS paper-facing SRU-Net row | `spectral_resnet_bottleneck_base` |
| BRDT visible row id | `sru_net` |
| BRDT internal model body | `hybrid_resnet` |

Do not use `--architecture srunet`; it is not a registered architecture.

## CDI Grid-Lines Usage

Use the comparison wrapper when generating full CDI benchmark rows:

```bash
python scripts/studies/grid_lines_compare_wrapper.py \
  --N 128 \
  --gridsize 1 \
  --models pinn_hybrid_resnet \
  --output-dir .artifacts/work/manual/srunet_$(date -u +%Y%m%dT%H%M%SZ) \
  --seed 3 \
  --nimgs-train 2 \
  --nimgs-test 2 \
  --nphotons 1e9 \
  --set-phi \
  --probe-npz datasets/Run1084_recon3_postPC_shrunk_3.npz \
  --probe-source custom \
  --probe-scale-mode pad_extrapolate \
  --probe-smoothing-sigma 0.5 \
  --torch-epochs 40 \
  --torch-batch-size 16 \
  --torch-learning-rate 2e-4 \
  --torch-scheduler ReduceLROnPlateau \
  --torch-plateau-factor 0.5 \
  --torch-plateau-patience 2 \
  --torch-plateau-min-lr 1e-4 \
  --torch-plateau-threshold 0.0 \
  --torch-loss-mode mae \
  --torch-output-mode real_imag \
  --fno-modes 12 \
  --fno-width 32 \
  --fno-blocks 4 \
  --fno-cnn-blocks 2
```

The wrapper writes row artifacts under `runs/pinn_hybrid_resnet/` and records
the run contract in the output manifest files.

## Direct Torch Runner Usage

Use the direct Torch runner only when train and test NPZs already exist:

```bash
python scripts/studies/grid_lines_torch_runner.py \
  --train-npz path/to/train.npz \
  --test-npz path/to/test.npz \
  --output-dir .artifacts/work/manual/srunet_direct \
  --architecture hybrid_resnet \
  --training-procedure pinn \
  --seed 3 \
  --epochs 40 \
  --batch-size 16 \
  --learning-rate 2e-4 \
  --output-mode real_imag \
  --torch-loss-mode mae
```

Use `--training-procedure supervised` for the objective-control row. If you
need the wrapper-visible row id to remain `supervised_hybrid_resnet`, prefer
the comparison wrapper or pass the direct runner's row override flags.

## Variant Rows

The CDI wrapper exposes append-only SRU-Net variants through model ids:

| Model id | Purpose |
|---|---|
| `pinn_hybrid_resnet` | Main CDI SRU-Net + PINN row |
| `supervised_hybrid_resnet` | Same body with supervised objective |
| `pinn_hybrid_resnet_encoder_conv_only` | Encoder branch ablation |
| `pinn_hybrid_resnet_encoder_spectral_only` | Encoder branch ablation |
| `pinn_hybrid_resnet_convnext_bottleneck` | Bottleneck-family ablation |
| `pinn_hybrid_resnet_ffno_ptychoblock_encoder` | Encoder mechanism probe |
| `pinn_hybrid_resnet_ptychoblock_ffno_encoder` | Encoder-order probe |

Most variants are decision-support only. Check the evidence matrix before
treating a row as paper-facing.

## PDEBench CNS

For PDEBench CNS, the paper-facing SRU-Net row is not the CDI
`pinn_hybrid_resnet` row. Current CNS SRU-Net evidence generally routes through
the PDEBench image-suite profiles, especially `spectral_resnet_bottleneck_base`.

Use the PDEBench run configuration and evidence docs rather than the CDI
grid-lines wrapper for CNS claims.

## BRDT

In BRDT studies, `sru_net` is a visible row id backed by the internal
`hybrid_resnet` body. The row label and architecture id are deliberately
distinct so manuscript naming does not leak into the model registry.

## Where To Look

- `ptycho_torch/generators/README.md` for architecture-level generator
  documentation.
- `scripts/studies/grid_lines_compare_wrapper.py` for CDI model ids and wrapper
  routing.
- `scripts/studies/grid_lines_torch_runner.py` for direct Torch runner
  architecture ids.
- `scripts/studies/born_rytov_dt/run_config.py` for BRDT `sru_net` row-label
  mapping.
- `docs/plans/NEURIPS-HYBRID-RESNET-2026/evidence_matrix.md` for completed
  results and claim boundaries.
- `docs/plans/NEURIPS-HYBRID-RESNET-2026/model_variant_index.json` for
  machine-readable row provenance.
