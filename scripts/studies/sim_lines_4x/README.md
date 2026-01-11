# SIM-LINES-4X (Nongrid Lines + TF Reconstruction)

This directory contains the SIM-LINES-4X scenario runners for generating a synthetic
"lines" object with a nongrid simulation, training a TensorFlow model, and running
inference on the contiguous test split.

## Scenarios

Four runner scripts map to the required scenarios:

- `run_gs1_ideal.py` - gridsize=1, idealized probe
- `run_gs1_integration_probe.py` - gridsize=1, integration-test probe
- `run_gs2_ideal.py` - gridsize=2, idealized probe
- `run_gs2_integration_probe.py` - gridsize=2, integration-test probe

## Locked Parameters

These scripts intentionally keep the core parameters fixed:

- `N=64`
- `object_size=392`
- `split_fraction=0.5`
- `base_total_images=2000` (gridsize=1)
- `group_count=1000` (per split)
- `nphotons=1e9`
- `neighbor_count=4`

For `gridsize=2`, total images scale by `gridsize^2` to keep the same number of
groups as the `gridsize=1` scenarios. Train and test splits are equal sized.

`--nepochs` can be passed to adjust training time.

## Usage

```bash
python scripts/studies/sim_lines_4x/run_gs1_ideal.py --output-root outputs/sim_lines_4x --nepochs 5
python scripts/studies/sim_lines_4x/run_gs1_integration_probe.py --output-root outputs/sim_lines_4x --nepochs 5
python scripts/studies/sim_lines_4x/run_gs2_ideal.py --output-root outputs/sim_lines_4x --nepochs 5
python scripts/studies/sim_lines_4x/run_gs2_integration_probe.py --output-root outputs/sim_lines_4x --nepochs 5
```

Optional gs2 scaling (e.g., 5x images and groups):

```bash
python scripts/studies/sim_lines_4x/run_gs2_ideal.py \
  --output-root outputs/sim_lines_4x \
  --nepochs 5 \
  --image-multiplier 5 \
  --group-multiplier 5
```

## Outputs

Each scenario writes to:

```
outputs/sim_lines_4x/<scenario>/
  train_outputs/
    wts.h5.zip
  inference_outputs/
    reconstructed_amplitude.png
    reconstructed_phase.png
  run.log
  run_metadata.json
```

The integration probe is sourced from:
`ptycho/datasets/Run1084_recon3_postPC_shrunk_3.npz`.
