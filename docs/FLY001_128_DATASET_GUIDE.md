# FLY001 N=128 Dataset Guide

This guide defines the reproducible preparation path for the `N=128` fly001 external-study dataset.

## Source Dataset

Raw source path:
- `~/Documents/128_res/fly001_128_train.npz`

Observed raw format:
- per-scan diffraction key: `diff3d`
- diffraction dtype: `uint16`
- shape: `(10304, 128, 128)`
- global arrays: `objectGuess`, `probeGuess`
- coordinate arrays: `xcoords`, `ycoords`, `xcoords_start`, `ycoords_start`
- `scan_index` may be non-unique (do not use as the primary provenance key)

## Canonical + Split Preparation

Use the deterministic prep script:

```bash
python scripts/studies/prepare_fly001_128_external_split.py \
  --input-npz ~/Documents/128_res/fly001_128_train.npz \
  --output-dir datasets/fly001_128
```

This writes:
- `datasets/fly001_128/fly001_128_train_converted.npz`
- `datasets/fly001_128/fly001_128_top_half_converted.npz`
- `datasets/fly001_128/fly001_128_full_test_converted.npz`
- `datasets/fly001_128/manifest.json`

## Split Policy

The train split uses `ycoords` midpoint threshold:
- train (`top_half`): `ycoords >= threshold`
- test (`full_test`): full canonical dataset (no spatial filtering)

This enforces spatial restriction only on training data while preserving complete-object evaluation.

## Manifest Provenance

`datasets/fly001_128/manifest.json` records:
- `source_file`
- `source_sha256`
- `canonical_npz`, `train_npz`, `test_npz`
- `split_axis`, `split_threshold`
- `n_total`, `n_train`, `n_test`

Use this manifest as the canonical provenance record for study documentation and index entries.

## Study Runbook

Runbook for the external `N=128` grid-lines comparison:
- `scripts/studies/runbooks/grid_lines_external_fly001_n128_top_train_full_test_e40.sh`

Study index entry:
- `docs/studies/index.md` → `grid-lines-external-fly001-n128-top-train-full-test-e40`
