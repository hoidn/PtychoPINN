# `lines_256` Dataset Note

## Purpose

`lines_256` is the repo-local name for the N=256 lines/structured NPZ pair used for PyTorch architecture experiments.

Use this dataset when you want a single fixed `N=256` lines benchmark without switching to the dual-profile canonical promotion flow.

## Dataset Identity

- Dataset alias: `lines_256`
- Runbook profile name: `custom_npz_pair_n256`
- Current compatibility train NPZ path: `outputs/lines_256_arch_improvement/datasets/N256/gs1/train.npz`
- Current compatibility test NPZ path: `outputs/lines_256_arch_improvement/datasets/N256/gs1/test.npz`

This pair is the `lines_256` working dataset for the architecture-improvement loop. The authoritative pair is now the regenerated `set_phi=True` version of this compatibility dataset. It supersedes both:

- the older archived `custom_npz_pair_n256` pair
- the earlier zero-phase `lines_256` compatibility pair where `Y_phi` existed but was all zeros

The zero-phase pair is historical only across this boundary. Do not compare new `set_phi=True` runs against baselines or accepted states recorded before the regeneration.

Important:

- These `outputs/...` paths are the current workflow compatibility location, not the preferred long-term storage convention.
- Datasets that must persist across cleanup should live under a durable git-ignored dataset location, not under `outputs/`.
- If this pair remains a pinned long-term input, it should be promoted out of `outputs/` and consumers should be updated intentionally.

Important:

- These `outputs/...` paths are the current workflow compatibility location, not the preferred long-term storage convention.
- Datasets that must persist across cleanup should live under a durable git-ignored dataset location, not under `outputs/`.
- If this pair remains a pinned long-term input, it should be promoted out of `outputs/` and consumers should be updated intentionally.

## Provenance

- The older hybrid-resnet search design used an archived `custom_npz_pair_n256` pair under `outputs/hybrid_resnet_structural_rerun_20260226T110719Z/...`.
- The current `lines_256` pair was regenerated on March 13, 2026 under `outputs/lines_256_arch_improvement/` with the same simulation settings but corrected custom-probe padding semantics.
- The authoritative pair is regenerated through `scripts/studies/runbooks/rebuild_lines_256_dataset.py` with the same simulation settings, corrected custom-probe padding semantics, and `set_phi=True`.
- The regenerated pair carries metadata showing:
  - `N=256`
  - `gridsize=1`
  - `nphotons=1e9`
  - `nimgs_train=2`
  - `nimgs_test=1`
  - `size=392`
  - `offset=4`
  - `outer_offset_train=8`
  - `outer_offset_test=20`
  - `probe_source=custom`
  - `probe_scale_mode=pad_preserve`
  - `probe_smoothing_sigma=0.5`
  - `set_phi=True`
  - `coords_type=relative`
  - `probe_npz=datasets/Run1084_recon3_postPC_shrunk_3.npz`
- Custom-probe padding rule for this regenerated pair:
  - use `probe_scale_mode=pad_preserve`
  - smooth the 64x64 source probe at `sigma=0.5`
  - center-pad the complex probe into the 256x256 canvas
  - do not rescale or extrapolate the phase as `N` grows
- This keeps `phase(n - center_N)` fixed across padded `N` values, so both the amplitude view and the phase view become relatively smaller as probe support becomes smaller relative to `N`.

## How To Use It

Preferred path for a single-model experiment:

```bash
python scripts/studies/run_lines_256_arch_experiment.py \
  --output-dir <output_dir> \
  --fno-modes 12 \
  --fno-width 32 \
  --fno-blocks 4 \
  --no-hybrid-skip-connections \
  --hybrid-downsample-steps 2 \
  --hybrid-downsample-op stride_conv \
  --hybrid-resnet-blocks 6 \
  --hybrid-skip-style add
```

Runbook path when you need the staged sweep/orchestration machinery:

```bash
python scripts/studies/runbooks/run_hybrid_resnet_mode_skip_sweep.py \
  --ns 256 \
  --dataset-profiles-n256 custom_npz_pair_n256 \
  --custom-n256-train-npz outputs/lines_256_arch_improvement/datasets/N256/gs1/train.npz \
  --custom-n256-test-npz outputs/lines_256_arch_improvement/datasets/N256/gs1/test.npz \
  --allow-n256-direct-diagnostic \
  --top-k-n256 0 \
  ...
```

Important:

- For canonical non-diagnostic `N=256` promotion runs, the runbook requires both `cameraman256_halfsplit_v1` and `custom_npz_pair_n256`.
- For a lines-only architecture experiment, use `scripts/studies/run_lines_256_arch_experiment.py` so the fixed dataset and epoch budget stay consistent across experiments.
- Because `outputs/` is cleanup-prone, do not assume these compatibility paths are durable archival storage.
- The wrapper pins `--train-npz`, `--test-npz`, `--seed 3`, `--epochs 20`, `--N 256`, `--gridsize 1`, `--architecture hybrid_resnet`, `--no-probe-mask`, and `--torch-mae-pred-l2-match-target`.
- The wrapper also pins `--scheduler ReduceLROnPlateau` and `--plateau-min-lr 0.0002`.
- Use the direct Torch runner only when you explicitly need to bypass the wrapper contract.
- If you need to rebuild the pair, use `python scripts/studies/runbooks/rebuild_lines_256_dataset.py` so the canonical `set_phi=True` contract is reproduced exactly.
- Rebuilding the pair with materially different simulation semantics, including `set_phi=False -> True`, invalidates prior `lines_256` baselines and accepted-state results.

## Metric Guidance

For the simplified architecture-improvement agent prompt, the optimization metric is `amp_ssim`.

- Treat `amp_ssim` as the only keep/discard metric.
- Supporting metrics may be logged, but they do not override a higher valid `amp_ssim`.

## References

- `docs/plans/2026-02-21-hybrid-resnet-skip-mode-search-design.md`
- `docs/plans/2026-02-21-hybrid-resnet-skip-mode-search-stage-a-execution.md`
- `docs/plans/2026-02-21-hybrid-resnet-skip-mode-search-stage-c-execution.md`
- `scripts/studies/runbooks/run_hybrid_resnet_mode_skip_sweep.py`
- `scripts/studies/grid_lines_torch_runner.py`
- `scripts/studies/run_lines_256_arch_experiment.py`
