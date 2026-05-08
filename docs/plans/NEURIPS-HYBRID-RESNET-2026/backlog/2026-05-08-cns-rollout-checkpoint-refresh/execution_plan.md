# CNS Rollout Checkpoint Refresh Execution Plan

## Goal

Produce real trained checkpoints for the current manuscript-facing CNS
matched-condition rows needed for rollout GIFs: FNO, SRU-Net, and FFNO.

## Contract

- task: `2d_cfd_cns`
- profiles:
  - `fno_base`
  - `spectral_resnet_bottleneck_base`
  - `author_ffno_cns_base`
- `history_len=5`
- `512 / 64 / 64` train/validation/test trajectories
- `max_windows_per_trajectory=8`
- `epochs=40`
- `batch_size=4`
- loss: normalized-field MSE
- dataset:
  `/home/ollie/Documents/pdebench-data/2d_cfd_cns/2D_CFD_Rand_M1.0_Eta0.01_Zeta0.01_periodic_128_Train.hdf5`

## Steps

1. Verify the checkpoint-capable runner.
   - Confirm `scripts/studies/pdebench_image128/cfd_cns.py` writes
     `model_state_<row_id>.pt` and `model_state_<row_id>.json`.
   - Run the focused rollout and runner artifact tests.

2. Launch checkpoint-producing h5 reruns.
   - Use `tmux` and the `ptycho311` environment.
   - Prefer one row per run root to make failures and checkpoints easy to
     audit.
   - Use item-local artifact roots under
     `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-08-cns-rollout-checkpoint-refresh/runs/`.

3. Audit each rerun.
   - Check exit status and invocation contract.
   - Confirm checkpoint and normalization artifacts exist.
   - Compare metrics with the current h5 table authorities.
   - Record whether each rerun is metric-compatible or visualization-lineage
     only.

4. Generate rollout GIFs.
   - Use `render_cns_rollout_video.py` against the trained checkpoint roots.
   - Generate at least density rollout GIFs with the layout:
     `initial | true | prediction | absolute_error`.
   - Use a fixed test sample and record split/sample/start/step metadata.

5. Publish the summary and indexes.
   - Write `cns_rollout_checkpoint_refresh_summary.md`.
   - Update evidence/index JSON only if downstream tooling needs checkpoint
     discovery.

## Verification

```bash
pytest -q tests/studies/test_pdebench_image128_rollout_video.py
pytest -q tests/studies/test_pdebench_image128_runner.py -k "cfd_cns or model_state or matched_condition"
python -m compileall -q scripts/studies/pdebench_image128 scripts/studies/run_pdebench_image128_suite.py
```

For each produced GIF, verify:

- GIF exists and has the expected frame count;
- first frame is nonblank;
- adjacent JSON manifest names the correct checkpoint and row id.
