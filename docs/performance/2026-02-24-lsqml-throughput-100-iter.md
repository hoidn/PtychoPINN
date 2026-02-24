# 2026-02-24 LSQML Throughput (100-Iteration Convergence Assumption)

- Dataset: `/home/ollie/Documents/128_res/fly001_128_train.npz`
- Frame resolution: `128 x 128`
- Total frames: `10304`
- Reconstructor: `pty-chi` `LSQML` (LS-MLE), `batch_size=96`, GPU (`cuda`)
- Measured steady epoch time: `1.4439925957537656 s/epoch`

## Fully Converged Throughput (Assume 100 Iterations)

Computation:

`throughput_fps = n_frames / (n_iters * epoch_time_s)`

`throughput_fps = 10304 / (100 * 1.4439925957537656) = 71.36 frames/s`

## Result

**LSQML fully converged throughput (100-iteration assumption): `71.36 frames/s`**
