# CNS Full-Training Cost Estimate

## Purpose

Estimate the wall-clock cost of promoting the CNS headline table from the
current capped same-contract lane to a same-contract full-training benchmark
lane on one RTX 3090.

## Fixed Assumptions

- dataset:
  `/home/ollie/Documents/pdebench-data/2d_cfd_cns/2D_CFD_Rand_M1.0_Eta0.01_Zeta0.01_periodic_128_Train.hdf5`
- total trajectories: `10000`
- split policy from the current PDEBench image-suite code/tests:
  `8000 / 1000 / 1000`
- selected contract for scaling:
  `history_len=2`, `mse`, batch size `4`, `40` epochs
- raw windows per trajectory at `history_len=2`: `19`
- current capped lane:
  `512 / 64 / 64` trajectories with `max_windows_per_trajectory=8`
- current capped train windows: `4096`
- projected full-train windows: `8000 * 19 = 152000`

Derived scaling factor from capped train windows to full-train windows:

- `(8000 / 512) * (19 / 8) = 37.109375x`

This estimate assumes first-order runtime scales roughly with training windows.
It is directionally useful, not a promise. It likely understates real elapsed
time because longer validation/test passes, I/O, checkpointing, and failed-run
recovery are not included.

## Per-Row Runtime Projections

Observed `40`-epoch capped runtimes:

| Row | Observed capped runtime sec | Estimated full-training hours |
| --- | ---: | ---: |
| `spectral_resnet_bottleneck_base` | `1861.6252` | `19.19` |
| `hybrid_resnet_cns` | `886.2512` | `9.14` |
| `fno_base` | `1137.9494` | `11.73` |
| `unet_strong` | `1366.6472` | `14.09` |
| `author_ffno_cns_base` | `4725.5117` | `48.71` |

Projected aggregate wall time:

- four-row headline table if `spectral_resnet_bottleneck_base` is chosen as the
  sole Hybrid-family row:
  `~93.72` GPU-hours (`~3.90` days)
- five-row hedge if `hybrid_resnet_cns` is also rerun before freezing the best
  Hybrid-family row:
  `~102.85` GPU-hours (`~4.29` days)

## Full-Training Practical Read

### Time

On one RTX 3090, the full-training path is a multi-day sequential commitment
before accounting for:

- reruns due to training instability or artifact mismatch
- row-lock/table-generation work
- any additional authored-FFNO rerun needed if the contract changed to
  `history_len=3`

### Memory

The current capped `40`-epoch runs do not indicate an immediate memory blocker:

- `spectral_resnet_bottleneck_base` peak CUDA memory:
  `476279296` bytes
- `hybrid_resnet_cns` peak CUDA memory:
  `432515584` bytes
- `fno_base` peak CUDA memory:
  `419467264` bytes
- `unet_strong` peak CUDA memory:
  `419467264` bytes
- `author_ffno_cns_base` peak CUDA memory:
  `3410884096` bytes

The bottleneck is therefore schedule risk and queue opportunity cost, not an
obvious capped-run memory failure.

## Decision Implication

This estimate does not prove full training is impossible. It does show that,
under the current deadline and one-GPU budget, choosing
`full_training_paper_benchmark` now would commit the queue to a high-risk,
multi-day execution branch before the paper contract is even locked.

That supports the selected `bounded_capped_decision_support` contract for the
current pass.
