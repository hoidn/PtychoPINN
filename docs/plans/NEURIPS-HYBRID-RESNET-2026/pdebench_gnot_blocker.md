# PDEBench GNOT CNS Compare Blocker

## Status

- Date: `2026-04-22`
- Initiative: `NEURIPS-HYBRID-RESNET-2026`
- Scope: execute the official GNOT compare plan on capped PDEBench `2d_cfd_cns`
- Result: `blocked before adapter implementation`

## Why It Blocked

The first execution gate in the plan was the external dependency/import path for
the official GNOT source. That gate failed in the current local environment.

Current environment:

- `torch==2.9.1+cu128`
- CUDA runtime reported by torch: `12.8`

Observed DGL outcomes:

1. `dgl==2.1.0` imported only after forcing an older `torchdata`, but then
   failed looking for a GraphBolt library keyed to this exact torch build:

   - missing `libgraphbolt_pytorch_2.9.1.so`

2. `dgl==1.1.3` imported successfully and the official GNOT module loaded, but
   it is CPU-only in this environment and cannot move graphs to CUDA:

   - `Device API cuda is not enabled. Please install the cuda version of dgl.`

So there is no working CUDA-enabled DGL path here for the current torch stack.

## Why CPU Fallback Was Rejected

I ran a representative CPU timing probe using the official GNOT model with a
regular-grid CNS-shaped sample:

- grid: `128x128` -> `16384` nodes per sample
- batch size `1`: forward `0.787s`, backward `0.394s`
- batch size `4`: forward `1.313s`, backward `1.582s`

That implies about `2.90s` per train step at batch size `4`.

Estimated training time from that probe:

- cap `512/64/64`, `8` windows/trajectory:
  - `1024` train steps per epoch
  - about `49.4` minutes per epoch
  - about `8.2` hours for `10` epochs
  - about `32.9` hours for `40` epochs

- cap `1024/128/128`, `8` windows/trajectory:
  - `2048` train steps per epoch
  - about `98.8` minutes per epoch
  - about `65.9` hours for `40` epochs

That is not a sensible execution path for the planned comparison matrix.

## Conclusion

The GNOT execution plan is blocked in this environment **before** adapter or
runner integration work:

- no usable CUDA-enabled DGL build is available for the active torch runtime
- CPU-only execution is too slow for the planned capped runs

The plan should stay blocked unless one of these changes:

1. a CUDA-enabled DGL build compatible with `torch 2.9.1+cu128` becomes
   available locally, or
2. the study scope is explicitly reduced to a much smaller CPU-only smoke.

## Evidence

- [gnot_environment_probe.json](/home/ollie/Documents/PtychoPINN/.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-gnot-cns-compare/gnot_environment_probe.json)
- [pdebench_gnot_cns_compare_plan.md](/home/ollie/Documents/PtychoPINN/docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_gnot_cns_compare_plan.md)
