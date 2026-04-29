# Active Work

- Auditing the fixed-contract stable root at
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-27-cdi-ffno-generator-lines-best-config/lines128_ffno_vs_hybrid_resnet`
  while the original wrapper process continues to own it.
- Preserving the no-duplicate-run decision from
  `docs/plans/NEURIPS-HYBRID-RESNET-2026/lines128_paper_benchmark_preflight.md`
  and publishing the current partial-completion state only.

# Current Status

- Active writer confirmed on `2026-04-29`:
  - tracked shell PID `831894`
  - live wrapper PID `831899`
- `pinn_hybrid_resnet` is complete inside the stable root:
  `runs/pinn_hybrid_resnet/metrics.json`, `history.json`, `model.pt`,
  `randomness_contract.json`, and `recons/pinn_hybrid_resnet/recon.npz`
  were written around `03:02` local time.
- The wrapper is still training the second row:
  `lightning_logs/version_1/metrics.csv` is fresh through epoch `18`, with the
  latest completed validation recorded at epoch `17`
  (`mae_val_loss=0.06248391419649124`).
- The stable root is therefore still incomplete at the wrapper level:
  there is no merged wrapper `metrics.json`, no wrapper tables/visual bundle,
  and no completed `runs/pinn_ffno/` tree yet.

# Next Resume Condition

- Resume after tracked PID `831894` exits with code `0` and the stable root has
  fresh wrapper-level `metrics.json`, table/visual artifacts, and a completed
  `runs/pinn_ffno/` result tree alongside the already-finished
  `pinn_hybrid_resnet` row.
