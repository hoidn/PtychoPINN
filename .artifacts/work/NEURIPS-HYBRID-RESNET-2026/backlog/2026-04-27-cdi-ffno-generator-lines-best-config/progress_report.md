# Active Work

- Wrote `docs/plans/NEURIPS-HYBRID-RESNET-2026/lines128_paper_benchmark_preflight.md`
  to freeze the recovered `N=128` fixed contract, active-root audit, and
  no-duplicate-run resume decision.
- The fixed-contract `pinn_hybrid_resnet` versus `pinn_ffno` wrapper compare
  remains live under:
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-27-cdi-ffno-generator-lines-best-config/lines128_ffno_vs_hybrid_resnet`

# Current Status

- Active writer confirmed on `2026-04-29`:
  - tracked shell PID `831894`
  - live wrapper PID `831899`
- `tmux.log` shows training still in progress; the latest audited output was
  around epoch `25/40`, so the stable root is not wrapper-complete yet.
- The stable root currently contains partial runtime artifacts only:
  `invocation.json`, `invocation.sh`, `tmux.log`, shared dataset outputs,
  transient checkpoints, and Lightning logs.
- Required completion artifacts are not present yet:
  merged wrapper metrics/tables/visuals and completed row outputs under
  `runs/pinn_hybrid_resnet/` and `runs/pinn_ffno/`.

# Next Resume Condition

- Resume after tracked PID `831894` exits with code `0` and the stable root has
  fresh wrapper-level `metrics.json`, table/visual artifacts, and completed
  row-local outputs for both `pinn_hybrid_resnet` and `pinn_ffno`.
