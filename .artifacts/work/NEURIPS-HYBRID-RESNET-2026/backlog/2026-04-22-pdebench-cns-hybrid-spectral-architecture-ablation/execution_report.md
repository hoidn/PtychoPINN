## Completed In This Pass

- Accepted the tracked Stage 4 finalist-confirmation run as complete after the launcher wrote exit code `0` and the run root contained the required metrics and compare outputs for `spectral_resnet_bottleneck_base` and `spectral_resnet_bottleneck_shared_blocks10`.
- Wrote the remaining Task 5 sidecars:
  - `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-hybrid-spectral-architecture-ablation/compare_finalists_1024cap_40ep_within_run.{json,csv}`
  - `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-hybrid-spectral-architecture-ablation/finalist_delta_1024cap.json`
- Published the durable repo-local summary and discoverability/state updates:
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_cns_hybrid_spectral_arch_ablation_summary.md`
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_2d_cfd_cns_summary.md`
  - `docs/studies/index.md`
  - `docs/index.md`
  - `state/NEURIPS-HYBRID-RESNET-2026/progress_ledger.json`
- Wrote the implementation-state bundle for this implementation phase.

## Completed Plan Tasks

- Task 5: selected the unique finalists from the fresh `40`-epoch sharing and shared-depth rankings, completed the larger-cap `1024 / 128 / 128` confirmation run, and published the within-run finalist compare plus the `512`-cap to `1024`-cap delta payload.
- Task 6: published the durable study summary, synced the broader CNS summary, updated docs discoverability, appended the progress-ledger completion entry, and reran the required deterministic checks against the final repo state.

## Remaining Required Plan Tasks

- None. The approved current-scope work for this backlog item is complete.

## Verification

- `pytest -q tests/studies/test_pdebench_image128_models.py tests/studies/test_pdebench_image128_runner.py` -> `65 passed in 42.32s`
  - log: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-hybrid-spectral-architecture-ablation/verification/final_pytest.log`
- `python -m compileall -q scripts/studies/pdebench_image128 scripts/studies/run_pdebench_image128_suite.py` -> exit `0`
  - log: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-hybrid-spectral-architecture-ablation/verification/final_compileall.log`
- Finalist launcher completion proof:
  - `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-hybrid-spectral-architecture-ablation/cns-hybrid-spectral-finalists-1024cap-40ep-20260428T054559Z.launch/exit_code.txt` contains `0`
  - the paired run root contains `comparison_summary.{json,csv}`, `metrics_spectral_resnet_bottleneck_base.json`, `metrics_spectral_resnet_bottleneck_shared_blocks10.json`, and both per-profile sample outputs

## Residual Risks

- This backlog item remains capped decision-support evidence only. It does not satisfy the PDEBench full-training benchmark gate.
- The larger-cap confirmation favors `spectral_resnet_bottleneck_base` on aggregate error, but `spectral_resnet_bottleneck_shared_blocks10` keeps a slight `fRMSE_mid/high` edge; the tradeoff may shift again under a full-training benchmark budget.
- The non-shared row’s `10`-epoch aggregate improvement did not persist at `40` epochs, so weight-sharing conclusions should stay limited to this capped lane.
