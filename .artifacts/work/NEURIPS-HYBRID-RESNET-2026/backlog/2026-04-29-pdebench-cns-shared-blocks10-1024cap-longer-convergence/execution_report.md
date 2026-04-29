# Execution Report

## Completed In This Pass

- Resumed the existing backlog item from the current checkout and confirmed the tracked `80`-epoch shared-blocks10 CNS run finished cleanly with exit code `0` at:
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-shared-blocks10-1024cap-longer-convergence/cns-shared-blocks10-1024cap-80ep-20260429T025641Z`
- Generated the required Task 4 reporting payloads:
  - `convergence_audit.json/csv`
  - `shared_blocks10_1024cap_40ep_vs_80ep.json/csv`
- Wrote the durable summary, synced the CNS summary and docs index, added the completion record to `state/NEURIPS-HYBRID-RESNET-2026/progress_ledger.json`, and wrote the implementation-state bundle.

## Completed Plan Tasks

- Task 1: frozen `40`-epoch reference manifest and frozen shell contract were present, validated, and reused.
- Task 2: exact `80`-epoch inspect proof was present and remained valid for the finished run contract.
- Task 3: the fresh `80`-epoch `spectral_resnet_bottleneck_shared_blocks10` pilot run completed with tracked exit code `0` and emitted the required fresh metrics/model-profile/comparison artifacts.
- Task 4: convergence audit, shell-validated epoch-budget delta payload, durable summary sync, progress-ledger update, docs-index update, execution report, and implementation-state output are complete.

## Remaining Required Plan Tasks

- None for the approved current scope.
- A same-budget `80`-epoch rerun of `spectral_resnet_bottleneck_base` was not part of this plan. If a later backlog item wants a strict same-budget architecture verdict, that needs a new approved scope.

## Verification

- `pytest -q tests/studies/test_pdebench_image128_models.py tests/studies/test_pdebench_image128_runner.py`
  - log: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-shared-blocks10-1024cap-longer-convergence/verification/final_pytest.log`
  - result: `79 passed in 51.91s`
- `python -m compileall -q scripts/studies/pdebench_image128 scripts/studies/run_pdebench_image128_suite.py`
  - log: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-shared-blocks10-1024cap-longer-convergence/verification/final_compileall.log`
  - result: exit `0`
- Payload validation:
  - log: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-shared-blocks10-1024cap-longer-convergence/verification/payload_validation.log`
  - result: `payload validation passed`
- Run completion proof:
  - `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-shared-blocks10-1024cap-longer-convergence/cns-shared-blocks10-1024cap-80ep-20260429T025641Z.launch/exit_code.txt = 0`
  - fresh run root contains `comparison_summary.json`, `metrics_spectral_resnet_bottleneck_shared_blocks10.json`, `model_profile_spectral_resnet_bottleneck_shared_blocks10.json`, `dataset_manifest.json`, `split_manifest.json`, and `invocation.json`

## Residual Risks

- The fixed convergence audit still marks the fresh shared-blocks10 row as materially improving at `80` epochs (`late_window_ratio=0.790954`), so this capped stop point is not fully converged.
- The fresh row is a mixed-budget comparison against the frozen `40`-epoch references. It materially changes the bounded `1024cap` read, but it does not settle the same-budget architecture ranking because `spectral_resnet_bottleneck_base` was not rerun at `80` epochs.
- The result remains capped decision-support evidence only and does not satisfy the full-training benchmark gate for CNS.
