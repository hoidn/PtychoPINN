# Execution Report

## Completed In This Pass

- Synced the durable Task 4 summary with the authoritative generated Task 2 inspect proof so the audit trail now cites the same `inspect-2048cap-20260428T232104Z/` root as this execution report.
- Reran the required backlog verification commands after the documentation/report repair and archived fresh logs under `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-hybrid-spectral-scaling-2048cap/verification/`.
- Repaired the review-blocking Task 2 preflight evidence by running the canonical `inspect` command into `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-hybrid-spectral-scaling-2048cap/inspect-2048cap-20260428T232104Z/`.
- Confirmed that the fresh inspect root contains the required generated runner artifacts: `invocation.json`, `dataset_manifest.json`, `split_manifest.json`, plus `hdf5_metadata.json` and `invocation.sh`.
- Verified from `split_manifest.json` that the inspect run staged the fixed `2048 / 256 / 256`, `history_len=2`, `max_windows_per_trajectory=8`, `batch_size=4`, `epochs=40` contract in `inspect` mode.
- Reconciled the existing launch sidecar with the written Task 3 artifact contract by adding `pid.txt -> python_pid.txt` and `train.log -> stdout.log` symlinks in `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-hybrid-spectral-scaling-2048cap/cns-hybrid-spectral-finalists-2048cap-40ep-20260428T201926Z.launch/`.
- Rewrote this execution report so it reflects the repaired state instead of the earlier incomplete Task 2 claim.

## Completed Current-Scope Work

- Task 1 remains complete: the archived pytest and compileall checks passed, and the frozen `512cap` / `1024cap` reference manifests are present under `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-hybrid-spectral-scaling-2048cap/`.
- Task 2 is now complete from generated artifacts rather than the superseded handwritten inspection summary:
  - inspect root: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-hybrid-spectral-scaling-2048cap/inspect-2048cap-20260428T232104Z/`
  - log: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-hybrid-spectral-scaling-2048cap/verification/inspect_2048cap_20260428T232104Z.log`
- Task 3 remains complete: the tracked pilot run exited `0`, wrote the required run-root artifacts for both finalist profiles, and the launch sidecar now exposes the plan-named `pid.txt`, `exit_code.txt`, and `train.log` paths without altering the underlying run outputs.
- Task 4 remains complete: the scaling JSON/CSV payload, durable summary, CNS summary sync, docs-index discoverability update, and progress-ledger update exist from the finished `2048cap` run, and the durable summary now cites the same generated inspect proof as the repaired Task 2 record.

## Follow-Up Work

- Same-protocol full-training CNS benchmark runs on the full available training split still remain outside this capped backlog item.
- Later suite-level interpretation across the broader PDEBench image-suite roadmap remains follow-up work after the full-training benchmark rows exist.

## Residual Risks

- All outcomes from this backlog item remain capped decision-support evidence only; this item still does not satisfy the roadmap full-training benchmark gate.
- The earlier handwritten `inspect-20260428T190521Z/inspection_summary.json` remains in the artifact tree as superseded context, but the authoritative Task 2 proof is now the generated inspect root at `inspect-2048cap-20260428T232104Z/`.
- The optional cross-run gallery remains blocked by saved-target mismatch across historical reference roots, but the required manifests, metrics, scaling payload, and durable summaries remain present and authoritative per `REPORTING-ARTIFACT-BOUNDARY-001`.
- `spectral_resnet_bottleneck_shared_blocks10` still retains narrower `fRMSE_mid/high` advantages, so manual-profile use may still be reasonable for that narrower objective even though the aggregate capped result does not justify promotion over the shared base row.

## Verification

- Fresh inspect repair: `python scripts/studies/run_pdebench_image128_suite.py --task 2d_cfd_cns --mode inspect --data-root /home/ollie/Documents/pdebench-data --output-root .artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-hybrid-spectral-scaling-2048cap/inspect-2048cap-20260428T232104Z --profiles spectral_resnet_bottleneck_base,spectral_resnet_bottleneck_shared_blocks10 --history-len 2 --epochs 40 --batch-size 4 --max-train-trajectories 2048 --max-val-trajectories 256 --max-test-trajectories 256 --max-windows-per-trajectory 8 --device cuda --num-workers 0` -> exit `0`, log archived at `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-hybrid-spectral-scaling-2048cap/verification/inspect_2048cap_20260428T232104Z.log`
- Generated inspect artifacts present: `invocation.json`, `dataset_manifest.json`, `split_manifest.json`, `hdf5_metadata.json`, `invocation.sh` under `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-hybrid-spectral-scaling-2048cap/inspect-2048cap-20260428T232104Z/`
- Split-contract check from `split_manifest.json`: `split_counts={"train": 2048, "val": 256, "test": 256}`, `history_len=2`, `max_windows_per_trajectory=8`, `run_mode="inspect"`
- Review-fix rerun checks:
  - `pytest -q tests/studies/test_pdebench_image128_models.py tests/studies/test_pdebench_image128_runner.py` -> `72 passed in 49.07s`, log archived at `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-hybrid-spectral-scaling-2048cap/verification/review_fix_pytest.log`
  - `python -m compileall -q scripts/studies/pdebench_image128 scripts/studies/run_pdebench_image128_suite.py` -> exit `0`, log archived at `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-hybrid-spectral-scaling-2048cap/verification/review_fix_compileall.log`
  - summary-sync assertion for the durable inspect-proof reference -> `summary_sync_ok`, log archived at `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-hybrid-spectral-scaling-2048cap/verification/review_fix_summary_sync.log`
- Earlier required backlog checks also remain green per `.artifacts/checks/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-28-pdebench-cns-hybrid-spectral-scaling-2048cap-checks.json`.
