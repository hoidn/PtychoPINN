## Completed In This Pass

- Added `test_cfd_cns_inspect_runner_writes_split_manifest` and fixed `scripts/studies/pdebench_image128/cfd_cns.py` so `inspect` mode writes `split_manifest.json` before returning.
- Reran the backlog-item deterministic checks and archived fresh preflight/final verification logs; the final targeted pytest gate now reports `67 passed`.
- Regenerated the inspect snapshot at `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-hybrid-spectral-architecture-ablation/inspect-20260428T075333Z`, updated `inspect_run_root.txt`, and verified that `hdf5_metadata.json`, `dataset_manifest.json`, and `split_manifest.json` all parse successfully.
- Resynchronized the durable summary and progress ledger so they now reference the repaired inspect artifact, the new test coverage, and the current verification counts.

## Completed Current-Scope Work

- The blocking inspect-contract defect is resolved: the fresh inspect artifact now satisfies the approved Task 1 contract instead of exiting before `split_manifest.json` was written.
- The medium-severity durable-state issue is resolved: the ledger and summary now reflect the repair-pass code/test surfaces and the current `67 passed` verification evidence.
- The current-scope backlog work is complete again after this repair pass. No additional approved-plan items remain open for this bounded architecture-ablation lane.

## Follow-Up Work

- Full-training CNS benchmark completeness remains outside this backlog item. The repaired inspect snapshot and capped pilots are still decision-support evidence only.
- If another backlog item adds an `inspect`-mode contract to a study runner, keep a direct runner test for the required manifests so future bookkeeping refactors cannot silently drop them.

## Residual Risks

- This backlog item still does not satisfy the PDEBench full-training benchmark gate.
- The larger-cap confirmation still favors `spectral_resnet_bottleneck_base` on aggregate error while `spectral_resnet_bottleneck_shared_blocks10` keeps a narrow `fRMSE_mid/high` edge; that tradeoff may move again under a full-training budget.
- The non-shared row’s `10`-epoch aggregate gain still disappears by `40` epochs, so any weight-sharing conclusion must stay limited to this capped lane.
