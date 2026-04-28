## Completed In This Pass

- Added a regression test and reporting-helper support for multi-fresh cross-run compares so sharing tranches can include both fresh spectral rows without mislabeling one as a frozen reference.
- Regenerated `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-hybrid-spectral-architecture-ablation/compare_sharing_{10ep,40ep}_against_existing.{json,csv}` from `reference_runs_10ep.json` and `reference_runs_40ep.json`, and rewrote `compare_manifest_sharing_{10ep,40ep}.json` to preserve the frozen reference rows plus the correct fresh profile set.
- Removed the non-authoritative pointer files `stage1_sharing_40ep_run_root.txt` and `stage2_depth_40ep_run_root.txt`.
- Updated the durable study summaries to note that the repaired anchored sidecars now point at the frozen context manifests and that the interpretation did not change.

## Completed Current-Scope Work

- The blocking review item is resolved: the sharing compare sidecars now implement the approved "against existing" contract instead of substituting the same-run shared base row as a required reference.
- The current-scope backlog work is complete again after the repair. No additional plan items remain open for this bounded architecture-ablation lane.

## Follow-Up Work

- Full-training CNS benchmark completeness remains outside this backlog item. These repaired sharing sidecars are still capped decision-support evidence only.
- If another backlog item needs anchored compares with multiple fresh rows, reuse the new `write_cross_run_compare(..., fresh_profile_ids=[...])` path instead of hand-assembling manifests.

## Residual Risks

- This backlog item still does not satisfy the PDEBench full-training benchmark gate.
- The larger-cap confirmation still favors `spectral_resnet_bottleneck_base` on aggregate error while `spectral_resnet_bottleneck_shared_blocks10` keeps a narrow `fRMSE_mid/high` edge; that tradeoff may move again under a full-training budget.
- The non-shared row’s `10`-epoch aggregate gain still disappears by `40` epochs, so any weight-sharing conclusion must stay limited to this capped lane.
