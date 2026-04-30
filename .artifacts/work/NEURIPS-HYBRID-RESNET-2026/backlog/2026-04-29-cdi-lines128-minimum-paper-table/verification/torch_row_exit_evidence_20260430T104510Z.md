# Torch Row Exit Evidence

- Root:
  `runs/minimum_subset_20260430T084339Z`
- Purpose:
  satisfy the implementation-review requirement for independent persisted
  Torch-row completion evidence, separate from the repaired row-local
  `invocation.json` / `exit_code_proof.json` pair

## Independent Persisted Evidence

- Wrapper-level completion:
  `runs/minimum_subset_20260430T084339Z/invocation.json:43-45` records the
  regenerated bundle pass as `status: completed`, `exit_code: 0`,
  `finished_at_utc: 2026-04-30T09:41:57.475512+00:00`.
- Torch row `pinn_hybrid_resnet`:
  `runs/minimum_subset_20260430T084339Z/launcher_stderr.log:73-78` records
  the training stop at `max_epochs=40`, the transition through inference and
  metric computation, and the persisted lines:
  `Saved artifacts to .../runs/pinn_hybrid_resnet` and
  `Torch runner complete. Artifacts in .../runs/pinn_hybrid_resnet`.
- Torch row `pinn_fno_vanilla`:
  `runs/minimum_subset_20260430T084339Z/launcher_stderr.log:130-135` records
  the same successful sequence and the persisted lines:
  `Saved artifacts to .../runs/pinn_fno_vanilla` and
  `Torch runner complete. Artifacts in .../runs/pinn_fno_vanilla`.

## Interpretation

- These launcher-stderr records are independent of the row-local
  `exit_code_proof.json` files that the implementation review flagged as
  circular under the old contract.
- This pass hardens future runs in code by requiring row-local proof emission
  to observe `invocation.exit_code == 0`.
- For the already-produced `084339Z` root, the launcher-stderr records above
  are the independent persisted Torch-row completion evidence used to retain
  the root as `paper_complete` without another expensive four-row rerun.
