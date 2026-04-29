---
priority: 18
plan_path: docs/plans/NEURIPS-HYBRID-RESNET-2026/paper_evidence_package_design.md
check_commands:
  - pytest -q tests/studies/test_pdebench_image128_runner.py tests/studies/test_pdebench_cfd_cns_data.py tests/studies/test_pdebench_cfd_cns_metrics.py
  - python -m compileall -q scripts/studies/pdebench_image128 scripts/studies/run_pdebench_image128_suite.py
prerequisites:
  - 2026-04-29-cns-paper-contract-decision
related_roadmap_phases:
  - phase-2-pdebench-128x128-image-suite
signals_for_selection:
  - After the CNS paper contract is fixed, the paper needs same-contract Hybrid-family, FNO, and U-Net/CNN-style core rows, plus an authored-FFNO row if it is available by the predeclared cutoff.
  - This item turns decision-support comparisons into a locked CNS evidence set without broadening the architecture search.
---

# Backlog Item: Lock CNS Paper Benchmark Rows

## Objective

- Produce or lock the CNS rows required by the selected paper evidence
  contract: best Hybrid/Hybrid-spectral, local FNO, U-Net/CNN-style local
  baseline, and authored FFNO if available by the predeclared cutoff.

Selected contract note:

- `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_cns_paper_contract_decision.md`
  now fixes the headline CNS lane and must be treated as the only contract
  authority for this item.
- Selected contract: `bounded_capped_decision_support`
- Selected history lane: `history_len=2`, `40` epochs, `512 / 64 / 64` trajectories, `max_windows_per_trajectory=8`, emitted windows `4096 / 512 / 512`
- Selected normalization contract: train-only per-field normalization fit on the `512` training trajectories, reused across all history slots and target channels, with evaluation reported in denormalized target space.
- Selected training recipe contract: keep the CNS task-local `mse` override relative to the design's generic `mae` baseline; use `Adam` with learning rate `2e-4`; use `ReduceLROnPlateau` with factor `0.5`, patience `2`, threshold `0.0`, and `min_lr=1e-5`; keep batch size `4`; keep the metric family `err_RMSE`, `err_nRMSE`, `relative_l2`, `fRMSE_low`, `fRMSE_mid`, `fRMSE_high`.
- Under that decision, the locked headline row roster is:
  `spectral_resnet_bottleneck_base`, `fno_base`, `unet_strong`,
  `author_ffno_cns_base`.
- Authored FFNO cutoff outcome: use only the accepted completed
  `author_ffno_cns_base` `history_len=2`, `40`-epoch row; do not wait for or
  imply any `history_len=3` authored rerun.
- `hybrid_resnet_cns` remains an audited continuity/support row rather than a
  required headline-table row in this pass.

## Scope

- Consume the CNS paper contract decision as the only authority for full
  training versus bounded capped evidence.
- Reuse the exact selected history lane, normalization contract, and training
  recipe contract above verbatim when writing row manifests and summaries.
- For `full_training_paper_benchmark`, run the required rows under the same
  official file, split, history length, normalization, loss, scheduler,
  training budget, and metric schema.
- For `bounded_capped_decision_support`, freeze audited row manifests for the
  required models and run only missing same-contract rows needed to make the
  bounded table coherent.
- For the selected bounded capped contract, do not reopen the stronger but
  incomplete `history_len=3` lane and do not wait for a new authored-FFNO rerun
  under a different temporal contract.
- For authored FFNO, either include the same-contract row by the predeclared
  cutoff or record a row-level `blocked` / `not_protocol_compatible` status and
  the resulting CNS claim limitation.
- Emit per-row provenance with invocation/config/git/environment/data/split
  fields, parameter counts, runtime, metrics, source prediction arrays, logs,
  and exit-code proof.
- Write a durable CNS row-lock summary under
  `docs/plans/NEURIPS-HYBRID-RESNET-2026/` that labels each row
  `full_training`, `capped_decision_support`, `blocked`, or
  `not_protocol_compatible`.

## Notes for Reviewer

- Do not mix rows from incompatible caps, histories, split counts, or metric
  schemas unless the incompatibility is an explicit table column and claim
  boundary.
- Do not use the local FFNO-close proxy as the authored FFNO row.
- If the authored FFNO or GNOT path is blocked, record the environment or
  protocol blocker and keep the headline table honest.
