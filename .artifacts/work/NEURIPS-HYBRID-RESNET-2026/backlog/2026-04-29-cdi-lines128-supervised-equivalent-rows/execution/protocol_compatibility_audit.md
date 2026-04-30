# Lines128 Supervised-Equivalent Protocol Compatibility Audit

- Date: `2026-04-30`
- Backlog item: `2026-04-29-cdi-lines128-supervised-equivalent-rows`
- Audit status: `executed_same_contract_comparison`

## Contract Check

- the locked dataset/split/probe/sample/metric contract remains the frozen
  `lines128` `N=128`, `gridsize=1`, `set_phi=True`, custom Run1084 probe,
  `pad_extrapolate`, fixed `seed=3`, `40`-epoch MAE Torch recipe already used
  by the authoritative minimum-subset and complete-table roots
- the new row id remains `supervised_ffno` with paper label
  `FFNO + supervised`
- the reused same-architecture comparator remains `pinn_ffno` from the
  authoritative complete-table root
- the reused supervised CNN reference remains `baseline` from the authoritative
  minimum-subset root

## Reviewed Failure Mode

The delivered extension was not protocol-correct because the claimed
`supervised_ffno` row did not actually execute the intended supervised FFNO
path, and the repaired current-root bundle recovery still under-described the
rerun.

- `Ptycho_Supervised` always instantiated the legacy `Autoencoder`, so the
  claimed `supervised_ffno` row could not actually bind to the configured FFNO
  generator even when the launcher requested `architecture: ffno`
- `PtychoPINN_Lightning(... mode='Supervised' ...)` did not pass the selected
  generator module or generator output contract into `Ptycho_Supervised`
- after the corrected direct rerun, the compare-wrapper recovery path still
  treated the current-root row like recovered evidence and omitted row-local
  `stdout.log`, `stderr.log`, and `exit_code_proof.json`, which kept the
  adjacent bundle at `decision_support`

## Narrow Fix Applied

- updated `ptycho_torch/model.py` so `Ptycho_Supervised` accepts an injected
  generator plus output contract and uses the same complex-prediction helper as
  the PINN path
- updated `PtychoPINN_Lightning` to pass the selected generator module and
  generator output mode into supervised runs, and to rebuild generator-backed
  modules from saved config state during checkpoint-only reload
- updated `scripts/studies/grid_lines_compare_wrapper.py` to:
  - bootstrap repo-root imports for direct script execution
  - mark current-root Torch rows as freshly rebuilt instead of recovered
  - materialize missing row-local logs during bundle repair so
    `exit_code_proof.json` and `paper_grade` promotion can be emitted
- refreshed the authoritative extension root's canonical `last.ckpt` from the
  valid rerun `last-v1.ckpt` after verifying the review-cited reload path still
  pointed at a stale pre-fix checkpoint
- added regressions covering:
  - supervised FFNO generator wiring in Lightning
  - generator-backed checkpoint round-trips across the registered Lightning
    architectures
  - fresh current-root Torch-row recovery semantics
  - recovered row-log enrichment to `paper_grade`
  - direct compare-wrapper script import bootstrap

## Verification Status

- targeted review regressions are green:
  - `tests/torch/test_generator_registry.py::test_ffno_generator_builds_supervised_lightning_model`
  - `tests/test_grid_lines_compare_wrapper.py::test_recover_torch_row_payload_marks_current_root_rows_as_fresh`
  - `tests/test_grid_lines_compare_wrapper.py::test_enrich_paper_row_payload_recovers_missing_direct_runner_logs`
  - `tests/test_grid_lines_compare_wrapper.py::test_compare_wrapper_script_path_bootstraps_repo_imports`
  - archived log:
    `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-supervised-equivalent-rows/verification/pytest_targeted_20260430_supervised_equivalent_rows.log`
- required deterministic gate is green:
  - archived log:
    `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-supervised-equivalent-rows/verification/pytest_20260430_supervised_equivalent_rows_checkpoint_fix.log`
- compile gate is green:
  - archived log:
    `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-supervised-equivalent-rows/verification/compileall_20260430_supervised_equivalent_rows_checkpoint_fix.log`
- checkpoint reload contract is green on both synthetic and real artifacts:
  - synthetic regression log:
    `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-supervised-equivalent-rows/verification/pytest_checkpoint_reload_20260430.log`
  - real reviewed artifact proof:
    `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-supervised-equivalent-rows/verification/checkpoint_reload_real_artifact_20260430.log`
- repo integration marker is green:
  - archived log:
    `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-supervised-equivalent-rows/verification/pytest_integration_20260430_supervised_equivalent_rows_checkpoint_fix.log`
- supervised launch completed successfully under the frozen contract:
  - archived log:
    `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-supervised-equivalent-rows/verification/supervised_ffno_launch_20260430T180217Z.log`
  - corrected run evidence:
    `lightning_logs/version_0/hparams.yaml` in the authoritative extension
    root records `mode: Supervised`, `architecture: ffno`, and
    `generator_output: real_imag`
- the rebuilt adjacent extension bundle is `paper_complete`:
  - authoritative root:
    `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-supervised-equivalent-rows/runs/supervised_ffno_extension_20260430T180217Z`
  - bundle regeneration log:
    `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-supervised-equivalent-rows/verification/supervised_ffno_bundle_20260430T180217Z.log`
- the corrected comparison audit is archived at:
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-supervised-equivalent-rows/execution/supervised_ffno_parity_audit.json`

## Final Outcome

`executed_same_contract_comparison`

The supervised FFNO row now runs truthfully under the frozen `lines128`
contract, the adjacent extension root validates as `paper_complete`, and the
rebuilt comparison audit shows the corrected supervised row is not identical to
the preserved `FFNO + PINN` comparator.
