## Completed In This Pass

- finalized the finished paired `80`-epoch `2d_cfd_cns` spectral `12/12` versus `24/24` capped compare at `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-spectral-modes24-convergence-compare/cns-spectral-modes24-vs-base-1024cap-80ep`
- generated `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-spectral-modes24-convergence-compare/convergence_audit.{json,csv}` from the fresh run and applied the plan’s fixed convergence rule
- wrote the durable summary `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_spectral_modes24_convergence_summary.md`
- updated `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_2d_cfd_cns_summary.md`, `docs/studies/index.md`, and `docs/index.md` so the new capped result is discoverable with the correct claim boundary
- wrote the final implementation-state bundle at `state/NEURIPS-HYBRID-RESNET-2026/backlog_drain/iterations/20/items/2026-04-28-pdebench-cns-spectral-modes24-convergence-compare/implementation-phase/implementation_state.json`

## Completed Plan Tasks

- Task 1: preflight verification, exact-contract inspect gate, and repo-surface audit remained green; no code patch was required
- Task 2: resolved identical batch size remained `16` for both rows with no fallback, as recorded in `resolved_batch_size.json`
- Task 3: the fresh paired `80`-epoch long run completed with launcher exit code `0` and the required run-root artifacts present
- Task 4: the convergence audit was emitted successfully with `expected_loss_count=80` and the fixed late-window rule
- Task 5: the durable summary and discoverability updates were published
- Task 6: final verification and artifact validation were completed

## Remaining Required Plan Tasks

- none for the approved current scope of this backlog item

## Verification

- `pytest -q tests/studies/test_pdebench_image128_models.py tests/studies/test_pdebench_image128_runner.py` -> `77 passed`
- `python -m compileall -q scripts/studies/pdebench_image128 scripts/studies/run_pdebench_image128_suite.py` -> exit `0`
- final artifact validation confirmed the summary, resolved batch-size record, and convergence-audit outputs exist
- launcher completion evidence: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-spectral-modes24-convergence-compare/.launch/cns-spectral-modes24-vs-base-1024cap-80ep/exit_code` contains `0`

## Residual Risks

- both rows still satisfied the fixed material-improvement rule at `80` epochs (`late_window_ratio < 0.95` for both), so the capped result is inconclusive rather than a clean spectral-mode verdict
- the shared `12/12` base row finished ahead on every final eval metric at this stop point, but that does not justify a promotion claim while the `24/24` row is still improving
- this remains capped decision-support evidence only; it does not satisfy the full-training PDEBench benchmark gate
