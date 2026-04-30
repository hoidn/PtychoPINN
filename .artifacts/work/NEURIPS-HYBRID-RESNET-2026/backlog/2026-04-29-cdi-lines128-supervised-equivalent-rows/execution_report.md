## Completed In This Pass

- added explicit supervised FFNO routing to the Lines128 Torch compare path,
  including the locked-dataset `Y_I` / `Y_phi` to `label_amp` / `label_phase`
  bridge and explicit paper labels for `FFNO + PINN` versus
  `FFNO + supervised`
- launched the required same-contract `supervised_ffno` row in tmux under the
  frozen `N=128`, `seed=3`, `40`-epoch custom-probe contract and completed the
  run successfully
- promoted the preserved `pinn_ffno` comparator into the new extension root,
  regenerated the adjacent pairwise bundle, repaired the wrapper tmux
  invocation metadata, and finalized the root as `paper_complete`
- verified exact supervised/PINN FFNO parity using a stronger-than-tolerance
  standard: SHA-256 identity for `recon.npz`, `history.json`, and `model.pt`,
  plus `numpy.allclose(..., rtol=0.0, atol=0.0)` on the reconstruction arrays

## Completed Plan Tasks

- Tranche 1: froze the extension authority, preserved prerequisite roots, and
  recorded the protocol-compatibility audit
- Tranche 2: closed the minimal supervised FFNO execution gap in the Torch
  runner, compare wrapper, dataset plumbing, and metrics-label surface
- Tranche 3: ran the required deterministic gate before the expensive launch
  and archived the pre-launch pytest / compile logs
- Tranche 4: completed the supervised FFNO tmux launch and validated the
  required row-local artifacts under
  `runs/supervised_ffno_extension_20260430T160218Z/runs/supervised_ffno`
- Tranche 5: built the adjacent extension bundle, paper benchmark manifest,
  parity audit, and durable summary / study-index entry
- Tranche 6: reran the focused selector, required deterministic gate, and
  `compileall` with fresh archived logs

## Remaining Required Plan Tasks

- None. Current-scope plan work is complete for this backlog item.

## Verification

- supervised FFNO tmux launch:
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-supervised-equivalent-rows/verification/supervised_ffno_launch_20260430T160218Z.log`
  - tracked launcher exited `0`
- extension bundle regeneration:
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-supervised-equivalent-rows/verification/supervised_ffno_bundle_20260430T162807Z.log`
  - final root status: `paper_complete`
- parity audit:
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-supervised-equivalent-rows/execution/supervised_ffno_parity_audit.json`
  - comparison standard: exact SHA-256 identity plus `numpy.allclose` with
    `rtol=0.0`, `atol=0.0`
- focused regression suite:
  `pytest -q tests/studies/test_lines128_paper_benchmark.py tests/studies/test_metrics_tables.py tests/test_grid_lines_compare_wrapper.py tests/torch/test_grid_lines_torch_runner.py`
  - `221 passed, 47 warnings in 42.13s`
  - log:
    `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-supervised-equivalent-rows/verification/pytest_focused_20260430T163023Z.log`
- required deterministic gate:
  `pytest -q tests/torch/test_grid_lines_hybrid_resnet_integration.py tests/torch/test_grid_lines_torch_runner.py tests/test_grid_lines_compare_wrapper.py`
  - `177 passed, 47 warnings in 301.19s (0:05:01)`
  - log:
    `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-supervised-equivalent-rows/verification/pytest_required_20260430T163124Z.log`
- compile gate:
  `python -m compileall -q ptycho_torch scripts/studies`
  - exit `0`
  - log:
    `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-supervised-equivalent-rows/verification/compileall_required_20260430T163124Z.log`

## Residual Risks

- the promoted `pinn_ffno` row inside the extension root remains recovered
  prerequisite evidence rather than a fresh rerun from this pass
- the extension is intentionally adjacent evidence and does not replace the
  preserved six-row primary CDI benchmark claim authority
- verification still emits the known non-fatal `tight_layout`, `skimage` SSIM,
  and FRC warnings already present on related Lines128 study surfaces
