# Execution Report

## Completed In This Pass

- Registered `pinn_ffno_depth24` as a first-class append-only FFNO row in
  `scripts/studies/grid_lines_compare_wrapper.py`, preserving the corrected
  four-block `pinn_ffno` default while routing depth-24 through row-local
  overrides (`fno_blocks=24`, `fno_cnn_blocks=0`).
- Added focused wrapper tests proving:
  - `pinn_ffno` remains the four-block no-refiner baseline
  - `pinn_ffno_depth24` resolves to `FFNO-24 + PINN`
  - the depth-24 row carries the expected row-local override contract
- Passed the required deterministic gates:
  - presence gate
  - `pytest -q tests/torch/test_generator_registry.py -k "ffno"`
  - `pytest -q tests/torch/test_grid_lines_torch_runner.py -k "ffno"`
  - `pytest -q tests/test_grid_lines_compare_wrapper.py -k "ffno"`
  - `python -m compileall -q ptycho_torch scripts/studies`
- Launched exactly one tmux-owned item-local run under
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-06-cdi-lines128-ffno-depth24-ablation/runs/ffno_depth24_20260507T052301Z`
  and waited for the tracked shell to print `__EXIT_CODE__=0`.
- Verified all required row-local artifacts exist for
  `runs/pinn_ffno_depth24/` and `recons/pinn_ffno_depth24/recon.npz`.
- Wrote the post-run contract audit and two-row comparison payloads:
  - `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-06-cdi-lines128-ffno-depth24-ablation/verification/contract_audit.json`
  - `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-06-cdi-lines128-ffno-depth24-ablation/runs/ffno_depth24_20260507T052301Z/comparison_depth24_vs_depth4.json`
  - `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-06-cdi-lines128-ffno-depth24-ablation/runs/ffno_depth24_20260507T052301Z/comparison_depth24_vs_depth4.csv`
- Published the durable depth-ablation summary and updated the discovery
  surfaces:
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/cdi_lines128_ffno_depth24_ablation_summary.md`
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/evidence_matrix.md`
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/model_variant_index.json`
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/ablation_index.json`
  - `docs/studies/index.md`

## Completed Plan Tasks

- Task 1: Lock the baseline and preflight the item
- Task 2: Add minimal depth-24 row plumbing
- Task 3: Run only the fresh depth-24 row
- Task 4: Publish the append-only comparison and update discoverability

## Remaining Required Plan Tasks

- None.

## Verification

- Deterministic pre-run/closeout gates passed:
  - `pytest -q tests/torch/test_generator_registry.py -k "ffno"`
  - `pytest -q tests/torch/test_grid_lines_torch_runner.py -k "ffno"`
  - `pytest -q tests/test_grid_lines_compare_wrapper.py -k "ffno"`
  - `python -m compileall -q ptycho_torch scripts/studies`
- Archived verification logs:
  - `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-06-cdi-lines128-ffno-depth24-ablation/verification/input_presence_gate.log`
  - `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-06-cdi-lines128-ffno-depth24-ablation/verification/pytest_generator_registry_ffno.log`
  - `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-06-cdi-lines128-ffno-depth24-ablation/verification/pytest_grid_lines_torch_runner_ffno.log`
  - `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-06-cdi-lines128-ffno-depth24-ablation/verification/pytest_grid_lines_compare_wrapper_ffno.log`
  - `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-06-cdi-lines128-ffno-depth24-ablation/verification/compileall.log`
- Long-run completion proof:
  - tmux session printed `__EXIT_CODE__=0`
  - root invocation:
    `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-06-cdi-lines128-ffno-depth24-ablation/runs/ffno_depth24_20260507T052301Z/invocation.json`
  - row-local proofs:
    `runs/pinn_ffno_depth24/launcher_completion.json`,
    `runs/pinn_ffno_depth24/exit_code_proof.json`
- Contract comparison standard:
  - locked config/invocation fields: exact equality
  - train/test datasets: exact equality after dropping only
    `_metadata.creation_info.timestamp`
- Contract audit outcome:
  - baseline `pinn_ffno`: `fno_blocks=4`, `fno_cnn_blocks=0`
  - fresh `pinn_ffno_depth24`: `fno_blocks=24`, `fno_cnn_blocks=0`
  - no unexpected contract drift
- Side-by-side FFNO result summary:
  - baseline `pinn_ffno`:
    amp/phase MAE `0.082043 / 0.137965`, SSIM `0.890305 / 0.959644`,
    PSNR `67.8819 / 63.2910`, MS-SSIM `0.979843 / 0.720259`,
    FRC50 `61.4650 / 58.7286`, parameters `124,968`,
    runtime `873.742 s` train / `1.231 s` inference
  - depth24 `pinn_ffno_depth24`:
    amp/phase MAE `0.056506 / 0.121740`, SSIM `0.944487 / 0.974460`,
    PSNR `71.1549 / 64.4167`, MS-SSIM `0.992207 / 0.811215`,
    FRC50 `68.6174 / 65.6225`, parameters `701,628`,
    runtime `4,754.923 s` train / `8.505 s` inference
- Delta (`depth24 - depth4`):
  - MAE `-0.025537 / -0.016225`
  - MSE `-0.005606 / -0.006960`
  - PSNR `+3.2730 / +1.1258`
  - SSIM `+0.054182 / +0.014817`
  - MS-SSIM `+0.012364 / +0.090956`
  - FRC50 `+7.1524 / +6.8939`
  - parameters `+576,660`
  - train time `+3,881.181 s`
  - inference time `+7.275 s`

## Residual Risks

- This remains a single-seed CDI ablation with only `2 / 2` train/test images;
  it is appropriate for same-contract directionality, not broad statistical
  claims.
- The depth-24 result is append-only and unpromoted. A later explicit paper
  refresh must decide whether its metric gain justifies the much larger
  parameter/runtime cost.
- The supervised depth-24 companion row is still a separate backlog item, so
  the final paper-refresh decision does not yet have a matched depth-24
  objective-control pair.
