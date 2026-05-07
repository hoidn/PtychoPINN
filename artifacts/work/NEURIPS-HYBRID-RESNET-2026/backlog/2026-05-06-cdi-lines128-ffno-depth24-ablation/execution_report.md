# Execution Report

## Completed In This Pass

- Addressed the `REVISE` blocker raised in the implementation review
  (`artifacts/review/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-06-cdi-lines128-ffno-depth24-ablation-implementation-review.md`):
  the authoritative top-level provenance now unambiguously reflects the
  delivered depth-24 contract and the published claim boundary.
- Regenerated the root model manifest at
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-06-cdi-lines128-ffno-depth24-ablation/runs/ffno_depth24_20260507T052301Z/model_manifest.json`:
  - `claim_boundary` updated from `decision_support_append_only` to
    `cdi_ffno_depth_ablation_only`
  - row-level `row_status` for `pinn_ffno_depth24` updated to the same
    delivered claim boundary
  - added a `claim_boundary_correction` block recording the previous label,
    the delivered label, the rationale, and the application timestamp
- Annotated the root wrapper invocation at
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-06-cdi-lines128-ffno-depth24-ablation/runs/ffno_depth24_20260507T052301Z/invocation.json`
  without falsifying the user-supplied `argv`. Added under `extra`:
  - `delivered_contract` (model_id, fno_blocks=24, fno_cnn_blocks=0,
    claim_boundary=`cdi_ffno_depth_ablation_only`, row-local truth pointer)
  - `effective_row_spec_overrides` showing the wrapper applied
    `pinn_ffno_depth24 -> {fno_blocks: 24, fno_cnn_blocks: 0}`
  - `effective_manifest_claim_boundary = cdi_ffno_depth_ablation_only`
  - `provenance_correction` describing the issue, the fix, and the timestamp
- Updated the contract audit at
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-06-cdi-lines128-ffno-depth24-ablation/verification/contract_audit.json`
  so `allowed_drift.wrapper_invocation.manifest_claim_boundary` records both
  the original wrapper argv label and the delivered claim boundary, plus
  `delivered_row_spec_overrides` and `delivered_claim_boundary` fields.
- Re-verified by direct file inspection that the row-local invocation at
  `runs/pinn_ffno_depth24/invocation.json` already records
  `--fno-blocks 24` and `--fno-cnn-blocks 0`, confirming that the actual
  training was launched against the depth-24 contract; only the wrapper-level
  argv had captured the user-supplied flags before the row-spec overrides
  were applied. No retraining was required.

## Completed Current-Scope Work

- Task 1 (lock baseline and preflight the item): complete; presence gate
  passed in the prior pass and is unaffected by these provenance edits.
- Task 2 (add minimal depth-24 row plumbing): complete; deterministic gates
  remain green in the prior pass.
- Task 3 (run only the fresh depth-24 row): complete; the long run finished
  with `__EXIT_CODE__=0`, the row-local artifacts exist, and the contract
  audit (now expanded) still PASSes with the only allowed delta being
  `fno_blocks: 4 -> 24`.
- Task 4 (publish the append-only comparison and update discoverability):
  complete; the durable summary, evidence matrix, model variant index,
  ablation index, and `docs/studies/index.md` already point at the
  `cdi_ffno_depth_ablation_only` claim boundary, and the corrected
  manifest now matches them.

## Verification

- Deterministic pre-run/closeout gates passed in the prior pass:
  - `pytest -q tests/torch/test_generator_registry.py -k "ffno"`
  - `pytest -q tests/torch/test_grid_lines_torch_runner.py -k "ffno"`
  - `pytest -q tests/test_grid_lines_compare_wrapper.py -k "ffno"`
  - `python -m compileall -q ptycho_torch scripts/studies`
- Archived verification logs unchanged:
  - `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-06-cdi-lines128-ffno-depth24-ablation/verification/input_presence_gate.log`
  - `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-06-cdi-lines128-ffno-depth24-ablation/verification/pytest_generator_registry_ffno.log`
  - `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-06-cdi-lines128-ffno-depth24-ablation/verification/pytest_grid_lines_torch_runner_ffno.log`
  - `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-06-cdi-lines128-ffno-depth24-ablation/verification/pytest_grid_lines_compare_wrapper_ffno.log`
  - `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-06-cdi-lines128-ffno-depth24-ablation/verification/compileall.log`
- Long-run completion proof unchanged:
  - tmux session printed `__EXIT_CODE__=0`
  - root invocation:
    `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-06-cdi-lines128-ffno-depth24-ablation/runs/ffno_depth24_20260507T052301Z/invocation.json`
  - row-local proofs:
    `runs/pinn_ffno_depth24/launcher_completion.json`,
    `runs/pinn_ffno_depth24/exit_code_proof.json`
- Contract audit outcome (post-correction):
  - baseline `pinn_ffno`: `fno_blocks=4`, `fno_cnn_blocks=0`
  - fresh `pinn_ffno_depth24`: `fno_blocks=24`, `fno_cnn_blocks=0`
  - allowed drift now records `manifest_claim_boundary` with both the
    user-argv label and the delivered `cdi_ffno_depth_ablation_only` label
- Side-by-side FFNO result summary (unchanged):
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

## Follow-Up Work

- The wrapper still does not auto-emit effective row-spec overrides into
  the root `invocation.json`. A future small enhancement to
  `scripts/studies/grid_lines_compare_wrapper.py` could write a
  `delivered_contract` / `effective_row_spec_overrides` block at run time so
  manual provenance corrections are not needed for analogous override-bearing
  rows. Out of scope for this ablation item.
- The supervised depth-24 companion row (`2026-05-06-cdi-lines128-supervised-ffno-depth24-no-refiner-rerun`)
  is still a separate backlog item.
- Manuscript-facing promotion remains deferred to
  `2026-05-06-cdi-lines128-ffno-depth24-final-paper-refresh`, which can decide
  whether the scalar metric gains justify the much larger parameter and
  runtime cost.

## Residual Risks

- Single-seed CDI ablation with only `2 / 2` train/test images; appropriate
  for same-contract directionality, not for broad statistical claims.
- The depth-24 result remains append-only and unpromoted; paper tables and
  figures are not refreshed in this item.
- The wrapper-level `argv` and `parsed_args` still record the user-supplied
  `--fno-blocks 4` / `--manifest-claim-boundary decision_support_append_only`
  because those were the literal CLI tokens. The corrective annotations under
  `extra` and the regenerated manifest are the authoritative description of
  the delivered contract; readers must consult the `extra.delivered_contract`
  block or the row-local `runs/pinn_ffno_depth24/invocation.json` for the
  effective fno_blocks=24 launch.
