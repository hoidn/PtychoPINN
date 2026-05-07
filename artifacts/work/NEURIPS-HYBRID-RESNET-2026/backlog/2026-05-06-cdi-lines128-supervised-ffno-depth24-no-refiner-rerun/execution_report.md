# Execution Report

## Completed In This Pass

- Materialized the plan-required stable row-local artifact paths directly under
  the item-local artifact root by adding symlinks that resolve into the
  timestamped wrapper invocation directory:
  - `runs/supervised_ffno_depth24/` -> `runs/supervised_ffno_depth24_20260507T192840Z/runs/supervised_ffno_depth24/`
  - `recons/supervised_ffno_depth24/` -> `runs/supervised_ffno_depth24_20260507T192840Z/recons/supervised_ffno_depth24/`
  - `visuals/amp_phase_supervised_ffno_depth24.png` -> `runs/supervised_ffno_depth24_20260507T192840Z/visuals/amp_phase_supervised_ffno_depth24.png`
  - `visuals/amp_phase_error_supervised_ffno_depth24.png` -> `runs/supervised_ffno_depth24_20260507T192840Z/visuals/amp_phase_error_supervised_ffno_depth24.png`
  This satisfies Task 3's blocking artifact-packaging contract without
  redoing the long run; the underlying evidence is the same single supervised
  depth-24 run, now discoverable at the contractual stable paths.
- Added a focused regression
  (`tests/test_grid_lines_compare_wrapper.py::test_enrich_paper_row_payload_emits_row_local_naming_for_supervised_ffno_depth24`)
  that exercises `_recover_torch_row_payload` and `_enrich_paper_row_payload`
  end-to-end for the new row id and asserts:
  - row-local payload paths target `runs/supervised_ffno_depth24/{metrics,history,stdout,stderr}` and `recons/supervised_ffno_depth24/recon.npz`
  - row-local visuals payload uses `visuals/amp_phase_supervised_ffno_depth24.png` and `visuals/amp_phase_error_supervised_ffno_depth24.png`
  - completion-finalization writes `runs/supervised_ffno_depth24/exit_code_proof.json`
  - the locked `decision_support_append_only` row status is preserved
  This closes the medium-severity gap that the prior tests covered preflight
  routing and row-spec registration only.
- Re-ran every required deterministic gate after both fixes and refreshed the
  archived logs under
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-06-cdi-lines128-supervised-ffno-depth24-no-refiner-rerun/verification/`:
  - `input_presence_gate.log` (presence gate)
  - `pytest_grid_lines_torch_runner_ffno.log` (`4 passed, 152 deselected`)
  - `pytest_grid_lines_compare_wrapper_ffno.log` (`13 passed, 76 deselected` -- one more than before, the new regression)
  - `compileall.log` (clean)
- Recorded the symlink materialization decision in
  `verification/stable_row_local_paths_manifest.json`, including each stable
  path, the symlink target, and SHA-256 digests of the resolved row-local
  files so downstream consumers can verify identity rather than re-derive it.

## Completed Current-Scope Work

- Task 1: Freeze Authorities And Run The Input Presence Gate.
- Task 2: Add Minimal `supervised_ffno_depth24` Row Plumbing -- the previously
  added preflight/registration regressions, plus the new row-local naming and
  completion-finalization regression added in this pass, now cover the full
  Task 2 explicit regression obligation (row-local artifact naming and
  completion-finalization under `supervised_ffno_depth24`).
- Task 3: Launch Exactly One Fresh Supervised Depth-24 No-Refiner Run -- the
  long run from the prior pass remains the single fresh run; this pass added
  the stable-path materialization that was the unmet blocking contract.
- Task 4: Audit Same-Contract Fairness And Depth-Only Delta -- audit and
  comparison payloads under `verification/` from the prior pass remain valid;
  no contract drift was introduced and no rerun was needed.
- Task 5: Write The Durable Summary And Refresh Discoverability -- the durable
  summary and discoverability surfaces from the prior pass still apply; this
  pass did not touch them because no claim boundary or evidence statement
  changed.

### Verification Summary

- Required stable row-local artifacts now resolve under the item-local root:
  - `runs/supervised_ffno_depth24/{invocation.json,config.json,history.json,metrics.json,model.pt,exit_code_proof.json,launcher_completion.json,stdout.log,stderr.log}`
  - `recons/supervised_ffno_depth24/recon.npz`
  - `visuals/amp_phase_supervised_ffno_depth24.png`
  - `visuals/amp_phase_error_supervised_ffno_depth24.png`
  - identity proof: `verification/stable_row_local_paths_manifest.json`
- Long-run completion proof from the prior pass still applies:
  - tmux pane capture: `verification/tmux_completion_capture.log`
  - shell exit marker `__EXIT_CODE__=0`
  - wrapper invocation:
    `runs/supervised_ffno_depth24_20260507T192840Z/invocation.json`
  - row-local proofs:
    `runs/supervised_ffno_depth24/launcher_completion.json` and
    `runs/supervised_ffno_depth24/exit_code_proof.json`
- Same-contract audit (`verification/contract_audit_supervised_depth24_vs_depth4.json`)
  and depth-only comparison payloads
  (`verification/comparison_supervised_depth24_vs_depth4.{json,csv}`) from
  the prior pass remain authoritative.
- Required deterministic gates re-passed in this pass; logs archived under
  `verification/` as listed above.
- JSON discovery surfaces still validate:
  - `python -m json.tool docs/plans/NEURIPS-HYBRID-RESNET-2026/model_variant_index.json > /dev/null`
  - `python -m json.tool docs/plans/NEURIPS-HYBRID-RESNET-2026/ablation_index.json > /dev/null`

## Follow-Up Work

- Align the wrapper-manifest claim boundary
  (`decision_support_append_only`) with the durable summary claim boundary
  (`cdi_supervised_ffno_depth_companion_only`), or document a durable
  precedence rule so downstream consumers do not have to infer which
  authority wins. This was the prior reviewer's follow-up and remains
  out-of-scope for this revise pass; flagging it for a later targeted item.

## Residual Risks

- This remains a single-seed, two-image-per-split CDI follow-up; it is useful
  for same-contract directionality, not broad statistical claims.
- The supervised depth-24 row improves several phase-side metrics and amplitude
  MAE versus the corrected four-block supervised row, but amplitude SSIM/FRC
  regress slightly, validation loss is worse, and runtime grows by more than
  `5x`; any paper-facing promotion still needs the later final-refresh
  judgment.
- The stable row-local paths under the item-local root are symlinks into the
  timestamped wrapper invocation directory, not duplicated bytes. Consumers
  that read through `Path.resolve()` will see the timestamped target; for
  downstream tools that only care about path-name stability this is
  intentional and preserves provenance, but tools that copy the item root
  with `cp -P` (no `-L`) must follow links to capture the underlying
  artifacts.
