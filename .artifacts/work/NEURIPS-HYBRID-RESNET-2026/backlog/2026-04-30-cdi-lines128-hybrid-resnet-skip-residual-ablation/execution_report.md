# Execution Report

## Completed In This Pass

- Kept the skip/residual ablation residual-scale controls execution-only by removing them from `ptycho_torch.config_params.ModelConfig`, threading them through `PyTorchExecutionConfig`, and teaching the Hybrid ResNet builder and Lightning checkpoint path to consume a separate `generator_overrides` surface.
- Changed the ablation helper and Torch runner so fresh rows use per-row training roots under `training_runs/<row_id>/...` while keeping the append-only comparison bundle rooted at the shared ablation artifact directory.
- Repaired the delivered ablation bundle metadata by fixing the encoder-fusion cross-reference, correcting the skip-add wording in the durable summary, and backfilling row-local training-output directories plus `training_output_recovery.json` from the already-completed shared Lightning outputs.

## Completed Current-Scope Work

- Resolved the blocking review items around execution-contract compliance, provenance isolation, and maintainability for the implemented skip/residual ablation.
- Updated the current-checkout execution report target, execution manifest, cross-reference manifest, and durable summary without changing the completed six-row CDI benchmark authority or broadening scope.
- Added focused tests covering the corrected cross-reference, row-local training-root planning, execution-only residual-config surface, and Hybrid ResNet fixed-residual build path.

## Follow-Up Work

- Optional only: `pinn_hybrid_resnet_skip_gated_add` remains the bounded next row if a later approved plan reopens this ablation family.

## Residual Risks

- The recovered `training_runs/<row_id>/` directories are post-review provenance repairs derived from the original shared Lightning output root. They isolate row-local checkpoints/logs for auditing, but the preserved invocation artifacts still show the original shared launch-root history.
- Historical `hparams.yaml` files inside the recovered training roots remain the original Lightning emissions from the pre-fix run. The authoritative contract fix now lives in the runner/workflow code, the updated manifests, and the recovery manifest rather than in rewritten historical YAML.
- Scientific interpretation is unchanged: this remains same-contract, two-test-image, decision-support CDI evidence rather than promoted paper-grade headline evidence.
