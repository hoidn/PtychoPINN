# Execution Report

## Completed In This Pass

- Added ablation-helper regressions that pin two failure modes: fresh rows being mislabeled as recovered during collation, and reuse short-circuiting without a valid row-local training root.
- Tightened the ablation helper so fresh rows only reuse existing outputs when `invocation.json` still points at the expected `training_runs/<row_id>` root and that row-local training root exists.
- Rebuilt the append-only comparison bundle from the existing row-local fresh outputs, materialized direct-row stdout/stderr plus exit-code proof artifacts for the three fresh rows, and rewrote `metrics.json` / `model_manifest.json` so fresh-row provenance is consistent.

## Completed Current-Scope Work

- Resolved the blocking implementation-review issue: fresh rows now emit `recovered_from_existing_artifacts: false`, no longer advertise nonexistent recovered logs, and carry valid `stdout.log`, `stderr.log`, and `exit_code_proof.json` outputs under `runs/<row_id>/`.
- Re-ran the required deterministic selectors and supporting gates: `tests/torch/test_fno_generators.py -k "hybrid_resnet or resnet_decoder_block or skip_style"`, `tests/torch/test_grid_lines_torch_runner.py -k "hybrid_skip or hybrid_resnet_blocks or resnet_width"`, `tests/studies/test_lines128_hybrid_resnet_skip_residual_ablation.py`, summary presence, `python -m compileall -q ptycho_torch scripts/studies`, and `pytest -v -m integration`.
- Preserved scope and authority boundaries: the completed six-row CDI benchmark remains unchanged, the reused `pinn_hybrid_resnet` baseline stays promoted from the authoritative source root, and optional `pinn_hybrid_resnet_skip_gated_add` remains deferred.

## Follow-Up Work

- Optional only: `pinn_hybrid_resnet_skip_gated_add` remains the bounded next row if a later approved plan reopens this ablation family.
- If this helper is extended again, keep the row-local training-root contract and bundle-consistency normalization in place so future collations cannot silently regress to recovered fresh-row provenance.

## Residual Risks

- Scientific interpretation is unchanged: this remains same-contract, two-test-image, decision-support CDI evidence rather than promoted paper-grade headline evidence.
- The fixed residual-scale read remains narrow to this Hybrid ResNet shell and frozen `lines128` contract; broader transfer claims are still unsupported.
