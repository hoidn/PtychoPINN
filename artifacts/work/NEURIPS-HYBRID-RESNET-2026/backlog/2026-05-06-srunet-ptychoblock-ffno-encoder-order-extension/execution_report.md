# Execution Report

## Completed In This Pass

- Implemented the reversed-order SRU-Net encoder family
  `hybrid_resnet_ptychoblock_ffno_encoder` / `hybrid_resnet_ptychoblock_ffno_encoder_cns`
  across generator, registry, config, CDI runner, compare-wrapper, and
  PDEBench CNS profile/model surfaces.
- Extracted one shared no-refiner FFNO stack helper and routed the end-to-end
  CDI FFNO generator plus both encoder-order SRU-Net variants through the same
  stack contract.
- Added focused test coverage for the new architecture id, registry routing,
  checkpoint rebuild path, CDI runner payload, compare-wrapper row, and CNS
  profile/model surface.
- Launched and completed the fresh CDI and CNS `20`-epoch mechanism rows under
  tracked tmux ownership, then validated fresh output artifacts.
- Wrote root-level `exit_code_proof.json` and
  `artifact_freshness_check.json` for both fresh run roots.
- Built the item-local `comparison_bundle.json`.
- Wrote the durable summary
  `docs/plans/NEURIPS-HYBRID-RESNET-2026/srunet_encoder_order_ffno_vs_ptychoblock_summary.md`
  and updated the evidence/discoverability indexes:
  `evidence_matrix.md`, `paper_evidence_index.md`, `model_variant_index.json`,
  `ablation_index.json`, and `docs/studies/index.md`.

## Completed Plan Tasks

- Task 1: Implement the shared FFNO stack and reversed-order SRU-Net
  architecture.
- Task 2: Wire the new architecture through CDI/CNS study entrypoints,
  manifests, and checkpoint reconstruction.
- Task 3: Add and pass the required focused tests, compile gate, and
  integration gate.
- Task 4: Run the fresh CDI mechanism row and validate fresh artifacts.
- Task 5: Run the fresh CNS mechanism row and validate fresh artifacts.
- Task 6: Publish the comparison bundle, durable summary, and discovery-index
  updates for the three-row encoder-order comparison.

## Remaining Required Plan Tasks

- None.

## Verification

- Required presence gate: pass.
- `pytest -q tests/torch/test_fno_generators.py -k "PtychoBlock or hybrid_resnet or ffno"`:
  `62 passed, 62 deselected`
- `pytest -q tests/torch/test_grid_lines_torch_runner.py -k "hybrid_resnet or ffno"`:
  `13 passed, 143 deselected`
- `pytest -q tests/studies/test_pdebench_image128_models.py -k "hybrid_resnet or ffno"`:
  `22 passed, 27 deselected`
- `pytest -q tests/test_grid_lines_compare_wrapper.py -k "ffno or hybrid_resnet"`:
  `18 passed, 67 deselected`
- `pytest -q tests/torch/test_generator_registry.py`:
  `12 passed`
- `pytest -q tests/torch/test_lightning_checkpoint.py -k "hybrid_resnet or ffno"`:
  `5 passed, 11 deselected`
- `python -m compileall -q ptycho_torch scripts/studies`: exit `0`
- `pytest -v -m integration`:
  `5 passed, 4 skipped, 2298 deselected, 2 warnings`
- Fresh CDI root
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-06-srunet-ptychoblock-ffno-encoder-order-extension/runs/cdi_ptychoblock_ffno_encoder_20260507T094629Z/`
  carries `exit_code_proof.json` with tracked PID `3232115` and a passing
  `artifact_freshness_check.json`.
- Fresh CNS root
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-06-srunet-ptychoblock-ffno-encoder-order-extension/runs/cns_ptychoblock_ffno_encoder_20260507T100829Z/`
  carries `exit_code_proof.json` with tracked PID `3234001` and a passing
  `artifact_freshness_check.json`.

## Residual Risks

- The new encoder-order rows remain bounded `20`-epoch mechanism probes only.
  They must not be promoted into the six-row CDI authority or the matched
  `40`-epoch CNS headline lane.
- Reversing the encoder order moves the FFNO stack onto the post-downsample
  feature map. That raises parameter count relative to the FFNO-first companion
  because the stack now operates at a wider channel count; this is an approved
  architectural consequence of the order swap, not a tuning deviation.
- Both pillars still rely on single-seed evidence for this mechanism family.
  Any promotion-grade claim would require a later explicit plan authorizing
  longer or replicated reruns.
