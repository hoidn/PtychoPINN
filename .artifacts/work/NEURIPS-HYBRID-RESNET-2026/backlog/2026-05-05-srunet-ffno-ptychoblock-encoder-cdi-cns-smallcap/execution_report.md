## Completed In This Pass

- Implemented the shared `hybrid_resnet_ffno_ptychoblock_encoder` family across
  the CDI Torch generator registry, checkpoint rebuild path, config validation,
  grid-lines runner/wrapper, and PDEBench CNS profile/model surfaces while
  preserving the fixed SRU-Net shell outside the encoder.
- Added focused regression coverage for the new architecture and profile wiring:
  `tests/torch/test_generator_registry.py`,
  `tests/torch/test_fno_generators.py`,
  `tests/torch/test_lightning_checkpoint.py`,
  `tests/torch/test_grid_lines_torch_runner.py`,
  `tests/test_grid_lines_compare_wrapper.py`, and
  `tests/studies/test_pdebench_image128_models.py`.
- Re-ran the backlog item's deterministic gates and archived fresh logs under
  `verification/`:
  - prerequisite presence script
  - `pytest -q tests/torch/test_fno_generators.py -k "PtychoBlock or hybrid_resnet or ffno"`
  - `pytest -q tests/torch/test_grid_lines_torch_runner.py -k "hybrid_resnet or ffno"`
  - `pytest -q tests/studies/test_pdebench_image128_models.py -k "hybrid_resnet or ffno"`
  - `pytest -q tests/test_grid_lines_compare_wrapper.py -k "hybrid_resnet or ffno"`
  - `pytest -q tests/torch/test_generator_registry.py`
  - `pytest -q tests/torch/test_lightning_checkpoint.py -k "hybrid_resnet or ffno"`
  - `python -m compileall -q ptycho_torch scripts/studies`
- Launched the fresh CDI row under tmux on the locked `lines128` contract and
  kept ownership of tracked PID `2982934` until the tmux shell recorded
  `EXIT_CODE:0`.
- Validated the fresh CDI run root
  `runs/cdi_ffno_ptychoblock_encoder_20260506T183959Z/`:
  root and row invocation payloads both report `status=completed` and
  `exit_code=0`; row-local `metrics.json`, `history.json`, `model.pt`,
  `exit_code_proof.json`, recon NPZ, and fixed-sample visuals are present; and
  run-root `artifact_freshness_check.json` plus `exit_code_proof.json` were
  written after validation.
- Launched the fresh matched-condition CNS row under tmux on the capped h5
  contract in run root
  `runs/cns_ffno_ptychoblock_encoder_20260506T190421Z/`; tracked PID
  `2991458` exited cleanly with tmux wait result `EXIT_CODE:0`.
- Validated the fresh CNS run root
  `runs/cns_ffno_ptychoblock_encoder_20260506T190421Z/`:
  `dataset_manifest.json`, `hdf5_metadata.json`, `split_manifest.json`,
  `metrics_hybrid_resnet_ffno_ptychoblock_encoder_cns.json`,
  `model_profile_hybrid_resnet_ffno_ptychoblock_encoder_cns.json`,
  fixed-sample field visuals/NPZ, and run-root
  `artifact_freshness_check.json` plus `exit_code_proof.json` are present and
  fresh.
- Collated the two fresh rows into the durable summary
  `docs/plans/NEURIPS-HYBRID-RESNET-2026/srunet_ffno_ptychoblock_encoder_cdi_cns_smallcap_summary.md`,
  refreshed `paper_evidence_index.md`, `evidence_matrix.md`,
  `ablation_index.json`, `model_variant_index.json`, `docs/studies/index.md`,
  and `docs/index.md`, and wrote the item-local
  `comparison_bundle.json`.

## Completed Plan Tasks

- Task 1: Freeze Authorities, Contracts, And Lineage Inputs.
- Task 2: Implement The Shared Encoder Variant And Metadata Contract.
- Task 3: Wire The Fresh CDI Row.
- Task 4: Wire The Fresh CNS Profile.
- Task 5: Run Deterministic Code Gates Before Any Expensive Launch.
- Task 6: Launch The Fresh CDI Row.
- Task 7: Launch The Fresh CNS Row.
  Current state: completed and validated.
- Task 8: Collate The Fresh Rows Into Durable Summary And Discoverability
  Surfaces.

## Remaining Required Plan Tasks

None.

## Verification

- Archived deterministic logs under
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-05-srunet-ffno-ptychoblock-encoder-cdi-cns-smallcap/verification/`.
- Current pass counts:
  - `test_fno_generators.log`: `61 passed, 62 deselected`
  - `test_grid_lines_torch_runner.log`: `12 passed, 143 deselected`
  - `test_pdebench_image128_models.log`: `21 passed, 25 deselected`
  - `test_grid_lines_compare_wrapper.log`: `13 passed, 66 deselected`
  - `test_generator_registry.log`: `11 passed`
  - `test_lightning_checkpoint.log`: `4 passed, 11 deselected`
  - `compileall.log`: exit `0`
- CDI completion proof:
  - tmux session `ffno-ptychoblock-cdi`, socket
    `/tmp/claude-tmux-sockets/ffno-ptychoblock-cdi.sock`
  - tracked PID `2982934`
  - tmux shell recorded `EXIT_CODE:0`
  - run root `runs/cdi_ffno_ptychoblock_encoder_20260506T183959Z/`
    contains root `exit_code_proof.json` and row-local
    `runs/pinn_hybrid_resnet_ffno_ptychoblock_encoder/exit_code_proof.json`
- Comparison standard for completed CDI evidence in this pass:
  same locked `lines128` contract, with direct metric comparison to lineage
  baselines and no additional `atol` / `rtol` tolerance because this is a fresh
  mechanism row rather than a parity oracle check.
- CNS completion proof:
  - tmux session `ffno-ptychoblock-cns`, socket
    `/tmp/claude-tmux-sockets/ffno-ptychoblock-cns.sock`
  - tracked PID `2991458`
  - tmux shell recorded `EXIT_CODE:0`
  - run root `runs/cns_ffno_ptychoblock_encoder_20260506T190421Z/`
    contains `artifact_freshness_check.json`,
    `exit_code_proof.json`,
    `metrics_hybrid_resnet_ffno_ptychoblock_encoder_cns.json`, and
    `model_profile_hybrid_resnet_ffno_ptychoblock_encoder_cns.json`
- Comparison standard for completed CNS evidence in this pass:
  same matched-condition capped h5 contract, with direct metric comparison to
  the existing `history_len=5` capped headline rows and no additional
  `atol` / `rtol` tolerance because this is a fresh mechanism row rather than a
  parity oracle check.

## Residual Risks

- The new CDI row is append-only mechanism evidence only. Even if its final
  metrics are strong, it must not replace the immutable six-row `lines128`
  authority or the matched-condition capped CNS headline authority.
- The CNS profile schema surfaces the fixed encoder recipe under
  `profile_config.*` rather than flattening those fields at the top level of
  `model_profile_hybrid_resnet_ffno_ptychoblock_encoder_cns.json`. The recipe
  is preserved, but downstream readers must inspect the nested config payload.
- `/home/ollie/Documents/neurips/` is not present in the current environment,
  so any paper-side index mirroring remains unavailable here; the durable record
  is currently repo-local.
