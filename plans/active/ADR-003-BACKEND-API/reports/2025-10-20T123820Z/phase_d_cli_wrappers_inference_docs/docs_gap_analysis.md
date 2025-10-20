# Phase D.C C4 Documentation Gap Analysis

**Initiative:** ADR-003-BACKEND-API
**Phase:** D.C — Inference CLI Thin Wrapper (Task C4)
**Date:** 2025-10-20
**Mode:** Docs — Evidence Collection

## Objective
Identify documentation deltas left after Phase D.C C3 implementation so the inference CLI thin wrapper docs (Task C4) can be updated in a single follow-up loop.

## Key References Reviewed
- `docs/workflows/pytorch.md` §§12–13 — CLI execution flags + backend selection guidance
- `ptycho_torch/inference.py` (post-C3 implementation)
- `plans/active/ADR-003-BACKEND-API/reports/2025-10-20T114500Z/phase_d_cli_wrappers_inference/inference_refactor.md`
- Findings: POLICY-001 (PyTorch required), FORMAT-001 (NPZ transpose guard)

## Findings

1. **Flag defaults drifted after helper refactor**
   - `docs/workflows/pytorch.md:360` still lists `--accelerator` default as `'cpu'`, but `cli_main()` now sets `default='auto'` with helper-driven resolution (auto → `cuda` if available, else `cpu`).
   - The same table lists `--inference-batch-size` default as `1`, but the CLI now passes `default=None` so the factory reuses training batch size unless overridden (per `_run_inference_and_reconstruct()` contract).

2. **Helper-based flow undocumented for inference**
   - Section 12 only documents the training helper delegation. Inference now shares the same helpers (`validate_paths`, `build_execution_config_from_args`) and routes through `_run_inference_and_reconstruct()` (see `ptycho_torch/inference.py:575-640`). Need to explain this flow + CONFIG-001 guarantees mirroring training language.

3. **Example usage missing / outdated**
   - CLI doc still lacks an inference example command. Epilog inside `cli_main()` shows `--device` examples; documentation should provide updated command highlighting `--accelerator`/`--quiet` usage and the minimal dataset fixture (`tests/fixtures/pytorch_integration/minimal_dataset_v1.npz`).

4. **Artifact expectations not documented**
   - Thin wrapper ensures amplitude/phase PNGs via `save_individual_reconstructions()` (`ptycho_torch/inference.py:622-633`), but workflow guide doesn’t mention these outputs. Add bullet or table entry under inference section.

5. **Deprecation timeline needs alignment**
   - Plan C4 requests explicit deprecation note for `--device`. Training section already states "will be dropped in a future release". Mirror wording for inference and reference Phase E governance (plan row D2/D3 → E1) so readers know removal happens post-ADR acceptance.

## Recommendations for Task C4

- Update `docs/workflows/pytorch.md` inference flag table to match current defaults and describe helper-driven accelerator resolution.
- Add a subsection (e.g., "Helper Delegation (Phase D.C)") summarizing the inference thin-wrapper flow with anchors to `ptycho_torch/cli/shared.py`, `_run_inference_and_reconstruct`, and the RED/GREEN tests (`tests/torch/test_cli_inference_torch.py`).
- Provide an example CLI command using the minimal dataset fixture, demonstrating `--accelerator cpu` + `--quiet` and highlighting expected output artifacts.
- Note that `--device` remains accepted but emits a deprecation warning; removal scheduled for Phase E (cite implementation plan §Phase E).
- After docs update, mark `phase_d_cli_wrappers/plan.md` row C4 `[x]`, refresh `summary.md`, and log Attempt #51 in `docs/fix_plan.md` referencing this report.

