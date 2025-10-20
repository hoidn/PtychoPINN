# ADR-003 Phase C4 — Inference CLI Debug Notes (2025-10-20T044344Z)

## Context
- Initiative: ADR-003-BACKEND-API, Phase C4 implementation/validation.
- Focus: Diagnose why C4.C6/C4.C7 acceptance tests remain RED after Ralph’s factory integration pass.
- Evidence collected from targeted pytest reproductions executed 2025-10-20T04:43Z–04:45Z.

## Key Findings
1. **Checkpoint discovery regression**
   - Command: `CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_cli_inference_torch.py::TestInferenceCLI::test_accelerator_flag_roundtrip -vv`
   - Log: `pytest_cli_inference_failure.txt`
   - Result: CLI raises `FileNotFoundError` because it only searches for `last.ckpt`, `wts.pt`, `model.pt`. Spec-mandated `wts.h5.zip` (created by `save_torch_bundle`) is ignored, so the factory-produced payload never reaches downstream consumers.
2. **Factory call still followed by legacy IO path**
   - Even after `create_inference_payload()` returns, `ptycho_torch/inference.py` continues with manual Lightning checkpoint loading and `RawData.from_file` against the CLI arguments.
   - Unit tests provide stub files (empty NPZ, placeholder model dir) to focus on flag wiring, so this real IO causes deterministic failures and violates the plan requirement to consume factory outputs instead of ad-hoc wiring.
3. **Integration workflow blocked upstream**
   - Command: `CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_integration_workflow_torch.py::test_run_pytorch_train_save_load_infer -vv`
   - Log: `pytest_integration_failure.txt`
   - Result: Training subprocess now aborts early with `ERROR: Training data file not found ... minimal_dataset_v1.npz`. The TEST-PYTORCH-001 fixture directory currently ships only `minimal_dataset_v1.json`; the RED→GREEN conversion must either materialize the NPZ fixture or regenerate it during the test setup before factory validation runs.
   - Note: Earlier logs (2025-10-20T044500Z artefacts) show the same test proceeded far enough to hit memmap shape drift (238×1 vs expected 4), so once the NPZ is restored we must re-validate for that legacy defect as well.

## Suggested Next Steps
1. **C4.C6/C4.C7 implementation**
   - Consume `InferencePayload` outputs to locate the spec-compliant archive (`wts.h5.zip`) and hand off to a dedicated loader instead of open-coding checkpoint selection. Ensure the CLI exits cleanly when the factory succeeds (unit tests patch the factory to assert execution_config wiring).
   - Gate RawData usage behind a follow-up workflow helper (or inject a test seam) so the CLI no longer attempts to open empty tmp NPZs during configuration-only tests.
2. **Restore/Generate PyTorch integration fixture**
   - Rehydrate `tests/fixtures/pytorch_integration/minimal_dataset_v1.npz` (likely derived from the JSON manifest) or adjust the test to synthesize the NPZ on demand before invoking the CLI.
   - After the dataset exists, rerun the integration test to uncover the previously observed memory-map mismatch (needs revert of `data/memmap/meta.json` to the canonical 34-sample/4-neighbor metadata or regeneration via the data prep toolchain).
3. **Plan / ledger updates**
   - Mark C4.C4 as complete (refactor notes delivered), but keep C4.C6/C4.C7 `[ ]` with updated guidance reflecting the new checkpoint/IO findings.
   - Capture the above failures in docs/fix_plan attempts to preserve traceability for Ralph’s implementation loop.

## Artefact Index
- `pytest_cli_inference_failure.txt` — targeted inference CLI selector (RED failure trace).
- `pytest_integration_failure.txt` — PyTorch integration workflow selector (training subprocess abort due to missing NPZ).

