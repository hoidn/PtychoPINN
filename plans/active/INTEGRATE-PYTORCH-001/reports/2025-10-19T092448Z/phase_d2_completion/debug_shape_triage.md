# Phase C4 Shape Triage — 2025-10-19T092448Z

## Context
- Focus: `INTEGRATE-PYTORCH-001-STUBS` Phase D2.C4 (modernize stitching tests and make them green).
- Trigger: Ralph's 2025-10-19T084016Z attempt showed `_reassemble_cdi_image_torch` raises a TensorFlow shape error (`cond/zeros` axis mismatch) once tests exercise the real stitching path.

## Hypotheses
1. **Channel axis ordering mismatch** — `_reassemble_cdi_image_torch` forwards PyTorch predictions with channel-first layout (`(n, C, H, W)` after the manual `unsqueeze(1)`), but `tf_helper.reassemble_position` expects channel-last tensors (`(n, H, W, C)`).
2. **Mock Lightning module output shape drift** — The new green tests supply a mock module returning `(batch, N, N)` tensors; this may hide bugs in the true Lightning output, which is likely `(batch, gridsize^2, N, N)` and still channel-first.
3. **Global offsets dtype/shape mismatch** — Incorrect casting of `global_offsets` could trigger the TensorFlow error (needs confirmation even though the stack trace points at the image tensor path).

## Triage Results
- **(1) Channel axis ordering mismatch** — Reproduced failure by calling `tf_helper.reassemble_position` with a `(n, 1, 64, 64)` tensor; TensorFlow raised the exact error seen in the pytest log. Transposing to `(n, 64, 64, 1)` succeeded (command: `python - <<'PY' ...` with channel-first vs channel-last experiment). ✅ root cause confirmed.
- **(2) Mock Lightning module output shape drift** — Examined `_build_inference_dataloader`: containers yield diffraction tensors shaped `(nsamples, 64, 64, gridsize^2)`. Real models will therefore hit the same axis-ordering bug unless `_reassemble_cdi_image_torch` explicitly moves the channel dimension to the end. Mock should emit complex tensors shaped `(batch, gridsize^2, N, N)` (channel-first) or `(batch, N, N, gridsize^2)` (channel-last) to mirror the production contract. ⚠️ Needs adjustment alongside production fix.
- **(3) Global offsets dtype/shape mismatch** — Verified via inspection that `global_offsets` retains the expected `(nsamples, 1, 2, 1)` float64 shape; TensorFlow accepts this when the image tensor is channel-last. No further action required. ❎ ruled out as blocker.

## Next Steps
1. Modify `_reassemble_cdi_image_torch` to convert any PyTorch prediction tensor to channel-last before delegating to TensorFlow, e.g. `obj_tensor_full = obj_tensor_full.permute(0, 2, 3, 1)` for channel-first inputs and `np.moveaxis` as final safety before calling `reassemble_position`.
2. Update the green tests to:
   - Keep one guard case asserting the `train_results=None` NotImplemented branch.
   - Ensure the mock Lightning module produces deterministic complex outputs with an explicit channel dimension matching `config.model.gridsize ** 2`.
   - Assert that the returned amplitude/phase arrays are finite and that `results['obj_tensor_full']` is channel-last.
3. Re-run `pytest tests/torch/test_workflows_components.py -k ReassembleCdiImageTorch -vv | tee plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-19T092448Z/phase_d2_completion/pytest_stitch_green.log` after the implementation fix.
4. Once green, refresh `phase_d2_completion.md` (C4 row), `summary.md`, and docs/fix_plan.md Attempts with the new artifact path (`2025-10-19T092448Z`).

## Evidence
- TensorFlow axis experiment (channel-first fails, channel-last succeeds): see Python REPL output captured in supervisor notes during this loop.
- Prior failing log: `plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-19T084016Z/phase_d2_completion/pytest_stitch_green.log` (lines 1148-1189).
