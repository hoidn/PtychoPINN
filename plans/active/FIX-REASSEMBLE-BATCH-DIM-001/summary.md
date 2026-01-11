### Turn Summary
Fixed batched reassembly to treat `padded_size=None` as unset, so it falls back to `get_padded_size()`.
Reran the targeted regression; it now passes.
Artifacts: `.artifacts/FIX-REASSEMBLE-BATCH-DIM-001/pytest_reassemble_batch_fix.log`.

### Turn Summary
Ran the targeted reassembly regression test; it failed with a ValueError in `_reassemble_position_batched` because `padded_size` was None.
Captured the failure log and traced the likely cause to `ReassemblePatchesLayer` passing `padded_size=None` into `mk_reassemble_position_batched_real`, which then skips the `get_padded_size()` fallback.
Next: treat `padded_size=None` as unset (use `params.get_padded_size()`) and rerun the regression test.
Artifacts: `.artifacts/FIX-REASSEMBLE-BATCH-DIM-001/pytest_reassemble_batch.log`.

### Turn Summary
Updated `_reassemble_position_batched` to accumulate per-sample canvases instead of collapsing the batch.
Aligned `ReassemblePatchesLayer` output shape metadata and updated the dense reassembly regression test to expect batch preservation.
Tests not run (per user request).
Artifacts: `.artifacts/FIX-REASSEMBLE-BATCH-DIM-001/`.
