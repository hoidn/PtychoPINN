### Turn Summary
Updated `_reassemble_position_batched` to accumulate per-sample canvases instead of collapsing the batch.
Aligned `ReassemblePatchesLayer` output shape metadata and updated the dense reassembly regression test to expect batch preservation.
Tests not run (per user request).
Artifacts: `.artifacts/FIX-REASSEMBLE-BATCH-DIM-001/`.
