### Turn Summary
Confirmed the non-XLA translation guard and its regression tests already exist in tree (`ptycho/tf_helper.py` and `tests/tf_helper/test_translation_shape_guard.py`), so the code side of Phase C1d is complete.
Found the scaled TF baseline slot empty (`tf_baseline/phase_c1_scaled/analysis/` has no stats and the CLI log is 0 bytes), so we still lack the TF debug bundle that unblocks parity.
Next: rerun the scaled TF training command with XLA disabled, capture the guard pytest log, and either archive the new artifacts plus inventory update or log a fresh blocker with the stack trace.
Artifacts: plans/active/FIX-PYTORCH-FORWARD-PARITY-001/reports/2025-11-13T000000Z/forward_parity/tf_baseline/phase_c1_scaled/cli/train_tf_phase_c1_scaled.log

### Turn Summary
Fixed non-XLA translation crash in TensorFlow training when gridsize > 1 by disabling ImageProjectiveTransformV3 fast path and using _translate_images_simple with broadcast logic to handle batch dimension mismatches between flattened images (b*c) and unflattened translations (b).
Resolved the "Shapes of all inputs must match" error at epoch 1; training now completes successfully with gridsize=2, though a separate reshape issue remains in inference.
Next: Document the inference blocker (reshape 0→4 in _translate_images_simple during eval) and escalate to supervisor for Phase C direction decision.
Artifacts: tests/tf_helper/test_translation_shape_guard.py (3/3 GREEN), docs/findings.md:TF-NON-XLA-SHAPE-001

Checklist:
- Files touched: ptycho/tf_helper.py, tests/tf_helper/test_translation_shape_guard.py, docs/findings.md
- Tests run: pytest tests/tf_helper/test_translation_shape_guard.py -vv (3 passed, 5.13s)
- Artifacts updated: docs/findings.md (TF-NON-XLA-SHAPE-001), commit 801780b6
