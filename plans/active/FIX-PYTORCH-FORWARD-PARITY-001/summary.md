### Turn Summary
Fixed non-XLA translation crash in TensorFlow training when gridsize > 1 by disabling ImageProjectiveTransformV3 fast path and using _translate_images_simple with broadcast logic to handle batch dimension mismatches between flattened images (b*c) and unflattened translations (b).
Resolved the "Shapes of all inputs must match" error at epoch 1; training now completes successfully with gridsize=2, though a separate reshape issue remains in inference.
Next: Document the inference blocker (reshape 0→4 in _translate_images_simple during eval) and escalate to supervisor for Phase C direction decision.
Artifacts: tests/tf_helper/test_translation_shape_guard.py (3/3 GREEN), docs/findings.md:TF-NON-XLA-SHAPE-001

Checklist:
- Files touched: ptycho/tf_helper.py, tests/tf_helper/test_translation_shape_guard.py, docs/findings.md
- Tests run: pytest tests/tf_helper/test_translation_shape_guard.py -vv (3 passed, 5.13s)
- Artifacts updated: docs/findings.md (TF-NON-XLA-SHAPE-001), commit 801780b6
