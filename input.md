Brief:
Set `HUB=$PWD/plans/active/FIX-PYTORCH-FORWARD-PARITY-001/reports/2025-11-13T000000Z/forward_parity`, `TF_XLA_FLAGS="--tf_xla_auto_jit=0"`, and `USE_XLA_TRANSLATE=0`, then rerun the scaled TF training command (train/test fly001, `n_images=64`, `n_groups=32`, `gridsize=2`, `--do_stitching`) so the current shape-mismatch failure reproduces under `$HUB/tf_baseline/phase_c1_scaled/cli/train_tf_phase_c1_scaled.log`.
Update `ptycho/tf_helper.py` so the non-XLA translation path realigns the batch dimension (or falls back to the safe `_translate_images_simple` path) before calling `tf.raw_ops.ImageProjectiveTransformV3`, and add a regression test `tests/tf_helper/test_translation_shape_guard.py::test_non_xla_translation_guard` that mocks `should_use_xla()` to `False` and asserts `_reassemble_patches_position_real` completes for a fake gridsize=2 tensor.
After the guard and test pass, rerun the scaled TF training command (same flags/env) so `tf_baseline/phase_c1_scaled/analysis/forward_parity_debug_tf/{stats.json,offsets.json,patch grids}` exist; update `$HUB/analysis/artifact_inventory.txt` with a “Phase C1 — TF scaled baseline” section or capture `$HUB/tf_baseline/phase_c1_scaled/red/blocked_<timestamp>_tf_translation_guard.md` if it still fails.

Summary: plans/active/FIX-PYTORCH-FORWARD-PARITY-001/summary.md
Plan: plans/active/FIX-PYTORCH-FORWARD-PARITY-001/implementation.md
Selector: tests/tf_helper/test_translation_shape_guard.py::test_non_xla_translation_guard
