Brief:
Run `pytest tests/tf_helper/test_translation_shape_guard.py::test_non_xla_translation_guard -vv` and tee the output to `$HUB/green/pytest_tf_translation_guard.log` to prove the existing guard is still GREEN before touching the TensorFlow CLI.
Export `AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md`, `HUB=$PWD/plans/active/FIX-PYTORCH-FORWARD-PARITY-001/reports/2025-11-13T000000Z/forward_parity`, `TF_XLA_FLAGS="--tf_xla_auto_jit=0"`, and `USE_XLA_TRANSLATE=0`, then rerun the scaled TF training command (train/test fly001 NPZ, `n_images=64`, `n_groups=32`, `batch_size=4`, `gridsize=2`, `--do_stitching`) so `$HUB/tf_baseline/phase_c1_scaled/analysis/forward_parity_debug_tf/{stats.json,offsets.json,png grids}` and a non-empty `cli/train_tf_phase_c1_scaled.log` land under the hub.
When the run completes, append a “Phase C1 — TF scaled baseline” section to `$HUB/analysis/artifact_inventory.txt` (parameters, env capture, sha1 of the new stats) and refresh `$HUB/summary.md`; if the CLI still fails (e.g., reshape 0→4 in `_translate_images_simple`), capture `$HUB/tf_baseline/phase_c1_scaled/red/blocked_<timestamp>_tf_translation_guard.md` with the new stack trace plus pytest log reference instead of partial edits.

Summary: plans/active/FIX-PYTORCH-FORWARD-PARITY-001/summary.md
Plan: plans/active/FIX-PYTORCH-FORWARD-PARITY-001/implementation.md
Selector: tests/tf_helper/test_translation_shape_guard.py::test_non_xla_translation_guard
