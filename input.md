Mode: Perf
Focus: FIX-TF-C1D-SCALED-RERUN-001
Selector: tests/tf_helper/test_translation_shape_guard.py::test_non_xla_translation_guard

Overview:
Phase C1d remains blocked because `$HUB/tf_baseline/phase_c1_scaled/analysis/` has never produced `forward_parity_debug_tf/*` and the only artifacts are the Nov 14 guard and CLI logs. Capture a fresh GREEN result for the TF-NON-XLA-SHAPE-001 guard, then rerun the scaled TensorFlow CLI with the same dataset knobs used for the PyTorch Phase B3 slice while `TF_XLA_FLAGS="--tf_xla_auto_jit=0"` and `USE_XLA_TRANSLATE=0` stay exported per spec-ptycho-runtime.md. Success means recording a non-empty CLI log plus `analysis/forward_parity_debug_tf/{stats.json,offsets.json,png grids}` and updating the hub inventory/summary with a stats sha1 + env capture; failure must instead produce `$TF_BASE/red/blocked_<timestamp>_tf_translation_guard.md` that cites both the pytest output and the CLI stack trace. No production code edits are needed—this loop is about delivering TensorFlow evidence or a blocker so Phase C parity can resume.

### Workload Spec
## Goal
Gather current TensorFlow guard + scaled CLI artifacts (or document a blocker) so the Phase C contract is satisfied: a new guard log under `$HUB/green/`, a non-zero `$TF_BASE/cli/train_tf_phase_c1_scaled.log`, and either a populated `$TF_BASE/analysis/forward_parity_debug_tf/` bundle with sha1 recorded in `analysis/artifact_inventory.txt` or a blocker referencing both logs. Use the scaled knobs from the PyTorch study (`n_images=64`, `n_groups=32`, `gridsize=2`, `neighbor_count=7`, `batch_size=4`, `max_epochs=1`, `--do_stitching --quiet`) to keep parity apples-to-apples.

## Contracts
- docs/specs/spec-ptycho-workflow.md:46 — Backend parity requires TensorFlow evidence (stats, offsets, PNGs) before advancing PyTorch forward-parity work.
- docs/specs/spec-ptycho-runtime.md:15 — Non-XLA translation runs MUST respect the `USE_XLA_TRANSLATE`/`TF_XLA_FLAGS` toggles; this rerun keeps XLA disabled to align with TF-NON-XLA-SHAPE-001.

## Interfaces
- tests/tf_helper/test_translation_shape_guard.py::test_non_xla_translation_guard — tests/tf_helper/test_translation_shape_guard.py:46-87. Ensures `_translate_images_simple` handles batch-size mismatches when XLA is off.
- scripts/training/train.py::main — scripts/training/train.py:300-420. CLI orchestrator that updates `params.cfg` (CONFIG-001) and runs train→eval; call it with the scaled dataset paths and knobs while teeing logs to `$TF_BASE/cli/train_tf_phase_c1_scaled.log`.
- plans/active/FIX-PYTORCH-FORWARD-PARITY-001/reports/2025-11-13T000000Z/forward_parity/analysis/artifact_inventory.txt::Phase C1 — TF scaled baseline — Inventory section that must reflect the new stats sha1/env capture or blocker reference so reviewers can audit this rerun.

## Pseudocode
1. `export AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md` and `export HUB="$PWD/plans/active/FIX-PYTORCH-FORWARD-PARITY-001/reports/2025-11-13T000000Z/forward_parity"`.
2. `export TF_BASE="$HUB/tf_baseline/phase_c1_scaled"; mkdir -p "$TF_BASE"/{cli,analysis,green,red}`.
3. `export TF_XLA_FLAGS="--tf_xla_auto_jit=0"; export USE_XLA_TRANSLATE=0; printf 'TF_XLA_FLAGS=%s\nUSE_XLA_TRANSLATE=%s\n' "$TF_XLA_FLAGS" "$USE_XLA_TRANSLATE" | tee "$TF_BASE/cli/env_capture_scaled.txt"`.
4. `pytest tests/tf_helper/test_translation_shape_guard.py::test_non_xla_translation_guard -vv | tee "$HUB/green/pytest_tf_translation_guard.log"` — stop immediately and file a blocker if this fails.
5. Run `python scripts/training/train.py --backend tensorflow --train_data_file datasets/fly001_reconstructed_prepared/fly001_reconstructed_final_downsampled_data_train.npz --test_data_file datasets/fly001_reconstructed_prepared/fly001_reconstructed_final_downsampled_data_test.npz --output_dir "$TF_BASE/run_scaled" --n_images 64 --n_groups 32 --batch_size 4 --gridsize 2 --neighbor_count 7 --max_epochs 1 --do_stitching --quiet |& tee "$TF_BASE/cli/train_tf_phase_c1_scaled.log"`.
6. On success: confirm `$TF_BASE/analysis/forward_parity_debug_tf/{stats.json,offsets.json,*.png}` exist, compute `shasum "$TF_BASE/analysis/forward_parity_debug_tf/stats.json"` and append the sha + env note to `$HUB/analysis/artifact_inventory.txt` and `$HUB/summary.md`.
7. On failure: create `$TF_BASE/red/blocked_<timestamp>_tf_translation_guard.md` summarizing the guard + CLI failures, then reference it from the inventory/summary instead.

## Tasks
- tests/tf_helper/test_translation_shape_guard.py::test_non_xla_translation_guard — Record a new GREEN log under `$HUB/green/` or emit a blocker if the selector fails.
- scripts/training/train.py::main — Execute the scaled TensorFlow rerun with the recorded env exports, tee CLI output to `$TF_BASE/cli/train_tf_phase_c1_scaled.log`, and produce `forward_parity_debug_tf/*` on success.
- plans/active/FIX-PYTORCH-FORWARD-PARITY-001/reports/.../analysis/artifact_inventory.txt::Phase C1 — TF scaled baseline — Update the inventory/summary with either the stats sha1 + env capture or the blocker pointer so reviewers know the outcome.

## Selector
```
pytest tests/tf_helper/test_translation_shape_guard.py::test_non_xla_translation_guard -vv | tee "$HUB/green/pytest_tf_translation_guard.log"
```
Pass when the selector exits 0 and the log is saved to `$HUB/green/`. If it fails, do not rerun; immediately create `$TF_BASE/red/blocked_<timestamp>_tf_translation_guard.md` referencing this log.

## Artifacts
- plans/active/FIX-PYTORCH-FORWARD-PARITY-001/reports/2025-11-13T000000Z/forward_parity/green/pytest_tf_translation_guard.log
- plans/active/FIX-PYTORCH-FORWARD-PARITY-001/reports/2025-11-13T000000Z/forward_parity/tf_baseline/phase_c1_scaled/cli/train_tf_phase_c1_scaled.log
- plans/active/FIX-PYTORCH-FORWARD-PARITY-001/reports/2025-11-13T000000Z/forward_parity/analysis/artifact_inventory.txt

Refs:
Summary: plans/active/FIX-PYTORCH-FORWARD-PARITY-001/summary.md
Plan: plans/active/FIX-PYTORCH-FORWARD-PARITY-001/implementation.md
Selector: tests/tf_helper/test_translation_shape_guard.py::test_non_xla_translation_guard
