Mode: Perf
Focus: FIX-TF-C1D-SCALED-RERUN-001
Selector: tests/tf_helper/test_translation_shape_guard.py::test_non_xla_translation_guard

Overview:
The TF-NON-XLA-SHAPE-001 blocker is still open because `$HUB/tf_baseline/phase_c1_scaled/analysis/` has never produced `forward_parity_debug_tf/*` and the only CLI log in that slot is the 55 KB failure from 2025-11-14. Re-run the guard selector so we have a fresh GREEN log proving the non-XLA translation fallback still works after the latest PyTorch changes, then execute the scaled TensorFlow CLI with the documented env exports (`AUTHORITATIVE_CMDS_DOC`, `TF_XLA_FLAGS="--tf_xla_auto_jit=0"`, `USE_XLA_TRANSLATE=0`) to mirror the PyTorch Phase B3 dataset slice (`n_images=64`, `n_groups=32`, `gridsize=2`, `neighbor_count=7`). Success means capturing `analysis/forward_parity_debug_tf/{stats.json,offsets.json,png grids}` plus sha1 + hub summary/inventory updates per POLICY-001/CONFIG-001; failure must be logged immediately under `$TF_BASE/red/blocked_<timestamp>_tf_translation_guard.md` with pointers to both the pytest log and the CLI stack trace. Do not edit production code—this pass is purely about producing TensorFlow evidence or a blocker so Phase C can advance again.

### Workload Spec
## Goal
Produce up-to-date TensorFlow guard + scaled CLI evidence (or a documented blocker) so Phase C1d satisfies the reassembly-parity contract. That requires a fresh pytest log in `$HUB/green/`, a non-empty `$TF_BASE/cli/train_tf_phase_c1_scaled.log`, and either a populated `$TF_BASE/analysis/forward_parity_debug_tf/` bundle with sha1 recorded in `analysis/artifact_inventory.txt` or a new `$TF_BASE/red/blocked_<ts>_tf_translation_guard.md` that cites the guard log and CLI failure signature. The CLI must reuse the scaled parameters from the PyTorch study so downstream comparisons remain apples-to-apples.

## Contracts
- docs/specs/spec-ptycho-workflow.md:46 — Backend parity requires TensorFlow reassembly evidence (stats, offsets, patch grids) before continuing the PyTorch forward-parity plan.
- docs/specs/spec-ptycho-runtime.md:15 — Translation runtime paths must obey the USE_XLA_TRANSLATE toggles; this rerun explicitly disables XLA by exporting `TF_XLA_FLAGS="--tf_xla_auto_jit=0"` and `USE_XLA_TRANSLATE=0`.

## Interfaces
- tests/tf_helper/test_translation_shape_guard.py::test_non_xla_translation_guard — File: tests/tf_helper/test_translation_shape_guard.py:46-87. Forces `should_use_xla=False` and asserts `_translate_images_simple` handles mismatched batch dimensions without crashing; this is the sentinel that guards against regressions in the fallback translation path.
- scripts/training/train.py::main — File: scripts/training/train.py:300-420. Parses CLI args, runs `update_legacy_dict(params.cfg, config)` per CONFIG-001, and orchestrates training/inference. For this loop, pass `--backend tensorflow --train_data_file datasets/fly001_reconstructed_prepared/fly001_reconstructed_final_downsampled_data_train.npz --test_data_file ..._test.npz --output_dir "$TF_BASE/run_scaled" --n_images 64 --n_groups 32 --batch_size 4 --gridsize 2 --neighbor_count 7 --max_epochs 1 --do_stitching --quiet` with the env exports described above.
- plans/active/FIX-PYTORCH-FORWARD-PARITY-001/reports/2025-11-13T000000Z/forward_parity/analysis/artifact_inventory.txt::Phase C1 — TF scaled baseline — Inventory entry that reviewers rely on; extend it with the new stats sha1 + env capture (or blocker reference) so Phase C progress is auditable.

## Pseudocode
1. `export AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md` and `export HUB=$PWD/plans/active/FIX-PYTORCH-FORWARD-PARITY-001/reports/2025-11-13T000000Z/forward_parity`.
2. `export TF_BASE="$HUB/tf_baseline/phase_c1_scaled"; mkdir -p "$TF_BASE"/{cli,analysis,green,red}`.
3. `export TF_XLA_FLAGS="--tf_xla_auto_jit=0"; export USE_XLA_TRANSLATE=0; printf 'TF_XLA_FLAGS=%s\nUSE_XLA_TRANSLATE=%s\n' "$TF_XLA_FLAGS" "$USE_XLA_TRANSLATE" | tee "$TF_BASE/cli/env_capture_scaled.txt"`.
4. Run `pytest tests/tf_helper/test_translation_shape_guard.py::test_non_xla_translation_guard -vv | tee "$HUB/green/pytest_tf_translation_guard.log"`; abort to blocker logging if it fails.
5. Execute the scaled CLI: `python scripts/training/train.py ... |& tee "$TF_BASE/cli/train_tf_phase_c1_scaled.log"` (use the exact dataset + knob set above).
6. On success: verify `$TF_BASE/analysis/forward_parity_debug_tf/{stats.json,offsets.json,*png}` exist, compute `shasum` for `stats.json`, and append the findings + env note to `$HUB/analysis/artifact_inventory.txt` and `$HUB/summary.md`.
7. On failure: `printf` the minimal stack trace + guard-log pointer into a new `$TF_BASE/red/blocked_<timestamp>_tf_translation_guard.md`, reference it from `analysis/artifact_inventory.txt`, and update `$HUB/summary.md` with the blocker status.
8. In both cases, ensure `$TF_BASE/cli/train_tf_phase_c1_scaled.log` and `$HUB/green/pytest_tf_translation_guard.log` are non-empty and captured in the hub.

## Tasks
- tests/tf_helper/test_translation_shape_guard.py::test_non_xla_translation_guard — Re-run the selector with `-vv`, tee into `$HUB/green/pytest_tf_translation_guard.log`, and stop immediately to write a blocker if it fails.
- scripts/training/train.py::main — Execute the scaled TensorFlow CLI with the env exports recorded in `env_capture_scaled.txt`, capturing stdout/stderr to `$TF_BASE/cli/train_tf_phase_c1_scaled.log` and generating the `forward_parity_debug_tf` bundle on success.
- plans/active/FIX-PYTORCH-FORWARD-PARITY-001/reports/2025-11-13T000000Z/forward_parity/analysis/artifact_inventory.txt::Phase C1 — TF scaled baseline — Extend the inventory (and `$HUB/summary.md`) with either the stats sha1 + env note or the blocker reference so reviewers can see exactly what happened.

## Selector
```
pytest tests/tf_helper/test_translation_shape_guard.py::test_non_xla_translation_guard -vv | tee "$HUB/green/pytest_tf_translation_guard.log"
```
Pass when the selector exits 0 and the tee’d log lives in `$HUB/green/`. If it fails, do not rerun; immediately author `$TF_BASE/red/blocked_<timestamp>_tf_translation_guard.md` referencing the pytest output and halt the CLI.

## Artifacts
- plans/active/FIX-PYTORCH-FORWARD-PARITY-001/reports/2025-11-13T000000Z/forward_parity/green/pytest_tf_translation_guard.log
- plans/active/FIX-PYTORCH-FORWARD-PARITY-001/reports/2025-11-13T000000Z/forward_parity/tf_baseline/phase_c1_scaled/cli/train_tf_phase_c1_scaled.log
- plans/active/FIX-PYTORCH-FORWARD-PARITY-001/reports/2025-11-13T000000Z/forward_parity/analysis/artifact_inventory.txt

Refs:
Summary: plans/active/FIX-PYTORCH-FORWARD-PARITY-001/summary.md
Plan: plans/active/FIX-PYTORCH-FORWARD-PARITY-001/implementation.md
Selector: tests/tf_helper/test_translation_shape_guard.py::test_non_xla_translation_guard
