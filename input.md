Mode: none
Focus: FIX-PYTORCH-FORWARD-PARITY-001
Selector: tests/tf_helper/test_translation_shape_guard.py::test_non_xla_translation_guard

Overview:
Phase C1d still lacks TensorFlow evidence—the scaled baseline slot under `$HUB/tf_baseline/phase_c1_scaled/` is empty and the CLI log is 0 bytes—so we need to prove the guard is still GREEN and then rerun the scaled TF training command with XLA disabled. Run the guard selector first and tee its output into the hub so we have a current pass/fail record alongside the env captures. Then export the documented env vars (`AUTHORITATIVE_CMDS_DOC`, `HUB`, `TF_XLA_FLAGS="--tf_xla_auto_jit=0"`, `USE_XLA_TRANSLATE=0`) and re-execute `scripts/training/train.py` with the scaled dataset parameters to produce `analysis/forward_parity_debug_tf/{stats.json,offsets.json,pngs}` plus a non-empty CLI log. When the CLI finishes, either update `analysis/artifact_inventory.txt` + `summary.md` with a new “Phase C1 — TF scaled baseline” section and sha1 of the stats file, or capture a fresh blocker under `$HUB/tf_baseline/phase_c1_scaled/red/blocked_<timestamp>_tf_translation_guard.md` if the run still fails (include the guard log reference in that blocker).

### Workload Spec
## Goal
Capture Phase C1d TensorFlow evidence by (a) re-running the non-XLA translation guard selector to verify the shipped fix still passes and (b) executing the scaled TF training CLI with the prescribed env variables so that `$HUB/tf_baseline/phase_c1_scaled/analysis/forward_parity_debug_tf/` finally contains stats/offsets/pngs plus a non-empty CLI log. Treat the CLI failure scenario as a first-class outcome: immediately record the stack trace and guard-log pointer under `red/blocked_<timestamp>.md` instead of leaving partial artifacts, and leave clear breadcrumbs in the inventory/summary either way. Finalize by refreshing `analysis/artifact_inventory.txt` with a “Phase C1 — TF scaled baseline” section (parameters, env capture, sha1) or by linking the new blocker if the CLI aborts.

## Contracts
- docs/specs/spec-ptycho-workflow.md:46 — Reassembly semantics must stay backend-consistent (object.big default) so the TF rerun needs gridsize 2 offsets + full debug bundle before C2/C3 comparisons resume.
- docs/specs/spec-ptycho-runtime.md:29 — USE_XLA flags define the authorized translation path; this rerun must set `TF_XLA_FLAGS="--tf_xla_auto_jit=0"` and `USE_XLA_TRANSLATE=0` to exercise the guarded non-XLA code path captured in TF-NON-XLA-SHAPE-001.

## Interfaces
- **tests/tf_helper/test_translation_shape_guard.py::test_non_xla_translation_guard** — File: tests/tf_helper/test_translation_shape_guard.py:46-87. Creates mismatched batch dimensions (flattened images vs. original translations) and asserts `translate_core` falls back to `_translate_images_simple` without crashing when XLA is disabled.
  - Inputs: fake tensors shaped `(gridsize^2, N, N, 1)` for images and `(1, 2)` for translations; uses `patch.object(tf_helper, 'should_use_xla', return_value=False)` to force the guarded path.
  - Invariants: selector must stay GREEN; failures indicate the guard regressed and need immediate blocker documentation before rerunning the CLI.
- **scripts/training/train.py::main** — File: scripts/training/train.py:300-420. Entry point for the TF training CLI that parses args, updates `params.cfg` via `update_legacy_dict`, loads the fly001 train/test NPZ files, and orchestrates training/stitching via `run_cdi_example_with_backend`.
  - Inputs for this loop: `--backend tensorflow --train_data_file datasets/fly001_reconstructed_prepared/fly001_reconstructed_final_downsampled_data_train.npz --test_data_file ..._test.npz --output_dir "$TF_BASE/run_scaled" --n_images 64 --n_groups 32 --batch_size 4 --gridsize 2 --neighbor_count 7 --max_epochs 1 --do_stitching --quiet`.
  - Invariants: with `TF_XLA_FLAGS="--tf_xla_auto_jit=0"` and `USE_XLA_TRANSLATE=0`, the run must emit `analysis/forward_parity_debug_tf/{stats.json,offsets.json,png grids}` plus log lines showing epoch completion; on failure, capture stderr/stdout to the hub log and file a blocker immediately.
- **plans/active/FIX-PYTORCH-FORWARD-PARITY-001/reports/2025-11-13T000000Z/forward_parity/analysis/artifact_inventory.txt::Phase C1 — TF scaled baseline** — Inventory file that reviewers consult; add a new section detailing the scaled parameters, env captures, sha1 of `forward_parity_debug_tf/stats.json`, and links to guard/CLI logs, or record the blocker path if the CLI failed so downstream phases understand the state.

## Pseudocode
1. `export AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md`
2. `export HUB=$PWD/plans/active/FIX-PYTORCH-FORWARD-PARITY-001/reports/2025-11-13T000000Z/forward_parity`
3. `pytest tests/tf_helper/test_translation_shape_guard.py::test_non_xla_translation_guard -vv | tee "$HUB/green/pytest_tf_translation_guard.log"`
4. `export TF_BASE="$HUB/tf_baseline/phase_c1_scaled"; mkdir -p "$TF_BASE"/{cli,analysis,green,red}`
5. `export TF_XLA_FLAGS="--tf_xla_auto_jit=0"; export USE_XLA_TRANSLATE=0`
6. `python scripts/training/train.py --backend tensorflow --train_data_file ...train.npz --test_data_file ...test.npz --output_dir "$TF_BASE/run_scaled" --n_images 64 --n_groups 32 --batch_size 4 --gridsize 2 --neighbor_count 7 --max_epochs 1 --do_stitching --quiet |& tee "$TF_BASE/cli/train_tf_phase_c1_scaled.log"`
7. If CLI succeeded: ensure `$TF_BASE/analysis/forward_parity_debug_tf/{stats.json,offsets.json,pngs}` exist, run `shasum` on `stats.json`, append new inventory section + summary note.
8. If CLI failed: create `$TF_BASE/red/blocked_<timestamp>_tf_translation_guard.md` with the stack trace + guard log reference instead of partial artifacts, and update inventory/summary to point to the blocker.

## Tasks
- tests/tf_helper/test_translation_shape_guard.py::test_non_xla_translation_guard — Run the selector with `-vv`, tee output to `$HUB/green/pytest_tf_translation_guard.log`, and stop if it fails (log blocker under `$TF_BASE/red/` before proceeding).
- scripts/training/train.py::main — Execute the scaled TF command with the documented env vars and dataset paths so `$TF_BASE/run_scaled/` contains the fresh training outputs plus `analysis/forward_parity_debug_tf/`; capture stdout/stderr to the existing CLI log.
- plans/active/FIX-PYTORCH-FORWARD-PARITY-001/reports/2025-11-13T000000Z/forward_parity/analysis/artifact_inventory.txt::Phase C1 — TF scaled baseline — Append or update the section summarizing the guard log, CLI log, env capture, scaled parameters, and sha1 (or point to the new blocker if the CLI still fails).

## Selector
```
pytest tests/tf_helper/test_translation_shape_guard.py::test_non_xla_translation_guard -vv | tee "$HUB/green/pytest_tf_translation_guard.log"
```
- Pass: selector exits 0 and the tee’d log lives under `$HUB/green/`.
- Blocker protocol: if pytest fails, stop immediately, write `$TF_BASE/red/blocked_<timestamp>_tf_translation_guard.md` with the failure trace + log path, and note the blocker in the inventory instead of re-running the CLI.

## Artifacts
- `plans/active/FIX-PYTORCH-FORWARD-PARITY-001/reports/2025-11-13T000000Z/forward_parity/green/pytest_tf_translation_guard.log`
- `plans/active/FIX-PYTORCH-FORWARD-PARITY-001/reports/2025-11-13T000000Z/forward_parity/tf_baseline/phase_c1_scaled/cli/train_tf_phase_c1_scaled.log`
- `plans/active/FIX-PYTORCH-FORWARD-PARITY-001/reports/2025-11-13T000000Z/forward_parity/analysis/artifact_inventory.txt`

Refs:
Summary: plans/active/FIX-PYTORCH-FORWARD-PARITY-001/summary.md
Plan: plans/active/FIX-PYTORCH-FORWARD-PARITY-001/implementation.md
Selector: tests/tf_helper/test_translation_shape_guard.py::test_non_xla_translation_guard
