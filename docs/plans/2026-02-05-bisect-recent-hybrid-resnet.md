# Bisect Plan (Last 2 Hours)

## Goal
Identify the first commit within the last two hours that causes the `grid_lines_hybrid_resnet` integration regression.

## Constraints
- **No worktrees.** Use `git bisect` in the main working tree only.

## Preconditions
- Working tree clean (no local modifications).
- Submodules checked out to recorded commits.
- Any prior stashes preserved for later restore.

## Steps
1. **Establish time window**
   - Record current UTC time.
   - Compute `since = now - 2 hours`.

2. **List commits in window (oldest → newest)**
   - `git log --since="$since" --reverse --pretty=format:%H`
   - If list is empty, expand window earlier until at least 2 commits exist.

3. **Select candidate good commit**
   - Choose the **earliest commit in the window** as the first candidate good.
   - Run the integration test **twice** at that commit:
     - `python -m pytest -v -m integration tests/torch/test_grid_lines_hybrid_resnet_integration.py`
   - If both pass, mark this commit **good**.
   - If it fails or flips, move to the next earliest commit and repeat.

4. **Start bisect**
   - `git bisect start`
   - `git bisect bad HEAD`
   - `git bisect good <GOOD_SHA>`

5. **Bisect loop**
   - For each checked-out commit:
     - Run the integration test once.
     - Save log: `.artifacts/bisect_recent/test_<sha>.log`.
     - Record result in `.artifacts/bisect_recent/bisect_log.csv`.
   - If a result is flaky (pass/fail), re-run once; if still inconsistent, re-run a third time and record as flaky.

6. **Finish**
   - `git bisect reset`
   - Summarize first bad commit + evidence logs in `.artifacts/bisect_recent/summary.md`.
   - Restore stashes if requested.

## Findings
- Window: 2026-02-05T06:38:48Z → 2026-02-05T08:38:48Z (24 commits).
- Earliest commit in window (verified good twice): `e0e8c1aff8390527e476efc5eb730b73cba53754`.
- **First bad commit:** `0cf58b67cf1825cfc2ab753fb1908f9211a7cf26` — `fix(torch): propagate object_big + coords_relative alias`.
  - Files: `ptycho_torch/data_container_bridge.py`, `ptycho_torch/workflows/components.py`.
- **Last good before first bad:** `2da54da82b5eee4332d5651f5145dd441264cdde`.
- Evidence logs: `.artifacts/bisect_recent/summary.md`, `.artifacts/bisect_recent/bisect_log.csv`, and `test_*.log` under `.artifacts/bisect_recent/`.

## Root Cause Investigation (Phase 1)
- Failure signature: `test_grid_lines_hybrid_resnet_metrics` fails on amp MAE.
  - Example failures on `0cf58b67`: amp MAE `0.1724303` and `0.15841551` vs baseline `0.10942369` + tol `0.02` (threshold `0.12942369`).
  - Evidence logs: `.artifacts/bisect_recent/test_0cf58b67.log`, `.artifacts/bisect_recent/test_0cf58b67_retry.log`.
- Reproducibility: two consecutive failures on `0cf58b67`; last-good `2da54da8` passed the same integration test.
- Change review (commit scope):
  - `ptycho_torch/workflows/components.py`: adds `object_big`, `probe_big`, `pad_object` into `factory_overrides` passed to `create_training_payload`.
  - `ptycho_torch/data_container_bridge.py`: adds `coords_relative` alias to `coords_nominal`.
- Data flow trace (grid-lines integration path):
  - `scripts/studies/grid_lines_torch_runner.py` builds `TrainingConfig` using TF `ModelConfig` defaults (object_big=True, probe_big=True, pad_object=True).
  - `_train_with_lightning` passes `factory_overrides` into `create_training_payload` (`ptycho_torch/config_factory.py`).
  - `create_training_payload` defaults `probe_big=False` when no override is provided, but uses the override when present.
  - Net effect of `0cf58b67` in this path: `probe_big` flips from default `False` (pre-commit) to `True` (post-commit), altering the decoder path in `ptycho_torch/model.py` (`Decoder_last.forward`).
- Preliminary suspect: change in `probe_big` propagation (architectural behavior shift) rather than `coords_relative` alias (grid-lines runner passes a dict with explicit `coords_relative` already).
- Probe-big override experiment:
  - Added env override `TORCH_PROBE_BIG=0` to force probe_big=False in the runner.
  - Result: integration still fails (amp MAE `0.16896562` > `0.12942369` threshold).
  - Evidence: `.artifacts/object_big_relative_offsets/pytest_grid_lines_hybrid_resnet_probe_big_false_retry.log`.
- Object-big override experiment:
  - Added env override `TORCH_OBJECT_BIG=0` to force object_big=False in the runner.
  - Result: integration **passes** (3/3 tests).
  - Evidence: `.artifacts/object_big_relative_offsets/pytest_grid_lines_hybrid_resnet_object_big_false.log`.
- Force-zero relative offsets experiment:
  - Added env override `TORCH_FORCE_ZERO_RELATIVE_OFFSETS=1` to zero coords_relative in the runner.
  - Result: integration still fails (amp MAE `0.1633376` > `0.12942369` threshold).
  - Evidence: `.artifacts/object_big_relative_offsets/pytest_grid_lines_hybrid_resnet_force_zero_relative_retry.log`.
- Skip reassembly/extraction experiment:
  - Added env override `TORCH_SKIP_OBJECT_BIG_REASSEMBLY=1` to bypass reassembly/extract when `C==1`.
  - Result: integration **passes** (3/3 tests).
  - Evidence: `.artifacts/object_big_relative_offsets/pytest_grid_lines_hybrid_resnet_skip_reassembly.log`.
- Helper inspection notes (no code changes):
  - `ptycho_torch/helper.py:reassemble_patches_position_real` always applies a central **N/2 mask** (slice `N//4 : N//4 + N//2`) before normalization, even when `C==1`.
  - For `gridsize=1` (C=1), this acts as a windowing operation (zeros outside center), which is not a no-op and could explain regression when `object_big=True`.
  - `extract_channels_from_region` translates back and crops, but it cannot recover data zeroed by the mask.
- Torch reassembly normalization parity change:
  - Updated `reassemble_patches_position_real` to mirror TF `mk_norm`: use `non_zeros + 0.001` and **no hard mask** on the output.
  - New unit test: `tests/torch/test_reassemble_patches_position_real_c1.py`.
  - Evidence (RED): `.artifacts/object_big_relative_offsets/pytest_reassemble_c1_red.log`.
  - Evidence (GREEN): `.artifacts/object_big_relative_offsets/pytest_reassemble_c1_green.log`.
- Integration after helper change (object_big still True) regressed further:
  - amp MAE `0.20412715` vs threshold `0.12942369`.
  - Evidence: `.artifacts/object_big_relative_offsets/pytest_grid_lines_hybrid_resnet_after_fix.log`.
- Fix for grid-lines torch parity:
  - Set `object_big=False` in `scripts/studies/grid_lines_torch_runner.py` to match TF grid_lines config.
  - Integration now passes (amp MAE `0.10942369`, phase MAE `0.10495743`).
  - Evidence: `.artifacts/object_big_relative_offsets/pytest_grid_lines_hybrid_resnet_object_big_false_default.log`.
