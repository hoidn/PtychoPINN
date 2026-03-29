# `lines_256` Architecture Improvement Loop

## Purpose

This document defines the exact loop for an autonomous architecture-improvement agent working on the single-dataset `lines_256` benchmark.

Use this loop when you want `autoresearch`-style branch advancement:

- make one source change or one parameter-only rerun change
- commit the candidate only when the experiment actually changes source
- run one experiment
- record the result in an untracked TSV
- keep the accepted experiment state only if `amp_ssim` improves
- otherwise reset back to the previous accepted source ref only when the candidate changed source

## Dedicated run checkout rule

This workflow has DSL-level git rollback/checkpoint behavior. It should run in a dedicated run checkout rooted at the intended base branch or ref.

Recommended split:

- run the autonomous `lines_256` session in a dedicated disposable checkout
- keep human development and integration work on `fno-stable` or another normal checkout
- if an urgent prompt or doc tweak must affect the next provider step, apply it in the run checkout only, then port it back deliberately later

Do not treat the run checkout as the authoritative integration branch. The loop already records `base_ref` explicitly for reset/restore behavior; a dedicated run checkout keeps that runtime assumption from colliding with unrelated branch history.

## Authoritative inputs

- Dataset note: `docs/studies/lines_256_dataset.md`
- Thin wrapper: `scripts/studies/run_lines_256_arch_experiment.py`

If the dataset note or thin wrapper is missing, stop and report a blocker. Do not guess alternate paths.

## Protected local changes rule

This loop must not stop just because the checkout already has unrelated tracked edits.

At the beginning of each session:

1. run:
   - `git status --short --untracked-files=no`
2. record the tracked dirty file paths from that command as `protected_local_paths`
3. do not stage, edit, reset, or commit any path in `protected_local_paths`
4. if a candidate would need to modify a path in `protected_local_paths`, stop and report that overlap as a blocker
5. preserve every path in `protected_local_paths` across `baseline`, `keep`, `discard`, and `crash` outcomes

This means the session protects the tracked edits that existed when the session started, while candidate cleanup is judged only on:
- whether protected paths are still intact
- whether candidate paths were restored cleanly

New unrelated tracked edits that appear later should be tolerated as long as they do not overlap the candidate path set.

## Fresh baseline rule

Do not depend on an archived baseline from an older study run.

At the beginning of every new experiment session:

1. start from a clean tracked git state
2. create a new `session_id`
3. run a fresh baseline from the current `HEAD`
4. use the thin wrapper `scripts/studies/run_lines_256_arch_experiment.py`
5. rely on the wrapper to pin:
   - train NPZ
   - test NPZ
   - `seed=3`
   - `epochs=20`
   - `scheduler=ReduceLROnPlateau`
   - `plateau_min_lr=2e-4`
   - `N=256`
   - `gridsize=1`
   - `architecture=hybrid_resnet`
   - `probe_mask=off`
   - `torch_mae_pred_l2_match_target=on`
6. use the default hybrid-resnet control:
   - `fno_modes=12`
   - `fno_width=32`
   - `fno_blocks=4`
   - `hybrid_skip_connections=off`
   - `hybrid_downsample_steps=2`
   - `hybrid_downsample_op=stride_conv`
   - `hybrid_resnet_blocks=6`
   - `hybrid_skip_style=add`
7. record that fresh baseline as the first row for the session in the TSV ledger

If the baseline run crashes or does not produce `amp_ssim`, stop. Do not start candidate experiments without a fresh baseline for the current session.

## Results ledger

The experiment ledger is an untracked TSV:

- path: `state/lines_256_arch_improvement/results.tsv`

Keep it untracked. Never stage or commit it. The point is to preserve experiment history even when bad candidate commits are reset away.

If the file does not exist, create it with exactly this header:

```tsv
session_id	timestamp_utc	ref_or_commit	decision	amp_ssim	compared_to_ref	compared_to_amp_ssim	delta_amp_ssim	output_root	comparison_png	command_or_source	notes
```

## Current accepted state and keepability

Within the current `session_id`, the current accepted state is the last row in `state/lines_256_arch_improvement/results.tsv` whose `decision` is:

- `baseline`
- `keep`

A candidate is keepable only if all of these are true:

1. the run exits with code `0`
2. `amp_ssim` is extracted successfully
3. the run matches the current accepted state on:
   - same dataset alias: `lines_256`
   - same underlying dataset paths
   - same requested wrapper seed: `seed=3`
   - same effective randomness contract recorded under `runs/pinn_hybrid_resnet/randomness_contract.json`
   - same epoch budget: `epochs=20`
4. candidate `amp_ssim` is strictly greater than the current accepted state's `amp_ssim`

Equal is not keepable. Lower is not keepable. Missing metric is not keepable.
Runs that exceed the fixed 30-minute budget are not keepable. Record them as a `TIMEOUT` outcome, discard the candidate, and continue.

Do not block just because an internal seed implementation detail differs from the requested wrapper seed, as long as the accepted baseline and candidate publish the same effective randomness contract for the session. Missing randomness metadata, or a mismatch between the candidate and accepted baseline contracts, is a real blocker because it invalidates comparability.

## Metric extraction

For `scripts/studies/run_lines_256_arch_experiment.py` runs, read amplitude SSIM from:

- preferred: `<output_root>/runs/pinn_hybrid_resnet/metrics.json`
- fallback: the run summary artifact if the direct metrics file is absent

When reading `metrics.json`, accept either of these shapes:

- if `amp_ssim` exists, use that value
- otherwise, if `ssim` is a two-element array, use `ssim[0]` as amplitude SSIM

Do not use phase SSIM or any other metric to override the keep/discard decision.

## Output and log paths

For each session:

- session id format: `<timestamp>`
- baseline output root: `outputs/lines_256_arch_improvement/<session_id>_baseline`
- baseline log: `state/lines_256_arch_improvement/<session_id>_baseline.log`
- comparison gallery dir: `outputs/lines_256_arch_improvement/comparison_pngs/<session_id>/`
- baseline gallery PNG: `outputs/lines_256_arch_improvement/comparison_pngs/<session_id>/<session_id>__baseline__compare_amp_phase_probe.png`

For each candidate run in that session:

- output root base: `outputs/lines_256_arch_improvement/`
- source-candidate output root format: `outputs/lines_256_arch_improvement/<timestamp>_<short_commit>`
- source-candidate stdout/stderr log: `state/lines_256_arch_improvement/<timestamp>_<short_commit>.log`
- source-candidate gallery PNG: `outputs/lines_256_arch_improvement/comparison_pngs/<session_id>/<timestamp>_<short_commit>__compare_amp_phase_probe.png`
- parameter-only output root format: `outputs/lines_256_arch_improvement/<timestamp>_<short_base_ref>__cfg`
- parameter-only stdout/stderr log: `state/lines_256_arch_improvement/<timestamp>_<short_base_ref>__cfg.log`
- parameter-only gallery PNG: `outputs/lines_256_arch_improvement/comparison_pngs/<session_id>/<timestamp>_<short_base_ref>__cfg__compare_amp_phase_probe.png`

## Comparison PNG rule

Core experiment success is defined by the launcher result, metrics, and
randomness contract, not by optional reporting visuals.

Do not rely on nested `visuals/` directories alone. The gallery dir is the easy-to-find location for visual inspection across the whole session.

For this loop, the wrapper should still attempt to publish
`visuals/compare_amp_phase_probe.png`.

If probe-inclusive rerendering is unavailable but a plain compare image exists,
the wrapper may publish that plain compare under the explicit probe-inclusive
filename as a best-effort fallback.

If no comparison PNG can be published after an otherwise successful run, do not
reclassify the experiment as a crash. Record `comparison_png=na` and a warning
instead.

If a candidate run truly crashes before any comparison PNG can be produced, set
`comparison_png=na` in the TSV row.

## Candidate viability smoke check

Before handing a candidate to the full `20`-epoch experiment, the agent should first prove that the edited code still supports a cheap end-to-end training and inference path.

The smoke check is not the scored experiment. It is only a viability gate.

Requirements:

- use the same `lines_256` dataset pair and fixed seed / architecture family as the real loop
- preserve the same no-probe-mask and `torch_mae_pred_l2_match_target=on` loss-path contract as the scored run
- exercise both training and inference through the Torch runner stack
- use a much cheaper budget than the scored experiment, e.g. `epochs=1`
- write smoke outputs and logs under the session-local `outputs/lines_256_arch_improvement/` and `state/lines_256_arch_improvement/` trees
- never stage or commit smoke outputs or logs

The agent should use the smoke check to catch and repair obvious breakage caused by its own edit before the full experiment runs.

If the candidate cannot be made smoke-green after a small number of focused fixes, stop and report a blocker rather than handing a broken candidate to the expensive `20`-epoch run.

## Exact loop

LOOP FOREVER:

1. Read `docs/studies/lines_256_dataset.md` and this loop document.
2. Check tracked git state with:
   - `git status --short --untracked-files=no`
3. Record the tracked dirty file list from step 2 as `protected_local_paths`.
4. Ensure `state/lines_256_arch_improvement/results.tsv` exists. If not, create it with the exact header above.
5. Create a new `session_id`.
6. Regenerate a fresh baseline from the current `HEAD` using the thin wrapper and default control.
7. If a baseline comparison PNG is available, publish it into the session gallery dir. If not, continue with `comparison_png=na` and a warning.
8. Append one `baseline` row for that `session_id`.
9. Read the current accepted state from the last TSV row in that `session_id` with `decision=baseline|keep`.
10. Choose the next coherent source or run-configuration hypothesis against that accepted state.
11. If the candidate is source-changing, make one coherent code change within the allowed editable surface and restrict edits to existing files that are not in `protected_local_paths`.
12. If the candidate is parameter-only, leave tracked source files unchanged and express the hypothesis through the smoke and scored run commands.
13. Run a cheap end-to-end smoke check for the prepared candidate and fix obvious breakage if needed.
14. If the candidate cannot be made smoke-green without guessing or thrashing, stop and report a blocker.
15. If the candidate is source-changing, record the exact staged candidate file list as `candidate_paths`.
16. Stage only the intended source changes for source-changing candidates. Never stage `state/` or `outputs/`.
17. Record `base_ref` as the accepted git sha from `accepted_state.json`.
18. If the candidate is source-changing, create exactly one candidate commit after the candidate is smoke-green.
19. If the candidate is parameter-only, create no commit and keep tracked source files unchanged.
20. Run exactly one `lines_256` experiment with the thin wrapper and the same fixed wrapper budget as the session baseline.
21. If a candidate comparison PNG is available, publish it into the session gallery dir. If not, continue with `comparison_png=na` and a warning.
22. Verify the candidate published `runs/pinn_hybrid_resnet/randomness_contract.json` and that it matches the accepted baseline's effective randomness contract for the session.
23. Read candidate `amp_ssim`.
24. Append exactly one TSV row for the candidate.
25. If candidate `amp_ssim` is strictly greater than the accepted state's `amp_ssim`, update the accepted state:
    - for a source-changing candidate, leave the commit in place and advance `accepted_ref`
    - for a parameter-only candidate, keep `accepted_ref` unchanged and update only the accepted run configuration and metric
26. If a source-changing candidate `amp_ssim` is equal or lower, append the `discard` row first, then run:
    - `git reset --mixed <base_ref>`
    - `git restore --source=<base_ref> --staged --worktree -- <candidate_paths>`
27. If a source-changing candidate run exceeded the fixed 30-minute budget, append the `timeout` row first, then run:
    - `git reset --mixed <base_ref>`
    - `git restore --source=<base_ref> --staged --worktree -- <candidate_paths>`
28. If a source-changing candidate run crashed, append the `crash` row first, then run:
    - `git reset --mixed <base_ref>`
    - `git restore --source=<base_ref> --staged --worktree -- <candidate_paths>`
29. If a parameter-only candidate is `discard`, `timeout`, or `crash`, append the row and continue without touching git state.
30. If the randomness contract is missing or mismatched, append a `blocked` row and stop instead of trusting the result.
31. Missing or fallback comparison publication should become a warning only. Do not reclassify a scored run as `crash` solely because the optional gallery artifact is unavailable.
32. only `CRASH` should trigger the focused debug path. After a source reset or after a parameter-only crash, attempt one focused crash-debug candidate that addresses the concrete failure and rerun the scored experiment once.
33. If that debugged candidate still crashes, or if no clean crash fix is available, stop and report a blocker instead of looping blindly.
34. After a source-changing discard, timeout, or crash reset, confirm:
    - `git status --short --untracked-files=no`
    - every path from `protected_local_paths` is still present in the tracked-dirty set
    - no candidate path remains dirty after the reset/restore sequence
35. Continue the loop using the last `baseline` or `keep` row from the current `session_id` as the accepted state.

Because each loop iteration records the accepted starting ref as `base_ref`, source-changing candidates can reset and restore against `base_ref` even if unrelated commits are created later while the session is live. Parameter-only candidates never need git rollback because they do not change tracked source files.

## Candidate row format

Append one TSV row per experiment using this schema:

```tsv
session_id	timestamp_utc	ref_or_commit	decision	amp_ssim	compared_to_ref	compared_to_amp_ssim	delta_amp_ssim	output_root	comparison_png	command_or_source	notes
```

Column meanings:

- `session_id`: the current experiment session
- `timestamp_utc`: candidate run timestamp in UTC
- `ref_or_commit`: candidate git commit hash for source-changing candidates, or the accepted base ref for parameter-only reruns
- `decision`: `baseline`, `keep`, `discard`, `timeout`, `crash`, or `blocked`
- `amp_ssim`: candidate amplitude SSIM, or `na` for timeouts/crashes without a metric
- `compared_to_ref`: the current accepted row's `ref_or_commit`
- `compared_to_amp_ssim`: the current accepted row's `amp_ssim`
- `delta_amp_ssim`: `candidate_amp_ssim - compared_to_amp_ssim`, or `na` on metric-free timeouts/crashes
- `output_root`: candidate run output directory
- `comparison_png`: best available comparison PNG in the session gallery dir, or `na` when no optional visual artifact was published
- `command_or_source`: exact run command
- `notes`: one short sentence explaining the change and decision

For a baseline row:

- `ref_or_commit` is the current `HEAD` commit before any candidate changes
- `decision=baseline`
- `compared_to_ref=na`
- `compared_to_amp_ssim=na`
- `delta_amp_ssim=na`
