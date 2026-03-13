# `lines_256` Architecture Improvement Loop

## Purpose

This document defines the exact loop for an autonomous architecture-improvement agent working on the single-dataset `lines_256` benchmark.

Use this loop when you want `autoresearch`-style branch advancement:

- make one code change
- commit it
- run one experiment
- record the result in an untracked TSV
- keep the commit only if `amp_ssim` improves
- otherwise reset back to the previous champion commit

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
5. keep `protected_local_paths` unchanged across `baseline`, `keep`, `discard`, and `crash` outcomes

This means unrelated tracked edits stay in place while the loop operates only on newly changed candidate files.

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
   - `N=256`
   - `gridsize=1`
   - `architecture=hybrid_resnet`
   - `probe_mask=off`
   - `torch_mae_pred_l2_match_target=off`
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

## Current champion and keepability

Within the current `session_id`, the current champion is the last row in `state/lines_256_arch_improvement/results.tsv` whose `decision` is:

- `baseline`
- `keep`

A candidate is keepable only if all of these are true:

1. the run exits with code `0`
2. `amp_ssim` is extracted successfully
3. the run matches the champion on:
   - same dataset alias: `lines_256`
   - same underlying dataset paths
   - same seed policy: `seed=3`
   - same epoch budget: `epochs=20`
4. candidate `amp_ssim` is strictly greater than champion `amp_ssim`

Equal is not keepable. Lower is not keepable. Missing metric is not keepable.

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
- output root format: `outputs/lines_256_arch_improvement/<timestamp>_<short_commit>`
- stdout/stderr log: `state/lines_256_arch_improvement/<timestamp>_<short_commit>.log`
- candidate gallery PNG: `outputs/lines_256_arch_improvement/comparison_pngs/<session_id>/<timestamp>_<short_commit>__compare_amp_phase_probe.png`

## Comparison PNG rule

Every run in the session must publish one comparison PNG into the session gallery dir.

Do not rely on nested `visuals/` directories alone. The gallery dir is the easy-to-find location for visual inspection across the whole session.

The gallery PNG must include:

- ground-truth object views
- predicted object views
- the probe

If the run's built-in `visuals/compare_amp_phase.png` already includes probe panels, copy that file into the session gallery dir.

If the built-in comparison PNG does not include probe panels, generate or rerender a comparison PNG that does include the probe, then place that probe-inclusive PNG in the session gallery dir.

Do not mark a successful run complete until its gallery PNG exists.

If a candidate run crashes before any comparison PNG can be produced, set `comparison_png=na` in the TSV row.

## Exact loop

LOOP FOREVER:

1. Read `docs/studies/lines_256_dataset.md` and this loop document.
2. Check tracked git state with:
   - `git status --short --untracked-files=no`
3. Record the tracked dirty file list from step 2 as `protected_local_paths`.
4. Ensure `state/lines_256_arch_improvement/results.tsv` exists. If not, create it with the exact header above.
5. Create a new `session_id`.
6. Regenerate a fresh baseline from the current `HEAD` using the thin wrapper and default control.
7. Publish the baseline comparison PNG into the session gallery dir.
8. Append one `baseline` row for that `session_id`.
9. Read the current champion from the last TSV row in that `session_id` with `decision=baseline|keep`.
10. Make one coherent code change within the allowed editable surface.
11. Restrict candidate edits to existing files that are not in `protected_local_paths`.
12. Record the exact staged candidate file list as `candidate_paths`.
13. Stage only the intended source changes. Never stage `state/` or `outputs/`.
14. Create exactly one candidate commit.
15. Run exactly one `lines_256` experiment with the thin wrapper and the same fixed wrapper budget as the session baseline.
16. Publish the candidate comparison PNG into the session gallery dir.
17. Read candidate `amp_ssim`.
18. Append exactly one TSV row for the candidate.
19. If candidate `amp_ssim` is strictly greater than champion `amp_ssim`, leave the commit in place. The branch has advanced.
20. If candidate `amp_ssim` is equal, lower, missing, or the run crashed, append the `discard` or `crash` row first, then run:
    - `git reset --mixed HEAD^`
    - `git restore --source=HEAD --staged --worktree -- <candidate_paths>`
21. After a discard, confirm the protected local-change set is unchanged with:
    - `git status --short --untracked-files=no`
22. Continue the loop using the last `baseline` or `keep` row from the current `session_id` as the champion.

Because each loop iteration creates exactly one candidate commit and candidate files must not overlap `protected_local_paths`, `git reset --mixed HEAD^` followed by restore of `candidate_paths` returns the checkout to the previous champion commit while preserving unrelated tracked edits.

## Candidate row format

Append one TSV row per experiment using this schema:

```tsv
session_id	timestamp_utc	ref_or_commit	decision	amp_ssim	compared_to_ref	compared_to_amp_ssim	delta_amp_ssim	output_root	comparison_png	command_or_source	notes
```

Column meanings:

- `session_id`: the current experiment session
- `timestamp_utc`: candidate run timestamp in UTC
- `ref_or_commit`: candidate git commit hash
- `decision`: `baseline`, `keep`, `discard`, or `crash`
- `amp_ssim`: candidate amplitude SSIM, or `na` for crashes without a metric
- `compared_to_ref`: the champion row's `ref_or_commit`
- `compared_to_amp_ssim`: the champion row's `amp_ssim`
- `delta_amp_ssim`: `candidate_amp_ssim - compared_to_amp_ssim`, or `na` on metric-free crashes
- `output_root`: candidate run output directory
- `comparison_png`: probe-inclusive comparison PNG in the session gallery dir, or `na` for crashes with no visual artifact
- `command_or_source`: exact run command
- `notes`: one short sentence explaining the change and decision

For a baseline row:

- `ref_or_commit` is the current `HEAD` commit before any candidate changes
- `decision=baseline`
- `compared_to_ref=na`
- `compared_to_amp_ssim=na`
- `delta_amp_ssim=na`
