# Multi-Arm Studies with `ptycho_study`

Generic Hydra-based study layer for `scripts/simulation/synthetic_pipeline.py`.
Ported from the `fno-stable` study layer (design contract:
`docs/superpowers/specs/2026-08-07-hydra-study-runner-design.md` on that
branch).

## Concept

A study is a directory `studies/<name>/conf/` in the runner's config schema:

- `config.yaml` — keys shared by every arm, plus a `study` node (wrapper
  settings: runner location, sentinel, output root) and a `hydra` node
  (run/sweep directory layout).
- One subdirectory per ablation axis (`family/`, `profile/`, ...) holding
  small delta files. Every delta file starts with `# @package _global_`.
- Single-key axes need no files — sweep the dotted key directly
  (`model.architecture=cnn,fno`).

`ptycho_study` composes base + selected deltas + CLI overrides, writes the
resolved config to `<arm>/arm.yaml`, and runs
`python <runner> --config arm.yaml` with `cwd` at the runner root. The runner
performs all validation on the composite.

## Commands

The `ptycho_study` console script becomes available once the package is
installed; before then, `python -m ptycho.workflows.study_runner` is the
equivalent invocation.

Single arm, with an ad-hoc override (the delta is recorded in
`.hydra/overrides.yaml`):

    ptycho_study --config-dir studies/<name>/conf --config-name config \
        family=lines profile=baseline model.architecture=cnn

Full matrix (`-m` = multirun; cartesian product, serial execution):

    ptycho_study --config-dir studies/<name>/conf --config-name config -m \
        family=lines,speckle profile=baseline,ablated \
        model.architecture=cnn,fno

Collation:

    python scripts/studies/collate_study_metrics.py <study_output_root>

Comparison figure (one row per arm, six columns, rendered from each arm's
`reconstruction.npz` with per-family shared color limits; fails if phase
panels carry no structure):

    python scripts/studies/render_study_comparison.py <study_output_root>

## Semantics

- **Arm directories:** sweeps write to `hydra.sweep.dir/subdir`; single runs
  to `hydra.run.dir` (timestamped `adhoc/` unless overridden). Every swept
  key must appear in the `hydra.sweep.subdir` template, or distinct arms
  collide on one directory and the sentinel check skips the later ones.
- **Resume:** an arm whose `study.sentinel` (default
  `reconstruction/metrics.json`) exists is skipped. Rerunning an aborted
  sweep command resumes it.
- **Failure:** a failed arm does not stop the sweep — Hydra's launcher runs
  the remaining arms and re-raises the first failure at sweep end (nonzero
  exit). Completed arms keep their artifacts; rerun the same command to
  resume. A failed arm keeps no sentinel but may keep partial artifacts,
  which the runner refuses to build on (partial or identity-mismatched
  state fails fast) — delete that arm's directory before the rerun. Full
  runner output: `<arm>/runner.log`.
- **Provenance per arm:** `.hydra/{config,overrides,hydra}.yaml`,
  `study_provenance.json` (git commit + dirty flag of the runner root, runner
  SHA-256, overrides, interpreter), `arm.yaml` (directly rerunnable via
  `python scripts/simulation/synthetic_pipeline.py --config arm.yaml`).
- **Simulate-only runs** must override the sentinel:
  `'workflow.stages=[simulate]' study.sentinel=datasets/manifest.json`. Note
  that shrinking `simulation.train_patterns`/`test_patterns` for a smoke also
  requires shrinking `training.train_raw_selection`, `training.training_groups`,
  and `training.validation_groups` in step, or the runner rejects the composite.
- **Shared datasets** (`study.shared_datasets=true`): arms sharing
  `(family_name, measurement_domain)` reuse one simulated dataset via a
  `datasets` symlink; the first arm per key simulates, later arms skip the
  simulate stage. Serial sweeps only. Default off.
- **Pinning a worktree:** set `study.runner_root` to the worktree path; the
  arm subprocess runs from there and provenance records its git state.

## Defining a new study

1. `mkdir -p studies/<name>/conf/<axis>` and write `config.yaml` in the
   runner's `--config` schema (see `docs/CONFIGURATION.md` for the key
   surface), plus `study.name`, `study.output_root`, and the `hydra`
   run/sweep layout.
2. One `# @package _global_` delta file per axis value.
3. Smoke one arm with tiny overrides (small `train_patterns`, `epochs=1`);
   remember to reduce the `training.*` selection and group counts alongside
   the pattern counts.
4. Sweep with `-m`. Add a conf-equivalence or composition test under
   `tests/studies/` if the study feeds a paper or gate.
