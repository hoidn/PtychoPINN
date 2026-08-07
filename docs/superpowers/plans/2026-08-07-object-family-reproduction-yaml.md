# Synthetic Object-Family Reproduction Implementation Plan

> **For agentic workers:** use `superpowers:subagent-driven-development` and
> `superpowers:test-driven-development`; do not create a worktree.

**Goal:** Make one compact YAML and one command per phase generate, validate,
train, reconstruct, evaluate, and collate the Lines/DeadLeaves matrix through
`ptycho_synthetic`.

**Design:**
`docs/superpowers/specs/2026-08-07-object-family-reproduction-yaml-design.md`

## Files

- Modify in the existing `fno-stable` checkout:
  `ptycho/simulation/object_producers.py`,
  `ptycho/simulation/flat_acquisition.py`,
  `ptycho/nongrid_simulation.py`,
  `ptycho/workflows/synthetic_config.py`,
  `ptycho/workflows/synthetic_pipeline.py`, the synthetic CLI, focused tests,
  and their configuration/runner guides.
- Modify in the artifact harness: `reproduce.yaml`, `run_study.py`,
  `collate_results.py`, `test_run_study.py`, and `check_immutable.sh`.
- Preserve: historical `study.yaml`, `preflight/`, and `full/` trees.

## Task 0: Immutable Baseline

- [x] Record the existing recursive manifests:

```text
study.yaml  files=1     bytes=5020        sha256=93ccfb4ec91ccd542282ec211f7db4a662139f04e4e2dd5c23ca6d0c527ce89f
preflight   files=575   bytes=562022019   sha256=8f7b9f5e5d919fd8354d4e5b35218fb2005e65e61cc29a184ce87e9e247431da
full        files=3385  bytes=3374308976  sha256=22f54f9c47c181ab42198f1c7496136254db19a8880501bb18b17217ebb75762
```

## Task 1: Compact Recipe

- [x] Replace dataset paths with shared simulation values and two object-family
  declarations.
- [x] Point `source.runner` at `scripts/simulation/synthetic_pipeline.py`.
- [x] Preserve every applicable architecture, optimizer, scheduler, loss, and
  physics choice explicitly; omit historical/provenance fields.
- [x] Assert the YAML has no hashes, historical paths, prose, or derived output
  filenames.

## Task 2: DeadLeaves In The Synthetic Runner (TDD)

- [x] Add failing tests for generic kind/recipe dispatch, deterministic
  `dead-leaves-object-v1`, manifest identity, mismatch rejection, and unchanged
  Lines output.
- [x] Implement one seeded producer registry and register Lines and DeadLeaves;
  do not add a dataset-specific execution path.
- [x] Make dataset-manifest validation derive expected producer identity from
  the same kind/recipe registry.
- [x] Run the focused flat-acquisition, synthetic-config, and CLI tests.

## Task 3: Study Expansion And Dataset Validation (TDD)

- [x] Replace grid-runner command-equivalence tests with failing tests for the
  two/twelve row expansions and one complete synthetic invocation per row.
- [x] Add failing tests for legacy/CI simulation config construction.
- [x] Add failing dataset-fixture tests covering fields, counts, shapes,
  finiteness, units, recipe identity, probes, raster geometry, and family
  distinction.
- [x] Implement compact loading, relative phase outputs, simulation identity
  derivation, arm-config generation, and semantic validation.

## Task 4: End-To-End Arm Execution (TDD)

- [x] Add a no-GPU fixture proving each selected row launches exactly one
  `ptycho_synthetic` invocation containing all four stages.
- [x] Make `--phase preflight` and `--phase full` run their complete arms with
  no separate dataset-resolution or preparation phase.
- [x] Validate equal dataset identities across arms that share family and
  measurement contract; allow lower-level memoization without depending on it.
- [x] Keep `--dry-run` side-effect free while printing every complete arm
  command/configuration.

## Task 5: Synthetic Results Collation (TDD)

- [x] Update fixtures and the collator for each arm's
  `reconstruction/reconstruction.npz`, `metrics.json`, `comparison.png`, and
  resolved workflow.
- [x] Derive phase figure names and write summary/completion JSON only under the
  phase root.
- [x] Verify finite amplitude/phase MAE and SSIM for every selected arm.

## Task 6: End-To-End Verification

- [x] Run focused unit tests and Python compilation.
- [x] Run both dry-run commands and confirm 2/12 complete synthetic runs.
- [x] Run a no-GPU orchestration fixture proving one top-level invocation owns
  simulation, validation, arm execution, and collation.
- [x] Run one real preflight row per family before any full matrix launch.
- [x] If the preflight passes, launch the full phase in tmux with exact PID
  tracking and required-artifact checks.
- [x] Recompute all three immutable historical manifests and confirm exact
  equality.
- [x] Review and commit the product-runner change separately from the root
  design/adapter change; do not include user-owned dirty paths.

## Result

- Source implementation: `99ec5f25d` on `fno-stable`.
- Focused source tests: 211 passed; artifact-harness tests: 15 passed.
- Real execution: 2/2 preflight rows and 12/12 full rows passed semantic
  validation on the recorded clean source commit.
- Historical `study.yaml`, `preflight/`, and `full/` recursive manifests match
  the Task 0 baselines exactly.
