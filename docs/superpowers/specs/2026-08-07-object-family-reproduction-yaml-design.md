# Synthetic Object-Family Reproduction Design

## Scope

This design governs the object-family reproduction harness under
`.artifacts/integration/object_family_n128_seed3_20ep_20260807_reproduction/`
and the minimum `ptycho_synthetic` extension needed to generate deterministic
DeadLeaves data. The completed historical `study.yaml`, `preflight/`, and
`full/` artifacts remain immutable.

## Goal

Provide one compact recipe and one top-level command per phase that generates
its own datasets, validates them, trains the requested matrix through
`ptycho_synthetic`, reconstructs, evaluates, and collates the result. The new
recipe is a semantic recreation of the historical study; it does not promise
historical NPZ hashes, identical stochastic draws, or identical metrics.

## Synthetic Runner Boundary

`ptycho_synthetic` is the execution boundary for simulation, training, strict
reload, reconstruction, and evaluation. `grid_lines_torch_runner.py` is not
used.

The runner selects object generation through one registry keyed by configured
`object.kind` plus `object_recipe`; it has no dataset-specific execution path.
It must register two deterministic object recipes:

- `lines-object-v1`, unchanged; and
- `dead-leaves-object-v1`, defined as
  `abs(create_dead_leaves((392, 392), args, rng=<seeded Generator>))` with
  `max_iters=700`, `r_min_frac=0.02`, `r_max_frac=0.18`, and
  `r_sigma=3`, followed by `diffsim.dummy_phi(amplitude)`.

Object kind and recipe identity are inseparable parameters. An unsupported or
contradictory pair fails before simulation. Every registered producer receives
the same explicit RNG interface. The manifest records the selected recipe,
producer symbols, source commit, and realized object-array hash.
Manifest consumers derive the expected producer symbols from that same
kind/recipe registry; they do not assume the Lines producer.

`SimulationConfig` and the producer registry remain authoritative for object
identity. The protected TensorFlow raw-data leaf receives an already
materialized `objectGuess`, so its scoped compatibility projection uses the
legacy `generic` source label; canonical kind and recipe are never recovered
from or selected by `params.cfg`.

## Authored Recipe

`reproduce.yaml` contains only executable choices:

- `source`: source checkout and synthetic runner;
- `output`: one path relative to the YAML;
- `simulation`: shared N=128, seed=3, raster, probe, photon, and split sizes;
- `families`: configured object-kind and versioned-recipe selections;
- `matrix`: ordered families, architectures, and training profiles;
- `preflight`: the two representative rows;
- `model`, `training`, and `inference`: shared explicit values;
- `profiles`: legacy MAE, legacy Poisson, and CI Poisson deltas; and
- `collation.crop_border`.

It contains no prose, historical paths, hashes, commit pins, lineage records,
duplicated phase paths, or derived filenames.

The train and test rasters contain 4,489 (67 x 67) and 729 (27 x 27) patterns.
The historical training archive contained two 67 x 67 objects; the canonical
flat-acquisition runner owns one truth canvas per split. Consequently this
recipe preserves the per-object scan geometry rather than the historical
8,978-row aggregate.

## One-Command Phase Execution

The user-facing commands are:

```bash
python run_study.py --config reproduce.yaml --phase preflight
python run_study.py --config reproduce.yaml --phase full
```

Dataset preparation is not a separate user step. The harness expands the
selected rows and launches one ordinary, complete `ptycho_synthetic` invocation
per row with stages `simulate,train,reconstruct,evaluate`. Each arm therefore
owns its datasets and all downstream artifacts. The harness does not resolve a
shared dataset pool, synthesize stage-completion records, copy simulation
roots, or bypass runner validation.

Identical family/measurement recipes are deterministic. Existing lower-level
memoization may accelerate their repeated simulation, but correctness and
completion do not depend on a cache hit.

## Dataset Validation

Historical SHA-256 values are provenance evidence only. Acceptance requires:

- a complete synthetic dataset manifest whose simulation identity matches the
  resolved family and measurement contract;
- the exact flat-acquisition field set and expected split counts;
- finite diffraction, coordinate, object, and probe arrays;
- N=128 detector and probe dimensions;
- nonnegative detector values;
- raster geometry of 67 x 67 train and 27 x 27 test;
- matching train/test transformed probes;
- `normalized_amplitude` arrays for legacy simulations and
  `count_intensity` arrays for CI simulations;
- the expected object recipe in the manifest; and
- different realized object hashes for Lines and DeadLeaves under the same
  measurement contract.

For arms with the same family and measurement contract, validation also
requires equal manifest dataset identities. This detects accidental recipe or
stochastic drift without introducing a shared-data lifecycle.

## Paths And Outputs

For `output: output/`, phase results are written beneath the YAML directory:

```text
output/
├── preflight/
│   ├── matrix_results/
│   └── preflight_comparison_table.png
└── full/
    ├── matrix_results/
    └── full_comparison_table.png
```

Dry runs resolve and print simulation and arm commands without creating these
directories. A real phase refuses an existing phase root. The collator writes
only beneath that phase root and creates no duplicate `tmp/` copy.

## Runtime Evidence And Failure Behavior

Runtime observations are generated outside the YAML. The harness records the
source commit, runner and harness hashes, generated dataset manifests, resolved
arm configurations, and runner invocations. It rejects a dirty source tree,
source mutation, incomplete or incompatible simulation stages, dataset
mutation, failed stages, non-finite metrics, and missing reconstruction or
comparison artifacts.

## Verification

Focused tests must establish:

1. deterministic Lines behavior is unchanged and deterministic DeadLeaves
   generation carries the correct recipe identity;
2. the compact YAML expands to the ordered two-row preflight and twelve-row
   full matrix;
3. every selected row expands to one complete synthetic-runner invocation;
4. dataset validation rejects schema, count, units, recipe, and family errors;
5. compatible arms produce equal deterministic dataset identities;
6. a no-GPU fixture proves the top-level phase includes simulation, arm
   execution, validation, and collation; and
7. the immutable historical evidence trees remain unchanged.
