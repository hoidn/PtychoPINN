# Regular CI Dose-Closure Convergence Design

## Purpose

Make deterministic dose closure the default rectangular-scale initialization
for Torch's training-only `ci` profile on `refactor`, `refactor-internal`, and
`fno-stable`. Retire the historical `rect_s1s2_init="data"` behavior completely.

This is a configuration-convergence change, not a new calibration subsystem.
`refactor` and `fno-stable` already contain the reference dose-closure
implementation; `refactor-internal` must adopt that implementation and delete
its older one-batch learned-model calibration.

The sampling and record clauses are amended by
[`2026-08-06-ci-dose-closure-representative-sampling-design.md`](2026-08-06-ci-dose-closure-representative-sampling-design.md).
That amendment replaces the historical first-256 prefix with pinned
fixed-seed uniform logical-slot sampling and makes fresh producers emit v2;
the profile, CLI, and `data`-retirement clauses below remain authoritative.

## Final Contract

All three branches expose the same contract:

| Surface | Behavior |
|---|---|
| Bare `ModelConfig.rect_s1s2_init` | Defaults to `"ones"` |
| Training-only `profile="ci"` | Defaults to `"dose_closure"` |
| Supported values | Exactly `"ones"` and `"dose_closure"` |
| Native Torch CLI | `--rect-s1s2-init {ones,dose_closure}` |
| Historical `"data"` | Rejected, with no alias or migration |

The bare default stays `ones` because a bare model has no resolved CI training
dataset from which to solve a gauge. The `ci` profile has that dataset contract,
so `dose_closure` is an appropriate profile default.

`rect_s1s2_init` remains an overrideable profile default, not one of the five
locked CI fields. A caller may explicitly select `ones` without changing the
count-intensity forward or loss contract.

## Resolution and CLI Precedence

Resolution preserves ordinary precedence:

```text
bare model default < selected profile default < explicit caller value
```

The native CLI parser uses `None` as the argparse default for
`--rect-s1s2-init` and adds the field to the factory override only when the user
supplied the flag. Therefore:

- `--profile ci` resolves to `dose_closure`;
- `--profile ci --rect-s1s2-init ones` resolves to `ones`; and
- no profile and no flag retains the bare `ones` behavior.

Structured synthetic configuration continues to use
`model.rect_s1s2_init`; programmatic training continues to use the flat factory
override. Existing CLIs that expose this choice use the same two spellings.

## Runtime Behavior

The pure selection in `ptycho_torch.rect_s1s2_sampling`, record validation in
`ptycho_torch.rect_s1s2_initialization`, and initialization helpers in
`ptycho_torch.workflows.components` on `refactor` are the reference. Do not
create a second solver.

Before fitting, `dose_closure` must:

1. reset all rectangular scalers to `s1=s2=1`;
2. derive the pinned fixed-seed sample of exactly 256 logical detector slots
   from the complete resolved training dataset without consuming ambient RNG
   or the original loader;
3. evaluate the real rectangular forward with a complex unit object over the
   selected logical rows and accumulate only their selected channel masks;
4. accumulate observed and predicted detector sums in float64;
5. solve `gauge = sqrt(observed_sum / predicted_sum)` and fill every `s1` and
   `s2` entry with that value; and
6. preserve module train/eval state and emit the strict fresh
   `rect-s1s2-initialization-v2` record while retaining strict read support for
   historical prefix-era v1 records.

The `ones` path resets the scalers, consumes no training loader, and emits the
existing unit record. Training persists the same record atomically in
`training_summary.json` and returns it in the training result. Supported
distributed execution retains the existing rank-zero publication and barrier.

Dose closure fails clearly for an incoherent/non-rectangular configuration,
missing or amplitude-domain CI fields, invalid shapes or counts, fewer than 256
patterns, or non-positive/non-finite sums or gauge. The short-data error names
the sampled and required counts and points to explicit `ones` initialization.
There is no smaller adaptive solve or automatic fallback.

## Full Retirement of `data`

`refactor-internal` currently computes `data` from one arbitrary batch and the
randomly initialized learned model. That result depends on model seed and batch
composition and is not the deterministic unit-object dose closure.

Convergence removes:

- `data` from config types, profile defaults, CLI choices, fixtures, and docs;
- `PtychoPINN_Lightning.calibrate_rect_s1s2`;
- the workflow's one-batch calibration branch; and
- the `rect_s1s2_calibration` result field.

Semantic validation attached to `ModelConfig` construction is the common
rejection boundary for an unsupported initialization value. Existing ModelSpec,
artifact, sidecar, and checkpoint decoders already reconstruct model identity
through that boundary. The maintained MLflow whole-model loaders bypass
construction by unpickling, so they must call the same validator immediately
after load. Structured mapping adapters may reject through their strict Literal
validation before construction. All paths reject `data`; the common semantic
error names `dose_closure` and `ones` and explains that historical `data`
artifacts need historical code or retraining.

Do not add decoder-specific migration branches, silently translate `data`, or
bump a schema solely because an enum value was removed. Tests should exercise
authored config, ModelSpec/artifact, checkpoint, and MLflow entry points to
prove the common validator is actually reached.

## Branch Application

### `refactor`

- change the regular-CI profile default;
- add native CLI authorship with omitted-versus-explicit forwarding;
- centralize current-value validation so serialized `data` is rejected; and
- update focused tests and public configuration/workflow docs.

### `fno-stable`

Apply the equivalent code and tests, then update the branch's additional
routed core-contract, normalization, command, testing, and index docs.

### `refactor-internal`

Port the focused reference initialization module and workflow helpers into the
branch's current loader/runtime structure. Remove its `data` method and result
shape rather than retaining compatibility. Update internal tests and its full
documentation routing. Do not use a broad branch merge for this port.

Shared code/test commits may be cherry-picked where blobs match. Branch-owned
documentation and the internal runtime port are adapted separately.

## Documentation Ownership

- `docs/CONFIGURATION.md` owns profile defaults and field meaning.
- `docs/workflows/pytorch.md` and the existing command reference own CLI usage.
- Existing core/normalization specs on branches that have them own mathematical
  and persistence semantics.
- Runner READMEs document only runner-specific profiles and flags.
- Indexes route to those authorities without duplicating the contract.

`refactor` must not gain a new `docs/index.md`; `refactor-internal` and
`fno-stable` update their existing indexes.

## Acceptance Evidence

Each branch must prove:

1. regular `ci` resolves to `dose_closure`, explicit `ones` wins, and a bare
   model remains `ones`;
2. native CLI help exposes the canonical choices, omission preserves profile
   resolution, and explicit values reach training;
3. `data` fails at direct config, structured mapping, ModelSpec/artifact,
   checkpoint, and MLflow-load boundaries;
4. initialization tests cover pinned fixed-seed selection, exact 256-slot
   masking, grouped channels, nested subsets, mmap/prebuilt access bounds,
   module-state restoration, invalid inputs, strict v1/v2 records, no-loader
   `ones`, and summary publication; and
5. one focused regular-CI integration with no initialization override persists
   a `dose_closure` initialization record.

Run branch-native focused suites first, then the relevant Torch integration
tests, then each branch's normal comprehensive gate. A final read-only
three-tip comparison confirms the same supported values, default, CLI spelling,
record schema, and `data` rejection policy.

## Non-Goals

- Renaming `ci` or synthetic profiles.
- Changing the five locked CI fields.
- Changing `rect_s1s2_trainable`, optimizer behavior, inference VarPro, or
  dataset-level inference refit.
- Changing the bare `ModelConfig` default.
- Preserving or migrating `data` artifacts.
- Introducing a general migration framework or bumping schema solely for enum
  retirement. The estimator-provenance amendment independently requires v2.
- Establishing a reconstruction-quality threshold or GPU baseline.
