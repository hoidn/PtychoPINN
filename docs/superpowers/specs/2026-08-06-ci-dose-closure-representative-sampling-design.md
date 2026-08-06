# CI Dose-Closure Representative Sampling Design

## Purpose

Replace dose closure's order-sensitive first-256 prefix with a reproducible,
uniform sample of 256 detector-pattern slots from the resolved training
population. Apply the same behavior to `refactor`, `fno-stable`, and
`refactor-internal`.

This design amends only the sampling and initialization-record clauses of
[`2026-08-05-regular-ci-dose-closure-convergence-design.md`](2026-08-05-regular-ci-dose-closure-convergence-design.md).
That design remains authoritative for profile defaults, supported modes, CLI
precedence, retirement of `rect_s1s2_init="data"`, and the three-branch
convergence order. Within that parent, this design specifically supersedes the
first-256 runtime selection, the v1-only runtime-record requirement, the
corresponding runtime/acceptance references to v1, and the non-goal that ruled
out a schema version change. The new v2 is justified by estimator provenance,
not by enum retirement, and does not introduce a general migration framework.

## Problem

The current reference solver rebuilds an unshuffled loader and consumes the
first 256 `(B, C)` detector slots. That is reproducible and contiguous for
memory-mapped input, but it makes the solved gauge depend on dataset order.
Spatially ordered scans can place one object region in the prefix, so its mean
transmission can dominate a gauge intended to condition the full training
population.

Some prebuilt mmap workflows happen to randomize the train split before the
prefix is read. That incidental ordering is not a sampling contract and is not
shared by inline dictionary loaders.

## Alternatives

### Chosen: fixed-seed uniform detector-slot sample

Select 256 logical detector slots uniformly without replacement from the
resolved training dataset using a private local RNG and fixed seed. This is
population-proportional across source experiments, does not privilege early
dataset indices, and requires no scan of `experiment_id` metadata. Reordering
the dataset changes which physical observations occupy the sampled logical
indices; the fixed dataset identity and fixed logical order remain inputs to
exact replay.

### Rejected: proportional experiment stratification

Stratification could guarantee that every experiment contributes, but it
requires an `O(N_train)` metadata scan through nested training subsets and a
weighted estimator to avoid over-weighting small experiments. Uniform global
sampling already gives every detector slot equal inclusion probability and
matches the current single shared gauge. Reconsider stratification only with a
separate design for per-experiment gauges or evidence that small experiments
must always be represented.

### Rejected: evenly spaced or freshly random samples

Evenly spaced indices can alias periodic grouping or scan order. Drawing from
ambient/global RNG state makes the gauge depend on training seed, rank, and
invocation history. Neither satisfies the reproducibility contract.

## Sampling Contract

`dose_closure` must perform these steps before optimization:

1. Resolve the complete logical training dataset after any train/validation
   split. Validation rows are never eligible.
2. Determine the constant channel count `C` from a private one-row,
   single-process inspection that does not consume the original loader or its
   RNG.
3. Treat each `(logical_row, channel)` pair as one detector-pattern slot. For
   `0 <= logical_row < len(training_dataset)` and `0 <= channel < C`, define
   `flat_slot = logical_row * C + channel`. Invert with
   `logical_row, channel = divmod(flat_slot, C)`. The population size is
   `len(training_dataset) * C`.
4. Fail if the population contains fewer than 256 slots.
5. Use a local `random.Random` instance seeded by the private constant
   `RECT_S1S2_DOSE_CLOSURE_SAMPLE_SEED = 20260806` and draw exactly 256 flat
   slot indices without replacement from `range(population_size)`. Do not read
   or mutate Python's module-global, NumPy, or Torch RNG state.
6. Convert selected slots back to logical row/channel pairs. For a nested
   `torch.utils.data.Subset(parent, indices)`, recursively map logical index `i`
   as `map(parent, indices[i])` until reaching the base dataset. Use that base
   index only as a physical-read sort key; never sample the base dataset
   directly. Distinct logical rows that map to the same base index remain
   distinct population members and retain their multiplicity.
7. Keep every selected channel mask keyed by its logical row. Order reads by
   `(base_index, logical_row)` when a base mmap index is available, otherwise by
   logical row. The row reader may reuse one physical read for duplicate base
   indices only if it applies each logical row's mask separately and counts its
   multiplicity. Physical sorting or reuse must not change sample membership or
   attach a channel mask to a different logical row.
8. Run the existing resolved rectangular forward with `s1=s2=1` and a complex
   unit object. The forward may evaluate unselected channels in a selected
   group row, but only the selected 256 slots contribute to the float64 sums.
9. Solve the unchanged estimator

   ```text
   gauge = sqrt(sum(selected measured counts)
                / sum(selected predicted intensity))
   ```

   and fill every `s1` and `s2` entry with that shared gauge.

Grouping-policy repeats remain distinct logical slots because they are also
distinct members of the resolved training population. Deduplicating raw scan
indices, changing the sample budget, or solving separate experiment gauges is
out of scope.

## Loader and Distributed Behavior

The selection helper is private and loader-agnostic at its boundary. A private
row-reader adapter must provide these behaviors:

- expose the resolved training dataset's logical length;
- inspect `C` from logical row zero without advancing the original loader;
- accept the ordered selected logical-row identities;
- yield batches paired with those same row identities, preserving the existing
  field/probe/scale collation contract; and
- use single-process, non-dropping reads without consuming the original loader,
  shuffle generator, sampler, or worker state.

The maintained adapters support ordinary `torch.utils.data.DataLoader` and
`TensorDictDataLoader`. They may rebuild a private loader from the original
loader's `dataset`, `batch_size`, and `collate_fn`, or use equivalent direct
indexed reads, but the behavioral contract above is authoritative. A loader
that lacks the required dataset/indexing or collation behavior fails clearly;
the implementation must not silently fall back to iterating its shuffled
sampler.

The selected population is the complete resolved training dataset, not a
Lightning `DistributedSampler` shard. Every DDP rank derives the same selection
and gauge from the fixed local seed. Inspection, selected-row reads, and forward
evaluation are rank-local; no collective occurs until the existing rank-zero
atomic publication and all-rank barrier path.

The design accepts sparse mmap reads for at most 256 unique rows as the cost of
removing prefix bias. It does not materialize a full permutation or full
detector array.

## Initialization Record Compatibility

The selected-slot policy changes initialization identity, so fresh producers
write `rect-s1s2-initialization-v2` while retaining the existing five fields:

```json
{
  "schema_version": "rect-s1s2-initialization-v2",
  "mode": "dose_closure",
  "solved_gauge": 1.0,
  "method": "dose_closure_seeded_uniform_unit_object",
  "sampled_patterns": 256
}
```

For `ones`, fresh v2 records retain `unit_default_no_solve`, gauge `1.0`, and
zero sampled patterns. The v2 schema fixes both the sampling policy and seed;
changing either requires another schema version. No user-facing seed field is
added to `ModelConfig`, CLI configuration, `ModelSpec`, or artifact identity.

Readers accept historical v1 records strictly and without rewriting them:

- v1 `dose_closure` requires `method="dose_closure_unit_object"` and denotes
  the historical prefix implementation;
- v2 `dose_closure` requires the new method and denotes fixed-seed uniform
  sampling; and
- both schema versions preserve their existing `ones` invariants.

Fresh runs always produce v2. Historical completed runs, metrics, and v1
summaries remain valid evidence for the code and sampling policy that produced
them; documentation labels them as prefix-era where comparison matters.

## Failure Behavior

Dose closure continues to fail closed for incoherent CI configuration, missing
count-intensity fields, invalid shapes or values, fewer than 256 slots, or
non-positive/non-finite sums and gauge. It additionally fails clearly when:

- the dataset has no rows or a non-positive/inconsistent channel count;
- a selected logical row cannot be mapped through a supported nested subset;
- a rebuilt loader yields rows or channels inconsistent with the selection;
  or
- the sampler does not yield all 256 selected slots exactly once.

There is no fallback to the old prefix, a smaller sample, or `ones`.

## Implementation Boundaries

- `ptycho_torch/rect_s1s2_sampling.py` owns pure flat-slot selection, recursive
  subset mapping, the immutable selected-row/channel plan, and the private
  row-reader adapter contract.
- `ptycho_torch/rect_s1s2_initialization.py` owns v1/v2 record validation and
  the fixed sample-count/seed/schema constants.
- `ptycho_torch/workflows/components.py` owns loader rebuilding, batch/device
  movement, the actual forward evaluation, and summary publication.

The new sampling module keeps selection logic out of the already large workflow
module and can be tested without constructing a model.

## Branch Application

### `refactor`

Implement and validate the reference sampler, v2 record, and workflow changes.
Update the existing dose-closure convergence design and plan plus the current
configuration/workflow and runner guides. Do not add `docs/index.md`.

### `fno-stable`

Port the settled reference symbols and tests. Update the normative CI gauge
section in `docs/specs/spec-ptycho-core.md`, then its normalization,
configuration, workflow, testing, findings, runner, and index surfaces. Preserve
the completed 2026-08-04 prefix-era plan and its recorded metrics as history.

### `refactor-internal`

Fold the v2 seeded sampler directly into the pending convergence port that
replaces `rect_s1s2_init="data"`. Do not first port the prefix solver. Apply the
full internal documentation set; documentation exclusions used for other
branches do not apply here.

Use the existing branch checkouts or serial branch checkouts. Do not create new
Git worktrees and do not use broad branch merges for the internal runtime port.

## Acceptance Evidence

Each branch must prove:

1. the selected set is the pinned fixed-seed set, contains exactly 256 unique
   flat slots, and includes non-prefix regions in the ordering-bias fixture;
2. selection is unchanged by loader shuffle seed, batch size, global RNG state,
   worker settings, or simulated DDP rank;
3. `C=1` and grouped `C=9` cases count exactly 256 selected detector slots with
   no channel-order truncation;
4. nested training subsets never select validation rows and mmap access is
   limited to the one-row inspection plus selected unique rows;
5. the known-gauge real forward, dictionary loader, TensorDict loader, and
   prebuilt mmap loader recover the expected gauge;
6. fresh `ones` and `dose_closure` runs emit strict v2 records while historical
   v1 records still round-trip and remain reusable;
7. existing invalid-input, module-state, rank-zero publication, and barrier
   tests remain green; and
8. branch-native CI integrations pass before each branch's broader suite.

At least one deterministic ordering-bias fixture must compare the new estimate
with the old prefix estimate on a spatially blocked population and show that
the fixed uniform sample is materially closer to the full-population gauge.
This is an estimator regression, not a reconstruction-quality threshold.

## Non-Goals

- A configurable sampling seed or sample count.
- Experiment-balanced or per-experiment gauge fitting.
- Deduplication of grouped acquisition frames.
- Changes to the unit-object convention, float64 sums, gauge equation,
  `rect_s1s2_trainable`, inference VarPro, or dataset refit.
- Rewriting historical v1 summaries or completed prefix-era metrics.
- Establishing new GPU reconstruction-quality thresholds.
