# CI Dose-Closure Representative Sampling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace order-sensitive first-256 CI dose closure with the same fixed-seed, uniformly sampled 256 detector slots on `refactor`, `fno-stable`, and `refactor-internal`.

**Architecture:** A new private `rect_s1s2_sampling` module owns the versioned SplitMix64 selection and immutable logical-row/channel plan. The workflow reads only the inspection row and selected logical rows, applies channel masks after the real forward, and leaves the gauge equation unchanged. Fresh runs emit a strict v2 initialization record; v1 records remain readable as prefix-era results. `refactor` is the reference implementation, `fno-stable` receives an adapted port, and `refactor-internal` receives the seeded-v2 solver directly as part of its pending dose-closure convergence.

**Tech Stack:** Python 3.11, PyTorch, Lightning, TensorDict/mmap datasets, dataclasses, pytest, Git. Do not create new worktrees.

---

**Design authority:** [`docs/superpowers/specs/2026-08-06-ci-dose-closure-representative-sampling-design.md`](../specs/2026-08-06-ci-dose-closure-representative-sampling-design.md)

**Parent contract:** [`docs/superpowers/specs/2026-08-05-regular-ci-dose-closure-convergence-design.md`](../specs/2026-08-05-regular-ci-dose-closure-convergence-design.md)

## Composition with the parent convergence plan

This plan is the sampling-and-record amendment to the parent convergence plan,
not a second independent rollout. Execute the two plans in this order:

1. On `refactor`, parent Tasks 1–2 are already complete in `a1f5d05ef` plus
   `6ab1716ec` (with the isolated prerequisite test fix in `80261d7a7`). Do not
   rerun setup or create more worktrees. Execute this plan's Tasks 1–2, then
   parent Task 3's CLI/MLflow code boundary. Fold the parent documentation step
   into this plan's Task 3 before running the branch gates.
2. On `fno-stable`, execute parent Task 4's config/identity convergence before
   this plan's Task 4 runtime port. Then execute parent Task 5's CLI/MLflow
   boundary and fold both plans' documentation into Task 4 before its branch
   gates.
3. On `refactor-internal`, this plan's Task 5 rewrites parent Tasks
   6–7: it activates the seeded-v2 runtime in the same commit that changes the
   profile/config identity and removes `data` calibration. Execute parent Task
   8's CLI/MLflow boundary next, then fold parent Task 9 into Task 5's
   documentation step before branch gates.
4. Use the parent Task 10 ordering for every branch: focused tests, integration
   tests, then the exact comprehensive command `python -m pytest -q`. This
   plan's final audit adds sampling identity to that same three-tip audit; it
   does not authorize a second, redundant comprehensive run.

## Branch and file map

| Concern | `refactor` | `fno-stable` | `refactor-internal` |
|---|---|---|---|
| Record | Existing `ptycho_torch/rect_s1s2_initialization.py` | Same existing v1 blob | Create/port during convergence |
| Sampler | Create `ptycho_torch/rect_s1s2_sampling.py` | Port settled file | Port settled file directly |
| Runtime | `ptycho_torch/workflows/components.py` | Same symbols, branch-local line drift | Replace old one-batch `data` path while converging |
| Focused tests | Add `tests/torch/test_rect_s1s2_sampling.py`; extend `tests/torch/test_rect_s1s2_initialization.py` | Same paths, adapted | Replace two historical calibration cases with settled coverage |
| Runner tests | `tests/torch/test_grid_lines_torch_runner_s1s2_init.py` | `tests/torch/test_grid_lines_torch_runner.py` | `tests/torch/test_grid_lines_torch_runner.py` |
| Normative docs | New design plus parent amendment | `docs/specs/spec-ptycho-core.md` | Same core spec after convergence |

Work in the existing branch checkout reported by `git worktree list`, if one
already exists, or execute branches serially after they are free. Do not create
another worktree. Preserve `.claude/`, `docs/plans/NEURIPS-HYBRID-RESNET-2026/`,
`scripts/orchestration/`, and the dirty notebook submodule.

### Task 1: Establish the selection and dual-reader foundation on `refactor`

**Files:**

- Create: `ptycho_torch/rect_s1s2_sampling.py`
- Create: `tests/torch/test_rect_s1s2_sampling.py`
- Modify: `ptycho_torch/rect_s1s2_initialization.py`
- Modify: `tests/torch/test_rect_s1s2_initialization.py`
- Modify: `tests/test_synthetic_pipeline.py`

- [ ] **Step 1: Write the pinned SplitMix64 and flat-slot tests**

Add tests for the exact v2 draw vector, flat-slot inversion, grouped channels,
and independence from ambient RNG state. Pin a compact digest rather than 256
literal integers:

```python
def test_v2_selection_vector_is_pinned():
    plan = build_dose_closure_sample_plan(
        dataset=_IdentityDataset(1024),
        channels=1,
    )
    encoded = b"".join(struct.pack("<Q", value) for value in plan.flat_slots)
    assert hashlib.sha256(encoded).hexdigest() == (
        "6c88290b749ff9ba972a2adeafce93985"
        "cb90c91ce3e682fe8f30add20f6e8f1"
    )
    assert plan.flat_slots[:8] == (705, 359, 847, 532, 312, 814, 888, 27)
    assert len(plan.flat_slots) == len(set(plan.flat_slots)) == 256
```

For `C=9`, assert every selected slot obeys
`flat == logical_row * 9 + channel`, every channel is in `[0, 9)`, and exactly
256 slots are represented without final-row truncation.

The `n=1024` digest does not exercise rejection because 1024 divides `2**64`.
Add a focused test of the private bounded-candidate helper with a non-power-of-
two bound: a candidate at or above `limit = 2**64 - (2**64 % n)` is rejected,
while `limit - 1` is accepted and mapped with modulo. This pins the rejection
step independently of the fixed stream's first 256 accepted values.

- [ ] **Step 2: Write nested-subset and duplicate-membership tests**

Construct two nested `torch.utils.data.Subset` objects whose outer logical rows
include two distinct paths to one base row. Require:

```python
assert all(
    row.logical_row in range(len(outer_subset))
    for row in plan.access_rows
)
selected_base_rows = {row.base_row for row in plan.access_rows}
assert validation_only_base_rows.isdisjoint(selected_base_rows)
assert duplicate_logical_members_keep_separate_masks(plan.access_rows)
assert plan.access_rows == tuple(
    sorted(plan.access_rows, key=lambda row: (row.base_row, row.logical_row))
)
```

Also snapshot and compare `random.getstate()`, NumPy RNG state, and
`torch.random.get_rng_state()` around plan construction.

- [ ] **Step 3: Run the new tests red**

```bash
python -m pytest tests/torch/test_rect_s1s2_sampling.py -q
```

Expected: collection fails because `ptycho_torch.rect_s1s2_sampling` does not
exist.

- [ ] **Step 4: Implement the bounded selection module**

Use immutable internal values with these boundaries:

```python
@dataclass(frozen=True, slots=True)
class SelectedDoseClosureRow:
    logical_row: int
    base_row: int
    channels: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class DoseClosureSamplePlan:
    population_patterns: int
    flat_slots: tuple[int, ...]       # canonical draw order
    access_rows: tuple[SelectedDoseClosureRow, ...]  # physical-read order


def build_dose_closure_sample_plan(
    dataset,
    *,
    channels: int,
) -> DoseClosureSamplePlan:
    ...
```

Implement the spec's unsigned-64-bit SplitMix64 transforms, unbiased rejection
for `randbelow`, duplicate-draw rejection, `divmod(flat_slot, channels)`, and
recursive `Subset` mapping. Use a set sized only to the 256 selected slots; do
not allocate `randperm(population_patterns)` or enumerate the population.
The production plan builder always uses
`RECT_S1S2_DOSE_CLOSURE_SAMPLE_SEED` and
`RECT_S1S2_DOSE_CLOSURE_PATTERNS`; it accepts no alternate seed or count. A
private candidate-mapping helper may accept a bound so the rejection rule can
be tested directly, but no runtime path may emit v2 for a different policy.

- [ ] **Step 5: Write strict v1/v2 decoder tests red**

Require existing `ones()` and `dose_closure()` factories to remain v1 in this
additive commit, while `from_mapping()` accepts an exact v2 payload. Add v1
round trips at counts 256 and 512; reject v1 below 256, v2 counts other than
256, cross-version methods, unknown schemas, malformed fields, and non-finite
gauges. Both versions of `ones` require gauge 1.0 and zero samples. Add the
matching historical-v1 synthetic-manifest reuse case, then run:

```bash
python -m pytest tests/torch/test_rect_s1s2_initialization.py -k "record" -q
```

- [ ] **Step 6: Add v2 constants and strict decoding without changing production**

In `rect_s1s2_initialization.py`, add:

```python
RECT_S1S2_INITIALIZATION_SCHEMA_V1 = "rect-s1s2-initialization-v1"
RECT_S1S2_INITIALIZATION_SCHEMA_V2 = "rect-s1s2-initialization-v2"
RECT_S1S2_DOSE_CLOSURE_SAMPLE_SEED = 20260806
RECT_S1S2_DOSE_CLOSURE_SAMPLE_POLICY = "splitmix64_rejection_v1"
```

Leave the existing `RECT_S1S2_INITIALIZATION_SCHEMA` alias pointing to v1 in
this task. Make `_validated_values()` choose strict method/count invariants by
`(schema_version, mode)`: v1 dose closure accepts counts at least 256, v2 dose
closure requires exactly 256, and both versions of `ones` require gauge 1.0
and zero samples. `from_mapping()` accepts both versions without rewriting v1.
Keep `ones()` and `dose_closure()` producing v1 until Task 2 switches them
atomically with the representative runtime. Add a synthetic-manifest reuse test
for an unchanged matching v1 record. Export only constants needed by the
sampler/tests; do not add config or CLI fields.

- [ ] **Step 7: Run green and commit**

```bash
python -m pytest \
  tests/torch/test_rect_s1s2_sampling.py \
  tests/torch/test_rect_s1s2_initialization.py \
  tests/test_synthetic_pipeline.py -q
git diff --check
git add ptycho_torch/rect_s1s2_sampling.py ptycho_torch/rect_s1s2_initialization.py tests/torch/test_rect_s1s2_sampling.py tests/torch/test_rect_s1s2_initialization.py tests/test_synthetic_pipeline.py
git commit -m "feat(torch): add representative sampling foundation"
```

### Task 2: Integrate selected-row reading and masked closure on `refactor`

**Files:**

- Modify: `ptycho_torch/rect_s1s2_sampling.py`
- Modify: `ptycho_torch/workflows/components.py`
- Modify: `ptycho_torch/rect_s1s2_initialization.py`
- Modify: `tests/torch/test_rect_s1s2_sampling.py`
- Modify: `tests/torch/test_rect_s1s2_initialization.py`
- Modify: `tests/torch/test_workflows_components.py`
- Modify: `tests/test_training_workflow_initialization_summary.py`
- Modify: `tests/test_synthetic_pipeline.py`
- Modify: `tests/torch/test_grid_lines_torch_runner_s1s2_init.py`

- [ ] **Step 1: Write row-reader adapter tests**

Cover ordinary `DataLoader`, `TensorDictDataLoader`, nested `Subset`, differing
batch sizes, worker settings, and shuffled original loaders. Require the
adapter to inspect row zero privately, yield `(logical_rows, batch)` pairs in
physical-read order, use `num_workers=0`, and leave the original
sampler/generator unconsumed. Verify the inspection plus selected logical rows
is the only dataset access, including for the prebuilt mmap path.

Keep loader construction out of `ptycho_torch.rect_s1s2_sampling`; its
immutable `SelectedDoseClosureRow` and `DoseClosureSamplePlan` values are the
adapter contract. Implement concrete private helpers in
`ptycho_torch.workflows.components` that:

- require a loader with an indexable `dataset`, a positive integer
  `batch_size`, and maintained collation behavior;
- rebuild the matching ordinary `DataLoader` or `TensorDictDataLoader` with
  `sampler=tuple(logical_rows)`, `drop_last=False`, `num_workers=0`,
  `pin_memory=False`, and no `shuffle` or generator;
- inspect logical row `(0,)` with private batch size one;
- read `row.logical_row for row in plan.access_rows` in physical-read order;
- chunk `plan.access_rows` by the private batch size and pair chunks with
  batches using strict cardinality checking; and
- reject missing, extra, duplicated, or reordered identities instead of
  silently attaching a channel mask to another row.

Group selected channels once per distinct logical row and sort channels within
that row. Distinct logical rows mapping to the same base row remain separate
access-plan members and retain their multiplicity.

- [ ] **Step 2: Replace prefix-dependent fixtures with red representative tests**

Rewrite `_known_gauge_loader` so every population member represents the same
known gauge. Add a separate `N=1024, C=1` blocked-dose fixture where the first
256 slots imply gauge `1.0` and the remaining 768 imply gauge `4.0`. The full
population gauge is `3.5`; the pinned selection contains 63 early and 193 late
slots and must solve `3.508360550171547` within numeric tolerance.
Also require
`abs(sampled_gauge - full_gauge) < abs(prefix_gauge - full_gauge)` so the
fixture proves the estimator improvement rather than only a hard-coded number.

Add assertions for:

- exactly 256 masked slots at `C=1` and `C=9`;
- every selected flat slot contributes exactly once, with no missing or extra
  contribution;
- identical gauge for different loader shuffle seeds, batch sizes, and worker
  settings;
- no change after perturbing Python/NumPy/Torch global RNG state;
- selected-row validation/nonfinite tests contaminate a slot obtained from the
  plan, not hard-coded row zero;
- simulated rank values do not affect the plan; and
- selected logical duplicates retain multiplicity.

Derive `C` only from canonical post-collation
`measured_intensity.shape == (B, C, H, W)` in the private inspection batch.
Add clear failures for an empty dataset, a non-positive inspected `C`, and a
selected row whose post-collation channel count differs from the inspection.
Instrument an unselected inconsistent row to prove the solver does not scan it
merely to validate dataset-wide shape.

Also require fresh constructors and workflow results to switch together:

```python
assert RectS1S2InitializationRecord.ones().schema_version.endswith("-v2")
assert RectS1S2InitializationRecord.dose_closure(2.0).method == (
    "dose_closure_seeded_uniform_unit_object"
)
```

- [ ] **Step 3: Run the runtime tests red**

```bash
python -m pytest \
  tests/torch/test_rect_s1s2_sampling.py \
  tests/torch/test_rect_s1s2_initialization.py \
  tests/torch/test_workflows_components.py \
  tests/test_training_workflow_initialization_summary.py \
  tests/test_synthetic_pipeline.py \
  tests/torch/test_grid_lines_torch_runner_s1s2_init.py -q
```

Expected: prefix runtime cannot honor the selected plan or v2 method.

- [ ] **Step 4: Implement the private row-reader adapter**

Implement the component helpers specified in Step 1. Preserve the distinct
ordinary-DataLoader and batched-`TensorDictDataLoader` indexing/collation
contracts, including nested subsets, and fail clearly for unsupported loader,
indexing, or collation behavior. Check yielded batch cardinality and identity
coverage before returning data to the solver.

- [ ] **Step 5: Replace prefix accumulation with selected masks**

In `_initialize_rect_s1s2_unmanaged()`:

1. inspect and validate canonical `(B,C,H,W)` CI fields;
2. build the immutable sample plan;
3. run the actual forward over selected group rows;
4. create a boolean `(B,C)` mask from each batch's logical-row plan;
5. accumulate only `target.reshape(B,C,-1)[mask]` and
   `predicted.reshape(B,C,-1)[mask]` in float64; and
6. require the final masked count to equal 256 before producing v2.

Keep scaler reset, train/eval restoration, gauge validation, rank publication,
barrier, and the no-loader `ones` path unchanged.

Remove the existing `sample_patterns` parameter from both initializer helpers
and from the fresh `RectS1S2InitializationRecord.dose_closure()` constructor.
Historical v1 counts remain readable only through `from_mapping()`; every fresh
v2 solve is fixed to 256.

```python
RectS1S2InitializationRecord.dose_closure(solved_gauge)
_initialize_rect_s1s2_unmanaged(model, *, mode, training_loader=None)
_initialize_rect_s1s2(model, *, mode, training_loader=None)
```

In the same implementation step, switch
`RECT_S1S2_INITIALIZATION_SCHEMA`, `ones()`, and `dose_closure()` to fresh v2
production and update every listed fresh-result fixture to the v2 schema/new
method. This atomic boundary prevents a prefix solve from ever being labeled
v2. Retain the v1 reuse case added in Task 1 unchanged.

- [ ] **Step 6: Run focused green and commit**

```bash
python -m pytest \
  tests/torch/test_rect_s1s2_sampling.py \
  tests/torch/test_rect_s1s2_initialization.py \
  tests/torch/test_workflows_components.py \
  tests/test_training_workflow_initialization_summary.py \
  tests/test_synthetic_pipeline.py \
  tests/torch/test_grid_lines_torch_runner_s1s2_init.py -q
git diff --check
git add ptycho_torch/rect_s1s2_sampling.py ptycho_torch/rect_s1s2_initialization.py ptycho_torch/workflows/components.py tests/torch/test_rect_s1s2_sampling.py tests/torch/test_rect_s1s2_initialization.py tests/torch/test_workflows_components.py tests/test_training_workflow_initialization_summary.py tests/test_synthetic_pipeline.py tests/torch/test_grid_lines_torch_runner_s1s2_init.py
git commit -m "fix(torch): sample dose closure across training data"
```

### Task 3: Close `refactor` documentation and hand off branch gates

**Prerequisite:** Complete parent Task 3 Steps 1–4 first: the native CLI must
forward only explicit `--rect-s1s2-init` values and all maintained MLflow
whole-model loaders must reject historical `data`. Keep that code/test commit
separate; fold only its documentation step into this task.

**Files:**

- Modify: `docs/superpowers/specs/2026-08-05-regular-ci-dose-closure-convergence-design.md`
- Modify: `docs/superpowers/plans/2026-08-05-regular-ci-dose-closure-convergence.md`
- Modify: `docs/CONFIGURATION.md`
- Modify: `docs/workflows/pytorch.md`
- Modify: `docs/COMMANDS_REFERENCE.md`
- Modify: `scripts/studies/README.md`
- Modify: `scripts/simulation/README.md`
- Modify: `ptycho_torch/config_params.py` (comment only)

- [ ] **Step 1: Remove current-contract prefix wording**

Make the parent design and plan defer sampling/schema semantics to the new
design. Replace active “first 256” descriptions with fixed-seed uniform
flat-slot sampling and v2 method/compatibility text. Do not create
`docs/index.md` on this branch and do not rewrite completed historical evidence.

- [ ] **Step 2: Verify docs and commit**

```bash
git diff --check
git grep -n -E 'first[- ]256|256-pattern prefix|dose_closure_unit_object' -- \
  docs/CONFIGURATION.md docs/workflows/pytorch.md docs/COMMANDS_REFERENCE.md \
  scripts/studies/README.md scripts/simulation/README.md \
  ptycho_torch/config_params.py
git add docs/superpowers/specs/2026-08-05-regular-ci-dose-closure-convergence-design.md docs/superpowers/plans/2026-08-05-regular-ci-dose-closure-convergence.md docs/CONFIGURATION.md docs/workflows/pytorch.md docs/COMMANDS_REFERENCE.md scripts/studies/README.md scripts/simulation/README.md ptycho_torch/config_params.py
git commit -m "docs(ci): explain representative dose sampling"
```

The grep is expected to return no current-contract match. Historical plans or
record-compatibility sections may retain explicitly labeled v1 wording.
Run the refactor portion of amended parent Task 10 after this commit; do not
duplicate its focused, integration, or comprehensive commands here.

### Task 4: Port the settled implementation to `fno-stable`

**Prerequisite:** Complete parent Task 4's branch-local config, identity,
historical-fixture, and `data`-rejection convergence first. After Step 3 below,
complete parent Task 5's CLI/MLflow code boundary before documentation or
branch gates. Fold the two plans' overlapping docs into this task.

**Files:**

- Create: `ptycho_torch/rect_s1s2_sampling.py`
- Modify: `ptycho_torch/rect_s1s2_initialization.py`
- Modify: `ptycho_torch/workflows/components.py`
- Create: `tests/torch/test_rect_s1s2_sampling.py`
- Modify: `tests/torch/test_rect_s1s2_initialization.py`
- Modify: `tests/torch/test_workflows_components.py`
- Modify: `tests/test_synthetic_pipeline.py`
- Modify: `tests/scripts/test_training_backend_selector.py`
- Modify: `tests/torch/test_grid_lines_torch_runner.py`
- Modify: `docs/specs/spec-ptycho-core.md`
- Modify: `docs/DATA_NORMALIZATION_GUIDE.md`
- Modify: `docs/CONFIGURATION.md`
- Modify: `docs/workflows/pytorch.md`
- Modify: `docs/COMMANDS_REFERENCE.md`
- Modify: `docs/TESTING_GUIDE.md`
- Modify: `docs/development/TEST_SUITE_INDEX.md`
- Modify: `docs/findings.md`
- Modify: `docs/index.md`
- Modify: `scripts/studies/README.md`
- Modify: `scripts/simulation/README.md`

- [ ] **Step 1: Confirm branch state and port tests first**

Record the branch SHA and compare each candidate file against the settled
`refactor` parent. Cherry-pick a commit only when its complete touched-file set
has identical parent blobs; otherwise port by symbol. Add the pure sampler,
dual-schema record, runtime, subset/mmap, bias, and consumer tests before
production edits.

- [ ] **Step 2: Run the branch-native red gate**

```bash
python -m pytest \
  tests/torch/test_rect_s1s2_sampling.py \
  tests/torch/test_rect_s1s2_initialization.py \
  tests/torch/test_workflows_components.py \
  tests/test_synthetic_pipeline.py \
  tests/scripts/test_training_backend_selector.py \
  tests/torch/test_grid_lines_torch_runner.py -q
```

- [ ] **Step 3: Port the settled symbols and run green**

Keep the sampler and record module behavior byte-identical where branch APIs
permit. Adapt only workflow imports/loader integration and branch-local tests.

```bash
python -m pytest \
  tests/torch/test_rect_s1s2_sampling.py \
  tests/torch/test_rect_s1s2_initialization.py \
  tests/torch/test_workflows_components.py \
  tests/test_synthetic_pipeline.py \
  tests/scripts/test_training_backend_selector.py \
  tests/torch/test_grid_lines_torch_runner.py -q
git diff --check
git add ptycho_torch/rect_s1s2_sampling.py ptycho_torch/rect_s1s2_initialization.py ptycho_torch/workflows/components.py tests/torch/test_rect_s1s2_sampling.py tests/torch/test_rect_s1s2_initialization.py tests/torch/test_workflows_components.py tests/test_synthetic_pipeline.py tests/scripts/test_training_backend_selector.py tests/torch/test_grid_lines_torch_runner.py
git commit -m "fix(torch): port representative dose sampling"
```

- [ ] **Step 4: Complete the parent CLI/MLflow boundary and its focused gate**

Execute parent Task 5 Steps 1–2. Require the native help/omission behavior and
all maintained MLflow loaders to enforce the two-value contract before
continuing:

```bash
python -m pytest \
  tests/torch/test_ci_profile.py \
  tests/torch/test_api_config_retirement.py \
  tests/scripts/test_training_backend_selector.py -q
python -m ptycho_torch.train --help
```

- [ ] **Step 5: Update normative and routed documentation**

Make `docs/specs/spec-ptycho-core.md` the branch source of truth for the
flat-slot formula, SplitMix64 seed/policy, v2 production, and strict v1 reading.
Update dependent guides/indexes. Preserve `docs/plans/2026-08-04-ci-gauge-invariant-scaling.md`
and its metrics as explicitly prefix-era evidence; do not silently relabel the
historical gauge as a uniform-sample result.

- [ ] **Step 6: Commit branch-owned docs**

```bash
git diff --check
git add docs/specs/spec-ptycho-core.md docs/DATA_NORMALIZATION_GUIDE.md docs/CONFIGURATION.md docs/workflows/pytorch.md docs/COMMANDS_REFERENCE.md docs/TESTING_GUIDE.md docs/development/TEST_SUITE_INDEX.md docs/findings.md docs/index.md scripts/studies/README.md scripts/simulation/README.md
git commit -m "docs(ci): specify representative dose sampling"
```

Run the fno-stable portion of amended parent Task 10 after this commit; it is
the sole owner of focused reruns, integration-before-comprehensive ordering,
and the full-suite result.

### Task 5: Rewrite and execute parent Tasks 6–9 on `refactor-internal`

This task replaces the parent plan's prefix-foundation instructions. It keeps
the parent's safe two-commit transition, but the additive foundation is seeded
v2 from the start. At no point may the internal branch contain a prefix solver
or advertise `dose_closure` without a working solver.

**Foundation files:**

- Create: `ptycho_torch/rect_s1s2_sampling.py`
- Create: `ptycho_torch/rect_s1s2_initialization.py`
- Modify: `ptycho_torch/scaling_contract.py`
- Modify: `ptycho_torch/workflows/components.py`
- Modify: `ptycho_torch/train_utils.py`
- Create: `tests/torch/test_rect_s1s2_sampling.py`
- Extend: `tests/torch/test_rect_s1s2_initialization.py`
- Modify: `tests/torch/test_workflows_components.py`

**Atomic-transition files:**

- Modify: `ptycho_torch/rect_s1s2_initialization.py`
- Modify: `ptycho_torch/config_params.py`
- Modify: `ptycho_torch/config_factory.py`
- Modify: `ptycho_torch/config_resolution.py`
- Modify: `ptycho_torch/model.py`
- Modify: `ptycho_torch/workflows/components.py`
- Modify: `ptycho/workflows/synthetic_config.py`
- Modify: `ptycho/workflows/training.py`
- Modify: `scripts/studies/grid_lines_torch_runner.py`
- Create: `tests/torch/test_ci_container_bridge.py`
- Create: `tests/test_training_workflow_initialization_summary.py`
- Modify the parent Task 7 config, identity, artifact, workflow-result, and
  runner tests plus `tests/studies/test_torch_ablation_manifest.py`

The internal branch's frozen fixtures already encode `ones`. Preserve the
generator, README, and serialized bytes; construct `data` rejection inputs by
mutating copies in the relevant tests.

- [ ] **Step 1: Port the complete seeded-v2 foundation tests red**

Port the settled pure selection, strict v1/v2 record, selected-row reader,
real-forward, dictionary, TensorDict, prebuilt mmap, subset/mmap-bound,
ordering-bias, invalid-input, module-state, and rank-publication tests. The
internal branch's existing `data` configuration and calibration tests remain
green in this first slice. Include a real prebuilt-mmap loader with
`num_workers=0`; fix `train_utils.py` to omit `prefetch_factor` for that case.

```bash
python -m pytest \
  tests/torch/test_rect_s1s2_sampling.py \
  tests/torch/test_rect_s1s2_initialization.py \
  tests/torch/test_workflows_components.py \
  tests/torch/test_rect_scaling.py \
  tests/torch/test_rectangular_scaled_forward.py -q
```

Expected: only the new seeded-v2 cases fail because the modules/helpers are
absent; the still-active historical calibration cases continue to pass.

- [ ] **Step 2: Add the inactive seeded-v2 foundation**

Port the settled sampler, strict v1/v2 reader with fresh v2 constructors, and
selected-row workflow helpers by symbol. Keep `ModelConfig`, the CI profile,
training entry wiring, `calibrate_rect_s1s2`, and
`rect_s1s2_calibration` unchanged. The new helper is complete and directly
testable but not yet selected by internal configuration. Do not port
`_slice_batch_prefix`, `_deterministic_rect_s1s2_loader`, or any other prefix
path. Add the atomic summary writer/barrier helpers, but defer
`_TrainingSummaryCallback` until activation so `_train_with_lightning` and its
callback contract change together.

- [ ] **Step 3: Run the foundation green and commit**

```bash
python -m pytest \
  tests/torch/test_rect_s1s2_sampling.py \
  tests/torch/test_rect_s1s2_initialization.py \
  tests/torch/test_workflows_components.py \
  tests/torch/test_rect_scaling.py \
  tests/torch/test_rectangular_scaled_forward.py \
  tests/torch/test_ci_profile.py -q
git diff --check
git add ptycho_torch/rect_s1s2_sampling.py ptycho_torch/rect_s1s2_initialization.py ptycho_torch/scaling_contract.py ptycho_torch/workflows/components.py ptycho_torch/train_utils.py tests/torch/test_rect_s1s2_sampling.py tests/torch/test_rect_s1s2_initialization.py tests/torch/test_workflows_components.py
git commit -m "feat(torch): add seeded gauge runtime"
```

- [ ] **Step 4: Write the parent Task 7 atomic-transition tests**

Require bare `ones`, CI-profile `dose_closure`, explicit `ones` precedence,
direct/mapping/ModelSpec/artifact/checkpoint `data` rejection, the
amplitude-only synthetic profile's `ones` constraint, pre-fit seeded-v2
initialization, strict summary publication, and shared/grid-runner result
propagation. Require the shared path to bridge raw grouped diffraction counts
into the CI container, and change the ablation manifest's valid nondefault mode
from `data` to `dose_closure`. Require `calibrate_rect_s1s2` and
`rect_s1s2_calibration` to be absent after the transition. Keep frozen
historical fixture bytes unchanged.

- [ ] **Step 5: Run the atomic transition red**

```bash
python -m pytest \
  tests/torch/test_ci_profile.py \
  tests/torch/test_config_resolution_internal_transaction.py \
  tests/torch/test_model_spec.py \
  tests/torch/test_artifact_schema.py \
  tests/torch/test_artifact_schema_v2.py \
  tests/torch/test_config_pydantic_artifacts.py \
  tests/torch/test_rect_s1s2_sampling.py \
  tests/torch/test_rect_s1s2_initialization.py \
  tests/torch/test_workflows_components.py \
  tests/torch/test_ci_container_bridge.py \
  tests/test_synthetic_workflow_config.py \
  tests/test_training_workflow_initialization_summary.py \
  tests/torch/test_grid_lines_torch_runner.py \
  tests/studies/test_torch_ablation_manifest.py -q
```

- [ ] **Step 6: Activate v2 and retire `data` atomically**

Apply the parent contract's common mode validator, profile-only default,
resolver defense, and frozen-fixture strategy. Add the summary callback and
wire the already-green seeded-v2 initializer before `trainer.fit`,
publish/return the strict record and summary path, remove the old `data` branch,
learned-model calibration method, `_last_calibration_means`, and obsolete loss
target. Preserve raw grouped diffraction counts in the shared container and
adapt them to `measured_intensity` at the CI boundary; do not estimate closure
from normalized `X`. Propagate the new result through the shared workflow and
grid runner. Keep the standalone runner default `ones`, inference refit, and
amplitude-only synthetic contract unchanged. Do not add a compatibility alias
or prefix fallback.

- [ ] **Step 7: Run green and commit the atomic transition**

```bash
python -m pytest \
  tests/torch/test_ci_profile.py \
  tests/torch/test_config_resolution_internal_transaction.py \
  tests/torch/test_model_spec.py \
  tests/torch/test_artifact_schema.py \
  tests/torch/test_artifact_schema_v2.py \
  tests/torch/test_config_pydantic_artifacts.py \
  tests/torch/test_rect_s1s2_sampling.py \
  tests/torch/test_rect_s1s2_initialization.py \
  tests/torch/test_workflows_components.py \
  tests/torch/test_rect_scaling.py \
  tests/torch/test_rectangular_scaled_forward.py \
  tests/torch/test_ci_container_bridge.py \
  tests/test_synthetic_workflow_config.py \
  tests/test_training_workflow_initialization_summary.py \
  tests/torch/test_grid_lines_torch_runner.py \
  tests/studies/test_torch_ablation_manifest.py -q
rg -n 'calibrate_rect_s1s2|rect_s1s2_calibration' ptycho ptycho_torch scripts tests
git diff --check
git add ptycho_torch/rect_s1s2_initialization.py ptycho_torch/config_params.py ptycho_torch/config_factory.py ptycho_torch/config_resolution.py ptycho_torch/model.py ptycho_torch/workflows/components.py ptycho/workflows/synthetic_config.py ptycho/workflows/training.py scripts/studies/grid_lines_torch_runner.py tests/torch/test_ci_profile.py tests/torch/test_config_resolution_internal_transaction.py tests/torch/test_model_spec.py tests/torch/test_artifact_schema.py tests/torch/test_artifact_schema_v2.py tests/torch/test_config_pydantic_artifacts.py tests/torch/test_rect_s1s2_initialization.py tests/torch/test_workflows_components.py tests/torch/test_ci_container_bridge.py tests/test_synthetic_workflow_config.py tests/test_training_workflow_initialization_summary.py tests/torch/test_grid_lines_torch_runner.py tests/studies/test_torch_ablation_manifest.py
git commit -m "fix(torch): replace data calibration with seeded dose closure"
```

- [ ] **Step 8: Execute parent Task 8, then fold sampling into parent Task 9**

Add and verify the native CLI plus MLflow boundaries only after Step 7. Update
the full internal core, normalization, configuration, command, workflow,
testing, findings, index, and runner documentation set with seeded-v2 and
strict-v1 semantics. Documentation exclusions used for other propagation paths
do not apply to `refactor-internal`.

Run the internal portion of amended parent Task 10 after those commits; do not
duplicate its focused, integration, comprehensive, or final-audit commands
here.

### Task 6: Amend and execute parent Task 10 once

**Files:** Modify the parent plan now; final execution is read-only across all
three settled branch tips unless a failure is causally traced to this work.

The sampling checks below are additions to parent Task 10, not a second final
gate. For each branch, run its focused contract, then integrations, then
`python -m pytest -q` exactly once after all branch code/docs are settled.

- [ ] **Step 1: Compare code contracts**

For each branch, verify:

```text
bare default       ones
regular-ci default dose_closure
explicit override  ones wins
supported modes    exactly ones and dose_closure
native CLI         --rect-s1s2-init {ones,dose_closure}
historical data    rejected at config/identity/checkpoint/MLflow boundaries
sample count       256
sampling seed      20260806
sampling policy    splitmix64_rejection_v1
fresh schema       rect-s1s2-initialization-v2
fresh dose method  dose_closure_seeded_uniform_unit_object
legacy schema      strict v1 read support
flat mapping       logical_row * C + channel
```

Require the pure sampling module to be identical unless a documented branch
API forces a minimal import-only difference.

- [ ] **Step 2: Sweep current authority surfaces**

Search current specs, guides, code comments, tests, and runner docs for stale
prefix behavior. Classify retained matches:

- completed 2026-08-04 plans, recorded metrics, and v1 compatibility text are
  intentional history;
- current normative or user-facing “first 256” wording is a failure; and
- unrelated “first-N grouping” or raster-test wording is out of scope.

- [ ] **Step 3: Confirm evidence and branch cleanliness**

Record all three final SHAs, focused/integration/comprehensive results, and
fresh artifact paths for long integration runs. Confirm only the user's
pre-existing dirty/untracked paths remain and no branch contains an
intermediate prefix port or broad merge used for the internal migration.

The intended commit shape is two focused code/test commits plus one boundary
commit and one docs commit on `refactor`; one config commit, one sampling/runtime commit, one
boundary commit, and one docs commit on `fno-stable`; and two internal runtime
commits followed by its boundary and docs commits. Squash only if later
requested; do not mix unrelated branch changes.
