# CI Dose-Closure Representative Sampling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace order-sensitive first-256 CI dose closure with the same fixed-seed, uniformly sampled 256 detector slots on `refactor`, `fno-stable`, and `refactor-internal`.

**Architecture:** A new private `rect_s1s2_sampling` module owns the versioned SplitMix64 selection and immutable logical-row/channel plan. The workflow reads only the inspection row and selected logical rows, applies channel masks after the real forward, and leaves the gauge equation unchanged. Fresh runs emit a strict v2 initialization record; v1 records remain readable as prefix-era results. `refactor` is the reference implementation, `fno-stable` receives an adapted port, and `refactor-internal` receives the seeded-v2 solver directly as part of its pending dose-closure convergence.

**Tech Stack:** Python 3.11, PyTorch, Lightning, TensorDict/mmap datasets, dataclasses, pytest, Git. Do not create new worktrees.

---

**Design authority:** [`docs/superpowers/specs/2026-08-06-ci-dose-closure-representative-sampling-design.md`](../specs/2026-08-06-ci-dose-closure-representative-sampling-design.md)

**Parent contract:** [`docs/superpowers/specs/2026-08-05-regular-ci-dose-closure-convergence-design.md`](../specs/2026-08-05-regular-ci-dose-closure-convergence-design.md)

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

### Task 1: Establish the pure selection contract on `refactor`

**Files:**

- Create: `ptycho_torch/rect_s1s2_sampling.py`
- Create: `tests/torch/test_rect_s1s2_sampling.py`
- Modify: `ptycho_torch/rect_s1s2_initialization.py`

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

- [ ] **Step 2: Write nested-subset and duplicate-membership tests**

Construct two nested `torch.utils.data.Subset` objects whose outer logical rows
include two distinct paths to one base row. Require:

```python
assert plan.rows_are_from_logical_population
assert all(row.logical_row in range(len(outer_subset)) for row in plan.rows)
assert validation_only_base_rows.isdisjoint(plan.selected_base_rows)
assert duplicate_logical_members_keep_separate_masks(plan)
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
    sample_patterns: int = RECT_S1S2_DOSE_CLOSURE_PATTERNS,
    seed: int = RECT_S1S2_DOSE_CLOSURE_SAMPLE_SEED,
) -> DoseClosureSamplePlan:
    ...
```

Implement the spec's unsigned-64-bit SplitMix64 transforms, unbiased rejection
for `randbelow`, duplicate-draw rejection, `divmod(flat_slot, channels)`, and
recursive `Subset` mapping. Use a set sized only to the 256 selected slots; do
not allocate `randperm(population_patterns)` or enumerate the population.

- [ ] **Step 5: Add the v2 constants without changing record production yet**

In `rect_s1s2_initialization.py`, add:

```python
RECT_S1S2_INITIALIZATION_SCHEMA_V1 = "rect-s1s2-initialization-v1"
RECT_S1S2_INITIALIZATION_SCHEMA_V2 = "rect-s1s2-initialization-v2"
RECT_S1S2_DOSE_CLOSURE_SAMPLE_SEED = 20260806
RECT_S1S2_DOSE_CLOSURE_SAMPLE_POLICY = "splitmix64_rejection_v1"
```

Leave the existing `RECT_S1S2_INITIALIZATION_SCHEMA` alias pointing to v1 in
this task; Task 2 switches fresh production to v2 together with dual-version
validation. Export only constants needed by the sampler/tests. Do not add
config or CLI fields.

- [ ] **Step 6: Run green and commit**

```bash
python -m pytest tests/torch/test_rect_s1s2_sampling.py -q
git diff --check
git add ptycho_torch/rect_s1s2_sampling.py ptycho_torch/rect_s1s2_initialization.py tests/torch/test_rect_s1s2_sampling.py
git commit -m "feat(torch): select representative dose samples"
```

### Task 2: Version the initialization record on `refactor`

**Files:**

- Modify: `ptycho_torch/rect_s1s2_initialization.py`
- Modify: `tests/torch/test_rect_s1s2_initialization.py`
- Modify: `tests/torch/test_workflows_components.py`
- Modify: `tests/test_training_workflow_initialization_summary.py`
- Modify: `tests/test_synthetic_pipeline.py`
- Modify: `tests/torch/test_grid_lines_torch_runner_s1s2_init.py`

- [ ] **Step 1: Write v1/v2 record tests**

Require strict version-specific methods and counts:

```python
assert RectS1S2InitializationRecord.ones().schema_version.endswith("-v2")
assert RectS1S2InitializationRecord.dose_closure(2.0).to_jsonable() == {
    "schema_version": "rect-s1s2-initialization-v2",
    "mode": "dose_closure",
    "solved_gauge": 2.0,
    "method": "dose_closure_seeded_uniform_unit_object",
    "sampled_patterns": 256,
}
```

Add historical v1 round trips for `sampled_patterns=256` and `512`. Reject v1
below 256, v2 values other than exactly 256, cross-version method strings,
unknown schemas, missing/extra fields, and non-finite gauges. Both versions of
`ones` require gauge `1.0` and zero samples.

- [ ] **Step 2: Run record tests red**

```bash
python -m pytest tests/torch/test_rect_s1s2_initialization.py -k "record" -q
```

- [ ] **Step 3: Implement strict dual-version decoding and v2 production**

Make `_validated_values()` choose the allowed method/count by
`(schema_version, mode)`. Keep the five-field set unchanged. `ones()` and
`dose_closure()` produce v2; `from_mapping()` accepts both versions and never
rewrites v1.

- [ ] **Step 4: Update exact record fixtures at maintained consumers**

Change fresh-result fixtures to v2/new method in the listed test files. Add at
least one synthetic-manifest reuse test that accepts an unchanged historical v1
record whose mode matches the resolved configuration.

- [ ] **Step 5: Run green and commit**

```bash
python -m pytest \
  tests/torch/test_rect_s1s2_initialization.py \
  tests/torch/test_workflows_components.py \
  tests/test_training_workflow_initialization_summary.py \
  tests/test_synthetic_pipeline.py \
  tests/torch/test_grid_lines_torch_runner_s1s2_init.py -q
git diff --check
git add ptycho_torch/rect_s1s2_initialization.py tests/torch/test_rect_s1s2_initialization.py tests/torch/test_workflows_components.py tests/test_training_workflow_initialization_summary.py tests/test_synthetic_pipeline.py tests/torch/test_grid_lines_torch_runner_s1s2_init.py
git commit -m "feat(torch): version dose-closure sampling records"
```

### Task 3: Integrate selected-row reading and masked closure on `refactor`

**Files:**

- Modify: `ptycho_torch/rect_s1s2_sampling.py`
- Modify: `ptycho_torch/workflows/components.py`
- Modify: `tests/torch/test_rect_s1s2_sampling.py`
- Modify: `tests/torch/test_rect_s1s2_initialization.py`

- [ ] **Step 1: Write row-reader adapter tests**

Cover ordinary `DataLoader`, `TensorDictDataLoader`, nested `Subset`, differing
batch sizes, and shuffled original loaders. Require the adapter to inspect row
zero privately, yield `(logical_rows, batch)` pairs in physical-read order, use
`num_workers=0`, and leave the original sampler/generator unconsumed. Verify the
inspection plus selected logical rows is the only dataset access.

- [ ] **Step 2: Replace prefix-dependent fixtures with red representative tests**

Rewrite `_known_gauge_loader` so every population member represents the same
known gauge. Add a separate `N=1024, C=1` blocked-dose fixture where the first
256 slots imply gauge `1.0` and the remaining 768 imply gauge `4.0`. The full
population gauge is `3.5`; the pinned selection contains 63 early and 193 late
slots and must solve `3.508360550171547` within numeric tolerance.

Add assertions for:

- exactly 256 masked slots at `C=1` and `C=9`;
- identical gauge for different loader shuffle seeds and batch sizes;
- no change after perturbing Python/NumPy/Torch global RNG state;
- selected-row validation/nonfinite tests contaminate a slot obtained from the
  plan, not hard-coded row zero;
- simulated rank values do not affect the plan; and
- selected logical duplicates retain multiplicity.

- [ ] **Step 3: Run the runtime tests red**

```bash
python -m pytest \
  tests/torch/test_rect_s1s2_sampling.py \
  tests/torch/test_rect_s1s2_initialization.py -q
```

Expected: prefix runtime cannot honor the selected plan or v2 method.

- [ ] **Step 4: Implement the private row-reader adapter**

The adapter may rebuild a private loader from `dataset`, `batch_size`, and
`collate_fn`, but it must pair each yielded batch with the exact requested
logical identities. Use `sampler=ordered_logical_rows`, `drop_last=False`,
`num_workers=0`, and no `shuffle` argument. Fail clearly for unsupported loader
or dataset indexing behavior.

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

- [ ] **Step 6: Run focused green and commit**

```bash
python -m pytest \
  tests/torch/test_rect_s1s2_sampling.py \
  tests/torch/test_rect_s1s2_initialization.py \
  tests/torch/test_workflows_components.py -q
git diff --check
git add ptycho_torch/rect_s1s2_sampling.py ptycho_torch/workflows/components.py tests/torch/test_rect_s1s2_sampling.py tests/torch/test_rect_s1s2_initialization.py
git commit -m "fix(torch): sample dose closure across training data"
```

### Task 4: Close `refactor` documentation and integration gates

**Files:**

- Modify: `docs/superpowers/specs/2026-08-05-regular-ci-dose-closure-convergence-design.md`
- Modify: `docs/superpowers/plans/2026-08-05-regular-ci-dose-closure-convergence.md`
- Modify: `docs/CONFIGURATION.md`
- Modify: `docs/workflows/pytorch.md`
- Modify: `scripts/studies/README.md`
- Modify: `scripts/simulation/README.md`
- Modify: `ptycho_torch/config_params.py` (comment only)

- [ ] **Step 1: Remove current-contract prefix wording**

Make the parent design and plan defer sampling/schema semantics to the new
design. Replace active “first 256” descriptions with fixed-seed uniform
flat-slot sampling and v2 method/compatibility text. Do not create
`docs/index.md` on this branch and do not rewrite completed historical evidence.

- [ ] **Step 2: Run the focused contract surface**

```bash
python -m pytest \
  tests/torch/test_rect_s1s2_sampling.py \
  tests/torch/test_rect_s1s2_initialization.py \
  tests/torch/test_ci_profile.py \
  tests/test_synthetic_workflow_config.py \
  tests/scripts/test_synthetic_pipeline_cli.py \
  tests/test_synthetic_pipeline.py \
  tests/torch/test_grid_lines_torch_runner_s1s2_init.py -q
```

- [ ] **Step 3: Run integration tests before any broader suite**

```bash
python -m pytest \
  tests/torch/test_grid_lines_ci_probe_roundtrip_integration.py \
  tests/torch/test_integration_workflow_torch.py -q
```

Investigate any failure before proceeding. For a long GPU or comprehensive
command, use the `tmux` skill, activate `ptycho311`, track the exact launched
PID, require exit code zero, and verify fresh expected artifacts.

- [ ] **Step 4: Run the branch comprehensive gate only after integration passes**

```bash
python -m pytest tests -q
```

- [ ] **Step 5: Verify docs and commit**

```bash
git diff --check
git grep -n -E 'first[- ]256|256-pattern prefix|dose_closure_unit_object' -- \
  docs/CONFIGURATION.md docs/workflows/pytorch.md scripts/studies/README.md \
  scripts/simulation/README.md ptycho_torch/config_params.py
git add docs/superpowers/specs/2026-08-05-regular-ci-dose-closure-convergence-design.md docs/superpowers/plans/2026-08-05-regular-ci-dose-closure-convergence.md docs/CONFIGURATION.md docs/workflows/pytorch.md scripts/studies/README.md scripts/simulation/README.md ptycho_torch/config_params.py
git commit -m "docs(ci): explain representative dose sampling"
```

The grep is expected to return no current-contract match. Historical plans or
record-compatibility sections may retain explicitly labeled v1 wording.

### Task 5: Port the settled implementation to `fno-stable`

**Files:**

- Create: `ptycho_torch/rect_s1s2_sampling.py`
- Modify: `ptycho_torch/rect_s1s2_initialization.py`
- Modify: `ptycho_torch/workflows/components.py`
- Create: `tests/torch/test_rect_s1s2_sampling.py`
- Modify: `tests/torch/test_rect_s1s2_initialization.py`
- Modify branch-local record consumers and runner tests
- Modify: `docs/specs/spec-ptycho-core.md`
- Modify: `docs/DATA_NORMALIZATION_GUIDE.md`
- Modify: `docs/CONFIGURATION.md`
- Modify: `docs/workflows/pytorch.md`
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
  tests/torch/test_grid_lines_torch_runner.py -k "rect_s1s2 or training_summary" -q
```

- [ ] **Step 3: Port the settled symbols and run green**

Keep the sampler and record module behavior byte-identical where branch APIs
permit. Adapt only workflow imports/loader integration and branch-local tests.

```bash
python -m pytest \
  tests/torch/test_rect_s1s2_sampling.py \
  tests/torch/test_rect_s1s2_initialization.py \
  tests/torch/test_workflows_components.py \
  tests/torch/test_grid_lines_torch_runner.py -k "rect_s1s2 or training_summary" -q
git diff --check
git add ptycho_torch/rect_s1s2_sampling.py ptycho_torch/rect_s1s2_initialization.py ptycho_torch/workflows/components.py tests/torch/test_rect_s1s2_sampling.py tests/torch/test_rect_s1s2_initialization.py tests/torch/test_workflows_components.py tests/torch/test_grid_lines_torch_runner.py
git commit -m "fix(torch): port representative dose sampling"
```

- [ ] **Step 4: Update normative and routed documentation**

Make `docs/specs/spec-ptycho-core.md` the branch source of truth for the
flat-slot formula, SplitMix64 seed/policy, v2 production, and strict v1 reading.
Update dependent guides/indexes. Preserve `docs/plans/2026-08-04-ci-gauge-invariant-scaling.md`
and its metrics as explicitly prefix-era evidence; do not silently relabel the
historical gauge as a uniform-sample result.

- [ ] **Step 5: Run fno-stable integration tests before the full suite**

```bash
python -m pytest \
  tests/torch/test_grid_lines_ci_probe_roundtrip_integration.py \
  tests/torch/test_integration_workflow_torch.py \
  tests/torch/test_grid_lines_hybrid_resnet_integration.py \
  tests/torch/test_synthetic_hybrid_resnet_gs2_integration.py -q
```

Investigate failures before running `python -m pytest tests -q`. Use the `tmux`
skill and exact-PID/artifact completion guardrail for long GPU runs.

- [ ] **Step 6: Commit branch-owned docs**

```bash
git diff --check
git add docs/specs/spec-ptycho-core.md docs/DATA_NORMALIZATION_GUIDE.md docs/CONFIGURATION.md docs/workflows/pytorch.md docs/TESTING_GUIDE.md docs/development/TEST_SUITE_INDEX.md docs/findings.md docs/index.md scripts/studies/README.md scripts/simulation/README.md
git commit -m "docs(ci): specify representative dose sampling"
```

### Task 6: Fold seeded v2 directly into `refactor-internal`

**Files:**

- Create: `ptycho_torch/rect_s1s2_sampling.py`
- Create or update: `ptycho_torch/rect_s1s2_initialization.py`
- Modify: `ptycho_torch/workflows/components.py`
- Modify/remove through the parent convergence plan: `ptycho_torch/model.py::calibrate_rect_s1s2`
- Create: `tests/torch/test_rect_s1s2_sampling.py`
- Replace/extend: `tests/torch/test_rect_s1s2_initialization.py`
- Modify internal consumer/runner tests
- Update the same full routed doc set listed for `fno-stable`

- [ ] **Step 1: Reconcile with the parent convergence state**

If `refactor-internal` still exposes `rect_s1s2_init="data"`, execute the
relevant config/identity retirement tasks from the parent convergence plan and
port the final seeded-v2 runtime directly. If convergence already landed, apply
only this plan's sampler, record, runtime, tests, and docs. Never port the
prefix solver as an intermediate state.

- [ ] **Step 2: Port the complete red test surface**

Bring across pure selection, v1/v2 record, real-forward, dictionary,
TensorDict, prebuilt mmap, subset leakage, ordering-bias, invalid-input,
module-state, summary, and runner tests. Adapt imports/fixtures to internal
branch APIs without weakening assertions.

- [ ] **Step 3: Run red, implement, and run green**

```bash
python -m pytest \
  tests/torch/test_rect_s1s2_sampling.py \
  tests/torch/test_rect_s1s2_initialization.py \
  tests/torch/test_workflows_components.py \
  tests/torch/test_grid_lines_torch_runner.py -k "rect_s1s2 or training_summary" -q
```

Commit one code/test commit after the settled behavior is green. Do not retain
`rect_s1s2_calibration`, `calibrate_rect_s1s2`, or the one-batch learned-model
path.

- [ ] **Step 4: Apply the full internal documentation update**

Update core, normalization, configuration, workflow, testing, suite index,
findings, index, and both runner READMEs. Documentation exclusions used on
other propagation paths do not apply to `refactor-internal`.

- [ ] **Step 5: Run internal integrations before the full suite**

```bash
python -m pytest \
  tests/torch/test_grid_lines_ci_probe_roundtrip_integration.py \
  tests/torch/test_integration_workflow_torch.py \
  tests/torch/test_grid_lines_hybrid_resnet_integration.py \
  tests/torch/test_synthetic_hybrid_resnet_gs2_integration.py -q
```

Investigate failures before `python -m pytest tests -q`. Use tmux/exact-PID
tracking for long commands, then commit branch-owned docs separately.

### Task 7: Perform the final three-tip consistency audit

**Files:** Read-only across all three branch tips.

- [ ] **Step 1: Compare code contracts**

For each branch, verify:

```text
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

The intended commit shape is three focused code/test commits plus one docs
commit on `refactor`, then one code/test and one docs commit on each ported
branch. Squash only if later requested; do not mix unrelated branch changes.
