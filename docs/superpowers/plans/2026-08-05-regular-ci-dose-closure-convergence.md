# Regular CI Dose-Closure Convergence Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Torch's training-only `ci` profile default to deterministic `dose_closure`, expose the same explicit override on maintained CLIs, and fully retire historical `rect_s1s2_init="data"` on `refactor`, `fno-stable`, and `refactor-internal`.

**Architecture:** Bare `ModelConfig` construction remains `ones`; only the regular-CI profile defaults to `dose_closure`, and explicit callers retain precedence. A single mode validator protects direct construction and every reconstructed model identity; maintained MLflow whole-model loaders call it after unpickling. The representative-sampling amendment replaces the historical prefix solver with the same fixed-seed, 256-slot runtime on all three branches and makes fresh producers emit v2 while retaining strict v1 readers. No alias, general migration framework, schema bump solely for enum retirement, or reconstruction-quality threshold is added.

**Tech Stack:** Python 3.11, dataclasses, argparse, Pydantic mapping adapters, PyTorch, Lightning, MLflow, pytest, Git worktrees.

---

**Design authority:** [`docs/superpowers/specs/2026-08-05-regular-ci-dose-closure-convergence-design.md`](../specs/2026-08-05-regular-ci-dose-closure-convergence-design.md), as amended for sampling and record identity by [`docs/superpowers/specs/2026-08-06-ci-dose-closure-representative-sampling-design.md`](../specs/2026-08-06-ci-dose-closure-representative-sampling-design.md).

**Execution amendment:** Follow [`docs/superpowers/plans/2026-08-06-ci-dose-closure-representative-sampling.md`](2026-08-06-ci-dose-closure-representative-sampling.md) for the interleaved branch order. Tasks 1–2 below are complete. The sampling plan replaces the prefix-runtime portions of Tasks 6–7 and augments Task 10; do not create more worktrees or run duplicate comprehensive suites.

## Branch and path matrix

| Concern | `refactor` | `fno-stable` | `refactor-internal` |
|---|---|---|---|
| Config resolution test | `tests/torch/test_config_resolution_transaction.py` | `tests/torch/test_config_resolution_internal_transaction.py` | `tests/torch/test_config_resolution_internal_transaction.py` |
| Study-runner tests | `tests/torch/test_grid_lines_torch_runner_s1s2_init.py` | Existing nodes in `tests/torch/test_grid_lines_torch_runner.py` | Add/adapt nodes in `tests/torch/test_grid_lines_torch_runner.py` |
| Shared-result test | `tests/test_training_workflow_initialization_summary.py` | Existing result coverage in `tests/scripts/test_training_backend_selector.py` and `tests/torch/test_grid_lines_torch_runner.py` | Port `tests/test_training_workflow_initialization_summary.py` |
| Representative runtime | Replace prefix via sampling Tasks 1–2 | Port settled seeded-v2 runtime after Task 4 | Add inactive seeded-v2 foundation in Task 6, then activate in Task 7 |
| Extra docs | None beyond the shared owners and runner README | Core, normalization, testing, findings, and index docs | Same extra owners as `fno-stable` |

At the audited tips, only these production blobs are identical on all three branches:

- `ptycho_torch/train.py`
- `ptycho_torch/api/api_helper.py`
- `ptycho_torch/api/base_api.py`
- `ptycho_torch/api/mlflow_utils.py`

`tests/torch/test_ci_profile.py` is identical only on `refactor` and `fno-stable`. Config, runtime, fixture, and most identity tests have branch drift and must be adapted branch-locally. Cherry-pick only a commit whose complete touched-file set is verified identical at its parent; otherwise apply the small change manually.

The common retirement error must identify `data` as unsupported, name `ones` and `dose_closure`, and say historical artifacts require historical code or retraining. Pydantic may reject a mapping at its `Literal` boundary first, but it must expose the same two supported spellings and must not translate `data`.

### Task 1: Verify the committed plan, isolate branches, and record baselines

**Status:** Complete. Existing worktrees are the only worktrees used by the
sampling amendment.

**Files:**

- Read: `docs/superpowers/specs/2026-08-05-regular-ci-dose-closure-convergence-design.md`
- Read: this committed plan from the primary checkout
- Preserve untouched: `.claude/`, `docs/plans/NEURIPS-HYBRID-RESNET-2026/`, `scripts/orchestration/`, and `notebooks/archive/ePIE_recon_simulation`

- [ ] **Step 1: Confirm this plan is committed before creating worktrees**

Run in `/home/ollie/Documents/PtychoPINN`:

```bash
git status --short --branch
git log -1 --oneline -- docs/superpowers/plans/2026-08-05-regular-ci-dose-closure-convergence.md
```

Expected: the plan is visible to new worktrees; only the listed user-owned paths remain untracked or modified.

- [ ] **Step 2: Refresh refs without changing branch content**

```bash
git fetch origin refactor
git fetch internal refactor-internal fno-stable
git rev-list --left-right --count refactor...origin/refactor
git rev-list --left-right --count refactor-internal...internal/refactor-internal
git rev-list --left-right --count fno-stable...internal/fno-stable
```

Fast-forward when a branch is only behind. If both sides moved, inspect the graph and reconcile deliberately; do not discard local commits.

- [ ] **Step 3: Create two isolated worktrees**

Use the repository's existing ignored `.worktrees/` directory, as required by
`superpowers:using-git-worktrees`:

```bash
git check-ignore -q .worktrees
git worktree add .worktrees/fno-dose-closure fno-stable
git worktree add .worktrees/internal-dose-closure refactor-internal
git worktree list
```

Keep `refactor` in the primary checkout. Record all three SHAs with `git rev-parse HEAD`.

- [ ] **Step 4: Record focused baselines**

Run in the primary `refactor` checkout:

```bash
python -m pytest tests/torch/test_ci_profile.py tests/torch/test_model_spec.py tests/torch/test_artifact_schema.py tests/torch/test_config_resolution_transaction.py -q
```

Run in each secondary worktree:

```bash
python -m pytest tests/torch/test_ci_profile.py tests/torch/test_model_spec.py tests/torch/test_artifact_schema.py tests/torch/test_config_resolution_internal_transaction.py -q
```

Also run the existing `tests/torch/test_rect_s1s2_initialization.py` on
`refactor` and `fno-stable`. Record failures before editing; investigate rather
than attributing them to this change.

### Task 2: Converge the `refactor` config, identity, and existing runtime

**Status:** Complete in `a1f5d05ef` and `6ab1716ec`, with prerequisite test-only
fix `80261d7a7`. This task intentionally left the then-current prefix runtime
untouched; representative-sampling Tasks 1–3 replace it before final docs and
branch gates.

**Files:**

- Modify: `ptycho_torch/rect_s1s2_initialization.py`
- Modify: `ptycho_torch/config_params.py`
- Modify: `ptycho_torch/config_factory.py`
- Modify: `ptycho_torch/config_resolution.py`
- Modify only for short-data guidance: `ptycho_torch/workflows/components.py`
- Test: `tests/torch/test_ci_profile.py`
- Test: `tests/torch/test_model_spec.py`
- Test: `tests/torch/test_artifact_schema.py`
- Test: `tests/torch/test_rect_s1s2_initialization.py`
- Test: `tests/torch/test_config_pydantic_artifacts.py`
- Test: `tests/test_synthetic_workflow_config.py`
- Modify: `tests/fixtures/config/generate_pre_migration_fixtures.py`
- Modify: `tests/fixtures/config/README.md`

- [ ] **Step 1: Write the failing contract tests**

Require all of the following:

```python
assert ModelConfig().rect_s1s2_init == "ones"
assert resolve_ci_profile()["rect_s1s2_init"] == "dose_closure"
assert resolve_ci_profile({"rect_s1s2_init": "ones"})["rect_s1s2_init"] == "ones"
```

Make the training payload and its derived `ModelSpec` carry `dose_closure` when `profile="ci"`. Add rejection cases for:

- direct `ModelConfig(rect_s1s2_init="data")`;
- an authored structured mapping with
  `model.rect_s1s2_init="data"` through the synthetic resolver;
- `ModelSpec.from_payload()` after mutating a valid payload to `data`;
- `decode_artifact_identity()` after the same mutation; and
- `PtychoPINN_Lightning.load_from_checkpoint()` after mutating the ModelSpec under `hyper_parameters` in a valid zero-epoch checkpoint.

The checkpoint case must fail at identity reconstruction, before state restoration.

- [ ] **Step 2: Preserve historical fixture bytes but change current expectations**

Do not rewrite the frozen pre-Pydantic JSON. Change its generator to construct a valid `ones` model and mutate only the serialized payload to historical `data` after encoding. Keep field-set and byte-reproduction assertions; current decoding must now reject the fixture. Test tensor-tag behavior on a copied payload changed to supported `ones`, so wire-format coverage remains independent of the retired value. Explain that intent in the fixture README.

- [ ] **Step 3: Make the existing runtime test exercise profile omission**

In `tests/torch/test_rect_s1s2_initialization.py`, let the resolved workflow fixture omit the initialization override. Require the omitted `ci` case to initialize before fit with `dose_closure` and return the same strict record written to `training_summary.json`. Keep an explicit `ones` control that consumes no loader. Strengthen short-data assertions to include sampled count, required count `256`, and `--rect-s1s2-init ones`.

- [ ] **Step 4: Run the red tests**

```bash
python -m pytest \
  tests/torch/test_ci_profile.py \
  tests/torch/test_model_spec.py \
  tests/torch/test_artifact_schema.py \
  tests/torch/test_config_pydantic_artifacts.py \
  tests/test_synthetic_workflow_config.py \
  tests/torch/test_rect_s1s2_initialization.py -q
```

Expected: failures expose the old profile default, unvalidated dataclass construction/identity reconstruction, and missing short-data guidance. Existing deterministic solver tests should not reveal a need for a second algorithm.

- [ ] **Step 5: Add one validator and change only the profile default**

In `rect_s1s2_initialization.py`:

```python
RECT_S1S2_INITIALIZATION_MODES = ("ones", "dose_closure")


def validate_rect_s1s2_initialization_mode(mode: object) -> str:
    if mode in RECT_S1S2_INITIALIZATION_MODES:
        return str(mode)
    if mode == "data":
        raise ValueError(
            "rect_s1s2_init='data' is no longer supported; use 'ones' or "
            "'dose_closure'. Historical 'data' artifacts require historical "
            "code or retraining."
        )
    raise ValueError(
        "rect_s1s2_init must be 'ones' or 'dose_closure', "
        f"got {mode!r}"
    )
```

Call it from the record's existing `_validated_values()` and from `ModelConfig.__post_init__`; preserve all existing generator validation there. Make resolver defense delegate to this helper. Set only `CI_PROFILE_BUNDLE["rect_s1s2_init"] = "dose_closure"`; keep the bare dataclass default `ones` and do not add this field to the five locked CI fields.

Extend only the insufficient-pattern error in `workflows/components.py`; this
completed config slice did not change the solver. The sampling amendment, not
this task, replaces the historical prefix before the branch is final.

- [ ] **Step 6: Run green tests and commit exact files**

```bash
python -m pytest \
  tests/torch/test_ci_profile.py \
  tests/torch/test_model_spec.py \
  tests/torch/test_artifact_schema.py \
  tests/torch/test_config_pydantic_artifacts.py \
  tests/torch/test_config_resolution_transaction.py \
  tests/test_synthetic_workflow_config.py \
  tests/torch/test_rect_s1s2_initialization.py \
  tests/test_training_workflow_initialization_summary.py \
  tests/torch/test_grid_lines_torch_runner_s1s2_init.py -q
git diff --check
git add ptycho_torch/rect_s1s2_initialization.py ptycho_torch/config_params.py ptycho_torch/config_factory.py ptycho_torch/config_resolution.py ptycho_torch/workflows/components.py tests/torch/test_ci_profile.py tests/torch/test_model_spec.py tests/torch/test_artifact_schema.py tests/torch/test_rect_s1s2_initialization.py tests/torch/test_config_pydantic_artifacts.py tests/test_synthetic_workflow_config.py tests/fixtures/config/generate_pre_migration_fixtures.py tests/fixtures/config/README.md
git commit -m "fix(config): converge CI initialization contract"
```

### Task 3: Add `refactor` CLI authorship, MLflow validation, and user docs

**Files:**

- Modify: `ptycho_torch/train.py`
- Modify: `ptycho_torch/api/api_helper.py`
- Modify: `ptycho_torch/api/base_api.py`
- Modify: `ptycho_torch/api/mlflow_utils.py`
- Test: `tests/torch/test_ci_profile.py`
- Test: `tests/torch/test_api_config_retirement.py`
- Modify: `docs/CONFIGURATION.md`
- Modify: `docs/workflows/pytorch.md`
- Modify: `docs/COMMANDS_REFERENCE.md`
- Modify only if its runner wording is stale: `scripts/studies/README.md`

- [ ] **Step 1: Write failing CLI and whole-model reload tests**

Require native help to expose `--rect-s1s2-init {ones,dose_closure}`. Parameterize native execution for omitted flag → profile `dose_closure`, explicit `ones` → `ones`, and explicit `dose_closure` → `dose_closure`. In the factory spy, prove omission does not author an override.

For MLflow, construct a valid `ModelConfig`, mutate its field to `data` after construction, attach it to a fake loaded model, and mock `mlflow.pytorch.load_model`. Require the common retirement error from:

- `api_helper.load_with_mlflow`;
- `PtychoModel.load_from_mlflow`;
- `mlflow_utils.load_model_from_mlflow`; and
- `mlflow_utils.load_model_and_configs`.

Patch registry/config lookups so tests make no network calls.

- [ ] **Step 2: Run the red boundary tests**

```bash
python -m pytest tests/torch/test_ci_profile.py tests/torch/test_api_config_retirement.py -q
```

- [ ] **Step 3: Implement omission-aware CLI forwarding and post-unpickle validation**

Add the argparse option with choices `("ones", "dose_closure")` and `default=None`. Add `overrides["rect_s1s2_init"]` only when the flag is not `None`; leave resolved profile forwarding unchanged. Help must distinguish startup dose closure from later scale trainability.

Immediately after each maintained `mlflow.pytorch.load_model(...)`, validate `model.model_config.rect_s1s2_init` with the common helper. Do not catch, translate, or mutate the invalid object.

- [ ] **Step 4: Verify and commit the reusable boundary change**

```bash
python -m pytest tests/torch/test_ci_profile.py tests/torch/test_api_config_retirement.py -q
python -m ptycho_torch.train --help
git diff --check
git add ptycho_torch/train.py ptycho_torch/api/api_helper.py ptycho_torch/api/base_api.py ptycho_torch/api/mlflow_utils.py tests/torch/test_ci_profile.py tests/torch/test_api_config_retirement.py
git commit -m "feat(cli): expose CI gauge initialization"
```

Record this commit SHA; it is the only planned cherry-pick candidate.

- [ ] **Step 5: Update and commit owning docs**

Document the bare/profile distinction, explicit `ones` precedence, fixed-seed
uniform selection of exactly 256 logical detector slots with no fallback,
fresh v2 records plus strict historical v1 reading, native examples, and
non-migrating `data` retirement. Do not create `docs/index.md` on `refactor` or
duplicate the algorithm in runner docs.

```bash
git diff --check
git grep -n -E 'dose_closure|rect-s1s2-init|historical.*data|256' -- docs/CONFIGURATION.md docs/workflows/pytorch.md docs/COMMANDS_REFERENCE.md scripts/studies/README.md
git add docs/CONFIGURATION.md docs/workflows/pytorch.md docs/COMMANDS_REFERENCE.md scripts/studies/README.md
git commit -m "docs(ci): explain dose-closure defaults"
```

If `scripts/studies/README.md` did not change, omit it from `git add`.

### Task 4: Port the core contract to fno-stable with a real red/green gate

**Files:**

- Modify: ptycho_torch/rect_s1s2_initialization.py
- Modify: ptycho_torch/config_params.py
- Modify: ptycho_torch/config_factory.py
- Modify: ptycho_torch/config_resolution.py
- Modify only for short-data guidance: ptycho_torch/workflows/components.py
- Test: tests/torch/test_ci_profile.py
- Test: tests/torch/test_model_spec.py
- Test: tests/torch/test_artifact_schema.py
- Test: tests/torch/test_config_pydantic_artifacts.py
- Test: tests/torch/test_rect_s1s2_initialization.py
- Test: tests/test_synthetic_workflow_config.py
- Modify: tests/fixtures/config/generate_pre_migration_fixtures.py
- Modify: tests/fixtures/config/README.md

- [ ] **Step 1: Adapt only tests and historical-fixture expectations**

In .worktrees/fno-dose-closure, add the same bare/profile/explicit, direct
construction, structured mapping, ModelSpec, artifact, checkpoint, frozen
fixture, omitted-profile runtime, and short-data assertions from Task 2. Use
fno-stable's existing fixture filenames and
tests/torch/test_config_resolution_internal_transaction.py. Do not edit
production yet.

- [ ] **Step 2: Run the branch-native red gate**

~~~bash
python -m pytest \
  tests/torch/test_ci_profile.py \
  tests/torch/test_model_spec.py \
  tests/torch/test_artifact_schema.py \
  tests/torch/test_config_pydantic_artifacts.py \
  tests/torch/test_config_resolution_internal_transaction.py \
  tests/torch/test_rect_s1s2_initialization.py \
  tests/test_synthetic_workflow_config.py -q
~~~

Expected: failures identify the old profile default, unvalidated direct/decoded
data, and missing short-data guidance. Existing deterministic-solver cases stay
green.

- [ ] **Step 3: Implement the branch-local core patch**

Apply Task 2's common validator, profile-only default change, resolver defense,
frozen-fixture strategy, and short-data guidance by symbol. Do not copy whole
files from refactor. This config commit may leave the historical v1 prefix
runtime temporarily intact; sampling-plan Task 4 replaces it before boundary
docs or branch gates.

- [ ] **Step 4: Rerun green and commit exact files**

~~~bash
python -m pytest \
  tests/torch/test_ci_profile.py \
  tests/torch/test_model_spec.py \
  tests/torch/test_artifact_schema.py \
  tests/torch/test_config_pydantic_artifacts.py \
  tests/torch/test_config_resolution_internal_transaction.py \
  tests/torch/test_rect_s1s2_initialization.py \
  tests/torch/test_workflows_components.py \
  tests/test_synthetic_workflow_config.py -q
git diff --check
git add ptycho_torch/rect_s1s2_initialization.py ptycho_torch/config_params.py ptycho_torch/config_factory.py ptycho_torch/config_resolution.py ptycho_torch/workflows/components.py tests/torch/test_ci_profile.py tests/torch/test_model_spec.py tests/torch/test_artifact_schema.py tests/torch/test_config_pydantic_artifacts.py tests/torch/test_rect_s1s2_initialization.py tests/test_synthetic_workflow_config.py tests/fixtures/config/generate_pre_migration_fixtures.py tests/fixtures/config/README.md
git commit -m "fix(config): converge fno-stable CI initialization"
~~~

### Task 5: Apply fno-stable entry boundaries and branch-owned docs

**Files:**

- Reuse or adapt the six Task 3 CLI/API/test files
- Test: tests/scripts/test_training_backend_selector.py
- Test: tests/torch/test_grid_lines_torch_runner.py
- Modify docs: docs/CONFIGURATION.md, docs/workflows/pytorch.md,
  docs/COMMANDS_REFERENCE.md, docs/specs/spec-ptycho-core.md,
  docs/DATA_NORMALIZATION_GUIDE.md, docs/TESTING_GUIDE.md,
  docs/development/TEST_SUITE_INDEX.md, docs/findings.md, docs/index.md
- Modify a runner README only if its field wording is stale

- [ ] **Step 1: Prove the boundary cherry-pick is exact or port manually**

Resolve the Task 3 SHA from the primary refactor history, require a clean
fno-stable worktree, assert the complete touched set, and compare every parent
blob before cherry-picking:

~~~bash
BOUNDARY_SHA="$(git -C /home/ollie/Documents/PtychoPINN log -1 --format=%H --fixed-strings --grep='feat(cli): expose CI gauge initialization' refactor)"
test -n "$BOUNDARY_SHA"
test -z "$(git status --porcelain)"
EXPECTED_PATHS="$(printf '%s\n' ptycho_torch/api/api_helper.py ptycho_torch/api/base_api.py ptycho_torch/api/mlflow_utils.py ptycho_torch/train.py tests/torch/test_api_config_retirement.py tests/torch/test_ci_profile.py)"
ACTUAL_PATHS="$(git diff-tree --no-commit-id --name-only -r "$BOUNDARY_SHA" | sort)"
test "$ACTUAL_PATHS" = "$EXPECTED_PATHS"
git diff --exit-code "${BOUNDARY_SHA}^" HEAD -- ptycho_torch/api/api_helper.py ptycho_torch/api/base_api.py ptycho_torch/api/mlflow_utils.py ptycho_torch/train.py tests/torch/test_api_config_retirement.py tests/torch/test_ci_profile.py
git cherry-pick "$BOUNDARY_SHA"
~~~

If any assertion or diff fails, abort the cherry-pick path. Instead,
write/adapt the CLI and MLflow tests first, run them red, apply the four small
production edits manually, and rerun green. Never resolve drift by taking an
entire side.

- [ ] **Step 2: Verify boundaries and branch-native result consumers**

~~~bash
python -m pytest tests/torch/test_ci_profile.py tests/torch/test_api_config_retirement.py tests/scripts/test_training_backend_selector.py -q
python -m pytest \
  tests/torch/test_grid_lines_torch_runner.py::TestRunGridLinesTorchScaffold::test_runner_exposes_training_summary_and_records_same_initialization \
  tests/torch/test_grid_lines_torch_runner.py::test_main_forwards_dose_closure_rect_s1s2_init \
  tests/torch/test_grid_lines_torch_runner.py::test_grid_training_rejects_dose_closure_on_legacy_contract \
  tests/torch/test_grid_lines_torch_runner.py::test_grid_lines_rect_s1s2_help_separates_ci_and_legacy_count_scaling -q
python -m ptycho_torch.train --help
~~~

If the boundary was ported manually, stage exactly the six listed files and
commit them before documentation.

- [ ] **Step 3: Update and commit fno-stable documentation**

Replace the current finding that generic CI initializes with ones, retain
historical findings as historical, and keep docs/index.md as routing rather
than policy duplication.

~~~bash
git diff --check
git grep -n -i -E 'generic CI.*ones|training-only.*ones|one.batch.*calibrat|data.calibrat|rect_s1s2_init.*data' -- docs scripts ptycho ptycho_torch tests
git add docs/CONFIGURATION.md docs/workflows/pytorch.md docs/COMMANDS_REFERENCE.md docs/specs/spec-ptycho-core.md docs/DATA_NORMALIZATION_GUIDE.md docs/TESTING_GUIDE.md docs/development/TEST_SUITE_INDEX.md docs/findings.md docs/index.md
git commit -m "docs(ci): converge fno-stable gauge semantics"
~~~

Stage scripts/studies/README.md or scripts/simulation/README.md explicitly only
if changed.

### Task 6: Add the inactive seeded-v2 runtime foundation to refactor-internal

This task is deliberately additive. It does not change the internal profile,
remove data, or wire the training entry yet, so its commit cannot advertise a
mode the runtime silently skips.

**Files:**

- Create: ptycho_torch/rect_s1s2_sampling.py
- Create: ptycho_torch/rect_s1s2_initialization.py
- Modify: ptycho_torch/scaling_contract.py
- Modify: ptycho_torch/workflows/components.py
- Modify: ptycho_torch/train_utils.py
- Create: tests/torch/test_rect_s1s2_sampling.py
- Extend: tests/torch/test_rect_s1s2_initialization.py
- Modify only for helper-level expectations: tests/torch/test_workflows_components.py

- [ ] **Step 1: Port only record, algorithm, and loader tests**

Adapt settled refactor coverage for pinned SplitMix64 selection, subset/mmap
bounds, strict v1/v2 record validation, seeded known-gauge forward, exact
256-slot channel masking, dict/TensorDict/prebuilt mmap loaders, no-loader ones,
invalid values/shapes/counts, and nested train/eval restoration. Include a real
prebuilt-mmap case with `num_workers=0`; the data module must omit
`prefetch_factor` in that case. Keep the old data-calibration and CI-profile
tests green temporarily.
Defer the omitted-profile training-entry, strict-result, and runner tests to
Task 7.

- [ ] **Step 2: Run the new helper tests red**

~~~bash
python -m pytest tests/torch/test_rect_s1s2_sampling.py tests/torch/test_rect_s1s2_initialization.py tests/torch/test_workflows_components.py -q
~~~

Expected: existing data tests pass and new cases fail because the record and
deterministic helpers do not exist.

- [ ] **Step 3: Port the settled foundation without activating it**

Port the pure fixed-seed sampler and the strict v1/v2 record with fresh v2
constructors, record-local `ones|dose_closure` validation, and
validate_rect_s1s2_initialization_contract. Do not
export or attach the common config-mode validator until Task 7, because its
retirement error would contradict the still-active data configuration in this
intermediate commit. Add these settled helpers to the
internal workflow module:

~~~text
private selected-row ordinary/TensorDict loader helpers
_initialize_rect_s1s2_unmanaged
_initialize_rect_s1s2
_write_training_summary_atomic
_publish_training_summary_and_barrier
_rect_s1s2_training_loader
~~~

Do not change ModelConfig, CI_PROFILE_BUNDLE, _train_with_lightning, the old
data branch, or calibrate_rect_s1s2 in this commit. Do not port a prefix helper
or a callable seed/sample-count override. Fix the existing zero-worker mmap
path in `train_utils.py` by omitting `prefetch_factor`; defer
`_TrainingSummaryCallback` and all training-entry wiring to Task 7.

- [ ] **Step 4: Run green helper tests and commit**

~~~bash
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
~~~

### Task 7: Activate and fully retire internal data calibration atomically

**Files:**

- Modify: ptycho_torch/rect_s1s2_initialization.py
- Modify: ptycho_torch/config_params.py
- Modify: ptycho_torch/config_factory.py
- Modify: ptycho_torch/config_resolution.py
- Modify: ptycho_torch/workflows/components.py
- Modify: ptycho_torch/model.py
- Modify: ptycho/workflows/synthetic_config.py
- Modify: ptycho/workflows/training.py
- Modify: scripts/studies/grid_lines_torch_runner.py
- Create: tests/torch/test_ci_container_bridge.py
- Create: tests/test_training_workflow_initialization_summary.py
- Modify internal config, identity, runtime, synthetic mapping, shared-result,
  existing grid-runner, and tests/studies/test_torch_ablation_manifest.py tests

The checked-in pre-migration fixture already records `ones`. Keep its generator,
README, and serialized bytes unchanged; mutate copied payloads inside rejection
tests when a historical `data` value is required.

- [ ] **Step 1: Write the failing atomic-transition tests**

Require:

- bare ones, CI-profile dose_closure, and explicit ones precedence;
- direct config, structured mapping, ModelSpec, artifact, and checkpoint data
  rejection;
- the amplitude-only synthetic profile remains ones and rejects both authored
  data and incoherent dose_closure;
- the shared workflow adapts raw grouped diffraction counts into the CI
  container before initialization instead of estimating from normalized `X`;
- an omitted CI initialization invokes seeded dose closure before fit and
  publishes one strict fresh-v2 record;
- the shared workflow and existing grid runner expose
  rect_s1s2_initialization and training_summary_path, never
  rect_s1s2_calibration; and
- the ablation manifest uses `dose_closure` as its valid nondefault mode.

Port tests/test_training_workflow_initialization_summary.py. Add/adapt the
runner cases in tests/torch/test_grid_lines_torch_runner.py rather than creating
the refactor-only runner module.

- [ ] **Step 2: Run the transition suite red**

~~~bash
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
~~~

- [ ] **Step 3: Make the state transition in one implementation slice**

Add the common config-mode validator to rect_s1s2_initialization.py and make its
record validation delegate to it. Change ModelConfig to
Literal["ones", "dose_closure"] with bare default ones and constructor
validation. Set only the regular-CI profile default to dose_closure and
delegate resolver defense to the same validator.

Add `_TrainingSummaryCallback`, wire `_train_with_lightning` to the Task 6
seeded-v2 initializer before `trainer.fit`, publish/return the strict record and
summary path, and delete the data branch and `rect_s1s2_calibration` result.
Remove `calibrate_rect_s1s2`, `_last_calibration_means`, and
`_loss_target_intensity` after confirming the transition leaves no callers.

Preserve raw grouped diffraction counts in the shared training container and
adapt them into `measured_intensity` at the CI boundary. Never derive closure
from normalized model inputs.

Exercise historical `data` rejection by mutating in-memory copies of the
already-frozen fixture; do not edit the fixture generator, README, or bytes. In
synthetic_config.py, invoke validate_rect_s1s2_initialization_contract at
resolution and public revalidation boundaries, matching refactor. This rejects
dose_closure under the branch's amplitude-only synthetic profile. Do not add a
CI synthetic profile, synthetic flag, stage-manifest migration, or schema bump.

Propagate the strict backend result through ptycho/workflows/training.py and the
existing grid runner. The standalone runner default remains ones; inference
rect_s1s2_refit remains unchanged.

- [ ] **Step 4: Run green and commit the atomic transition**

~~~bash
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
git add ptycho_torch/rect_s1s2_initialization.py ptycho_torch/config_params.py ptycho_torch/config_factory.py ptycho_torch/config_resolution.py ptycho_torch/workflows/components.py ptycho_torch/model.py ptycho/workflows/synthetic_config.py ptycho/workflows/training.py scripts/studies/grid_lines_torch_runner.py tests/torch/test_ci_profile.py tests/torch/test_config_resolution_internal_transaction.py tests/torch/test_model_spec.py tests/torch/test_artifact_schema.py tests/torch/test_artifact_schema_v2.py tests/torch/test_config_pydantic_artifacts.py tests/torch/test_rect_s1s2_initialization.py tests/torch/test_workflows_components.py tests/torch/test_ci_container_bridge.py tests/test_synthetic_workflow_config.py tests/test_training_workflow_initialization_summary.py tests/torch/test_grid_lines_torch_runner.py tests/studies/test_torch_ablation_manifest.py
git commit -m "fix(torch): replace data calibration with seeded dose closure"
~~~

### Task 8: Close internal native CLI and MLflow boundaries

**Files:**

- Modify: ptycho_torch/train.py
- Modify: ptycho_torch/api/api_helper.py
- Modify: ptycho_torch/api/base_api.py
- Modify: ptycho_torch/api/mlflow_utils.py
- Test: tests/torch/test_ci_profile.py
- Test: tests/torch/test_api_config_retirement.py
- Test: tests/torch/test_cli_train_torch.py

- [ ] **Step 1: Write and run failing boundary tests**

Port/adapt Task 3's omission-aware native CLI/help cases and all four mocked
post-unpickle data rejections.

~~~bash
python -m pytest tests/torch/test_ci_profile.py tests/torch/test_api_config_retirement.py tests/torch/test_cli_train_torch.py -q
~~~

Expected: core construction already rejects data, while the absent native flag
and unvalidated whole-model loaders fail these focused cases.

- [ ] **Step 2: Apply the four production edits manually**

The production parents matched at audit time, but internal test_ci_profile.py
did not. Add the native option with default None and explicit-only forwarding;
validate immediately after each maintained MLflow whole-model load. Do not
cherry-pick a mixed commit.

- [ ] **Step 3: Verify and commit boundaries**

~~~bash
python -m pytest tests/torch/test_ci_profile.py tests/torch/test_api_config_retirement.py tests/torch/test_cli_train_torch.py -q
python -m ptycho_torch.train --help
git diff --check
git add ptycho_torch/train.py ptycho_torch/api/api_helper.py ptycho_torch/api/base_api.py ptycho_torch/api/mlflow_utils.py tests/torch/test_ci_profile.py tests/torch/test_api_config_retirement.py tests/torch/test_cli_train_torch.py
git commit -m "feat(cli): expose internal CI gauge initialization"
~~~

### Task 9: Update refactor-internal documentation

**Files:**

- Modify: docs/CONFIGURATION.md
- Modify: docs/workflows/pytorch.md
- Modify: docs/COMMANDS_REFERENCE.md
- Modify: docs/specs/spec-ptycho-core.md
- Modify: docs/DATA_NORMALIZATION_GUIDE.md
- Modify: docs/TESTING_GUIDE.md
- Modify: docs/development/TEST_SUITE_INDEX.md
- Modify: docs/findings.md
- Modify: docs/index.md
- Modify either runner README only where stale

- [ ] **Step 1: Remove supported data-calibration wording**

Document the same bare/profile/explicit precedence, fixed-seed uniform
selection of exactly 256 logical detector slots, no fallback, fresh v2/strict
historical v1 records, native spelling, and historical data retirement. Update
current findings and test catalogs; keep the index as routing.

- [ ] **Step 2: Sweep, validate, and commit exact docs**

~~~bash
git grep -n -i -E 'rect_s1s2_init.*data|one.batch.*calibrat|data.calibrat|rect_s1s2_calibration' -- ptycho ptycho_torch scripts docs tests
git diff --check
git add docs/CONFIGURATION.md docs/workflows/pytorch.md docs/COMMANDS_REFERENCE.md docs/specs/spec-ptycho-core.md docs/DATA_NORMALIZATION_GUIDE.md docs/TESTING_GUIDE.md docs/development/TEST_SUITE_INDEX.md docs/findings.md docs/index.md
git commit -m "docs(ci): converge internal gauge semantics"
~~~

Stage scripts/studies/README.md or scripts/simulation/README.md explicitly only
if changed.

### Task 10: Run integrations, comprehensive gates, and final review

**Files:** Test and inspect only unless a failure is causally traced to this
work.

- [ ] **Step 1: Rerun every branch's focused gate**

Reuse the exact green commands in Tasks 2–3, 4–5, or 6–8. Run the
branch-present synthetic pipeline/config tests, the branch's
`tests/torch/test_rect_s1s2_sampling.py`, and inspect
`python -m ptycho_torch.train --help`. Include this omitted-profile node on
every branch:

~~~bash
python -m pytest tests/torch/test_rect_s1s2_initialization.py::test_training_entry_initializes_before_fit_and_persists_same_summary_record -q
~~~

It must resolve CI to dose_closure, initialize before fit, and persist/return
identical records.

- [ ] **Step 2: Run integration modules before any comprehensive suite**

On all three branches:

~~~bash
python -m pytest tests/torch/test_grid_lines_ci_probe_roundtrip_integration.py tests/torch/test_integration_workflow_torch.py -q
~~~

On fno-stable and refactor-internal, also run:

~~~bash
python -m pytest tests/torch/test_grid_lines_hybrid_resnet_integration.py tests/torch/test_synthetic_hybrid_resnet_gs2_integration.py -q
~~~

Diagnose and fix any failure before moving to the full suite. Reproduce an
alleged environment-only failure at the recorded branch baseline before
excluding it.

- [ ] **Step 3: Run each comprehensive gate**

Only after integrations pass:

~~~bash
python -m pytest -q
~~~

Record totals, skips, duration, and exact failing nodes.

- [ ] **Step 4: Audit all three final tips**

~~~bash
git grep -n -i -E 'rect_s1s2_init.*data|one.batch.*calibrat|rect_s1s2_calibration' -- ptycho ptycho_torch scripts docs
git grep -n 'rect_s1s2_init' -- ptycho_torch/config_params.py ptycho_torch/config_factory.py ptycho_torch/train.py
git grep -n 'RECT_S1S2_INITIALIZATION_SCHEMA\|RECT_S1S2_DOSE_CLOSURE_PATTERNS' -- ptycho_torch/rect_s1s2_initialization.py
git grep -n 'RECT_S1S2_DOSE_CLOSURE_SAMPLE_SEED\|RECT_S1S2_DOSE_CLOSURE_SAMPLE_POLICY' -- ptycho_torch/rect_s1s2_initialization.py
git diff --check
git status --short --branch
git log --oneline --decorate -8
~~~

Classify explicit historical-retirement prose; no supported data path may
remain. Confirm bare ones, profile dose_closure, explicit ones precedence, two
CLI choices, fixed count 256, seed 20260806, policy
splitmix64_rejection_v1, fresh schema v2/method
dose_closure_seeded_uniform_unit_object, strict historical v1 reading, logical
`row * C + channel` mapping, strict result persistence, small
branch-appropriate histories, and no unrelated staged files.

- [ ] **Step 5: Review before push**

Use superpowers:requesting-code-review on the complete three-branch diff.
Resolve spec compliance before code quality, rerun affected tests after every
correction, and use superpowers:verification-before-completion before claiming
success or pushing.
