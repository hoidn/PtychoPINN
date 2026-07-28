# Public Configuration Resolution Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Resolve public ModelConfig, TrainingConfig, and InferenceConfig sources through one deterministic, side-effect-free boundary with explicit CLI precedence, canonical `n_groups`, and separate structural, runnable, and resource validation.

**Architecture:** Add a focused public resolution module that owns file/CLI merge, field ownership, duplicate handling, and deprecated grouping aliases while returning unchanged stdlib dataclasses. Training and inference entry points consume it and preserve explicit CLI suppliedness. Validation remains manual and layered; Pydantic, persistence, and the legacy projection remain unchanged.

**Tech Stack:** Python stdlib dataclasses and argparse, PyYAML, pathlib, pytest.

**Execution constraint:** Work in the current checkout. Do not create a
worktree; the repository `AGENTS.md` prohibition overrides generic execution
skill advice. Immediately before Task 1, the execution controller records the
current commit as the implementation-start SHA for Task 6's exclusion check.

---

## File Map

- Create `ptycho/config/resolution.py`: pure family-specific source resolvers and public validation layers.
- Modify `ptycho/config/__init__.py`: export the supported resolver/validator entry points.
- Modify `ptycho/workflows/components.py`: training parser/setup integration and duplicate parser reduction.
- Modify `ptycho/workflows/backend_selector.py`: validate, bridge, then dispatch
  already-resolved public records in CONFIG-001 order.
- Modify `scripts/training/train.py`: consume the shared public config argument builder.
- Modify `scripts/inference/inference.py`: complete inference file/CLI resolution and canonical grouping.
- Modify `ptycho/config/config.py`: retain the exact compatibility behavior of
  the exported legacy validation facades.
- Add `tests/test_public_config_resolution.py`: pure source-resolution and validation contract tests.
- Modify `tests/test_model_config_architecture.py`: focused public architecture
  and object-policy validation.
- Modify focused training/inference CLI and backend-selector tests only.
- Modify `docs/CONFIGURATION.md`: document the public source precedence,
  grouping alias, and validation layers.

### Task 1: Implement pure public source resolution

**Files:**

- Create: `ptycho/config/resolution.py`
- Create: `tests/test_public_config_resolution.py`
- Modify: `ptycho/config/__init__.py`

- [ ] **Step 1: Write failing source-resolution tests**

Cover:

```python
def test_training_file_value_survives_omitted_cli_value():
    config = resolve_training_config(
        {"nepochs": 9, "model": {"N": 128}},
        {},
    )
    assert config.nepochs == 9
    assert config.model.N == 128


def test_explicit_cli_value_overrides_file():
    config = resolve_training_config(
        {"nepochs": 9},
        {"nepochs": 3},
    )
    assert config.nepochs == 3


def test_equal_flat_and_nested_model_values_are_canonicalized_once():
    config = resolve_training_config(
        {"N": 128, "model": {"N": 128}},
        {},
    )
    assert config.model.N == 128


def test_conflicting_flat_and_nested_model_values_fail():
    with pytest.raises(ValueError, match="N.*flat.*model"):
        resolve_training_config(
            {"N": 64, "model": {"N": 128}},
            {},
        )


def test_cli_flat_model_value_overrides_file_nested_value():
    config = resolve_training_config(
        {"model": {"N": 64}},
        {"N": 128},
    )
    assert config.model.N == 128


def test_cli_nested_model_value_overrides_file_flat_value():
    config = resolve_training_config(
        {"N": 64},
        {"model": {"N": 128}},
    )
    assert config.model.N == 128
```

Also test sorted unknown root/nested names, unchanged input mappings, Path
conversion, fresh returned dataclasses, and backend-aware object-policy
resolution. The public spelling `backend="pytorch"` must be mapped to the
object-policy helper's `backend="torch"`; TensorFlow remains `"tensorflow"`.
Add structural-invalid cases (architecture, activation, gridsize, and a
backend/object-policy mismatch) so a resolver cannot return an unvalidated
candidate. Test root-envelope rejection for falsy non-mappings such as `[]`
and nested `model` values that are not mappings.

- [ ] **Step 2: Run the pure tests and verify RED**

```bash
python -m pytest tests/test_public_config_resolution.py \
  -k "training or inference" -q
```

Expected: import failure because the resolution module does not exist.

- [ ] **Step 3: Implement structural validation and narrow family resolvers**

First implement and export from `ptycho.config` the no-filesystem validators
needed by this task:

```python
validate_model_config_structure(config)
validate_training_config_structure(config)
validate_inference_config_structure(config)
```

They own the approved local types, closed domains, ranges, and semantic joins.
Task 5 adds runnable/resource APIs and compatibility facades; it does not defer
the structural validation required for Task 1 to reach GREEN.

Normalize and partition each source independently before applying precedence:

```python
def resolve_training_config(
    file_mapping: Mapping[str, Any] | None,
    explicit_cli_patch: Mapping[str, Any] | None,
) -> TrainingConfig:
    file_values = _normalize_public_source(
        {} if file_mapping is None else file_mapping,
        source="file",
        workflow_type=TrainingConfig,
    )
    cli_values = _normalize_public_source(
        {} if explicit_cli_patch is None else explicit_cli_patch,
        source="explicit CLI",
        workflow_type=TrainingConfig,
    )
    candidate = _construct_training_candidate(file_values, cli_values)
    validate_training_config_structure(candidate)
    return candidate
```

Within one source, equal flat/nested duplicates are accepted once and unequal
duplicates fail. Across sources, a canonicalized explicit CLI value replaces
the canonicalized file value regardless of flat/nested location. Use explicit
public-family owner sets (they may be asserted against stdlib dataclass fields)
and implement the equivalent inference resolver. Resolve model object policy
using the selected workflow backend (`pytorch -> torch`) and perform structural
validation before returning.

Validate the root and optional nested `model` envelopes with `isinstance(...,
Mapping)` before truthiness/default handling. Only `None` means an absent
source.

Keep helpers public-family specific; do not introduce a registry shared with
Torch, simulation, or execution config. Do not validate with Pydantic.

- [ ] **Step 4: Run pure resolution tests and verify GREEN**

```bash
python -m pytest tests/test_public_config_resolution.py -q
```

Expected: all pure source tests pass.

- [ ] **Step 5: Commit**

```bash
git add ptycho/config/resolution.py ptycho/config/__init__.py \
  tests/test_public_config_resolution.py
git commit -m "feat(config): resolve public configuration sources"
```

### Task 2: Make grouping alias resolution explicit

**Files:**

- Modify: `ptycho/config/resolution.py`
- Modify: `tests/test_public_config_resolution.py`

- [ ] **Step 1: Write failing alias tests**

Add cases proving:

- `n_images` alone resolves to `n_groups`;
- equal `n_images` and `n_groups` are accepted once;
- unequal values fail with both names;
- file `n_images` followed by CLI `n_groups` uses the CLI value without a
  false cross-source conflict;
- file `n_groups` followed by CLI `n_images` uses the CLI value without a
  false cross-source conflict;
- one deprecation warning is emitted only after successful resolution; and
- direct `TrainingConfig` and `InferenceConfig` construction retains current
  `__post_init__` compatibility.

- [ ] **Step 2: Run alias tests and verify RED**

```bash
python -m pytest tests/test_public_config_resolution.py \
  -k "n_images or n_groups" -q
```

Expected: conflicting aliases do not yet fail at the source boundary.

- [ ] **Step 3: Implement boundary alias resolution**

Resolve the alias independently inside each source before file/CLI precedence:

```python
def _resolve_group_alias(values, *, source):
    resolved = dict(values)
    if "n_images" not in resolved:
        return resolved, False
    legacy = resolved["n_images"]
    canonical = resolved.get("n_groups")
    if canonical is not None and legacy is not None and canonical != legacy:
        raise ValueError(
            "n_images conflicts with canonical n_groups"
        )
    if canonical is None:
        resolved["n_groups"] = legacy
    resolved["n_images"] = None
    return resolved, legacy is not None
```

Then overlay the canonicalized CLI source on the canonicalized file source.
Only alias/canonical duplicates within one source conflict; values in a
higher-precedence source replace lower-precedence values. Emit the existing
deprecation warning once after the complete candidate is structurally valid,
even when a lower-precedence alias was later replaced. Preserve
direct-constructor behavior.

- [ ] **Step 4: Run alias tests and verify GREEN**

Run the same command as Step 2.

Expected: all alias and direct-constructor compatibility tests pass.

- [ ] **Step 5: Commit**

```bash
git add ptycho/config/resolution.py tests/test_public_config_resolution.py
git commit -m "fix(config): canonicalize public grouping aliases"
```

### Task 3: Preserve explicit CLI suppliedness and remove parser duplication

**Files:**

- Modify: `ptycho/workflows/components.py`
- Modify: `scripts/training/train.py`
- Modify: `tests/scripts/test_training_backend_selector.py`
- Modify: `tests/test_public_config_resolution.py`

- [ ] **Step 1: Write failing CLI precedence tests**

Build a YAML file with `nepochs: 9`, parse argv containing only `--config`,
then assert `setup_configuration()` returns 9. Add the inverse case with
explicit `--nepochs 3`.

Also prove explicitly supplying the default value still counts as explicit,
and that source setup itself does not mutate `params.cfg` or its sealed state.
Exercise the shared parser builder with:

- a direct `Literal` field and an `Optional[Literal[...]]` field, proving
  primitive choices are unwrapped correctly;
- each supported boolean action form, proving omitted versus explicit
  true/false suppliedness;
- required and optional `Path` fields, proving stored values remain `Path`;
  and
- existing CLI spellings/help choices.

- [ ] **Step 2: Run the focused training CLI tests and verify RED**

```bash
python -m pytest \
  tests/test_public_config_resolution.py \
  tests/scripts/test_training_backend_selector.py \
  -k "yaml or precedence or explicit or literal or boolean or path or sealed_state" -q
```

Expected: reflected argparse defaults overwrite the YAML value.

- [ ] **Step 3: Extract one public config argument builder**

Create one helper in `ptycho.workflows.components` used by both training parser
entry points. Configuration override arguments use
`default=argparse.SUPPRESS`; non-configuration operational flags retain their
existing defaults.

`setup_configuration()` passes only present namespace fields as the explicit
CLI patch:

```python
file_values = load_yaml_config(yaml_path) if yaml_path else {}
cli_patch = {
    name: value
    for name, value in vars(args).items()
    if name in PUBLIC_TRAINING_INPUT_NAMES
}
config = resolve_training_config(file_values, cli_patch)
```

Remove the duplicated reflected ModelConfig/TrainingConfig argument-building
body from `scripts/training/train.py`; do not change CLI spellings or help
choices. `setup_configuration()` now returns the structurally validated
resolved value without calling `update_legacy_dict()`; runnable validation and
the CONFIG-001 bridge belong to the consuming execution boundary in Task 5.
Remove `@configured_legacy_params`, `params.unseal()`, and `params.seal()` from
`setup_configuration()` as well; the setup function must not alter either the
legacy mapping or its lifecycle state.

- [ ] **Step 4: Run focused parser/setup tests and verify GREEN**

```bash
python -m pytest \
  tests/test_public_config_resolution.py \
  tests/scripts/test_training_backend_selector.py -q
```

Expected: file/CLI precedence and existing backend arguments pass.

- [ ] **Step 5: Commit**

```bash
git add ptycho/workflows/components.py scripts/training/train.py \
  tests/test_public_config_resolution.py \
  tests/scripts/test_training_backend_selector.py
git commit -m "refactor(cli): preserve explicit public overrides"
```

### Task 4: Route inference through the complete public resolver

**Files:**

- Modify: `scripts/inference/inference.py`
- Modify: `tests/scripts/test_inference_backend_selector.py`
- Modify: `tests/test_public_config_resolution.py`

- [ ] **Step 1: Write failing inference source tests**

Prove inference YAML root values such as `n_groups`, `neighbor_count`,
`subsample_seed`, `debug`, `output_dir`, and `backend` are consumed, while an
explicit CLI value wins. Prove unknown root/nested fields fail. Add parser and
setup cases proving:

- omitted inference options are absent from the explicit patch;
- `model_path` and `test_data_file` can be supplied only by YAML;
- CLI `--test_data` maps to canonical `test_data_file`;
- the main inference path opens/loads `config.test_data_file`, rather than
  dereferencing the now-optional raw `args.test_data` attribute;
- an explicit CLI value equal to its presentation default still wins;
- `interpret_sampling_parameters()` reads `config.n_groups`, not
  `config.n_images`; and
- TensorFlow inference forwards `config.neighbor_count`, not a hard-coded
  `K=4`.

- [ ] **Step 2: Run inference tests and verify RED**

```bash
python -m pytest \
  tests/test_public_config_resolution.py \
  tests/scripts/test_inference_backend_selector.py \
  -k "inference and (yaml or unknown or precedence or n_groups or neighbor_count or parser or file_only)" -q
```

Expected: current setup reads only nested model YAML and ignores root inference
values; current parser defaults overwrite file values and current consumers
still read the deprecated grouping field and fixed neighbor count.

- [ ] **Step 3: Integrate `resolve_inference_config`**

Set `default=argparse.SUPPRESS` (or retain an equivalent explicit-name set) for
every inference configuration override. Stop making `--model_path` and
`--test_data` unconditionally argparse-required so YAML-only values can reach
resolution; the resolver/validation layer reports a missing required value
after both sources are considered. Map the existing CLI destination
`test_data` to canonical `test_data_file` when building the explicit patch.
Operational-only flags such as plotting and debug-dump controls remain outside
the public config patch and retain their presentation behavior.

Build the CLI patch only from present inference config values, then call the
pure resolver. Remove the manual ModelConfig default/YAML update path.

After resolution, make the runtime data-loading path consume
`config.test_data_file`. It must not read `args.test_data`: that argparse
attribute may be absent for a YAML-only request. Add a main-path test that
supplies the test-data path only in YAML and asserts that exact resolved path
is passed to the loader.

Update `interpret_sampling_parameters()` and the selected inference path to
read canonical `config.n_groups`. Keep `n_images` only as a boundary alias.
Thread `config.neighbor_count` instead of a hard-coded `K=4` where that current
path owns the grouping value.

- [ ] **Step 4: Run focused inference tests and verify GREEN**

```bash
python -m pytest \
  tests/test_public_config_resolution.py \
  tests/scripts/test_inference_backend_selector.py -q
```

Expected: complete inference source resolution, canonical grouping, backend
selection, and prior CLI tests pass.

- [ ] **Step 5: Commit**

```bash
git add scripts/inference/inference.py \
  tests/test_public_config_resolution.py \
  tests/scripts/test_inference_backend_selector.py
git commit -m "fix(inference): resolve complete public configuration"
```

### Task 5: Split structural, runnable, and resource validation

**Files:**

- Modify: `ptycho/config/resolution.py`
- Modify: `ptycho/config/__init__.py`
- Modify: `ptycho/config/config.py`
- Modify: `ptycho/workflows/components.py`
- Modify: `ptycho/workflows/backend_selector.py`
- Modify: `scripts/training/train.py`
- Modify: `scripts/inference/inference.py`
- Modify: `tests/test_public_config_resolution.py`
- Modify: `tests/test_model_config_architecture.py`
- Modify: `tests/torch/test_backend_selection.py`
- Modify: `tests/torch/test_execution_config_defaults.py`
- Modify: `tests/test_legacy_params_lifecycle.py`

- [ ] **Step 1: Write failing layered-validation tests**

Prove:

```python
inspectable = TrainingConfig(model=ModelConfig(), nepochs=0)
validate_training_config_structure(inspectable)  # passes
with pytest.raises(ValueError, match="nepochs"):
    validate_runnable_training_config(inspectable)
```

Also prove structural inference validation performs no `Path.exists()` call,
while `validate_inference_resources()` checks model and test-data paths.
Characterize the five-architecture domain, public activation spellings, and
object-policy joins.

Characterize the existing exported facades before refactoring:

- `validate_model_config()` retains its current structural checks;
- `validate_training_config()` retains its current checks for model, batch
  power-of-two, positive epochs, weights, and photons without acquiring a
  file-existence requirement; and
- `validate_inference_config()` retains its current model-structure and
  model-archive compatibility behavior without newly requiring test-data
  existence.

The compatibility facades and the new structural validators intentionally
have different predicates. In particular, characterize that the current
`validate_model_config()` compatibility facade continues to accept values
such as an otherwise unsupported activation that it does not currently check,
while `validate_model_config_structure()` rejects them. This prevents an
implementation from accidentally broadening an established exported facade
by delegation.

Add ordering tests for supported consumers:

- pure resolution performs structural validation before returning;
- runnable/resource failure occurs before any `params.cfg` bridge;
- a valid resolved request is bridged before backend-specific delegation;
- inference bridges the resolved request before the selected loader, while the
  loader remains free to restore authoritative archived state; and
- metadata-derived `nphotons` is reconstructed and revalidated before the
  final bridge and data/model consumption.

Update the directly affected selector fixtures:

- `tests/test_legacy_params_lifecycle.py::test_backend_workflow_entrypoint_contains_legacy_bridge`
  must pass a valid resolved `TrainingConfig` with an existing readable
  `train_data_file`, while preserving the exact rollback assertion; and
- the GPU-first and CPU-fallback backend-selector cases in
  `tests/torch/test_execution_config_defaults.py` must use an existing
  temporary training-data path so their delegation assertions exercise the
  valid runnable route.

For `load_inference_bundle_with_backend(bundle_dir, config)`, prove that the
actual `bundle_dir` and resolved `config.model_path` cannot diverge: unequal
normalized Paths fail before any bridge, and the one equal authoritative path
is resource-validated before delegation.

- [ ] **Step 2: Run validation tests and verify RED**

```bash
python -m pytest \
  tests/test_public_config_resolution.py \
  tests/test_model_config_architecture.py \
  tests/torch/test_backend_selection.py \
  -k "structural or runnable or resource or bridge_order or metadata or compatibility_facade" -q
```

Expected: current validators combine or omit these layers.

- [ ] **Step 3: Implement explicit validation layers**

Keep existing public names as locked compatibility facades:

- `validate_model_config()` retains its current predicate in a separate
  compatibility implementation and does **not** delegate to the broader new
  model structural validator;
- `validate_training_config()` preserves its exact existing validation set and
  exception categories; it does not gain path existence or new grouping
  requirements; and
- `validate_inference_config()` preserves its exact existing structure plus
  model-archive compatibility checks; it does not gain the new test-data
  resource requirement.

Route supported entry points through the more explicit APIs:

```python
validate_model_config_structure(config)
validate_training_config_structure(config)
validate_runnable_training_config(config)
validate_inference_config_structure(config)
validate_inference_resources(config)
```

Use manual exact checks and existing object-policy helpers. Do not validate
filesystem state from structural functions and do not add constructor or
assignment validation. `validate_runnable_training_config()` requires a
non-`None`, existing, readable training-data path in addition to positive
epochs/batch/photons and sampling coherence; it must fail before the bridge.
Both `resolve_training_config()` and
`resolve_inference_config()` must call their structural validator before
returning. Export all supported validation APIs from `ptycho.config`.

- [ ] **Step 4: Call the correct layer at each consumer**

Training and inference source setup remain side-effect-free. In
`scripts/training/train.py`, apply a metadata photon override by reconstructing
the dataclass, then rerun structural and runnable validation on that final
record before any data/model consumption or legacy bridge.

In `ptycho/workflows/backend_selector.py`, make direct programmatic consumers
follow the same order:

```text
already-resolved record
    -> structural plus appropriate runnable/resource validation
    -> update_legacy_dict(params.cfg, record)
    -> inspect backend and delegate
```

Training dispatch validates runnable state. Inference dispatch validates
structure and both model/test-data resources, bridges the resolved bootstrap
request, and then calls the backend loader; authoritative archive restoration
inside that loader remains unchanged. At the inference selector boundary,
normalize `bundle_dir` and `config.model_path`, require equality, and validate
that actual path before bridging so the validated resource is the resource
that the loader consumes.

Remove the early bridge from `setup_configuration()`. Ensure the CLI does not
bridge a pre-metadata training candidate and then mutate `params.cfg` by hand.
Legacy projection and persistence do not acquire runnable/resource checks.

Update
`tests/test_legacy_params_lifecycle.py::test_outer_bundle_route_rolls_back_inner_success`
to use a valid resolved `InferenceConfig` instead of `SimpleNamespace`, while
preserving its exact outer rollback assertion.
Also update
`tests/test_legacy_params_lifecycle.py::test_backend_workflow_entrypoint_contains_legacy_bridge`
and the two backend-selector default/fallback tests described in Step 1 with
real temporary paths; runnable validation must not be bypassed or caught and
ignored in tests whose contract is successful delegation.

- [ ] **Step 5: Run focused validation and CLI tests**

```bash
python -m pytest \
  tests/test_public_config_resolution.py \
  tests/test_model_config_architecture.py \
  tests/scripts/test_training_backend_selector.py \
  tests/scripts/test_inference_backend_selector.py \
  tests/torch/test_backend_selection.py \
  tests/torch/test_execution_config_defaults.py \
  tests/test_legacy_params_lifecycle.py::test_backend_workflow_entrypoint_contains_legacy_bridge \
  tests/test_legacy_params_lifecycle.py::test_outer_bundle_route_rolls_back_inner_success -q
```

Expected: layered validation and supported entry points pass.

- [ ] **Step 6: Commit**

```bash
git add ptycho/config/resolution.py ptycho/config/__init__.py \
  ptycho/config/config.py \
  ptycho/workflows/components.py ptycho/workflows/backend_selector.py \
  scripts/training/train.py scripts/inference/inference.py \
  tests/test_public_config_resolution.py \
  tests/test_model_config_architecture.py \
  tests/scripts/test_training_backend_selector.py \
  tests/scripts/test_inference_backend_selector.py \
  tests/torch/test_backend_selection.py \
  tests/torch/test_execution_config_defaults.py \
  tests/test_legacy_params_lifecycle.py
git commit -m "refactor(config): separate public validation layers"
```

### Task 6: Focused compatibility verification

**Files:**

- Modify only directly stale symbol-routing text if implementation naming
  differs from the approved design.

- [ ] **Step 1: Run the claim-matched public set**

```bash
python -m pytest \
  tests/test_public_config_resolution.py \
  tests/test_model_config_architecture.py \
  tests/scripts/test_training_backend_selector.py \
  tests/scripts/test_inference_backend_selector.py \
  tests/torch/test_backend_selection.py -q

python -m pytest \
  tests/torch/test_config_bridge.py::TestConfigBridgeMVP::test_mvp_config_bridge_populates_params_cfg \
  tests/torch/test_config_bridge.py::TestConfigBridgeParity::test_params_cfg_matches_baseline \
  tests/torch/test_config_bridge.py::TestConfigBridgeArchitecture -q
```

Expected: public resolution, CLI, grouping, legacy/backend bridge, and
architecture tests pass.

- [ ] **Step 2: Freeze the legacy projections with focused exact assertions**

In `tests/test_public_config_resolution.py`, compare
`dataclass_to_legacy_dict()` for a resolved config with the exact projection of
the equivalent directly constructed config. Assert equal key order/value
content, Path-to-string conversion, and retained `None` in the pure
projection. Seed a target mapping with sentinel values, call
`update_legacy_dict()`, and assert non-`None` values update while the sentinel
for every projected `None` remains unchanged. Do not update expected legacy
keys, Path/string values, or skip-`None` behavior to make failures pass.

Run only those focused tests:

```bash
python -m pytest tests/test_public_config_resolution.py \
  -k "legacy_projection or update_legacy_skip_none" -q
```

- [ ] **Step 3: Verify exclusions**

```bash
BASE_SHA=<execution-start SHA recorded immediately before Task 1>
git diff --name-only "$BASE_SHA"..HEAD
git diff --check
```

Expected: no Pydantic adapter for public configs, enum catalog, ModelSpec,
artifact codec, checkpoint, MLflow, or simulation implementation change.

- [ ] **Step 4: Document the supported public boundary**

Update `docs/CONFIGURATION.md` with:

- defaults < file < explicitly supplied CLI precedence;
- per-source flat/nested duplicate handling;
- `n_groups` canonicalization and `n_images` deprecation; and
- structural versus runnable versus resource validation ownership.

Update `docs/index.md` only to route the exported public resolver APIs and this
implementation plan. Do not add another design or schema.

```bash
git add docs/CONFIGURATION.md docs/index.md \
  tests/test_public_config_resolution.py
git commit -m "docs(config): route public resolution APIs"
```
