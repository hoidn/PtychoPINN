# Torch Configuration Resolution Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Torch training and inference configuration resolution explicit, return-new, fail-closed, and transactional while preserving the legacy bridge and persistence contracts.

**Architecture:** Introduce a private phase-aware resolution module with explicit input registries, canonical alias normalization, fresh candidate construction, semantic bundle validation, and deterministic audit data. Keep `create_training_payload()` and `create_inference_payload()` as the public facades; they perform read-only resolution first and commit `params.cfg` only after the complete candidate is valid. Preserve `update_existing_config()` as a characterized legacy island and leave ModelSpec, artifacts, checkpoints, MLflow, and Pydantic outside this change.

**Tech Stack:** Python stdlib dataclasses, pathlib, NumPy metadata inspection, PyTorch configuration dataclasses, pytest.

**Prerequisite and execution constraint:** Execute
`docs/plans/2026-07-28-execution-config-ownership-implementation.md` first so
the resolver can consume its provenance-carrying `ExecutionRequest`. Work in
the current checkout and do not create a worktree; the repository `AGENTS.md`
prohibition overrides generic execution skill advice. Immediately before Task
1, record the current commit as the implementation-start SHA for Task 6.

---

## File Map

- Create `ptycho_torch/config_resolution.py`: explicit training/inference registries, patch normalization, candidate records, semantic validation, and audit construction.
- Modify `ptycho_torch/config_factory.py`: delegate resolution, translate validated candidates, and commit the legacy bridge atomically.
- Modify `ptycho_torch/config_params.py`: document and preserve the legacy updater boundary; do not broaden it.
- Add `tests/torch/test_config_resolution_transaction.py`: focused resolver, alias, audit, derivation, and side-effect tests.
- Modify `tests/torch/test_config_factory.py`: retain factory compatibility and bridge ordering assertions.
- Modify `tests/torch/test_structural_config_ownership.py`: assert explicit ownership and derived channel conflicts.

## Explicit phase input inventory

The implementation must encode this inventory as declared constants/rules.
Dataclass reflection may assert constructor completeness but must not decide
external acceptance.

Training patch inputs by owner:

- Data: `nphotons`, `scale_contract_version`, `measurement_domain`, `N`, `K`,
  `K_quadrant`, `n_subsample`, `subsample_seed`, `grid_size`,
  `neighbor_function`, `min_neighbor_distance`, `max_neighbor_distance`,
  `scan_pattern`, `normalize`, `probe_scale`, `probe_normalize`,
  `data_scaling`, `phase_subtraction`, `x_bounds`, and `y_bounds`.
- Model: `mode`, `architecture`, `fno_modes`, `fno_width`, `fno_blocks`,
  `fno_cnn_blocks`, `learned_input_channels`, `fno_input_transform`,
  `max_hidden_channels`, `resnet_width`, `spectral_bottleneck_blocks`,
  `spectral_bottleneck_modes`, `spectral_bottleneck_share_weights`,
  `spectral_bottleneck_gate_init`, `spectral_bottleneck_gate_mode`,
  `generator_output_mode`, `cnn_output_mode`, `use_shared_decoder`,
  `intensity_scale_trainable`, `intensity_scale`, `max_position_jitter`,
  `num_datasets`, `n_filters_scale`, `amp_activation`, `batch_norm`,
  `probe_mask`, `probe_mask_tensor`, `probe_mask_sigma`,
  `probe_mask_diameter`, `edge_pad`, `decoder_last_c_outer_fraction`,
  `decoder_last_amp_channels`, `use_legacy_decoder_channel_override`,
  `eca_encoder`, `cbam_encoder`, `cbam_bottleneck`, `cbam_decoder`,
  `eca_decoder`, `spatial_decoder`, `decoder_spatial_kernel`, `object_big`,
  `object_layout`, `training_canvas`, `probe_big`, `offset`,
  `training_patch_weighting`, `physics_forward_mode`,
  `rect_s1s2_trainable`, `rect_s1s2_init`, `amplitude_physics_gain`,
  `pad_object`, `gaussian_smoothing_sigma`, `amp_loss`, `phase_loss`,
  `amp_loss_coeff`, and `phase_loss_coeff`.
- Training: `training_directories`, `framework`, `orchestrator`,
  `learning_rate`, `epochs`, `batch_size`,
  `epochs_fine_tune`, `fine_tune_gamma`, `scheduler`, `lr_warmup_epochs`,
  `lr_min_ratio`, `plateau_factor`, `plateau_patience`, `plateau_min_lr`,
  `plateau_threshold`, `accum_steps`, `gradient_clip_val`,
  `gradient_clip_algorithm`, `optimizer`, `momentum`, `weight_decay`,
  `adam_beta1`, `adam_beta2`, `log_grad_norm`, `grad_norm_log_freq`,
  `stage_1_epochs`, `stage_2_epochs`, `stage_3_epochs`,
  `physics_weight_schedule`, `stage_3_lr_factor`, `torch_loss_mode`,
  `torch_mae_pred_l2_match_target`, `experiment_name`, `notes`,
  `model_name`, `test_data_file`, and `n_groups`.
- Training-phase inference controls: `patch_weighting`, `varpro_scaling`,
  `log_patch_stats`, and `patch_stats_limit`.
- Derived/argument constraints, accepted only for equality checking:
  `C`, `C_model`, `C_forward`, `loss_function`, `nll`,
  `train_data_file`, and `output_dir`.
- Compatibility aliases: `gridsize -> grid_size`,
  `neighbor_count -> K`, `model_type -> mode`, and
  `max_epochs -> epochs`.
- Rejected legacy duplicate owners: `device`, `strategy`, `n_devices`, and
  `num_workers`. Supported factory patches direct callers to the execution
  request; these same-named Torch TrainingConfig fields survive only inside
  the separately characterized legacy standalone lane and are not effective
  factory inputs or audit entries.

Inference patch inputs by owner:

- Data: `N`, `K`, `grid_size`, `probe_scale`, `subsample_seed`,
  `scale_contract_version`, and `measurement_domain`.
- Model bridge projection: `mode`, `amp_activation`, `n_filters_scale`,
  `object_big`, `object_layout`, `training_canvas`,
  `training_patch_weighting`, `probe_big`, `probe_mask`,
  `probe_mask_tensor`, `probe_mask_sigma`, `probe_mask_diameter`,
  `pad_object`, and `gaussian_smoothing_sigma`.
- Inference: `batch_size`, `patch_weighting`, `varpro_scaling`,
  `log_patch_stats`, and `patch_stats_limit`.
- Bridge-only runtime values: `n_groups` and `n_subsample`.
- Derived/argument constraints: `C`, `C_model`, `C_forward`, `model_path`,
  `test_data_file`, and `output_dir`.
- Compatibility aliases: `gridsize -> grid_size`,
  `neighbor_count -> K`, and `model_type -> mode`.

The execution request is a separate input owner. For training, its deprecated
optimizer-adjacent compatibility fields are `learning_rate`, `scheduler`,
`gradient_clip_val`, `gradient_clip_algorithm`, and `accum_steps`. Its
deprecated topology aliases are the five
`spectral_bottleneck_{blocks,modes,share_weights,gate_init,gate_mode}` fields.
Every other execution field remains owned by the request/config and is audited
only if the execution plan marks it effective for the phase.

## Locked factory baselines

Omission preserves these factory-specific baselines; do not substitute raw
dataclass defaults where current factory behavior differs:

| Value | Training baseline | Inference baseline |
|---|---|---|
| `grid_size` / `C` | `(1, 1)` / `1` | `(1, 1)` / `1` |
| `K` | `DataConfig.K` (`6`) | Ptychodus bridge contract (`4`) |
| `N` | observed from training NPZ | observed from test NPZ |
| `nphotons` | NPZ metadata, else public `TFTrainingConfig` default (`1e9`) | `DataConfig`/bridge profile value |
| scale contract | current DataConfig/profile resolution | `ci_intensity_v2` / `count_intensity` unless an explicit coherent pair is supplied |
| model fields | current `PTModelConfig` defaults plus derived `C_*`, resolved loss, and object policy | current selective bridge projection defaults plus derived `C_*` and object policy |
| Torch training fields | `PTTrainingConfig` defaults plus positional paths and required `n_groups` | not constructed |
| Torch inference `batch_size` | `PTInferenceConfig` default (`1000`) | factory compatibility value (`16`) |
| inference patch controls | `patch_weighting="probe"`, `varpro_scaling=True`, stats disabled | same |
| remaining inference fields | `PTInferenceConfig` defaults | `PTInferenceConfig` defaults |

Factory path arguments (`train_data_file`, `model_path`, `test_data_file`, and
`output_dir`) remain authoritative observations/constraints. Tests must lock
every differing baseline above before the return-new refactor.

### Task 1: Add explicit phase input normalization

**Files:**

- Create: `ptycho_torch/config_resolution.py`
- Create: `tests/torch/test_config_resolution_transaction.py`

- [ ] **Step 1: Write failing unknown-key and alias tests**

Cover both phases:

```python
def test_training_patch_rejects_sorted_unknown_names():
    with pytest.raises(ValueError, match=r"unknown training input.*alpha_typo.*zeta_typo"):
        normalize_training_patch(
            {"zeta_typo": 1, "alpha_typo": 2}
        )


def test_inference_patch_rejects_unknown_names():
    with pytest.raises(ValueError, match="unknown inference input.*batch_szie"):
        normalize_inference_patch({"batch_szie": 4})


@pytest.mark.parametrize(
    ("normalizer", "patch", "canonical", "expected"),
    [
        (normalize_training_patch, {"max_epochs": 7, "epochs": 7}, "epochs", 7),
        (normalize_training_patch, {"neighbor_count": 4, "K": 4}, "K", 4),
        (
            normalize_inference_patch,
            {"gridsize": 2, "grid_size": (2, 2)},
            "grid_size",
            (2, 2),
        ),
    ],
)
def test_equal_alias_and_canonical_are_consumed_once(
    normalizer, patch, canonical, expected
):
    normalized = normalizer(patch)
    assert normalized.values[canonical] == expected
    assert canonical in normalized.audit
    assert set(normalized.audit) == {canonical}
    assert normalized.aliases[canonical]


def test_unequal_alias_and_canonical_fail():
    with pytest.raises(ValueError, match="alias.*max_epochs.*epochs.*conflict"):
        normalize_training_patch({"max_epochs": 6, "epochs": 7})
```

Also assert that input mappings are unchanged after success and failure.
Using an `ExecutionRequest`, cover each deprecated topology alias: equal
canonical/alias values are accepted once under the Model owner, unequal values
fail, and no deprecation warning is emitted until complete factory success.

- [ ] **Step 2: Run the focused tests and confirm RED**

```bash
python -m pytest tests/torch/test_config_resolution_transaction.py -q
```

Expected: import or behavior failures because the resolver does not exist.

- [ ] **Step 3: Implement declared phase registries**

In `ptycho_torch/config_resolution.py`, add small immutable internal records such
as:

```python
@dataclass(frozen=True)
class InputRule:
    canonical: str
    owner: Literal[
        "data",
        "model",
        "training",
        "inference",
        "bridge",
        "derived_constraint",
        "execution_compatibility",
    ]
    aliases: tuple[str, ...] = ()


@dataclass(frozen=True)
class NormalizedPatch:
    values: Mapping[str, object]
    audit: Mapping[str, object]
    aliases: Mapping[str, tuple[str, ...]]
    notices: tuple[ResolutionNotice, ...]
```

Declare training and inference registries explicitly. Include only names the
phase actually consumes. Do not derive accepted external names with
`dataclasses.fields()`, and do not route a name to every record that happens to
have a same-named field. Reflection may be used only in an assertion that an
internal constructor mapping remains complete.

Canonicalize the existing compatibility aliases:

- `max_epochs -> epochs`;
- `neighbor_count -> K`;
- `model_type -> mode`; and
- scalar `gridsize -> grid_size=(n, n)`.

If canonical and alias forms are both present, compare their normalized values.
Accept equal values once under the canonical name and retain alias provenance;
reject unequal values deterministically. Reject every unregistered input.

Normalize explicitly supplied deprecated execution topology aliases from the
`ExecutionRequest` through the same canonical Model rules. Accumulate
deprecation notices in `NormalizedPatch`; do not call `warnings.warn()` during
normalization.

- [ ] **Step 4: Re-run the focused tests**

```bash
python -m pytest tests/torch/test_config_resolution_transaction.py -q
```

Expected: normalization tests pass.

- [ ] **Step 5: Commit**

```bash
git add ptycho_torch/config_resolution.py tests/torch/test_config_resolution_transaction.py
git commit -m "refactor(torch): declare phase config inputs"
```

### Task 2: Construct fresh candidates and enforce derived joins

**Files:**

- Modify: `ptycho_torch/config_resolution.py`
- Modify: `tests/torch/test_config_resolution_transaction.py`
- Modify: `tests/torch/test_structural_config_ownership.py`

- [ ] **Step 1: Write failing return-new and join tests**

Create baseline records and prove:

- every value in the locked factory-baseline table, including both phase
  divergences from raw dataclass defaults;
- omitted patch keys retain baseline values;
- successful resolution returns fresh outer dataclass instances;
- the caller mapping and all baseline records are unchanged;
- `C=prod(grid_size)`, `C_model=C`, and `C_forward=C` are named derivations;
- an explicitly supplied `C`, `C_model`, or `C_forward` that disagrees with the
  derived value fails instead of being overwritten;
- `torch_loss_mode` and `ModelConfig.loss_function` resolve to one coherent
  objective;
- scale-profile and object-policy validation occur before publication; and
- inference resolution does not claim to reconstruct checkpoint ModelSpec
  identity.

Also prove `device`, `strategy`, `n_devices`, and `num_workers` are rejected as
supported training patch inputs with an error naming execution ownership.

Force a probe-size fallback followed by a later invalid semantic join and
assert no fallback warning is emitted. In the corresponding successful
factory case, assert the same deferred warning is emitted once.

Use an internal baseline bundle rather than mutating default instances:

```python
baseline = TorchConfigBaseline(
    data=DataConfig(grid_size=(1, 1), C=1),
    model=ModelConfig(C_model=1, C_forward=1),
    training=TrainingConfig(epochs=3),
    inference=InferenceConfig(batch_size=8),
)
resolved = resolve_training_bundle(
    baseline=baseline,
    normalized=normalize_training_patch({"epochs": 9, "n_groups": 16}),
    observations=TrainingObservations(...),
)
assert resolved.training.epochs == 9
assert baseline.training.epochs == 3
assert resolved.training is not baseline.training
```

- [ ] **Step 2: Run and confirm RED**

```bash
python -m pytest \
  tests/torch/test_config_resolution_transaction.py \
  tests/torch/test_structural_config_ownership.py -q
```

- [ ] **Step 3: Implement private candidate bundles**

Add frozen internal records for:

- default/baseline Torch records;
- read-only training observations (`train_data_file`, `output_dir`, inferred
  probe size, and photon metadata);
- read-only inference observations (`model_path`, `test_data_file`,
  `output_dir`, and inferred probe size); and
- resolved training/inference bundles plus deterministic audit.

Read-only observations carry deferred `ResolutionNotice` values rather than
emitting warnings. Preserve public `infer_probe_size()` behavior for direct
legacy callers, but have supported factory resolution use a non-emitting
observation helper that returns the inferred/fallback value and any notice.
Notices from probe fallback, deprecated topology aliases, and other
resolver-owned normalization are emitted only after complete payload
construction and a successful legacy commit.

Use dataclass constructors or `dataclasses.replace()` to create candidates.
Never mutate caller-owned records. Keep tensor-valued configuration fields
read-only during resolution.

Give each derived value one named rule:

- scalar `gridsize` normalization;
- `C`, `C_model`, and `C_forward` from `grid_size`;
- loss identity from `torch_loss_mode`;
- `nll` from the selected objective compatibility rule;
- `N` and `nphotons` from explicit value, metadata observation, then declared
  default; and
- object compatibility through
  `resolve_torch_model_object_policy()`.

Conflict-check supplied derived fields before constructing the bundle.

- [ ] **Step 4: Validate the complete bundle**

Reuse existing focused physics and policy validators:

- `validate_amplitude_physics_gain()`;
- `_reject_half_configured_ci()` or a moved private equivalent;
- `validate_contract_coherence()`; and
- explicit channel/loss joins.

Keep ModelSpec derivation outside the resolver. The resolver may supply the
validated inputs later used by `derive_model_spec()`, but it must not alter
ModelSpec or persistence behavior.

- [ ] **Step 5: Re-run the focused tests**

```bash
python -m pytest \
  tests/torch/test_config_resolution_transaction.py \
  tests/torch/test_structural_config_ownership.py -q
```

- [ ] **Step 6: Commit**

```bash
git add \
  ptycho_torch/config_resolution.py \
  tests/torch/test_config_resolution_transaction.py \
  tests/torch/test_structural_config_ownership.py
git commit -m "refactor(torch): resolve fresh config bundles"
```

### Task 3: Delegate the public factories to pure resolution

**Files:**

- Modify: `ptycho_torch/config_factory.py`
- Modify: `ptycho_torch/config_resolution.py`
- Modify: `tests/torch/test_config_factory.py`
- Modify: `tests/torch/test_config_resolution_transaction.py`

- [ ] **Step 1: Write failing factory delegation tests**

Use spies to prove:

- profile normalization, raw key classification, file checks, candidate
  construction, bridge translation, and complete bundle validation happen
  before any global mutation;
- invalid training and inference patches leave `params.cfg` and the seal state
  byte-for-value unchanged;
- no warning is emitted for a failed resolution;
- a successful factory commits the complete legacy projection exactly once;
- the returned payload contains the fresh resolved Torch records; and
- training and inference audit excludes raw aliases, unknown names, and
  superseded values.

Retain existing assertions for `n_groups`, inferred `N`, photon metadata, CI
profiles, and TensorFlow bridge values.

Retain the prerequisite execution-plan matrix for each of `learning_rate`,
`scheduler`, `gradient_clip_val`, `gradient_clip_algorithm`, and
`accum_steps`:

```text
explicit canonical TrainingConfig patch
    > explicitly supplied ExecutionRequest compatibility value
    > resolved/baseline TrainingConfig
    > TrainingConfig default
```

Assert that only the resolved `TrainingConfig` carries the effective value and
that the execution request's compatibility copy is not downstream ownership.

- [ ] **Step 2: Run and confirm RED**

```bash
python -m pytest \
  tests/torch/test_config_resolution_transaction.py \
  tests/torch/test_config_factory.py -q
```

- [ ] **Step 3: Make factories thin facades**

In `create_training_payload()`:

1. defensively copy the raw patch;
2. resolve a named profile into raw values without side effects;
3. collect read-only observations;
4. call the training resolver;
5. translate the resolved records through the existing config bridge;
6. materialize a default execution request/config only now, if input was
   `None`;
7. preserve the existing legacy commit position for this task;
8. derive ModelSpec through the unchanged existing function;
9. construct the complete deterministic audit and local `TrainingPayload`;
10. emit accumulated successful-resolution notices; and
11. return the payload.

Apply the equivalent ordering to inference, without reconstructing structural
model identity from loose inference overrides.

The prerequisite execution normalizer must return a no-input marker for
`None`; it must not construct `PyTorchExecutionConfig()` during raw
normalization. This ensures an invalid patch/bundle cannot trigger capability
inspection or the CPU-fallback warning. Task 4 deliberately moves the
remaining successful legacy commit after ModelSpec/audit/payload construction.

Remove `_TRAINING_CONFIG_TYPES`, reflection-based accepted-name discovery,
raw-copy audit construction, and duplicated procedural routing from the
facades. Preserve public signatures in this task. The companion execution plan
has already added the explicit execution request envelope. Move/preserve its
optimizer-adjacent precedence inside the pure training resolver: explicit
canonical patch wins over explicitly supplied request compatibility input,
which wins over the baseline/default. Write the effective value only into the
fresh `PTTrainingConfig`.

- [ ] **Step 4: Build deterministic consumed/effective audit**

Audit must contain only:

- canonical recognized inputs;
- intentionally exposed effective values;
- named derivations and their provenance; and
- declared environmental observations.

Keep current compatibility audit keys only where they describe a value actually
consumed or resolved. Record aliases as provenance metadata rather than a
second applied value. Never use audit as ModelSpec or persistence input.

- [ ] **Step 5: Re-run focused factory tests**

```bash
python -m pytest \
  tests/torch/test_config_resolution_transaction.py \
  tests/torch/test_config_factory.py \
  tests/torch/test_structural_config_ownership.py -q
```

- [ ] **Step 6: Commit**

```bash
git add \
  ptycho_torch/config_factory.py \
  ptycho_torch/config_resolution.py \
  tests/torch/test_config_factory.py \
  tests/torch/test_config_resolution_transaction.py
git commit -m "refactor(torch): delegate payload resolution"
```

### Task 4: Delay and verify the transactional legacy commit

**Files:**

- Modify: `ptycho_torch/config_factory.py`
- Modify: `tests/torch/test_config_resolution_transaction.py`
- Modify: `tests/torch/test_config_factory.py`

- [ ] **Step 1: Characterize the existing transaction and write a failing ordering test**

First characterize `@configured_legacy_params`: patch
`populate_legacy_params()` to mutate one legacy key and then raise, and assert
that the existing scope restores:

- the exact prior `params.cfg` mapping is restored; and
- the prior sealed/unsealed state is restored.

This is a GREEN characterization, not a premise for adding a second transaction
mechanism.

Then patch `derive_model_spec()` or local payload construction to raise and
assert `populate_legacy_params()` was never called. This must be RED on the
current ordering, which commits before ModelSpec/payload construction.

- [ ] **Step 2: Run and confirm RED**

```bash
python -m pytest \
  tests/torch/test_config_resolution_transaction.py \
  tests/torch/test_config_factory.py -q
```

- [ ] **Step 3: Move the existing commit to the final boundary**

Do not add another snapshot/rollback helper. Retain
`@configured_legacy_params`, which already snapshots and restores both the
mapping and seal state on exceptions.

Remove `params.unseal()` from factory entry. Construct the entire bridge
projection, ModelSpec where applicable, audit, and payload locally. Only then:

1. unseal;
2. call the existing canonical legacy bridge exactly once;
3. seal;
4. emit accumulated notices; and
5. return the already constructed payload.

If the commit raises, the surrounding configured scope restores both legacy
mapping and seal state. A successful training or inference call remains
persistently committed and sealed.

- [ ] **Step 4: Re-run the focused tests**

```bash
python -m pytest \
  tests/torch/test_config_resolution_transaction.py \
  tests/torch/test_config_factory.py -q
```

- [ ] **Step 5: Commit**

```bash
git add \
  ptycho_torch/config_factory.py \
  tests/torch/test_config_resolution_transaction.py \
  tests/torch/test_config_factory.py
git commit -m "fix(torch): delay legacy config commit"
```

### Task 5: Seal the compatibility islands

**Files:**

- Modify: `ptycho_torch/config_params.py`
- Modify: `tests/torch/test_config_resolution_transaction.py`
- Modify: `tests/torch/test_config_factory.py` only if a missing Datagen case is demonstrated

- [ ] **Step 1: Add characterization tests for the legacy updater**

Prove that `update_existing_config()` retains its existing compatibility
behavior:

- it mutates the passed object in place;
- it applies known partial values;
- it ignores unknown names unless verbose output is requested; and
- supported training/inference factories do not call it.

Do not tighten or remove this behavior in this plan.

- [ ] **Step 2: Verify canonical Datagen delegation**

Use existing tests to prove `DatagenConfig`:

- enters canonical simulation validation through `to_simulation_config()`;
- round-trips its six representable fields;
- surfaces canonical validation errors; and
- rejects a requested lossless projection when canonical probe source
  semantics cannot be represented.

Only add a focused missing test if current coverage does not establish one of
these claims. Do not create a Torch-specific simulation schema or validator.
The existing selectors are:

```text
tests/torch/test_config_factory.py::test_datagen_config_converts_owned_fields_to_simulation_without_changing_payload_shape
tests/torch/test_config_factory.py::test_datagen_config_round_trip_preserves_only_representable_owned_fields
tests/torch/test_config_factory.py::test_datagen_config_rejects_lossy_probe_or_object_conversion
```

- [ ] **Step 3: Mark the boundary in code**

Add a concise docstring warning to `update_existing_config()`:

- legacy compatibility only;
- new code must use the return-new resolver; and
- tolerant unknown-key behavior is intentionally not a supported resolver
  contract.

Do not alter its executable behavior.

- [ ] **Step 4: Run focused compatibility tests**

```bash
python -m pytest \
  tests/torch/test_config_resolution_transaction.py \
  tests/torch/test_config_factory.py::test_datagen_config_converts_owned_fields_to_simulation_without_changing_payload_shape \
  tests/torch/test_config_factory.py::test_datagen_config_round_trip_preserves_only_representable_owned_fields \
  tests/torch/test_config_factory.py::test_datagen_config_rejects_lossy_probe_or_object_conversion -q
```

- [ ] **Step 5: Commit**

```bash
git add ptycho_torch/config_params.py tests/torch/test_config_resolution_transaction.py
git add tests/torch/test_config_factory.py
git commit -m "docs(torch): isolate legacy config updater"
```

### Task 6: Focused resolver regression

**Files:**

- Modify only directly affected tests or routing text if a demonstrated
  mismatch remains.

- [ ] **Step 1: Run the complete focused evidence set**

```bash
python -m pytest \
  tests/torch/test_config_resolution_transaction.py \
  tests/torch/test_structural_config_ownership.py \
  tests/torch/test_config_factory.py::TestTrainingPayloadStructure \
  tests/torch/test_config_factory.py::TestInferencePayloadStructure \
  tests/torch/test_config_factory.py::TestConfigBridgeTranslation \
  tests/torch/test_config_factory.py::TestLegacyParamsPopulation \
  tests/torch/test_config_factory.py::TestOverridePrecedence \
  tests/torch/test_config_factory.py::TestFactoryValidation \
  tests/torch/test_config_factory.py::test_datagen_config_converts_owned_fields_to_simulation_without_changing_payload_shape \
  tests/torch/test_config_factory.py::test_datagen_config_round_trip_preserves_only_representable_owned_fields \
  tests/torch/test_config_factory.py::test_datagen_config_rejects_lossy_probe_or_object_conversion \
  tests/torch/test_config_bridge.py::TestConfigBridgeMVP::test_mvp_config_bridge_populates_params_cfg \
  tests/torch/test_config_bridge.py::TestConfigBridgeParity::test_params_cfg_matches_baseline \
  tests/torch/test_config_bridge.py::TestConfigBridgeArchitecture -q
```

Expected: resolver, semantic join, transaction, legacy bridge, canonical
simulation delegation, and bridge projection tests pass.

- [ ] **Step 2: Run the named direct factory consumers**

```bash
python -m pytest \
  tests/torch/test_amplitude_physics_gain.py::TestConfigPlumbing::test_training_payload_plumbs_gain_and_audits_it \
  tests/torch/test_amplitude_physics_gain.py::TestConfigPlumbing::test_training_payload_audits_default_gain \
  tests/torch/test_ci_profile.py::test_create_training_payload_ci_profile_resolves_coherent_payload \
  tests/torch/test_ci_profile.py::test_create_training_payload_ci_profile_rejects_contradiction \
  tests/torch/test_ci_profile.py::test_create_training_payload_rejects_unknown_profile \
  tests/torch/test_ci_profile.py::test_profile_none_is_bit_identical_to_default \
  tests/torch/test_ci_profile.py::test_half_configured_ci_intent_via_overrides_raises \
  tests/torch/test_ci_profile.py::test_count_intensity_mae_rectangular_raises_at_factory \
  tests/torch/test_model_spec.py::test_training_payload_carries_current_model_spec \
  tests/torch/test_patch_stats_cli.py::TestPatchStatsCLI::test_factory_creates_inference_config_with_patch_stats \
  tests/torch/test_patch_stats_cli.py::TestPatchStatsCLI::test_factory_inference_config_defaults \
  tests/torch/test_absolute_scaling_entrypoints.py::test_rectangular_factory_defaults_to_ci_profile \
  tests/torch/test_absolute_scaling_entrypoints.py::test_factory_rejects_partial_or_contradictory_profile_overrides \
  tests/torch/test_absolute_scaling_entrypoints.py::test_factory_accepts_explicit_legacy_pair_for_training_and_inference -q
```

If a class qualifier differs, confirm the exact node with
`python -m pytest --collect-only <file> -q` and update only the selector, not
the evidence scope. Classify unrelated failures against the governing contract
rather than expanding the plan.

- [ ] **Step 3: Verify exclusion boundaries**

```bash
BASE_SHA=<execution-start SHA recorded immediately before Task 1>
git diff --name-only "$BASE_SHA"..HEAD
git diff --check
rg -n "update_existing_config\\(" ptycho_torch/config_factory.py
```

Expected:

- no supported factory call to `update_existing_config()`;
- no changes to ModelSpec, artifacts, checkpoints, sidecars, MLflow codecs, or
  Pydantic ownership;
- no output directory creation before resolver success; and
- no changed persisted schema or bytes.

- [ ] **Step 4: Commit only a directly required compatibility fix**

If the focused checks required no change, do not make an empty commit. If a
direct consumer needed a bounded compatibility correction, commit only that
correction:

```bash
git add <directly-affected-files>
git commit -m "fix(torch): preserve resolver consumers"
```

The prerequisite execution ownership implementation retains ownership of
`ExecutionRequest` construction and Lightning argument derivation. This plan
integrates its declared optimizer-adjacent precedence into the return-new
resolver without creating a competing mechanism.
