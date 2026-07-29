# Public Configuration Resolution Design

**Status:** Approved on 2026-07-28

**Parent architecture:** `docs/superpowers/specs/2026-07-28-configuration-boundary-architecture.md`

**Implementation state on `refactor-internal`:** The
`ptycho/config/resolution.py` resolver and its focused fixtures on the public
`refactor` branch are reference implementation evidence only. That module and
those fixtures are not present here; roadmap Slice 8 Stage A owns their
internal-safe port while preserving the internal 14-architecture domain.

## Purpose

Define one supported boundary for resolving public ModelConfig,
TrainingConfig, and InferenceConfig values from configuration files, explicit
CLI overrides, bridges, and programmatic instances while preserving their
stdlib dataclass API and one-way legacy projection.

Pydantic is held until boundary consolidation demonstrates that an adapter
would delete meaningful production code. This design first makes the existing
contracts coherent enough to validate.

## Current Problem

The public records are presented as the canonical configuration API, but their
supported construction paths do not have one resolution contract:

- training reflects dataclass fields into two duplicated parsers, merges YAML
  with argparse values, silently filters unknown fields, and can let argparse
  defaults overwrite YAML even when the user did not supply the option;
- inference reads only the nested YAML model section and ignores root
  inference values;
- notebook updates warn and discard unknown names;
- the Torch training factory rejects unknown overrides while Torch inference
  silently ignores some;
- ModelConfig is frozen, while TrainingConfig and InferenceConfig are mutable
  and normalize the deprecated `n_images` alias during `__post_init__`;
- `validate_training_config()` has no production caller;
- `validate_inference_config()` mixes partial structural validation with
  filesystem checks; and
- several current callers construct values outside the annotations, so
  constructor-time enforcement would be a compatibility change rather than a
  refactor.

The normalized dataclasses and `params.cfg` projection are also used in
persistence and backend bridges. Validation must not change their wire values
or CONFIG-001 ordering incidentally.

## Governing Contract Decisions

### Architecture domain

This branch's public ModelConfig, Torch registry, ModelSpec, and current
configuration guide expose:

```text
cnn
ffno
fno
hybrid
stable_hybrid
fno_vanilla
neuralop_uno
hybrid_resnet
hybrid_resnet_ffno_ptychoblock_encoder
hybrid_resnet_ptychoblock_ffno_encoder
spectral_resnet_bottleneck_net
spectral_resnet_bottleneck_linear_decoder
hybrid_resnet_ffno_bottleneck
hybrid_resnet_convnext_bottleneck
```

The external contract and configuration bridge carry the same 14 supported
values. Validation work preserves this complete Hybrid/ResNet-inclusive
architecture domain.

### Public spellings

- Public amplitude activation uses `sigmoid`, `swish`, `softplus`, or `relu`.
- Torch `silu` and `SiLU` remain Torch spellings and normalize to public
  `swish` in the existing Torch-to-public bridge.
- Stored public closed-domain values remain exact built-in strings.
- No enum type or generic string normalizer is introduced.

### Structural versus runnable validity

A dataclass can be structurally useful for inspection, compatibility decoding,
or test construction without being a runnable training or inference request.
Therefore validation is split:

- structural validation checks types, closed domains, local ranges, and
  cross-field coherence;
- runnable training validation adds requirements such as positive epochs,
  available training data, and sampling coherence; and
- runnable inference resource validation adds archive and test-data
  availability.

This prevents `nepochs=0` in a non-running compatibility record from silently
becoming a valid training plan while avoiding unnecessary constructor breaks.

### Canonical grouping field

`n_groups` is canonical. `n_images` is accepted only as a deprecated boundary
alias:

- when only `n_images` is supplied, it resolves to `n_groups`;
- when both are supplied with different values, resolution fails;
- when both are supplied with the same value, one canonical value survives
  and the compatibility warning is emitted once; and
- supported consumers read `n_groups`, not the deprecated input field.

Direct dataclass construction retains current `__post_init__` behavior during
the compatibility phase. The stricter conflict rule belongs to supported
source resolvers until a public constructor change is separately approved.

## Decision

Introduce two conceptual family-specific resolution boundaries:

```python
resolve_training_config(
    file_mapping,
    explicit_cli_patch,
) -> TrainingConfig

resolve_inference_config(
    file_mapping,
    explicit_cli_patch,
) -> InferenceConfig
```

Exact exported names and module placement are locked by the implementation
plan. There is no generic repository-wide resolver framework.

Each resolver:

1. receives an unmodified file mapping and a mapping containing only
   explicitly supplied CLI values;
2. validates the supported root envelope;
3. partitions flat model fields, an optional nested model section, and
   workflow fields;
4. rejects unknown and multiply owned keys;
5. resolves declared compatibility aliases and precedence;
6. constructs a complete private candidate;
7. resolves the model object policy for the selected backend;
8. performs structural and semantic validation;
9. returns a fresh public dataclass; and
10. leaves `params.cfg`, files, and other external state untouched.

Runnable and resource validation occurs at the consuming workflow boundary
after source resolution.

## Source And Precedence Contract

Supported public configuration sources use:

```text
dataclass defaults
    < file values
    < explicitly supplied CLI values
```

Argparse defaults are presentation defaults, not proof that a value was
supplied. Supported parsers preserve explicit presence, preferably by using
`argparse.SUPPRESS` for override arguments or an equivalent explicit-name set.

A file may retain the existing flat public training form and may contain a
nested `model` mapping. The resolver owns one deterministic conflict rule:

- a value may appear in either its supported flat location or its nested model
  location;
- equal duplicate values are accepted once, with the nested/flat duplicate
  recorded as source provenance rather than a second applied value;
- unequal duplicates fail with both locations named; and
- a key that belongs to neither ModelConfig nor the selected workflow fails.

Inference uses the same root-resolution rules rather than reading only the
nested model section.

The resolver never modifies the raw file or CLI mappings.

## Validation Layers

### Model structural validation

Owns:

- the 14 supported architectures, including the Hybrid/ResNet families;
- declared public closed strings;
- supported public `N` values at raw authoring and runnable boundaries;
- positive grid/filter/probe scale values;
- non-negative smoothing and mask constraints; and
- object layout, training canvas, weighting, and legacy object-policy joins.

Direct construction remains non-validating. Any programmatic instance entering
a supported runnable boundary is explicitly validated there.

### Training structural validation

Owns:

- nested ModelConfig structural validity;
- numeric and closed-domain field validity;
- loss-weight ranges and objective coherence;
- optimizer and scheduler spellings;
- sampling aliases and local sampling relationships; and
- Path representation without checking resource existence.

### Runnable training validation

Adds only requirements needed to begin training, including:

- `nepochs > 0`;
- a usable training-data path;
- positive photon and batch values;
- any required group/sample counts; and
- backend-specific coherence that belongs to the selected workflow.

### Inference structural validation

Owns:

- nested ModelConfig structural validity;
- canonical grouping and sampling values;
- backend and local scalar domains; and
- Path representation without filesystem access.

### Inference resource validation

Owns:

- model archive/path existence and supported layout;
- test-data existence; and
- any backend-specific artifact preconditions.

Filesystem checks do not run during generic construction, serialization,
legacy projection, or compatibility inspection.

## Mutation And Revalidation

ModelConfig remains frozen and uses `dataclasses.replace()`.

TrainingConfig and InferenceConfig remain mutable for compatibility. Supported
workflow code must either:

- resolve a new candidate after each source of mutation; or
- explicitly revalidate the instance after its final mutation and immediately
  before consumption.

Metadata-derived updates, such as photon count, reconstruct or replace the
candidate and rerun the affected structural/runnable validation. Direct
mutation followed by silent consumption is not a supported resolved boundary.

No assignment validation is added. It would alter existing mutation timing and
would not cover mutation inside list values.

## Legacy Projection

Resolution precedes the CONFIG-001 bridge:

```text
resolved and appropriately validated public config
    -> dataclass_to_legacy_dict()
    -> update_legacy_dict(params.cfg, config)
    -> legacy consumer
```

This design preserves:

- exact legacy key mappings;
- nested model flattening;
- Path-to-string behavior;
- the pure projection's representation of `None`;
- `update_legacy_dict()` skipping `None`;
- current public dataclass reflection and field order; and
- current persisted TensorFlow/Torch compatibility values.

Pydantic is not invoked by `dataclass_to_legacy_dict()` or
`update_legacy_dict()`. Runnable resource checks are not a projection
precondition.

Supported inference ordering must be reconciled with CONFIG-001 explicitly:
the workflow bridges the currently resolved request before inspecting
backend-specific legacy consumers, while archived state restoration remains
the load boundary's declared operation. Validation must not hide or reorder
that ownership.

## CLI Contract

CLI code continues to expose raw strings, numbers, booleans, and paths.
Argparse choices may be derived from `Literal` annotations only through a
focused helper that correctly unwraps direct and optional Literal types.

The helper:

- returns primitive choice values;
- does not construct enums or Pydantic objects;
- does not decide configuration ownership;
- does not apply defaults to a partial patch; and
- is shared only if it deletes the duplicated public parser logic.

File parsing remains with `yaml.safe_load`. A successfully parsed YAML value is
still untrusted until the appropriate public resolver accepts it.

## Pydantic Decision

Pydantic adoption is held. The only acceptable later shape is:

- unchanged stdlib dataclasses;
- unchanged stored Literal strings;
- one cached root adapter per adopted complete family boundary;
- mapping validation after file/CLI merge;
- strict instance revalidation only at explicit validation boundaries;
- explicit semantic and resource validators afterward; and
- existing serializers and projections unchanged.

It is not used for:

- direct constructor validation;
- assignment validation;
- partial CLI or bridge overrides;
- argparse types or choices;
- `params.cfg`;
- ModelSpec, artifacts, checkpoints, or MLflow; or
- filesystem and environment validation.

Before adoption, executable feasibility evidence must prove:

1. exact accepted scalar/coercion behavior, including booleans and numeric
   subclasses;
2. unknown-key handling at root and nested levels;
3. mapping construction and existing-instance revalidation;
4. Path input and stored representation;
5. `n_images` warnings, conflicts, and canonicalization;
6. direct and optional Literal CLI behavior;
7. exact public error facade;
8. dataclass signatures, reflection, positional construction, equality,
   freezing, replacement, and mutable behavior;
9. exact legacy and persistence fixtures; and
10. material net production deletion.

A Model-only adapter is insufficient. Adoption must delete the schema-aware
field filtering, manual path conversion, duplicated membership/type branches,
and preferably the duplicated training parser. Otherwise manual validation
remains the accepted architecture.

## Explicitly Rejected Designs

### Pydantic dataclasses or BaseModel

Rejected because they change constructor validation, exceptions, mutation,
reflection, `replace`, and wire behavior.

### Public string enums

Rejected because Literal validation is sufficient at an adopted boundary and
enums require normalization through CLI, bridges, artifacts, and legacy
projection.

### One input model parallel to each dataclass

Rejected because it creates a second field schema and conversion layer rather
than deleting duplication.

### Validation inside every constructor or assignment

Rejected because structurally useful compatibility records and mutable
workflow assembly do not share runnable-state requirements.

### Pydantic on partial overrides

Rejected because omitted patch values must not acquire defaults before source
precedence and alias resolution.

### Filesystem checks in structural validation

Rejected because persistence inspection and configuration transformation must
not depend on the local filesystem.

### Generic cross-family resolver

Rejected because public, Torch, simulation, execution, and artifact boundaries
have different ownership, completeness, and compatibility contracts.

## Complexity Budget

Implementation is acceptable only if it:

- leaves direct stdlib dataclass construction compatible;
- removes a duplicated or inconsistent source-resolution path;
- uses one family-specific ownership table or equivalent explicit code, not a
  generic registry;
- prevents argparse defaults from masquerading as explicit overrides;
- makes training and inference unknown-key behavior consistent;
- makes `n_groups` the actual canonical consumer field;
- separates structural, runnable, and resource validation;
- does not introduce Pydantic until its independent adoption gate passes;
- does not change persistence or legacy representations; and
- has a net reduction in production branching or duplicate parsing.

## Focused Acceptance Evidence

### Resolution

- file-only values survive when their CLI option was omitted;
- explicit CLI values override file values;
- unknown flat, nested model, and workflow keys fail deterministically;
- duplicate flat/nested model values follow the declared conflict rule;
- input mappings remain unchanged;
- training and inference return fresh complete dataclasses; and
- inference consumes root inference values rather than silently ignoring them.

### Aliases and mutation

- `n_images` alone resolves to `n_groups`;
- equal and conflicting dual inputs follow the declared rules;
- supported consumers use `n_groups`;
- metadata-derived updates are reconstructed or revalidated before use; and
- direct constructor compatibility remains characterized.

### Validation layering

- a structurally inspectable non-running record does not pass runnable
  training validation;
- structural inference validation performs no filesystem access;
- resource validation checks both model and test-data requirements; and
- backend/model/object-policy semantic joins fail when deliberately broken.

### CLI and compatibility

- direct and optional Literal choices remain correct;
- booleans and paths retain expected CLI behavior;
- legacy projection dictionaries are exact, including `None`;
- `update_legacy_dict()` retains skip-`None` behavior and CONFIG-001 ordering;
- Torch bridge public values and ModelSpec handshakes remain unchanged; and
- no Pydantic model, enum, normalizer, or serializer appears in the
  implementation.
