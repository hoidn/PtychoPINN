# Torch Configuration Resolution and Transactional Patch Design

**Status:** Approved  
**Approved:** 2026-07-28  
**Scope:** PyTorch runtime configuration resolution and compatibility patching  
**Primary implementation surfaces:** `ptycho_torch/config_factory.py`, `ptycho_torch/config_params.py`

## 1. Purpose

This design defines how the PyTorch configuration boundary turns user- or
compatibility-supplied partial inputs into validated runtime configuration
bundles.

The selected architecture is an explicit, phase-aware resolver that constructs
return-new candidates, validates the whole candidate bundle, and only then
allows side effects. Supported training and inference paths do not mutate a
caller-owned configuration object in place.

This design also separates three contracts that must not be conflated:

1. a **partial patch**, in which omission means "leave the baseline value
   unchanged";
2. a **resolved runtime bundle**, in which every runtime field has an effective
   value and cross-record joins hold; and
3. a **complete persisted snapshot**, whose exact keys, versions, tensor
   encoding, and migration rules belong to its persistence schema.

The third contract remains outside this design. Runtime resolver convenience
must not weaken or silently redefine persistence identity.

## 2. Governing context

This design follows the repository authority stack routed by `docs/index.md`.
In particular:

- `docs/CONFIGURATION.md` defines factory-resolved Torch dataclasses,
  phase-local payloads, the one-way legacy bridge, and the channel-count and
  model-policy joins.
- `specs/ptychodus_api_spec.md` requires strict portable artifacts and keeps
  the legacy TensorFlow bridge one-way.
- `docs/specs/spec-ptycho-interfaces.md` requires explicit, versioned handling
  of known legacy artifact eras.
- `docs/superpowers/specs/2026-07-28-pydantic-simulation-validation-design.md`
  owns the canonical simulation-validation boundary and explicitly excludes
  Torch runtime configuration, execution configuration, ModelSpec, and
  artifact persistence.
- `docs/superpowers/specs/2026-07-28-execution-config-ownership-design.md`
  owns the field-by-field handoff and precedence between execution
  compatibility inputs, effective Torch TrainingConfig values, and resolved
  Trainer arguments.

When these sources evolve, their current contract text outranks historical
resolver behavior and this design must be revised explicitly if its boundary
changes.

## 3. Scope and ownership

### 3.1 In scope

This design owns:

- resolution of the Torch `DataConfig`, `ModelConfig`, `TrainingConfig`, and
  `InferenceConfig` records used by training and inference payloads;
- parsing and classification of flat phase-specific override mappings;
- explicit aliases and derived runtime values;
- return-new application of partial patches;
- consistent unknown-key and audit behavior across training and inference;
- semantic joins across records in a resolved runtime bundle;
- ordering of environmental checks and mutating side effects;
- the compatibility boundary around `update_existing_config()`; and
- delegation from `DatagenConfig` to the canonical simulation configuration.

### 3.2 Outside scope

This design does not change:

- `ModelSpec`, its field sets, equality, tensor-copy behavior, or schema
  migrations;
- the portable artifact envelope, artifact tensor codec, artifact versions, or
  exact snapshot field sets;
- Lightning checkpoint hyperparameters or checkpoint loading;
- artifact identity sidecars or bundle manifests;
- MLflow parameter serialization or MLflow's role as non-authoritative
  metadata;
- `PyTorchExecutionConfig` ownership or its environment-dependent resolution;
- canonical simulation schema ownership;
- public CLI syntax; or
- the external Ptychodus API contract.

No runtime resolver output is authoritative persisted identity merely because
it is complete in memory. The existing ModelSpec and artifact codecs remain the
only owners of their respective wire contracts.

Although execution ownership is outside this design, the training resolver
implements the companion execution design's declared handoff. Explicit
canonical TrainingConfig overrides have priority over explicit
optimizer-adjacent execution compatibility inputs, which have priority over
the resolved TrainingConfig value and its defaults. The resolver writes the
effective value only to Torch TrainingConfig; any Lightning clipping or
accumulation argument is derived from that effective owner. It does not invent
a second precedence rule.

## 4. Current problem

The present Torch boundary mixes several behaviors:

- training rejects unknown overrides while inference can silently ignore them;
- both factories initially copy raw inputs into `overrides_applied`, so an
  unused key can look applied;
- field-name reflection can make ownership implicit;
- aliases, inference, environmental defaults, semantic joins, and audit
  recording occur within the same procedural flow;
- `update_existing_config()` mutates an existing record with `setattr`,
  bypasses constructor and `__post_init__` validation, and tolerates unknown
  keys; and
- legacy JSON and MLflow loaders apply partial mappings to defaults even when
  their names suggest that they loaded a complete configuration.

These properties make failure non-atomic and make it difficult to distinguish
consumed input, derived state, and ignored input. They also make a direct
Pydantic `TypeAdapter` substitution look smaller than it is: the hard contract
is bundle resolution, not individual field annotation.

## 5. Architectural decision

Supported Torch paths shall use an explicit resolver with two phase-specific
entry points:

- a training resolver that produces the configuration records and audit data
  needed by `TrainingPayload`; and
- an inference resolver that produces the configuration records and audit data
  needed by `InferencePayload`.

`create_training_payload()` and `create_inference_payload()` remain the public
factory facades. Their internal resolution contract is:

```text
baseline records + partial phase patch + explicit observations
    -> classify names and aliases
    -> parse supplied values
    -> construct return-new candidate records
    -> calculate declared derivations
    -> validate bundle-level semantic joins
    -> perform remaining read-only environmental validation
    -> publish resolved payload and audit
    -> perform explicitly requested side effects
```

Every step through environmental validation is transactional with respect to
caller-visible state. Failure returns an error and leaves all caller-owned
records, mappings, global configuration, filesystem state, and emitted audit
unchanged.

The resolver is explicit rather than reflection-driven. Each supported phase
has a declared input registry that records, for every accepted external name:

- its canonical name;
- its owning target record or execution input;
- its accepted aliases, if any;
- its value parser or validator;
- whether it is user-supplied, derived, or environmentally observed; and
- how its effective value is represented in runtime audit data.

Dataclass field discovery may help assert that an internal constructor remains
complete. It must not decide which external names are accepted or fan one
external value into every record with a matching field name.

## 6. Configuration lifecycle and boundaries

### 6.1 Raw patch boundary

A raw patch is an untrusted partial mapping. Missing values have no meaning
beyond "not supplied." The resolver must preserve suppliedness until aliases,
conflicts, and derivations have been decided.

At this boundary:

- no caller mapping is modified;
- no default is inserted into the raw mapping;
- no dataclass instance is treated as validated merely because it has the
  expected Python type;
- no global configuration is updated; and
- no warning or audit event describes an input as applied.

### 6.2 Candidate boundary

The resolver constructs fresh outer dataclass records from the baseline plus
the normalized patch. It must use constructors or an equivalently complete
validation path, never a sequence of `setattr` operations on the baseline.

A candidate is private to resolution until all field and bundle checks pass.
The resolver must not mutate values owned by the caller. Mutable values such as
tensors require an explicit field policy: either they remain read-only through
this boundary or they are defensively copied before any downstream operation
that may mutate them. Return-new refers at minimum to the configuration
containers; it is not permission for hidden mutation through aliased values.

### 6.3 Resolved bundle boundary

A resolved bundle is a complete runtime candidate for its phase. It has:

- effective values for every required runtime field;
- declared derived values;
- valid cross-record joins;
- completed read-only environmental validation required by the factory; and
- deterministic audit data describing consumed and resolved effective inputs.

Only a resolved bundle may be published as a training or inference payload.
Downstream code may rely on its joins without reinterpreting the original
patch.

### 6.4 Side-effect boundary

Mutation is a separate commit step after resolution succeeds. Examples include:

- populating `ptycho.params.cfg` through the required one-way legacy bridge;
- emitting deferred deprecation warnings;
- creating an output directory, when an owning workflow requests it; and
- writing logs or tracking metadata.

The commit step receives a resolved bundle; it does not repair or finish
validation. If a global bridge update can fail partway, the bridge must stage
the complete replacement or restore its prior state before propagating the
error. A failed commit must not leave a partially updated `params.cfg`.

Read-only observations needed to decide validity—such as inspecting an NPZ
shape, checking path readability, or resolving available accelerator
capabilities—may occur before commit. They must be explicit resolver inputs or
isolated validation dependencies so tests can prove that no mutation occurred.

## 7. Complete snapshots and partial patches are different APIs

No API may infer snapshot-versus-patch semantics from which keys happen to be
present.

| Contract | Missing key | Unknown key | Alias | Defaulting | Result |
|---|---|---|---|---|---|
| Partial patch | leaves baseline unchanged | error | only from an explicit phase registry | baseline values may survive | return-new resolved runtime bundle |
| Complete snapshot | error relative to its owning schema | error | forbidden unless that schema versions it | forbidden | exact decoded snapshot |

A complete snapshot is complete relative to the schema that owns it. For
example, the artifact schema intentionally persists selected complete
`DataConfig`, `TrainingConfig`, and `InferenceConfig` projections; it does not
therefore persist every dataclass field.

The runtime factory accepts partial patches. If a new mapping-based complete
snapshot API is introduced, it must:

- have a distinct name and type from the patch API;
- compare the incoming key set with the exact schema-owned field set before
  construction;
- reject missing and extra keys;
- reject implicit aliases and environment-dependent defaults;
- perform explicit version migration before current-schema validation; and
- preserve the representation and error contract of its owning schema.

Existing artifact and ModelSpec decoders already own such exact boundaries and
are not rerouted through the runtime patch resolver by this design.

Legacy JSON and MLflow loaders that fill missing values from defaults are
partial compatibility loaders, regardless of historical names or log
messages. Their output is not a trusted complete snapshot until it has passed
the supported resolver boundary.

## 8. Transactional patch semantics

Patch resolution shall have the following observable behavior:

1. Snapshot or otherwise protect the phase baseline without mutating it.
2. Classify every raw key using the phase's explicit registry.
3. Reject unknown names and ambiguous alias/canonical combinations.
4. Parse all supplied values without updating a candidate incrementally in
   externally visible state.
5. Apply normalized values to fresh candidate records.
6. Calculate only declared derivations.
7. Validate every relevant bundle-level semantic join.
8. Complete required read-only environmental checks.
9. Build audit data from consumed inputs and resolved effective values.
10. Return the new bundle. Side-effecting consumers may then commit it.

An error at any step discards the private candidate. There is no rollback of a
partly mutated caller object because supported resolution never mutates that
object.

Aliases are compatibility syntax, not additional stored fields. If both an
alias and canonical name are supplied:

- different effective values are an error;
- equal effective values are accepted once under the canonical name in both
  training and inference; and
- audit records the canonical effective input plus alias provenance, never two
  independently applied settings.

Derived fields are owned by named rules. For example, channel counts derived
from `grid_size` are not ordinary user-overridable values in a path that owns
that derivation. A supplied value that conflicts with a derived invariant is
an error, not a value that is silently overwritten.

## 9. Unknown-key and audit policy

Training and inference shall use the same fail-closed policy:

- every supplied raw key is consumed by a declared rule or causes resolution
  to fail;
- error reporting identifies unknown keys deterministically, preferably in
  sorted order;
- spelling suggestions may be offered, but never turn an unknown key into an
  accepted one;
- a field accepted by one phase is not automatically accepted by another; and
- execution controls and configuration values retain distinct owners even
  when they share a flat input surface.

`overrides_applied` is a runtime audit view, not a copy of raw input and not a
persistence snapshot. It contains only:

- recognized supplied values after canonicalization;
- resolved effective values that the factory intentionally exposes for audit;
- declared derived values and their provenance; and
- declared environmental selections and their provenance.

It never contains:

- silently ignored unknown keys;
- an alias as though it were a second setting;
- values observed only during a failed transaction;
- arbitrary unconsumed input; or
- claims of application for values that a later rule replaced without
  recording the replacement.

Audit ordering and provenance labels must be deterministic enough for focused
tests. Audit representation must not be reused as authoritative artifact or
checkpoint identity.

## 10. Bundle-level semantic joins

Individual dataclass construction is necessary but insufficient. The resolver
shall validate the coherent phase bundle before publishing it.

At minimum, applicable joins include:

- `DataConfig.C` equals the product represented by
  `DataConfig.grid_size`;
- `ModelConfig.C_model` and `ModelConfig.C_forward` equal
  `DataConfig.C`;
- the Torch model projection agrees with the canonical shared model
  configuration for fields owned by that projection;
- the resolved training loss identity agrees across training and model
  consumers;
- scale-profile name and scale values form an allowed pair;
- object representation and object-policy decisions are resolved once and
  agree across consumers;
- simulation scan geometry agrees with the model grid where a simulation
  configuration participates in the boundary; and
- a training-only control cannot silently alter inference structure, or vice
  versa.

The exact validator may call existing policy helpers such as
`resolve_torch_model_object_policy()`, but policy resolution must return a new
candidate or occur before publication. Downstream model constructors must not
silently repair a supposedly resolved bundle.

Inference resolution intentionally does not reconstruct authoritative model
identity from loose overrides. Structural identity continues to come from the
saved ModelSpec or artifact at the application boundary. The inference
resolver validates only the runtime records and joins it owns.

Execution topology and environment selection remain separate from scientific
model identity. They may be included in runtime audit, but must not leak into
ModelSpec derivation.

## 11. Ordering of validation and side effects

The supported ordering is:

1. raw key classification;
2. value parsing and local validation;
3. return-new candidate construction;
4. derivation;
5. bundle semantic validation;
6. read-only environmental validation;
7. resolved audit construction;
8. payload publication;
9. deferred side effects.

In particular:

- `update_legacy_dict(params.cfg, config)` runs only with a fully resolved
  configuration and only when the owning workflow requires the legacy bridge;
- a failed patch does not touch `params.cfg`;
- output directory creation does not precede configuration validity;
- warnings are accumulated during normalization and emitted after successful
  resolution; and
- logging and tracking receive resolved values, not speculative candidates.

If a workflow requires a side effect before another subsystem can validate,
that subsystem is not part of the pure resolver. Its adapter must make the
temporary state explicit and provide atomic restore-on-failure behavior.

## 12. `update_existing_config()` compatibility island

`ptycho_torch.config_params.update_existing_config()` remains available only
for named legacy compatibility consumers that depend on its current behavior:

- in-place mutation;
- partial mapping semantics;
- ignored unknown keys unless verbose output is requested; and
- no reconstruction through dataclass validation.

This behavior is not a supported resolver contract. Therefore:

- new code must not call `update_existing_config()`;
- `create_training_payload()` and `create_inference_payload()` must not use it;
- supported loaders must not expose its mutated object as a resolved bundle
  without passing through the return-new resolver;
- its tolerant unknown-key behavior must not leak into factory audit; and
- no additional configuration type or consumer may be added to this island
  without a separately approved compatibility requirement.

Tests for the island are characterization tests: they preserve existing legacy
behavior while preventing it from becoming the model for new code. Removing
or tightening the function requires migration of every named legacy caller and
is a separate compatibility decision.

## 13. `DatagenConfig` delegates to canonical simulation validation

`DatagenConfig` is a six-field legacy projection, not an independent complete
simulation schema. It shall retain its current external field names and use:

- `DatagenConfig.to_simulation_config()` to enter the canonical simulation
  boundary; and
- `DatagenConfig.from_simulation_config()` to return to the legacy projection.

Torch resolution must not duplicate scan geometry, probe, sample, noise, or
simulation semantic rules in a second validator. Unknown simulation fields,
canonical defaults, representation normalization, and semantic errors belong
to the accepted simulation-validation design.

The delegation contract requires:

- conversion through the public canonical simulation constructors;
- canonical validation before a projection is treated as resolved;
- round-trip preservation for the six supported legacy fields; and
- an explicit error for information that cannot be represented when a caller
  requests a lossless legacy projection.

This design does not expand `DatagenConfig` into persistence identity and does
not create a Torch-specific Pydantic simulation model.

## 14. Pydantic decision: held, not selected

Pydantic is not prescribed for Torch runtime configuration by this design.
The decision is based on executable feasibility probes against Pydantic
2.12.3 and the current stdlib dataclasses:

- `TypeAdapter(DataConfig)`, `TypeAdapter(TrainingConfig)`,
  `TypeAdapter(InferenceConfig)`, and `TypeAdapter(DatagenConfig)` construct,
  but their default mapping behavior ignores extra keys, fills missing
  defaults, and coerces values such as numeric strings, integral floats, and
  booleans;
- `validate_python(mapping, strict=True)` rejects a mapping wholesale instead
  of providing the required exact, non-coercing mapping boundary;
- validating an already constructed invalid dataclass instance returns the
  same instance without revalidation under default settings;
- `TypeAdapter(ModelConfig)` fails schema generation on `torch.Tensor`;
- passing `ConfigDict` directly to `TypeAdapter` for a stdlib dataclass raises
  a Pydantic usage error because configuration must be attached to the class;
- an ephemeral configured dataclass can enable arbitrary tensor types, forbid
  extras, and revalidate instances, but it constructs another dataclass while
  retaining tensor aliasing; and
- JSON dumping that configured model still fails for a tensor without a custom
  serializer.

These probes demonstrate that a local adapter call would neither replace
bundle semantics nor preserve current tensor and wire behavior by default.
They are feasibility facts, not a permanent prohibition.

Pydantic may be reconsidered only when a proposed boundary satisfies all of
the following adoption gates with executable evidence:

1. It receives a complete raw mapping at one owned boundary rather than
   validating values after legacy preprocessing.
2. It enforces the selected exactness and coercion policy for every accepted
   field, including Python's `bool`-as-`int` edge.
3. It implements identical unknown-key and alias-conflict behavior for
   training and inference.
4. It revalidates pre-existing instances or excludes them from the raw
   boundary explicitly.
5. It defines tensor validation, ownership, aliasing, copying, and
   serialization behavior without changing ModelSpec or artifact bytes.
6. It preserves current public exception categories and stable message
   fragments, or an approved compatibility facade translates them.
7. It still invokes explicit bundle-level semantic validators.
8. It preserves the separation of patch APIs from complete snapshot decoders.
9. It deletes a concrete, material amount of hand-written parsing or branching
   rather than adding a parallel representation.
10. It passes byte- and value-sensitive persistence fixtures even though
    persistence remains outside the adopted boundary.

Until every gate passes, explicit Python resolution and semantic validation
remain the approved architecture.

## 15. Persistence and ModelSpec exclusion

Resolution ends with validated runtime records and phase audit. Existing
downstream persistence steps retain their current authority:

```text
resolved runtime model config
    -> derive_model_spec(...)
    -> existing ModelSpec schema and copy rules
    -> existing artifact/checkpoint encoders
```

This design does not:

- add resolver audit to ModelSpec;
- infer ModelSpec from `overrides_applied`;
- add `DatagenConfig` or execution controls to artifact identity;
- replace exact artifact field-set checks with dataclass defaults;
- replace the tagged tensor codec with Pydantic JSON; or
- make MLflow config parameters sufficient to reload a model.

Any change to those behaviors requires a persistence design with its own
version, migration, compatibility, and fixture evidence.

## 16. Required invariants

An implementation conforming to this design maintains all of these invariants:

1. Supported training and inference resolution is return-new.
2. A failed resolution leaves the baseline, caller mapping, global
   configuration, filesystem, warnings, and audit unchanged.
3. Every raw key is consumed or rejected.
4. Training and inference share the same unknown-key and audit rules.
5. Aliases are explicit, phase-scoped, conflict-checked, and canonicalized.
6. Derived values have named owners and cannot silently overwrite conflicting
   user input.
7. `overrides_applied` contains only consumed or resolved effective inputs.
8. Resolved records satisfy constructor-level validation and all applicable
   bundle joins.
9. Read-only environmental observations are explicit and precede mutation.
10. Mutating side effects occur only after successful validation.
11. The legacy in-place updater is isolated from supported resolution.
12. Datagen semantics come from the canonical simulation boundary.
13. Runtime audit is not persistence identity.
14. ModelSpec, artifact, checkpoint, sidecar, and MLflow wire behavior are
    unchanged by this design.

## 17. Error contract

Resolution errors shall identify the boundary and cause without exposing
partly applied state. Stable categories include:

- unknown input name;
- alias/canonical conflict;
- invalid field value;
- conflicting derived value;
- invalid bundle semantic join;
- failed environmental precondition; and
- failed side-effect commit.

Multiple independent unknown names should be reported together when practical.
Semantic validation may fail fast where later checks depend on an earlier
invariant. Error ordering must be deterministic.

A side-effect commit failure is distinct from a resolution error because the
candidate was valid. The adapter must still satisfy its atomicity contract,
including restoration of global bridge state where applicable.

## 18. Rejected alternatives

### 18.1 Direct `TypeAdapter` wrapping of current dataclasses

Rejected because default extra handling, coercion, missing-value behavior,
instance revalidation, tensor schema support, and tensor serialization do not
match the boundary. It also leaves aliases, derivations, and semantic joins in
the existing procedural code.

### 18.2 New Pydantic domain models parallel to every Torch dataclass

Rejected because it adds another representation and conversion layer without
demonstrated branch deletion. It would also create pressure to reuse runtime
models for persistence, where the contracts differ.

### 18.3 One generic reflective patcher

Rejected because matching on dataclass field names makes phase ownership and
fan-out implicit. It cannot express safely that some identical names are
joined, derived, phase-local, or persistence-excluded.

### 18.4 In-place mutation followed by validation and rollback

Rejected because rollback must account for post-init effects, nested mutable
values, tensors, warnings, global state, and partial failure. Return-new
candidates make invalid intermediate state unobservable.

### 18.5 Expanding `update_existing_config()` into the supported path

Rejected because its tolerant unknown handling and `setattr` mutation are the
opposite of the selected resolver contract. Its behavior survives only for
legacy callers.

### 18.6 Treating runtime audit or MLflow parameters as a snapshot

Rejected because audit is selective and provenance-oriented, while MLflow
serialization is intentionally lossy for some values. Neither has the exact
field-set, tensor, version, or migration guarantees of the artifact schema.

### 18.7 Unifying runtime, simulation, execution, and persistence models

Rejected because these boundaries have different owners, completeness rules,
environment dependencies, and compatibility obligations. Shared scalar
helpers are allowed; shared ownership is not implied.

## 19. Focused acceptance evidence

Conformance is established with the smallest fresh evidence that can falsify
the complete resolver claim.

### 19.1 Resolver contract tests

Focused tests shall demonstrate:

- successful training and inference patches return fresh configuration
  containers and do not mutate baselines or input mappings;
- an error after one or more values have been parsed still exposes no partial
  candidate;
- training and inference both reject unknown names and report them
  deterministically;
- canonical/alias duplicates and conflicts follow the same rule in both
  phases;
- accepted fields are routed only to their declared owners;
- missing patch keys retain baseline values;
- conflicting supplied and derived channel counts fail;
- channel, loss, scale-profile, object-policy, and applicable simulation-grid
  joins fail when deliberately broken;
- audit contains consumed canonical values, declared derivations, and
  provenance, but excludes unknown or superseded raw input; and
- inference does not claim to reconstruct saved structural model identity.

### 19.2 Side-effect tests

Spies or isolated state fixtures shall demonstrate:

- no `params.cfg` update, directory creation, warning, log, or tracking event
  occurs on failed resolution;
- required side effects occur only after a valid bundle exists;
- a failing global bridge commit restores its prior state; and
- a successful bridge receives the complete resolved legacy projection
  exactly once.

### 19.3 Compatibility tests

Focused characterization shall demonstrate:

- named legacy callers of `update_existing_config()` retain their existing
  partial, in-place behavior;
- no supported factory calls the legacy updater;
- legacy-loaded partial mappings become trusted only after supported
  resolution; and
- `DatagenConfig` round-trips its supported six fields through the canonical
  simulation configuration and surfaces canonical validation failures.

### 19.4 Regression boundary

Because persistence is excluded, acceptance does not require redesigning or
rewriting broad artifact suites. Existing value- and byte-sensitive ModelSpec,
artifact, checkpoint, and tensor fixtures are targeted regression evidence
when a resolver change can affect their inputs. Any changed persisted bytes,
schema keys, version, tensor ownership, or checkpoint identity is a design
violation, not an incidental fixture update.

The implementation is conforming only when the focused tests pass in the
repository's required Python environment and the diff contains no unapproved
changes to the excluded persistence surfaces.
