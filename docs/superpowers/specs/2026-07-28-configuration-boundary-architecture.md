# Configuration Boundary Architecture

**Status:** Approved and implemented on `refactor-internal` on 2026-07-28

**Companion designs:**

- `docs/superpowers/specs/2026-07-28-pydantic-simulation-validation-design.md`
- `docs/superpowers/specs/2026-07-28-public-config-resolution-design.md`
- `docs/superpowers/specs/2026-07-28-torch-config-resolution-design.md`
- `docs/superpowers/specs/2026-07-28-execution-config-ownership-design.md`
- `docs/superpowers/specs/2026-07-28-configuration-persistence-boundaries.md`

**Implementation state on `refactor-internal`:** Roadmap Slice 8 is
implemented. The representation-preserving simulation adapter, return-new
public and Torch resolvers, and execution-request split are present on this
branch. The post-isolation family gates adopted cached Pydantic adapters for
the public Model/Training/Inference family and retained explicit manual
validation for the Torch family. Existing internal ModelSpec and artifact
codecs remain current and were not replaced.

## Purpose

Define one architecture for configuration boundaries without forcing one
validation or persistence mechanism across configuration families with
different lifecycle contracts.

The architecture is a portfolio of independently justified treatments.
Pydantic is an optional structural validator for a complete closed snapshot.
It is not the configuration domain model, a partial-patch engine, a runtime
resolver, an artifact codec, or a replacement for the legacy bridge.

## Governing Context

- `specs/ptychodus_api_spec.md` owns the external configuration and
  compatibility contract.
- `docs/CONFIGURATION.md` explains current field ownership and representation
  layers.
- `docs/workflows/pytorch.md` owns supported PyTorch workflow sequencing.
- CONFIG-001 requires supported legacy-touching workflows to project resolved
  public configuration into `ptycho.params.cfg` before legacy consumers run.
- The accepted simulation design defines one narrow,
  representation-preserving use of Pydantic, now implemented internally.
  Adoption by the public family followed its own measured gate; the Torch
  family's separate gate terminated with `retain manual`.

Where current specifications, guides, and implementation disagree, the
conflict must be resolved in the owning contract before a validator freezes
one interpretation into runtime behavior.

## Decision

Every supported configuration path is decomposed into these stages:

```text
syntax bytes
    -> parser
    -> raw mapping or explicit partial patch
    -> precedence and compatibility resolution
    -> complete candidate snapshot
    -> structural validation
    -> semantic and cross-record validation
    -> resource/environment validation when required
    -> resolved stdlib dataclass bundle
    -> one-way compatibility projection or explicit persistence
```

One function may implement adjacent stages where that is simpler, but the
contracts remain distinct. In particular:

1. A partial patch is never treated as a complete configuration.
2. Defaults are applied during resolution, not inferred by a patch validator.
3. Side effects occur only after the complete candidate passes the validation
   required by that consumer.
4. Serialization and artifact identity remain explicit and versioned.
5. Environmental resolution remains explicit and testable independently of
   structural validation.

## Boundary Kinds

### Complete authoring document

A complete document boundary may accept omitted fields that have declared
defaults, then construct one complete snapshot. Unknown-key and coercion rules
must be declared for the whole document. `SimulationConfig` is the reference
case.

### Partial override

CLI arguments, study overrides, notebook updates, and compatibility payloads
are patches. A patch contract declares:

- which fields it owns;
- whether unknown fields are rejected, warned, or retained for a later owner;
- how explicit presence is distinguished from a default;
- alias conflict behavior; and
- precedence relative to other sources.

A partial patch is resolved into a complete candidate before structural
validation. Pydantic defaults must not turn an omitted patch value into an
explicit override.

### Resolved runtime snapshot

A runtime snapshot contains the effective values consumed by one workflow.
Mutable source records must be revalidated after their last supported mutation
and before consumption, or the resolver must return a replacement snapshot
that no supported path mutates.

### Environmental request

Filesystem existence, NPZ metadata, CUDA availability, device counts, and
optional dependency availability are environmental facts. They remain outside
generic structural validation. A request and its resolved runtime value must
not be silently conflated where provenance matters.

### Versioned persistence

ModelSpec, Torch artifact identity, checkpoints, and tensor envelopes have
versioned schemas and explicit compatibility upgrades. Exact historical field
sets, tensor representation, and defensive-copy semantics belong to those
codecs.

### Legacy projection

`ptycho.params.cfg` is a one-way compatibility projection. It is never a new
structured configuration source. Projection follows successful configuration
resolution and preserves its existing declared `None` behavior.

## Shared Invariants

1. Public and Torch configuration carriers remain stdlib dataclasses unless a
   separately approved design proves a replacement is compatible and simpler.
2. Closed string domains remain `Literal[...]` and stored values remain exact
   built-in strings. No enum catalog or enum normalization layer is introduced.
3. Pydantic serialization, `BaseModel`, assignment validation, and generic
   repository-wide configuration registries are outside this architecture.
4. Syntax parsing remains with `yaml`, `tomllib`, and `json`.
5. Argparse produces primitive values. It does not use an enum or Pydantic
   model as its `type`.
6. Domain semantics remain explicit Python: object-policy joins, scale
   contracts, channel joins, alias conflicts, runnable-state requirements, and
   resource checks are not hidden in generic schemas.
7. Raw Pydantic exception formatting is not a public error contract. A family
   that adopts an adapter owns a stable domain error facade.
8. A validator must not change artifact bytes, legacy keys, dataclass
   reflection, constructor signatures, mutation timing, tensor aliasing, or
   failure ordering without an explicit governing contract change.

## Portfolio Decisions

| Boundary | Treatment | Pydantic decision |
|---|---|---|
| Simulation recipe | Cached complete-document adapter over unchanged stdlib dataclasses | Adopted and implemented |
| Public Model/Training/Inference | Return-new resolution plus cached complete-snapshot adapters; semantics and resources remain explicit | Adopted and implemented after a 17-line conservatively measured adapter-attributable net deletion |
| Torch Data/Model/Training/Inference | Explicit resolver and transactional patch architecture | Retain manual: the 157-field exactness surface and 194-line addition floor exceed the 109-line adoption-favorable deletion ceiling |
| `DatagenConfig` | Compatibility view delegated to `SimulationConfig` | No independent schema |
| `PyTorchExecutionConfig` | Separate ownership, provenance, and environmental resolution | Retain manual validation |
| CLI patches | Preserve explicit presence, merge, then validate complete result | No direct adapter |
| `params.cfg` | Validated one-way compatibility projection | Reject |
| ModelSpec/artifacts/checkpoints | Existing exact, versioned codecs | Reject |
| MLflow configuration dictionaries | Observability and legacy compatibility only | Reject as authoritative identity |

## Pydantic Adoption Gate

Another family may adopt a cached `TypeAdapter` only when all of the following
are demonstrated:

1. The complete raw or instance boundary is named and has one accepted-value
   table.
2. Partial patches do not enter the adapter before resolution.
3. Mutation timing and revalidation points are complete and enforceable.
4. Strict scalar, container, unknown-key, default, and instance behavior are
   proven with the installed Pydantic version rather than assumed.
5. Existing dataclass reflection, signatures, equality, `replace`, warnings,
   and public error behavior remain compatible.
6. Existing wire representations, artifact fixtures, and legacy projections
   remain exact.
7. The implementation deletes schema-aware production branches or duplicate
   parsing code and has a clear net complexity reduction.
8. No parallel input model, enum catalog, generic normalizer, or Pydantic
   serializer is added.

Failure of any gate means the family retains explicit manual validation.

## Contract Alignment Outcomes

The implementation resolves the former alignment prerequisites as follows:

- The public architecture domain is the 14 values, including the Hybrid/ResNet
  families, shared by the public dataclass, Torch registry, ModelSpec,
  configuration guide, configuration bridge, and external API contract.
  Future expansion requires an explicit registry and persistence decision.
- The public accepted-value table distinguishes structural validity from
  runnable/resource validity for `N`, amplitude activation, numeric fields,
  and incomplete training records.
- Mapping-based training, inference, notebook, and Torch factory entry points
  fail closed on unknown keys.
- CLI presence is explicit, so omitted argparse values do not overwrite YAML
  or baseline values.
- `n_groups` is canonical and the deprecated `n_images` input is normalized at
  the public boundary.
- Execution topology and optimizer aliases are retired. `ExecutionRequest`
  owns unresolved runtime provenance, Torch Model/Training records own
  scientific choices, and `PyTorchExecutionConfig` is the capability-resolved
  runtime output.

## Explicitly Rejected Architecture

### Repository-wide Pydantic models or dataclasses

Rejected because constructor validation, mutable patches, execution
provenance, tensors, and versioned artifacts do not share one lifecycle.

### Central schema or validation registry

Rejected because it would become a second ownership system spanning public,
Torch, execution, simulation, and persistence fields.

### Pydantic as YAML/TOML or argparse bridge

Rejected because syntax parsing and partial override provenance precede
complete-snapshot validation.

### Pydantic artifact serialization

Rejected because it cannot replace exact versioned field sets, tensor
envelopes, deterministic upgrades, and ModelSpec clone isolation.

## Complexity Budget

Implementations under this architecture must:

- prefer family-specific resolvers over a generic configuration framework;
- remove or quarantine an old path when adding its supported replacement;
- avoid moving fields between owners without an explicit compatibility design;
- avoid new persistent formats unless a version change is the stated goal;
- keep supplemental checks from becoming unrelated completion gates; and
- use the smallest focused evidence set that can falsify the affected
  compatibility claim.

## Acceptance

This architecture is satisfied when each child design:

- names its complete and partial boundaries;
- assigns ownership for resolution, validation, resources, projection, and
  persistence;
- records an adopt, hold, or reject decision for Pydantic;
- states compatibility and complexity gates; and
- can be implemented and verified independently.
