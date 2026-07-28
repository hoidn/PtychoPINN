# Configuration Persistence And Compatibility Boundaries

**Status:** Approved on 2026-07-28

**Parent architecture:** `docs/superpowers/specs/2026-07-28-configuration-boundary-architecture.md`

## Purpose

Define which configuration representations are authoritative at persistence
and compatibility boundaries, and record why Pydantic is not their serializer.

## Representation Ownership

| Representation | Owner | Authority |
|---|---|---|
| Public stdlib dataclasses | Public workflow configuration | Current model/data/training/inference request |
| Torch runtime dataclasses | Torch factory/runtime | Effective backend-specific carrier |
| `ModelSpec` portable schema | Torch construction and reload | Sealed model graph/state identity |
| Torch artifact identity | Versioned artifact codec | Portable runtime configuration identity |
| Lightning checkpoint dictionaries | Checkpoint compatibility lane | Dual-written state checked against ModelSpec |
| MLflow config dictionaries | Observability/legacy reload lane | Informative, not authoritative model identity |
| `ptycho.params.cfg` | Legacy compatibility bridge | Flat runtime projection, never a structured source |
| `DatagenConfig` | Legacy data-generation compatibility | Lossy view delegated to SimulationConfig |

## Decision

Keep all current persistence boundaries explicit and versioned. Pydantic does
not dump or load ModelSpec, Torch artifact identity, Lightning checkpoints,
MLflow configuration dictionaries, TensorFlow configuration payloads, or
`params.cfg`.

Runtime validation may be applied only after a versioned decoder has:

1. identified the schema era;
2. checked the exact envelope and field set required by that era;
3. decoded tensor envelopes;
4. performed the declared deterministic upgrade; and
5. constructed the current runtime records.

The runtime validator does not replace historical-schema validation.

## Torch ModelSpec

ModelSpec remains a frozen, versioned structural identity with:

- explicit portable field tuples independent of current dataclass reflection;
- deterministic portable-v1 to portable-v2 upgrade;
- exact unknown/missing-field rejection;
- defensive copies of tensor and mutable values;
- canonical/Torch ownership joins; and
- a fresh materialized Torch ModelConfig for each consumer.

These behaviors are domain and compatibility logic. A generic schema dump
would weaken the requirement that adding a runtime field forces an explicit
schema-version decision.

## Torch Artifact Identity

The artifact codec remains responsible for:

- backend and schema markers;
- the exact top-level envelope;
- version-specific data, training, and inference field sets;
- ModelSpec payload decoding;
- tensor dtype, shape, real/complex data envelopes;
- CI statistics normalization; and
- cross-record channel and model joins.

The accepted lifecycle is:

```text
JSON/container value
    -> tensor-envelope decode
    -> exact versioned field-set validation
    -> declared upgrade
    -> runtime dataclass construction
    -> current semantic/join validation
    -> publish decoded identity
```

Pydantic must not run before the versioned decoder or silently apply current
defaults to an older schema.

## Lightning Checkpoints

Current checkpoints may dual-write ModelSpec and complete configuration
dictionaries. ModelSpec is the structural authority. Dual-written values must
agree before strict state loading.

Legacy checkpoints without ModelSpec retain only their explicitly supported
upgrade/fallback behavior. A current configuration validator must not turn
legacy default filling into a claim that the checkpoint carried current
identity.

Adding a configuration field to a Lightning checkpoint is not automatically an
artifact-schema change, but any field that affects construction must be
represented in ModelSpec under a deliberate schema decision.

## MLflow And Sidecars

Existing MLflow configuration dictionaries have tolerant, era-specific
behavior:

- missing values may receive defaults;
- unknown fields may be ignored;
- tensor fields may be omitted or encoded differently; and
- some loaders fall back rather than failing closed.

Therefore they are observability and compatibility data, not portable model
identity.

New integrations should log or reference the authoritative artifact identity.
A future strict MLflow or sidecar envelope requires a new versioned contract;
it must coexist with, rather than reinterpret, historical loose dictionaries.

## Legacy `params.cfg`

The compatibility direction is one-way:

```text
validated resolved public dataclass
    -> dataclass_to_legacy_dict()
    -> complete compatibility patch
    -> scoped update of ptycho.params.cfg
    -> legacy consumer
```

`dataclass_to_legacy_dict()` remains the pure projection boundary.
`update_legacy_dict()` remains the mutation boundary.

The current update skips `None` values, while the pure projected dictionary may
contain them. That distinction preserves previously populated legacy values
and must not change incidentally during validation work.

Supported changes to this bridge must:

- construct and validate the whole patch before mutating `params.cfg`;
- retain exact key mappings and Path/string behavior;
- preserve CONFIG-001 ordering;
- avoid reading `params.cfg` back into new structured configuration; and
- remain independent of Pydantic serialization.

Eliminating legacy reads is a separate strangler migration.

## `DatagenConfig`

`DatagenConfig` remains a six-field compatibility view rather than an
authoritative simulation persistence format.

- New simulation fields are added only to SimulationConfig.
- Conversion to SimulationConfig receives a base for unrepresented values and
  delegates validation to the canonical simulation boundary.
- Reverse conversion fails when representation would be lossy.
- Datagen values do not enter ModelSpec or Torch artifact identity.

## Pydantic Decision

Pydantic persistence is rejected because executable feasibility checks show
that generic adapters and dumps do not preserve:

- exact historical field sets;
- strict scalar/coercion behavior without per-field machinery;
- torch.Tensor schema and JSON behavior;
- complex tensor tagging;
- defensive tensor clone isolation;
- old-schema default and upgrade semantics; or
- exact current wire bytes.

Version-specific Pydantic DTOs may be reconsidered only for a new schema
version that explicitly chooses them and demonstrates net deletion. They must
not reinterpret existing schema identifiers.

## Explicitly Rejected Designs

### `model_dump()` or `dump_json()` as artifact codec

Rejected because tensor and historical-version semantics are not generic
configuration serialization.

### Current dataclass fields as the portable schema

Rejected because a runtime field addition must not silently alter an existing
artifact version.

### MLflow dictionaries as reload identity

Rejected because their tolerance, omission, fallback, and tensor behavior
differs from the portable artifact contract.

### Reverse bridge from `params.cfg`

Rejected because it would create a second mutable configuration authority and
violate CONFIG-001's one-way architecture.

### One serializer for simulation, runtime, and artifacts

Rejected because simulation recipe identity, runtime configuration, model
structural identity, and legacy projection have intentionally different wire
contracts.

## Complexity Budget

- Do not add a generic normalization or serializer layer.
- Do not duplicate frozen schema field lists in a second validation model.
- Prefer reusing a post-decode runtime semantic validator over duplicating
  semantic joins inside each codec.
- Add a new schema version only for a real wire-contract change.
- Keep MLflow compatibility handling isolated from authoritative artifact
  loading.
- Do not broaden artifact tests into gates for unrelated configuration
  changes.

## Acceptance Evidence

Focused persistence evidence must prove, where affected:

1. byte- or value-exact portable-v1 and portable-v2 fixtures;
2. deterministic upgrade results;
3. exact unknown and missing field failures per version;
4. tensor dtype, shape, complex data, and defensive-copy isolation;
5. ModelSpec/config/checkpoint agreement before strict state loading;
6. unchanged TensorFlow and Torch legacy projections, including `None`;
7. unchanged `params.cfg` mutation ordering;
8. MLflow dictionaries remain non-authoritative; and
9. no Pydantic model or serializer appears in persistence paths.
