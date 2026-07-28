# Representation-Preserving Pydantic Simulation Validation Design

**Status:** Accepted target design

**Accepted:** 2026-07-28

**Supersedes:** The rejected 2026-07-27 Pydantic/string-enum migration

## Purpose

Reduce the hand-written structural and scalar validation at the simulation
configuration boundary without changing the configuration objects, their wire
representation, or unrelated configuration systems.

Pydantic is a boundary validator in this design. It is not the domain model and
does not become a repository-wide configuration substrate.

## Governing Context

This design follows:

- `AGENTS.md` for authority, scope, and evidence rules;
- `docs/CONFIGURATION.md` for configuration ownership and the closed simulation
  mapping boundary;
- the existing public API in `ptycho/config/config.py`;
- the current simulation configuration and CLI contract tests.

The normative architecture vocabulary in `specs/ptychodus_api_spec.md` is
outside this initiative. Any conflict between that specification and a branch
registry requires a separate contract decision.

## Problem

The five frozen simulation records are useful plain dataclasses, but raw
TOML/YAML/JSON mappings currently require hand-written recursion for:

- unknown-key rejection;
- nested dataclass construction;
- string-to-`Path` conversion;
- list-to-tuple conversion;
- strict scalar validation;
- `Literal` membership validation; and
- error-path construction.

Some of this structural/type validation is repeated in
`validate_simulation_config()`, which also owns the genuinely domain-specific
rules.

The repository does not require constructor-time validation. Existing callers
intentionally may construct or `dataclasses.replace()` a recipe and then call
`validate_simulation_config()`. Changing all configuration records into
Pydantic dataclasses would change failure timing and exception behavior without
a governing requirement.

## Decision

Keep these five records as standard frozen dataclasses:

- `ProbeSimulationConfig`
- `SyntheticObjectConfig`
- `ScanSimulationConfig`
- `DetectorSimulationConfig`
- `SimulationConfig`

Create one cached `pydantic.TypeAdapter(SimulationConfig)` and use it in exactly
two places:

1. `simulation_config_from_mapping()` validates and normalizes a raw complete
   mapping into the existing dataclass graph.
2. `validate_simulation_config()` strictly revalidates an existing dataclass
   graph before applying domain-semantic and cross-field rules.

Closed string domains remain `Literal[...]`. Values stored in every dataclass
remain exact built-in strings. No runtime enum representation is introduced.

## Core Invariants

1. **Domain representation stays standard-library-only.**
   `dataclasses.is_dataclass`, `fields`, `asdict`, `replace`, signatures,
   equality, hashing, frozen behavior, and positional construction retain their
   standard dataclass behavior.

2. **Construction timing stays unchanged.**
   Direct construction and `replace()` do not validate. A recipe becomes
   trusted only after `validate_simulation_config()` or construction through
   `simulation_config_from_mapping()`.

3. **Raw parsing and programmatic validation share one structural schema.**
   The cached TypeAdapter is the single implementation of field shapes,
   strict scalar types, nested construction, extras policy, and `Literal`
   membership.

4. **Semantic rules remain explicit domain code.**
   Probe-pipeline grammar, terminal-operation constraints, source/path
   coherence, square geometry, and the relationship between pipeline output
   size and `SimulationConfig.N` remain named Python validation logic.

5. **Pydantic does not cross persistence boundaries.**
   Existing explicit serializers remain authoritative. No `model_dump`,
   Pydantic JSON schema, enum normalizer, or generic artifact serializer is
   introduced.

6. **No automatic broader migration follows.**
   Public model/training/inference configs, Torch configs,
   `PyTorchExecutionConfig`, CLI reflection, ModelSpec, artifacts, Lightning,
   MLflow, checkpoints, and sidecars are outside scope.

## Structural Validation

### Standard dataclass configuration

Each of the five standard dataclasses is decorated with Pydantic's public
`with_config()` helper using:

```python
_SIMULATION_ADAPTER_CONFIG = ConfigDict(
    extra="forbid",
    revalidate_instances="always",
    validate_default=True,
)

@with_config(_SIMULATION_ADAPTER_CONFIG)
@dataclass(frozen=True)
class ProbeSimulationConfig:
    ...
```

`with_config()` supplies TypeAdapter behavior without replacing the standard
dataclass decorator. The shown decorator order is required. Every nested record
receives the configuration because a parent configuration does not implicitly
govern nested standard dataclasses. `validate_default=True` makes an omitted
raw field pass through the same adapter schema as an explicitly supplied value;
it does not add validation to direct stdlib construction.

### Strict private annotation aliases

Private `Annotated` aliases express recurring scalar constraints:

- strict positive integer;
- strict non-negative integer;
- strict integer or `None`;
- strict boolean; and
- strict finite positive number.

Integer aliases have a `BeforeValidator` requiring `type(value) is int`.
Finite-number aliases have a `BeforeValidator` requiring
`type(value) in {int, float}`. This rejects `bool`, `IntEnum`, `Decimal`,
`Fraction`, NumPy scalar objects, and other values that Pydantic could
otherwise normalize into a built-in number while leaving a directly
constructed original object unchanged.

The three closed string fields use private `Annotated[Literal[...], ...]`
aliases with a `BeforeValidator` that requires `type(value) is str`. This
rejects `str` subclasses and string-valued enums before Pydantic can normalize
them into a built-in string. It is necessary because instance validation must
validate the object the caller supplied, not a normalized copy that is then
discarded.

The finite-number alias is a union of strict integer and strict float plus a
finite/positive validator. This preserves whether an accepted value was `4` or
`4.0`; optional probe and detector diameters participate in the canonical
simulation dictionary and digest, so Pydantic must not silently normalize one
wire value into the other.

The dimension-pair alias has a `BeforeValidator` accepting only `list` or
`tuple` containers. Pydantic must not broaden the current boundary by
normalizing sets, deques, generators, or other iterables into tuples.

These aliases are validation implementation details, not a new public domain
type catalog.

### Accepted raw values

| Field shape | Accepted raw input | Stored value | Rejected |
|---|---|---|---|
| closed string | an exact declared `Literal` spelling | exact built-in `str` | aliases, case variants, other strings |
| positive integer | exact built-in integer greater than zero | exact built-in `int` | `bool`, integer subclasses/enums, numeric string, zero, negative, float |
| non-negative integer | exact built-in integer at least zero | exact built-in `int` | `bool`, integer subclasses/enums, numeric string, negative, float |
| optional integer | exact built-in integer or `None` | exact built-in `int` or `None` | `bool`, integer subclasses/enums, numeric string, float |
| finite positive number | exact built-in integer or float greater than zero and finite | the original exact numeric kind | `bool`, numeric subclasses, `Decimal`, `Fraction`, numeric string, zero, negative, NaN, infinity |
| boolean | boolean | `bool` | integers and strings |
| dimension pair | two-element list or tuple of exact positive integers | two-element tuple | other containers, wrong length, booleans, subclasses/enums, strings, floats |
| optional path | path string, `Path`, or `None` at the raw boundary | `Path` or `None` | other values |

Square-pair requirements remain semantic checks after structural validation.

### Mapping mode versus instance mode

The same cached adapter has two deliberate modes:

```python
_SIMULATION_CONFIG_ADAPTER.validate_python(values)
```

Before adapter validation, a small schema-agnostic helper recursively
materializes `collections.abc.Mapping` instances as built-in dictionaries
while preserving list and tuple containers. This retains the existing
`Mapping` API, including `UserDict`-style top-level and nested mappings, because
Pydantic's dataclass adapter otherwise accepts only dictionaries. The helper
does not select fields, reject keys, coerce scalars, or construct dataclasses.

Mapping mode then permits only the documented structural conversions, notably
a path string to `Path` and a two-element list to tuple. Strict scalar aliases
still reject numeric and boolean coercions. The call deliberately omits the
call-level `strict` argument: passing `strict=False` would override field-level
strict types, while `strict=True` would also disable the documented structural
conversions.

```python
_SIMULATION_CONFIG_ADAPTER.validate_python(config, strict=True)
```

Instance mode revalidates every nested field and rejects unnormalized values
inside directly constructed dataclasses. It does not silently validate a copy
while leaving the caller's object malformed.

## Data Flow

```text
TOML / YAML / JSON
        |
        v
format parser -> raw mapping
        |
        v
simulation_config_from_mapping()
        |
        +-- recursively materialize Mapping objects as dictionaries
        |
        +-- TypeAdapter, mapping mode
        |     - reject extras
        |     - validate strict scalars and Literals
        |     - construct nested standard dataclasses
        |     - normalize documented structural forms
        |
        +-- validate_simulation_config()
              - TypeAdapter, strict instance mode
              - semantic and cross-field rules
        |
        v
validated standard SimulationConfig
        |
        +-- simulation_config_to_dict() -> stable canonical mapping
        +-- simulation_config_sha256()  -> stable identity
        +-- dataclass_to_legacy_dict()  -> existing legacy projection
```

Programmatic callers enter at `validate_simulation_config()` before a recipe is
used by simulation, serialization, digesting, or compatibility validation.

## Semantic Validation Ownership

After TypeAdapter validation, `validate_simulation_config()` owns only rules
that are not ordinary field shapes:

- an ideal probe cannot carry `source_path`;
- the transform pipeline must be syntactically valid;
- the boundary-matched operation must be terminal;
- the pipeline's final size must equal `SimulationConfig.N`;
- object and scan dimension pairs must be square; and
- any other relationship between independently valid fields.

Membership checks, scalar type checks, positivity loops, and mapping recursion
must not remain duplicated in that function.

`validate_simulation_compatibility()` remains separate because it compares a
valid simulation recipe with a model configuration.

## Errors

`simulation_config_from_mapping()` and `validate_simulation_config()` remain the
stable domain error facade.

`validate_simulation_config()` retains its explicit top-level `isinstance`
guard, including the existing `TypeError` for a value that is not a
`SimulationConfig`. Pydantic validation errors inside a correctly typed graph
are translated as described below.

Pydantic `ValidationError.errors()` locations are translated to paths beginning
with `simulation`, such as:

```text
simulation.probe.source
simulation.object.image_size.0
simulation.scan.unknown
```

Callers continue to receive `ValueError`-compatible failures. Error messages
must describe the invalid domain field and must not expose a Pydantic model
dump or make Pydantic exception formatting part of the public contract.

Domain validators retain their existing domain-specific messages where tests
or documentation depend on them.

## Serialization And Identity

The following functions remain explicit and authoritative:

- `simulation_config_to_dict()`
- `simulation_config_sha256()`
- `dataclass_to_legacy_dict()`

They do not call a Pydantic serializer. Their field names, tuple/list
decisions, path spelling, numeric kinds, JSON settings, and SHA-256 bytes remain
unchanged.

Exact pre-change simulation fixtures cover:

- the canonical default dictionary and digest;
- optional diameter values represented as both integer and float;
- JSON, YAML, and TOML round trips; and
- the legacy bridge output.

Torch portable-artifact fixtures are independent regression coverage. They do
not become gates for this initiative and do not authorize changes to Torch
configuration or persistence code.

## Dependency Policy

Pydantic is declared as a direct dependency everywhere this repository declares
runtime/test dependencies:

```text
pydantic>=2.12,<3
```

The upper bound is the next major version, not a minor-version lock. This
design uses only public Pydantic APIs:

- `TypeAdapter`
- `ConfigDict`
- `with_config`
- strict scalar types
- `Annotated` validators
- `BeforeValidator`
- `ValidationError.errors()`

No internal schema hooks or `ArgsKwargs` behavior is required.

## Explicitly Rejected Designs

### One enum class per closed string domain

Rejected because Pydantic validates `Literal` directly while preserving
built-in strings. Enums would add dozens of runtime types and force
normalization across persistence boundaries.

### Pydantic dataclasses for all five records

Rejected because constructor-time validation is not required and would change
the established direct-construction and `replace()` workflow.

### Repository-wide Pydantic migration

Rejected because mutable public/Torch configs, execution provenance, CLI
reflection, and artifact persistence have different contracts. Combining them
creates cross-layer migration risk rather than reducing debt.

### TypeAdapter only at raw loading

Rejected because programmatic recipes would retain a separate hand-written
type validator. The accepted design reuses the same adapter in strict instance
mode and leaves only semantic rules in domain code.

## Complexity Budget

The implementation is acceptable only if:

- no `ConfigStrEnum`, enum catalog, enum normalizer, or enum-aware CLI helper is
  added;
- no Torch, artifact, checkpoint, MLflow, Lightning, or architecture-spec file
  changes;
- the schema-aware hand-written nested mapping helpers are deleted;
- at most one small schema-agnostic recursive helper remains solely to
  materialize arbitrary `Mapping` objects as dictionaries;
- structural/type/membership checks are not duplicated after migration;
- production configuration code has a net reduction in validation branches;
- the only production Python module changed is
  `ptycho/config/config.py`; and
- dependency declarations, focused tests, and directly relevant
  configuration documentation are the only other touched surfaces.

Test and fixture lines are evidence and are not counted as production
complexity reduction.

## Acceptance Evidence

Focused evidence must prove:

1. raw nested mappings construct the same standard dataclass graph;
2. unknown keys fail at every simulation level with stable domain paths;
3. every row in the accepted-value table behaves exactly as specified;
4. top-level and nested non-dict `Mapping` implementations retain their
   accepted behavior;
5. direct construction and `replace()` remain non-validating;
6. strict instance validation catches malformed programmatic recipes;
7. all semantic and cross-field rules remain enforced;
8. dataclass reflection, positional construction, frozen behavior, `asdict`,
   equality, hashing, and `replace()` remain compatible;
9. canonical dictionaries, digests, format round trips, and legacy projections
   are exact;
10. every stored closed-domain value has `type(value) is str`; and
11. direct validation rejects string-valued enums, numeric subclasses, and
    other values that would be accepted only through normalization of a copy;
12. dimension pairs reject non-list/tuple iterables; and
13. focused simulation configuration and CLI tests pass.

No repository-wide test sweep, Torch artifact sweep, or implementation review
loop is required by this design.

## Future Changes

Another configuration family may adopt Pydantic only under a separate accepted
design showing:

- which complete raw boundary benefits;
- an always-coherent mutation and validation model;
- exact representation and error behavior;
- concrete production code deletion; and
- no normalization fanout into unrelated consumers.

Similarity to this simulation boundary is not sufficient authority to migrate
another family.
