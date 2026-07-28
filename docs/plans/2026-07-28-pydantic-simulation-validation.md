# Pydantic Simulation Boundary Validation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace hand-written simulation mapping/type validation with one
representation-preserving Pydantic TypeAdapter while retaining standard
dataclasses and exact simulation identity.

**Architecture:** The five frozen simulation records remain ordinary stdlib
dataclasses. One cached `TypeAdapter(SimulationConfig)` validates raw mappings
in conversion mode and revalidates existing dataclass graphs in strict mode;
the existing validator retains only semantic and cross-field rules.

**Tech Stack:** Python 3.10, stdlib dataclasses, Pydantic 2,
pytest, TOML/YAML/JSON.

**Governing design:**
`docs/superpowers/specs/2026-07-28-pydantic-simulation-validation-design.md`

---

## Scope And Stop Rule

This plan ends after the simulation boundary is migrated and its focused
evidence passes.

Do not add enums, a domain catalog, enum normalization, CLI annotation helpers,
Pydantic dataclass decorators, or changes to public/Torch mutable configs,
`PyTorchExecutionConfig`, ModelSpec, artifacts, Lightning, MLflow,
checkpoints, sidecars, or `specs/ptychodus_api_spec.md`.

The already committed Torch artifact fixtures are independent regression
coverage. Do not modify, regenerate, or use them as a completion gate for this
plan.

## File Map

- Modify `ptycho/config/config.py`: strict simulation annotations, standard
  dataclass Pydantic configuration, cached TypeAdapter, domain error
  translation, simplified mapping conversion, and semantic-only validation.
- Modify `pyproject.toml`: declare the direct Pydantic dependency.
- Modify `setup.py`: keep the legacy installation declaration coherent.
- Modify `requirements-ci.txt`: install the direct dependency in CPU CI.
- Modify `tests/test_simulation_config.py`: exact structural/type,
  constructor-timing, dataclass, identity, and legacy-bridge evidence.
- Modify `tests/scripts/test_simulation_config_cli.py`: exact format-loading and
  error-path evidence where needed.
- Modify `docs/CONFIGURATION.md`: explain boundary validation and the explicit
  validation lifecycle.

No other production module is in scope.

## Task 1: Lock The Exact Simulation Contract

**Files:**

- Modify: `tests/test_simulation_config.py`
- Modify: `tests/scripts/test_simulation_config_cli.py`

- [ ] **Step 1: Add raw-value acceptance tests**

Add focused parameterized cases proving:

```python
@pytest.mark.parametrize(
    ("mapping", "path"),
    [
        ({"N": "64"}, "simulation.N"),
        ({"N": True}, "simulation.N"),
        ({"seed": True}, "simulation.seed"),
        ({"object": {"set_phi": 1}}, "simulation.object.set_phi"),
        ({"object": {"set_phi": "true"}}, "simulation.object.set_phi"),
        ({"object": {"kind": "LINES"}}, "simulation.object.kind"),
        ({"scan": {"offset": 1.0}}, "simulation.scan.offset"),
        (
            {"detector": {"photons_per_pattern": float("inf")}},
            "simulation.detector.photons_per_pattern",
        ),
    ],
)
def test_simulation_mapping_rejects_coercive_values(mapping, path):
    with pytest.raises(ValueError, match=re.escape(path)):
        api.simulation_config_from_mapping(mapping)
```

Keep the existing boolean/numeric cases; extend rather than duplicate them.

- [ ] **Step 2: Lock structural normalization**

Prove that raw path strings become `Path`, two-element lists become tuples, all
closed-domain fields store exact built-in strings, and unknown keys fail at
every nested level.

Also construct a programmatic recipe containing both a string-valued enum and
a custom `str` subclass in closed-domain fields. Direct construction remains
permitted, but `validate_simulation_config()` must reject both rather than
validating a normalized copy.

Pass `collections.UserDict` instances at both the top level and a nested
section. They must retain the currently accepted generic `Mapping` behavior.
Pass sets and deques as dimension pairs; both must be rejected rather than
normalized into tuples.

- [ ] **Step 3: Lock numeric-kind identity**

Construct otherwise-identical mappings with:

```python
{"probe": {"mask_diameter": 4}}
{"probe": {"mask_diameter": 4.0}}
{"detector": {"beamstop_diameter": 4}}
{"detector": {"beamstop_diameter": 4.0}}
```

Assert the canonical dictionaries preserve `int` versus `float`. Record and
assert their pre-migration SHA-256 values before implementation.

Also pass `Decimal`, `Fraction`, `IntEnum`, and representative NumPy scalar
values through raw mappings and directly constructed recipes. Validation must
reject them rather than accepting a normalized copy.

- [ ] **Step 4: Lock constructor and dataclass behavior**

Prove:

```python
invalid = replace(SimulationConfig(), N=0)
assert invalid.N == 0
with pytest.raises(ValueError, match="simulation.N"):
    validate_simulation_config(invalid)
```

Also cover `is_dataclass`, `fields`, `asdict`, positional construction, frozen
assignment, equality, hashing, signatures, independent nested defaults, and
`replace()`.

- [ ] **Step 5: Lock exact boundary outputs**

Add exact fixtures/assertions for:

- the default canonical dictionary and SHA-256;
- JSON, YAML, and TOML representations of the same recipe;
- `simulation_config_to_dict()` round-trip; and
- `dataclass_to_legacy_dict()` output for a non-default recipe.

- [ ] **Step 6: Run characterization and RED evidence**

Run:

```bash
pytest -q tests/test_simulation_config.py \
  tests/scripts/test_simulation_config_cli.py
```

Expected before implementation:

- existing characterization cases pass;
- the new strict `set_phi` cases fail because the current mapping boundary
  accepts coercive boolean inputs.

- [ ] **Step 7: Preserve the RED working tree**

Do not commit the deliberately failing strict-boundary cases. Continue directly
to Task 2 so the next commit contains both the falsifying tests and their
minimal implementation and is green.

## Task 2: Install One Representation-Preserving TypeAdapter

**Files:**

- Modify: `ptycho/config/config.py`
- Modify: `pyproject.toml`
- Modify: `setup.py`
- Modify: `requirements-ci.txt`

- [ ] **Step 1: Declare the direct dependency**

Add the same requirement to all three dependency surfaces:

```text
pydantic>=2.12,<3
```

Do not minor-pin Pydantic and do not add `pydantic-settings`.

- [ ] **Step 2: Add private strict annotation aliases**

Use public Pydantic APIs to define private aliases for:

```python
_StrictPositiveInt
_StrictNonNegativeInt
_StrictOptionalInt
_StrictBool
_StrictFinitePositiveNumber
_StrictPositivePair
_ProbeSource
_SyntheticObjectKind
_ScanKind
```

`_StrictFinitePositiveNumber` must use strict integer-or-float branches and an
exact-built-in-number `BeforeValidator` plus an after-validator so `4` remains
`int`, `4.0` remains `float`, booleans, numeric subclasses, `Decimal`,
`Fraction`, NumPy scalars, and numeric strings fail, and
NaN/infinity/non-positive values fail. Integer aliases likewise require
`type(value) is int` before their range constraint.

The pair alias accepts a two-element list or tuple at the mapping boundary,
stores a tuple, and requires exact positive integer elements. A
`BeforeValidator` must reject every other container, including sets and
deques. Square geometry remains an explicit semantic rule in
`validate_simulation_config()`.

The three closed-domain aliases wrap their existing `Literal` values with a
`BeforeValidator` requiring `type(value) is str`. This must reject
string-valued enums and custom `str` subclasses before Pydantic normalization.

- [ ] **Step 3: Configure the five standard dataclasses**

Retain `dataclasses.dataclass(frozen=True)`. Add Pydantic's public
`with_config()` decorator to each of the five simulation records with:

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

Use this decorator order for all five classes. Update only simulation
annotations to use the private strict aliases. Retain all current field names,
order, defaults, `Literal` declarations, and standard dataclass decorators.

- [ ] **Step 4: Create the cached adapter**

After all five classes are defined, create exactly one module-private adapter:

```python
_SIMULATION_CONFIG_ADAPTER = TypeAdapter(SimulationConfig)
```

Do not create per-field adapters or a generic repository configuration helper.

- [ ] **Step 5: Add domain error translation**

Implement one simulation-local helper that translates
`ValidationError.errors()` locations to `simulation.<nested path>` and raises a
`ValueError`-compatible domain failure.

It must preserve field discoverability without exposing serialized Pydantic
models or making Pydantic's display formatting authoritative.

- [ ] **Step 6: Simplify raw mapping construction**

Replace `_reject_unknown_mapping_keys()`, `_pair_from_mapping()`, and
`_section_from_mapping()` with one schema-agnostic mapping materializer and the
adapter:

```python
materialized = _materialize_simulation_mappings(values)
validated = _SIMULATION_CONFIG_ADAPTER.validate_python(materialized)
validate_simulation_config(validated)
return validated
```

The materializer recursively converts arbitrary `Mapping` instances to
built-in dictionaries and recurses through list/tuple values without changing
their container kind. It performs no field selection, unknown-key validation,
scalar coercion, or dataclass construction.

Do not pass `strict=False`: it overrides the field-level strict scalar aliases.
Do not pass `strict=True`: it rejects the documented path-string and
list-to-tuple conversions. Wrap structural validation failures with the domain
error translator. Retain the initial requirement that the top-level input is a
mapping.

- [ ] **Step 7: Simplify explicit validation**

At the start of `validate_simulation_config()`:

```python
if not isinstance(config, SimulationConfig):
    raise TypeError("config must be a SimulationConfig")
_SIMULATION_CONFIG_ADAPTER.validate_python(config, strict=True)
```

Preserve the existing top-level `TypeError`. Translate structural failures
inside a correctly typed graph through the same error helper. Delete duplicated
membership, scalar type, finiteness, positivity, pair-shape, and extras checks.

Retain only semantic/cross-field rules:

- probe source/path coherence;
- transform-pipeline grammar and terminal-operation behavior;
- final probe size versus `SimulationConfig.N`; and
- other relationships not represented by the field annotations.

Do not perform filesystem, CUDA, simulation, or other environmental checks.

- [ ] **Step 8: Run the focused tests**

```bash
pytest -q tests/test_simulation_config.py \
  tests/scripts/test_simulation_config_cli.py
```

Expected: all pass.

- [ ] **Step 9: Confirm the complexity budget**

Run:

```bash
git diff --stat HEAD
git diff -- ptycho/config/config.py
rg -n "ConfigStrEnum|normalize_config_enums|pydantic\\.dataclasses" \
  ptycho ptycho_torch scripts tests
```

Expected:

- the three schema-aware hand-written mapping helpers are gone;
- only the small schema-agnostic Mapping materializer remains;
- structural/type branches have a net reduction;
- none of the rejected enum/Pydantic-dataclass mechanisms were added;
- no production module other than `ptycho/config/config.py` changed.

- [ ] **Step 10: Commit the implementation**

```bash
git add ptycho/config/config.py pyproject.toml setup.py requirements-ci.txt \
  tests/test_simulation_config.py \
  tests/scripts/test_simulation_config_cli.py
git commit -m "refactor(config): validate simulation mappings with Pydantic"
```

## Task 3: Document The Boundary

**Files:**

- Modify: `docs/CONFIGURATION.md`

- [ ] **Step 1: Document validation timing**

In the generated-data configuration section, state:

- simulation records remain standard frozen dataclasses;
- direct construction and `replace()` do not validate;
- programmatic callers call `validate_simulation_config()` before use;
- raw TOML/YAML/JSON mappings go through
  `simulation_config_from_mapping()`; and
- Pydantic performs structural/type validation only at those boundaries.

- [ ] **Step 2: Document strict raw scalar behavior**

Summarize the accepted-value table from the governing design, especially:

- no boolean-as-integer or numeric-string coercion;
- exact `Literal` spellings;
- strict booleans;
- documented path/list structural normalization; and
- preservation of integer-versus-float diameter values in canonical identity.

- [ ] **Step 3: Run focused text checks**

```bash
rg -n "standard.*dataclass|validate_simulation_config|TypeAdapter|coerc" \
  docs/CONFIGURATION.md
rg -n "string enum|ConfigStrEnum|normalize_config_enums" \
  docs/CONFIGURATION.md docs/index.md \
  docs/superpowers/specs/2026-07-28-pydantic-simulation-validation-design.md
```

Expected: the current boundary is discoverable and no rejected enum design is
described as active.

- [ ] **Step 4: Commit the documentation**

```bash
git add docs/CONFIGURATION.md
git commit -m "docs(config): explain simulation boundary validation"
```

## Task 4: Focused Final Verification

**Files:** No new files.

- [ ] **Step 1: Run claim-matched tests**

```bash
pytest -q tests/test_simulation_config.py \
  tests/scripts/test_simulation_config_cli.py
```

Expected: all pass.

- [ ] **Step 2: Verify dependency metadata parses**

```bash
python - <<'PY'
import ast
import pathlib
import tomllib

tomllib.loads(pathlib.Path("pyproject.toml").read_text())
ast.parse(pathlib.Path("setup.py").read_text())
print("dependency metadata parse: PASS")
PY
```

Expected: `dependency metadata parse: PASS`.

- [ ] **Step 3: Verify scope and stale-mechanism absence**

```bash
IMPLEMENTATION_COMMIT="$(
  git log -1 --format=%H \
    --grep='refactor(config): validate simulation mappings with Pydantic'
)"
git diff --name-only "${IMPLEMENTATION_COMMIT}^"..HEAD
rg -n "ConfigStrEnum|normalize_config_enums|AutomaticDeviceCount|DeterminismMode" \
  ptycho ptycho_torch scripts tests
```

The search must return no implementation of the rejected mechanisms. Mentions
inside the governing design and plan are intentional rejection/guard text, not
stale active requirements.

No repository-wide test sweep or Torch artifact sweep is required.

- [ ] **Step 4: Report**

Report:

- exact focused test count and result;
- exact files changed;
- deleted hand-written helpers/branches;
- confirmation that standard dataclass and wire identity stayed exact; and
- confirmation that no broader configuration/persistence migration occurred.
