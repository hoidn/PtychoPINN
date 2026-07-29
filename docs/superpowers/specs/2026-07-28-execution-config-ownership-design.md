# PyTorch Execution Configuration Ownership Design

**Status:** Approved on 2026-07-28

**Parent architecture:** `docs/superpowers/specs/2026-07-28-configuration-boundary-architecture.md`

**Implementation state on `refactor-internal`:** The `ExecutionRequest` and
split ownership/resolution implementation on the public `refactor` branch is
reference evidence only. It is not present here; roadmap Slice 8 Stage A owns
the internal-safe port while preserving the internal execution aliases and
ownership contracts.

## Purpose

Separate runtime request, explicit-input provenance, configuration ownership,
and environment-dependent resolution without converting
`PyTorchExecutionConfig` into a Pydantic domain model.

## Current Problem

`PyTorchExecutionConfig` currently combines:

- user-requested Lightning and DataLoader settings;
- optimizer-adjacent inputs;
- deprecated model-topology aliases;
- explicit-versus-omitted provenance captured in `__new__`;
- structural validation in `__post_init__`; and
- environment-dependent accelerator resolution in `__post_init__`.

This makes the constructor both a data carrier and an effectful resolver.
Pydantic conversion cannot simplify that mixture without changing constructor
timing, hidden provenance, warnings, signatures, or direct programmatic use.

Several fields also have more than one apparent owner. Learning rate,
scheduler, gradient clipping, and accumulation values appear across public
TrainingConfig, Torch TrainingConfig, execution config, CLI helpers, and
Lightning consumers. Deprecated spectral fields already belong to Torch
ModelConfig but remain accepted as execution aliases.

## Decision

Retain the stdlib `PyTorchExecutionConfig` and its manual validation for
compatibility. Move supported workflow architecture toward three explicit
stages:

```text
primitive CLI/programmatic values
    -> execution request + explicit field names
    -> pure structural and ownership resolution
    -> environment capability resolution
    -> resolved execution inputs consumed by Lightning
```

The migration is staged. It does not change direct-constructor behavior until
the external contract explicitly allows that change.

## Ownership

### Torch ModelConfig

Owns model graph and topology:

- spectral bottleneck structure;
- FFNO encoder structure;
- decoder/output representation; and
- all other generator construction fields.

Execution config may accept deprecated topology aliases only at a declared
compatibility boundary. Resolved generators never read topology from execution
config.

### TrainingConfig

Owns optimization semantics:

- optimizer family and optimizer hyperparameters;
- base learning rate;
- scheduler family and scheduler-specific values; and
- the selected gradient-clipping policy where that policy changes parameter
  updates.

During compatibility migration, execution input may remain a higher-precedence
source for a subset of these values, but the factory resolves them one-way into
the effective TrainingConfig. Downstream optimization code reads only the
resolved training owner.

### PyTorchExecutionConfig

Owns runtime and Trainer mechanics:

- accelerator, devices, strategy, precision, and determinism;
- DataLoader workers and memory/prefetch behavior;
- gradient accumulation as a Trainer execution mechanism;
- checkpoint, early stopping, logging, and progress behavior;
- reconstruction logging mechanics; and
- inference batch and evaluation execution controls.

No field on this record is persisted as model structural identity or projected
to `params.cfg`.

## Selected Field Ownership And Precedence

The target effective owners and compatibility handoff are:

| Effective value | Single downstream owner | Execution-config role during migration |
|---|---|---|
| Model topology and spectral fields | Torch ModelConfig | Deprecated input alias only |
| Optimizer, momentum, weight decay, and Adam betas | Torch TrainingConfig | None |
| Learning rate and scheduler family/settings | Torch TrainingConfig | Deprecated priority-2 input alias |
| Gradient clip value and algorithm | Torch TrainingConfig | Deprecated priority-2 input alias; resolved Trainer arguments may mirror the effective training value |
| Gradient accumulation steps | Torch TrainingConfig | Deprecated priority-2 input alias; resolved Trainer arguments may mirror the effective training value |
| Accelerator, devices, strategy, precision, determinism | Resolved execution inputs | Authoritative request |
| Workers, pinning, persistence, and prefetch | Resolved execution inputs | Authoritative request |
| Checkpointing, early stopping, logging, progress, reconstruction logging | Resolved execution inputs | Authoritative request |
| Inference execution batch and evaluation mechanics | Resolved execution inputs | Authoritative request |

The factory resolves each optimizer-adjacent value once using this precedence:

1. an explicitly supplied canonical/factory TrainingConfig override;
2. an explicitly supplied execution compatibility input;
3. an already resolved TrainingConfig value; and
4. the Torch TrainingConfig default.

Defaults inserted by argparse or dataclass construction are not explicit
inputs. `ExecutionRequest.explicit_fields` carries that distinction.

A bare programmatic `PyTorchExecutionConfig` without explicit-field provenance
remains a legacy compatibility input during migration. To preserve its current
factory behavior, all of its optimizer-adjacent values are treated as supplied
at priority 2. New supported CLI and programmatic builders return provenance
and do not use this ambiguous lane.

After resolution:

- optimizer construction reads learning rate, scheduler, clipping policy, and
  accumulation only from the resolved Torch TrainingConfig;
- Lightning Trainer arguments that need clipping or accumulation are derived
  from that same resolved training value and the selected automatic/manual
  optimization mode; and
- execution config is not read again as an independent optimization owner.

Equal canonical and compatibility inputs are accepted once under the canonical
TrainingConfig owner. Different values resolve by the precedence above; this
is not a conflict because priority is explicit. Deprecated topology aliases
retain their stricter rule: equal dual input is accepted once and unequal dual
input fails because graph identity must not depend on source precedence.

## Request And Resolution

### Explicit-input provenance

CLI helpers and supported programmatic factories know which fields were
explicit. They must carry that set alongside the request rather than infer it
from values after defaults are applied.

The target internal value is conceptually:

```python
@dataclass(frozen=True)
class ExecutionRequest:
    values: Mapping[str, object]
    explicit_fields: frozenset[str]
```

`values` is a canonical primitive field mapping, not a constructed
`PyTorchExecutionConfig`. This distinction is required because the compatible
public dataclass constructor still resolves `accelerator="auto"` immediately;
constructing it at request time would observe hardware before pure structural
and ownership resolution. The mapping is copied into an immutable internal
view and checked against the declared execution field names, so it is an
internal resolution envelope rather than a replacement public configuration
schema. It is introduced only if it deletes the hidden
`_explicit_structural_aliases` behavior from supported factory paths. Direct
construction remains compatible while deprecated aliases exist.

### Structural resolution

A pure resolver:

- validates field domains and scalar relationships;
- maps deprecated topology aliases into a candidate Torch ModelConfig patch;
- rejects conflicting dual input;
- maps optimizer-adjacent compatibility inputs into the effective training
  patch;
- records deprecation warnings and consumed provenance; and
- returns candidates without inspecting CUDA or mutating global state.

It does not instantiate Lightning, inspect files, import optional logging
backends, or project `params.cfg`.

### Environment resolution

A separate resolver receives a capability provider and resolves:

- `accelerator="auto"`;
- device counts where `"auto"` is supported;
- CUDA-dependent pin-memory or precision constraints; and
- optional logger availability if that is required before Trainer
  construction.

The provider is injectable so CPU/CUDA behavior is testable without modifying
global hardware state. Its minimum snapshot includes CUDA availability and
CUDA device count. Capability observation is lazy: it occurs only after the
pure scientific, structural, topology, and optimizer candidates are valid and
only when an unresolved runtime value needs it (for example,
`accelerator="auto"` or CUDA `devices="auto"`).

The selected supported-path policy is:

- `devices="auto"` resolves to the available CUDA count for a resolved CUDA
  accelerator and to one device for CPU or MPS;
- `pin_memory=True` resolves to `False` with a recorded notice when the
  accelerator is not CUDA;
- CPU `precision="16-mixed"` resolves to `"bf16-mixed"`, matching Lightning's
  effective CPU precision, while CUDA and MPS retain a supported requested
  precision; and
- requested and resolved accelerator, devices, pin-memory, and precision are
  recorded separately in the runtime audit.

Only after this stage does the supported path instantiate
`PyTorchExecutionConfig`, passing resolved values so its compatibility
constructor does not repeat capability observation.

Compatibility constructors may continue performing current auto-resolution
until the external constructor contract is deliberately revised. Supported
CLI/factory paths should converge on the explicit resolver.

## Selected Contract Alignment

The execution contract is aligned to the supported runtime as follows:

- the request default is `accelerator="auto"`, resolved to CUDA when available
  and otherwise CPU with the existing policy warning;
- TPU is rejected because this runtime has no Torch-XLA support;
- `checkpoint_save_top_k` must be non-negative; `-1` is not a supported
  save-all spelling;
- the effective scheduler domain is owned and validated by Torch
  TrainingConfig, while the execution field is a compatibility input;
- logger input accepts `csv`, `tensorboard`, `mlflow`, or disabled (`None`;
  CLI spelling `none`);
- strategy remains an open Lightning string validated at Trainer
  construction, while the repository explicitly tests supported `auto` and
  DDP paths;
- `persistent_workers=True` requires `num_workers>0`;
- optimizer-adjacent execution fields follow the ownership and precedence
  table above; and
- structural validation must use these selected semantics rather than stale
  annotations or duplicated CLI choice lists.

## Pydantic Decision

Pydantic is rejected for the current execution constructor because:

- raw-argument provenance is meaningful;
- environment resolution is effectful;
- direct construction is a public programmatic boundary;
- constructor exception and warning timing is established behavior; and
- the remaining schema cannot be separated without first resolving ownership.

Pydantic may be reconsidered only after:

1. deprecated topology aliases are removed from execution config;
2. optimizer ownership is singular;
3. explicit-input provenance is an ordinary resolver input;
4. environment resolution is outside construction;
5. the remaining structural checks have one complete accepted-value table; and
6. an adapter deletes meaningful manual validation without introducing a
   parallel request model.

Manual validation remains the accepted result if these gates do not justify an
adapter.

## Compatibility Invariants

- `PyTorchExecutionConfig` remains a stdlib dataclass with its public field
  names and current signature during the compatibility phase.
- It never populates `params.cfg`.
- Equal deprecated topology inputs may resolve once; conflicting dual input
  fails before model construction.
- Resolved model construction reads topology only from Torch ModelConfig.
- Optimizer construction reads only the resolved TrainingConfig owner.
- Environmental resolution is recorded in the payload audit trail without
  rewriting model or artifact identity.
- No execution record becomes part of ModelSpec or versioned Torch artifact
  identity.

## Explicitly Rejected Designs

### Pydantic dataclass with custom internal hooks

Rejected because it would replace transparent manual provenance with
Pydantic-specific constructor internals while environment and ownership
complexity remain.

### One generic configuration precedence engine

Rejected because model, training, execution, CLI, and artifact inputs have
different owners and compatibility rules.

### Persist execution config as model identity

Rejected because devices, workers, logging, and runtime resolution do not
define the model graph or scientific state.

### Immediate public request/resolved type split

Rejected as an unversioned public API break. An internal envelope may support a
staged migration, but direct-constructor behavior changes require their own
contract decision.

## Complexity Budget

An implementation is acceptable only if it:

- removes downstream dual reads when resolving an ownership conflict;
- reduces hidden provenance or constructor branches rather than wrapping them;
- does not add a second public execution schema;
- does not add Pydantic, enum, serialization, or artifact machinery;
- keeps capability checks injectable and outside structural validation; and
- deletes deprecated alias code when its compatibility window closes.

## Acceptance Evidence

Focused evidence must cover:

1. explicit default versus omitted topology alias provenance;
2. positional and keyword compatibility where currently supported;
3. equal dual input, conflicting dual input, and one-way ModelConfig
   resolution;
4. singular downstream optimizer ownership;
5. CPU, CUDA-available, and unsupported-accelerator resolution through an
   injected capability provider;
6. unchanged warnings, public signatures, and current constructor behavior
   during the compatibility phase;
7. DDP devices, strategy, precision, and Trainer arguments;
8. absence of execution values in `params.cfg`, ModelSpec, and artifact
   identity; and
9. exact payload audit records for requested and resolved runtime values.
