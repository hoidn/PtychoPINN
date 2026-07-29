# PyTorch Execution Configuration Ownership Design

**Status:** Approved and implemented on `refactor-internal` on 2026-07-28

**Parent architecture:** `docs/superpowers/specs/2026-07-28-configuration-boundary-architecture.md`

**Implementation state on `refactor-internal`:** `ExecutionRequest`, explicit
request provenance, capability-late runtime resolution, and canonical
model/training ownership are implemented. Supported entry points no longer
accept a bare resolved `PyTorchExecutionConfig` as an unresolved request, and
the retired topology/optimizer compatibility lanes are absent.

## Purpose

Separate runtime request, explicit-input provenance, configuration ownership,
and environment-dependent resolution without converting
`PyTorchExecutionConfig` into a Pydantic domain model.

## Problem addressed

Before this design, `PyTorchExecutionConfig` combined:

- user-requested Lightning and DataLoader settings;
- optimizer-adjacent inputs;
- deprecated model-topology aliases;
- explicit-versus-omitted provenance captured in `__new__`;
- structural validation in `__post_init__`; and
- environment-dependent accelerator resolution in `__post_init__`.

That made the constructor both a data carrier and an effectful resolver.
Pydantic conversion cannot simplify that mixture without changing constructor
timing, hidden provenance, warnings, signatures, or direct programmatic use.

Several fields also had more than one apparent owner. Learning rate,
scheduler, gradient clipping, and accumulation values appear across public
TrainingConfig, Torch TrainingConfig, execution config, CLI helpers, and
Lightning consumers. Deprecated spectral fields already belonged to Torch
ModelConfig but remained accepted as execution aliases.

## Decision

Retain the stdlib `PyTorchExecutionConfig` as a pure, manually validated
resolved runtime carrier. Supported workflows implement three explicit stages:

```text
primitive CLI/programmatic values
    -> execution request + explicit field names
    -> pure structural and ownership resolution
    -> environment capability resolution
    -> resolved execution inputs consumed by Lightning
```

`ExecutionRequest` owns unresolved values and explicit-input provenance.
`PyTorchExecutionConfig` accepts resolved runtime values only and performs no
hardware observation.

## Ownership

### Torch ModelConfig

Owns model graph and topology:

- spectral bottleneck structure;
- FFNO encoder structure;
- decoder/output representation; and
- all other generator construction fields.

Execution configuration accepts no topology aliases. Resolved generators read
topology only from Torch ModelConfig.

### TrainingConfig

Owns optimization semantics:

- optimizer family and optimizer hyperparameters;
- base learning rate;
- scheduler family and scheduler-specific values; and
- the selected gradient-clipping policy where that policy changes parameter
  updates.

Execution input is not a source for these values. Downstream optimization code
reads only the resolved training owner.

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

## Selected Field Ownership

The implemented effective owners are:

| Effective value | Single downstream owner | Execution role |
|---|---|---|
| Model topology and spectral fields | Torch ModelConfig | None |
| Optimizer, momentum, weight decay, and Adam betas | Torch TrainingConfig | None |
| Learning rate and scheduler family/settings | Torch TrainingConfig | None |
| Gradient clip value and algorithm | Torch TrainingConfig | Resolved Trainer arguments mirror the effective training value |
| Gradient accumulation steps | Torch TrainingConfig | Resolved Trainer arguments mirror the effective training value |
| Accelerator, devices, strategy, precision, determinism | Resolved execution inputs | Authoritative request |
| Workers, pinning, persistence, and prefetch | Resolved execution inputs | Authoritative request |
| Checkpointing, early stopping, logging, progress, reconstruction logging | Resolved execution inputs | Authoritative request |
| Inference execution batch and evaluation mechanics | Resolved execution inputs | Authoritative request |

Factories resolve optimizer-adjacent values from canonical/factory
TrainingConfig overrides and the resolved TrainingConfig baseline/default.
`ExecutionRequest.explicit_fields` records presence only for execution-owned
runtime fields. A bare `PyTorchExecutionConfig` is rejected as factory input
because it is already a resolved output carrier.

After resolution:

- optimizer construction reads learning rate, scheduler, clipping policy, and
  accumulation only from the resolved Torch TrainingConfig;
- Lightning Trainer arguments that need clipping or accumulation are derived
  from that same resolved training value and the selected automatic/manual
  optimization mode; and
- execution config is not read again as an independent optimization owner.

## Request And Resolution

### Explicit-input provenance

CLI helpers and supported programmatic factories know which fields were
explicit. They must carry that set alongside the request rather than infer it
from values after defaults are applied.

The implemented request value is:

```python
@dataclass(frozen=True)
class ExecutionRequest:
    values: Mapping[str, object]
    explicit_fields: frozenset[str]
```

`values` is a canonical primitive field mapping, not a constructed
`PyTorchExecutionConfig`. This keeps unresolved `accelerator="auto"` and
presence provenance outside the resolved carrier. The mapping is copied into
an immutable view and checked against the declared execution field names.

### Structural resolution

A pure resolver:

- validates field domains and scalar relationships;
- records consumed provenance and deferred resolution notices; and
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
`PyTorchExecutionConfig` from resolved values. Its constructor validates but
never performs capability observation.

## Selected Contract Alignment

The execution contract is aligned to the supported runtime as follows:

- the request default is `accelerator="auto"`, resolved to CUDA when available
  and otherwise CPU with the existing policy warning;
- TPU is rejected because this runtime has no Torch-XLA support;
- `checkpoint_save_top_k` must be non-negative; `-1` is not a supported
  save-all spelling;
- the effective scheduler domain is owned and validated by Torch
  TrainingConfig and is absent from execution configuration;
- logger input accepts `csv`, `tensorboard`, `mlflow`, or disabled (`None`;
  CLI spelling `none`);
- strategy remains an open Lightning string validated at Trainer
  construction, while the repository explicitly tests supported `auto` and
  DDP paths;
- `persistent_workers=True` requires `num_workers>0`;
- optimizer-adjacent fields follow the single-owner table above; and
- structural validation must use these selected semantics rather than stale
  annotations or duplicated CLI choice lists.

## Pydantic Decision

Pydantic is rejected for the current execution constructor because:

- raw-argument provenance is meaningful;
- environment resolution is effectful;
- direct construction is a public programmatic boundary;
- constructor exception and warning timing is established behavior; and
- the remaining schema cannot be separated without first resolving ownership.

Topology aliases are removed, optimizer ownership is singular,
explicit-input provenance is an ordinary request input, and environment
resolution is outside construction. Manual validation remains the accepted
result because the remaining request/environment lifecycle is not simplified
by a snapshot adapter.

## Implementation Invariants

- `PyTorchExecutionConfig` remains a stdlib dataclass containing resolved
  runtime fields only.
- It never populates `params.cfg`.
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

### Treating a resolved carrier as an unresolved request

Rejected because it loses explicit-input provenance and makes capability
resolution timing ambiguous. Supported factories accept `ExecutionRequest` or
`None`; they return `PyTorchExecutionConfig` only after environment resolution.

## Complexity Budget

An implementation is acceptable only if it:

- removes downstream dual reads when resolving an ownership conflict;
- reduces hidden provenance or constructor branches rather than wrapping them;
- does not add a second public execution schema;
- does not add Pydantic, enum, serialization, or artifact machinery;
- keeps capability checks injectable and outside structural validation; and
- contains no retired topology or optimizer alias machinery.

## Acceptance Evidence

Focused evidence must cover:

1. explicit versus omitted runtime-field provenance;
2. resolved-carrier dataclass signature and validation;
3. rejection of topology or optimizer fields at the execution boundary;
4. singular downstream optimizer ownership;
5. CPU, CUDA-available, and unsupported-accelerator resolution through an
   injected capability provider;
6. stable warnings, request provenance, and resolved-carrier behavior;
7. DDP devices, strategy, precision, and Trainer arguments;
8. absence of execution values in `params.cfg`, ModelSpec, and artifact
   identity; and
9. exact payload audit records for requested and resolved runtime values.
