# PyTorch Execution Configuration Ownership Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give optimizer-adjacent values one effective Torch `TrainingConfig` owner while preserving `PyTorchExecutionConfig` construction compatibility and making explicit CLI provenance available to the factory.

**Architecture:** Add a small internal `ExecutionRequest` envelope containing canonical primitive values and explicit fields, plus a new CLI builder that creates that request without constructing `PyTorchExecutionConfig`. The factory first resolves scientific, structural, topology, and optimizer ownership purely; only then does an injectable capability resolver select accelerator/devices and CUDA-dependent runtime values and construct the compatible public execution dataclass. The existing config-returning helper uses the same stages immediately and preserves its warnings. A separate priority-3 training-baseline channel prevents public resolved values from masquerading as explicit factory overrides. Downstream optimizer, scheduler, clipping, and accumulation behavior reads only the resolved Torch `TrainingConfig`. No public execution schema, Pydantic, or persistence change is introduced.

**Tech Stack:** Python stdlib dataclasses, argparse, PyTorch Lightning workflow helpers, pytest.

**Prerequisite and execution constraint:** Execute
`docs/plans/2026-07-28-public-config-resolution-implementation.md` first so the
unified public parser and layered validation boundary are stable. Work in the
current checkout. Do not create a worktree; the repository `AGENTS.md`
prohibition overrides generic execution skill advice. Immediately before Task
1, record the current commit as the implementation-start SHA for Task 7.

---

## File Map

- Create `ptycho_torch/execution_request.py`: internal request/provenance value,
  capability snapshot, and pure environment-resolution helpers.
- Modify `ptycho_torch/cli/shared.py`: build explicit execution requests while retaining the public compatibility helper.
- Modify `ptycho_torch/config_factory.py`: accept and normalize request/config inputs and apply the selected ownership precedence.
- Modify `ptycho_torch/workflows/components.py`: consume resolved TrainingConfig optimization values for model and Trainer behavior.
- Modify `ptycho/config/config.py`: enforce the selected remaining execution structural contract manually.
- Modify the native and unified training/inference CLI entry points to build
  provenance-aware requests; the unified inference path uses the shared
  phase-aware resolution boundary because it does not own a Torch factory
  payload.
- Modify focused CLI and factory/workflow tests only.

### Task 1: Characterize the selected execution contract

**Files:**

- Modify: `tests/torch/test_cli_shared.py`
- Modify: `tests/torch/test_execution_config_defaults.py`
- Modify: `tests/torch/test_train_lightning_execution_contract.py`

- [ ] **Step 1: Write failing structural-contract tests**

Add focused cases proving:

```python
def test_execution_config_rejects_persistent_workers_without_workers():
    with pytest.raises(ValueError, match="persistent_workers"):
        PyTorchExecutionConfig(
            accelerator="cpu",
            num_workers=0,
            persistent_workers=True,
        )


def test_execution_config_rejects_unknown_logger_backend():
    with pytest.raises(ValueError, match="logger_backend"):
        PyTorchExecutionConfig(accelerator="cpu", logger_backend="bogus")


def test_execution_config_explicit_none_disables_logger():
    config = PyTorchExecutionConfig(accelerator="cpu", logger_backend=None)
    assert config.logger_backend is None
```

Retain the existing TPU rejection, non-negative checkpoint count, accelerator
default/resolution, signature, and deprecated structural-provenance tests.
Add actual compatibility evidence rather than relying on a comment: freeze the
ordered `inspect.signature(PyTorchExecutionConfig)` parameter names and prove
that a positional prefix such as
`PyTorchExecutionConfig("cpu", 2, "ddp", False)` still binds to
`accelerator`, `devices`, `strategy`, and `deterministic`. Also retain a
keyword-construction case. These tests protect the public direct-constructor
surface while supported request paths change internally.
Characterize its interleaved warning order as well: with CUDA unavailable,
`PyTorchExecutionConfig(accelerator="auto", learning_rate=0)` emits the
existing POLICY-001 CPU fallback warning and then raises the late
learning-rate `ValueError`. This exact test prevents Task 4's shared validator
extraction from moving all checks ahead of constructor auto-resolution.
Replace the stale downstream-normalization expectation
`test_lightning_data_module_normalizes_zero_worker_settings`: under the
approved contract, `persistent_workers=True, num_workers=0` is invalid at the
execution boundary and never reaches a DataLoader. Keep a separate valid
`num_workers>0` test for forwarding.


- [ ] **Step 2: Run the focused tests and verify RED**

Run:

```bash
python -m pytest \
  tests/torch/test_execution_config_defaults.py \
  tests/torch/test_train_lightning_execution_contract.py \
  -k "persistent_workers or logger_backend" -q
```

Expected: the new persistent-worker and logger-domain cases fail because
`PyTorchExecutionConfig.__post_init__` does not yet enforce them.

- [ ] **Step 3: Implement the minimal manual checks**

In `PyTorchExecutionConfig.__post_init__`, add only:

```python
if self.persistent_workers and self.num_workers <= 0:
    raise ValueError("persistent_workers=True requires num_workers > 0")

if self.logger_backend not in {"csv", "tensorboard", "mlflow", None}:
    raise ValueError(
        "logger_backend must be 'csv', 'tensorboard', 'mlflow', or None"
    )
```

Do not add Pydantic, enums, filesystem checks, logger imports, or assignment
validation. Task 4 validates optimizer-owned domains on the effective Torch
TrainingConfig; do not make the execution-config annotation the scheduler
authority.

- [ ] **Step 4: Run focused execution tests and verify GREEN**

Run:

```bash
python -m pytest \
  tests/torch/test_execution_config_defaults.py \
  tests/torch/test_train_lightning_execution_contract.py \
  -q
```

Expected: all selected execution contract tests pass.

- [ ] **Step 5: Commit**

```bash
git add ptycho/config/config.py \
  tests/torch/test_execution_config_defaults.py \
  tests/torch/test_train_lightning_execution_contract.py
git commit -m "fix(config): enforce execution runtime contract"
```

### Task 2: Add an internal explicit-input request envelope

**Files:**

- Create: `ptycho_torch/execution_request.py`
- Modify: `tests/torch/test_cli_shared.py`

- [ ] **Step 1: Write failing request-value tests**

Add tests for a frozen internal primitive request:

```python
def test_execution_request_records_only_explicit_fields():
    request = ExecutionRequest(
        values={"accelerator": "auto", "devices": "auto"},
        explicit_fields=frozenset({"accelerator"}),
        notices=(ResolutionNotice(DeprecationWarning, "--device is deprecated"),),
    )
    assert request.as_dict() == {
        "accelerator": "auto",
        "devices": "auto",
    }
    assert request.explicit_fields == frozenset({"accelerator"})


def test_execution_request_rejects_unknown_explicit_field():
    with pytest.raises(ValueError, match="unknown explicit execution field"):
        ExecutionRequest(
            values={"accelerator": "cpu"},
            explicit_fields=frozenset({"bogus"}),
        )
```

Also prove the request defensively copies its input mapping, rejects unknown
value keys, rejects explicit fields absent from the request mapping, and does
not call `PyTorchExecutionConfig` or inspect Torch/CUDA. Prove an explicitly
supplied topology value equal to its dataclass default remains in
`explicit_fields`, while the same default inserted for an omitted option does
not. That distinction is acceptance evidence, not value comparison.

- [ ] **Step 2: Run the tests and verify RED**

Run:

```bash
python -m pytest tests/torch/test_cli_shared.py \
  -k "execution_request" -q
```

Expected: import or name failure because `ExecutionRequest` does not exist.

- [ ] **Step 3: Implement the internal record**

Create:

```python
from dataclasses import dataclass, fields
from types import MappingProxyType
from typing import Any, Mapping

from ptycho.config.config import PyTorchExecutionConfig


@dataclass(frozen=True)
class ExecutionCapabilities:
    cuda_available: bool
    cuda_device_count: int


@dataclass(frozen=True)
class EnvironmentResolution:
    requested: Mapping[str, Any]
    resolved: Mapping[str, Any]
    capabilities: ExecutionCapabilities | None


@dataclass(frozen=True)
class ResolutionNotice:
    category: type[Warning]
    message: str


@dataclass(frozen=True)
class ExecutionRequest:
    values: Mapping[str, Any]
    explicit_fields: frozenset[str]
    notices: tuple[ResolutionNotice, ...] = ()

    def __post_init__(self) -> None:
        known = {item.name for item in fields(PyTorchExecutionConfig)}
        copied = dict(self.values)
        unknown = set(copied) - known
        if unknown:
            raise ValueError(
                "unknown execution request field(s): "
                + ", ".join(sorted(unknown))
            )
        unknown_explicit = self.explicit_fields - known
        if unknown_explicit:
            raise ValueError(
                "unknown explicit execution field(s): "
                + ", ".join(sorted(unknown_explicit))
            )
        absent_explicit = self.explicit_fields - set(copied)
        if absent_explicit:
            raise ValueError(...)
        object.__setattr__(self, "values", MappingProxyType(copied))

    def as_dict(self) -> dict[str, Any]:
        return dict(self.values)
```

The exact internal immutable representation may differ if it has the same
behavior. It is provenance plus canonical primitive values, not a second
configuration schema. It deliberately cannot contain a constructed
`PyTorchExecutionConfig`, because that would resolve hardware before the pure
factory stages.

- [ ] **Step 4: Run the request tests and verify GREEN**

Run:

```bash
python -m pytest tests/torch/test_cli_shared.py \
  -k "execution_request" -q
```

Expected: all request-value tests pass.

- [ ] **Step 5: Commit**

```bash
git add ptycho_torch/execution_request.py tests/torch/test_cli_shared.py
git commit -m "refactor(config): represent explicit execution requests"
```

### Task 3: Build provenance-aware CLI requests without breaking the helper

**Files:**

- Modify: `ptycho_torch/cli/shared.py`
- Modify: `tests/torch/test_cli_shared.py`

- [ ] **Step 1: Write failing CLI request tests**

Add tests for a new function:

```python
request = build_execution_request_from_args(
    args,
    mode="training",
    explicit_options={"--learning-rate", "--accelerator"},
)
assert request.values["learning_rate"] == 2e-4
assert request.values["accelerator"] == "auto"
assert request.explicit_fields == frozenset(
    {"learning_rate", "accelerator"}
)
```

Also prove raw CLI `logger_backend="none"` resolves to Python `None` and that
an omitted logger uses `"csv"`. Prove request construction does not call
`torch.cuda.is_available()`, `torch.cuda.device_count()`, or
`PyTorchExecutionConfig`, even for `accelerator="auto"`.

Characterize compatibility-helper warning timing with an otherwise invalid
Namespace: deprecated-device, deprecated disable-MLflow, and
deterministic-worker warnings are emitted before
`PyTorchExecutionConfig` raises on the invalid field. This is existing
observable behavior and must not be deferred on the config-returning helper.

Add exact option-to-field suppliedness tests for the complete supported
execution surface. Encode one shared declarative registry with these
one-to-one and one-to-many bindings:

| Option spelling(s) | Canonical explicit field(s) |
|---|---|
| `--accelerator`, `--torch-accelerator`, deprecated `--device` | `accelerator` |
| `--deterministic`, `--no-deterministic`, `--torch-deterministic` | `deterministic` |
| `--num-workers`, `--torch-num-workers` | `num_workers` |
| `--learning-rate`, `--torch-learning-rate` | `learning_rate` |
| native `--scheduler`; unified `--torch-scheduler` | `scheduler` |
| `--accumulate-grad-batches`, `--torch-accumulate-grad-batches` | `accum_steps` |
| `--logger`, `--torch-logger` | `logger_backend` |
| `--quiet` | `enable_progress_bar` |
| deprecated `--disable_mlflow` | `logger_backend`, `enable_progress_bar` |
| `--enable-checkpointing`, `--disable-checkpointing`, `--torch-enable-checkpointing` | `enable_checkpointing` |
| `--checkpoint-save-top-k`, `--torch-checkpoint-save-top-k` | `checkpoint_save_top_k` |
| `--checkpoint-monitor` | `checkpoint_monitor_metric` |
| `--checkpoint-mode` | `checkpoint_mode` |
| `--early-stop-patience` | `early_stop_patience` |
| `--torch-recon-log-every-n-epochs` | `recon_log_every_n_epochs` |
| `--torch-recon-log-num-patches` | `recon_log_num_patches` |
| `--torch-recon-log-fixed-indices` | `recon_log_fixed_indices` |
| `--torch-recon-log-stitch` | `recon_log_stitch` |
| `--torch-recon-log-max-stitch-samples` | `recon_log_max_stitch_samples` |
| `--inference-batch-size`, `--torch-inference-batch-size` | `inference_batch_size` |

Both `--flag value` and `--flag=value`, plus paired boolean spellings, must map
to canonical execution fields. Argparse destination names are never passed
untranslated to `ExecutionRequest`. The fan-out for deprecated
`--disable_mlflow` is explicit: its historical value sets
`logger_backend=None` and `enable_progress_bar=False`, and both fields are
marked explicit.

Each registry binding declares the supported entry point/lane in which the
spelling is execution-owned. In particular, native
`ptycho_torch.train --scheduler` is the deprecated execution-compatibility
input, whereas unified `scripts/training/train.py --scheduler` is a canonical
public TrainingConfig option and is excluded from that entry point's execution
request; only its `--torch-scheduler` spelling enters the compatibility
request. The separate Task 6 public-optimizer raw-option map captures the
canonical spelling.

Prove omitted presentation defaults do not become explicit compatibility
inputs: an omitted deprecated `--device` default must not defeat
`accelerator="auto"`, and an omitted wrapper `learning_rate=None` must select
the execution/TrainingConfig default rather than fail validation.

- [ ] **Step 2: Run the CLI tests and verify RED**

Run:

```bash
python -m pytest tests/torch/test_cli_shared.py \
  -k "execution_request_from_args or logger_backend" -q
```

Expected: the new builder is absent.

- [ ] **Step 3: Extract one construction implementation**

Implement:

```python
def build_execution_request_from_args(
    args,
    mode="training",
    *,
    explicit_options=(),
):
    explicit_fields, explicit_sources = canonicalize_execution_options(
        explicit_options
    )
    values, notices = _normalize_execution_namespace(
        args,
        mode=mode,
        explicit_sources=explicit_sources,
    )
    return ExecutionRequest(
        values=values,
        explicit_fields=frozenset(explicit_fields),
        notices=notices,
    )


def build_execution_config_from_args(args, mode="training"):
    # Share canonical field mapping, but preserve the historical effect order:
    # device resolution/deprecation, logger/performance warnings, then public
    # dataclass construction and validation.
    values = _normalize_compatibility_namespace_with_immediate_notices(
        args, mode=mode
    )
    return PyTorchExecutionConfig(**values)
```

Move Namespace-to-canonical-value normalization into the private helper rather
than duplicating it. The request-returning builder must stop before
environment resolution and public dataclass construction. Accumulate
device/logger/performance deprecations as `ResolutionNotice` values. Existing
callers of the config-returning helper continue receiving
`PyTorchExecutionConfig`, treat the supplied Namespace with historical
semantics, perform their existing immediate auto-resolution, and see
deprecated-device, disable-MLflow, deterministic-worker, and CPU-fallback
warnings before any later constructor validation failure exactly as today.
Task 4 moves supported request/factory paths through the new pure and
environment stages; it does not change this compatibility surface.

The option registry is shared by the four supported entry points. It handles
`--name=value`; it does not infer suppliedness by comparing values to defaults.
When an argparse Namespace contains `None` or a presentation default for an
omitted option, normalization uses the selected execution default without
marking it explicit. Retain source-option provenance long enough to
distinguish deprecated `--device` from canonical `--accelerator`.

- [ ] **Step 4: Run all focused CLI helper tests and verify GREEN**

Run:

```bash
python -m pytest tests/torch/test_cli_shared.py -q
```

Expected: existing helper behavior and new request behavior pass.

- [ ] **Step 5: Commit**

```bash
git add ptycho_torch/cli/shared.py tests/torch/test_cli_shared.py
git commit -m "refactor(cli): preserve execution input provenance"
```

### Task 4: Resolve optimizer-adjacent compatibility inputs once

**Files:**

- Modify: `ptycho_torch/config_factory.py`
- Modify: `ptycho_torch/execution_request.py`
- Modify: `ptycho_torch/workflows/components.py`
- Modify: `ptycho/config/config.py`
- Modify: `ptycho/workflows/backend_selector.py`
- Modify: `tests/torch/test_config_factory.py`
- Modify: `tests/torch/test_structural_config_ownership.py`
- Modify: `tests/torch/test_workflows_components.py`
- Modify: `tests/torch/test_execution_config_defaults.py`

- [ ] **Step 1: Write failing precedence and ownership tests**

Cover:

```python
def test_canonical_training_override_beats_execution_compatibility_input(...):
    request = ExecutionRequest(
        values={
            "accelerator": "cpu",
            "learning_rate": 2e-4,
            "scheduler": "Exponential",
            "gradient_clip_val": 5.0,
            "accum_steps": 4,
        },
        explicit_fields=frozenset({
            "learning_rate",
            "scheduler",
            "gradient_clip_val",
            "accum_steps",
        }),
    )
    payload = create_training_payload(
        ...,
        overrides={
            "n_groups": 4,
            "learning_rate": 3e-4,
        },
        training_baseline=PTTrainingConfig(scheduler="WarmupCosine"),
        execution_config=request,
    )
    assert payload.pt_training_config.learning_rate == 3e-4
    assert payload.pt_training_config.scheduler == "Exponential"
    assert payload.pt_training_config.gradient_clip_val == 5.0
    assert payload.pt_training_config.accum_steps == 4
```

Also prove:

- an execution value not listed in `explicit_fields` does not override the
  training value;
- a bare legacy `PyTorchExecutionConfig` treats its optimizer-adjacent fields
  as supplied at priority 2;
- equal optimizer values are recorded once under TrainingConfig ownership;
- the workflow does not promote execution learning-rate/clipping into the
  explicit canonical patch;
- the complete public scheduler value is a resolved baseline (priority 3),
  while a caller's explicit `overrides["scheduler"]` is canonical priority 1;
- in the unified training CLI, raw suppliedness of canonical public optimizer
  options is threaded into that explicit factory override channel. If
  `--scheduler WarmupCosine` and compatibility
  `--torch-scheduler Exponential` are both explicit, the canonical
  `WarmupCosine` value wins at priority 1; the resolved public record alone
  remains only priority 3;
- `create_training_payload(..., training_baseline=...)` is a distinct typed
  keyword-only `TFTrainingConfig | PTTrainingConfig | None` channel from
  `overrides`; the resolver never guesses which mapping entries came from the
  public resolved record;
- topology alias conflicts retain fail-closed behavior;
- a request whose topology alias equals the dataclass default is consumed and
  warned only when that field is explicit; the same normalized default with
  the field omitted from `explicit_fields` is not consumed;
- the topology resolver leaves both input mappings unchanged and emits no
  warning before a successful payload exists;
- explicit `ffno_encoder_blocks` and `ffno_encoder_modes` map to canonical
  `fno_blocks` and `fno_modes`;
- the four execution FFNO fields with no current ModelConfig owner
  (`ffno_encoder_share_weights`, `ffno_encoder_gate_init`,
  `ffno_encoder_norm`, and `ffno_encoder_mlp_ratio`) fail explicitly in a
  supported request instead of remaining silently dead;
- inference accepts and normalizes `ExecutionRequest`, retains no optimizer
  handoff, audits requested/resolved runtime values, and rejects explicitly
  supplied training-only optimizer compatibility fields instead of ignoring
  them;
- an invalid factory patch with `execution_config=None` does not construct the
  environment-dependent default or emit the CPU fallback warning; and
- audit contains one canonical TrainingConfig value plus source provenance,
  along with exact requested/resolved accelerator, devices, pin-memory, and
  precision provenance.

Add injected-capability cases proving:

- `accelerator="auto", devices="auto"` resolves to CUDA plus the injected
  positive CUDA device count when available;
- the same request resolves to CPU plus one device and records the existing
  POLICY-001 notice when CUDA is unavailable;
- CPU `devices="auto"` resolves to one, CPU `pin_memory=True` resolves to
  `False` with a notice, and CPU `"16-mixed"` resolves to `"bf16-mixed"`;
- explicit CPU and explicit CUDA with a concrete device count do not call the
  default capability observer;
- TPU and structurally invalid values fail before capability observation; and
- the final `PyTorchExecutionConfig` is constructed only with a resolved
  accelerator, so its compatibility constructor does not perform a second
  hardware lookup.

Add effective-owner domain tests for the Torch TrainingConfig scheduler set
(`Default`, `Exponential`, `MultiStage`, `Adaptive`, `WarmupCosine`,
`ReduceLROnPlateau`), clip algorithms (`norm`, `value`, `agc`), positive
learning rate/accumulation, and finite non-negative clip magnitude.

- [ ] **Step 2: Run factory ownership tests and verify RED**

Run:

```bash
python -m pytest \
  tests/torch/test_config_factory.py \
  tests/torch/test_structural_config_ownership.py \
  tests/torch/test_workflows_components.py \
  tests/torch/test_execution_config_defaults.py \
  -k "execution and (precedence or ownership or provenance or topology or inference or domain or deferred)" -q
```

Expected: request objects are not accepted and effective optimizer values do
not follow the selected precedence.

- [ ] **Step 3: Implement request normalization and precedence**

Add shared pure helpers in `execution_request.py`:

```python
def normalize_execution_input(value, *, mode):
    if value is None:
        return None  # materialize the primitive default request only later
    if isinstance(value, ExecutionRequest):
        return NormalizedExecutionInput.from_request(value)
    if isinstance(value, PyTorchExecutionConfig):
        return NormalizedExecutionInput.from_legacy_config(
            value,
            mode=mode,
        )
    raise TypeError(...)
```

`NormalizedExecutionInput` is an internal resolver result containing copied
canonical values, explicit fields, deferred notices, and (only for the bare
legacy lane) the already-constructed compatibility config. It is not exported
as a public config schema. In training, a bare legacy config treats all
optimizer-adjacent values as priority-2 inputs and retains
`_explicit_structural_aliases`. In inference, the ambiguous optimizer defaults
on a bare config remain ignored for compatibility, but actually captured
structural aliases remain explicit and fail as phase-inappropriate.

Before topology, optimizer, or environment resolution, run the complete pure
structural checks over the normalized primitive values. Extract the rules
already owned by `PyTorchExecutionConfig.__post_init__` into shared private
mapping-based validation stages in `ptycho/config/config.py`, so rules are not
duplicated while timing remains compatible:

- the compatible constructor keeps its exact sequence—pre-environment
  accelerator/devices/precision checks, current auto-resolution and warning,
  then the remaining scalar/relationship checks; and
- the supported request path invokes both pure validation stages before
  capability observation.

Thus a direct `PyTorchExecutionConfig(accelerator="auto", learning_rate=0)`
retains its existing auto-resolution/warning-before-late-error behavior, while
the new request/factory path rejects the same invalid primitive request before
hardware observation. Unsupported accelerators/TPU, logger domain, worker
relationships, numeric ranges, checkpoint mode/count, precision spelling, and
deprecated-alias scalar domains have one implementation.

Add `resolve_optimizer_ownership()` that receives:

- the separate resolved `TFTrainingConfig | PTTrainingConfig` owner at
  priority 3;
- the normalized execution request at priority 2; and
- the explicit canonical factory patch at priority 1.

It constructs a fresh Torch TrainingConfig, validates the effective domains,
and returns deterministic provenance. Before constructing it:

```python
effective_training_patch = dict(canonical_training_patch)
for name in OPTIMIZER_EXECUTION_COMPAT_FIELDS:
    if (
        name in explicit_execution_fields
        and name not in effective_training_patch
    ):
        effective_training_patch[name] = normalized_execution.values[name]
        provenance[name] = "execution_compatibility"
```

Canonical `overrides` win. Store the effective value in Torch TrainingConfig
and record only that canonical owner in `overrides_applied`, with separate
source provenance.

In `_train_with_lightning()`, stop copying
`execution_config.learning_rate`/`gradient_clip_val` into
`factory_overrides`. Pass the complete resolved public `TrainingConfig` object
through the new typed keyword-only factory parameter. A declared
owner-specific extractor reads only optimizer fields that exist on that
dataclass; a Torch TrainingConfig baseline uses the same explicit owner list.
Only values actually present in the explicit caller `overrides` mapping are
priority-1 canonical patch values. Do not convert the baseline to another
untyped patch at the call site or let a public dataclass default masquerade as
explicit.

Preserve all existing positional factory parameters and add only keyword-only
channels:

```python
def create_training_payload(
    train_data_file,
    output_dir,
    overrides=None,
    execution_config=None,
    profile=None,
    *,
    training_baseline: TFTrainingConfig | PTTrainingConfig | None = None,
    execution_capabilities=None,
): ...
```

`create_inference_payload()` gains only the analogous keyword-only
`execution_capabilities` test seam.

Use `ExecutionRequest.explicit_fields` as the authority on supported paths.
For a bare compatibility config only, retain `_explicit_structural_aliases`.
Extend that legacy capture set to all six FFNO names. Map
`ffno_encoder_blocks -> fno_blocks` and
`ffno_encoder_modes -> fno_modes`; reject explicit use of the four
unrepresentable dead aliases with a stable actionable error. Preserve
equal/conflicting dual-input behavior for all representable topology aliases.

Replace `_merge_deprecated_execution_model_aliases()` with a pure
`resolve_topology_compatibility(canonical_model_patch, normalized_execution)`
that copies its inputs and returns a new canonical model patch, consumed-source
audit, and deferred notices. It must never mutate the caller's override mapping
or emit warnings. The factory emits the returned notices only after the entire
payload is valid.

Implement `resolve_execution_environment()` as a pure function of the
structurally valid primitive runtime candidate and an optional immutable
`ExecutionCapabilities(cuda_available, cuda_device_count)` snapshot. A small
orchestrator checks whether a snapshot is needed and calls
`observe_execution_capabilities()` lazily only after scientific, topology, and
optimizer resolution succeeds. Apply the selected environment policy from the
design: accelerator auto resolution, accelerator-aware `devices="auto"`,
non-CUDA pin-memory downgrade with notice, and CPU fp16-to-bf16 resolution.
Return an `EnvironmentResolution` with separate requested/resolved mappings,
then instantiate `PyTorchExecutionConfig` using only the resolved values.

Training and inference factories accept
`ExecutionRequest | PyTorchExecutionConfig | None`, but payloads continue to
carry the resolved `PyTorchExecutionConfig` for compatibility. When input is
`None`, validate the scientific/config candidate first, then create the
primitive default request, collect capabilities if required, and construct the
config. Add optional keyword-only `execution_capabilities` injection to both
factories for deterministic tests. A caller-created bare config has already
performed accelerator resolution in its compatibility constructor. Normalize
its copied field values into the same environment lane, treating that
accelerator as already resolved so no second accelerator lookup or POLICY-001
warning occurs. Still resolve `devices="auto"`, non-CUDA pin-memory, and CPU
fp16 precision, then construct the final payload config from those resolved
values. Direct constructor timing/signature remains unchanged; the factory
does not pretend the constructor's earlier effects can be rolled back.

Remove the PyTorch default-construction branch from
`ptycho/workflows/backend_selector.py`: when a programmatic caller supplies
`torch_execution_config=None`, pass `None` through to the Torch workflow so the
factory performs scientific validation before lazy capability observation.
Update the existing GPU-first/CPU-fallback selector tests to assert final
factory/workflow resolution instead of requiring the selector itself to
construct the config. Directly supplied compatibility configs continue to pass
through unchanged.

Inference normalization is phase-aware: execution runtime fields remain
accepted, but explicit optimizer/topology compatibility fields that inference
cannot consume fail with a deterministic phase-owner error.

Accumulate request notices and emit them only after successful payload
construction. Include requested and resolved environment fields in runtime
audit, never in `params.cfg`, ModelSpec, or artifact identity. Lock the exact
nested record:

```python
overrides_applied["execution_runtime"] == {
    "explicit_fields": [...],  # sorted canonical names
    "requested": {
        "accelerator": ...,
        "devices": ...,
        "pin_memory": ...,
        "precision": ...,
    },
    "resolved": {
        "accelerator": ...,
        "devices": ...,
        "pin_memory": ...,
        "precision": ...,
    },
    "capabilities": (
        {"cuda_available": ..., "cuda_device_count": ...}
        if capability observation was required
        else None
    ),
}
```

Optimizer and topology source provenance remains in separate deterministic
owner-specific audit entries.

- [ ] **Step 4: Run focused factory tests and verify GREEN**

Run:

```bash
python -m pytest \
  tests/torch/test_config_factory.py \
  tests/torch/test_structural_config_ownership.py \
  tests/torch/test_execution_config_defaults.py \
  -q
```

Expected: selected precedence, topology conflicts, and prior factory behavior
pass.

- [ ] **Step 5: Commit**

```bash
git add ptycho_torch/config_factory.py \
  ptycho_torch/execution_request.py \
  ptycho_torch/workflows/components.py \
  ptycho/config/config.py \
  ptycho/workflows/backend_selector.py \
  tests/torch/test_config_factory.py \
  tests/torch/test_structural_config_ownership.py \
  tests/torch/test_workflows_components.py \
  tests/torch/test_execution_config_defaults.py
git commit -m "refactor(config): resolve execution optimization inputs once"
```

### Task 5: Derive Lightning arguments from resolved TrainingConfig

**Files:**

- Modify: `ptycho_torch/workflows/components.py`
- Modify: `ptycho_torch/train.py`
- Modify: `ptycho_torch/train_lightning_only.py`
- Modify: `tests/torch/test_train_lightning_execution_contract.py`
- Modify: `tests/torch/test_workflows_components.py`
- Modify: `tests/torch/test_model_training.py`

- [ ] **Step 1: Write failing singular-owner tests**

Use a deliberately different source execution value and resolved training
value, then assert:

```python
assert model.training_config.accum_steps == resolved_training.accum_steps
assert trainer_kwargs["accumulate_grad_batches"] == (
    1 if manual_optimization else resolved_training.accum_steps
)
assert trainer_kwargs["gradient_clip_val"] == (
    None if manual_optimization else resolved_training.gradient_clip_val
)
```

Prove manual optimization reads the resolved training values inside the model
and does not independently read execution config.

Also prove:

- optimizer construction and learning-rate scaling read
  `resolved_training.learning_rate`;
- scheduler construction reads `resolved_training.scheduler` and its
  scheduler-specific fields;
- manual clipping reads both
  `resolved_training.gradient_clip_val` and
  `resolved_training.gradient_clip_algorithm`;
- manual optimization accepts `agc` and performs it at model level, while an
  automatic-optimization model with an effective `agc` policy fails before
  `L.Trainer` construction because Lightning only accepts `norm` or `value`;
- `ptycho_torch.train.main()` and `train_lightning_only.main()` never select
  learning rate from execution config after ownership resolution; and
- `_train_with_lightning()` passes the resolved execution `devices`,
  `strategy`, and `precision` to `L.Trainer` (rather than hard-coding
  `devices=1` or omitting precision), while worker settings and rank behavior
  remain execution-owned.

- [ ] **Step 2: Run focused workflow tests and verify RED**

Run:

```bash
python -m pytest \
  tests/torch/test_train_lightning_execution_contract.py \
  tests/torch/test_workflows_components.py \
  tests/torch/test_model_training.py \
  -k "accum or gradient_clip or learning_rate or scheduler or singular_owner or ddp" -q
```

Expected: Trainer arguments still come directly from execution config.

- [ ] **Step 3: Replace downstream dual reads**

After payload resolution, bind:

```python
execution_config = payload.execution_config
effective_accum_steps = payload.pt_training_config.accum_steps
effective_clip_val = payload.pt_training_config.gradient_clip_val
effective_clip_algorithm = payload.pt_training_config.gradient_clip_algorithm
```

Use these values for Trainer arguments. Preserve the existing manual
optimization rule that disables Lightning's own accumulation/clipping when the
model performs those operations itself.

Remove every optimizer selection of the form
`execution_config.learning_rate if ...` in `ptycho_torch/train.py` and
`ptycho_torch/train_lightning_only.py`. Their model LR and scheduler setup use
the already resolved TrainingConfig. Where a legacy direct entry point has not
passed through the factory, invoke the same one-way
`resolve_optimizer_ownership()` once at entry and then discard execution's
optimizer role.

Prevent the current new-CLI double resolution: thread an optional keyword-only
`resolved_payload` through the internal Torch workflow delegation. The
`ptycho_torch.train` CLI passes the payload it already created;
`_train_with_lightning()` uses that payload directly and does not call the
factory again. Direct public workflow callers omit it and resolve exactly once
inside `_train_with_lightning()`. Do not use value comparison or a hidden
attribute to guess whether ownership was already resolved.

Derive automatic Lightning clip value and algorithm together from the resolved
TrainingConfig. Preserve manual optimization's disabling of Lightning clipping
and its internal use of that same policy. If automatic optimization is active,
pass only `norm` or `value` to Lightning; reject effective `agc` with an
actionable error before constructing the Trainer. AGC remains valid for the
existing manual model-level implementation.

In `_train_with_lightning()`, replace the hard-coded `devices=1` and omitted
precision with `payload.execution_config.devices` and
`payload.execution_config.precision`. Use the same resolved execution config
for strategy selection, datamodule/DDP routing, workers, deterministic mode,
checkpointing, and logging. No downstream path may continue reading the
pre-resolution request or a different compatibility config.

Apply that same final runtime projection to the legacy-direct
`ptycho_torch.train.main()` Trainer path, which currently selects devices from
`training_config.n_devices` and omits precision.
`train_lightning_only.main()` already passes both but must be covered by the
same assertion. Across all three Trainer construction paths,
devices/strategy/precision come from the resolved execution config while
optimization comes from the resolved PT TrainingConfig.

Do not move device, strategy, precision, workers, checkpointing, logging, or
inference batch mechanics out of execution config.

- [ ] **Step 4: Run focused workflow tests and verify GREEN**

Run:

```bash
python -m pytest \
  tests/torch/test_train_lightning_execution_contract.py \
  tests/torch/test_workflows_components.py \
  tests/torch/test_model_training.py \
  -k "accum or gradient_clip or learning_rate or scheduler or singular_owner or ddp" -q
```

Expected: singular-owner tests and current manual/automatic optimization
contracts pass.

- [ ] **Step 5: Commit**

```bash
git add ptycho_torch/workflows/components.py \
  ptycho_torch/train.py ptycho_torch/train_lightning_only.py \
  tests/torch/test_train_lightning_execution_contract.py \
  tests/torch/test_workflows_components.py \
  tests/torch/test_model_training.py
git commit -m "refactor(torch): derive trainer optimization inputs"
```

### Task 6: Route supported CLI entry points through explicit requests

**Files:**

- Modify: `ptycho_torch/train.py`
- Modify: `ptycho_torch/inference.py`
- Modify: `scripts/training/train.py`
- Modify: `scripts/inference/inference.py`
- Modify: `ptycho/workflows/backend_selector.py`
- Modify focused CLI/backend-selector tests.

- [ ] **Step 1: Write failing supported-entry-point provenance tests**

Patch the request builder in CLI tests and assert each supported entry point:

- calls `build_execution_request_from_args`;
- supplies the exact canonical execution fields mapped from options present in
  its argv;
- passes the request to the backend/factory; and
- does not mark argparse defaults as explicit.

Cover `--name=value`, deprecated `--device`, paired boolean options, the
`torch-*` wrapper spellings, and
`accumulate_grad_batches -> accum_steps`. Exercise every binding in Task 3's
registry, including checkpoint/reconstruction fields, inference batch size,
and the two-field `--disable_mlflow` fan-out.

For `scripts/training/train.py`, cover simultaneous canonical and compatibility
optimizer options with different values. Assert the parser's actual presence
set is used to build an explicit canonical factory override mapping and that
this mapping reaches `run_cdi_example_torch(..., overrides=...)`; canonical
`--scheduler` must beat explicit compatibility `--torch-scheduler`.

Assert native `ptycho_torch.inference` request provenance survives until
`create_inference_payload()` and is not dereferenced as though the request
itself were a config. The unified `scripts/inference/inference.py` has no
factory call: assert that it passes the request through the shared
phase-aware `resolve_runtime_execution_request()` boundary exactly once, then
uses only that result's resolved config for `model.to()` and inference while
retaining its requested/resolved audit.

- [ ] **Step 2: Run focused CLI tests and verify RED**

Run:

```bash
python -m pytest \
  tests/torch/test_cli_train_torch.py \
  tests/torch/test_cli_inference_torch.py \
  tests/scripts/test_training_backend_selector.py \
  tests/scripts/test_inference_backend_selector.py \
  -k "execution and explicit" -q
```

Expected: entry points still call the compatibility config builder.

- [ ] **Step 3: Adopt the request builder**

At each parser boundary, derive explicit destinations from the actual argv
tokens through the shared option-to-canonical-field map and pass them to
`build_execution_request_from_args`. Keep
`build_execution_config_from_args` supported for external compatibility.

Do not infer explicitness by comparing a value with its default: a user may
explicitly supply the default.

Pass `ExecutionRequest` through the supported backend/workflow boundary until
the factory normalizes it. After factory resolution, downstream runtime code
uses `payload.execution_config`; it never dereferences the request directly.
Remove the public wrapper scripts' ad hoc `sys.argv` membership checks.

The unified training wrapper must also preserve suppliedness for canonical
public optimizer fields. Use the same raw argv token scan as execution
suppliedness, with a separate declared public-optimizer option map; then read
the values for exactly those present canonical destinations from the parsed
Namespace. This remains correct when a user explicitly supplies the
presentation default. Thread that mapping through a new optional
`torch_factory_overrides` argument on the backend selector into the existing
Torch workflow `overrides` channel. File/YAML-resolved values remain in the
typed priority-3 `training_baseline`; do not copy the whole resolved public
dataclass into the explicit mapping. This sidecar is required only at the CLI
boundary and is not persisted or attached to the public dataclass.

For the unified inference wrapper, which loads a backend bundle directly and
does not call `create_inference_payload()`, invoke the same
`resolve_runtime_execution_request(request, mode="inference", ...)` orchestration
used by the factory after public configuration and bundle validation. Use its
resolved config and audit; do not unwrap request values or construct a second
config locally.

Preserve the native `ptycho_torch.train` four-value scheduler set exactly; it
is normative in `specs/ptychodus_api_spec.md` §7.1 and must not be broadened by
this ownership change. Correct the unified `--torch-scheduler` parser's stale
`CosineAnnealing` spelling to its documented public TrainingConfig set:
`Default`, `Exponential`, `WarmupCosine`, and `ReduceLROnPlateau`. The effective
factory resolver still validates the complete Torch TrainingConfig domain for
supported programmatic inputs. Also correct the stale checkpoint help that
advertises unsupported `save_top_k=-1`.

- [ ] **Step 4: Run the focused CLI/backend tests and verify GREEN**

Run the same command as Step 2.

Expected: supported entry points preserve explicit provenance and existing
CLI behavior.

- [ ] **Step 5: Commit**

```bash
git add ptycho_torch/train.py ptycho_torch/inference.py \
  scripts/training/train.py scripts/inference/inference.py \
  ptycho/workflows/backend_selector.py \
  tests/torch/test_cli_train_torch.py \
  tests/torch/test_cli_inference_torch.py \
  tests/scripts/test_training_backend_selector.py \
  tests/scripts/test_inference_backend_selector.py
git commit -m "refactor(cli): thread execution request provenance"
```

### Task 7: Focused ownership regression

**Files:**

- Modify only a directly stale execution/configuration guide sentence if code
  symbol naming changed during implementation.

- [ ] **Step 1: Run the claim-matched focused set**

```bash
python -m pytest \
  tests/torch/test_cli_shared.py \
  tests/torch/test_cli_train_torch.py \
  tests/torch/test_cli_inference_torch.py \
  tests/torch/test_execution_config_defaults.py \
  tests/torch/test_config_factory.py \
  tests/torch/test_structural_config_ownership.py \
  tests/torch/test_train_lightning_execution_contract.py \
  tests/torch/test_workflows_components.py \
  tests/torch/test_model_training.py \
  tests/scripts/test_training_backend_selector.py \
  tests/scripts/test_inference_backend_selector.py \
  -q
```

Expected: all focused execution ownership, provenance, factory, DDP/Trainer,
and CLI tests pass.

- [ ] **Step 2: Verify excluded boundaries are untouched**

Run:

```bash
BASE_SHA=<execution-start SHA recorded immediately before Task 1>
git diff --name-only "$BASE_SHA"..HEAD
```

Expected: no ModelSpec, artifact schema, checkpoint codec, MLflow serializer,
or simulation implementation file appears.

- [ ] **Step 3: Prove identity exclusion and inspect production ownership**

Add/retain a focused test that constructs a request with distinctive
runtime-only values (`devices`, `precision`, `pin_memory`, `logger_backend`,
`persistent_workers`, `checkpoint_save_top_k`) and proves they are absent
from:

- the CONFIG-001 `params.cfg` projection;
- `ModelSpec` fields/payload; and
- the existing portable artifact identity/config sections.

The test must also prove requested/resolved accelerator, devices, pin-memory,
and precision values appear in runtime audit only. Run that exact selector:

```bash
python -m pytest \
  tests/torch/test_config_factory.py \
  -k "execution_values_excluded_from_identity" -q
```

Then run:

```bash
rg -n "execution_config\\.(learning_rate|scheduler|gradient_clip|accum_steps)" \
  ptycho_torch ptycho scripts
```

Expected: remaining reads are compatibility input collection or runtime
resolution only; model/optimizer and Trainer parameter-update behavior use the
resolved Torch TrainingConfig owner.

- [ ] **Step 4: Commit any directly required routing correction**

If no documentation symbol changed, do not create an empty commit. Otherwise:

```bash
git add docs/CONFIGURATION.md docs/workflows/pytorch.md
git commit -m "docs(config): route execution ownership"
```
