# `params.cfg` Strangler Design

**Status:** Proposed for repository promotion. The target design was approved
on 2026-07-28, but it is not authoritative on a branch until its parent
transition design and governing configuration authorities coexist there.

**Parent authority:**
`docs/specs/2026-07-28-configuration-compatibility-retirement-and-schema-convergence-design.md`.

**Related authorities:**

- `docs/superpowers/specs/2026-07-28-configuration-boundary-architecture.md`
  owns the one-way projection invariant.
- `docs/superpowers/specs/2026-07-28-configuration-persistence-boundaries.md`
  owns artifact and checkpoint restoration.
- `docs/specs/spec-ptycho-config-bridge.md` and
  `specs/ptychodus_api_spec.md` own current externally visible bridge
  requirements.
- The repository `AGENTS.md` restrictions continue to govern protected core
  files.

## 1. Goal

Remove `ptycho.params.cfg` as a runtime dependency of supported modern code
without introducing another global object or changing versioned artifact
semantics.

The first target is **zero supported modern reads**. The final target is
deletion of the global projection path once every declared legacy consumer has
migrated and the specifications that require it have been amended.

This is a strangler migration: explicit modern paths grow around legacy reads;
the global bridge contracts only as consumers disappear.

## 2. Terminology and non-equivalence

- **Resolved owner:** an existing resolved `SimulationConfig`, `DataConfig`,
  `ModelConfig`, `TrainingConfig`, `InferenceConfig`, execution request, or
  phase payload.
- **Legacy projection:** the flat mapping produced from resolved owners and
  committed through `update_legacy_dict()`.
- **Legacy consumer:** code whose current governing contract still reads
  `params.cfg`.
- **Modern consumer:** a supported path whose public boundary can receive a
  resolved owner or explicit primitive input.

This design does not retire `update_existing_config()`; transition Stage 1
owns that separate tolerant dataclass mutator. Maintained callers migrate, and
an enclosing API is deleted only under its own support contract.

## 3. Target dataflow

```text
resolved phase payload
     |
     +--> modern subsystem(owner records / explicit inputs)
     |
     +--> legacy projection --scoped commit--> legacy subsystem
```

The projection branch is one-way. A legacy wrapper may read the exact primitive
arguments required by a legacy call, but it must not reconstruct a supposedly
authoritative dataclass or runtime bundle from the global dictionary.

New code must not cache, alias, or retain `params.cfg` outside the existing
scoped lifecycle. Nested, exceptional, threaded, task-local, and process
behavior continues to use `ptycho/config/legacy_state.py`.

## 4. Consumer classification

Every occurrence considered for migration is classified before editing:

| Class | Examples | Disposition |
|---|---|---|
| Projection writer | Configuration bridge/factory commit | Retain until its last declared reader is gone |
| Supported modern reader | Torch workflow, model manager, data bridge, maintained orchestration | Inject a resolved owner or explicit primitive |
| Shared adapter reader | Backend selector, shared data/workflow component | Split modern explicit core from bounded legacy wrapper |
| Legacy core reader | TensorFlow-era model, simulator, helper, protected physics module | Migrate only under explicit plan authority; otherwise keep scoped |
| Persistence/restoration | Versioned artifact or checkpoint loader | Preserve codec ownership; emit resolved owners, never global authority |
| Tooling/study/notebook | Maintained script or analysis path | Use supported payloads; archive paths tied to obsolete artifacts |
| Test observation | Lifecycle, isolation, compatibility, or behavioral fixture | Retain only if it proves a current contract |
| Dead/import-only reference | Unused import, comment, unreachable branch | Delete without treating it as migration evidence |

A raw repository occurrence count is discovery evidence, not an acceptance
gate.

Writers are classified separately because deleting readers before
consolidating mutation can leave invisible state changes:

- canonical public projection through `dataclass_to_legacy_dict()` and
  `update_legacy_dict()`;
- Torch factory projection through its named bridge;
- duplicate workflow or script projections;
- versioned historical restoration;
- simulation-then-runtime sequential projection;
- test-only mutation used to prove lifecycle behavior.

No new writer is permitted. Remaining writers converge on named bridge or
archive boundaries before their readers are removed.

## 5. Migration seam

The preferred seam separates an explicit implementation from a temporary
legacy adapter:

```python
def _operation_impl(data, *, detector_size, grid_size):
    ...


def operation_from_resolved(data, *, data_config, model_config):
    return _operation_impl(
        data,
        detector_size=data_config.N,
        grid_size=model_config.gridsize,
    )


def legacy_operation(data):
    return _operation_impl(
        data,
        detector_size=params.cfg["N"],
        grid_size=params.cfg["gridsize"],
    )
```

The legacy adapter may disappear when its callers migrate. It must not construct
`ModelConfig(**params.cfg)` or copy the entire global dictionary into a new
context record.

When an existing resolved owner is too broad for a leaf function, pass the
smallest meaningful primitive set. Introduce a new subsystem-specific record
only when it owns a stable domain concept and replaces more duplication than it
adds. A generic repository-wide runtime context is prohibited.

## 6. Ordered tranches

### Tranche A: inventory and characterize boundaries

- Classify reads, writes, imports, and restoration behavior.
- Identify the resolved owner for every maintained consumer.
- Record which legacy readers force the bridge to remain.
- Pin only affected observable behavior, including lifecycle/isolation where
  the scoped global remains.

This tranche changes no runtime ownership.

### Tranche B: consolidate writers and containment

Remove duplicate modern projection calls and route every remaining runtime
commit through the existing named bridge and
`ptycho/config/legacy_state.py` scopes:

- `configured_params_scope()` contains resolved projection and rollback;
- `archived_params_scope()` contains historical restoration;
- `legacy_params_scope()` contains an already-projected legacy leaf.

Scopes move inward as explicit paths replace global readers. Once a caller and
all of its descendants are explicit, remove its outer scope instead of keeping
redundant containment.

Requirements:

- no supported ad hoc `params.cfg.update`, assignment, or duplicate
  `update_legacy_dict()` remains outside the writer allowlist;
- a failed resolution or legacy call restores exact dictionary contents and
  seal state;
- nested, exceptional, threaded, task-local, and process behavior remains
  characterized.

### Tranche C: supported Torch leaves and tooling

Migrate maintained Torch workflow components, model management, raw-data
bridges, backend-neutral orchestration, studies, and notebooks that already
have access to phase payloads.

Requirements:

- caller-owned configurations are not mutated;
- payload records remain the exact resolved records;
- sealed model/artifact identity is unchanged;
- no new global or reverse projection is introduced.

### Tranche D: shared data and workflow components

Split shared preprocessing, backend selectors, and workflow helpers into
explicit modern cores plus bounded legacy adapters where TensorFlow or an
external compatibility path still needs the global state.

The modern route must be callable and testable without any `params.cfg`
mutation.

### Tranche E: archive seam

Decode historical snapshots at their versioned persistence boundary. Values
that belong to public or Torch owners are returned explicitly; genuinely
historical extras remain owned by their era-specific codec.

Use `archived_params_scope()` only around a still-unconverted legacy
construction leaf. Do not reinterpret `ModelSpec`, artifact schema identifiers,
tensor envelopes, or unversioned MLflow scalars.

Requirements:

- existing bundles decode to the same effective reconstruction inputs;
- malformed loads roll back atomically;
- modern reload consumers receive explicit resolved owners;
- only a declared legacy leaf receives temporary global restoration.

### Tranche F: legacy backend and protected core

Migrate remaining TensorFlow-era and core physics readers only under an
explicit plan that authorizes the affected stable files. Preserve exact
forward-physics, data-shape, and external API contracts.

This tranche may retain small legacy adapters when deleting them would require
an independently governed backend or external-interop removal. Such adapters
must be named as legacy and remain scoped.

### Tranche G: zero-modern-read milestone

Remove redundant outer scopes from modern entry points and update configuration
guidance so projection is required only immediately before invoking a remaining
legacy leaf.

Requirements:

- no supported modern consumer reads `params.cfg`;
- every remaining production read is confined to a named legacy/archive
  adapter with an owner and removal condition;
- the pure projection, mutation bridge, and lifecycle helpers may remain;
- this milestone is not described as global-bridge deletion.

### Tranche H: final bridge retirement

Delete the global projection path only when:

1. no supported or declared legacy runtime consumer reads `params.cfg`;
2. versioned artifacts restore directly to resolved owners;
3. external bridge specifications are amended atomically;
4. lifecycle tests are deleted or converted according to the removed
   contract;
5. a focused end-to-end path for every retained backend succeeds without a
   global commit.

If declared legacy consumers remain, completion stops at zero modern reads and
the scoped bridge remains an explicit compatibility boundary.

Final retirement removes:

- `params.cfg` runtime storage and supported imports;
- `update_legacy_dict()` and factory population wrappers;
- global archive restoration;
- lifecycle scopes in `legacy_state.py` that have no remaining named consumer;
- `dataclass_to_legacy_dict()` unless a versioned historical codec explicitly
  retains its pure output.

The same change updates the configuration-boundary architecture's
legacy-projection portfolio row, the persistence-boundaries design, external
bridge specifications, the configuration guide, and documentation routing.
Acceptance requires zero supported production definitions, imports, reads, or
writes for the retired bridge; historical artifact loading without global
restoration; no replacement singleton or context registry; and exact
compatibility behavior through the removal of the final declared consumer.

## 7. Tranche acceptance

Each migrated subsystem must prove:

- its supported entry point accepts resolved owners or explicit primitives;
- it produces the same contract-relevant output for a pinned representative
  input;
- its modern path succeeds when access to `params.cfg` is denied or poisoned;
- invalid configuration fails at the resolver or subsystem boundary before
  global mutation;
- remaining legacy callers still execute inside the existing scoped lifecycle;
- artifact bytes, identity fields, and historical upgrade behavior remain
  unchanged unless the persistence design explicitly authorizes a version
  transition.

Deletion of one reader does not require a repository-wide test sweep. Evidence
is focused on the migrated subsystem and affected bridge invariant.

## 8. Completion states

Two states are intentionally distinct:

1. **`modern_isolation_complete`:** supported modern code has zero direct
   `params.cfg` reads; the one-way bridge remains solely for named legacy
   consumers.
2. **`global_bridge_retired`:** no consumer remains, external contracts are
   updated, and the projection/lifecycle machinery is deleted.

The first state is a valid intermediate architectural improvement. It must not
be mislabeled as deletion of `params.cfg`.

## 9. Explicitly rejected

- A universal runtime context or replacement global singleton.
- Reconstructing dataclasses, phase payloads, or artifact identity from
  `params.cfg`.
- Passing an unclassified full dictionary through new APIs.
- Moving versioned artifact upgrade behavior into the global bridge.
- Editing protected physics modules as incidental cleanup.
- Rewriting functioning legacy consumers before an explicit owner and
  falsifiable parity boundary are identified.
- Treating grep-count reduction as proof of behavioral migration.
