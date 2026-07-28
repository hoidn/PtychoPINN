# Configuration Compatibility Retirement and Schema Convergence Design

**Status:** Proposed for repository promotion. The target design was approved
on 2026-07-28, but it is not authoritative on a branch until its named parent
and child authorities coexist there and `docs/index.md` routes to it.

**Parent authority:**
`docs/superpowers/specs/2026-07-28-configuration-boundary-architecture.md`.
That architecture continues to own configuration lifecycle boundaries,
family-specific resolution, the one-way legacy projection, and the Pydantic
Adoption Gate.

**Child authorities:**

- `docs/superpowers/specs/2026-07-28-torch-config-resolution-design.md` owns
  Torch configuration resolution and the remaining mutation compatibility
  surface.
- `docs/superpowers/specs/2026-07-28-execution-config-ownership-design.md` owns
  execution/runtime provenance and compatibility aliases.
- `docs/specs/2026-07-28-params-cfg-strangler-design.md` owns migration away
  from global runtime reads.
- `docs/specs/2026-07-28-pydantic-family-adoption-design.md` owns the final,
  conditional per-family structural-validation decision.
- `docs/superpowers/specs/2026-07-28-configuration-persistence-boundaries.md`
  continues to own artifact, checkpoint, and historical-load behavior.

This document coordinates their dependency order. It does not replace their
family-specific contracts.

## 1. Decision

Configuration compatibility is retired by migrating known consumers and then
deleting the obsolete surface. Deprecated Python, CLI, JSON, and MLflow
configuration APIs do not remain indefinitely once their declared public and
internal consumers have moved to supported resolvers and versioned artifacts.

No replacement generic mutation helper is introduced. In particular,
`update_existing_config()` is not renamed, hidden behind `ConfigManager`, or
reimplemented with equivalent tolerant semantics.

Schema convergence is evaluated only after compatibility aliases, mutation
islands, and modern dependencies on `params.cfg` have been reduced. Pydantic is
an optional implementation mechanism at a complete family boundary, not the
driver of the migration. A family adopts it only when the existing adoption
gate proves net simplification.

## 2. Target architecture

```text
YAML / TOML / CLI / supported Python API
                        |
                        v
           family-specific return-new resolver
                        |
                        v
          validated resolved owner records
               |                       |
               |                       +--> modern consumers
               |
               +--> one-way legacy projection
                              |
                              v
                    scoped params.cfg state
                              |
                              +--> declared legacy consumers only
```

Artifact and checkpoint restoration joins this flow through its versioned
codec and produces resolved owner records. Unversioned MLflow scalar
dictionaries and `params.cfg` never become reverse configuration sources.

The intended end state has:

- one supported resolution path per public or Torch phase;
- explicit provenance for execution overrides;
- explicit resolved-owner inputs for maintained runtime consumers;
- a bounded, scoped legacy adapter only while declared legacy consumers
  remain;
- no ownerless mapping mutation;
- no generic repository-wide configuration registry or mega-context object.

## 3. Ownership boundaries

| Concern | Owning boundary | Retirement rule |
|---|---|---|
| File/CLI/API precedence and complete snapshot construction | Public and Torch family resolvers | Remains supported |
| Tolerant dataclass mutation (`update_existing_config`) | Transition Stage 1 | Migrate maintained callers; delete the mutator, while enclosing APIs follow their own support contracts |
| Optimizer and topology compatibility fields on execution config | Execution compatibility lane | Delete after consumers use canonical owners |
| Legacy global runtime reads | `params.cfg` strangler | Replace with explicit owner inputs, subsystem by subsystem |
| One-way legacy projection (`update_legacy_dict`) | Legacy compatibility bridge | Retain until the final declared legacy consumer is migrated |
| Artifact and checkpoint restoration | Versioned persistence codecs | Remains separate; never replaced by generic resolution |
| Structural type/domain validation | Each configuration family | Re-evaluate after the compatibility/state migration |

`update_existing_config()` and `update_legacy_dict()` are intentionally
different:

- `update_existing_config()` tolerantly mutates configuration records and is
  retired.
- `update_legacy_dict()` projects already-resolved configuration into the
  global compatibility dictionary and remains temporarily mandatory for
  legacy consumers.

## 4. Consumer-gated compatibility removal

A compatibility surface may be deleted when all of the following are true:

1. Its in-repository callers and imports are classified as maintained,
   migrated, archived, or dead.
2. Every maintained caller uses the supported resolver, factory, execution
   request, or versioned artifact loader.
3. Known internal-only consumers have migrated without losing their extended
   model fields or sealed artifact identities.
4. The owning external specification is updated in the same change as any
   externally visible removal.
5. Focused evidence proves the supported replacement and the affected
   compatibility fixtures that remain contractual.

An unknown hypothetical consumer does not justify an indefinite shim.
Conversely, a known consumer cannot be deleted merely because the modern route
exists.

Compatibility loaders for historical artifacts remain only where the
persistence contract explicitly names their era and upgrade semantics.
Unversioned MLflow configuration dictionaries do not acquire such authority by
being routed through a modern resolver.

## 5. Migration sequence

### Entry precondition: colocated authority and substrate

Before compatibility removal begins, every authoritative branch being changed
must contain this design's governing documents and the supported return-new
resolver substrate. A public variant must remain free of excluded internal
families; an internal variant must preserve its permitted extensions. Each
variant independently reaches its consumer-deletion gates.

Branch choreography, completed-plan status reconciliation, and stable
implementation-evidence pointers belong to the implementation roadmap rather
than this durable architecture. They do not expand the acceptance criteria
below.

### Stage 1: retire the tolerant mutation surface

The supported public and Torch factories already use return-new resolution.
The remaining `update_existing_config()` calls are concentrated in legacy
API/JSON/MLflow/notebook/runner surfaces rather than maintained factories.
That concentration does not make every enclosing package disposable.

Before editing, classify each surface as maintained, compatibility-only,
archival, or dead:

- maintained functionality migrates to a supported resolver, bundle loader, or
  explicit `dataclasses.replace()` operation;
- a runner keeps its active resolved-input path while an obsolete loader
  fallback may be removed;
- MLflow remains observability data, while unversioned scalar restoration is
  retired unless a separately governed versioned converter owns it;
- tracked notebooks migrate or are explicitly archived according to their
  support contract;
- an API or entry point is deleted only when its owning specification removes
  that support, not merely because it currently calls the updater.

External specifications are changed to record an approved support decision,
not amended retroactively to manufacture evidence that a surface was
deprecated. Removal may target updater-dependent configuration loading without
deleting unrelated supported API functionality.

No generic `replace_config(config, patch)` or equivalent is permitted.

Acceptance requires zero production definitions, imports, or calls of
`update_existing_config()` on each branch being changed; a recorded disposition
for every former caller; maintained train/inference/API paths entering through
a resolver or already-resolved bundle; unchanged caller-owned records and
internal architecture fields; versioned artifact loading remaining
authoritative; and the `update_legacy_dict()` bridge remaining unchanged for
the next stage.

### Stage 2: close execution compatibility lanes

The execution boundary completes an input/output split:

```text
primitive runtime patch --> ExecutionRequest ----+
                                                  |
canonical topology patch --> ModelConfig --------+--> pure ownership and
                                                  |    structural validation
canonical optimizer patch -> TrainingConfig -----+             |
                                                                v
                                                     capability resolution
                                                                |
                                                                v
                                             resolved PyTorchExecutionConfig

ModelConfig ------> ModelSpec
TrainingConfig ---> optimizer / Trainer mirrors
```

`ExecutionRequest` becomes the sole supported provenance-carrying input.
`PyTorchExecutionConfig` remains the pure, resolved Trainer/DataLoader output
carrier and owns runtime fields only. The execution child design and revised
external API own the request representation and constructor exposure; this
parent requires only one ordinary request boundary and forbids
constructor-inferred provenance.

Request, topology, optimizer, and structural validation complete before any
capability observation. Hardware is inspected only after those candidates are
valid and only when an unresolved runtime value requires it.

The execution child design and external specification are amended first. The
contract delta removes promises for the old config-returning CLI helper,
standalone immediate accelerator resolution, bare resolved-carrier factory
input, and the execution-compatibility optimizer priority tier; maps optimizer
CLI flags to canonical `TrainingConfig`; and defines
`PyTorchExecutionConfig` as a resolved runtime carrier.
Consumers then migrate before the following are deleted in order:

1. the unused config-returning CLI helper;
2. the standalone immediate accelerator resolver;
3. the ambiguous bare-`PyTorchExecutionConfig` programmatic lane;
4. the hidden `_explicit_structural_aliases` constructor sentinel;
5. topology aliases now owned by canonical `ModelConfig`;
6. optimizer, scheduler, clipping, and accumulation aliases now owned by
   canonical `TrainingConfig`;
7. constructor-time capability observation on the resolved output carrier.

Runtime-only fields remain owned by execution configuration. Requested/resolved
runtime audit may be written to non-authoritative observability outputs, but it
is excluded from `ModelSpec`, portable artifact identity, and authoritative
checkpoint identity.

Acceptance requires:

- one supported unresolved execution-request boundary and one resolved runtime
  output;
- rejection of a bare resolved carrier before capability observation, payload
  creation, or global mutation;
- no structural alias registry, hidden explicit-field sentinel, optimizer
  compatibility precedence tier, or compatibility provenance owner;
- explicitly supplied CLI optimizer values reaching canonical
  `TrainingConfig`, while omitted flags do not overwrite file or baseline
  values;
- topology producing the same canonical `ModelConfig` and `ModelSpec`;
- exact frozen v1/v2 artifact fields and fixtures, canonical optimizer values
  remaining in persisted `TrainingConfig`, canonical topology remaining in
  `ModelSpec`, and no schema-version bump or historical-decoder
  reinterpretation;
- the resolved runtime carrier continuing to drive accelerator, devices,
  strategy, precision, determinism, and DataLoader mechanics, while Trainer
  clipping and accumulation derive solely from resolved `TrainingConfig`.

### Stage 3: strangle modern `params.cfg` dependencies

Follow `docs/specs/2026-07-28-params-cfg-strangler-design.md` one subsystem at
a time.
Modern code receives existing resolved owner records or the smallest explicit
primitive inputs required by its operation. It never receives a new
repository-wide context bag.

The intermediate completion condition is zero supported modern reads from
`params.cfg`. The one-way bridge may remain for explicitly declared legacy
consumers. Final bridge deletion is a later contract transition requiring zero
remaining readers and an atomic update to specifications that currently
mandate the bridge.

### Stage 4: re-evaluate schema convergence

For schema evaluation, **post-strangler** means the child design's
`modern_isolation_complete` milestone: supported modern code has zero
`params.cfg` reads. A named, scoped legacy bridge may still remain; final
`global_bridge_retired` is not a prerequisite.

Only after Stages 1–3 and that milestone, evaluate public and Torch families
independently under the Pydantic Adoption Gate:

- remeasure the manual structural-validation surface after compatibility
  deletion;
- produce a dry-run deletion ledger before production changes;
- prove constructor, reflection, error, and accepted-value behavior with the
  installed Pydantic version;
- adopt only if adapter-attributable production code and decision complexity
  decrease;
- otherwise record `retain manual` as the successful terminal decision.

Simulation remains the proven adopted family. Execution configuration, partial
patches, `params.cfg`, MLflow identity, and versioned persistence remain
outside Pydantic.

### Stage 5: close routing and portfolio state

Update the parent architecture's family portfolio only for outcomes actually
implemented and verified. Update child statuses and documentation routing
without copying tranche chronology or temporary evidence into baseline
contracts.

## 6. Acceptance

The transition is complete only when:

- completed-plan status and routing surfaces agree;
- the supported factories and CLIs have no dependency on
  `update_existing_config()`;
- the tolerant mutator and all executable imports are gone, and every former
  caller has a contract-backed migrated, retained, archived, or deleted
  disposition;
- supported execution accepts explicit provenance and no longer infers
  compatibility ownership from a bare execution record;
- canonical model and training owners no longer compete with execution aliases;
- supported modern runtime code does not read `params.cfg`;
- no path reconstructs structured authority from `params.cfg` or unversioned
  MLflow scalars;
- remaining legacy projection and versioned artifact behavior matches its
  current governing specification;
- each Pydantic candidate records either a gate-backed adoption or a
  gate-backed retain-manual decision.

Evidence is tranche-local and claim-matched. Raw grep counts, historical task
checkboxes, and optional broad test sweeps do not independently create
completion requirements.

## 7. Explicitly rejected

- Replacing `update_existing_config()` with another tolerant mutation helper.
- Keeping deprecated configuration APIs indefinitely after known consumers
  migrate.
- Treating unversioned MLflow scalar dictionaries as authoritative
  configuration snapshots.
- Reconstructing resolved dataclasses from `params.cfg`.
- Introducing a universal `RuntimeContext`, schema registry, reflective
  patcher, or mega-config record.
- Making Pydantic a YAML/TOML/CLI bridge or a persistence format.
- Adopting Pydantic before measuring net deletion at the post-strangler
  boundary.
- Removing the mandatory one-way legacy bridge while declared legacy consumers
  or governing external specifications still require it.
