# PtychoPINN Refactoring Roadmap — Vertical Boundary Migrations

> **Status authority:** Active roadmap for the `fno-stable` line, revised
> 2026-07-15 after a source, contract, and architecture audit.
>
> **Scope:** This document owns live refactoring status, sequencing, and
> dependency decisions. Normative behavior remains owned by `specs/` and
> `docs/specs/`; this roadmap cannot weaken or replace those contracts.
>
> **Historical plans:** The 2026-07-06 pipeline plans and the original 2026-07-07
> Phase 0–3 plans are retained as provenance. Their task bodies are not an
> executable queue and do not create completion requirements. Each has a current
> disposition header pointing back here.

## 1. Why this revision exists

The original roadmap correctly identified the major risks:

- scientific behavior is distributed across TensorFlow, Torch, legacy global
  configuration, and workflow code;
- silent fallbacks can return plausible but scientifically different output;
- configuration ownership and precedence are not explicit enough;
- geometry, scaling, support, patch layout, and reconstruction conventions are
  encoded in control flow rather than typed contracts;
- central modules and study runners have accumulated unrelated responsibilities.

Its sequencing instinct also remains sound: contain hidden state and preserve
behavioral evidence before extracting abstractions. However, the original
horizontal phases grouped unrelated changes and proposed abstractions that no
longer match the repository:

- a backend-neutral core containing `RawData` and `params.cfg` would carry
  framework and lifecycle coupling into the new package;
- a single `Reassembler` would conflate training-forward assembly, ordinary
  inference stitching, and VarPro calibration;
- a broad `TrainingBackend` would hide different training, bundle, and inference
  artifact models;
- moving the whole grid-lines workflow into `scripts/` would relocate reusable
  application logic instead of decomposing it;
- reflective dataclass comparison cannot determine semantic field ownership;
- deleting `ptycho_torch.api` conflicts with a currently acceptable external API
  route in `specs/ptychodus_api_spec.md`.

This revision therefore organizes work as vertical boundary migrations. Each
slice establishes one explicit contract, migrates a bounded production path,
and gathers evidence matched to the behavior it claims to preserve.

## 2. Authority, branch, and state model

### 2.1 Authority stack

Apply the repository authority order:

1. external contracts in `specs/` and internal normative contracts in
   `docs/specs/`;
2. an explicitly approved design or plan within its stated scope;
3. this roadmap for live refactoring status and sequencing;
4. historical phase plans, reports, reviews, and run artifacts as evidence only.

Checks produce evidence; they do not independently add requirements. Open todos,
historical task lists, and reviewer suggestions do not expand a slice's gate.

### 2.2 Branch scope

This roadmap describes `fno-stable`, where the Hybrid/FFNO model families,
current Torch reconstruction paths, and CI compatibility work live. Before a
slice is ported to `main`, classify its changes as one of:

- portable contract or framework-neutral utility;
- backend-specific implementation;
- `fno-stable` compatibility behavior that needs a separate main-side design.

Do not assume one commit sequence or package migration applies identically to
both histories.

### 2.3 Status storage

There is no machine-readable selector, tranche manifest, or active workflow
state for this initiative. This file is the sole live status surface. The
ignored `.superpowers/sdd/progress-refactor-*.md` files are historical execution
evidence, not routing inputs. `docs/plans/README.md` and `docs/index.md` route to
this file but do not duplicate its status.

## 3. Audited current state (2026-07-15)

| Surface | Current state | Disposition |
|---|---|---|
| Original Phase 0 | Core zero-consumer deletions completed; residual items are independent, obsolete, or were rejected after live-consumer discovery | Closed; residual hygiene moves to Slice 7 only when independently justified |
| Original Phase 1 | `params` warn-mode seal landed; test lifecycle containment, fail-loud handling, and explicit contract work remain | Partially superseded by Slice 2 |
| Original Phase 2 | Reassembly coverage and inference routing advanced through later compatibility work; named config, geometry, metrics, API, and runner moves remain pending or invalidated | Superseded by Slices 3–7 |
| Original Phase 3 | No neutral-core, reassembler, or backend interface extraction landed; grid-lines and central Torch modules grew | Superseded by Slices 3–6 |
| 2026-07-06 pipeline plans | Dead forks remain; CLI routing partly changed; old global scaling diagnosis is legacy-profile-only | Historical input to Slices 5 and 7 |
| Simulation configuration | The [simulation-config implementation plan](2026-07-14-simulation-config-boundary-matched-probe.md) was implemented and committed as `8d2a2c11d`. The repository-wide suite exposed one attributable compatibility defect—the probe-stress smoke changed flat geometry while retaining the prior nested `SimulationConfig`—and that caller now updates both views in lockstep; its exact falsifier passes. | The post-fix repository-wide rerun required by Section 8 remains pending before this major chunk is closed. This blocks further work on the same grid-lines/config path, not non-overlapping model-ownership work in Slice 3. |

The audit also found that `ptycho_torch` itself imports without TensorFlow. The
important framework-crossing edges are narrower: legacy `ptycho.params`, the
TensorFlow-coupled operations in `RawData`, and runtime TF helper/reassembly
calls. Refactoring should remove those specific edges rather than treating every
Torch import from `ptycho` as invalid.

## 4. Target architecture

### 4.1 Boundaries

| Boundary | Owns | Must not own |
|---|---|---|
| Scientific/data contracts | measurement-domain and scale-profile identity; `N`/channel/layout requirements; physical probe/gauge lineage; frozen CI population statistics; coordinate, support, and crop semantics | backend runtime state or study-specific defaults |
| Simulation configuration | probe source/transforms and simulation-time masks, synthetic object, scan, detector/noise, and generation seed | optimizer, architecture, training-time masks, or model reconstruction policy |
| Torch model specification | graph topology, state-dict shape, structural physics/output interpretation, and versioned identity needed to construct and reload a Torch model | data loading, devices, loggers, trainer scheduling, or Lightning lifecycle |
| Canonical public configuration | the `ModelConfig`, `TrainingConfig`, and `InferenceConfig` Ptychodus handshake and its one-way projection through `update_legacy_dict` | compiled Torch structure or process-global semantic authority |
| Runtime configuration | accelerator, devices, workers, precision, logging, trainer controls | model structure or generated-data identity |
| Legacy bridge | bounded translation into `params.cfg` for code that still requires it | canonical ownership or long-lived process-global truth |
| Backend-native kernels | differentiable TF or Torch numerical implementation | independent definition of scientific semantics |
| Application/model factory | composition of a model specification, data/scientific contracts, and training/runtime configuration into a Lightning or TensorFlow application object | redefining or silently overriding any composed section |
| Workflow/application layer | use-case composition and normalized result/artifact DTOs | study CLI parsing or package imports from `scripts/` |

Where simulation and data/model records repeat `N`, grid, or channel values,
those repetitions are validated join keys: they must agree, but do not create
co-ownership. Current externally specified execution fields retain their public
precedence until their governing contract is migrated atomically; an internal
ownership cleanup cannot silently reinterpret them.

### 4.2 Shared semantics, native kernels

Scientific behavior is defined once by normative equations, layouts, and
independent reference oracles. TensorFlow and Torch may keep backend-native
differentiable kernels. A shared implementation is desirable only when it is
actually framework-free and does not compromise autograd, tracing, or numerical
behavior.

### 4.3 Reconstruction decomposition

Do not introduce one universal `Reassembler` or one configuration bag carrying
every reconstruction choice. Keep these contracts distinct (type names below
are illustrative, not required public symbols):

- **training-forward patch assembly:** differentiable overlap merge used inside
  the loss/model path;
- **patch assembly / `AssemblySpec`:** spatial placement and overlap
  normalization under immutable geometry, support, and weighting choices;
- **scale estimation / `CalibrationSpec`:** measured-count, scale-contract, and
  probe-gauge-dependent coefficient estimation, including VarPro;
- **output / `OutputSpec`:** crop, gauge presentation, channel/mode semantics,
  and diagnostics after assembly/calibration;
- **reconstruction policy:** a small versioned composition record referencing
  those separate specifications. Its identity is distinct from the normative
  `scale_contract_version`; selecting a scaling profile must not implicitly
  select a stitching algorithm.

Training and inference may share value objects and reference math, but neither
is required to call the other's orchestration path.

## 5. Roadmap slices

### Slice 1 — Authority and current-state reset

**Status:** Complete (2026-07-15). The revision, routing, disposition, stale-text,
and added-link checks passed.

**Outcome:** One current status authority, no executable historical queue, and no
standing task-local coordination hold promoted into policy.

**Included work:**

- route `docs/index.md` and `docs/plans/README.md` here;
- give the old phase/pipeline plans dated disposition headers;
- remove stale claims that CI/parity exclusions are mere CPU flakes;
- distinguish affected-contract evidence, the checked-in CI harness, and
  optional diagnostics;
- replace mutable line-number/LOC anchors with paths and symbols in future task
  plans.

**Gate:** Links resolve; historical documents identify this file as the only live
status authority; no machine-readable refactor state exists that needs updating.

### Slice 2 — Legacy-state and fail-loud migrations

**Status:** Pending. The warn-mode seal exists; 2A is the first bounded
candidate. Lifecycle containment and the named failure paths remain pending.

**Outcome:** The legacy parameter module is framework-neutral, supported
entrypoints bound its state lifecycle explicitly, and named failures cannot
silently change the scientific algorithm or pretend a model loaded successfully.

**Work:**

**2A — Framework-neutral parameter foundation**

1. Make `ptycho.params` TensorFlow-free without moving the global dictionary into
   a new neutral-core package.
2. Inventory the supported production entrypoints and every mutation form they
   use, including `set()`, direct item assignment, `.update()`,
   `update_legacy_dict`, and bundle restoration. Preserve the externally required
   dictionary object/API and load-time overwrite side effect.

**2B — Lifecycle containment**

3. Choose a containment mechanism only after the inventory. Its contract must
   state nested-context behavior, exception restoration, dictionary object
   identity, and process/worker/DDP boundaries; a `MutableMapping` replacement
   and a scoped populate/restore context are not presumed equivalent. In
   particular, archive load must restore the archived flat state before model
   construction and leave it observable for the returned model's lifetime, or
   use the documented alternative hook; it cannot immediately restore the
   caller's prior dictionary behind that model.
4. Migrate the inventoried entrypoints through an explicit compatibility adapter
   without requiring unrelated tests or imports to initialize TensorFlow.
5. Add test and temporary-entrypoint isolation that restores already-imported
   singleton/global state without rebinding a dictionary behind modules that
   retained its object reference. This isolation does not supersede the public
   post-load state requirement above.

**2C — Independent fail-loud corrections**

6. Migrate each specifically identified broad handler independently: handlers
   that return `None`, retain fresh weights after a failed load, or substitute
   mean reassembly after a TensorFlow reconstruction failure. Partial results
   must be an explicit caller-selected policy where genuinely supported. A load
   path must never report success or return a model after falling back to fresh
   weights, permissive state loading, or today's structural defaults.

Generated-data identity and scaling metadata remain owned by
`SimulationConfig` and the normative scaling/data contracts; they are not a
legacy-lifecycle deliverable.

**Gate:** Each sub-slice has its own claim-matched gate. 2A proves a TF-free
import and preserves the dictionary API; 2B covers the named entrypoint inventory
and its nested/exception/process semantics, including the required post-load
`params.cfg` state; each 2C correction has its own focused falsifier and cannot
return a successful fresh/default model. Unselected mutation routes or fallbacks
do not become a repository-wide completion sweep.

### Slice 3 — Configuration and model identity

**Status:** Partial. `SimulationConfig` ownership and its bounded flat/nested
caller correction are implemented. Structural field ownership, model schema,
and artifact migration remain pending.

**Outcome:** Every field has one semantic owner, model bundles carry enough
versioned structure to rebuild themselves, and legacy translation is explicit.

**Work:**

**3A — Field ownership and structural-side-channel removal**

1. **Complete (2026-07-15):** Correct the recorded flat/nested simulation caller
   through the established compatibility boundary before further changes to that
   same path.
2. Publish an explicit field-ownership and mapping table covering
   `SimulationConfig`, scientific/data contracts, the structural model owner,
   training configuration, runtime execution configuration, and the legacy
   bridge.
3. Move Hybrid/FFNO/Spectral topology fields out of the normatively runtime-only
   `PyTorchExecutionConfig` into the structural model owner. Treat the existing
   public fields as deprecated one-way input aliases during the migration: map
   them explicitly, reject conflicting old/new values, and stop accepting the
   old location only after supported callers, artifacts, and the governing
   execution-config contract have moved.
4. Replace permissive or name-reflective semantic guessing with closed
   constructors, declared mappings, unknown-field rejection, and intentional
   compatibility aliases.

**3B — Versioned structural model schema**

5. Introduce a versioned structural model specification (name illustrative) that
   owns generator topology, structural physics/output choices, construction, and
   rebuild identity for Torch. Derive it through a closed adapter from canonical
   `ModelConfig` plus declared Torch extensions; do not promote it into a second
   public configuration handshake. It must not own Lightning wrapping,
   optimizer/scheduler state, devices, logging, or data loading. An application
   factory composes the separate model, data/scientific, training, and runtime
   sections into `PtychoPINN_Lightning`.

**3C — Explicit artifact upgrade path**

6. Declare the supported artifact formats and schema eras for Lightning `.ckpt`
   hyperparameters, sidecar config JSON, and `wts.h5.zip` bundles. Inspect the
   root manifest's backend/version before model construction; unsupported
   cross-backend loads fail descriptively.
7. Implement a pure decode-old/upgrade-to-current path that normalizes each
   declared era into one current Torch model specification plus its required data
   contract snapshot. Never infer unknown structure from today's defaults.
   Construct the exact module, apply only versioned deterministic key migrations,
   strict-load state, restore required CI statistics and `params.cfg`, and fail
   before returning when any step is unsupported.
8. Dual-write the new versioned identity while required old readers remain
   supported, then fold existing sidecar identity into that schema rather than
   retaining competing sources of truth. New PyTorch bundles preserve the
   external `wts.h5.zip` contract: root manifest backend/version, logical
   `autoencoder` and `diffraction_to_obj` roles, Lightning checkpoint payloads,
   and serializable hyperparameters sufficient for state-free reload. Preserve
   the CONFIG-001 load-time legacy update required by the external API.
9. Fail closed with the exact unsupported schema/version and artifact format when
   no declared migration exists; do not infer compatibility from current
   dataclass fields, use `strict=False`, return fresh weights, or ask for
   regeneration when a supported deterministic upgrade is defined.

**3D — Construction consolidation**

10. Consolidate duplicate Torch architecture dispatch only after 3A–3C establish
   one structural owner and the declared artifact support matrix passes.

**Gate:** Field mappings cover the declared ownership table; runtime config no
longer changes graph topology; every artifact era in the declared support matrix
rebuilds the exact Torch structure, strict-loads its state, restores governed
configuration/statistics, and reloads state-free through the current structural
model specification—or fails before returning with its declared
unsupported-version error. Current bundles retain the required archive name,
backend tag, and two logical roles. Simulation-owned fields do not leak into
model or runtime ownership. An informal "representative" checkpoint sample is
not a support matrix.

### Slice 4 — Geometry and neutral data records

**Status:** Pending. Later work aligned both main local-offset paths to the same
semantic sign, invalidating the original sign-divergence premise.

**Outcome:** One minimal framework-neutral acquisition record crosses a real
producer/consumer boundary, and geometry types are introduced only where that
migration proves they are independently useful.

**Work:**

1. Extract the smallest framework-neutral acquisition/data record needed by one
   named producer/consumer pair currently coupled through `RawData`. Keep
   TensorFlow loading and transformation operations behind an adapter. This is
   an internal record, not a replacement public Ptychodus data surface.
2. Prove the record's destination imports without TensorFlow or Torch and round
   trip the selected pair before moving another consumer.
3. Introduce immutable geometry values incrementally as that migration needs
   independently reusable concepts; do not predeclare a comprehensive coordinate,
   origin, units, canvas, support, crop, and patch-layout type system.
4. Preserve the normative `(x, y)` and `local_offset_sign = -1` semantics; do not
   expose an unrestricted sign knob that permits invalid conventions.
5. Treat transpose/order differences as explicit layout adapters, not competing
   scientific conventions, and keep compatibility adapters at package boundaries.
   Until external specs change atomically, those adapters preserve
   `RawData.generate_grouped_data`, `PtychoDataContainer`, the public grouped
   keys and exact shapes, row-major channel mapping, `(x, y)` pixel convention,
   and `local_offset_sign = -1`.

**Gate:** The selected producer/consumer round trip preserves exact shape/index
mappings, public keys/surfaces, and registered-tolerance numerical behavior;
importing its neutral record does not load either framework. No unselected
geometry abstraction is a gate for that migration.

### Slice 5 — Assembly and reconstruction policies

**Status:** Pending architecturally; substantial focused coverage and an explicit
inference route resolver already exist and must be reused.

**Outcome:** Training assembly, spatial stitching, calibration, and output
handling are separate, composable operations with no physics-changing fallback
or ambiguity between reconstruction-policy and scaling-profile identity.

**Work:**

1. Inventory current training-forward, TF reference, uniform inference,
   probe-weighted inference, VarPro, and grid-tiling paths by scientific contract.
2. Establish independent oracles for placement/denominator math and for scale
   estimation before changing production routing. Do not freeze a current output
   merely because it exists.
3. Introduce separate internal ports and immutable specifications for patch
   assembly, calibration/scale estimation, and output crop/gauge/diagnostics.
   Keep training-forward assembly a distinct differentiable model-spec port;
   inference reconstruction policy must not alter its gradients implicitly.
4. If named reconstruction policies are useful, make them versioned composition
   records that reference those specifications. A reconstruction-policy identity
   is not `scale_contract_version`, and no scaling profile implicitly selects
   assembly or output behavior.
5. Preserve the landed uniform-versus-barycentric route decision unless a
   governed behavior migration explicitly changes it.
6. Replace Torch runtime calls into TF reconstruction only when the native path
   satisfies the same contract and missing physical probes fail closed where the
   profile requires them. Preserve the externally specified TF
   `reassemble_position` route until native Torch parity is established, and keep
   public `middle_trim`/`pad_eval` inputs as compatibility aliases until their
   governing execution-config contract migrates atomically.

**Gate:** Each selected named route passes the oracle for the specification it
changes plus affected reload/inference selectors. A visual/end-to-end check is a
gate only when the active acceptance claim covers rendered reconstruction
behavior; unselected routes do not create a repository-wide migration gate.

### Slice 6 — Workflow layering and backend adapters

**Status:** Pending. The package grid-lines workflow remains large and has one
runtime import from `scripts.studies.invocation_logging`.

**Outcome:** Scripts orchestrate; package modules expose cohesive application
services; backend differences are explicit without a plugin framework.

**Work:**

1. Decompose grid-lines behavior by cohesion—simulation/data preparation,
   reconstruction/evaluation, artifact rendering, and orchestration—while
   retaining compatibility exports during migration.
2. Move invocation/provenance utilities below the package boundary or inject
   them from scripts; eliminate package imports from `scripts/`.
3. Split a Torch workflow use case only when a selected migration establishes a
   stable DTO/port boundary; do not make subjective whole-module cohesion a
   repository-wide task or mechanically expose every internal step.
4. Prefer explicit two-backend construction/dispatch. Introduce narrow trainer
   and inference-session protocols only where they reduce an observed coupling;
   do not add global registration machinery.
5. Extract only small framework-neutral metric primitives and artifact helpers.
   Dataset/study-specific alignment, masks, aggregation, and control flow remain
   with their studies.

**Gate:** The named package-to-`scripts` import edge is removed, and each selected
use-case split preserves its public signature or follows an explicit migration.
Affected workflow smokes are run only when needed to support that bounded
acceptance claim; unselected modules are not part of the gate.

### Slice 7 — Compatibility migrations and independent hygiene

**Status:** Opportunistic. This slice is not a prerequisite for unrelated
architecture work.

**Outcome:** Compatibility switches and obsolete forks are retired through
evidence-based migrations rather than repository-search assumptions.

**Work:**

- Replace `object_big` with separate versioned axes for output/object layout,
  merge strategy, and support/canvas geometry. This depends on Slices 3 and 5 and
  must preserve checkpoint interpretation explicitly.
- Retire `ptycho_torch.api` only after the external Ptychodus API contract,
  supported replacement surface, consumers, tests, and migration guidance change
  atomically.
- Delete dead reassembly forks, obsolete loaders/solvers, and abandoned study
  helpers only after verification that notebooks, external contracts, and live
  entrypoints do not depend on them.
- Extract shared study utilities only for genuinely common mechanics such as
  invocation provenance, artifact writing, or config resolution. Do not build a
  generic scientific study framework in the Torch model package.

**Gate:** Each cleanup or compatibility migration has its own affected-contract
evidence. No global cleanup quota or residual sweep is a roadmap completion gate.

## 6. Dependency and concurrency model

```text
Slice 1 (authority reset)
   ├── Slice 2A (TF-free params) ──> Slice 2B (named-entrypoint lifecycle)
   │       └── Slice 2C corrections may proceed independently when they do not
   │           overlap the lifecycle implementation.
   ├── Slice 3A (field ownership/runtime side channels)
   │       └── Slice 3B (model schema) ──> 3C (artifact upgrades) ──> 3D (dispatch)
   └── Slice 4A (one neutral acquisition record)

Slice 5 (assembly/calibration/output) starts from Slice 3A plus only the geometry
contracts proved necessary by its selected route. Selected Slice 6 application
splits follow the stable ports produced by Slices 4/5; the existing
package-to-scripts import edge may be removed independently.

Slice 7 hygiene may run independently except:
  object_big migration requires Slices 3 + 5;
  API retirement requires its external-contract migration;
  deletion waits only on the exact consumers/contracts it affects.
```

The bounded simulation flat/nested correction is implemented, but its post-fix
repository-wide gate remains pending and blocks the same grid-lines/config path.
With Slice 1 complete, Slice 2A, non-overlapping Slice 3A ownership
analysis/migration, and Slice 4A are dependency-eligible. Do not start 3B
model-schema construction before 3A has removed the structural runtime side
channel, and do not consolidate dispatch before the 3C artifact support matrix
is operational.

Before editing shared hot files, check current working-tree ownership and active
executors. This is a task-local collision preflight, not a standing hold tied to
a historical initiative.

## 7. Feasibility prerequisites

The following claims require a focused proof before an implementation plan may
treat them as available:

| Proposed capability | Required proof |
|---|---|
| TF-free neutral data record/core values | Import smoke demonstrating neither TensorFlow nor Torch is loaded, plus one producer/consumer round trip |
| Controlled legacy-state lifecycle | The declared entrypoint/mutation inventory is covered, dictionary identity and required `.update()` behavior are preserved, and nested/exception/process semantics are explicit |
| Structural-field ownership migration | Old and new input locations dual-read identically, conflicts fail closed, and runtime-only configuration no longer changes topology |
| Versioned Torch model construction | Every `.ckpt`, sidecar-JSON, and `wts.h5.zip` era in the declared support matrix decodes without current-default inference, reconstructs the exact structure, strict-loads state, restores governed metadata, or raises its declared unsupported-version error before returning |
| Native Torch replacement for TF reconstruction | Independent placement/denominator/scale oracle and governed profile parity |
| Public workflow or API compatibility | Minimal executable consumer fixture against the normative external signature and result semantics |
| Replacement of `object_big` | Versioned old-to-new mapping across construction, forward merge, persistence, and reload |

A failed feasibility proof changes the design or leaves the capability as an
explicit prerequisite; it must not be papered over by a compatibility fallback
that changes results silently.

## 8. Evidence policy

For each slice or bounded task:

1. Identify the affected governing contract and the claim being made.
2. Run the smallest fresh selector set capable of falsifying that complete claim.
3. For numerical parity, preserve fixture identity and require the exact named
   selector at its registered tolerance; shapes and index mappings remain exact.
   Do not require byte-identical floating-point output unless a governing
   contract explicitly does.
4. Run cross-backend parity only for changes affecting shared physics, scale,
   coordinates, layouts, or reconstruction semantics.
5. Run an end-to-end workflow or visual check only when the acceptance claim
   includes workflow completion or rendered reconstruction behavior.
6. Run the checked-in CI harness only when a bounded slice's acceptance claim
   spans that harness, and name the exact command in that slice's implementation
   plan. There is no standing "appropriate boundary" CI gate.
7. A **major chunk** is a lettered sub-slice, or a separately gated production
   migration within a slice. After its focused gates pass and before it is marked
   complete or receives its final commit, run the comprehensive repository suite
   on the exact candidate tree:

   ```bash
   python -m pytest tests/ -q
   ```

   This includes every unit and integration test collectable in the current
   environment; declared environment-dependent skips remain skips. The full
   suite is not required after every small subtask, documentation-only edit, or
   intermediate commit.
8. Classify every full-suite failure against the chunk's base tree and affected
   contracts. A failure reproduced on the base is reported but does not create
   unrelated repair work. A failure introduced by or causally attributable to
   the chunk is a regression: fix it before completion, rerun its focused
   falsifier, then rerun the comprehensive suite until no attributable regression
   remains.
9. Classify other supplemental failures against current requirements. A
   supplemental check does not become a new gate merely because it was run.
10. Retain durable logs only when the active plan or testing contract requires
    them.

CI exclusions in `ci/known_failures.txt` and `ci/collect_ignores.txt` are not
passing evidence. Audit an exclusion with its exact selector and remove it only
after fresh evidence shows the governed behavior is green.

## 9. Completion definition

Completion is bounded to the migration surfaces named by this roadmap. The
initiative is complete when:

- `ptycho.params` is TensorFlow-free and legacy state is bounded for the
  production entrypoints enumerated by Slice 2; unenumerated future callers do
  not retroactively expand the gate, while supported bundle loads retain the
  required `params.cfg` overwrite for the returned model's lifetime;
- simulation, structural model, training, runtime, and legacy-bridge fields in
  the Slice 3 ownership table have one owner, and runtime-only configuration no
  longer changes graph topology except through explicitly declared deprecated
  one-way aliases whose conflicts fail closed;
- every artifact era declared in the Slice 3 support matrix rebuilds state-free
  through the versioned Torch structural model specification with strict state
  loading, or fails before returning with its declared unsupported-version
  error; current PyTorch bundles preserve `wts.h5.zip`, the backend tag, both
  logical model roles, and serializable reload identity;
- the producer/consumer route selected by Slice 4 uses a framework-neutral
  acquisition record with explicit layout adapters while its public grouped
  data keys, shapes, and model I/O contract remain compatible;
- the production reconstruction routes frozen in Slice 5's approved inventory
  use distinct training assembly, inference assembly, calibration, and output
  specifications, retain declared public compatibility aliases, and do not
  silently fall back after a native-route failure;
- the named package-to-`scripts` import edge is removed and each application
  split explicitly selected under Slice 6 preserves its contract;
- every completed major chunk has a fresh `python -m pytest tests/ -q` result on
  its final candidate tree and no unresolved regression attributable to that
  chunk.

Slice 7's unrelated compatibility flags and hygiene items remain opportunistic
unless one is explicitly required by a named migration above. No requirement
that every workflow module satisfy a subjective cohesion judgment, and no
arbitrary module-size target, full-repository cleanup sweep, review loop, or
supplemental test count, is part of this completion definition.

## 10. Historical mapping

| Historical source | Current owner |
|---|---|
| Phase 0 cleanup | Slice 7, only for independently justified residual hygiene |
| Phase 1 safety net | Slice 2 for legacy lifecycle/fail-loud work; generated-data identity remains with `SimulationConfig` and normative data/scaling contracts |
| Phase 2 config/geometry/metrics/scripts consolidation | Slices 3, 4, 6, and 7 |
| Phase 3 neutral core | Slice 4's framework-neutral records/value objects only |
| Phase 3 `Reassembler` | Slice 5's separate assembly, calibration, output, and composition contracts |
| Phase 3 `TrainingBackend` | Slice 6's narrow trainer/inference adapters |
| Phase 3 `object_big` replacement | Slice 7 after Slices 3 and 5 |
| 2026-07-06 pipeline consolidation | Landed route retained in Slice 5; remaining dead forks/bugs classified under Slices 5 or 7 |

Historical commit identifiers and execution reports remain useful provenance,
but current source symbols and normative contracts determine future task scope.
