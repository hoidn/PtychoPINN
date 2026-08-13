# Subtractive Refactoring Roadmap

> **Status:** Complete. Approved 2026-08-12; completed 2026-08-13.
>
> **Scope:** Governs refactoring sequence and status on `fno-stable`,
> `refactor-internal`, and `refactor`. Implementation starts on `fno-stable`.
>
> **Authority:** Normative behavior remains owned by `specs/` and
> `docs/specs/`. This roadmap owns only refactoring scope, dependencies, and
> status.
>
> **Supersedes:** the 2026-07-07 refactoring roadmap on branches where that
> historical record exists. Its completed work remains historical fact; its
> unfinished items are not carried forward unless the 2026-08-12 audit
> independently justifies them below.

## 1. Objective

Reduce the number of production paths that independently load, group, batch,
train, reload, and reconstruct the same ptychography data. Delete proven
orphans first; consolidate live paths only after their distinct scientific and
external contracts are explicit.

The target is one shared Torch training service, not one implementation of
everything:

```text
ptycho_synthetic      ptycho_train      ptycho_study adapter
       └──────────────────┬──────────────────┘
                          ▼
                 one NPZ decoder/selector
                          ▼
                 one grouping/index plan
                          ▼
                 one Torch batch contract
                          ▼
              ┌───────────┴───────────┐
              ▼                       ▼
       RAM materializer        TensorDict mmap
              └───────────┬───────────┘
                          ▼
                one loader/sampler path
                          ▼
                one Lightning train/save
```

The RAM and mmap rails remain distinct because they solve different storage
problems. The duplicated scientific transformations before and around them are
the consolidation target.

## 2. Audit basis and reset

The fresh audit used these branch tips:

| Branch | Audited commit | Role |
| --- | --- | --- |
| `fno-stable` | `acd6c2aa6129ca984c5213238b509db0bcabcc4e` | Initial implementation target |
| `refactor-internal` | `56db36ef5ea4e355f64e8b3a39a1e47e1609cf53` | Internal feature superset requiring semantic adaptation |
| `refactor` | `f2523bfb23a0f8a0c79c12c14a987dd59777cfdf` | Public line with older runner/training surfaces |

The audit found:

| Surface | Evidence | Decision |
| --- | --- | --- |
| Packaging | Unrestricted namespace discovery identifies 3,108 package candidates; the regenerated source manifest contains 2,240 files, including docs, tests, outputs, notebooks, archive paths, `loaders`, and top-level `torch` | Establish an explicit wheel boundary first and remove stale `setup.py` |
| Proven orphans | Approximately 5.0 kLOC across alternate reassembly forks, `beta_modules`, shadowed/dead modules, loaders, and a deprecated wrapper have no live production callers | Delete in Phase 0 |
| NPZ ingestion | `RawData.from_file`, workflow `load_data`, `PtychoDataset`, and compatibility bridges decode or reshape overlapping NPZ contracts | Choose one decoder before deleting live loaders |
| Grouping | `RawData.generate_grouped_data` performs sample-then-group selection; `patch_generator.group_coords` implements bounds, nearest/min-distance/quadrant policies, coverage, and center IDs | Preserve named policies in one index-plan result; do not choose either implementation by accident |
| Storage | `MemmapDatasetBridge` opens NPZ with NumPy mmap syntax but casts/materializes before delegating; `PtychoDataset` owns the real TensorDict mmap rail | Do not describe the bridge as the mmap implementation |
| Training | `train_lightning_only.py`, native `ptycho_torch.train`, and shared workflow components duplicate parts of data/module/trainer construction | Converge only after the canonical batch rail exists |
| Grid-lines runner | `grid_lines_torch_runner.py` is 3.1–3.4 kLOC and still has branch-dependent study callers and a named normative contract | Retire after public synthetic/study parity and spec migration, not by wrapping it |
| Public surfaces | `ptycho_torch.api`, native Torch CLIs, persistence eras, and several unused exported loader classes are externally visible or contract-sensitive | Treat as conditional migrations, not immediate deletions |

This evidence replaces the stale roadmap queue. Line counts rank opportunities;
they are not completion quotas.

## 3. Decisions

1. **Delete before abstracting.** A zero-caller path is removed. It is not put
   behind a new interface.
2. **One owner per transformation.** NPZ decoding, selection, grouping,
   normalization, batch conversion, sampling, and training each get one owner.
3. **Storage rails are not duplicate behavior.** RAM remains for
   already-materialized/programmatic/TensorFlow use; file-backed Torch and DDP
   use TensorDict mmap.
4. **Grouping policy is explicit data.** The future index-plan result must name
   the selected policy and retain center indices, neighbor indices, coverage,
   object/experiment partitions, and seeded selection needed by current callers.
5. **Keep backend-native computation.** TensorFlow and Torch kernels,
   serialization, and reconstruction implementations remain separate where
   their runtime or artifact contracts differ.
6. **Keep distinct reconstruction policies.** Tiled, barycentric, and VarPro
   reconstruction are not collapsed into one generic reassembler.
7. **Keep `ptycho_study` where present.** It is the multi-arm composition
   layer, not another trainer. Do not reintroduce it on `refactor-internal`,
   where that surface was intentionally removed.
8. **No compatibility fiction.** An exported name or normative entry point is
   removed only after its consumer and contract decision is explicit.
9. **Propagate semantics, not commits.** Each branch receives the same outcome
   adapted to its current callers and contracts; blind cherry-picks are not the
   propagation strategy.

## 4. Ordered roadmap

| Phase | Outcome | Status | Dependency |
| --- | --- | --- | --- |
| 0 | Bound the installable package and delete proven orphans | **Complete on all three branches** | [Execution plan and evidence](2026-08-12-fno-stable-orphan-removal-plan.md) |
| 1 | One NPZ decoder/selector and one backend-neutral grouping/index plan | **Complete on all three branches** | Phase 0; policy/fixture inventory |
| 2 | One Torch batch contract and loader/sampler path over RAM and mmap materializers | **Complete on all three branches** | Phase 1 |
| 3 | One shared Torch training/save service; retire duplicate trainer construction | **Complete on all three branches** | Phase 2 |
| 4 | Retire `grid_lines_torch_runner.py` after maintained callers move to public workflows | **Complete on all three branches** | Phase 3 plus public parity and spec migration |
| 5 | Evaluate contract-sensitive and lower-value parallel paths | **Evaluated; none promoted** | Named prerequisite per item |

### Phase 0 — Package boundary and proven orphan removal

The first tranche is intentionally behavior-neutral:

- constrain package discovery to `ptycho`, `ptycho_torch`, the retained legacy
  `frc` namespace, and the three installed
  `scripts.{training,inference,simulation}` namespaces;
- preserve each branch's installed commands while keeping repository datasets
  out of the wheel: four commands on `fno-stable`/`refactor`, and the existing
  three-command boundary on `refactor-internal`; the tracked Run1084 fixture
  remains at its existing repo-relative
  `datasets/Run1084_recon3_postPC_shrunk_3.npz` path;
- declare the existing POLICY-001 minimum as `torch>=2.2` in the surviving
  packaging authority;
- delete stale `setup.py` so `pyproject.toml` is the sole packaging authority;
- delete:
  - `ptycho_torch/reassembly_alpha.py`;
  - `ptycho_torch/reassembly_beta.py`;
  - `ptycho_torch/beta_modules/`;
  - shadowed `ptycho_torch/datagen.py`;
  - `ptycho_torch/model_finetuner_modified.py`;
  - top-level `loaders/`;
  - both duplicate top-level `torch/tf_helper.py` and
    `torch/tests/tf_helper.py` files plus the permanently skipped
    `tests/torch/test_tf_helper.py` that targets the absent relative module;
  - `scripts/simulation/run_with_synthetic_lines.py` and its wrapper-only
    tests;
  - deprecated, zero-caller
    `ptycho.workflows.components.load_and_prepare_data`.

This phase does **not** delete `MemmapDatasetBridge`,
`PtychoDataset.from_np`, `InMemoryPtychoDataModule`, or dead-looking methods in
canonical `reassembly.py`; their public/exported-name status must be decided in
Phase 2 or 5.

### Phase 1 — Canonical ingestion and grouping plan

Produce one decoded acquisition record and a separate, deterministic selection
and grouping result. The design must first inventory current callers and freeze
the behaviors they actually select:

- accepted NPZ keys and axis compatibility;
- amplitude versus intensity identity;
- coordinate transforms and object/experiment indices;
- seeded subsampling and train/test selection;
- `random`/`sequential` sample-then-group behavior;
- `Nearest`, `Min_dist`, and `4_quadrant` grouping;
- bounds filtering, complete reconstruction coverage, centers, neighbor IDs,
  and K-choose-C oversampling;
- normalization ownership.

The phase may then replace duplicate parsers and grouping implementations with
one decoder and one index-plan result. It must not combine decoding, grouping,
normalization, and storage into another all-purpose dataset class.

Minimum evidence is deterministic record parity for representative standalone
and grouped NPZ fixtures plus policy-specific grouping parity. Scientific
differences found by that comparison require an approved design decision; they
are not silently normalized away.

### Phase 2 — Batch, storage, and sampler convergence

The implementation and integration gates are complete on all three branches.
The selected design builds the existing normative Torch
batch tuple `(tensor_dict, probe, probe_scaling)` from the Phase 1 record and
grouping plan. One conversion path must own channel/layout conversion, named CI
fields, experiment identity, and probe expansion. It then feeds either:

- a RAM materializer for in-process arrays; or
- `PtychoDataset`'s TensorDict mmap store for file-backed training and DDP.

One loader factory owns batch size, workers, shuffling, and explicit sampler
selection. Lightning owns default DDP sharding; a caller must not pre-shard
data and then apply a distributed sampler again.

The completed implementation removed these redundant surfaces:

- `MemmapDatasetBridge`;
- `PtychoDataset.from_np`;
- `InMemoryPtychoDataModule`;
- `RawDataTorch`.

Minimum evidence compares named batches from RAM and mmap for the same grouping
plan, includes one multi-experiment case, and includes a CPU DDP smoke where the
environment supports it.

### Phase 3 — Training entry-point convergence

Route `ptycho_synthetic`, `ptycho_train`, and study adapters through one service
that receives resolved model/runtime records and a selected data rail, then
constructs the Lightning module, callbacks, trainer, fit call, and saved bundle
once.

Migrate the live ablation/VarPro callers of
`ptycho_torch/train_lightning_only.py`, preserve only demonstrated checkpoint
and DDP controls, then delete that module and its redundant Lightning
DataModule. Reduce `ptycho_torch.train` to a translation/compatibility shim over
the same service. Removing the native CLI itself is conditional because
`specs/ptychodus_api_spec.md` currently names it.

Minimum evidence covers RAM and mmap training smoke paths, strict saved-bundle
reload, configured optimizer/scheduler propagation, and one supported DDP
invocation.

### Phase 4 — Grid-lines Torch runner retirement

`scripts/studies/grid_lines_torch_runner.py` is retired only after all of these
are true:

1. maintained callers are separated from historical recipes;
2. active dataset generation and single-run training are expressible through
   parameterized `ptycho_synthetic` configuration;
3. multi-arm studies use `ptycho_study` adapters over the shared training
   service;
4. strict bundle reload, barycentric/tiled reconstruction, VarPro, CI probe
   identity, supervised labels, and scaled reconstruction retain their selected
   public behavior;
5. paper-row collation and historical comparison output live in study-level
   adapters rather than the training service;
6. maintained YAML/TOML and commands are migrated, while obsolete historical
   adapters/tests are deleted rather than ported; and
7. `docs/specs/spec-ptycho-interfaces.md` no longer assigns normative producer
   status to the runner or its dict-container-only path.

The frozen `cnn-lines-ci-v1` recipe may remain reproducible as configuration;
its runner implementation does not remain for that reason.

Minimum evidence is one current GS2 synthetic integration, one representative
CI/grid-lines replay, strict reload/reassembly checks, and zero live imports or
subprocess calls to the retired runner.

### Phase 5 — Conditional simplifications

These are roadmap candidates, not an executable backlog:

| Candidate | Required reason to proceed |
| --- | --- |
| Modern strict bundle loading versus `model_manager.py` legacy reconstruction | A versioned artifact-era inventory shows a shared decoder can preserve strict loading and declared legacy support |
| Installed versus native inference CLIs | Public command and `specs/ptychodus_api_spec.md` consumers can use one dispatcher without losing return/artifact behavior |
| Generator wrapper classes, registry, architecture literal, and application factory | Construction parity proves one dispatch owner can preserve every supported architecture and state signature |
| Dead methods in canonical `reassembly.py` | Public-name and dynamic-caller audit proves they are not a supported extension surface |
| `ptycho_torch.api` (about 2.6 kLOC) | External consumers and the Ptychodus API contract explicitly migrate away from it |
| TensorFlow `grid_lines_workflow.py` decomposition | Duplicate application behavior remains after the Torch service converges and a bounded TF design preserves its contract |

Tiny duplicated helpers such as JSON serializers or equality checks do not
justify a cross-codebase utility campaign. Consolidate them only when a live
owner is already being changed.

The 2026-08-13 closeout evaluation promoted none of these candidates. Each
still requires its named evidence and a separately approved implementation
plan.

## 5. Branch propagation

Each completed phase is implemented and verified on `fno-stable` first. Then:

| Branch | Adaptation rule |
| --- | --- |
| `refactor-internal` | Preserve internal-only architecture and synthetic/study behavior. Documentation exclusions used for public branches do not apply. |
| `refactor` | Reconcile its older `scripts/training/train.py`, smaller grid-runner caller set, and leaner docs rather than importing `fno-stable` study history. Pull its upstream before integration. |

For every propagation:

1. re-audit live callers and governing contracts at the destination tip;
2. reproduce the phase's outcome with the smallest branch-native diff;
3. omit source-only features that do not exist on the destination rather than
   adding them as migration scaffolding;
4. run the destination's affected selectors; and
5. update this roadmap's status only after all intended branches have either
   landed the outcome or recorded a concrete branch-specific non-applicability.

There is no machine-readable refactoring manifest. This file is the live status
surface; `docs/index.md` and `plans/README.md` route to it.

## 6. Evidence and completion

Every phase gets a bounded implementation plan before code changes. Evidence is
claim-matched:

- deletion requires caller/import absence plus tests for the surviving public
  path;
- data consolidation requires record/group/batch parity, not only unit tests of
  a new class;
- DDP claims require an actual distributed smoke;
- runner retirement requires public end-to-end replacement evidence;
- scientific quality is rerun only when the phase can change scientific output;
- a repository-wide suite is required only when the affected contract or the
  phase's accepted plan makes that breadth necessary.

The roadmap is complete when Phase 4 has landed on all applicable branches,
there is one shared Torch ingestion-to-training route with two explicit storage
rails, the grid-lines Torch runner is gone, and no unresolved live duplicate
path remains within Phases 0–4. Phase 5 candidates do not block completion
unless a later approved plan promotes one into the required path.
