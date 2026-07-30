# Native Torch CLI Mmap Ingestion Design

## Metadata And Status

- ID: `TORCH-NATIVE-MMAP-001`
- Title: Native Torch CLI mmap ingestion
- Status: approved
- Owner: PtychoPINN maintainers
- Date: 2026-07-29
- Approval source: User approval in the 2026-07-29 Codex design discussion
- Related implementation: `ptycho_torch/train.py`,
  `ptycho_torch/dataloader.py`, and
  `ptycho_torch/workflows/components.py`

## Context And Authority

The native training CLI currently resolves a complete `TrainingPayload`, loads
the selected NPZ files through `RawData.from_file`, and passes `RawData` into
`run_cdi_example_torch`. The workflow consequently selects its in-memory
`PtychoDataContainerTorch` loader branch even though the Torch stack already
has a supported `PtychoDataset` mmap branch.

The governing sources are:

- `docs/specs/spec-ptycho-interfaces.md`, which owns the normative Torch batch
  contract and identifies `PtychoDataset` as the native mmap producer.
- `docs/specs/spec-ptycho-core.md`, which owns standalone-NPZ keys, shapes,
  units, and scaling profiles.
- `docs/workflows/pytorch.md`, which owns the native CLI's user-facing
  invocation and configuration behavior.
- `docs/architecture_torch.md`, which owns the two supported Torch data-path
  distinction and shared workflow boundary.

This design owns only the native CLI's selection and preparation of its
training substrate. It does not replace the normative batch or scientific data
contracts.

## Problem

`python -m ptycho_torch.train` does not currently use the native mmap loader.
It materializes the selected NPZ through `RawData`, groups it in memory, and
then delegates to the shared workflow. The old mmap-capable local training
implementation was removed, but the active CLI had already bypassed it.

A direct switch to `PtychoDataset` is insufficient because:

- `PtychoDataset` consumes every NPZ in a directory, while the CLI accepts one
  exact train file and one optional exact test file.
- its repo-relative default mmap path can collide across runs;
- its writer currently treats `objectGuess` as required even though the
  governing standalone-NPZ contract defines it as optional;
- its current grouping builds the candidates implied by
  `DataConfig.n_subsample` and does not enforce the CLI's
  `--n_images`/`n_groups` count; and
- its DDP prebuilt-data-module branch does not currently consume an explicit
  test mmap as validation data.

## Goals And Non-Goals

### Goals

- Make the native Torch training CLI feed mmap-backed `PtychoDataset` objects
  into `run_cdi_example_torch`.
- Select only the exact NPZ path named for each CLI role.
- Rebuild isolated train and optional test maps inside each run's
  `output_dir`.
- Preserve `--n_images` as the exact number of groups exposed by each CLI
  dataset.
- Preserve explicit test-file validation for single-process and DDP-style
  execution.
- Accept contract-valid standalone NPZ inputs without `objectGuess`.
- Preserve the resolved configuration, shared Lightning workflow, checkpoint,
  and `wts.h5.zip` persistence boundaries.
- Fail clearly instead of silently falling back to the in-memory path.

### Non-Goals

- Changing programmatic callers that intentionally pass `RawData`,
  `RawDataTorch`, or `PtychoDataContainerTorch`.
- Making mmap caches reusable across independent runs.
- Changing bundle, checkpoint, ModelSpec, or configuration schemas.
- Reviving locally duplicated model, DataModule, or Trainer construction in
  `ptycho_torch/train.py`.
- Changing existing uncapped `PtychoDataset` construction when no explicit
  group limit is supplied.

## Decision And Alternatives

### Decision

The native CLI will use a single-file mmap preparation adapter. For each
selected role, the adapter will:

1. create a fresh role workspace under
   `{output_dir}/mmap_workspace/{train|test}`;
2. stage only the selected NPZ through a hard link, with a byte-for-byte copy
   fallback when linking is unavailable;
3. construct `PtychoDataset` with the resolved Torch data, model, and training
   configurations, a role-local mmap path, `remake_map=True`, and the requested
   exact group count;
4. remove the staged link or copy after construction; and
5. return the mmap-backed dataset to the existing shared workflow.

The adapter is supported only on Linux with procfs mounted and accessible at
`/proc/self/fd`, plus descriptor-relative and no-follow filesystem operations.
Before creating `output_dir`, removing a role workspace, or staging a source,
it opens a read-only probe descriptor and proves that its procfs path identifies
the same inode. Missing platform capabilities or inaccessible procfs produce an
actionable terminal error. A pathname-based or in-memory fallback is not
permitted.

The mmap writer will align its input validation with
`docs/specs/spec-ptycho-core.md`: `probeGuess` remains required and
`objectGuess` is optional. When `objectGuess` is absent, the writer leaves its
full-object reference collection empty. In supervised mode, `label` remains
required by that mode; if the optional object reference is absent, the stored
phase-correction factor is `0.0`, so configured phase subtraction is a
deterministic no-op instead of making an optional NPZ field mandatory.

`PtychoDataset` will gain an opt-in group-limit contract. The limit is applied
to the generated group records before mmap allocation, so the map and exposed
dataset length both equal `n_groups`. Existing callers that omit the limit
retain current behavior.

When candidate groups exceed the requested count, selection is without
replacement. Sequential sampling takes the first groups in stable source
order. Random sampling uses `subsample_seed`, falling back to the workflow's
existing seed `42` when the resolved value is absent. When fewer groups exist
than requested, construction fails with a count-bearing diagnostic; it does
not silently reduce the dataset or invent duplicate groups.

The workflow's mmap branch will continue to return ordinary mmap loaders for
non-DDP execution and a prebuilt mmap DataModule for DDP-style execution.
That DataModule will accept an optional validation-map path. An explicit test
map means the full capped train map is training data and the full capped test
map is validation data. Without an explicit test map, the existing
deterministic mmap DDP train/validation split remains unchanged.
For non-DDP execution without an explicit test map, the direct-loader branch
returns no validation loader; it does not construct a loader around `None` or
invent a split.

For the CI scaling profile, validation batches must receive the finalized
training-map statistics. Test-map-derived statistics must not become the
validation authority.

### Alternatives Considered

| Alternative | Advantages | Disadvantages | Why not selected |
|---|---|---|---|
| Teach `PtychoDataset` to accept arbitrary file lists directly | Removes staging | Broadens a core loader API and file-discovery contract beyond this CLI need | Exact single-file staging is already established by Torch inference and study paths |
| Restore the deleted local `PtychoDataModule`/model/Trainer path | Closely resembles old training code | Reintroduces duplicated orchestration and bypasses the current bundle-saving workflow | Conflicts with the shared-workflow boundary |
| Point `PtychoDataset` at the NPZ's parent directory | Very small code change | Silently includes unrelated NPZ files and uses collision-prone cache state | Violates the exact CLI file contract |
| Map all candidates and cap only the DataLoader | Avoids changing map construction | Wastes disk and mmap creation time; DDP must persist a separate subset contract | The requested count should constrain the produced map itself |

## Components And Data Flow

| Component | Responsibility | Source of truth |
|---|---|---|
| Native CLI ingestion adapter | Isolate an exact NPZ, own the run-local workspace, and build a capped `PtychoDataset` | `ptycho_torch.train` ingestion boundary |
| `PtychoDataset` group limiter | Select exactly the requested group records before mmap allocation | Torch loader implementation under the normative batch contract |
| Mmap workflow dispatcher | Select mmap loaders when the supplied object is `PtychoDataset` | `ptycho_torch.workflows.components` |
| Prebuilt mmap DataModule | Reopen maps per DDP rank and bind optional explicit validation data | Lightning/DDP loader boundary |

```text
exact train NPZ
  -> run-local train staging
  -> capped train PtychoDataset mmap
  -> run_cdi_example_torch
  -> mmap Lightning loader or prebuilt DDP DataModule
  -> Trainer.fit
  -> checkpoints and wts.h5.zip

optional exact test NPZ
  -> run-local test staging
  -> capped test PtychoDataset mmap
  -> validation loader using finalized training statistics
```

## Contracts And Invariants

- The public CLI flags and their configuration precedence do not change.
- `--n_images=N` means each constructed CLI dataset exposes exactly `N`
  groups, or construction fails.
- Only the explicitly named NPZ is eligible for each role.
- Absence of the optional `objectGuess` key does not invalidate an otherwise
  contract-valid NPZ. Supervised mode still requires its `label` input, and
  missing object-reference phase correction resolves to the documented `0.0`
  no-op.
- The persistent role-local map, state file, and schema manifest live below
  the run's `output_dir`; the source staging entry is temporary.
- Every yielded batch continues to satisfy
  `(tensor_dict, probe, probe_scaling)` from
  `docs/specs/spec-ptycho-interfaces.md`.
- Explicit test data remains validation data under both direct-loader and
  DDP-style paths.
- CI validation uses immutable statistics finalized from training samples.
- Bundle and checkpoint identity remains derived from the already-resolved
  training payload, not from workspace paths.
- Mmap creation failures propagate. There is no automatic `RawData` fallback.
- Native mmap construction requires Linux with accessible procfs descriptor
  aliases at `/proc/self/fd`; capability failure is detected before output or
  staging mutation and has no pathname fallback.

## Failure And Recovery

| Failure | Required behavior |
|---|---|
| Non-Linux runtime, inaccessible `/proc/self/fd`, or missing descriptor/no-follow capability | Fail before creating or changing `output_dir`, report the Linux/procfs requirement, and do not use a pathname fallback |
| Selected NPZ is missing or invalid | Fail before training with the existing path/data-contract diagnostic |
| Optional `objectGuess` is absent | Continue without a full-object reference; use `0.0` for supervised phase correction |
| Role workspace contains a previous map | Remove only that resolved role workspace and rebuild it |
| Hard linking is unsupported | Copy the exact source file and continue |
| Fewer candidate groups exist than `n_groups` | Fail with requested and available counts |
| Mmap or state creation fails | Remove temporary staging and partial role state, then re-raise |
| Explicit validation map is unavailable to a DDP rank | Fail rather than substituting a train split |
| Mmap schema/profile is incompatible | Preserve the existing rebuild-required failure |

Re-running the same command against the same `output_dir` is safe because the
two role workspaces are fresh-build state. No cross-run cache validity or
fingerprinting promise is made.

## Evidence And Feasibility

The substrate-preserving capability is already exercised by real code:

- `PtychoDataset` constructs and reads a standalone-NPZ TensorDict mmap.
- `_build_dataloaders_from_ptycho_dataset` already selects the mmap and DDP
  branches from the supplied type.
- native inference and study utilities already isolate a single NPZ through
  staging before `PtychoDataset` construction.

The focused feasibility command run before approval was:

```text
python -m pytest \
  tests/torch/test_ptycho_dataset_normalized_amplitude.py \
  tests/torch/test_workflows_components.py \
  -k 'ptycho_dataset_does_not_zero_normalized_amplitude_data or mmap_ddp_loader_uses_resolved_execution' \
  -q
```

Result: `3 passed, 47 deselected`.

The exact pre-allocation group cap, explicit DDP validation map, and
training-statistics propagation are implementation obligations and require
their own RED/GREEN tests before production changes.

## Compatibility And Migration

This is an ingestion-substrate migration for the native CLI. It intentionally
adopts the native mmap loader's grouping, normalization, and batch production
while preserving the CLI's exact group-count and selected-file contracts.
Scientific parity with the former `RawData` grouping algorithm is not claimed.

Existing programmatic input-type dispatch remains valid:

- `PtychoDataset` selects mmap behavior;
- `RawData`, `RawDataTorch`, and `PtychoDataContainerTorch` retain their current
  in-memory behavior.

Existing mmap callers remain uncapped unless they explicitly opt into the new
group limit. Existing DDP mmap callers without a test dataset retain their
deterministic internal split.

Rollback consists of restoring the native CLI's `RawData.from_file` adapter;
no persisted model or configuration format migration is required.

## Verification Strategy

Implementation evidence must cover:

- a RED/GREEN CLI routing test proving `run_cdi_example_torch` receives
  `PtychoDataset`, not `RawData`;
- exact single-file isolation when unrelated NPZ files share the source
  directory;
- exact train and test group counts for random and sequential selection;
- deterministic random selection from the resolved seed;
- fail-closed behavior when the requested count exceeds candidates;
- contract-valid unsupervised and supervised NPZ construction without
  `objectGuess`, including the supervised `0.0` phase-correction behavior;
- optional-test handling for direct loaders;
- explicit test-map use and per-rank reopen behavior for DDP and spawn-style
  strategies;
- CI validation batches receiving finalized training statistics;
- continued checkpoint and bundle persistence through the shared workflow; and
- capability tests proving non-Linux and inaccessible-procfs simulations fail
  before `output_dir` mutation with the documented diagnostic; and
- focused native CLI, mmap loader, workflow-component, and batch-contract
  regression suites.

An end-to-end smoke must confirm that a real native CLI invocation creates a
TensorDict mmap below `output_dir/mmap_workspace`, trains through Lightning,
and emits its normal model artifacts.

## Declarative Acceptance Scenarios

### Native CLI with explicit validation

- **Given** valid train and test NPZ files and `--n_images 64`
- **When** the native Torch training CLI starts
- **Then** separate fresh train and test maps each expose exactly 64 groups
- **And** the shared workflow receives `PtychoDataset` objects
- **And** the test map is used for validation under the selected execution
  strategy.

### Source directory contains unrelated datasets

- **Given** the selected train NPZ shares a directory with other NPZ files
- **When** the native CLI builds its train map
- **Then** only the selected file contributes groups or state.

### Optional object reference is absent

- **Given** a contract-valid NPZ without `objectGuess`
- **When** the native CLI builds its mmap dataset
- **Then** construction succeeds without a full-object reference
- **And** supervised mode, when selected with a valid `label`, records a `0.0`
  phase correction rather than failing on the absent optional key.

### Non-DDP training without explicit validation

- **Given** no test NPZ and a non-DDP execution strategy
- **When** the workflow builds direct mmap loaders
- **Then** it returns the train loader and no validation loader.

### Insufficient candidate groups

- **Given** the selected NPZ produces fewer groups than `--n_images`
- **When** mmap construction applies the exact group count
- **Then** construction fails before training and reports requested and
  available counts.

### Mmap construction failure

- **Given** mmap creation cannot complete
- **When** the ingestion adapter encounters the error
- **Then** it cleans temporary/partial role state and propagates the error
- **And** it does not fall back to in-memory training.

## Success And Stop Criteria

The implementation succeeds when the native CLI always reaches the shared
workflow through `PtychoDataset`, exact file/count and validation semantics are
proven under direct and DDP-style execution, CI statistics retain training
ownership, and the normal persisted model artifacts remain intact.

Stop and revise before rollout if exact pre-allocation capping cannot preserve
the normative batch contract, if DDP cannot reliably reopen both role maps, or
if a real CLI smoke reveals that the mmap path changes bundle identity rather
than only ingestion.

## Documentation Impact

Implementation must update:

- `docs/workflows/pytorch.md` to describe native CLI mmap preparation and
  run-local workspaces;
- `docs/architecture_torch.md` so its training flow distinguishes native CLI
  mmap ingestion from programmatic `RawData` input; and
- relevant test-index content if the repository's generated index workflow
  requires regeneration.

Normative NPZ and batch contracts do not require semantic changes.

## Implementation Handoff

The implementation plan must:

1. establish RED tests for exact capping and CLI type routing;
2. align mmap NPZ validation and writing with optional `objectGuess`;
3. add the opt-in pre-allocation group limit without changing uncapped callers;
4. add isolated run-local single-file preparation;
5. make optional validation safe for direct mmap loaders;
6. extend the DDP prebuilt path for an explicit validation map and training
   statistics authority;
7. switch the native CLI from `RawData` to the new preparation adapter;
8. update the architecture and workflow documentation; and
9. run focused regression and real-CLI smoke evidence.

The implementation may choose private helper names and internal file layout
below each role workspace, but it must not reopen the decisions on fresh
run-local caches, exact group counts, explicit validation, or shared workflow
delegation.
