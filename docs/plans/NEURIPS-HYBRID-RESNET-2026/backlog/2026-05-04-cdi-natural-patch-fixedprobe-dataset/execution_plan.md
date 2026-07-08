# CDI Natural-Patch Fixed-Probe Dataset Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:executing-plans` or `superpowers:subagent-driven-development` to implement this plan task-by-task. Keep this file as the execution authority for the selected backlog item.

**Goal:** Generate and lock `natural_patches128_fixedprobe_v1` as a git-ignored expanded-object CDI dataset with deterministic train/validation/test splits, fixed Run1084 probe lineage, complete manifests, and a small visual contact sheet, without training any model rows or changing the existing `lines128` paper table authority.

**Architecture:** Treat this item as dataset-contract work, not benchmark execution. Reuse the existing Run1084 probe lineage and CDI runner/schema references, add one dedicated natural-patch dataset builder path that can emit durable source/probe/split/simulation manifests plus canonical array payloads under `.artifacts/data/NEURIPS-HYBRID-RESNET-2026/natural_patches128_fixedprobe_v1/`, and prove downstream compatibility either by writing runner-ready grouped NPZs or by emitting a checked adapter contract backed by tests. Split assignment must happen at the source-image level before crop sampling so train/validation/test never share a parent natural image.

**Tech Stack:** PATH `python`, NumPy/Pillow or equivalent image I/O already available in repo dependencies, existing CDI/grid-lines study helpers under `scripts/studies/`, Run1084 probe asset `datasets/Run1084_recon3_postPC_shrunk_3.npz`, Markdown/JSON manifests, git-ignored `.artifacts/data/` outputs, pytest, `compileall`.

---

## Selected Backlog Objective

- Implement backlog item `2026-05-04-cdi-natural-patch-fixedprobe-dataset`.
- Generate the frozen dataset id `natural_patches128_fixedprobe_v1`.
- Keep the dataset bounded to `N=128`, fixed probe, deterministic source-level splits, and at most `10_000` object patches total.
- Produce durable non-code outputs:
  - dataset artifacts under `.artifacts/data/NEURIPS-HYBRID-RESNET-2026/natural_patches128_fixedprobe_v1/`
  - summary at `docs/plans/NEURIPS-HYBRID-RESNET-2026/cdi_natural_patch_fixedprobe_dataset_summary.md`
  - evidence/discoverability updates in `docs/plans/NEURIPS-HYBRID-RESNET-2026/evidence_matrix.md`, `docs/studies/index.md`, and `docs/index.md`

## Scope Boundaries

### In Scope

- Natural-image corpus selection from the first locally available, documentable source that can satisfy provenance and split-isolation requirements.
- Fixed-probe lineage capture from the current `lines128` CDI authority.
- Deterministic source-image assignment, patch extraction, luminance/grayscale handling, intensity-to-object mapping, simulation settings, and emitted manifests.
- Canonical dataset packaging for later expanded-object CDI benchmarking, including either:
  - direct runner-compatible grouped NPZ payloads, or
  - raw split payloads plus an explicit adapter contract proving later consumption without model-semantics drift.
- A small contact sheet showing representative source patches and diffraction samples.

### Explicit Non-Goals

- Do not train Hybrid ResNet, CNN, FNO, FFNO, U-NO, or any other model row in this item.
- Do not update manuscript result tables or `/home/ollie/Documents/neurips/` paper-facing assets.
- Do not rerun or rewrite the existing `lines128` complete table, U-NO extension, or other CDI authorities.
- Do not substitute a synthetic lines/shapes corpus if no usable local natural-image source is available.
- Do not tune crop policy, object encoding, or source selection after inspecting model metrics.
- Do not commit bulky dataset arrays into git.
- Do not modify `ptycho/model.py`, `ptycho/diffsim.py`, or `ptycho/tf_helper.py`.

## Steering, Roadmap, And Policy Constraints

- Steering requires roadmap gates and fairness constraints to remain explicit. This item is a Phase 3 dataset prerequisite only; it must not quietly expand into the later expanded-object benchmark or other optional follow-ups.
- The roadmap authorizes this work at item `3.3h`: create a frozen expanded-object CDI dataset before any expanded-object benchmark rows run.
- The selected backlog item is the execution authority for queue location and scope. Treat `selection_source_path` as provenance only.
- Preserve the current CDI fixed-probe lineage from the `lines128` benchmark unless implementation records a reviewed replacement note before generation. The default expectation is the same Run1084 probe source and preprocessing family used by the current `lines128` authority.
- Preserve apples-to-apples discipline:
  - split at the source-image level so train/validation/test never share a parent image
  - keep one fixed probe contract across all splits
  - keep one fixed object encoding/simulation contract across all splits
  - keep the dataset at `<= 10_000` total objects unless a later roadmap amendment raises the cap
- Long-running generation stays under implementation ownership until terminal success or recoverable failure handling is complete:
  - use `tmux` when the live generation is expected to run longer than an interactive command
  - keep PATH `python`
  - if `tmux` is used, activate `ptycho311`
  - track the exact launched PID
  - accept completion only when the tracked PID exits `0` and required fresh dataset artifacts exist
- Do not mark the item `BLOCKED` for normal test/import/path/validation failures. Diagnose, patch narrowly, and rerun first. Reserve `BLOCKED` for:
  - no acceptable local natural-image corpus
  - missing required probe asset
  - unavailable hardware or storage required for generation
  - roadmap conflict
  - user decision required
  - failure still unrecoverable after a documented narrow fix attempt

## Prerequisite Status

- Effective satisfied CDI prerequisite authority already exists in checked-in summaries even though the progress ledger does not enumerate every later Phase 3 CDI completion:
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/lines128_paper_benchmark_summary.md`
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/lines128_uno_table_extension_summary.md`
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/lines128_minimum_paper_table_summary.md`
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/evidence_matrix.md`
- Progress-ledger context that matters:
  - the initiative already completed Phase 0 and Phase 1 discovery/selection work
  - current selection authority has intentionally moved into a Phase 3 CDI dataset prerequisite
- Downstream dependency to preserve:
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/backlog-gaps/2026-05-04-cdi-natural-patch-expanded-benchmark.md` depends on this dataset but is not part of this item

## Fixed Contract To Preserve

- Dataset id: `natural_patches128_fixedprobe_v1`
- Resolution: `N=128`
- Total object cap: `<= 10_000`
- Recommended initial split target: `8_000 / 1_000 / 1_000` train/validation/test objects
- Split invariant: no source-image overlap across train/validation/test
- Probe lineage invariant:
  - source asset `datasets/Run1084_recon3_postPC_shrunk_3.npz`
  - record the exact preprocessing pipeline inherited from the current `lines128` authority, including any smoothing, padding/scaling, and crop/extrapolation choices
- Object-contract invariant:
  - record one deterministic source-image ordering rule
  - record one deterministic crop-selection rule and seed
  - record one luminance/grayscale conversion rule
  - record one intensity-to-complex-object mapping rule before generation begins
  - keep the same object-contract rule for all splits
- Consumer-compatibility invariant:
  - later benchmark work must be able to consume the emitted dataset without changing model semantics
  - implementation must prove this either through runner-compatible grouped NPZ outputs or through a checked adapter contract plus tests

## Implementation Architecture

- **Unit 1: Contract and provenance capture**
  - Freeze the natural-image source, Run1084 probe lineage, split policy, object encoding, and output schema before live generation.
- **Unit 2: Deterministic dataset builder**
  - Add one focused dataset-generation path that emits split-isolated source patches, simulated CDI arrays, and machine-readable manifests under the git-ignored dataset root.
- **Unit 3: Consumer-compatibility and discoverability**
  - Prove the generated dataset can feed the later expanded-object benchmark without semantic drift, then publish one durable summary and index updates so later tasks can find the dataset and its claim boundary.

## Concrete File And Artifact Targets

### Mandatory code and test surfaces

- Add core builder module:
  - `scripts/studies/cdi_natural_patch_dataset.py`
- Add thin entrypoint:
  - `scripts/studies/run_cdi_natural_patch_dataset.py`
- Add focused test coverage:
  - `tests/studies/test_cdi_natural_patch_dataset.py`
- Modify only if reuse is clean and reduces duplicated schema logic:
  - `scripts/studies/grid_study_dataset_builder.py`
  - `tests/studies/test_grid_study_dataset_builder.py`
- Reference-only compatibility surfaces unless a small schema helper is genuinely required:
  - `scripts/studies/grid_lines_torch_runner.py`
  - `scripts/studies/grid_lines_compare_wrapper.py`

### Mandatory contract outputs

- Dataset root:
  - `.artifacts/data/NEURIPS-HYBRID-RESNET-2026/natural_patches128_fixedprobe_v1/`
- Under that root, emit at minimum:
  - `dataset_manifest.json`
  - `source_manifest.json` or `source_manifest.jsonl`
  - `split_manifest.json`
  - `probe_manifest.json`
  - `simulation_manifest.json`
  - `contact_sheet.png`
  - canonical split payloads for `train`, `val`, and `test`
  - either runner-ready grouped NPZs for the later CDI workflow or an explicit `adapter_contract.json`
- Durable summary:
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/cdi_natural_patch_fixedprobe_dataset_summary.md`
- Required discoverability updates:
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/evidence_matrix.md`
  - `docs/studies/index.md`
  - `docs/index.md`

### Preferred packaging only after core completion

- `verification/` under the dataset root with archived command logs and post-generation audits
- a small `samples/` or `preview/` directory with a few representative source/object/diffraction arrays referenced by the contact sheet
- a concise adapter/readme note near the dataset root if the grouped payloads are not directly runner-ready

## Execution Checklist

### Tranche 1: Freeze The Dataset Contract And Resource Preconditions

- [ ] Create the durable summary scaffold at `docs/plans/NEURIPS-HYBRID-RESNET-2026/cdi_natural_patch_fixedprobe_dataset_summary.md` before code changes. It must state:
  - selected backlog item id
  - this plan path
  - intended dataset root
  - current claim boundary: dataset prerequisite only, not benchmark evidence
  - explicit statement that this item does not train models or replace the `lines128` authority
- [ ] Identify the local natural-image corpus candidate and record:
  - source root/path pattern
  - approximate usable image count
  - license/access note if known locally
  - checksum or size/mtime strategy
- [ ] Audit the fixed Run1084 probe lineage from current `lines128` authority and record the exact preprocessing pipeline that will be reused.
- [ ] Choose and freeze the dataset schema strategy:
  - raw split payloads only plus checked adapter contract, or
  - raw split payloads plus direct runner-ready grouped NPZ exports
- [ ] Run the backlog item’s prerequisite presence check exactly as written before any expensive generation:

```bash
python - <<'PY'
from pathlib import Path
required = [
    Path("datasets/Run1084_recon3_postPC_shrunk_3.npz"),
    Path("scripts/studies/grid_lines_torch_runner.py"),
    Path("scripts/studies/grid_lines_compare_wrapper.py"),
]
missing = [str(path) for path in required if not path.exists()]
if missing:
    raise SystemExit(f"missing natural-patch dataset prerequisites: {missing}")
print("natural-patch fixed-probe dataset prerequisites present")
PY
```

Verification for Tranche 1:

- [ ] **Blocking:** prerequisite presence command passes before any full generation run.
- [ ] **Blocking:** summary scaffold and planned dataset root agree on the same dataset id and claim boundary.
- [ ] **Blocking:** if no acceptable local natural-image corpus is available, stop here, write the blocker details into the summary, and mark the item `BLOCKED` with the required source contract instead of substituting synthetic content.

### Tranche 2: Add Deterministic Builder Logic And Tests Before Live Generation

- [ ] Implement the natural-patch dataset builder in a dedicated study module rather than scattering logic across ad hoc shell commands.
- [ ] Split source images first, then sample patches only within each split so train/validation/test cannot share a parent image.
- [ ] Make patch selection deterministic and manifest-backed:
  - stable source-image ordering
  - fixed split seed
  - fixed crop seed/policy
  - recorded crop coordinates per emitted patch
- [ ] Record one fixed object encoding contract before generation:
  - luminance/grayscale conversion
  - normalization
  - intensity-to-object amplitude and/or phase mapping
  - any clipping/bounds
- [ ] Reuse or minimally extend existing dataset-builder helpers only if that keeps the schema clearer; do not force this item through the synthetic grid-lines path.
- [ ] Add focused tests that cover:
  - deterministic source-level split assignment
  - no source-image overlap across train/validation/test
  - `<= 10_000` total object cap enforcement
  - manifest completeness and stable fields
  - probe-manifest capture
  - emitted split payload keys/shapes for the chosen contract
  - adapter-contract or grouped-NPZ compatibility proof for later CDI consumption

Verification for Tranche 2:

- [ ] **Blocking:** `pytest -q tests/studies/test_cdi_natural_patch_dataset.py` passes before any full dataset generation run.
- [ ] **Supporting:** if `scripts/studies/grid_study_dataset_builder.py` is modified, run a narrow compatible selector such as `pytest -q tests/studies/test_grid_study_dataset_builder.py`.
- [ ] **Supporting:** run `python -m compileall -q scripts/studies ptycho_torch` after code edits to catch syntax/import drift early.

### Tranche 3: Generate And Lock The Dataset Root

- [ ] Create the git-ignored dataset root `.artifacts/data/NEURIPS-HYBRID-RESNET-2026/natural_patches128_fixedprobe_v1/`.
- [ ] Emit canonical machine-readable manifests:
  - source image provenance
  - split membership
  - patch coordinates and patch ids
  - probe source and preprocessing
  - simulation parameters
  - dataset-level array inventory and schema version
- [ ] Emit canonical split payloads for train/validation/test under the locked contract.
- [ ] Emit either:
  - runner-ready grouped NPZ payloads that match the later CDI consumer contract, or
  - an explicit adapter contract plus proof artifact showing how later work derives grouped payloads without changing semantics
- [ ] Build the required small contact sheet from representative source patches and simulated diffraction samples.
- [ ] If the generation run is non-trivial, execute it under `tmux`, track the exact PID, and require `rc=0` plus fresh required artifacts before considering the run complete.

Verification for Tranche 3:

- [ ] **Blocking:** a post-generation audit proves:
  - total objects `<= 10_000`
  - split counts match the manifest
  - no source-image overlap across train/validation/test
  - all required manifests exist
  - contact sheet exists
  - split payloads exist for all three splits
- [ ] **Blocking:** consumer-compatibility proof exists, either as runner-ready grouped NPZs or as a tested adapter contract.
- [ ] **Supporting:** if the first live generation attempt fails for ordinary data/path/schema reasons, narrow-fix and rerun instead of marking `BLOCKED`.

### Tranche 4: Publish Durable Summary And Discoverability Updates

- [ ] Finalize `docs/plans/NEURIPS-HYBRID-RESNET-2026/cdi_natural_patch_fixedprobe_dataset_summary.md` with:
  - chosen source corpus identity
  - locked split counts
  - fixed Run1084 probe lineage and preprocessing
  - object encoding/simulation contract
  - dataset root and manifest paths
  - explicit claim boundary: prerequisite dataset only, not benchmark-performance evidence
  - downstream consumer note for the later expanded-object benchmark item
- [ ] Update `docs/plans/NEURIPS-HYBRID-RESNET-2026/evidence_matrix.md` so later manuscript-planning work can discover the dataset artifact root and summary authority.
- [ ] Update `docs/studies/index.md` with a concise study entry describing the dataset contract, summary path, and boundary.
- [ ] Update `docs/index.md` because this becomes durable project knowledge discoverable outside the initiative-local study index.
- [ ] Update `docs/findings.md` only if implementation uncovers a durable future-facing pitfall that later workers are likely to repeat.

Verification for Tranche 4:

- [ ] **Blocking:** the summary, evidence matrix entry, and study/index entries all point to the same dataset id and summary authority.
- [ ] **Supporting:** if no `docs/findings.md` update is needed, state that explicitly in the execution report with the reason.

## Required Deterministic Checks Before Closing The Item

Run these checks exactly unless a stronger replacement is explicitly justified in the execution report. For this item, they remain required as written:

```bash
python - <<'PY'
from pathlib import Path
required = [
    Path("datasets/Run1084_recon3_postPC_shrunk_3.npz"),
    Path("scripts/studies/grid_lines_torch_runner.py"),
    Path("scripts/studies/grid_lines_compare_wrapper.py"),
]
missing = [str(path) for path in required if not path.exists()]
if missing:
    raise SystemExit(f"missing natural-patch dataset prerequisites: {missing}")
print("natural-patch fixed-probe dataset prerequisites present")
PY
pytest -q tests/studies/test_cdi_natural_patch_dataset.py
python -m compileall -q scripts/studies ptycho_torch
```

Check classification:

- **Blocking before expensive generation:** prerequisite presence check and `pytest -q tests/studies/test_cdi_natural_patch_dataset.py`
- **Blocking before close:** rerun `pytest -q tests/studies/test_cdi_natural_patch_dataset.py` after any late narrow fix and run `python -m compileall -q scripts/studies ptycho_torch`

## Execution Boundary And Handoff

- This item is complete when:
  - `natural_patches128_fixedprobe_v1` exists under the git-ignored artifact root
  - required manifests and contact sheet exist
  - source-level split isolation is proven
  - the dataset is shown to be consumable by later CDI benchmarking without semantic drift
  - the durable summary and index updates are complete
- This item is not complete merely because a builder script exists or because raw arrays were dumped without manifests.
- This item must not launch the later expanded-object benchmark. Handoff to the later benchmark item should reference the summary/manifests created here and keep the existing `lines128` paper table authority unchanged.
