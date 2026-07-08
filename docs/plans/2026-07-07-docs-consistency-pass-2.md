# Documentation Consistency Pass 2 — Internal Architecture, APIs & Data Conventions

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development. Briefs/reports under `.superpowers/sdd/ext/` (task-docpass2-t*-{brief,report}.md), ledger `.superpowers/sdd/progress.md`.

**Date:** 2026-07-07 · **Branch:** `fno-stable` · **Status:** COMPLETE (2026-07-07; see Execution record)
**Input:** four-partition read-only audit (internal-architecture docs; NPZ/data contracts vs loaders; TF↔PyTorch tensor correspondence; API/config-bridge contracts), 2026-07-07. Follows pass 1 (`2026-07-07-docs-consistency-pass.md`, COMPLETE).

**Goal:** Make the developer-facing internal-architecture story current and discoverable from `docs/index.md`, and make the cross-backend data/API contracts true: one canonical TF↔torch tensor-correspondence spec, corrected external API spec, reconciled config-bridge spec, and data-contract key names that route to the right arrays.

## Headline audit results this plan addresses
- The only existing TF↔torch shape table (`specs/ptychodus_api_spec.md:191-226` §4.4.1, duplicated at `docs/architecture.md:55-70`) is wrong about torch real/imag output — claims `[B,C,N,N,2]`; code uses raw `(B,H,W,C,2)` (`ptycho_torch/model.py:52-81,94-115`) then returns **complex `(B,C,N,N)`** from `forward_predict` (`model.py:1924-1931,2006-2015`). Everything deeper (probe layouts, padded-size formulas, reassembly sign) is undocumented.
- `docs/architecture_torch.md` (last touched 2026-04-30) omits `ptycho_torch/reassembly.py` entirely, lists 9 of 14 registry generators, calls native reassembly "planned" (false since ≤2026-07-02), and omits the ADR-003 `config_factory` layer. `docs/index.md` has no adjacent reading path from "Architecture & Development" to the `spec-ptycho-*` contracts; `docs/adr/ADR-0007` is orphaned.
- `ptycho_torch/dataloader.py:36,48-72` and `ptycho_torch/memmap_bridge.py:110-124` docstrings invert the NPZ contract: they call `'diffraction'` "canonical per DATA-001" — canonical standalone-NPZ key is `diff3d` (`docs/specs/spec-ptycho-core.md:99`, `ptycho/raw_data.py:322`); `'diffraction'` names an H5 group in a different container (`specs/data_contracts.md:207`).
- `specs/ptychodus_api_spec.md` (external contract): omits the `architecture` routing field; claims a nonexistent `local_offsets` dict key from `generate_grouped_data` (it is a `PtychoDataContainer` attribute, `ptycho/loader.py:508`); specifies a scheduler enum (`Default/Exponential/MultiStage/Adaptive`) whose values don't exist — real enum is `TrainingConfig.scheduler: Literal['Default','Exponential','WarmupCosine','ReduceLROnPlateau']` (`ptycho/config/config.py:157`), and `PyTorchExecutionConfig.scheduler` is an unvalidated `str` (`config.py:267`).
- Real code/spec divergence needing adjudication, not silent patching: `ptycho/loader.py:462-465` sources `coords_true` and `coords_nominal` from the SAME key (`coords_start_relative`), so they can never diverge, while `docs/specs/spec-ptycho-interfaces.md:49` implies divergence when true positions are provided.
- Backend-default divergences unflagged in the bridge spec: TF `gridsize=1` vs PT `grid_size=(2,2)` (`config.py:97` vs `config_params.py:30`); `TrainingConfig.positions_provided=True` vs legacy `params.py:68 False`; spec claims PT `amp_activation` is a 6-value Literal but it is unvalidated `str='silu'` (`config_params.py:117`).
- `docs/CONFIGURATION.md`: `plateau_min_lr` documented `1e-4`, code default `5e-5` (`config.py:162`); ModelConfig/TrainingConfig/InferenceConfig tables silently incomplete (no "subset" label, unlike the PyTorchExecutionConfig table fixed in pass 1).
- Stale/wrong code docstrings: `run_cdi_example_torch` still says "raises NotImplementedError" though fully implemented (`ptycho_torch/workflows/components.py:149`); `probes_indexed` documented 4D but is 5D `(B,C,P,N,N)` (`ptycho_torch/dataloader.py:782-783`, `dset_loader_pt_mmap.py:389`); `PadAndDiffractLayer` docstring claims `(B,N,N,C)` input while wrapped `pad_and_diffract` asserts a trailing 1-channel (`ptycho/custom_layers.py:262-263` vs `ptycho/tf_helper.py:353,358`) — needs a static trace before deciding docstring-fix vs bug finding.

## Global constraints
- Same regime as pass 1: never weaken a contract (supersede/annotate, don't delete normative text); index-routing invariant (any doc added/renamed/re-scoped patches `docs/index.md` in the same commit); pathspec-scoped commits + scoped purity checks (shared tree, concurrent session); no "claude" in messages, no trailers, never `--no-verify`, no pushes; one commit per task.
- Protected files untouchable: `ptycho/model.py`, `ptycho/diffsim.py`, `ptycho/tf_helper.py`, `ptycho/config/config.py`, `ptycho_torch/config_params.py`, `ptycho_torch/model.py`. ADDITIONALLY no-touch (concurrent gs2 session): `ptycho_torch/helper.py`, `ptycho_torch/data_container_bridge.py`, `ptycho_torch/patch_generator.py`. Code edits in this plan are DOCSTRING/COMMENT-ONLY, in non-protected files, except nothing else.
- Every shape/convention claim written into a doc must be re-verified against the tree by the writing implementer (cite file:line current at write time). The auditor citations above were verified 2026-07-07 but the tree moves.
- `specs/ptychodus_api_spec.md` is an EXTERNAL contract: edits must correct it toward implementation truth, never re-specify behavior.
- No behavior changes anywhere. The `coords_true` divergence is documented + escalated, not fixed here.

## Wave schedule (disjoint files per wave)
- **Wave 1 (parallel): T3, T4, T5, T6**
- **Wave 2 (parallel): T1, T7**
- **Wave 3: T2** (needs T1's shard to point at; shares files with T1/T7)
- **Wave 4:** opus batch review over all commits → fix round if needed → T8 close-out.

---

### Task 3 — Data-contract corrections (docs + findings) (ONE commit)
**Files:** `docs/specs/spec-ptycho-interfaces.md`, `specs/data_contracts.md`, `docs/DATA_MANAGEMENT_GUIDE.md`, `docs/DATA_NORMALIZATION_GUIDE.md`, `docs/findings.md`.
1. `DATA_MANAGEMENT_GUIDE.md:92-95` and `:254-256`: standalone-NPZ key `'diffraction'` → `'diff3d'` (both the Standard Keys prose and the `validate_dataset()` required list). Add one sentence noting `'diffraction'` is the H5 `/raw_data` bundle name (`specs/data_contracts.md` §12), not the standalone-NPZ key.
2. `DATA_MANAGEMENT_GUIDE.md` NPZ section: one note — torch loaders auto-transpose legacy `(H,W,N)` arrays (FORMAT-001), TF `RawData.from_file` does not (square-shape assert only); cross-ref `docs/findings.md` FORMAT-001.
3. `DATA_NORMALIZATION_GUIDE.md:159-163`: scope the blanket PyTorch intensity_scale claim — true for the `_attach_physics_scale` container path; the grid-lines dict path attaches NO physics scale by default (`--count-scale-mode` default `off`; `auto` opt-in, not outcome-preserving; POISSON-SCALE-001). Add `docs/findings.md` (POISSON-NORM-001, POISSON-SCALE-001) to Related Documentation.
4. `specs/data_contracts.md:207` (§12 `/raw_data` `diffraction` dataset): append units annotation — "amplitude (sqrt of counts), consistent with the standalone-NPZ `diff3d` contract in docs/specs/spec-ptycho-core.md".
5. `docs/specs/spec-ptycho-interfaces.md`: add a **"Torch grid-lines dict-container contract"** block next to the grouped-dict section: keys `X`, `observed_images` (loss-side raw diffraction, = `train_data['diffraction']` in all input-conditioning modes), `coords_relative`/`positions`, `probe`, `physics_scaling_constant` (optional; absent by default), `rms_scaling_constant`, `label_amp`/`label_phase` — semantics + producing/consuming code refs (`ptycho_torch/workflows/components.py:394-447,583-594`, runner `:1382/:1420`). Shapes may cross-ref the Task-1 correspondence spec.
6. `docs/specs/spec-ptycho-interfaces.md:49` (coords_nominal/coords_true): add known-gap note — current TF loader sources both from `coords_start_relative` (`ptycho/loader.py:462-465`), so they never diverge; the `coords_relative`/`coords_offsets` pair computed in `raw_data.py:550,585-586` is not read by the loader. Requirement stands; implementation is the acknowledged deviation.
7. `docs/findings.md`: new entry **TF-LOADER-COORDS-001** (Active): the divergence in (6), evidence lines, impact (true-position jitter datasets silently ignored by TF loader path), resolution options (fix loader vs narrow spec) deferred to a user decision; cross-ref spec-ptycho-interfaces known-gap note. Index-table row included.
8. Commit: `docs: correct standalone-NPZ key routing, scope normalization claims, add dict-container contract`

### Task 4 — External API spec corrections (ONE commit)
**Files:** `specs/ptychodus_api_spec.md` only.
1. §2.1/§5.1 `ModelConfig`: add the `architecture` field (`Literal`, 14 values, default `'cnn'`, source of truth `ptycho/config/config.py:100-102`) and note dependent fields (`fno_modes`, `generator_output_mode`, …) enforced by `validate_model_config`.
2. §"generate_grouped_data" (~:171-174): remove the phantom `local_offsets` dict key; state it is a `PtychoDataContainer` attribute derived in `ptycho/loader.py:508`; list the actual dict keys from `ptycho/raw_data.py:582-593`.
3. ~:328 scheduler: correct to `TrainingConfig.scheduler: Literal['Default','Exponential','WarmupCosine','ReduceLROnPlateau']` (`config.py:157`) and note `PyTorchExecutionConfig.scheduler` is an unvalidated `str` (`config.py:267`); delete the nonexistent `MultiStage`/`Adaptive` values.
4. §4.4.1 (:191-226) shape-table correction: torch generator real/imag RAW output is `(B, H, W, C, 2)` (`ptycho_torch/model.py:52-81,94-115`), converted to **complex `(B, C, N, N)`** before `forward_predict` returns (`model.py:1924-1931,2006-2015`); the public inference output is always complex — remove the "either/or real-imag-stacked" framing. Re-verify each cited line before writing.
5. Commit: `docs: correct ptychodus API spec — architecture field, grouped-dict keys, scheduler enum, torch output shapes`

### Task 5 — Config-bridge spec + CONFIGURATION.md reconciliation (ONE commit)
**Files:** `docs/specs/spec-ptycho-config-bridge.md`, `docs/CONFIGURATION.md`, `docs/cli_config_dataflow.md`.
1. Bridge spec: correct `amp_activation` row — PT field is unvalidated `str = 'silu'` (`config_params.py:117`); TF-side Literal stands.
2. Bridge spec: add default-divergence caveats — TF `gridsize=1` vs PT `grid_size=(2,2)` (`config.py:97` / `config_params.py:30`); `positions_provided` TF dataclass `True` vs legacy `params.py:68` `False` pre-bridge.
3. Bridge spec: add rows or an explicit "non-bridged Torch-only knobs" note for `generator_output_mode`, `cnn_output_mode`, `physics_forward_mode`, `training_patch_weighting`, `rect_s1s2_trainable`, and the loss-weight mapping asymmetry (TF `mae_weight`/`realspace_mae_weight` vs PT `loss_function`/`amp_loss`/`phase_loss`) with code refs.
4. `CONFIGURATION.md`: `plateau_min_lr` default `1e-4` → `5e-5` (`config.py:162`); extend the pass-1 "illustrative subset — full list in ptycho/config/config.py" label to the ModelConfig, TrainingConfig, and InferenceConfig tables; note the architecture Literal is enumerated authoritatively in `spec-ptycho-config-bridge.md` §3.
5. `cli_config_dataflow.md:204-212`: strike or correct the fabricated `KEY_MAPPINGS = {'nepochs':'epochs',…}` example (real mapping `config.py:720-730` has no such entry; legacy key stays `nepochs`) — one surgical edit under the existing historical banner.
6. Commit: `docs: reconcile config-bridge spec defaults and knobs, fix CONFIGURATION table gaps`

### Task 6 — Code docstring corrections (docstring/comment-only; ONE commit)
**Files:** `ptycho_torch/dataloader.py`, `ptycho_torch/memmap_bridge.py`, `ptycho_torch/dset_loader_pt_mmap.py`, `ptycho_torch/workflows/components.py`, `ptycho/custom_layers.py`. (None protected; gs2 no-touch files NOT in this list.)
1. `dataloader.py:36,48-72,93-130` + `memmap_bridge.py:110-124`: fix the inverted canonical-key claims — canonical standalone-NPZ key is `diff3d` (`docs/specs/spec-ptycho-core.md`); `'diffraction'` is accepted as an alias here but is canonically the H5 `/raw_data` dataset name. Keep behavior untouched; docstrings/comments only.
2. `dataloader.py:782-783` + `dset_loader_pt_mmap.py:389`: `probes_indexed` documented 4D `(N,C,H,W)` → actual 5D `(B,C,P,N,N)` per the `unsqueeze(1).expand(…)` chain (`dataloader.py:805`).
3. `components.py:149` `run_cdi_example_torch`: replace the "Phase D2.A Scaffold … raises NotImplementedError" docstring with a description of the implemented behavior (steps 1–5, return contract).
4. `custom_layers.py:262-263` `PadAndDiffractLayer`: STATIC trace first — does a channel→flat conversion (e.g. `_channel_to_flat`) sit between `ExtractPatchesPositionLayer` output and this layer in `ptycho/model.py:561-572` (read-only)? If yes: fix the docstring to state the flat `(B·C,N,N,1)` input contract. If genuinely absent for gridsize>1: do NOT edit; add finding candidate to your report and mark this step ESCALATE (controller files it).
5. Verify: `python -c "import ptycho_torch.dataloader, ptycho_torch.memmap_bridge, ptycho_torch.dset_loader_pt_mmap, ptycho_torch.workflows.components, ptycho.custom_layers"` (CUDA_VISIBLE_DEVICES=""); `git diff` shows only docstring/comment hunks (no executable-line changes).
6. Commit: `docs: correct stale docstrings — NPZ key routing, probe tensor rank, implemented workflow status`

### Task 1 — Canonical TF↔Torch tensor-correspondence spec (NEW shard; ONE commit)
**Files:** create `docs/specs/spec-ptycho-tensor-correspondence.md`; modify `docs/architecture.md`, `docs/specs/spec-ptychopinn.md` (shard index), `docs/index.md`.
1. Author the new shard: a code-cited correspondence table covering (re-verify every citation; mark each row Documented-elsewhere / Divergence / Previously-undocumented):
   1. Grouped diffraction `X`: TF `(B,N,N,C)` channel-last (`ptycho/loader.py:112`) ↔ torch container `(B,N,N,C)` identical (`data_container_bridge.py:210`) ↔ torch model input `(B,C,N,N)` after Dataset permute (`workflows/components.py:521-530`).
   2. Positions/offsets: TF `(B,1,2,C)` `[x,y]` (`loader.py:117-118`) ↔ torch `(B,C,1,2)` (`components.py:543-548`).
   3. Reassembly: torch `(B,C,N,N)` complex + offsets `(B,C,1,2)` → canvas `(B,M,M)` (`helper.py:38,55-56,68`), with the `Translation(…, -offsets_flat)` sign negation (`helper.py:110,254`) and TORCH-REASSEMBLY-SIGN-001 cross-ref.
   4. Generator raw real/imag `(B,H,W,C,2)` → complex `(B,C,H,W)` (`ptycho_torch/model.py:52-119`); the three output modes (real_imag / amp_phase / amp_phase_logits) and their bounded activations.
   5. Probe layouts (the four torch variants + TF `(N,N)`-no-modes): container `(N,N)`; grid-lines dict `(B,H,W)`/`(B,P,H,W)` (`components.py:612-631`); mmap loader `(B,C,P,N,N)` (`dataloader.py:791-806`); rect-scaled `(C,P,H,W)` (`components.py:619-629`); dispatch on `probe.ndim` (`helper.py:257-282`).
   6. `pad_and_diffract`: TF flat `(B·C,N,N,1)` single-channel asserted (`tf_helper.py:353,358` — cite read-only) ↔ torch `(N,C,P,H,W)` with mode-sum (`helper.py:651-674`).
   7. Padded-size formulas (DIVERGENCE): TF `N+(gs-1)·offset+jitter` (`params.py:88-104`) vs torch `N+(gs-1)·ceil(max_neighbor_dist, even)+0` (`helper.py:493-507`, TORCH-PADDED-SIZE-001).
   8. Amplitude-not-intensity arrays both backends (`loader.py:528`, `data_container_bridge.py:121`, spec-ptycho-core §Raw NPZ).
   9. Complex representation: TF single complex64 tensor vs torch mode-dependent intermediates.
   10. `local_offset_sign = -1` shared constant (spec-ptycho-core:44, workflow:18, conformance:37) — the exemplar convention.
   11. Channel↔(row,col) row-major `c//gs, c%gs` (`loader.py:78`, `architecture.md:53`).
   12. Single-patch inference reshape `(1,-1,N,N)` (`architecture_torch.md` §3.1).
2. `docs/architecture.md:55-70`: replace the duplicated shape table with a two-line summary + pointer to the new shard and to `ptychodus_api_spec.md` §4.4.1 (post-Task-4 corrected) — one canonical statement per fact.
3. Route the shard: `docs/specs/spec-ptychopinn.md` shard index + `docs/index.md` Specifications section + a "TF↔Torch correspondence" line in the Architecture & Development section.
4. Consistency gate: the shard's §4.4.1-adjacent facts must match Task 4's corrected text exactly (both derive from the Verified-facts in this plan's headline section).
5. Commit: `docs: add canonical TF-torch tensor correspondence spec, deduplicate shape tables`

### Task 7 — IDL-style component contracts section (ONE commit)
**Files:** `docs/architecture_torch.md` (new "Component Contracts" section only — Wave-3 Task 2 edits other sections), `docs/DEVELOPER_GUIDE.md` (one pointer), plus one-line docstring cross-refs in `ptycho/raw_data.py`, `ptycho/loader.py`, `scripts/studies/grid_lines_torch_runner.py`, `ptycho_torch/workflows/components.py`, `ptycho_torch/generators/registry.py` (docstring-only).
1. Author contracts (signature, inputs/outputs with types+shapes, dependencies incl. params.cfg reads, behavior, error modes) for: `RawData` (`raw_data.py:127`), `PtychoDataContainer` (`loader.py:97`), `run_grid_lines_torch` (`grid_lines_torch_runner.py:1912`), `run_cdi_example_torch` (`components.py:149` — post-Task-6 docstring), `resolve_generator`/registry (`registry.py:39`). Shapes cross-ref the Task-1 shard rather than restating.
2. Each of the five code docstrings gains one line: `Contract: docs/architecture_torch.md §Component Contracts.`
3. `DEVELOPER_GUIDE.md`: one pointer line in the architecture section.
4. Commit: `docs: add IDL-style component contracts for core data/workflow/generator APIs`

### Task 2 — Architecture-doc currency + reading path (ONE commit; Wave 3)
**Files:** `docs/architecture_torch.md`, `docs/architecture_tf.md`, `ptycho_torch/generators/README.md`, `docs/index.md`.
1. `architecture_torch.md`: add `ptycho_torch/reassembly.py` to §4 Component Reference (barycentric/VarPro reassembly: `reconstruct_image_barycentric`, `VarProScaler`, `compute_varpro_basis`; REASSEMBLY-BRIDGE-001 cross-ref); replace `:124` "native Torch reassembly planned" with the actual two-path story (helper.py MVP path + reassembly.py VarPro/barycentric path).
2. `architecture_torch.md` §4.1 generators table: sync to the 14-entry registry OR replace the enumeration with a pointer to `spec-ptycho-config-bridge.md` §3 + `registry.py` as the authorities (prefer the pointer; keep a 3-4 row illustrative table labeled as such). Same treatment in `ptycho_torch/generators/README.md`.
3. `architecture_torch.md` §1 diagram/prose: insert the ADR-003 `config_factory.py` layer between CLI and `config_bridge` (text bullet acceptable if the diagram is ASCII-art-fragile).
4. `architecture_torch.md` §3.1 Reassembly Contract: add the offset-sign convention + TORCH-REASSEMBLY-SIGN-001 link (coordinate with Task 1's shard — link, don't restate).
5. `architecture_tf.md:157`: fix the "uses TF helper for MVP parity / native planned" claim the same way; cross-link §6 ↔ `architecture_torch.md` §5 ↔ the Task-1 shard.
6. `docs/index.md`: "See also: Specifications (normative contracts: docs/specs/spec-ptycho-*)" pointer at the end of Architecture & Development; route `docs/adr/ADR-0007-remove-hubs.md` (new one-line Decisions entry).
6b. (added during execution, T1 observation) `docs/specs/spec-ptychopinn.md`: register `spec-ptycho-config-bridge.md` in the shard index (it was missing; T1 added `spec-ptycho-tensor-correspondence.md` there, leaving config-bridge the only unlisted shard). Add `docs/specs/spec-ptychopinn.md` to this task's files/pathspec.
7. Commit: `docs: bring torch architecture docs current — reassembly, registry, config factory, reading path`

### Task 8 — Close-out (ONE commit)
1. `docs/findings.md`: extend DOCS-CONSISTENCY-001 (or add a dated addendum) with pass-2 scope, the new spec shard, TF-LOADER-COORDS-001, and commit hashes.
2. This plan: statuses + hashes, Status → COMPLETE.
3. Commit: `docs: close out documentation consistency pass 2`

## Execution record (2026-07-07)
All eight tasks complete. Commits (in landing order):
- **T3** `26f60ff7` — data-contract corrections + dict-container contract + TF-LOADER-COORDS-001 finding
- **T4** `69b42962` — ptychodus API spec: architecture field, grouped-dict keys, scheduler enum, §4.4.1 output shapes
- **T5** `f3fd9eb6` — config-bridge caveats + non-bridged knobs; CONFIGURATION.md fixes; KEY_MAPPINGS strike
- **T6** `12a8843b` — docstring-only code corrections (canonical-key inversion, probes 5D, run_cdi_example_torch, PadAndDiffractLayer). Deviations: `dset_loader_pt_mmap.py` probes docstring was already correct (genuinely 4D there — plan step 2 partially wrong); runtime error-message literals with inverted canonical wording left untouched (executable lines; deferred follow-up, endorsed by review). Task 6.4 static trace confirmed `_channel_to_flat` upstream of `PadAndDiffractLayer` — docstring fix, NOT a bug; no escalation.
- **T1** `0f07c00b` — NEW `docs/specs/spec-ptycho-tensor-correspondence.md` (12 code-verified rows), architecture.md dedup→pointer, shard-index + index.md routing
- **T7** `e0768df1` — `architecture_torch.md` §6 Component Contracts + DEVELOPER_GUIDE pointer + 5 docstring cross-refs
- **T2** `2ab53c10` — architecture currency (reassembly two-path, registry pointer, config-factory, sign-convention link), generators README, ADR-0007 + Specifications routing, config-bridge shard registered (step 6b)
- Fix round `6f37d49c` — citation line-drift cleanup (sole review finding; findings.md TF-LOADER-COORDS-001 anchors). Drift in the other three docs was independently resolved by the symbol-reference conversion commits `1b5348b8`/`a21afa7e`.
- **T8** — this close-out commit (findings.md DOCS-CONSISTENCY-001 pass-2 addendum + this record).

**Batch review (opus, 2026-07-07): Approved (PASS_WITH_CONCERNS).** All normative claims verified true against the live tree; per-task spec verdicts all ✅; both code commits confirmed docstring-only; no protected/no-touch file modified; hygiene clean. Sole finding (COULD-level): citation line-drift from same-batch docstring insertions — fixed above. Reviewer endorsed deferring the error-message literal fix as a small separate executable-line follow-up.

## Escalations to the user (recorded, not blocking)
- **TF-LOADER-COORDS-001**: fix `ptycho/loader.py` to read `coords_relative`/`coords_offsets` (behavior change, needs tests) vs narrow the spec. Decision requested after this pass.
- `PadAndDiffractLayer` static trace (Task 6.4) may surface a real gridsize>1 shape bug — will be escalated with evidence if found.
- The six `tests/study` dose_overlap tests remain archival candidates (pass-1 follow-up).

## Deferred
- Full IDL coverage beyond the five sampled components; deeper spec-shard modernization; `docs/models/srunet.md` placement; NEURIPS index consolidation (unchanged from pass 1).
