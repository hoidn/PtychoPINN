# BRDT Dataset Preflight Execution Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. This item authorizes only the BRDT smoke-dataset contract, deterministic generation path, manifest, and dry-run geometry summary. Do not add BRDT training adapters, do not run the four-row BRDT preflight, do not register BRDT as a normal CDI generator, do not promote BRDT into manuscript evidence, and do not create worktrees. If dataset generation becomes long-running, keep it under implementation ownership until terminal success or recoverable failure handling is complete; use `tmux` plus the `ptycho311` environment when appropriate.

**Goal:** Create the minimal reproducible BRDT smoke dataset and manifest path needed for later adapter and bounded four-row preflight work, while locking the physical `q` target and the normalization rule that keeps physics loss in physical units.

**Architecture:** Split the work into three units: a dataset-contract layer that freezes the physical target, split policy, and normalization fields; a deterministic generator/dry-run CLI that consumes the locked Born operator contract and writes smoke outputs under an ignored artifact root; and durable reporting/index updates that make the preflight discoverable without promoting BRDT beyond feasibility status. The operator itself is already locked by the passed validation report and must be consumed, not redefined, in this item.

**Tech Stack:** PATH `python`, PyTorch for operator execution, NumPy plus `h5py` for smoke dataset serialization, JSON/HDF5 manifests, pytest, compileall, Markdown evidence docs.

---

## Selected Objective

- Implement backlog item `2026-04-29-brdt-dataset-preflight`.
- Lock the BRDT physical supervision target to:

  ```math
  q(x,z)=k_m^2\left(\left(\frac{n(x,z)}{n_m}\right)^2-1\right).
  ```

- Generate only a small smoke/preflight dataset under the validated Born operator contract.
- Store physical `q_true`, normalized `q_true`, complex sinograms, angle masks, split metadata, train-only normalization statistics, generation command, git state, and environment information.
- Make the future physics-loss rule explicit and durable:

  ```math
  L_{\mathrm{phys}} = \|A(\mathrm{unnormalize}(\hat q))-y\|.
  ```

- Emit a machine-readable dataset manifest and a dry-run geometry validation summary that later BRDT items can consume directly.

## Scope

- Consume the operator authority from:
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/brdt_operator_validation_report.md`
  - `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-brdt-operator-validation/operator_validation.json`
- Consume the candidate-lane boundaries from:
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/born_rytov_dt_candidate_lane_design.md`
- Implement the minimum study-local dataset generation surface under `scripts/studies/born_rytov_dt/`.
- Add dataset-contract tests under `tests/studies/test_born_rytov_dt_dataset.py`.
- Write a durable checked-in preflight summary and update the NeurIPS evidence indexes so downstream adapter/preflight items can discover the smoke dataset authority.

## Explicit Non-Goals

- Do not modify `ptycho_torch/physics/born_rytov_dt.py` unless a true operator-contract bug is discovered; if the locked operator authority is wrong, stop and open a separate narrow follow-up instead of silently changing the contract here.
- Do not add Lightning modules, dataloaders, row metadata schemas, model wrappers, or loss wrappers for BRDT training. Those belong to `2026-04-29-brdt-task-adapters`.
- Do not run the bounded classical/U-Net/FNO/SRU-Hybrid four-row preflight. That belongs to `2026-04-29-brdt-four-row-preflight`.
- Do not add Rytov mode, limited-angle stress tests, FFNO rows, external FDTD mismatch data, or paper-facing BRDT tables.
- Do not use CDI line-pattern objects as the only phantom family.
- Do not generate the later larger `128 x 128` decision-support split in this item.
- Do not write manuscript prose or create `/home/ollie/Documents/neurips/` artifacts.

## Steering, Roadmap, and Candidate-Lane Constraints

- Steering and the roadmap keep CDI `lines128` and PDEBench CNS as the required manuscript pillars. BRDT remains additive candidate work only and cannot support manuscript claims without a later checked-in roadmap or evidence-package amendment.
- This item is bounded to dataset feasibility. It must not drift into adapter or benchmark execution work just because the smoke dataset exists.
- The passed operator validation report is binding authority for mode, geometry, FFT normalization, wavelength, medium refractive index, angle convention, detector layout, and output sinogram layout. The dataset generator must consume that authority rather than restating or drifting it.
- The physical forward operator always consumes physical `q`. Normalized `q` exists only as model-output/storage convenience; the manifest and checked-in summary must both state `forward_input_is_physical_q: true` and make the unnormalize-before-physics rule explicit.
- Smoke outputs are feasibility artifacts only. Even if a tiny smoke dataset later supports a toy model run, this item does not authorize ranking models, claiming benchmark readiness, or promoting BRDT beyond feasibility.
- Artifact hygiene is binding: large generated arrays stay under an ignored or external data root, while the checked-in plan summary and index updates remain lean and human-readable.

## Prerequisite Status

- The backlog prerequisite `2026-04-29-brdt-operator-validation` is satisfied by the completed backlog item and the report `docs/plans/NEURIPS-HYBRID-RESNET-2026/brdt_operator_validation_report.md`, whose verdict is `pass_with_documented_limits`.
- The operator validation report explicitly clears `2026-04-29-brdt-dataset-preflight` to proceed and records the only current optional gap as `odtbrain` unavailable, which is non-blocking for this dataset item.
- The progress ledger does not yet track a BRDT-specific tranche. Do not infer broader BRDT readiness from the ledger; treat the operator report plus the selected backlog item as the authoritative prerequisite state for this plan.

## Execution Rules

- Diagnose, fix, and rerun ordinary import, path, environment, serialization, or test failures before considering the item blocked.
- Reserve `BLOCKED` for missing external storage/resources, unrecoverable dependency or hardware failure after a documented narrow fix attempt, roadmap conflict, or an unresolved mismatch between the passed operator authority and the dataset contract.
- Keep long-running generation under implementation ownership until the launched process exits cleanly and the expected artifacts are freshly written. Do not launch duplicate runs against the same `--output-root`.
- Use PATH `python` in commands. If a long-running generation command is needed, run it in `tmux` with `ptycho311` active per repo guidance.
- Keep scratch data under `tmp/` or the item artifact root and remove non-durable scratch before completion.

## Implementation Architecture

1. **Dataset Contract Layer**
   - Owns the locked physical target, the train-only normalization statistics, split metadata, manifest schema, and the explicit separation between `physical_q` and `normalized_q`.
2. **Smoke Generator And Dry-Run CLI**
   - Owns phantom-family sampling, deterministic split/sample seeds, operator invocation using the locked Born contract, HDF5/JSON output writing, and the geometry-only dry-run mode.
3. **Durable Reporting And Discoverability**
   - Owns the checked-in preflight summary plus `docs/index.md`, `docs/studies/index.md`, `evidence_matrix.md`, and `paper_evidence_index.md` updates needed to keep the new dataset authority discoverable and correctly labeled as feasibility-only.

## File and Artifact Targets

Mandatory contract outputs:

- `scripts/studies/born_rytov_dt/generate_brdt_dataset.py`
- `tests/studies/test_born_rytov_dt_dataset.py`
- `docs/plans/NEURIPS-HYBRID-RESNET-2026/brdt_dataset_preflight.md`
- `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-brdt-dataset-preflight/dataset_manifest.json`
- `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-brdt-dataset-preflight/dry_run_summary.json`

Likely code/support files to create or modify:

- `scripts/studies/born_rytov_dt/__init__.py`
- `scripts/studies/born_rytov_dt/dataset_contract.py`
- `scripts/studies/born_rytov_dt/phantoms.py`

Preferred packaging for generated smoke data:

- `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-brdt-dataset-preflight/dataset/brdt128_sparse_fullview_preflight_train.h5`
- `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-brdt-dataset-preflight/dataset/brdt128_sparse_fullview_preflight_val.h5`
- `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-brdt-dataset-preflight/dataset/brdt128_sparse_fullview_preflight_test.h5`
- `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-brdt-dataset-preflight/logs/`

Mandatory discoverability and evidence updates if the durable summary/artifacts are produced:

- `docs/index.md`
- `docs/studies/index.md`
- `docs/plans/NEURIPS-HYBRID-RESNET-2026/evidence_matrix.md`
- `docs/plans/NEURIPS-HYBRID-RESNET-2026/paper_evidence_index.md`

Leave unchanged unless a separate approved follow-up authorizes it:

- `ptycho_torch/physics/born_rytov_dt.py`
- `ptycho/model.py`
- `ptycho/diffsim.py`
- `ptycho/tf_helper.py`
- `docs/plans/2026-04-20-neurips-hybrid-resnet-submission-design.md`
- `docs/plans/2026-04-20-neurips-hybrid-resnet-submission-roadmap.md`

## Mandatory Durable Output Contract

Before this item is complete, the checked-in summary plus machine-readable manifest/summary must cover:

- dataset identity:
  - backlog item ID
  - output root
  - exact generation command
  - git SHA and dirty-state note
  - Python/environment summary
  - pointer to the operator validation report and `operator_validation.json`
- locked physical target:
  - exact `q(x,z)` formula
  - `forward_input_is_physical_q: true`
  - `model_output_space: normalized_q`
  - explicit statement that later physics loss must unnormalize before calling `A(...)`
- geometry/operator settings copied from the locked operator authority:
  - `mode: born`
  - `grid_size=128`
  - `detector_size=128`
  - `angles=64` full-view coverage
  - `wavelength_px=8.0`
  - `medium_ri=1.333`
  - sinogram layout `(A, D, 2)` or split real/imag arrays with equivalent meaning
- split contract:
  - deterministic split seed
  - exact train/val/test counts
  - disjoint object-seed policy
  - train-only normalization statistics
- sample content:
  - `q_true_physical`
  - `q_true_norm`
  - complex sinogram real/imag channels
  - `angle_mask`
  - per-sample generator family
  - per-sample seed
- phantom and noise contract:
  - at least two non-CDI-only weak-scattering phantom families, with the preferred minimum being overlapping ellipses plus blob/inclusion families
  - refractive-index contrast range kept in the weak-scattering regime
  - fixed recorded `noise_sigma` in physical sinogram units
  - measured SNR summary
- dry-run geometry summary:
  - requested versus locked operator geometry
  - estimated artifact paths and counts
  - any missing dependency/storage/path issue
  - explicit verdict of `ready_for_smoke_generation` or `not_ready`
- claim boundary:
  - feasibility-only language
  - explicit statement that the later larger decision-support split, adapters, and four-row preflight remain out of scope

## Execution Tranches

### Tranche 1: Lock The Dataset Contract

**Purpose:** Freeze the physical target, normalization semantics, split policy, and output schema before any arrays are generated.

- [ ] Create or update a dataset-contract helper surface under `scripts/studies/born_rytov_dt/` that exposes:
  - the canonical physical-target formula for `q`
  - train-only normalization-stat computation
  - `normalize_q` / `unnormalize_q` helpers or equivalent explicit logic
  - manifest-building helpers with stable keys
- [ ] Make the manifest contract carry `forward_input_is_physical_q: true` and `model_output_space: normalized_q`.
- [ ] Lock the canonical smoke geometry to the operator-validated BRDT preflight contract:
  - `grid_size=128`
  - `detector_size=128`
  - `64` full-view angles
  - `wavelength_px=8.0`
  - `medium_ri=1.333`
  - Born mode only
- [ ] Lock a small deterministic smoke split. Use a fixed split seed and a small explicit count budget such as `16 train / 4 val / 4 test` unless a stricter local runtime constraint forces a smaller documented count.
- [ ] Lock the minimum phantom-family roster so the dataset is not just CDI line patterns. The preferred minimum is:
  - overlapping ellipses
  - soft blob/cell-like phantoms
  - sparse fine inclusions or annular inclusions
- [ ] Add tests first in `tests/studies/test_born_rytov_dt_dataset.py` for:
  - manifest-key stability
  - train-only normalization statistics
  - normalize/unnormalize round-trip
  - split determinism and object-seed disjointness
  - explicit rejection of routes that would feed normalized `q` to the operator contract without an unnormalize step

**Verification**

- Blocking: run the relevant focused selector(s) from `tests/studies/test_born_rytov_dt_dataset.py` that cover manifest, normalization, and split helpers after the helper surface exists.
- Supporting: inspect the resulting manifest helper output with a tiny in-memory fixture and confirm the locked operator fields match the passed operator report exactly.

### Tranche 2: Implement The Dry-Run And Smoke Dataset Generator

**Purpose:** Build the deterministic generator that can validate geometry without arrays and then emit the small smoke dataset under the locked contract.

- [ ] Implement `python -m scripts.studies.born_rytov_dt.generate_brdt_dataset` with:
  - an ignored/external `--output-root`
  - deterministic `--split-seed`
  - a geometry-only `--dry-run-manifest` mode
  - small explicit split counts
  - recorded noise configuration
  - recorded operator-validation input path
- [ ] In dry-run mode, load/consume the operator authority from `operator_validation.json`, validate the requested geometry against it, and write `dry_run_summary.json` plus a manifest skeleton without generating full arrays.
- [ ] In live smoke mode, generate physical weak-scattering phantoms, convert them to physical `q`, evaluate the locked Born operator to create complex sinograms, compute train-only normalization statistics from the train split only, and write:
  - train/val/test HDF5 files
  - `dataset_manifest.json`
  - provenance/log artifacts under the item artifact root
- [ ] Store the required arrays and metadata:
  - `q_true_physical`
  - `q_true_norm`
  - `sinogram_real`
  - `sinogram_imag`
  - `angle_mask`
  - per-sample seed and generator family
- [ ] Keep the dry-run summary and live manifest explicit that this is the smoke/preflight dataset, not the later decision-support split.
- [ ] Do not add classical backprop initialization images unless they are needed purely for a geometry summary; the selected backlog scope does not require `init_born_*` outputs in this item.

**Verification**

- Blocking: run the dry-run CLI first and verify that `dry_run_summary.json` is emitted with a `ready_for_smoke_generation` verdict before launching the live smoke generation.
- Blocking: run one live smoke generation into the default artifact root and verify that fresh train/val/test files plus `dataset_manifest.json` are written.
- Supporting: inspect one generated file/schema and confirm the expected datasets, dtypes, and shapes are present for the stored fields above.
- Supporting: if generation runtime is long enough to matter operationally, relaunch once in `tmux` with `ptycho311` active and keep the tracked process under ownership until completion.

### Tranche 3: Durable Summary, Discoverability, And Final Gates

**Purpose:** Check in the human-readable preflight authority, keep the evidence indexes current, and finish with deterministic repo checks.

- [ ] Write `docs/plans/NEURIPS-HYBRID-RESNET-2026/brdt_dataset_preflight.md` with:
  - the selected backlog item ID
  - prerequisite/operator-validation status
  - the locked `q` formula and normalization rule
  - smoke split counts and phantom-family roster
  - output-root and artifact paths
  - dry-run outcome
  - feasibility-only claim boundary
  - explicit handoff note that adapters and four-row preflight are next items
- [ ] Update `docs/index.md` and `docs/studies/index.md` so the BRDT dataset preflight summary is discoverable next to the candidate design and operator validation report.
- [ ] Update `docs/plans/NEURIPS-HYBRID-RESNET-2026/evidence_matrix.md` and `docs/plans/NEURIPS-HYBRID-RESNET-2026/paper_evidence_index.md` so the new summary and artifact root are visible and correctly labeled as `feasibility`, not manuscript evidence.
- [ ] Keep any recorded limitations explicit. If the smoke dataset had to shrink counts or omit optional fields, document that in the summary rather than hiding it.

**Verification**

- Blocking: `pytest -q tests/studies/test_born_rytov_dt_dataset.py`
- Blocking: `python -m compileall -q scripts/studies/born_rytov_dt`
- Supporting: reopen the checked-in summary and confirm it points to the live artifact root, operator report, and machine-readable manifest paths actually produced in Tranche 2.

## Completion Standard

This item is complete only when:

- the operator validation prerequisite is explicitly consumed rather than redefined;
- the BRDT smoke dataset contract is locked around physical `q`, train-only normalization, deterministic splits, and feasibility-only claim boundaries;
- the dry-run summary and live manifest exist under the item artifact root;
- `tests/studies/test_born_rytov_dt_dataset.py` passes;
- `python -m compileall -q scripts/studies/born_rytov_dt` passes;
- the durable summary and evidence-index updates make the dataset preflight discoverable for `2026-04-29-brdt-task-adapters` and `2026-04-29-brdt-four-row-preflight`.
