# PtychoViT Workflow (Planned)

**Status: Draft (planned, not yet implemented in PtychoPINN).**

This document defines the intended integration contract for running a `pinn_ptychovit` model arm from the grid-lines study workflow. It is a design-time guide and should not be treated as a runnable workflow until the implementation plan reaches first green verification.

## Scope

- Add `pinn_ptychovit` as a selectable model arm in studies.
- Use subprocess execution against an external `ptycho-vit` checkout.
- Keep data adaptation isolated in a dedicated interop layer (`NPZ -> paired HDF5`).
- Compare explicitly selected models only.
- Evaluate cross-model metrics on a canonical `256 x 256` grid.

## Backend Contract

- Model ID: `pinn_ptychovit`
- Supported resolution: `N=256` only (initial contract).
- Input format expected by PtychoViT: paired files
  - `*_dp.hdf5` with `dp` dataset
  - `*_para.hdf5` with `object`, `probe`, `probe_position_x_m`, `probe_position_y_m`
- Adapter-owned artifacts (planned):
  - paired HDF5 files
  - normalization pickle
  - manifest/provenance file

## Checkpoint Restore

Planned artifact semantics for restore paths:

- `best_model.pth`
  - best validation checkpoint for inference/default fine-tuning bootstrap.
- `checkpoint_model.pth`
  - latest training-step model weights for interrupted-run resume.
- `checkpoint.state`
  - epoch, optimizer/scheduler state, and tracked loss history.

Planned restore decision table:

- Resume interrupted run:
  - load `checkpoint_model.pth` + `checkpoint.state`
  - set `resume_from_checkpoint=true`
- Fine-tune from prior run:
  - initialize from `best_model.pth`
  - start new run metadata/output directory
- Inference-only:
  - load `best_model.pth` unless explicitly overridden

## Fine-Tuning

Planned policy: progressive unfreezing.

- Stage 1: decoder-only warmup
- Stage 2: unfreeze top encoder blocks (lower LR than decoders)
- Stage 3: optional full unfreeze if validation justifies it

Planned config topics to document with concrete examples:

- `resume_from_checkpoint`
- component learning rates (`encoder_lr`, decoder learning rates)
- run numbering and output directory isolation
- checkpoint selection policy (`best_model.pth` vs `checkpoint_model.pth`)

## Inference

Planned inputs:

- model artifacts (`best_model.pth`, config snapshot)
- paired test HDF5 files (`*_dp.hdf5`, `*_para.hdf5`)
- normalization source (dictionary/fallback policy)

Planned outputs:

- study run logs (stdout/stderr)
- reconstruction artifact under studies output (`recons/pinn_ptychovit/recon.npz`)
- per-model metrics entry in `metrics_by_model.json`

## Evaluation Policy

- Native model execution may use different `N` across selected arms.
- Cross-model metrics are computed after harmonizing reconstructions to canonical `256 x 256`.
- No baseline-comparison requirement by default.
- No physical-unit harmonization in v1 (pixel-space only).

## Runbook

Planned runbook sections to be finalized after implementation:

- restore + resume command sequence
- fresh fine-tune from prior checkpoint
- inference-only command sequence
- smoke-validation checklist
- common failure signatures and remediation

## Publication Gate

Before this workflow is linked from `docs/index.md` or command references:

1. Adapter conversion tests pass.
2. Compatibility validator tests pass.
3. Runner subprocess tests pass.
4. End-to-end smoke run emits expected artifacts and metrics.
