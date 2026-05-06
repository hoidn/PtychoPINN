# BRDT Supervised+Born 40-Epoch Paper Evidence Summary

> **Owning backlog item:** `2026-05-05-brdt-supervised-born-40ep-paper-evidence`
> **Final claim boundary:** `paper_evidence_brdt_additive`
> **Promotion status:** `passed`
> **Caveat:** additive bounded evidence only. This does **not** replace the
> required CDI `lines128` or PDEBench CNS pillars, and it does not authorize
> `/home/ollie/Documents/neurips/` publication in this phase.

## 1. Identity And Locked Contract

- Initiative: `NEURIPS-HYBRID-RESNET-2026`.
- Lane: Born-Rytov diffraction tomography (BRDT) additive candidate study.
- Fresh immutable artifact root:
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-05-brdt-supervised-born-40ep-paper-evidence/`
- Read-only lineage inputs:
  - `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-brdt-four-row-preflight/`
  - `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-04-brdt-ffno-row-extension/`
- Locked same-contract fields inherited from the earlier BRDT bundle:
  - dataset id `brdt128_decision_support_preflight`
  - split `2048 / 256 / 256`
  - operator `BornRytovForward2D`, Born mode, `N=D=128`, `A=64`,
    `wavelength_px=8.0`, `medium_ri=1.333`, `normalize=unitary_fft`
  - input mode `born_init_image`
  - train-only physical-`q` normalization
  - loss weights `image=1.0`, `physics=0.1`, `relative_physics=0.1`,
    `tv=1e-5`, `positivity=1e-4`
  - fixed samples `[145, 83, 255, 126]` with sample `255` required for the
    paper-facing visual bundle
- Changed fields for this additive promotion attempt only:
  - rows rerun: `hybrid_resnet`, `ffno`
  - epochs `40`
  - optimizer Adam, `lr=2e-4`
  - scheduler `ReduceLROnPlateau(factor=0.5, patience=2, threshold=0.0, min_lr=1e-5)`
  - batch size `16`
  - seed `42`
  - per-epoch `history.json` / `history.csv`

## 2. Outcome

The fresh 40-epoch rerun improves both neural rows against their frozen
20-epoch authorities under the same dataset, operator, split, input, and loss
contract. The gate now passes because the bundle has complete provenance,
sample-`255` compare/error/source-array assets, same-contract lineage to the
frozen 20-epoch rows, and a checked-in evidence-package amendment that matches
the run artifacts.

Hybrid ResNet remains the strongest image-space BRDT row. FFNO remains
competitive at much lower parameter count, but does not displace Hybrid ResNet
on this capped contract.

## 3. 20-Epoch To 40-Epoch Delta Read

| Row | image_relative_l2_phys | meas_relative_l2 | PSNR_phys | SSIM_phys | Final LR | LR drops | Improving at stop |
|---|---:|---:|---:|---:|---:|---:|---|
| `hybrid_resnet` | `0.3190 -> 0.2875` (`-0.0315`) | `0.1992 -> 0.1739` (`-0.0253`) | `29.74 -> 30.64` (`+0.90`) | `0.9471 -> 0.9556` (`+0.0085`) | `2e-4` | `0` | `true` |
| `ffno` | `0.3421 -> 0.3324` (`-0.0097`) | `0.1859 -> 0.1784` (`-0.0075`) | `29.13 -> 29.38` (`+0.25`) | `0.9420 -> 0.9455` (`+0.0035`) | `2e-4` | `0` | `true` |

Convergence read from `convergence_audit.json`:

- Both rerun rows reached the planned `40` history records and retained the
  planned scheduler configuration.
- Both rows improved on every tracked blocking metric versus the frozen
  20-epoch lineage.
- Neither row triggered a plateau LR reduction, and both remained materially
  improving over the last 5 and 10 epochs. This is a bounded additive-evidence
  result, not a claim that either row is fully converged.

## 4. Sample-255 Visual Provenance Decision

The paper-facing visual authority is the fresh sample-`255` bundle under:

- `visuals/sample_0255_compare_q.png`
- `visuals/sample_0255_error_q.png`
- `visuals/sample_0255_sinogram_residual.png`
- `figures/source_arrays/sample_0255_*`

The classical/model-based comparator for sample `255` is accepted from the
frozen `2026-04-29` baseline bundle's recorded same-contract row
`classical_born_backprop`, whose baseline `metrics.json` and source arrays mark
that row `row_status="completed"` with `paper_label="Model-based Born inverse"`.
This summary resolves the earlier stale narrative inconsistency in
`brdt_preflight_summary.md` in favor of the concrete baseline artifact payload
actually consumed by the fresh 40-epoch bundle.

## 5. Gate Result And Reasons

- Final claim boundary: `paper_evidence_brdt_additive`
- Promotion status: `passed`
- Lane status: completed, additive-only

Promotion passed because all required gate conditions were satisfied:

- both rerun rows completed successfully
- both rows emitted `40` history records
- scheduler fields matched the locked plan contract
- runtime provenance, dataset identity, split manifest, run-log, and exit-code
  proof were present
- sample-`255` compare/error/source-array assets were present with the
  same-contract classical comparator
- checked-in evidence surfaces were updated to the same backlog item, artifact
  root, and claim boundary

This promotion remains deliberately narrow:

- BRDT is still bounded capped evidence, not a new primary manuscript pillar
- the additive lane does not replace CDI `lines128` or PDEBench CNS
- no full-training BRDT competitiveness claim is authorized

## 6. Residual Risks

- The rows were still improving at stop and never reduced LR, so the correct
  interpretation is additive bounded evidence rather than fully converged BRDT
  performance.
- The classical comparator used for the sample-`255` figure is inherited by
  lineage from the baseline bundle rather than regenerated in this pass.
- This item is repo-local only; `/home/ollie/Documents/neurips/` remains out of
  scope.

## 7. Reproducibility And Meta Provenance

The fresh artifact root is reproducible end-to-end from the approved runner
in `scripts/studies/born_rytov_dt/run_brdt_40ep_paper_evidence.py`:

- The full live training run is invoked with the locked CLI listed in the
  execution plan and is guarded by a per-output-root writer lock that refuses
  duplicate writers and refuses to silently overwrite a populated bundle.
- The top-level `preflight_manifest.json` claim boundary and promotion status
  are re-seeded from the recomputed `paper_evidence_gate.json` after training
  completes, so the manifest cannot present a passing additive label without
  the gate honestly passing.
- The gate's `provenance_checks` payload validates that runtime provenance
  carries `git_sha`/`git_dirty`, host identity and GPU count, model profiles
  and the run log are present, dataset identity and split manifest exist,
  the sample-`255` visual bundle (including the same-contract classical
  comparator) is materialized, and the durable summary at this path is
  checked-in and references this backlog item.
- Meta artifacts (manifest, runtime provenance, run-exit status, convergence
  audit, gate, visuals) can be deterministically re-derived from existing
  per-row `history.{json,csv}`, `row_summary.json`, `model_profile.json`, and
  source arrays via the runner's `--rebuild-meta-only` mode without
  retraining. Run-exit-status preserves the original tracked PID; runtime
  provenance documents the rebuild step's git SHA, host, and GPU count.
