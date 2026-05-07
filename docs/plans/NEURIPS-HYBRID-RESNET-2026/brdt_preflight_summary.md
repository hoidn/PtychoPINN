# BRDT Preflight Summary And Promotion Decision

> **Owning backlog item:** `2026-04-29-brdt-preflight-summary-promotion-decision`
> (closed-out summary authority; this item itself does **not** authorize any
> BRDT promotion into manuscript tables, figures, or `/home/ollie/Documents/neurips/`).

## 1. Identity And Scope Boundary

- Initiative: `NEURIPS-HYBRID-RESNET-2026`.
- Lane: Born-Rytov diffraction tomography (BRDT) candidate study.
- Manuscript role per steering and roadmap: **additional candidate work only.**
  The required NeurIPS pillars remain CDI `lines128` and PDEBench CNS; BRDT is
  not a roadmap pillar.
- Bundle claim boundary: `decision_support_preflight_only`.
- This summary is the durable evidence authority for the BRDT four-row
  preflight. Later backlog selection or BRDT umbrella close-out can consume
  this file instead of reopening the raw artifact tree.

## 2. Final Recommendation

**Recommendation token:** `defer_after_preflight`

**Follow-up routing:**

- Preserve the BRDT four-row preflight as deferred candidate evidence only.
- Do **not** spin up a BRDT evidence-amendment plan against the current
  manuscript budget.
- Do **not** promote any BRDT row into manuscript tables, figures, or the
  `/home/ollie/Documents/neurips/` evidence bundle.
- If a future campaign elects to revisit BRDT, that decision must (a) restore
  the classical Born backprop reference under ODTbrain or an equally
  authoritative non-ML inversion, (b) lift the bundle from
  `decision_support_preflight_only` to a full-training contract under a fresh
  written plan, and (c) author a separate roadmap or evidence-amendment plan
  that names exact rows, budgets, fairness contract, artifacts, and claim
  boundaries.

### Rationale

The Born/forward operator, decision-support dataset, and task-adapter
contracts are scientifically valid and internally consistent (see Section 3).
Hybrid ResNet shows a clear and large differentiating advantage over both
U-Net and the vanilla FNO row at this capped budget on every locked image- and
measurement-space metric (see Section 6). However, the classical Born
backprop row is blocked because ODTbrain is unavailable in the local
environment, and the bundle is a capped 2048/256/256 decision-support split
trained for 20 epochs rather than a full-training benchmark. With the
non-ML baseline unrun, the BRDT comparison is materially incomplete relative
to a fair-promotion contract, and the current submission roadmap continues to
prioritize the CDI `lines128` and PDEBench CNS pillars. The lane is real and
worth preserving, but immediate manuscript-evidence follow-up is not
justified — `defer_after_preflight` is the rule-fit outcome.

## 3. Prerequisite Status

All four prerequisite backlog items are closed and their authorities are
checked in:

| Prerequisite | Authority | Status |
|---|---|---|
| `2026-04-29-brdt-operator-validation` | [BRDT operator validation report](brdt_operator_validation_report.md) | locked: `BornRytovForward2D` Born mode, `unitary_fft`, Wolf-1969 sampling |
| `2026-04-29-brdt-dataset-preflight` | [BRDT dataset preflight summary](brdt_dataset_preflight.md) | locked: physical `q(x,z)`, train-only normalization, deterministic seeding, unnormalize-before-physics |
| `2026-04-29-brdt-task-adapters` | [BRDT task adapters summary](brdt_task_adapters.md) | locked: `born_init_image` input, supervised + Born consistency loss, real-channel adapters; ODTbrain already noted as unavailable locally |
| `2026-04-29-brdt-four-row-preflight` | this summary | bundle complete; classical row blocked on ODTbrain |

## 4. Operator Validation Result

- Validation artifact:
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-brdt-operator-validation/operator_validation.json`.
- Geometry locked in the four-row bundle: Born mode, `grid_size=128`,
  `detector_size=128`, `angle_count=64`, `wavelength_px=8.0`,
  `medium_ri=1.333`, `normalize=unitary_fft`.
- The same validation artifact path is recorded as `operator_version` on
  every per-row `row_summary.json` and on the bundle-level
  `preflight_manifest.json`, so all four rows share one operator authority.

## 5. Dataset And Normalization Validity

- Dataset id: `brdt128_decision_support_preflight` (capped decision-support
  tier).
- Dataset manifest:
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-brdt-four-row-preflight/decision_support_dataset/dataset_manifest.json`.
- Split counts: `train=2048`, `val=256`, `test=256`.
- Train-only physical-`q` normalization recorded by the bundle:
  - `q_min_train = -0.02745276702843244`
  - `q_max_train =  0.028077640329534873`
  - `q_mean_train = 0.00128355410049707`
  - `q_std_train  = 0.005439489003905311`
- The bundle is consuming the capped decision-support dataset (not the earlier
  smoke dataset). All four rows record `dataset_id =
  brdt128_decision_support_preflight`.

## 6. Four-Row Contract, Source Artifact Root, And Row Roster

- Source artifact root:
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-brdt-four-row-preflight/`.
- Bundle artifacts (all present):
  `preflight_manifest.json`, `metrics.json`, `metrics.csv`,
  `metric_schema.json`, `visual_manifest.json`, `rows/`,
  `visuals/`, `figures/source_arrays/`,
  `decision_support_dataset/dataset_manifest.json`.
- Training contract: `epochs=20`, `batch_size=16`, `learning_rate=2e-4`,
  optimizer Adam, fixed `seed=42`, fixed-sample seed `17` with sample ids
  `[145, 83, 255, 126]`.
- Loss weights: `image=1.0`, `physics=0.1`, `relative_physics=0.1`,
  `tv=1e-5`, `positivity=1e-4`.
- Input mode: `born_init_image` for every row in this historical preflight
  contract. Sinogram-input manuscript work is a separate 2026-05-07 contract:
  learned models consume measured complex sinograms directly, and the Born
  inverse is a non-learned reference rather than model input.
- Per-row contract fingerprints (lifted from `preflight_manifest.json`):
  - `classical_born_backprop`: `dd2f447822fd6566db3c308f0b4ae009b63c79673fed628e72902dc1e5f15b6c`
  - `unet`: `bbc3f26d427bed00d741e2b7c6f14fc75a48b24f10f9d86ef7555203023c0a93`
  - `fno_vanilla`: `792dea76c13bfdb14c5e81800fa2be501169a6c3f468ef47ae9de29736842da2`
  - `hybrid_resnet`: `070559ea52d2e3d1192cec79ce740665bdb3c3c21d9f9d09224a11c9353d8915`
- No rows were resumed; `resumed_rows` is empty in the manifest.

### Row Metrics (Image-Space Physical `q`)

Image-space metrics use physical `q`. Lower is better for
`image_mae_phys` / `image_rmse_phys` / `image_relative_l2_phys`; higher is
better for `psnr_phys` / `ssim_phys`.

| Row | Status | image_mae_phys | image_rmse_phys | image_relative_l2_phys | PSNR_phys | SSIM_phys | Params | Train wall (s) | Eval wall (s) |
|---|---|---|---|---|---|---|---|---|---|
| `classical_born_backprop` | **blocked** | n/a | n/a | n/a | n/a | n/a | 0 | 0.00 | 0.00 |
| `unet` | completed | 0.001827 | 0.004899 | 0.8638 | 21.088 | 0.6129 | 18,465 | 69.001 | 0.611 |
| `fno_vanilla` | completed | 0.001947 | 0.005624 | 0.9916 | 19.889 | 0.6779 | 44,465 | 85.110 | 0.656 |
| `hybrid_resnet` | completed | 0.000650 | 0.001809 | 0.3190 | 29.741 | 0.9471 | 142,018 | 79.254 | 0.641 |

### Row Metrics (Measurement Space)

| Row | Status | meas_mae | meas_rmse | meas_relative_l2 |
|---|---|---|---|---|
| `classical_born_backprop` | blocked | n/a | n/a | n/a |
| `unet` | completed | 0.002803 | 0.005823 | 0.6875 |
| `fno_vanilla` | completed | 0.003568 | 0.008307 | 0.9808 |
| `hybrid_resnet` | completed | 0.001080 | 0.001687 | 0.1992 |

### Read

- `hybrid_resnet` is the clear leader on every blocking image-space and
  measurement-space metric and on both supporting metrics, beating the U-Net
  and `fno_vanilla` rows by a wide margin.
- `unet` and `fno_vanilla` both fail to reduce image-space relative L2 below
  `0.86`, i.e. the residual error is on the order of the target field itself.
- These outcomes are not paper-grade evidence under the current claim
  boundary; they are decision-support inputs.

### Runtime And Provenance

- Device: `cuda` on `NVIDIA GeForce RTX 3090`.
- Python `3.11.13`, PyTorch `2.9.1+cu128`, dataset writer `numpy 1.26.4`,
  `h5py 3.14.0`.
- Per-row `invocation.json` and `invocation.sh` files record the exact
  orchestrator command (`scripts/studies/born_rytov_dt/run_preflight.py`)
  and the seeded fixed-sample contract.
- The four-row execution report records the seeded rerun contract
  (writer-lock + per-row provenance fixes were committed under the
  `cb1faaea`/`0a7e0a0b` series before this summary).

## 7. Visual Bundle And Source-Array Availability

- Three figure roll-ups under `visuals/`:
  `brdt_compare_q.png`, `brdt_error_q.png`, `brdt_sinogram_residual.png`.
- Source `.npy` arrays present under `figures/source_arrays/` for every
  fixed sample id and every neural row, plus shared
  `*_q_target.npy` and `*_sino_obs.npy` arrays for sample ids
  `[145, 83, 255, 126]` (12 row-specific predictions across U-Net,
  FNO vanilla, Hybrid ResNet, plus 4 targets and 4 sinogram observations).
- The classical row contributes no source arrays, consistent with its
  blocked status.

## 8. Dependency And Environment Issues

- ODTbrain remains unavailable in the local environment after two narrow-fix
  attempts recorded directly in `preflight_manifest.json`:
  - `import_odtbrain` → `ModuleNotFoundError: No module named 'odtbrain'`
  - `retry_after_invalidate_caches` → same error.
- Effect: only the local-adjoint backend is available, and that backend is
  `feasibility_only`. Rather than silently fall through to local adjoint,
  the orchestrator marked the classical row as `row_status=blocked` with
  `blocker_reason=odtbrain_unavailable`.
- Scope of the blocker: lane-wide for any classical reference row, not
  row-local in a recoverable sense — every classical reference row in this
  bundle would face the same import error.
- `neuralop` is available (`true` in `dependency_status`).
- No environment-side issues affect the U-Net, FNO vanilla, or Hybrid ResNet
  rows.

## 9. Known Limitations And Claim Boundary

- BRDT remains additional candidate work; it is **not** a manuscript pillar.
- The bundle is `decision_support_preflight_only`. It is **not** a
  full-training BRDT benchmark and may not be cited as benchmark-grade
  evidence. Capped/sub-paper rows under the steering rules are decision
  support only.
- The classical Born backprop comparator is missing because ODTbrain is
  unavailable. The Hybrid ResNet advantage over U-Net and FNO vanilla cannot
  be promoted to a fairness claim against a non-ML inversion baseline from
  this bundle alone.
- 20 epochs at batch 16 over a 2048-sample train split is a deliberately
  short budget. The metric ordering may be valid as routing evidence but does
  not establish converged-row outcomes.
- The image-space residual `relative_l2_phys ≈ 0.32` for the strongest row
  still indicates a noticeable error fraction relative to the target field
  norm; even the leader is not a clean inversion.
- This summary item itself does **not** authorize promotion into manuscript
  tables, figures, or `/home/ollie/Documents/neurips/` outputs. Any later
  manuscript use must be authorized by a separately written
  roadmap/evidence-amendment plan.

## 10. Why Not Promote, Why Not Reject

The other two recommendation tokens defined for this item (the
"promote-to-evidence-amendment-plan" and "reject-for-current-manuscript"
outcomes) were considered and rejected.

- The promotion-style outcome is rejected here because:
  - the classical reference row is blocked, so the fairness contract against
    a non-ML inversion is incomplete;
  - the bundle is decision-support only and not on a full-training contract;
  - the steering pillars (CDI `lines128`, PDEBench CNS) and the current
    progress ledger do not authorize redirecting manuscript budget to BRDT
    follow-up at this point.
- The rejection-style outcome is rejected here because:
  - the operator/dataset/adapter chain remains scientifically valid;
  - Hybrid ResNet shows a real and large differentiating advantage on every
    locked metric, which is not a result a healthy program would discard;
  - the ODTbrain blocker is an environment problem, not a deeper scientific
    or fairness flaw, and is recoverable by a future plan that resolves the
    dependency.
- The chosen recommendation token (see Section 2) therefore remains the
  rule-fit outcome.

## 11. Verification Pointers

Deterministic checks for this summary live under the work directory:

`.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-brdt-preflight-summary-promotion-decision/verification/`

Run logs are archived there for reproducibility.

## 12. Discoverability

- The durable index entry for `2026-04-29-brdt-four-row-preflight` in
  [paper_evidence_index.md](paper_evidence_index.md) points at this summary
  as the owning authority.
- `docs/index.md` is intentionally not edited from this item:
  `paper_evidence_index.md` is the intended discoverability surface for this
  candidate-lane summary, since BRDT is candidate work and not a manuscript
  pillar.
