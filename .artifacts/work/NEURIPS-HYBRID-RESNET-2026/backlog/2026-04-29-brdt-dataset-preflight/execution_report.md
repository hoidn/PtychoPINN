# BRDT Dataset Preflight Execution Report

- Backlog item: `2026-04-29-brdt-dataset-preflight`
- Plan: `docs/plans/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-brdt-dataset-preflight/execution_plan.md`
- State: `COMPLETED`
- Tier: `feasibility` (additive candidate work; not manuscript evidence)

## Completed In This Pass

- Fixed the review-blocking reproducibility defect in
  `scripts/studies/born_rytov_dt/generate_brdt_dataset.py` by replacing
  the process-randomized `hash(split)` noise seed path with the stable
  dataset-contract helper
  `dataset_contract.deterministic_noise_seed(split_seed, split)`.
  Fresh-process reruns now reproduce identical noisy sinograms and
  identical measured SNRs for the same CLI inputs.
- Fixed provenance capture by deriving the stored generation command
  from the actual invoked CLI arguments. Dry-run artifacts now record
  `python -m scripts.studies.born_rytov_dt.generate_brdt_dataset --dry-run-manifest`
  instead of a flag-less placeholder, and live artifacts retain the
  exact flag set used for generation.
- Completed the missing dry-run manifest-skeleton task. Dry-run mode now
  writes `dry_run_manifest.json` alongside `dry_run_summary.json`, using
  the live manifest schema with `normalization: null`,
  `noise.measured_snr: null`, and
  `extra.generation_mode: dry_run_manifest`.
- Added regression coverage in
  `tests/studies/test_born_rytov_dt_dataset.py` for:
  - dry-run manifest-skeleton emission
  - exact command capture with non-default flags
  - fresh-process live-generation reproducibility via stable HDF5 hashes
    and stable `measured_snr`
- Refreshed the default smoke artifacts under
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-brdt-dataset-preflight/`
  so the checked-in machine-readable evidence matches the fixed
  generator:
  - `dataset_manifest.json`
  - `dry_run_manifest.json`
  - `dry_run_summary.json`
  - `dataset/brdt128_sparse_fullview_preflight_{train,val,test}.h5`
- Updated durable discoverability/docs surfaces to mention the dry-run
  manifest skeleton and deterministic-noise/provenance fixes:
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/brdt_dataset_preflight.md`
  - `docs/index.md`
  - `docs/studies/index.md`
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/paper_evidence_index.md`

## Completed Current-Scope Work

- The BRDT smoke-dataset contract remains locked around physical
  `q(x,z)=k_m^2((n/n_m)^2-1)`, train-only normalization,
  `forward_input_is_physical_q: true`, `model_output_space:
  normalized_q`, deterministic disjoint object seeds, and feasibility-
  only claim boundaries.
- The generator now satisfies the full current plan contract:
  dry-run geometry validation, dry-run summary plus manifest skeleton,
  exact command provenance, deterministic live generation, and live
  HDF5/manifest emission under the approved artifact root.
- The approved backlog verification contract still passes, and the new
  regression coverage directly covers the review findings that caused
  `REVISE`.

## Follow-Up Work

- `2026-04-29-brdt-task-adapters`: add Lightning/data-loading/loss
  surfaces that consume the locked smoke-dataset contract without
  renaming the stored HDF5 fields.
- `2026-04-29-brdt-four-row-preflight`: run the bounded classical/U-Net/
  FNO/SRU-or-Hybrid candidate-lane preflight on top of this dataset
  authority.

## Residual Risks

- ODTbrain inverse-side recovery is still not exercised locally
  (`brdt_operator_validation_report.md` remains
  `pass_with_documented_limits`); this stays a documented BRDT-lane
  limit but does not block dataset preflight.
- The smoke dataset is intentionally tiny (24 samples). It is suitable
  for feasibility and adapter bring-up only; anything benchmark-like
  must regenerate the later larger decision-support split under a
  separately authorized item.
- The HDF5 schema is now stable enough for downstream consumers to rely
  on, including the new dry-run skeleton, but renaming core stored
  fields (`q_true_physical`, `q_true_norm`, `sinogram_real`,
  `sinogram_imag`, `angle_mask`, `sample_seed`, `phantom_family`) would
  still be a contract break requiring explicit follow-up approval.

## Verification

- Review regression subset:
  `pytest -q tests/studies/test_born_rytov_dt_dataset.py -k 'dry_run_writes_manifest_skeleton or live_generation_reproducible_across_fresh_processes'`
  → `2 passed`.
- Blocking pytest:
  `pytest -q tests/studies/test_born_rytov_dt_dataset.py`
  → passes.
- Blocking compileall:
  `python -m compileall -q scripts/studies/born_rytov_dt`
  → success (no output, exit `0`).
- Dry-run refresh:
  `python -m scripts.studies.born_rytov_dt.generate_brdt_dataset --dry-run-manifest`
  → wrote `dry_run_summary.json` and `dry_run_manifest.json` with
  `verdict: ready_for_smoke_generation`, zero geometry mismatches, and
  the exact dry-run command string.
- Live refresh:
  `python -m scripts.studies.born_rytov_dt.generate_brdt_dataset`
  → rewrote `dataset_manifest.json` plus the three HDF5 splits with
  stable deterministic content and recorded SNR values:
  `train_db=18.465376487786692`, `val_db=16.3747234610113`,
  `test_db=19.309521118789167`.
