# Execution Report

## Completed In This Pass

- recovered direct distributed-dataset evidence by extracting and inspecting
  the public
  `wavebench_dataset/time_varying/is/thick_lines_gaussian_lens.beton`
  member from the Zenodo archive, then recorded observed train/validation/test
  sample stats in
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-wavebench-inverse-source-preflight/verification/dataset_slice_inspection.json`
- corrected the durable contract to reflect the published `.beton` member
  rather than nonexistent staged `.npy` files, while keeping the raw
  `334 x 128` measurement shape explicit as an upstream generator/loader fact
- separated supervised readiness from physics-readiness, recorded the stable
  `1 x 128 x 128` measurement-image contract, named the in-repo adapter
  precedents, and fixed the first follow-up width decision to `C=32` plus
  `C=64`
- reclassified native FNO and U-Net checkpoint availability from
  `checkpoint_reusable` to `retrain_required`, recorded stable numeric
  forward-model thresholds, filled in the missing paper-bundle follow-up route,
  and added a narrow validator plus pytest selector for the preflight contract
- completed the missing Tranche 1 variant inventory by recording all six
  published inverse-source variants, their split semantics, and the
  fixed-medium versus GRF/OOD distinctions directly in the durable metadata and
  summary
- replaced the transient temp-checkout staging-path pin with the stable
  follow-up target
  `<wavebench repo>/wavebench_dataset/time_varying/is/`, then tightened the
  validator and pytest coverage so summary/metadata agreement on that contract
  is enforced

## Completed Current-Scope Work

- Tranche 1 complete: repo identity, dataset DOI/source, archive identity,
  access notes, setup risks, native baseline entry points, full inverse-source
  variant inventory, split semantics, and fixed-versus-GRF/OOD distinctions are
  now all recorded durably
- Tranche 2 complete: the selected variant remains
  `thick_lines_gaussian_lens`; the selected split and `9000 / 500 / 500`
  counts remain fixed; the distributed on-disk schema is now directly observed
  from the public `.beton` member; and observed sample-level value ranges are
  recorded for train, validation, and test examples
- Tranche 3 complete: native FNO and U-Net reuse surfaces remain identified,
  but the durable classification now correctly records `retrain_required`
  because exact checkpoint identifiers were not recovered
- Tranche 4 complete to preflight outcome: the summary and metadata now
  explicitly separate supervised readiness from physics-informed readiness,
  identify the shared measurement-image contract and adapter gap, and keep
  physics deferred because no reproduction metrics were measured
- Tranche 5 complete: the checked-in summary, machine-readable metadata,
  execution report, verification artifacts, and docs-hub discoverability all
  agree on the final status token, the full inverse-source variant inventory,
  and the stable follow-up staging target

## Verification

- required deterministic input check:
  `python - <<'PY' ...`
  - result: `wavebench preflight inputs present`
  - log:
    `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-wavebench-inverse-source-preflight/verification/required_inputs_check.log`
- distributed dataset-slice inspection:
  `python - <<'PY' ...`
  - result: the public selected-variant `.beton` member was inspected
    directly; the archive contains `10000` samples with `input` and `target`
    fields shaped `1 x 128 x 128`, and split-aligned sample stats were
    recorded
  - artifacts:
    `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-wavebench-inverse-source-preflight/verification/dataset_slice_inspection.json`
    and
    `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-wavebench-inverse-source-preflight/verification/dataset_slice_inspection.log`
- contract validator:
  `python scripts/studies/validate_wavebench_preflight_contract.py`
  - result: `wavebench preflight contract validated`
  - log:
    `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-wavebench-inverse-source-preflight/verification/contract_validator.log`
- stronger final consistency check:
  `python - <<'PY' ...`
  - result:
    `wavebench preflight summary, metadata, and index references are consistent`
  - log:
    `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-wavebench-inverse-source-preflight/verification/final_consistency_check.log`
- targeted regression selector:
  `pytest tests/studies/test_wavebench_preflight_contract.py -q`
  - result: `3 passed in 0.92s`
  - log:
    `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-wavebench-inverse-source-preflight/verification/pytest_wavebench_preflight_contract.log`

## Follow-Up Work

- stage the selected `.beton` member under
  `<wavebench repo>/wavebench_dataset/time_varying/is/` before any later loader
  smoke or training item
- resolve exact FNO and U-Net checkpoint identifiers or choose a native
  baseline retraining path explicitly
- provision a WaveBench-capable environment with `ffcv`, `jax`, `jwave`, and
  `ml_collections`, then run the forward-model reproduction gate before any
  physics-informed row is authorized
- keep paper-table, figure, and evidence-bundle assembly out of scope until
  supervised/native/physics follow-ups produce claim-bearing evidence

## Residual Risks

- the upstream README and code still disagree on archive and extracted-folder
  naming, so later execution must continue to treat the code path
  `wavebench_dataset/` as authoritative unless upstream documentation changes
- the public checkpoint folder is known, but exact selected-variant checkpoint
  filenames are still unresolved, so equal-footing native baseline reuse
  remains blocked pending manual provenance recovery or retraining
- physics-informed readiness remains unproven because waveform MAE, waveform
  RMSE, relative L2, and normalized residual were not measured; the approved
  reproduction thresholds are now durable, but they are still unsatisfied
