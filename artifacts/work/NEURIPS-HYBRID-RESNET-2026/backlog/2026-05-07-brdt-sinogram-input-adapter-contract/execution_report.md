# Execution Report

## Completed In This Pass

- Hardened `scripts/studies/born_rytov_dt/run_sinogram_input_smoke.py` so the
  smoke runner now owns the adapter-contract root end to end:
  - default root moved to
    `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-07-brdt-sinogram-input-adapter-contract/smoke`
  - per-row `invocation.json` / `invocation.sh` are emitted for `ffno` and
    `sru_net`
  - top-level `smoke_summary.json` is written at the backlog-item root
  - per-row adapter contracts are checked to confirm `input_mode="sinogram"`
- Added a regression test proving the smoke runner writes
  `smoke_summary.json`, keeps row artifacts under `smoke/`, and records the
  required sinogram-contract proof fields.
- Wrote the durable summary
  `docs/plans/NEURIPS-HYBRID-RESNET-2026/brdt_sinogram_input_adapter_contract.md`.
- Updated `docs/index.md`,
  `docs/plans/NEURIPS-HYBRID-RESNET-2026/evidence_matrix.md`, and
  `docs/plans/NEURIPS-HYBRID-RESNET-2026/ablation_index.json` so the new
  contract is discoverable as readiness-only authority while preserving the
  historical `born_init_image` lineage honestly.
- Executed the dedicated smoke command against the required adapter-contract
  root and verified the emitted artifacts.

## Completed Plan Tasks

- Task 1: completed
  - `SUPPORTED_INPUT_MODES` remains limited to `("born_init_image", "sinogram")`
  - `direct_sinogram` remains rejected
  - the sinogram-input successor dry-run continues to surface only
    `classical_born_backprop`, `ffno`, and `sru_net`
- Task 2: completed
  - targeted adapter tests confirm `ffno` and `sru_net` accept
    `(B, 2, 64, 128)` sinograms and emit `(B, 1, 128, 128)`
  - wrong-shape image tensors remain rejected for the sinogram adapter
- Task 3: completed
  - the smoke runner now writes the dedicated readiness-only root
  - `smoke_summary.json` records `input_mode="sinogram"`, the consumed dataset
    manifest, learned rows `ffno` / `sru_net`, per-row status, and proof that
    learned-model input plus Born-consistency targets both use the measured
    complex sinogram while the Born inverse stays non-learned reference only
- Task 4: completed
  - durable summary created
  - docs index, evidence matrix, and ablation index updated
  - `model_variant_index.json` intentionally left unchanged because this item
    added readiness-only adapter/smoke evidence and no benchmark-performance row
- Task 5: completed
  - all required deterministic gates below ran successfully

## Remaining Required Plan Tasks

- None.

## Verification

- `pytest --collect-only -q tests/studies/test_born_rytov_dt_adapters.py tests/studies/test_born_rytov_dt_preflight.py`
  - passed, `140` tests collected
- `pytest -q tests/studies/test_born_rytov_dt_adapters.py -k "sinogram or input_mode or model"`
  - passed, `12` passed
- `pytest -q tests/studies/test_born_rytov_dt_preflight.py -k "sinogram_input_40ep or input_mode"`
  - passed, `1` passed
- `pytest -q tests/studies/test_born_rytov_dt_preflight.py -k "run_sinogram_input_smoke_writes_summary_and_row_artifacts"`
  - passed, regression test for the new smoke-summary contract
- `python -m compileall -q scripts/studies/born_rytov_dt ptycho_torch`
  - passed
- docs/index parse gate from the plan
  - passed; `ablation_index.json` parses and all required docs exist
- `python -m scripts.studies.born_rytov_dt.run_sinogram_input_smoke --device cpu --output-root .artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-07-brdt-sinogram-input-adapter-contract/smoke`
  - passed
  - top-level artifact:
    `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-07-brdt-sinogram-input-adapter-contract/smoke_summary.json`
  - per-row artifacts confirmed:
    `smoke/ffno/{adapter_contract.json,invocation.json,invocation.sh}`
    and `smoke/sru_net/{adapter_contract.json,invocation.json,invocation.sh}`
  - summary proof fields confirmed:
    `input_mode="sinogram"`,
    `learned_rows=["ffno","sru_net"]`,
    `model_input_source="measured complex sinogram"`,
    `born_consistency_target_source="measured complex sinogram"`,
    `born_inverse_role="non_learned_reference_only"`

## Residual Risks

- This item is still readiness-only. It does not establish benchmark behavior
  or full-training BRDT competitiveness under the new sinogram-input contract.
- The smoke run was a short CPU fast-dev execution; the successor `40`-epoch
  item still owns any longer budget or paper-evidence interpretation.
- Historical BRDT bundles remain `born_init_image` lineage only. Downstream
  consumers must not mix those rows with the new sinogram-input lane unless a
  later same-contract summary explicitly authorizes it.
