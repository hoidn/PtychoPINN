# Execution Report

## Completed In This Pass

- Fixed the smoke runner default packaging contract flagged in the
  implementation review (Medium #1):
  - both the function default and the CLI `--output-root` default in
    `scripts/studies/born_rytov_dt/run_sinogram_input_smoke.py` now resolve
    to `DEFAULT_OUTPUT_ROOT` directly (`.../2026-05-07-brdt-sinogram-input-adapter-contract/smoke`),
    eliminating the prior `smoke/smoke` double-segment that landed
    `smoke_summary.json` under `smoke/` instead of the backlog-item root
    on default invocations.
  - confirmed via `inspect.signature(...)` that both defaults end in a
    single `smoke` segment with no `smoke/smoke` substring.
- Added the negative regression guard required by Task 2 of the plan
  (Medium #2):
  - `tests/studies/test_born_rytov_dt_adapters.py::test_sinogram_input_mode_does_not_route_through_derive_born_init_image`
    monkeypatches `derive_born_init_image` to raise on both `train` and
    `evaluate` modules, then exercises `train._prepare_input` with
    `input_mode="sinogram"` and asserts the returned tensor has the
    `(B, 2, angles, detectors)` channels-first sinogram contract.
  - the test now fires if a future refactor silently routes the learned
    sinogram path through the Born inverse helper, locking in the
    "Born inverse remains non-learned reference only" contract.
- Re-ran the dedicated smoke command and confirmed:
  - `smoke_summary.json` lands at the backlog-item root (not under
    `smoke/`),
  - per-row `adapter_contract.json`, `eval_summary.json`,
    `invocation.json`, `invocation.sh` artifacts remain under
    `smoke/{ffno,sru_net}/`,
  - summary proof fields are unchanged (`input_mode="sinogram"`,
    `model_input_source="measured complex sinogram"`,
    `born_consistency_target_source="measured complex sinogram"`,
    `born_inverse_role="non_learned_reference_only"`).

## Completed Current-Scope Work

- Task 1 - Input-mode and runner-selection contract: completed
  (unchanged this pass).
- Task 2 - Sinogram-input adapter surface and regression coverage:
  completed; this pass added the explicit negative test for
  `derive_born_init_image` so the contract is now defended end-to-end.
- Task 3 - Train/eval/preflight/smoke routing: completed; this pass
  fixed the smoke runner default-packaging contract bug so an
  unparameterized invocation now writes to the correct adapter-contract
  root.
- Task 4 - Durable contract summary and discoverability updates:
  completed (unchanged this pass).
- Task 5 - Required final deterministic gate: re-run this pass; all
  required commands still pass with the new test in scope.

## Verification

- `pytest --collect-only -q tests/studies/test_born_rytov_dt_adapters.py tests/studies/test_born_rytov_dt_preflight.py`
  - passed; `141` tests collected (one more than the previous pass due
    to the new regression guard).
- `pytest -q tests/studies/test_born_rytov_dt_adapters.py -k "sinogram or input_mode or model"`
  - passed; `13` tests passed (12 prior + the new
    `test_sinogram_input_mode_does_not_route_through_derive_born_init_image`).
- `pytest -q tests/studies/test_born_rytov_dt_preflight.py -k "sinogram_input_40ep or input_mode"`
  - passed; `1` test passed.
- `pytest -q tests/studies/test_born_rytov_dt_preflight.py -k "run_sinogram_input_smoke_writes_summary_and_row_artifacts"`
  - passed; covers the smoke-runner summary contract.
- `python -m compileall -q scripts/studies/born_rytov_dt ptycho_torch`
  - passed.
- `python -m scripts.studies.born_rytov_dt.run_sinogram_input_smoke --device cpu --output-root .artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-07-brdt-sinogram-input-adapter-contract/smoke`
  - passed; `smoke_summary.json` written at the backlog-item root with
    the required proof fields.
- Docs/index parse gate from the plan
  - passed; `ablation_index.json` parses and all required docs exist.
- Default-output-root sanity check (Medium #1 fix verification):
  - both `inspect.signature(run_sinogram_input_smoke).parameters['output_root'].default`
    and the `argparse` default resolve to
    `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-07-brdt-sinogram-input-adapter-contract/smoke`
    with no `smoke/smoke` substring.

## Follow-Up Work

- The successor `2026-05-07-brdt-sinogram-input-40ep-paper-evidence` item
  still owns the `40`-epoch run, manuscript-asset retargeting, and
  `paper_evidence_index.md` / `paper_evidence_manifest.json` updates
  under the new contract; nothing in this pass changes that handoff.

## Residual Risks

- This item remains readiness-only. It does not establish benchmark
  behavior or full-training BRDT competitiveness under the new
  sinogram-input contract.
- The smoke run was a short CPU fast-dev execution; the successor
  `40`-epoch item still owns any longer budget or paper-evidence
  interpretation.
- Historical BRDT bundles remain `born_init_image` lineage only.
  Downstream consumers must not mix those rows with the new
  sinogram-input lane unless a later same-contract summary explicitly
  authorizes it.
- `model_variant_index.json` remains intentionally unchanged because
  this item produced only readiness-only adapter/smoke evidence and no
  benchmark-performance row.
