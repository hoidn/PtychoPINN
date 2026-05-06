# Execution Report

## Completed In This Pass

- Added scheduler/history coverage and a dedicated BRDT 40-epoch paper-evidence
  runner plus convergence/gate helpers under `scripts/studies/born_rytov_dt/`.
- Executed the live same-contract 40-epoch BRDT rerun for exactly
  `hybrid_resnet` and `ffno`, producing the fresh immutable artifact root at
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-05-brdt-supervised-born-40ep-paper-evidence/`.
- Updated the durable BRDT evidence surfaces so the final gate, summary, index,
  and manifest all agree on the additive promotion result.

## Completed Plan Tasks

- Task 1: locked the 40-epoch contract with regression coverage for scheduler,
  history, convergence, and gate payloads
- Task 2: implemented per-epoch history capture and scheduler provenance
- Task 3: added the immutable 40-epoch paper-evidence runner
- Task 4: emitted convergence audit and paper-evidence gate payloads
- Task 5: ran the live 40-epoch rows and validated the artifact root
- Task 6: wrote the durable summary and updated the evidence surfaces

## Remaining Required Plan Tasks

- None.

## Verification

- Required deterministic checks passed:
  - `pytest -q tests/studies/test_born_rytov_dt_preflight.py -k 'history or scheduler or 40ep or paper_evidence or convergence'`
  - `pytest -q tests/studies/test_born_rytov_dt_adapters.py -k 'ffno or hybrid or row_schema'`
  - `pytest -q tests/studies/test_born_rytov_dt_adapters.py tests/studies/test_born_rytov_dt_preflight.py`
  - `python -m compileall -q scripts/studies/born_rytov_dt ptycho_torch`
- Live runner completed in tmux with tracked PID exit `0`.
- Fresh artifact-root checks confirmed:
  - both rows carry `history.json` and `history.csv`
  - `convergence_audit.json` and `paper_evidence_gate.json` exist
  - sample-`255` visuals and source arrays exist
  - top-level manifest records separate claim-boundary and promotion-status
    fields
- Edited JSON evidence surfaces were validated with `python -m json.tool`.

## Residual Risks

- Both rerun rows were still materially improving at stop and never reduced LR,
  so the result is bounded additive evidence rather than a full convergence
  claim.
- The sample-`255` classical comparator is accepted from the frozen baseline
  lineage rather than regenerated in this pass.
