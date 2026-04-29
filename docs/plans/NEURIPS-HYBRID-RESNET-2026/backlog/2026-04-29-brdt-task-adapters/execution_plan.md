# BRDT Task Adapters Execution Plan

## Scope

Add task-local BRDT loaders, adapters, training wrappers, and evaluation
surfaces. This plan must not register BRDT as a normal CDI generator.

## Required Outputs

- Dataset loader/collator for the BRDT preflight dataset.
- Adapter contract for classical Born, U-Net, FNO vanilla, and SRU/Hybrid-family
  rows.
- Loss wrapper that unnormalizes predicted `q` before the physics term.
- Row metadata schema separating model, training procedure, input mode,
  operator version, and row status.

## Checks

- `pytest -q tests/studies/test_born_rytov_dt_adapters.py`
- `python -m compileall -q scripts/studies/born_rytov_dt ptycho_torch`
