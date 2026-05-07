# Execution Report

## Completed In This Pass

- Hardened `scripts/studies/born_rytov_dt/run_sinogram_input_40ep.py` so the
  sinogram-input BRDT bundle emits the required provenance, split/dataset
  manifests, convergence audit, gate payload, combined manifest, and visual
  manifest under the task root.
- Updated the repo-local BRDT paper refresh helpers and tests so the active
  manuscript/table/model-config/efficiency surfaces now read from the
  `2026-05-07-brdt-sinogram-input-40ep-paper-evidence` root and treat
  `sru_net` as the learned SRU-Net row.
- Completed the live sinogram-input `40`-epoch rerun under tracked PID
  `3253923`, validated `run_exit_status.json.exit_code == 0`, and confirmed
  `paper_evidence_gate.json` passed with
  `claim_boundary=paper_evidence_brdt_additive`.
- Refreshed repo-local manuscript assets, rebuilt the PDF twice, reran the
  paper-evidence audit, rebuilt the manuscript zip, and updated durable BRDT
  summary/index/manifest/design surfaces to make the new sinogram-input bundle
  the current additive-secondary BRDT authority while preserving older
  `born_init_image` bundles as historical lineage only.

## Completed Plan Tasks

- Task 1: Sinogram-input evidence runner implementation and targeted tests.
- Task 2: Live BRDT rerun, artifact validation, convergence audit, and gate.
- Task 3: Passing-gate manuscript/table/figure/model-config/efficiency refresh
  plus manuscript PDF rebuild.
- Task 4: Durable BRDT summary publication, discoverability updates,
  paper-evidence manifest update, audit rerun, and zip rebuild/validation.

## Remaining Required Plan Tasks

- None.

## Verification

- `pytest -q tests/studies/test_born_rytov_dt_preflight.py -k "sinogram_input_40ep or sinogram_input_smoke"`
  `3 passed, 95 deselected`
- `pytest -q tests/studies/test_paper_results_refresh.py tests/studies/test_paper_efficiency_table.py tests/studies/test_paper_model_config_table.py`
  `66 passed`
- `python scripts/studies/paper_results_refresh.py --write-brdt-assets --write-brdt-context-figure --write-model-config-table --write-efficiency-table`
  completed successfully
- `pdflatex -interaction=nonstopmode -halt-on-error hybrid_resnet_neurips_first_draft.tex`
  completed successfully twice
- `python scripts/studies/paper_evidence_audit.py --repo-root .`
  completed successfully
- `zip -T docs/plans/NEURIPS-HYBRID-RESNET-2026/scr_ptychography_neurips_draft_package.zip`
  `OK`
- Live bundle validation:
  - `runtime_provenance.json.tracked_pid == 3253923`
  - `run_exit_status.json.exit_code == 0`
  - `paper_evidence_gate.json.promotion_status == "passed"`
  - learned rows `ffno` and `sru_net` each wrote `40` history records

## Residual Risks

- Both learned rows remain materially improving at stop, so the correct read is
  bounded additive evidence rather than fully converged BRDT performance.
- The new learned `sinogram` contract regresses materially versus the preserved
  historical `born_init_image` authority; the old and new lanes must remain
  contract-separated in future analysis.
- The external `/home/ollie/Documents/neurips/` publication tree referenced by
  repo guidance is absent in this environment, so this pass updates only the
  repo-local manuscript/package surfaces.
- The live bundle's `preflight_manifest.json` keeps the authoritative
  row-local `input_mode="sinogram"` fields, but its legacy top-level
  `input_mode` / `in_channels` fields remain unset.
