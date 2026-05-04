# BRDT FFNO Row Extension Plan

## Objective

Add one append-only FFNO neural row to the completed BRDT decision-support
preflight so the candidate inverse-scattering lane can test whether a
factorized Fourier operator improves on the existing FNO vanilla and
Hybrid ResNet/SRU-Net rows.

## Scope

- Consume the existing BRDT authorities:
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/born_rytov_dt_candidate_lane_design.md`
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/brdt_task_adapters.md`
  - `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-brdt-four-row-preflight/preflight_manifest.json`
  - `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-brdt-four-row-preflight/metrics.json`
- Add a task-local FFNO adapter row under `scripts/studies/born_rytov_dt/`
  without registering BRDT as a CDI generator.
- Preserve the existing BRDT contract: `born_init_image` input, physical `q`
  target, same decision-support dataset, same split, same normalization,
  same supervised plus Born-consistency loss, same fixed sample IDs, same
  metric schema, and `decision_support_preflight_only` claim boundary.
- Run only the new FFNO row unless a required compatibility preflight fails.
- Append the FFNO row to the BRDT metrics/table/manifest surfaces and update
  durable indexes by lineage reference to the original four-row bundle.

## Non-Goals

- Do not rerun or overwrite the existing classical, U-Net, FNO vanilla, or
  Hybrid ResNet rows.
- Do not call BRDT paper-grade evidence or promote it into the manuscript.
- Do not add direct-sinogram input, Rytov mode, limited-angle rows, multi-seed
  robustness, or external FDTD mismatch checks.
- Do not conflate FNO vanilla and FFNO. The new row must have a distinct
  `row_id`, architecture metadata, parameter count, invocation, and
  row-status payload.

## Implementation Notes

The first implementation should prefer the narrowest task-local extension:

1. Extend BRDT row/config surfaces to admit a single visible row such as
   `ffno` or `ffno_factorized`.
2. Add or adapt an FFNO body behind `BRDTModelAdapter`, recording dependency
   blockers explicitly if the required implementation is unavailable.
3. Extend the preflight runner with an append-only single-row mode or an
   equivalent helper that writes the FFNO row under a separate extension root
   and then emits a combined metrics view by referencing the original bundle.
4. Preserve the original row contract fingerprints and source arrays; write
   fresh invocation and row-summary artifacts only for the FFNO row.
5. Update `model_variant_index.json`, `evidence_matrix.md`, and
   `paper_evidence_index.md` only after the FFNO row has a completed or
   explicitly blocked row-status artifact.

## Expected Artifacts

- Extension artifact root under:
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-04-brdt-ffno-row-extension/`
- FFNO row directory with `invocation.json`, `invocation.sh`,
  `row_summary.json`, optional `model_state.pt`, and source arrays for the
  fixed visual sample set.
- Combined BRDT metrics JSON/CSV or an append-only extension manifest that
  clearly references:
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-brdt-four-row-preflight/`
- Updated durable indexes if and only if the FFNO row is completed or blocked
  with a structured reason.

## Verification

```bash
python - <<'PY'
from pathlib import Path
required = [
    Path("docs/plans/NEURIPS-HYBRID-RESNET-2026/born_rytov_dt_candidate_lane_design.md"),
    Path("docs/plans/NEURIPS-HYBRID-RESNET-2026/brdt_task_adapters.md"),
    Path(".artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-brdt-four-row-preflight/preflight_manifest.json"),
    Path(".artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-brdt-four-row-preflight/metrics.json"),
]
missing = [str(path) for path in required if not path.exists()]
if missing:
    raise SystemExit(f"missing BRDT FFNO extension inputs: {missing}")
print("brdt ffno extension inputs present")
PY
pytest -q tests/studies/test_born_rytov_dt_adapters.py tests/studies/test_born_rytov_dt_preflight.py
python -m compileall -q scripts/studies/born_rytov_dt ptycho_torch
```
