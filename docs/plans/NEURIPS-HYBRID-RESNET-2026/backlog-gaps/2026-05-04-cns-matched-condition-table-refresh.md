# Seed Plan: CNS Matched-Condition Table Refresh

## Objective

Refresh the manuscript-facing PDEBench CNS table so the headline model ranking
uses one matched condition rather than mixing best-observed rows across history
lengths, caps, or variants.

## Scope

- Consume the current `history_len=2`, `2048 / 256 / 256`, `40`-epoch capped
  CNS authority bundle and the completed `history_len=5` comparator gap-fill
  evidence.
- Select exactly one headline CNS table condition:
  - prefer the all-row `history_len=5` capped condition only if
    `author_ffno_cns_base`, `spectral_resnet_bottleneck_base` / SRU-Net*,
    `fno_base`, and `unet_strong` all have complete same-condition rows;
  - otherwise keep the locked `history_len=2`, `2048 / 256 / 256` capped
    authority as the headline CNS table.
- Move any mixed best-observed rows to appendix/context only, with row lineage
  and condition labels preserved.
- Do not rerun CNS training in this item. This is a deterministic table,
  figure, index, and manuscript-authority refresh. If a required row is missing,
  record the row-level blocker and choose the complete matched condition.

## Non-Goals

- Do not reopen the CNS paper-contract decision into full training.
- Do not mix `history_len=5` FFNO/SRU-Net* rows with `history_len=2` FNO/U-Net
  rows in the headline ranking.
- Do not change model architectures, history lengths, caps, normalization,
  metric definitions, or fixed visual samples after seeing metrics.
- Do not discard best-observed evidence; keep it as labeled context outside the
  headline table.

## Expected Artifacts

- A durable summary:
  `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_cns_matched_condition_table_refresh_summary.md`
- Refreshed CNS JSON, CSV, and TeX payloads under the existing paper-results
  table structure.
- Refreshed CNS visual bundle references only when the chosen matched condition
  has corresponding fixed-sample visuals.
- Updates to:
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/paper_evidence_index.md`
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/evidence_matrix.md`
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/model_variant_index.json`
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/hybrid_resnet_neurips_first_draft.tex`
  - `docs/studies/index.md`

## Verification Commands

```bash
python - <<'PY'
from pathlib import Path
required = [
    Path("docs/backlog/done/2026-04-29-cns-paper-2048cap-row-extension.md"),
    Path("docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_cns_paper_2048cap_extension_summary.md"),
    Path("docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_cns_history5_comparator_gap_fill_summary.md"),
    Path("scripts/studies/paper_results_refresh.py"),
]
missing = [str(path) for path in required if not path.exists()]
if missing:
    raise SystemExit(f"missing CNS matched-table inputs: {missing}")
print("CNS matched-table inputs present")
PY
pytest -q tests/studies/test_paper_results_refresh.py tests/studies/test_pdebench_cfd_cns_metrics.py
python -m compileall -q scripts/studies
```
