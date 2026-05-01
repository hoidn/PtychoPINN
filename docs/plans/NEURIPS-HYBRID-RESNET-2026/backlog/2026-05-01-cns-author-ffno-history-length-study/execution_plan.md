# CNS Authored FFNO History-Length Study Seed Plan

This is the seed plan referenced by
`docs/backlog/active/2026-05-01-cns-author-ffno-history-length-study.md`.
The NeurIPS backlog workflow may rewrite this into a fuller execution plan
before implementation.

## Intent

Measure whether the authored FFNO CNS row benefits from additional temporal
context, analogous to the completed SRU-Net history-length study, without
changing the locked CNS paper-table contract.

## Required Context

- `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_author_ffno_equal_footing_summary.md`
- `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_cns_spectral_history_len4plus_compare_summary.md`
- `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_cns_paper_contract_decision.md`
- `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_cns_paper_row_lock_summary.md`
- `docs/plans/NEURIPS-HYBRID-RESNET-2026/evidence_matrix.md`
- `docs/plans/NEURIPS-HYBRID-RESNET-2026/model_variant_index.json`
- `docs/plans/NEURIPS-HYBRID-RESNET-2026/ablation_index.json`

## Execution Shape

1. Audit the existing `author_ffno_cns_base` adapter and model profile to
   confirm how input channels are derived from `history_len`.
2. Freeze the existing `history_len=2`, `40`-epoch authored-FFNO root as the
   comparison anchor.
3. Run a fresh `history_len=3`, `40`-epoch authored-FFNO row under the same
   capped CNS contract.
4. Compare absolute metrics and deltas against the frozen `history_len=2`
   anchor.
5. Gate `history_len=4` and `history_len=5` runs from the observed prior
   history point, preserving the same contract.
6. Write a durable summary and update the evidence indexes for any evaluated
   rows.

## Verification

At minimum, run:

```bash
pytest -q tests/studies/test_pdebench_image128_models.py -k 'author_ffno'
pytest -q tests/studies/test_pdebench_image128_runner.py tests/studies/test_pdebench_cfd_cns_data.py tests/studies/test_pdebench_cfd_cns_metrics.py
python -m compileall -q scripts/studies/pdebench_image128 scripts/studies/run_pdebench_image128_suite.py
```

Any long training run must be launched through tmux in the `ptycho311`
environment and must track the exact launched PID.

## Claim Boundary

This item is a within-model temporal-context ablation for authored FFNO. It is
not a replacement for the locked CNS headline bundle and does not authorize
mixing history lengths in a model-ranking table.
