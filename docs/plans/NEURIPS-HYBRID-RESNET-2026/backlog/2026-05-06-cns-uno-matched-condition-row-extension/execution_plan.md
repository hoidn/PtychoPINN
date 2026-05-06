# CNS U-NO Matched-Condition Row Extension Execution Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Do not create worktrees.

**Goal:** Append one NeuralOperator U-NO row to the PDEBench CNS matched-condition result set under the existing `history_len=5`, `512 / 64 / 64`, `40`-epoch capped contract.

**Architecture:** Add or reuse a CNS-compatible U-NO model profile in the PDEBench image-suite runner, prove it can instantiate and run under the CNS tensor contract, launch only the new row, then publish a derived plus-U-NO table bundle by lineage.

## Source Of Truth

- Backlog item: `docs/backlog/active/2026-05-06-cns-uno-matched-condition-row-extension.md`
- CNS matched-condition summary: `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_cns_matched_condition_table_refresh_summary.md`
- Current CNS table: `docs/plans/NEURIPS-HYBRID-RESNET-2026/tables/pdebench_cns_matched_condition_metrics.json`
- U-NO generator adapter: `ptycho_torch/generators/neuralop_uno.py`
- CNS runner config/model surfaces:
  - `scripts/studies/pdebench_image128/run_config.py`
  - `scripts/studies/pdebench_image128/models.py`
  - `scripts/studies/run_pdebench_image128_suite.py`

## Task 1: Preflight CNS U-NO Shape Contract

- [ ] Add focused tests showing the CNS runner can build a U-NO profile with the CNS input/output channel counts and `128x128` spatial size.
- [ ] If a wrapper is required, document whether it only pads/crops/channels maps and does not change the benchmark target.
- [ ] Record blocked status if NeuralOperator U-NO cannot satisfy the CNS contract without changing data, target, loss, or row semantics.

## Task 2: Add CNS U-NO Profile

- [ ] Add a profile such as `neuralop_uno_cns_base` to `scripts/studies/pdebench_image128/run_config.py`.
- [ ] Wire model construction in `scripts/studies/pdebench_image128/models.py` using the existing U-NO body or a narrow task-local adapter around it.
- [ ] Add tests for profile lookup, model construction, forward shape, missing dependency behavior, and profile manifest fields.

## Task 3: Run Only The U-NO Row

- [ ] Launch the U-NO row under the matched CNS contract: `history_len=5`, `512 / 64 / 64`, `40` epochs, batch size `4`, MSE, same optimizer/normalization recipe as the current matched-condition table.
- [ ] Use tmux for long-running execution and the `ptycho311` environment.
- [ ] Write row-local invocation, config, profile, metrics, history, checkpoint, and provenance artifacts under the item-local artifact root.

## Task 4: Publish Derived Plus-U-NO Tables

- [ ] Add a table assembler or extend the existing CNS table refresh code to publish a derived plus-U-NO table by lineage.
- [ ] Preserve the existing four-row table as the current authority unless the summary explicitly recommends switching.
- [ ] Emit JSON/CSV/TeX assets with U-NO appended and row roles/source paths preserved.

## Task 5: Update Durable Indexes

- [ ] Create `pdebench_cns_uno_row_extension_summary.md`.
- [ ] Update `evidence_matrix.md`, `paper_evidence_index.md`, `model_variant_index.json`, and `docs/studies/index.md`.
- [ ] State whether the manuscript should consume the plus-U-NO table or treat U-NO as adjacent context.

## Verification

- [ ] Run the backlog item input check unchanged.
- [ ] `pytest -q tests/studies/test_pdebench_image128_models.py -k "uno or profile or cns"`
- [ ] `pytest -q tests/studies/test_pdebench_image128_runner.py -k "matched_condition or pdebench_cns or uno"`
- [ ] `python -m compileall -q ptycho_torch scripts/studies`
- [ ] Validate the derived plus-U-NO JSON has exactly the original four matched-condition rows plus one U-NO row, all under the same CNS contract.
- [ ] Refresh the active backlog manifest and confirm this item is valid.
