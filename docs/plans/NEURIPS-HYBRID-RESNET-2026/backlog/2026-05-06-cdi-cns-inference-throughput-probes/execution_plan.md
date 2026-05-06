# CDI/CNS Inference Throughput Probe Execution Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Do not create worktrees.

**Goal:** Measure inference throughput for CDI Lines128 and PDEBench CNS rows that currently have missing throughput in the paper efficiency table, without rerunning training or changing paper row selection.

**Architecture:** Add benchmark-local inference probe utilities that load existing row checkpoints and datasets by lineage, run deterministic warmup/timed inference loops, write item-local timing artifacts, and teach the paper efficiency table generator to consume those timing artifacts.

## Source Of Truth

- Backlog item: `docs/backlog/active/2026-05-06-cdi-cns-inference-throughput-probes.md`
- Current efficiency table: `docs/plans/NEURIPS-HYBRID-RESNET-2026/tables/paper_efficiency_table.json`
- CDI authority: `docs/plans/NEURIPS-HYBRID-RESNET-2026/tables/cdi_lines128_metrics_extended.json`
- CNS authority: `docs/plans/NEURIPS-HYBRID-RESNET-2026/tables/pdebench_cns_matched_condition_metrics.json`
- Efficiency generator: `scripts/studies/paper_efficiency_table.py`
- CDI FFNO caveat: historical `pinn_ffno` and `supervised_ffno` rows are
  `FFNO-local proxy` rows with `fno_cnn_blocks=2`. Probe them only under that
  label unless the corrected no-refiner table refresh has replaced their
  source roots.

## Task 1: Recover Row Lineage

- [ ] Enumerate CDI and CNS rows from `paper_efficiency_table.json` whose `inference_throughput_status` is `missing`.
- [ ] Resolve each row to its checkpoint path, model config, dataset/split contract, and claim boundary from existing manifests.
- [ ] For CDI FFNO rows, record whether the source is `fno_cnn_blocks=0`
  no-refiner evidence or historical `fno_cnn_blocks=2` proxy evidence.
- [ ] Write a row-lineage audit JSON that marks rows as `ready_for_probe`, `missing_checkpoint`, `missing_dataset`, or `not_comparable`.

## Task 2: Implement Timing Protocol

- [ ] Add a shared inference-timing helper with fixed warmups, fixed timed passes, device label, batch size, sample count, elapsed seconds, samples/sec, and optional latency.
- [ ] Synchronize CUDA before and after each timed region when timing on GPU.
- [ ] Keep checkpoint loading, dataloader construction, data generation, metric computation, and figure rendering outside the timed region.

## Task 3: Add Benchmark Probes

- [ ] Add a CDI Lines128 inference probe that loads existing CDI rows and times only model forward/inference work under the locked Lines128 contract.
- [ ] Add a PDEBench CNS inference probe that loads matched-condition CNS rows and times only model forward/inference work under the locked `history_len=5`, `512 / 64 / 64`, 40-epoch contract.
- [ ] Write item-local `throughput_probe_results.{json,csv}` and `probe_provenance.json`.

## Task 4: Regenerate Paper Efficiency Assets

- [ ] Extend `scripts/studies/paper_efficiency_table.py` to consume the item-local throughput artifact.
- [ ] Regenerate `paper_efficiency_table.{json,csv,tex}` and `paper_efficiency_table_summary.md`.
- [ ] Preserve BRDT rows only as `historical_brdt_40ep_proxy_context` unless
  the corrected no-refiner BRDT FFNO rerun has regenerated the BRDT efficiency
  rows. Do not carry forward `paper_approved_secondary_brdt` for the historical
  FFNO-local-refiner proxy.
- [ ] Preserve CDI FFNO rows only as `FFNO-local proxy` unless the corrected
  no-refiner CDI table refresh has regenerated the CDI efficiency rows.

## Task 5: Update Discovery Docs

- [ ] Update `paper_evidence_index.md`, `evidence_matrix.md`, and `docs/studies/index.md` with the throughput-probe artifact root and claim boundary.
- [ ] State explicitly whether CDI/CNS headline tables can receive throughput columns or should defer to the standalone efficiency table.

## Verification

- [ ] `pytest -q tests/studies/test_paper_efficiency_table.py tests/studies/test_paper_results_refresh.py -k "efficiency_table or throughput"`
- [ ] `python scripts/studies/paper_results_refresh.py --write-efficiency-table`
- [ ] Validate that regenerated `paper_efficiency_table.json` has measured CDI/CNS throughput for every probed row, with skipped-row reasons for any remaining non-measured rows.
- [ ] `python -m compileall -q scripts/studies ptycho_torch`
- [ ] Refresh the active backlog manifest and confirm this item is valid.
