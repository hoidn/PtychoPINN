# Corrected BRDT FFNO 40-Epoch Rerun Execution Plan

> **For agentic workers:** Use `superpowers:executing-plans` or
> `superpowers:subagent-driven-development` when executing this plan. Do not
> create worktrees. Use `tmux` plus `ptycho311` for multi-minute BRDT runs, track
> the exact launched PID, and consider the run complete only when that PID exits
> `0` and the expected artifacts are freshly written.

**Goal:** Produce a clean 40-epoch BRDT rerun whose FFNO row uses the corrected
no-refiner adapter, with provenance, convergence, sample-255 visuals, and table
inputs suitable for later manuscript refresh if the evidence gate passes.

**Architecture:** Reuse the existing BRDT 40-epoch runner design, but replace
the historical FFNO lineage with the corrected 20-epoch no-refiner authority.
The preferred clean evidence path reruns both `hybrid_resnet`/SRU-Net and
corrected `ffno` under the same invocation family so provenance and runtime
comparisons are coherent.

**Tech Stack:** PATH `python`, PyTorch, existing BRDT 40-epoch runner,
convergence audit and paper-refresh helpers, JSON/CSV/PNG/NumPy artifacts,
pytest, compileall.

## Source Of Truth

- Corrected 20-epoch prerequisite:
  `2026-05-06-brdt-corrected-ffno-row-rerun`
- Historical 40-epoch caveat authority:
  `docs/plans/NEURIPS-HYBRID-RESNET-2026/brdt_supervised_born_40ep_paper_evidence_summary.md`
- BRDT same-contract baseline:
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-brdt-four-row-preflight/`

## Tasks

- [ ] Validate that the corrected 20-epoch no-refiner FFNO row exists and that
      its model profile excludes `cnn_blocks` and post-bottleneck refiners.
- [ ] Rerun corrected `ffno` for 40 epochs with supervised+Born loss,
      `ReduceLROnPlateau`, batch `16`, seed `42`, and per-epoch history.
- [ ] Rerun `hybrid_resnet`/SRU-Net in the same corrected root if the goal is a
      fresh two-row paper-evidence gate; otherwise label it explicitly as reused
      lineage and do not claim a fresh two-row bundle.
- [ ] Regenerate `metrics.{json,csv}`, `combined_metrics.{json,csv}`,
      `history.{json,csv}`, `convergence_audit.{json,csv}`,
      `paper_evidence_gate.json`, `model_profile.json`, runtime provenance, and
      sample-255 source arrays/figures.
- [ ] Regenerate dependent paper assets only after the corrected gate decision:
      BRDT manuscript table/figure, model-config table, efficiency table, and
      package zip if those assets still include BRDT FFNO.
- [ ] Update durable summaries and indexes to mark the old 40-epoch row as
      superseded for pure-FFNO use when the corrected row completes.

## Required Checks

```bash
pytest -q tests/studies/test_born_rytov_dt_adapters.py tests/studies/test_born_rytov_dt_preflight.py
python -m compileall -q scripts/studies/born_rytov_dt ptycho_torch
python -m json.tool docs/plans/NEURIPS-HYBRID-RESNET-2026/model_variant_index.json >/tmp/model_variant_index.json
python -m json.tool docs/plans/NEURIPS-HYBRID-RESNET-2026/ablation_index.json >/tmp/ablation_index.json
```

## Completion Gate

- The corrected FFNO row is no-refiner by profile and source inspection.
- The sample-255 visual and table inputs come from the corrected row, not the
  historical local-refiner proxy.
- The evidence gate either passes with clean provenance or fails with explicit
  failed checks; no paper-facing surface silently consumes the old proxy row.
