# Corrected BRDT FFNO 20-Epoch Rerun Execution Plan

> **For agentic workers:** Use `superpowers:executing-plans` or
> `superpowers:subagent-driven-development` when executing this plan. Do not
> create worktrees. Use `tmux` plus `ptycho311` for multi-minute BRDT runs, track
> the exact launched PID, and consider the run complete only when that PID exits
> `0` and the expected artifacts are freshly written.

**Goal:** Replace the historical BRDT FFNO-local-refiner proxy row with a fresh
append-only 20-epoch row from the corrected no-refiner BRDT FFNO adapter.

**Architecture:** Keep the original BRDT same-contract row-extension design, but
change the FFNO adapter contract to the current source of truth:
`SpatialLifter -> SharedFactorizedFfnoBottleneck -> 1x1 output projection`. The
runner must reject `cnn_blocks` and must not instantiate or report post-bottleneck
CNN refiners.

**Tech Stack:** PATH `python`, PyTorch, existing BRDT task-local runner and
artifact bundle code, JSON/CSV/PNG/NumPy artifacts, pytest, compileall.

## Source Of Truth

- Corrected adapter code:
  `scripts/studies/born_rytov_dt/models.py`
- Corrected default kwargs:
  `scripts/studies/born_rytov_dt/run_config.py`
- Historical proxy summary:
  `docs/plans/NEURIPS-HYBRID-RESNET-2026/brdt_ffno_row_extension_summary.md`
- Baseline read-only BRDT root:
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-brdt-four-row-preflight/`

## Tasks

- [ ] Verify that `models.build_neural_adapter(architecture="ffno", ...)`
      reports no `refiners`, no `cnn_blocks`, and `parameter_count=27394`.
- [ ] Add or update runner metadata so the corrected root records
      `architecture_contract="brdt_ffno_no_refiner_v1"` or equivalent.
- [ ] Run only the corrected FFNO row under the original 20-epoch
      supervised+Born contract.
- [ ] Write a new append-only corrected artifact root, for example:
      `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-06-brdt-corrected-ffno-row-rerun/`
- [ ] Emit corrected `metrics.{json,csv}`, `metric_schema.json`,
      `visual_manifest.json`, `rows/ffno/model_profile.json`, source arrays,
      and a combined lineage view against the original four-row baseline.
- [ ] Update durable surfaces:
      `brdt_ffno_row_extension_summary.md`, `paper_evidence_index.md`,
      `evidence_matrix.md`, `model_variant_index.json`, `ablation_index.json`,
      and `docs/index.md`.

## Required Checks

```bash
pytest -q tests/studies/test_born_rytov_dt_adapters.py tests/studies/test_born_rytov_dt_preflight.py
python -m compileall -q scripts/studies/born_rytov_dt ptycho_torch
python -m json.tool docs/plans/NEURIPS-HYBRID-RESNET-2026/model_variant_index.json >/tmp/model_variant_index.json
python -m json.tool docs/plans/NEURIPS-HYBRID-RESNET-2026/ablation_index.json >/tmp/ablation_index.json
```

## Completion Gate

- The corrected row exists under a new root and does not overwrite the
  historical proxy root.
- The corrected row's profile does not contain `cnn_blocks` or post-bottleneck
  refiner modules.
- Historical artifacts remain discoverable only as proxy evidence.
