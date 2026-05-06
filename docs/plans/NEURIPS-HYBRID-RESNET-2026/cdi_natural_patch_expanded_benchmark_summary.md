# NeurIPS CDI Natural-Patch Expanded Benchmark Summary

- Date: `2026-05-05`
- Backlog item: `2026-05-04-cdi-natural-patch-expanded-benchmark`
- State: `benchmark_incomplete_recovered_non_authoritative`
- Dataset id: `natural_patches128_fixedprobe_v1`
- Recovered (non-authoritative) run root:
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-04-cdi-natural-patch-expanded-benchmark/runs/natural-patch-benchmark-20260505T213458Z`
- Fixed seed policy: `seed=3`
- Fixed sample ids: `0`, `500`, `999`

## Authoritative Status

The approved Task 3 completion gate (a canonical tmux `--mode benchmark` launch
whose tracked PID exits `0`) **has not been satisfied**. The original launcher
exited `1` during bundle collation. The current bundle is a recollation of the
on-disk row artifacts and is published as `benchmark_incomplete` with every
required row in `recovered_non_authoritative` state. The bundle therefore is
**not paper-grade evidence** and must not be cited as `paper_complete`.

A clean from-scratch rerun under a single contiguous tmux launcher is the only
path to a paper-grade natural-patch authority on this dataset.

## Implementation State

- Provenance scaffolding helper
  `scripts/studies/cdi_natural_patch_benchmark.py::_attach_natural_patch_row_provenance`
  emits per-run `dataset_identity_manifest.json`, `split_manifest.json`, and
  per-row `runs/<model_id>/exit_code_proof.json` for live runs and the recollate
  path. Live rows fill `randomness.requested_seed`, full
  `environment` (host/torch/cuda/gpu), `git.dirty_state_note`,
  `dataset.manifest_json`, `splits.manifest_json`, and
  `outputs.exit_code_proof_json` so a clean future launch can satisfy
  `write_paper_benchmark_bundle(require_row_provenance=True)` directly.
- The `recollate` mode (`--mode recollate`) re-publishes an existing run root
  from its on-disk per-row artifacts. It does **not** rewrite per-row
  invocation envelopes; failed/1 envelopes remain failed/1 and are surfaced as
  `row_invocation_status="failed"` / `row_invocation_exit_code=1` in the
  bundle's `row_statuses`. The path stamps the per-row provenance with the
  original execution commit recorded in each row's `invocation.json
  extra.git_commit`, never the recollation-time HEAD commit.
- Bundle row statuses from recollate are always
  `recovered_non_authoritative`; the bundle writer downgrades the run to
  `benchmark_incomplete`.

## Accepted Six-Row Roster

- `baseline` -> `CDI CNN + supervised`
- `pinn` -> `CDI CNN + PINN`
- `pinn_hybrid_resnet` -> `Hybrid ResNet + PINN`
- `pinn_fno_vanilla` -> `FNO Vanilla + PINN`
- `pinn_ffno` -> `FFNO + PINN`
- `pinn_neuralop_uno` -> `U-NO + PINN`

## Recovered Bundle Row State

| Row | Original invocation | Recovered bundle row state |
|---|---|---|
| `baseline` | `completed/0` | `recovered_non_authoritative` |
| `pinn` | `completed/0` | `recovered_non_authoritative` |
| `pinn_hybrid_resnet` | `failed/1` (`Invalid shape (1, 128, 128) for image data`) | `recovered_non_authoritative` |
| `pinn_fno_vanilla` | `failed/1` | `recovered_non_authoritative` |
| `pinn_ffno` | `failed/1` | `recovered_non_authoritative` |
| `pinn_neuralop_uno` | `failed/1` | `recovered_non_authoritative` |

Per-row execution commit `5c4deddfd9b81431c063276720e7e4d3bf911ff7` is preserved
in the recollated metrics. The four torch rows' on-disk training artifacts
(model checkpoint, `metrics.json`, `history.json`, recon) are preserved from the
original launch, but the row processes themselves reported failure, so those
metrics are **not** authoritative without a fresh end-to-end run.

## Recovered Single-Seed Metrics (Non-Authoritative)

These numbers are the per-row metrics surfaced from the recovered
`metrics.json`. They reflect the original in-process training/inference but
were not produced by a launcher-clean end-to-end run; they are **not**
paper-citable.

| Row | Amp MAE | Phase MAE | Amp SSIM | Phase SSIM | Amp FRC50 | Phase FRC50 |
|---|---:|---:|---:|---:|---:|---:|
| `baseline` | 0.0716 | 0.3954 | 0.4864 | 0.6456 | 0.7902 | 0.7817 |
| `pinn` | 0.2923 | 1.4472 | 0.2440 | 0.2366 | 0.8565 | 0.8349 |
| `pinn_hybrid_resnet` | 0.2609 | 0.4374 | 0.0275 | 0.4425 | 0.8621 | 0.9395 |
| `pinn_fno_vanilla` | 0.1571 | 0.4237 | 0.0420 | 0.5307 | 0.8300 | 0.9338 |
| `pinn_ffno` | 0.1567 | 0.3961 | 0.0594 | 0.6041 | 0.8581 | 0.8339 |
| `pinn_neuralop_uno` | 0.1708 | 0.3997 | 0.0517 | 0.5981 | 0.8429 | 0.8793 |

## Claim Boundary

- Single-seed expanded-object CDI evidence on
  `natural_patches128_fixedprobe_v1` only.
- This benchmark does **not** replace the `lines128` paper-table authority in
  `lines128_paper_benchmark_summary.md` and is not a same-contract substitute
  for the synthetic grid-lines table.
- Until a clean from-scratch rerun produces a `paper_complete` bundle, no
  paper-facing claim should reference this benchmark as authoritative
  evidence.

## Verification

- Required dataset/benchmark unit tests:
  `pytest -q tests/studies/test_cdi_natural_patch_dataset.py tests/studies/test_cdi_natural_patch_benchmark.py`
  (`30 passed`).
- Compile gate: `python -m compileall -q scripts/studies ptycho_torch`.
- Recollate launcher proof:
  `verification/recollate-honest-<UTC>.exit_code` reports `0`. The recollate
  bundle reports `benchmark_status="benchmark_incomplete"` and every row in
  `recovered_non_authoritative` state, with the original execution commit
  preserved per row.

## Follow-Up Work

- Authoritative path: relaunch the full six-row natural-patch benchmark on the
  locked dataset under one contiguous tmux launcher (`--mode benchmark`). The
  tracked PID must exit `0` end-to-end. Only then can the run root be
  republished as `paper_complete` and the discoverability surfaces promoted to
  paper-grade authority.
- Recollation remains available as a republication tool for bundle-collation-
  only failures, but the path now refuses to upgrade a recovered bundle past
  `benchmark_incomplete` so a future bundle-collation crash cannot be silently
  rewritten into paper-grade evidence.
