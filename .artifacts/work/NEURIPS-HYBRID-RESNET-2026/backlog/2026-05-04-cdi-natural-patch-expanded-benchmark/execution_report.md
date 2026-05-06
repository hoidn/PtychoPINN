# Execution Report

## Completed In This Pass

This pass addresses the implementation review's three HIGH findings.
HIGH-1 (unfinished current-scope work, the clean tmux `--mode benchmark`
relaunch) is recorded honestly as outstanding follow-up work; HIGH-2 (false
success provenance attached to failed-row recollations) and HIGH-3 (drifted
discoverability surfaces) are fixed in this pass.

- **HIGH-2 (false success provenance on failed-row recollation):** the
  recollate path now refuses to surface stale per-row `exit_code_proof.json`
  for any row whose original invocation did not report `completed/0`.
  Concretely:
  - `_PROVENANCE_FIELDS_TO_DROP` now includes `outputs`
    (`scripts/studies/cdi_natural_patch_benchmark.py`). The recollate path
    no longer carries the prior `outputs.exit_code_proof_json` reference
    forward; it rebuilds the per-row outputs dict from on-disk artifacts
    each republication.
  - `_attach_natural_patch_row_provenance` now removes the on-disk
    `runs/<row>/exit_code_proof.json` file and pops
    `exit_code_proof_json` from `outputs_payload` whenever
    `write_exit_code_proof(...)` refuses to write proof. That single
    helper governs both the live and recollate paths, so failed rows can
    no longer leave dishonest proof artifacts behind.
  - Re-published the existing
    `runs/natural-patch-benchmark-20260505T213458Z` run root via
    `--mode recollate`. The four failed torch rows
    (`pinn_hybrid_resnet`, `pinn_fno_vanilla`, `pinn_ffno`,
    `pinn_neuralop_uno`) now have no `exit_code_proof.json` on disk and
    no `outputs.exit_code_proof_json` reference in `metrics.json`. The
    two TF rows (`baseline`, `pinn`) still carry honest proof because
    their invocation records `completed/0`.
- **HIGH-3 (docs index drift):** synchronized the discoverability surfaces
  with the `benchmark_incomplete_recovered_non_authoritative` state.
  - `docs/index.md` no longer claims `paper_complete`; the entry now states
    that the bundle is `benchmark_incomplete_recovered_non_authoritative`,
    that four torch rows still report `failed/1`, and that a clean
    `--mode benchmark` rerun is required before paper-grade citation.
  - `docs/studies/index.md` is rewritten to label the run root as the
    "recovered (non-authoritative) bundle root", explain the recollate
    refuse-proof behavior for failed rows, and reiterate that this is
    diagnostic context, not paper-grade authority.
- **HIGH-2 regression test:**
  `test_recollate_failed_row_drops_stale_exit_code_proof` in
  `tests/studies/test_cdi_natural_patch_benchmark.py` seeds a stale
  `exit_code_proof.json` (claiming `completed/0`) under a row whose
  invocation reports `failed/1`, runs `--mode recollate`, and asserts:
  - the on-disk `exit_code_proof.json` is removed for the failed row;
  - `metrics.json` `rows.<row>.outputs` does not contain
    `exit_code_proof_json` for the failed row;
  - the still-completed sibling row keeps a freshly-issued proof file
    that the bundle writer can dereference.

## Completed Current-Scope Work

- Task 1: prerequisite presence gate, dataset-contract preflight, dry-run
  inspection, and prepared-input contract handling are unchanged and still
  green.
- Task 2: the natural-patch harness now refuses to publish a
  fictitious `exit_code_proof.json` for any row whose invocation did not
  report `completed/0`. The recollate path drops `outputs` from the
  carried-over provenance and rebuilds the outputs dict from on-disk
  artifacts; the live path inherits the same proof-removal guard via the
  shared `_attach_natural_patch_row_provenance` helper.
- Task 3: the existing run root is re-published as a
  `benchmark_incomplete_recovered_non_authoritative` bundle with honest
  per-row state. The approved Task 3 completion gate (clean tmux
  `--mode benchmark` PID exit `0` end-to-end) is **not** satisfied; this
  is recorded honestly rather than masked. The clean rerun remains the
  outstanding current-scope work and is enumerated under Follow-Up Work.
- Task 4: `cdi_natural_patch_expanded_benchmark_summary.md`,
  `paper_evidence_index.md`, `evidence_matrix.md`, and
  `model_variant_index.json` already described the recovered/non-authoritative
  state from the prior pass; this pass synchronizes `docs/index.md` and
  `docs/studies/index.md` so the entrypoint surfaces match the summary.

## Follow-Up Work

- **Authoritative path (HIGH-1):** relaunch the full six-row natural-patch
  benchmark on the locked dataset under one contiguous tmux launcher
  (`--mode benchmark`). The tracked PID must exit `0` end-to-end and the
  authoritative run root must contain fresh
  `metrics.json`, `metric_schema.json`, `model_manifest.json`,
  `paper_benchmark_manifest.json`, `metrics_table.csv`, and
  `metrics_table.tex`, with every required row in a `paper_grade` row
  status, before this benchmark can be promoted to `paper_complete` and
  the discoverability surfaces re-promoted to `paper_grade`. The
  provenance scaffolding emitted by `_attach_natural_patch_row_provenance`
  already satisfies `require_row_provenance=True` for live runs, so no
  further harness work is required to support a clean rerun.
- Recollation remains available as a republication tool for
  bundle-collation-only failures, but the path now refuses to upgrade a
  recovered bundle past `benchmark_incomplete` and refuses to attach
  proof to any row that did not invoke cleanly, so a future
  bundle-collation crash cannot be silently rewritten into paper-grade
  evidence.
- Optional defensive follow-up suggested by the prior review: derive the
  one-line `docs/index.md` and `docs/studies/index.md` natural-patch
  blurbs from the summary's status field at publish time so future
  status drift becomes a single-source-of-truth violation rather than a
  divergence between three hand-edited surfaces.

## Verification

- Required input presence gate (this pass log):
  `verification/required_input_gate_revise_20260506T010804Z.log`.
- Required pytest gate (this pass log):
  `verification/pytest_selected_revise_20260506T010804Z.log`
  (`31 passed`, including the new
  `test_recollate_failed_row_drops_stale_exit_code_proof` regression).
- Compile gate (this pass log):
  `verification/compileall_revise_20260506T010804Z.log`.
- Repo integration marker (this pass log):
  `verification/pytest_integration_revise_20260506T010804Z.log`.
- Honest recollate launcher proof for the cleanup pass:
  `verification/recollate-honest-20260506T010645Z.exit_code` reports `0`,
  `verification/recollate-honest-20260506T010645Z.log` records the harness
  JSON result, and the run root's `metrics.json` has the failed torch rows
  with `outputs` lacking `exit_code_proof_json` and the on-disk
  `runs/<failed_row>/exit_code_proof.json` files removed.
- On-disk verification of HIGH-2 fix on the published run root:
  - `runs/baseline/exit_code_proof.json`: present, `completed/0`.
  - `runs/pinn/exit_code_proof.json`: present, `completed/0`.
  - `runs/pinn_hybrid_resnet/exit_code_proof.json`: absent.
  - `runs/pinn_fno_vanilla/exit_code_proof.json`: absent.
  - `runs/pinn_ffno/exit_code_proof.json`: absent.
  - `runs/pinn_neuralop_uno/exit_code_proof.json`: absent.
  - `metrics.json` `rows.<failed_row>.outputs` carries no
    `exit_code_proof_json` for any of the four failed torch rows.

## Residual Risks

- The approved Task 3 completion gate (clean tmux `--mode benchmark` PID
  exit `0` end-to-end) is **not** satisfied. The bundle is honestly
  published as `benchmark_incomplete_recovered_non_authoritative` and the
  discoverability surfaces no longer claim paper-grade authority. A clean
  from-scratch rerun is the only path to a paper-citable natural-patch
  authority.
- The four torch rows surface `row_invocation_status="failed"` /
  `row_invocation_exit_code=1` in the bundle. The on-disk training
  artifacts (`config.json`, `metrics.json`, `history.json`, model
  checkpoints, recons) survive from the original launch, but the row
  processes themselves reported failure. The recovered metrics are
  diagnostic context only and must not be cited as authoritative
  expanded-object CDI evidence.
- TF-row `train_wall_time_sec` (~0.0003) and `inference_time_sec=null`
  remain advisory-only telemetry inherited from the original recovery
  path. They do not block paper completion under the current validator
  but should be regenerated by the clean rerun follow-up.
- This remains single-seed expanded-object CDI evidence on
  `natural_patches128_fixedprobe_v1` and does not replace the `lines128`
  paper-table authority under any reading.
