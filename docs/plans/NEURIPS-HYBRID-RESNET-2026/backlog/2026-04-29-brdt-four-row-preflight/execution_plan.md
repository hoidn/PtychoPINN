# BRDT Four-Row Preflight Execution Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. This item authorizes only the bounded BRDT decision-support preflight after the BRDT operator, dataset-smoke, and task-adapter prerequisites. Do not broaden scope to Rytov, limited-angle, FFNO, physics-only, external FDTD mismatch, multi-seed, or paper-grade promotion work. Do not create worktrees. Any long-running dataset generation or row execution must remain under implementation ownership until the tracked PID exits successfully and the expected artifacts are freshly written; use `tmux` plus `ptycho311` for multi-minute runs.

**Goal:** Execute the first bounded BRDT decision-support preflight under one larger capped dataset, one operator contract, one input representation, one split/normalization policy, and one metric schema, yielding the exact four-row roster plus a self-contained machine-readable bundle for the later summary/promotion-decision backlog item.

**Architecture:** Reuse the locked BRDT operator, smoke-dataset, and adapter surfaces already checked in under `scripts/studies/born_rytov_dt/`, but add a distinct decision-support dataset profile and a preflight orchestration/reporting layer. The work splits into three units: upgrade dataset generation from smoke-only to a separate capped decision-support manifest, run the four-row equal-footing execution with explicit row-level blocker handling, and assemble the machine-readable bundle plus discovery/index updates that the follow-up summary item can consume without re-deriving scope. Keep the classical, U-Net, FNO vanilla, and SRU/Hybrid-family rows on the same `born_init_image` input, the same physical-`q` target semantics, and the same evaluation bundle.

**Tech Stack:** PATH `python`, PyTorch, HDF5, NumPy/JSON/CSV, existing BRDT task-local train/eval surfaces, optional `odtbrain` and `neuralop` with explicit blocker semantics, pytest, compileall, tmux in the `ptycho311` environment for long runs.

---

## Selected Objective

- Implement backlog item `2026-04-29-brdt-four-row-preflight`.
- Run the bounded BRDT decision-support preflight under one dataset, operator, input, split, metric, and training contract.
- Execute exactly this first-row roster unless a narrow row-level blocker remains after a documented fix attempt:
  - `classical_born_backprop`;
  - `unet` with `supervised + Born consistency`;
  - `fno_vanilla` with `supervised + Born consistency`;
  - one Hybrid-family row surfaced as either `sru_net` or `hybrid_resnet`, with the internal adapter body still `hybrid_resnet`.
- Emit image-space metrics on physical `q`, measurement-space residuals, parameter counts for neural rows, runtime/hardware metadata, row statuses, fixed-sample visuals, source arrays, table JSON/CSV, metric schema, and a manifest sufficient to regenerate the bundle.
- Leave the durable checked-in summary and promotion/deferral recommendation to backlog item `2026-04-29-brdt-preflight-summary-promotion-decision`; this item must instead leave a complete bundle and manifest fields that make that follow-up deterministic.

## Scope

- Consume the binding BRDT authorities from:
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/born_rytov_dt_candidate_lane_design.md`
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/brdt_operator_validation_report.md`
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/brdt_dataset_preflight.md`
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/brdt_task_adapters.md`
  - `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-brdt-operator-validation/operator_validation.json`
  - `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-brdt-dataset-preflight/dataset_manifest.json`
  - `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-brdt-task-adapters/fast_dev_run/adapter_contract.json` if needed for row-schema provenance
- Extend the BRDT task-local surfaces only as needed to support:
  - a distinct capped decision-support dataset profile and manifest;
  - checkpointed or otherwise reproducible row execution for the four-row bundle;
  - aggregation of per-row metrics, visuals, runtime metadata, and row statuses into one equal-footing bundle.
- Keep all BRDT work task-local under `scripts/studies/born_rytov_dt/`; do not register BRDT in the CDI generator registry or alter the CDI/PDE pillars.

## Explicit Non-Goals

- Do not add Rytov rows, limited-angle stress tests, FFNO rows, physics-only rows, external FDTD mismatch rows, or multi-seed robustness.
- Do not treat this item as paper-grade evidence or as a manuscript-table promotion. The output tier is `decision_support` or row-level `blocked`, not `paper_grade`.
- Do not mix `born_init_image` rows with direct-sinogram rows.
- Do not silently substitute incompatible inputs, splits, or normalization policies to make a row run.
- Do not overwrite or relabel the existing smoke dataset artifacts from `2026-04-29-brdt-dataset-preflight`; the larger preflight dataset must be a separate artifact set with its own identity and claim boundary.
- Do not modify `ptycho_torch/physics/born_rytov_dt.py` unless a true operator-contract bug is discovered; if that occurs, stop and open a narrow follow-up instead of drifting the locked authority here.
- Do not rewrite the roadmap, alter the steering queue, or expand BRDT into a new required manuscript pillar.
- Do not author `docs/plans/NEURIPS-HYBRID-RESNET-2026/brdt_preflight_summary.md` in this item; the queued summary/promotion-decision backlog item owns that durable narrative output.

## Steering, Roadmap, and Fairness Constraints

- Steering keeps CDI `lines128` and PDEBench CNS as the required manuscript pillars. BRDT remains additive candidate work only.
- The roadmap and BRDT candidate-lane design allow a bounded `candidate-brdt-preflight` chain, not a broad benchmark suite.
- The selected backlog reviewer notes are binding:
  - do not add Rytov, limited-angle, FFNO, physics-only, external FDTD mismatch, or multi-seed rows;
  - do not call this paper-grade evidence;
  - do not mix input representations, splits, or normalization policies across rows.
- The first BRDT input contract remains `born_init_image` for every row in this table.
- Every completed row in the bundle must share the same dataset manifest, operator authority, input representation, split counts, normalization rule, loss family, fixed sample IDs, and metric schema.
- If one row cannot satisfy the shared contract after a narrow fix attempt, record it as a row-level blocker with an explicit reason; do not relax the protocol to keep the row alive.

## Prerequisite Status

- The selected backlog prerequisite `2026-04-29-brdt-task-adapters` is satisfied by `docs/plans/NEURIPS-HYBRID-RESNET-2026/brdt_task_adapters.md` and its machine-readable adapter-contract artifacts.
- The adapter item already locked:
  - the `born_init_image` first input mode;
  - row metadata fields `row_id`, `model`, `training`, `input_mode`, `dataset_id`, `operator_version`, `row_status`;
  - local-adjoint fallback semantics for deriving `born_init_image` when `odtbrain` is unavailable;
  - task-local train/eval entrypoints under `scripts/studies/born_rytov_dt/`.
- The dataset preflight artifact is intentionally smoke-only. Its summary explicitly says the larger decision-support split still belongs to this backlog item. This is the key missing prerequisite the implementation must address before any row comparison can be called decision-support.
- The initiative progress ledger shows no BRDT-specific tranche completion beyond the upstream backlog summaries and no blocked initiative tranches that pre-empt this bounded candidate-lane work.

## Execution Rules

- Diagnose, fix, and rerun ordinary import, path, dependency, test-harness, or CLI failures before escalating.
- Reserve item-level `BLOCKED` for:
  - missing external resources or data generation capacity after a narrow fix attempt;
  - an unrecoverable dependency or hardware failure that prevents the shared contract from running at all;
  - a roadmap or contract contradiction that cannot be resolved within this backlog authority.
- Use row-level `blocked` statuses, not item-level `BLOCKED`, when only `odtbrain`, `neuralop`, CUDA, or one model path remains unavailable after a documented narrow fix attempt.
- Long-running commands must not be duplicated against the same `--output-root`. Track the exact launched PID and confirm both exit code `0` and freshly written expected artifacts before moving on.
- Keep scratch under `tmp/` and durable run outputs under `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-brdt-four-row-preflight/`.
- Preserve PATH `python` per `PYTHON-ENV-001`, and keep invocation/runtime provenance for every durable CLI surface.

## Implementation Architecture

1. **Decision-Support Dataset Layer**
   - Owns the separate capped dataset identity, counts, manifest semantics, and non-destructive reuse of the locked BRDT generator/operator contract.
2. **Four-Row Execution Layer**
   - Owns the exact row roster, training/evaluation budgets, checkpoint or reproducible model-state handling, fixed sample IDs, and row-level blocker semantics under one contract.
3. **Bundle Assembly and Discovery Layer**
   - Owns machine-readable metrics/manifests, fixed-sample visuals and source arrays, bundle-level handoff metadata, and the evidence-index updates that make the completed preflight discoverable.

## File and Artifact Targets

Mandatory contract outputs:

- `tests/studies/test_born_rytov_dt_preflight.py`
- `scripts/studies/born_rytov_dt/run_preflight.py` as the authoritative four-row orchestration entrypoint
- `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-brdt-four-row-preflight/decision_support_dataset/dataset_manifest.json`
- `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-brdt-four-row-preflight/preflight_manifest.json`
- `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-brdt-four-row-preflight/metrics.json`
- `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-brdt-four-row-preflight/metrics.csv`
- `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-brdt-four-row-preflight/metric_schema.json`
- `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-brdt-four-row-preflight/visual_manifest.json`
- `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-brdt-four-row-preflight/visuals/`
- `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-brdt-four-row-preflight/figures/source_arrays/`

Likely code surfaces to modify:

- `scripts/studies/born_rytov_dt/dataset_contract.py`
- `scripts/studies/born_rytov_dt/generate_brdt_dataset.py`
- `scripts/studies/born_rytov_dt/run_config.py`
- `scripts/studies/born_rytov_dt/train.py`
- `scripts/studies/born_rytov_dt/evaluate.py`
- `scripts/studies/born_rytov_dt/reporting.py`
- `tests/studies/test_born_rytov_dt_adapters.py` only if shared adapter/train/eval behavior changes

Required documentation/index updates after durable outputs exist:

- `docs/plans/NEURIPS-HYBRID-RESNET-2026/evidence_matrix.md`
- `docs/plans/NEURIPS-HYBRID-RESNET-2026/model_variant_index.json`
- `docs/plans/NEURIPS-HYBRID-RESNET-2026/paper_evidence_index.md`
- `docs/plans/NEURIPS-HYBRID-RESNET-2026/ablation_index.json` only if implementation emits a separate harness/probe/audit artifact outside the primary four-row bundle
- `docs/index.md` only if the new BRDT preflight summary becomes a durable standalone reference beyond the BRDT section already discoverable through the evidence indexes

Preferred packaging, if helper logic becomes too large:

- add adjacent helpers such as `scripts/studies/born_rytov_dt/preflight_metrics.py` or `scripts/studies/born_rytov_dt/preflight_visuals.py` instead of making `reporting.py` monolithic

## Contract and Claim Invariants

- **Dataset identity:** the smoke dataset remains intact. The preflight must generate or stage a separate capped dataset/manifest under this item's artifact root with a distinct claim boundary and counts.
- **Bounded dataset size:** use the candidate-design capped decision-support split by default: `2048 / 256 / 256` train/val/test at `N=128`, `A=64`. If a smaller cap becomes necessary, record the reduced contract explicitly before any row runs and keep it identical across all rows.
- **Input contract:** every row uses `born_init_image`; no direct-sinogram row is allowed in this bundle.
- **Target semantics:** image-space metrics are computed on physical `q`, and any physics loss or measurement prediction path must unnormalize model outputs before invoking the operator.
- **Hybrid-family identity:** choose one visible Hybrid-family row label once for the bundle (`sru_net` preferred if manuscript-facing naming is desired, otherwise `hybrid_resnet`), record it in the preflight manifest, and do not surface both labels in one table.
- **Classical row semantics:** the classical comparison row should prefer ODTbrain. The local-adjoint path remains valid for generating `born_init_image` for neural rows, but if ODTbrain remains unavailable after a narrow fix attempt, the classical table row must be marked `blocked` rather than silently presented as equal-footing decision-support evidence.
- **Neural-row fairness:** use one shared optimizer/loss family and one bounded epoch budget for U-Net, FNO vanilla, and Hybrid-family rows. Start from the checked-in BRDT defaults (`Adam`, `lr=2e-4`, `LossWeights` from `run_config.py`) and lock the budget once before launch. Default to `20` epochs for all neural rows; a single documented increase to `40` is allowed only if the same increase is applied uniformly across neural rows.
- **Metric schema minimum:** blocking metrics for this item are physical-`q` MAE/RMSE/relative-L2 plus measurement-space MAE/RMSE/relative-L2, parameter count, runtime/hardware fields, and row status. Additional diagnostics such as PSNR, SSIM, per-angle curves, per-frequency curves, or FRC may be included as supporting outputs if they come from the same saved arrays without broadening scope.
- **Visual bundle minimum:** fixed-sample `q` comparisons, `q` error maps, and sinogram residual panels are required, with source arrays and a visual manifest. Optional FRC or refractive-index panels are supporting only.
- **Claim boundary:** every emitted artifact must label the bundle as `decision_support_preflight_only`, not manuscript or paper-grade evidence.

## Execution Tranches

### Tranche 1: Add the Capped Decision-Support Dataset Contract

**Purpose:** Replace the current smoke-only dataset assumption with a separate larger dataset contract that this item can honestly use for decision-support.

- [ ] Write failing tests first in `tests/studies/test_born_rytov_dt_preflight.py` for:
  - a distinct decision-support dataset identity and claim boundary;
  - non-destructive separation between the smoke artifact root and the four-row preflight artifact root;
  - expected split counts, manifest fields, and operator/dataset authority pointers;
  - refusal to treat the original smoke manifest as the authoritative decision-support dataset.
- [ ] Extend `scripts/studies/born_rytov_dt/dataset_contract.py` and `scripts/studies/born_rytov_dt/generate_brdt_dataset.py` so implementation can emit a capped decision-support dataset under this item's artifact root without overwriting the smoke dataset metadata or `backlog_item` identity.
- [ ] Keep geometry, `q` semantics, normalization helpers, and operator-authority validation sourced from the existing locked contract; only extend the parts needed to represent the larger split and new claim boundary.
- [ ] Preserve the generator's `--dry-run-manifest` path and make it produce the new decision-support dataset manifest skeleton before live generation.
- [ ] If implementation needs a separate dataset-profile flag rather than raw count overrides, add it in a narrow way and keep the tests authoritative.

**Verification**

- [ ] **Blocking:** `pytest -q tests/studies/test_born_rytov_dt_preflight.py` must cover the new dataset-profile contract before any live dataset generation.
- [ ] **Blocking:** `python -m compileall -q scripts/studies/born_rytov_dt` must pass before any long-running generation or training.
- [ ] **Supporting:** if shared dataset helpers change, rerun `pytest -q tests/studies/test_born_rytov_dt_dataset.py`.

### Tranche 2: Add the Four-Row Preflight Orchestrator and Row-Execution Contract

**Purpose:** Create one authoritative preflight entrypoint that can launch or resume the exact row roster under one bounded contract.

- [ ] Add `scripts/studies/born_rytov_dt/run_preflight.py` that:
  - consumes the decision-support dataset manifest and locked operator authority;
  - resolves the exact four-row roster;
  - freezes the visible Hybrid-family label once;
  - chooses fixed sample IDs from the test split before any row is evaluated;
  - writes a top-level `preflight_manifest.json` recording dataset identity, operator pointer, row roster, epoch budget, optimizer/loss contract, metric schema version, fixed sample IDs, and claim boundary.
- [ ] Extend `scripts/studies/born_rytov_dt/train.py` and `scripts/studies/born_rytov_dt/evaluate.py` as needed so the preflight can train a neural row, persist enough model state or outputs to evaluate that same trained row later, and emit runtime/device metadata.
- [ ] Keep classical-row handling explicit:
  - try ODTbrain first;
  - if unavailable, attempt only narrow environment fixes within item scope;
  - if still unavailable, emit a row-level blocker and keep the rest of the bundle on the same contract.
- [ ] Keep `neuralop` handling explicit for `fno_vanilla`; a missing dependency after a narrow fix attempt becomes a row-level blocker, not a silent omission.
- [ ] Ensure the preflight runner prevents duplicate active writers to the same output root and can resume or skip already-complete row artifacts when the manifest contract matches.
- [ ] Add or extend tests in `tests/studies/test_born_rytov_dt_preflight.py` for:
  - row roster resolution and exact ordering;
  - fixed sample ID locking;
  - row-level blocker serialization;
  - preflight-manifest schema;
  - train/eval handoff or checkpoint-loading behavior;
  - refusal to mix mismatched dataset/operator/input contracts across rows.

**Verification**

- [ ] **Blocking:** `pytest -q tests/studies/test_born_rytov_dt_preflight.py` must pass with orchestration coverage before any multi-row execution.
- [ ] **Supporting:** rerun `pytest -q tests/studies/test_born_rytov_dt_adapters.py` if `run_config.py`, `train.py`, `evaluate.py`, or shared reporting helpers change.

### Tranche 3: Execute the Bounded Dataset Generation and Four-Row Runs

**Purpose:** Run the actual decision-support preflight under the locked bounded contract.

- [ ] Run the decision-support dataset dry gate first, using the new distinct output root under `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-brdt-four-row-preflight/decision_support_dataset/`.
- [ ] If the dry gate passes, run live capped dataset generation under the same contract and confirm the new `dataset_manifest.json` plus split HDF5 files are freshly written.
- [ ] Launch row execution only after the blocking checks and live dataset artifacts are present.
- [ ] Train/evaluate the neural rows under one locked contract:
  - same dataset manifest;
  - same optimizer/loss configuration;
  - same bounded epoch budget;
  - same batch-size policy unless a hardware-limited adjustment is required and applied uniformly.
- [ ] Execute or record the classical row under the same dataset and metric contract.
- [ ] For each row, persist invocation artifacts, runtime/hardware metadata, metrics, and row status under a stable per-row run directory.
- [ ] If a row remains blocked, keep its slot in the final table with explicit `blocker_reason`, `blocker_message`, and claim-boundary notes. Do not reroute to a different representation or a different dataset just to fill the slot.

**Verification**

- [ ] **Blocking:** before the live multi-row run, rerun `pytest -q tests/studies/test_born_rytov_dt_preflight.py`.
- [ ] **Blocking:** before the live multi-row run, rerun `python -m compileall -q scripts/studies/born_rytov_dt`.
- [ ] **Blocking:** live execution is complete only when the tracked PID exits `0` and the expected dataset/row artifacts for the selected output root exist freshly.
- [ ] **Supporting:** if shared adapter logic changed while fixing runtime failures, rerun `pytest -q tests/studies/test_born_rytov_dt_adapters.py`.

### Tranche 4: Assemble the Bundle and Discovery Metadata

**Purpose:** Turn per-row outputs into the durable decision-support bundle that the follow-up summary/promotion-decision item can consume directly.

- [ ] Aggregate the completed and blocked row outputs into:
  - `metrics.json`;
  - `metrics.csv`;
  - `metric_schema.json`;
  - `visual_manifest.json`;
  - `preflight_manifest.json` updated with final row statuses, artifact pointers, dependency/blocker notes, claim boundary, and an explicit `next_backlog_item: 2026-04-29-brdt-preflight-summary-promotion-decision` handoff field.
- [ ] Emit fixed-sample visuals and source arrays under one stable manifest. Minimum required visuals:
  - `visuals/brdt_compare_q.png`
  - `visuals/brdt_error_q.png`
  - `visuals/brdt_sinogram_residual.png`
  - matching `.npy` source arrays for every fixed sample / row prediction used in the rendered figures
- [ ] Ensure the final bundle metadata is sufficient for the follow-up summary/promotion item without rereading backlog notes. At minimum, the top-level manifest must identify:
  - operator validity authority consumed from the locked validation report;
  - dataset and normalization validity for the new decision-support manifest;
  - exact row roster and row statuses;
  - metric bundle and visual/source-array artifact pointers;
  - dependency/environment issues and blocker reasons;
  - known limitations and the explicit `decision_support_preflight_only` claim boundary.
- [ ] Update the durable evidence indexes so the BRDT preflight bundle outputs are discoverable without rereading backlog notes, while leaving the checked-in narrative summary to the next queued backlog item.

**Verification**

- [ ] **Blocking:** confirm the primary bundle files exist:
  - `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-brdt-four-row-preflight/preflight_manifest.json`
  - `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-brdt-four-row-preflight/metrics.json`
  - `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-brdt-four-row-preflight/metrics.csv`
  - `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-brdt-four-row-preflight/metric_schema.json`
  - `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-brdt-four-row-preflight/visual_manifest.json`
- [ ] **Supporting:** if implementation adds a separate audit or probe artifact outside the main bundle, update `ablation_index.json` or explicitly state in the execution report why no ablation-index update was needed.

## Deterministic Checks

Required backlog checks from the selected item:

- [ ] **Blocking:** `pytest -q tests/studies/test_born_rytov_dt_preflight.py`
- [ ] **Blocking:** `python -m compileall -q scripts/studies/born_rytov_dt`

Additional proportional checks:

- [ ] **Supporting:** `pytest -q tests/studies/test_born_rytov_dt_adapters.py` whenever shared adapter/train/eval surfaces change.
- [ ] **Supporting:** `pytest -q tests/studies/test_born_rytov_dt_dataset.py` whenever dataset-generation or dataset-contract helpers change.

Suggested final artifact-presence check:

```bash
python - <<'PY'
from pathlib import Path
root = Path(".artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-brdt-four-row-preflight")
required = [
    root / "preflight_manifest.json",
    root / "metrics.json",
    root / "metrics.csv",
    root / "metric_schema.json",
    root / "visual_manifest.json",
]
missing = [str(p) for p in required if not p.exists()]
if missing:
    raise SystemExit(f"missing BRDT preflight outputs: {missing}")
print("brdt preflight outputs present")
PY
```

## Long-Run Command Ownership

Blocking pre-run checks should pass before expensive work begins:

```bash
pytest -q tests/studies/test_born_rytov_dt_preflight.py
python -m compileall -q scripts/studies/born_rytov_dt
```

Use `tmux` plus `ptycho311` for long runs. The exact CLI surface may evolve during implementation, but the ownership pattern is fixed:

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ptycho311
python -m scripts.studies.born_rytov_dt.generate_brdt_dataset \
  --output-root .artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-brdt-four-row-preflight/decision_support_dataset \
  --train-count 2048 --val-count 256 --test-count 256 --dry-run-manifest
```

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ptycho311
python -m scripts.studies.born_rytov_dt.run_preflight \
  --manifest .artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-brdt-four-row-preflight/decision_support_dataset/dataset_manifest.json \
  --output-root .artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-brdt-four-row-preflight
```

The implementation phase owns those runs until:

- the tracked PID exits with code `0`;
- the expected dataset or bundle artifacts are freshly written;
- any recoverable runtime failure has been diagnosed, fixed narrowly, and rerun or converted into an explicit row-level blocker.

## Completion Criteria

- The BRDT four-row preflight has a separate capped decision-support dataset contract and did not overwrite the smoke dataset artifacts.
- The final bundle includes the exact required row roster with equal-footing metadata and explicit row statuses.
- Image-space physical-`q` metrics, measurement-space residuals, parameter counts, runtime/hardware metadata, fixed-sample visuals, source arrays, JSON/CSV tables, metric schema, and manifests all exist under this item's artifact root.
- `preflight_manifest.json` clearly labels the bundle as `decision_support_preflight_only`, records the follow-up owner `2026-04-29-brdt-preflight-summary-promotion-decision`, and points to the full artifact set needed for that later durable summary.
- `docs/plans/NEURIPS-HYBRID-RESNET-2026/evidence_matrix.md`, `model_variant_index.json`, and `paper_evidence_index.md` are updated to point at the new BRDT preflight outputs.
- The backlog-required checks pass, and any remaining unmet row is represented as an explicit row-level blocker rather than hidden protocol drift.
