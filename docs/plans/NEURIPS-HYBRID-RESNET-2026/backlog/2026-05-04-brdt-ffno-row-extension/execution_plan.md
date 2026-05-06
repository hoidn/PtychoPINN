# BRDT FFNO Row Extension Execution Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. This item authorizes only one append-only BRDT FFNO row plus the minimal task-local adapter, execution, and reporting support needed to compare that row against the completed BRDT decision-support preflight. Do not rerun or overwrite the existing classical, U-Net, FNO vanilla, or Hybrid/SRU rows. Do not promote BRDT into manuscript evidence, do not create worktrees, and keep any long-running command under implementation ownership until the tracked PID exits successfully and the expected artifacts are freshly written; use `tmux` plus `ptycho311` when the FFNO row run is multi-minute.

**Goal:** Add a single append-only BRDT FFNO row under the same locked BRDT decision-support contract as the completed four-row preflight, then emit a read-only-lineage five-row comparison surface and durable summary/index updates without modifying the baseline bundle.

**Architecture:** Keep BRDT fully task-local under `scripts/studies/born_rytov_dt/`. Reuse the completed four-row preflight as an immutable baseline authority, add a distinct task-local `ffno` neural adapter that preserves ordinary real-channel `model(x) -> q_pred` semantics, run only that row under the same dataset/operator/input/split/normalization/training contract, and assemble an append-only extension bundle that references the original bundle by lineage while publishing a combined five-row metrics/manifests view.

**Tech Stack:** PATH `python`, PyTorch, existing BRDT task-local loader/train/eval/preflight code, `ptycho_torch` FFNO building blocks reused behind a task-local wrapper, JSON/CSV/PNG/NumPy artifact bundles, pytest, compileall, Markdown/JSON evidence indexes.

---

## Selected Objective

- Implement backlog item `2026-05-04-brdt-ffno-row-extension`.
- Add exactly one new BRDT row representing factorized Fourier operators under the locked BRDT decision-support contract.
- Compare that new row against the already-generated BRDT neural baseline rows:
  - `unet`
  - `fno_vanilla`
  - `hybrid_resnet` or `sru_net` (preserve the baseline bundle’s visible label choice)
- Preserve the completed four-row preflight as a read-only authority and produce append-only extension artifacts plus durable summary/index updates.

## Scope

- Consume these binding authorities and artifacts exactly as they exist:
  - `docs/plans/2026-04-20-neurips-hybrid-resnet-submission-design.md`
  - `docs/plans/2026-04-20-neurips-hybrid-resnet-submission-roadmap.md`
  - `docs/steering.md`
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/born_rytov_dt_candidate_lane_design.md`
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/brdt_task_adapters.md`
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/brdt_preflight_summary.md`
  - `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-brdt-four-row-preflight/preflight_manifest.json`
  - `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-brdt-four-row-preflight/metrics.json`
- Reuse the completed BRDT decision-support dataset, operator authority, `born_init_image` input mode, physical-`q` target, split counts, normalization stats, fixed sample IDs, training seed, fixed-sample seed, and supervised-plus-Born training objective from the baseline four-row preflight.
- Add only the new FFNO row and the narrow extension machinery needed to:
  - construct or explicitly block the FFNO adapter;
  - run only the FFNO row under the same contract;
  - emit the FFNO row’s provenance, runtime, parameter count, row status, image/measurement metrics, visuals, and source arrays;
  - emit a combined five-row metrics/manifest view that references the baseline bundle by lineage rather than by copy-overwrite.
- Update durable discoverability only after the FFNO row finishes as `completed` or `blocked` with a structured reason.

## Explicit Non-Goals

- Do not rerun or overwrite the existing classical, U-Net, FNO vanilla, Hybrid ResNet, or SRU-Net BRDT rows.
- Do not change the BRDT input contract away from `born_init_image`.
- Do not mix direct-sinogram input, Rytov mode, limited-angle data, multi-seed robustness, classical rework, or objective-preset changes into this item.
- Do not relabel the result as paper-grade, manuscript-ready, or a new NeurIPS pillar. BRDT remains deferred candidate evidence.
- Do not rewrite the current `defer_after_preflight` recommendation in `brdt_preflight_summary.md`; this item may add adjacent decision support only.
- Do not register BRDT or FFNO in the CDI generator registry, and do not alter the core CDI/TensorFlow physics modules guarded by project policy.
- Do not silently collapse `ffno` into `fno_vanilla`. The new row must have a distinct row identity, architecture metadata, parameter count, provenance, and row-status payload.

## Steering, Roadmap, and Fairness Constraints

- Steering keeps CDI `lines128` and PDEBench CNS as the required manuscript pillars. BRDT remains additive candidate work only.
- The roadmap allows candidate-lane BRDT follow-up only as bounded decision-support evidence; it does not authorize BRDT manuscript promotion.
- The selected backlog reviewer notes are binding:
  - add only the FFNO row and minimal support needed to run or explicitly block it;
  - preserve the same BRDT operator, dataset, `born_init_image` input mode, physical `q` target, split, normalization, metric schema, and supervised-plus-Born training procedure;
  - do not rerun or overwrite the existing four-row bundle;
  - do not mix direct-sinogram, Rytov, limited-angle, or multi-seed variants into this item.
- Apples-to-apples contract for the new FFNO row:
  - same dataset id: `brdt128_decision_support_preflight`
  - same split counts: `2048 / 256 / 256`
  - same operator geometry: `N=D=128`, `A=64`, `wavelength_px=8.0`, `medium_ri=1.333`, Born mode, `unitary_fft`
  - same fixed-sample seed/id contract from the baseline bundle
  - same training budget: `20` epochs, Adam, `lr=2e-4`, batch size `16`, seed `42`
  - same loss weights as the completed four-row preflight: `image=1.0`, `physics=0.1`, `relative_physics=0.1`, `tv=1e-5`, `positivity=1e-4`
- If the FFNO row cannot satisfy that exact contract after a narrow fix attempt, record a structured row-level blocker instead of relaxing the protocol.

## Prerequisite Status

- The initiative progress ledger shows no blocked core NeurIPS tranches that pre-empt this bounded candidate-lane work; the completed initiative tranches remain Phase 0/1 and selected Phase 2 items, with no BRDT-specific tranche tracked there yet.
- BRDT prerequisite authority is instead carried by already-completed backlog outputs:
  - operator validation: complete
  - dataset preflight: complete
  - task adapters: complete
  - four-row decision-support preflight and summary/promotion decision: complete
- The baseline BRDT bundle is therefore the operational prerequisite for this item. Implementation must treat its `preflight_manifest.json` and `metrics.json` as authoritative and immutable.

## Execution Rules

- Diagnose, fix, and rerun ordinary import, environment, path, CLI, optional-dependency, or test-harness failures before escalating.
- Reserve item-level `BLOCKED` for missing external resources, hardware unavailability after a documented narrow fix attempt, roadmap conflict, or an unrecoverable contract contradiction between the baseline bundle and the new FFNO row machinery.
- Use a row-level `blocked` result for FFNO-specific adapter or dependency failures when the baseline bundle itself remains valid.
- Keep durable outputs under:
  - `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-04-brdt-ffno-row-extension/`
- Keep scratch under `tmp/` and remove non-durable scratch before completion.
- Do not launch duplicate runs against the same extension `--output-root`.
- Completion of the live FFNO execution requires both:
  - the tracked PID exiting with code `0`
  - the expected row and bundle artifacts being freshly written under the extension root

## Implementation Architecture

1. **Row Contract and Adapter Layer**
   - Owns the new distinct `ffno` row schema, architecture defaults, dependency gating, and ordinary real-channel FFNO adapter construction without touching the CDI generator registry.
2. **Append-Only Extension Execution Layer**
   - Owns baseline-bundle validation, single-row FFNO launch/dry-run behavior, backlog-item identity, claim-boundary labeling, and row-local provenance/status artifacts under a separate extension root.
3. **Combined Bundle and Discoverability Layer**
   - Owns the read-only-lineage five-row combined metrics/manifest/visual surfaces plus the checked-in summary and durable evidence-index updates that make the extension discoverable without reopening raw artifacts.

## File and Artifact Targets

Mandatory contract outputs:

- `tests/studies/test_born_rytov_dt_adapters.py`
- `tests/studies/test_born_rytov_dt_preflight.py`
- `scripts/studies/born_rytov_dt/run_config.py`
- `scripts/studies/born_rytov_dt/models.py`
- `scripts/studies/born_rytov_dt/run_preflight.py` if reusable helper extraction or row-roster support is needed
- `scripts/studies/born_rytov_dt/run_ffno_extension.py`
- `scripts/studies/born_rytov_dt/extension_bundle.py`
- `scripts/studies/born_rytov_dt/preflight_visuals.py` if combined five-row figure emission needs extension support
- `docs/plans/NEURIPS-HYBRID-RESNET-2026/brdt_ffno_row_extension_summary.md`
- `docs/plans/NEURIPS-HYBRID-RESNET-2026/evidence_matrix.md`
- `docs/plans/NEURIPS-HYBRID-RESNET-2026/model_variant_index.json`
- `docs/plans/NEURIPS-HYBRID-RESNET-2026/ablation_index.json`
- `docs/plans/NEURIPS-HYBRID-RESNET-2026/paper_evidence_index.md`
- `docs/index.md`
- `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-04-brdt-ffno-row-extension/preflight_manifest.json`
- `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-04-brdt-ffno-row-extension/metrics.json`
- `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-04-brdt-ffno-row-extension/metrics.csv`
- `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-04-brdt-ffno-row-extension/metric_schema.json`
- `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-04-brdt-ffno-row-extension/visual_manifest.json`
- `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-04-brdt-ffno-row-extension/combined_metrics.json`
- `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-04-brdt-ffno-row-extension/combined_metrics.csv`
- `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-04-brdt-ffno-row-extension/combined_manifest.json`
- `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-04-brdt-ffno-row-extension/rows/ffno/`

Preferred packaging:

- Keep the baseline-vs-extension merge logic isolated in `scripts/studies/born_rytov_dt/extension_bundle.py` rather than bloating `comparison.py`, which is currently specific to the physics-only objective ablation.
- Keep the backlog-item/claim-boundary wrapper logic in `run_ffno_extension.py` rather than making the generic four-row preflight CLI carry BRDT-extension-specific identity rules.

## Contract Invariants To Preserve

- **Baseline immutability:** the original four-row BRDT bundle under `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-brdt-four-row-preflight/` stays read-only.
- **Distinct FFNO identity:** the new row is surfaced as `row_id="ffno"` and `paper_label="FFNO"` unless an unavoidable implementation detail requires a more specific internal helper name; if that happens, manifests and summaries must still present a distinct visible FFNO row and must not reuse `fno_vanilla`.
- **Same-contract training:** the FFNO row inherits the baseline dataset/operator/input/split/normalization/fixed-sample/training recipe exactly.
- **Append-only claim boundary:** the extension root and checked-in summary are `decision_support_append_only`, not `decision_support_preflight_only`, `bounded_capped`, or `paper_grade`.
- **Five-row combined view:** the combined manifest/metrics view preserves the existing four-row baseline context, including the blocked classical row, and adds the FFNO row by lineage rather than by rerun.
- **No protocol drift for convenience:** if FFNO requires a different history length, input representation, normalization, or objective preset to function, that is a blocker or a separate future plan, not a silent adjustment here.

## Execution Tranches

### Tranche 1: Lock the FFNO Extension Contract and Baseline Lineage

**Purpose:** Freeze the exact append-only contract before any adapter or runtime work.

- [ ] Write failing tests first in `tests/studies/test_born_rytov_dt_adapters.py` and `tests/studies/test_born_rytov_dt_preflight.py` for:
  - acceptance of a distinct `ffno` row/model in the BRDT row schema;
  - rejection of any attempt to alias `ffno` to `fno_vanilla`;
  - validation that the FFNO extension runner reads the baseline `preflight_manifest.json` and refuses contract mismatches on dataset id, operator pointer, `born_init_image`, normalization, fixed sample IDs, training contract, or claim boundary;
  - validation that the combined metrics/manifest view preserves the baseline rows read-only and adds exactly one FFNO row.
- [ ] Extend `scripts/studies/born_rytov_dt/run_config.py` so the BRDT row schema can represent a distinct FFNO row and default arch kwargs for that row without broadening the original four-row roster by default.
- [ ] Add a narrow helper, preferably in `scripts/studies/born_rytov_dt/extension_bundle.py` or `run_ffno_extension.py`, that fingerprints or validates the baseline bundle contract before any live launch.
- [ ] Keep the baseline preflight’s visible Hybrid-family label authoritative for the combined five-row view; do not relabel the existing baseline row in this item.

**Verification**

- [ ] **Blocking:** the updated adapter/preflight tests must prove the FFNO row schema and baseline-contract validation behavior before any live FFNO execution is attempted.
- [ ] **Supporting:** `python -m compileall -q scripts/studies/born_rytov_dt ptycho_torch` may be run early for syntax/import feedback but does not replace the test gate.

### Tranche 2: Implement the Task-Local BRDT FFNO Adapter

**Purpose:** Make FFNO constructible under ordinary BRDT real-channel semantics.

- [ ] Add task-local FFNO adapter support in `scripts/studies/born_rytov_dt/models.py`.
  - Reuse the repo’s existing FFNO building blocks from `ptycho_torch/generators/ffno.py` and related factorized Fourier components only through a BRDT-local wrapper.
  - Keep the adapter contract as `model(x) -> q_pred`, with input `(B, C_in, 128, 128)` and output `(B, 1, 128, 128)`.
  - Preserve parameter-count reporting and runtime metadata parity with the existing BRDT neural rows.
- [ ] Add or adjust FFNO architecture defaults in `run_config.py` so the row has explicit hidden-width / modes / block-count settings recorded in provenance.
- [ ] If a reusable FFNO body import fails, surface a controlled `AdapterBuildError` with a structured blocker reason rather than falling back to a different architecture.
- [ ] Extend adapter tests to cover:
  - FFNO adapter construction;
  - forward pass on BRDT-shaped tensors;
  - distinct architecture metadata and parameter count;
  - controlled missing-dependency behavior if any optional import boundary exists.

**Verification**

- [ ] **Blocking:** the adapter test module must pass with explicit FFNO coverage before any extension dry-run or live row launch.
- [ ] **Supporting:** if the FFNO wrapper introduces a small helper module, compile it together with `scripts/studies/born_rytov_dt` and `ptycho_torch` before moving to the runner tranche.

### Tranche 3: Add the Append-Only FFNO Extension Runner and Combined Bundle Assembly

**Purpose:** Run only the FFNO row under a separate extension identity, then merge it with the baseline bundle by lineage.

- [ ] Add `scripts/studies/born_rytov_dt/run_ffno_extension.py` as the authoritative entrypoint for this backlog item.
  - It should consume the baseline BRDT bundle root plus the baseline dataset manifest path.
  - It should validate the baseline contract before launch.
  - It should execute only `row_id="ffno"` under the supervised-plus-Born contract.
  - It should record `backlog_item: "2026-05-04-brdt-ffno-row-extension"` and `claim_boundary: "decision_support_append_only"` in invocation and manifest surfaces.
- [ ] Reuse the existing BRDT train/eval/preflight internals where practical, but do not let the FFNO extension masquerade as the original four-row backlog item.
- [ ] Add `scripts/studies/born_rytov_dt/extension_bundle.py` to build:
  - `combined_metrics.json`
  - `combined_metrics.csv`
  - `combined_manifest.json`
  These must preserve the original four-row rows by lineage and append the FFNO row as the fifth row.
- [ ] Ensure the extension root emits:
  - row-local provenance under `rows/ffno/`
  - FFNO image-space physical-`q` metrics
  - FFNO measurement-space residual metrics
  - FFNO parameter count and runtime
  - FFNO fixed-sample source arrays
  - a visual manifest and rendered figure set that makes FFNO comparable to the baseline neural rows
- [ ] Keep the baseline paper-local assets and baseline bundle files untouched.
- [ ] Add or extend tests in `tests/studies/test_born_rytov_dt_preflight.py` for:
  - dry-run FFNO extension manifest writing
  - baseline-lineage validation failures
  - live-path simulated row emission for FFNO only
  - combined five-row metrics/manifest generation
  - refusal to overwrite baseline outputs

**Verification**

- [ ] **Blocking:** a dry-run FFNO extension command must succeed and emit fresh `preflight_manifest.json` plus `metric_schema.json` under the new extension root before any live training.
- [ ] **Blocking:** the preflight test module must pass with FFNO extension coverage before the live run.
- [ ] **Supporting:** rerun the adapter test module if FFNO-runner work changes shared model-construction or row-schema code.

### Tranche 4: Execute the FFNO Row and Publish the Durable Outcome

**Purpose:** Produce the append-only FFNO result, then make it discoverable without overstating the claim boundary.

- [ ] Run the FFNO extension dry gate first. Expected command shape:

```bash
python -m scripts.studies.born_rytov_dt.run_ffno_extension \
  --baseline-root .artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-brdt-four-row-preflight \
  --manifest .artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-brdt-four-row-preflight/decision_support_dataset/dataset_manifest.json \
  --output-root .artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-04-brdt-ffno-row-extension \
  --epochs 20 --batch-size 16 --learning-rate 2e-4 --seed 42 \
  --fixed-sample-seed 17 --device cuda --dry-run
```

- [ ] If the dry gate passes, launch the live FFNO row under the same contract. If the run is multi-minute, use `tmux` in `ptycho311` and keep ownership until the tracked PID exits `0` and the expected extension artifacts are fresh.
- [ ] If FFNO cannot build or run under the locked contract after a narrow fix attempt, write a structured `blocked` row outcome and still emit the append-only manifest/summary/index updates for that blocked state.
- [ ] Write `docs/plans/NEURIPS-HYBRID-RESNET-2026/brdt_ffno_row_extension_summary.md` as the checked-in summary authority.
  - It must state the unchanged BRDT candidate-lane boundary.
  - It must describe the same-contract five-row comparison read.
  - It must reaffirm that BRDT remains deferred candidate evidence rather than manuscript authority.
- [ ] Update durable discoverability after the FFNO row is completed or blocked:
  - `evidence_matrix.md`: add the new append-only BRDT FFNO extension outcome and artifact root
  - `paper_evidence_index.md`: add the completed backlog outcome row with `decision_support_append_only`
  - `model_variant_index.json`: add the FFNO variant entry if the row completed; if it blocked, record that explicitly in the summary/index surfaces rather than inventing a completed variant
  - `ablation_index.json`: add the append-only row-extension record because this item is a same-contract BRDT architecture extension relative to the completed preflight
  - `docs/index.md`: add the new summary authority so it is discoverable from the documentation hub

**Verification**

- [ ] **Blocking:** the live run is complete only when `rows/ffno/`, `metrics.json`, `combined_metrics.json`, and `combined_manifest.json` exist freshly under the extension root and the tracked PID exited `0`.
- [ ] **Blocking:** the checked-in summary and required evidence indexes must all point to the same extension root and claim boundary.
- [ ] **Supporting:** if the final implementation adds a new durable helper summary or changes the BRDT discoverability surface materially, update any adjacent doc index only where it improves actual lookup behavior; do not broaden this into a general roadmap rewrite.

## Deterministic Checks

These are the required final deterministic checks from the selected backlog item. Earlier narrower selectors may be used during implementation, but they do not replace this final gate.

- [ ] **Blocking:**

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
```

- [ ] **Blocking:**

```bash
pytest -q tests/studies/test_born_rytov_dt_adapters.py tests/studies/test_born_rytov_dt_preflight.py
```

- [ ] **Blocking:**

```bash
python -m compileall -q scripts/studies/born_rytov_dt ptycho_torch
```

## Completion Criteria

- The extension root contains a truthful append-only FFNO row result or structured row-level blocker under the locked BRDT contract.
- The combined five-row metrics/manifest surfaces are readable without reopening the baseline bundle internals.
- The checked-in summary and evidence indexes all agree on:
  - backlog item id `2026-05-04-brdt-ffno-row-extension`
  - claim boundary `decision_support_append_only`
  - baseline bundle lineage to `2026-04-29-brdt-four-row-preflight`
- No baseline BRDT row artifacts were rerun, overwritten, or relabeled.
