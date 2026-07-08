# Corrected BRDT FFNO 20-Epoch Rerun Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. Do not create worktrees. Use `tmux` plus `ptycho311` for multi-minute BRDT runs, track the exact launched PID, and consider the run complete only when that PID exits `0` and the expected artifacts are freshly written.

**Goal:** Produce a fresh BRDT `ffno` 20-epoch row under the corrected no-refiner adapter contract, preserve the historical FFNO-local-refiner proxy as lineage-only context, and make the corrected row the discoverable pure-BRDT-FFNO authority for this capped candidate lane.

**Architecture:** Reuse the completed BRDT four-row preflight as immutable read-only lineage, keep the exact same dataset/operator/input/split/fixed-sample/training contract, and launch only one corrected `ffno` row. Before any long run, either prove the existing FFNO-extension runner can emit truthful `2026-05-06` metadata and no-refiner audits or make the smallest runner/bundle change needed to do so. After the run, publish a new append-only corrected root, a same-contract audit showing only the FFNO architecture contract changed, and durable summary/index updates that point pure-FFNO readers to the corrected root without rewriting the historical proxy lineage.

**Tech Stack:** PATH `python`, `ptycho311` for long-running launches, PyTorch/Lightning, `scripts/studies/born_rytov_dt/`, JSON/CSV/PNG/NumPy artifacts, pytest, compileall, Markdown/JSON evidence indexes.

---

## Selected Objective

- Rerun exactly the BRDT `ffno` row under the original supervised+Born `20`-epoch contract using the corrected no-refiner adapter.
- Keep the completed BRDT four-row preflight as read-only lineage and do not rerun `unet`, `fno_vanilla`, `hybrid_resnet`, or `classical_born_backprop`.
- Publish a corrected append-only five-row view and durable discoverability updates so the historical `2026-05-04` FFNO row remains explicit proxy context only.

## Scope Boundaries

### In Scope

- A narrow audit of the current BRDT FFNO entrypoint and lineage helpers to confirm they can emit truthful `2026-05-06` backlog metadata, corrected row labels, and a no-refiner architecture contract.
- Any minimal code/test changes needed to launch only the corrected `ffno` row and write corrected lineage/manifest outputs without altering the frozen BRDT dataset, operator, split, metrics, or historical artifact roots.
- One fresh corrected `ffno` run under the same:
  - dataset id `brdt128_decision_support_preflight`
  - `born_init_image` input contract
  - split counts `2048 / 256 / 256`
  - fixed sample ids `[145, 83, 255, 126]`
  - normalization stats
  - training contract `epochs=20`, `batch_size=16`, `optimizer=Adam`, `learning_rate=2e-4`, `seed=42`, and baseline loss weights
- A corrected same-contract combined bundle and a row-level audit proving the executed FFNO row is no-refiner and same-footing with the baseline bundle except for the intended FFNO architecture correction.
- Durable summary and evidence-index updates that make the corrected row discoverable as the pure-BRDT-FFNO 20-epoch authority.

### Explicit Non-Goals

- Do not rerun U-Net, FNO vanilla, Hybrid ResNet, or the model-based Born row.
- Do not overwrite `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-04-brdt-ffno-row-extension/` or any part of `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-brdt-four-row-preflight/`.
- Do not relax equal-footing by changing dataset identity, split counts, fixed sample ids, normalization, input mode, operator pointer, epoch budget, batch size, learning rate, loss weights, seed, or metric schema.
- Do not promote BRDT into a required NeurIPS pillar, manuscript-ready evidence lane, or paper-grade claim from this item. Claim boundary remains `decision_support_append_only`.
- Do not touch CDI `lines128`, PDEBench CNS, WaveBench, OpenFWI, manuscript tables, efficiency packaging, or the BRDT `40`-epoch rerun assets unless a change is unavoidable fallout of the corrected `20`-epoch lineage update.
- Do not mark the item `BLOCKED` for ordinary import/path/test/harness failures. Diagnose, apply a documented narrow fix, and rerun first. Reserve `BLOCKED` for missing prerequisite artifacts, unavailable hardware/resources, roadmap conflict, external dependency outside current authority, required user decision, or a failure that remains unrecoverable after a narrow fix attempt.

## Steering, Roadmap, And Policy Constraints

- Steering keeps priority on core paper evidence and forbids silent fairness drift. This item is allowed only as bounded candidate-lane maintenance that repairs an already-cited BRDT FFNO comparison without expanding the BRDT lane.
- The roadmap’s required pillars remain CDI `lines128` and PDEBench CNS. BRDT stays optional candidate context and the `defer_after_preflight` decision in `brdt_preflight_summary.md` remains in force.
- Equal-footing is binding:
  - use the completed four-row preflight as the immutable baseline authority
  - keep the same dataset/operator/input/split/fixed-sample/training contract
  - if the corrected row fails, record a row-level blocker rather than reintroducing `cnn_blocks` or other refiners
- Long-running execution remains under implementation ownership until terminal success or recoverable failure handling is complete:
  - launch in `tmux`
  - activate `ptycho311`
  - keep PATH `python`
  - track the exact launched PID
  - only accept completion when that PID exits `0` and required artifacts are freshly written
- PyTorch and legacy-bridge policies still apply if code changes are needed. If any path touches legacy modules, preserve the `update_legacy_dict(params.cfg, config)` ordering rules from project policy.

## Prerequisite Status

- Satisfied prerequisite inputs from the selected backlog item:
  - baseline four-row bundle:
    `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-brdt-four-row-preflight/`
  - baseline manifest:
    `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-brdt-four-row-preflight/preflight_manifest.json`
  - baseline metrics:
    `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-brdt-four-row-preflight/metrics.json`
  - dataset manifest reused by the corrected rerun:
    `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-brdt-four-row-preflight/decision_support_dataset/dataset_manifest.json`
- Satisfied code prerequisites:
  - `scripts/studies/born_rytov_dt/models.py` already enforces that BRDT `ffno` rejects `cnn_blocks`.
  - `scripts/studies/born_rytov_dt/run_config.py` already defines the corrected no-refiner FFNO defaults.
- Progress-ledger status that matters:
  - `state/NEURIPS-HYBRID-RESNET-2026/progress_ledger.json` shows early roadmap tranches complete and no active blocked tranche for this initiative.
  - The ledger does not enumerate this late candidate-lane repair directly, so the effective prerequisite authority for this item is the selected backlog context plus the completed BRDT preflight/extension summaries and artifacts above.

## Implementation Architecture

- **Unit 1: Runner and metadata truthfulness**
  - Verify whether the current `run_ffno_extension.py` and `extension_bundle.py` can emit a corrected `2026-05-06` backlog id, corrected summary/labeling, and explicit no-refiner architecture metadata without corrupting the historical `2026-05-04` defaults.
  - If not, extract shared logic and add a narrow corrected entrypoint or parameterization that leaves the historical path stable.

- **Unit 2: Corrected same-contract FFNO rerun**
  - Launch exactly one corrected `ffno` row under the locked `20`-epoch BRDT contract.
  - Capture row-local provenance, model profile, metrics, visuals, and combined lineage artifacts in a new append-only root.

- **Unit 3: Audit and discoverability refresh**
  - Write a same-contract audit proving only the FFNO architecture contract changed and that the executed corrected row contains no post-bottleneck local refiners.
  - Add or update durable summaries and evidence indexes so downstream users discover the corrected authority first while the historical proxy remains explicit preserved lineage.

## File And Artifact Targets

### Mandatory contract outputs

- Fresh corrected artifact root:
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-06-brdt-corrected-ffno-row-rerun/`
- Corrected top-level artifacts under that root, at minimum:
  - `preflight_manifest.json`
  - `metrics.json`
  - `metrics.csv`
  - `metric_schema.json`
  - `visual_manifest.json`
  - `combined_metrics.json`
  - `combined_metrics.csv`
  - `combined_manifest.json`
  - top-level `invocation.json` and `invocation.sh`
- Corrected row-local outputs under `rows/ffno/`, at minimum:
  - `row_summary.json`
  - `model_profile.json`
  - `invocation.json`
  - `invocation.sh`
  - `model_state.pt`
  - row-local metrics/history artifacts if emitted by the runner
- Corrected visual/source-array outputs for the fixed sample set under the new root.
- A no-refiner and same-contract audit artifact under the corrected root, preferably machine-readable JSON plus a short Markdown note if needed.
- Durable summary authority for the corrected rerun:
  `docs/plans/NEURIPS-HYBRID-RESNET-2026/brdt_corrected_ffno_row_rerun_summary.md`
- Required discoverability updates:
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/brdt_ffno_row_extension_summary.md`
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/evidence_matrix.md`
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/model_variant_index.json`
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/ablation_index.json`
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/paper_evidence_index.md`
  - `docs/index.md`

### Likely code and test surfaces if a narrow fix is needed

- `scripts/studies/born_rytov_dt/run_ffno_extension.py`
- `scripts/studies/born_rytov_dt/extension_bundle.py`
- optionally a narrow sibling entrypoint under `scripts/studies/born_rytov_dt/` for the corrected rerun if rebadging the historical script would blur lineage
- `scripts/studies/born_rytov_dt/models.py` only if the audit finds an actual contract drift
- `scripts/studies/born_rytov_dt/run_config.py` only if the audit finds the defaults do not match the current no-refiner contract
- `tests/studies/test_born_rytov_dt_adapters.py`
- `tests/studies/test_born_rytov_dt_preflight.py`

### Preferred packaging after core completion

- A concise corrected-vs-historical contract-diff note under the corrected root for human review convenience.
- Optional convenience validation transcripts under the corrected root showing the no-refiner inspection and lineage checks.
- Keep broader BRDT `40`-epoch paper-evidence refreshes, efficiency-table refreshes, and manuscript-facing figure/table rewrites out of scope for this item.

## Execution Checklist

### Task 1: Freeze Inputs And Audit The Current Corrected Contract

- [ ] Confirm the required selected-item inputs exist before editing or launching:
  - baseline `preflight_manifest.json`
  - baseline `metrics.json`
  - baseline `metrics.csv`
  - current historical summary `docs/plans/NEURIPS-HYBRID-RESNET-2026/brdt_ffno_row_extension_summary.md`
  - current plan path and target execution plan
- [ ] Confirm the baseline contract fields to inherit exactly:
  - dataset manifest path `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-brdt-four-row-preflight/decision_support_dataset/dataset_manifest.json`
  - operator validation artifact path from the baseline manifest
  - fixed sample ids `[145, 83, 255, 126]`
  - training contract `20` epochs, batch `16`, Adam `2e-4`, seed `42`, and baseline loss weights
- [ ] Instantiate the current BRDT `ffno` adapter and verify the corrected no-refiner contract:
  - `parameter_count == 27394`
  - `cnn_blocks` is rejected
  - no post-bottleneck local refiner modules are instantiated
- [ ] Inspect the existing `run_ffno_extension.py` and `extension_bundle.py` path and determine whether they can already emit truthful corrected metadata for backlog item `2026-05-06-brdt-corrected-ffno-row-rerun`.
- [ ] If the existing path is already truthful, record that no production code change is required before the rerun.
- [ ] If it is not truthful, write down the smallest required fix surface before editing:
  - parameterize the existing runner/bundle while keeping historical defaults stable, or
  - add a narrow corrected-rerun sibling entrypoint that reuses shared helpers but emits `2026-05-06` lineage and no-refiner contract fields

Verification for Task 1:

- Blocking before any long run:
  - `python - <<'PY'
from pathlib import Path
required = [
    Path("docs/plans/NEURIPS-HYBRID-RESNET-2026/brdt_ffno_row_extension_summary.md"),
    Path("docs/plans/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-06-brdt-corrected-ffno-row-rerun/execution_plan.md"),
    Path(".artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-brdt-four-row-preflight/preflight_manifest.json"),
    Path(".artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-brdt-four-row-preflight/metrics.json"),
]
missing = [str(path) for path in required if not path.exists()]
if missing:
    raise SystemExit(f"missing corrected BRDT FFNO rerun inputs: {missing}")
print("corrected BRDT FFNO rerun inputs present")
PY`
  - `python - <<'PY'
from scripts.studies.born_rytov_dt.models import build_neural_adapter
adapter = build_neural_adapter("ffno", in_channels=1)
info = adapter.info()
assert info.parameter_count == 27394, info.parameter_count
assert "cnn_blocks" not in info.arch_kwargs, info.arch_kwargs
try:
    build_neural_adapter("ffno", in_channels=1, arch_kwargs={"cnn_blocks": 2})
except ValueError as exc:
    assert "does not accept cnn_blocks" in str(exc)
else:
    raise AssertionError("cnn_blocks should be rejected")
print("BRDT FFNO no-refiner adapter contract verified")
PY`
- Supporting:
  - a short written note naming whether the corrected rerun will reuse the historical entrypoint or introduce a narrow corrected sibling

### Task 2: Apply The Smallest Runner, Bundle, And Test Fixes Needed

- [ ] Only if Task 1 found a real gap, implement the smallest viable change so the corrected rerun writes truthful `2026-05-06` backlog metadata, corrected labels, and no-refiner contract fields without mutating the historical `2026-05-04` lineage.
- [ ] Keep any fix strictly scoped to the corrected BRDT FFNO rerun path and the shared helper logic it depends on.
- [ ] Preserve historical behavior for `2026-05-04-brdt-ffno-row-extension`; do not silently rewrite its backlog id, claim boundary, row label, or output-root semantics.
- [ ] Add or update focused tests that prove:
  - corrected rerun metadata/manifest fields are emitted under the new backlog id
  - the historical extension path remains stable
  - missing required lineage files still fail closed
  - corrected no-refiner contract metadata can be discovered from the emitted artifacts

Verification for Task 2:

- Blocking before any long run:
  - `pytest -q tests/studies/test_born_rytov_dt_adapters.py tests/studies/test_born_rytov_dt_preflight.py`
  - `python -m compileall -q scripts/studies/born_rytov_dt ptycho_torch`
- Supporting:
  - any new focused selector added for corrected metadata or lineage validation

### Task 3: Launch The Corrected Same-Contract FFNO Rerun

- [ ] Launch exactly one corrected `ffno` row with the baseline root and dataset manifest above, writing into:
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-06-brdt-corrected-ffno-row-rerun/`
- [ ] Use the authoritative corrected entrypoint from Tasks 1-2.
- [ ] Keep the launch arguments on the locked same-contract surface:
  - baseline root:
    `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-brdt-four-row-preflight/`
  - manifest:
    `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-brdt-four-row-preflight/decision_support_dataset/dataset_manifest.json`
  - output root:
    `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-06-brdt-corrected-ffno-row-rerun/`
  - inherited training contract `epochs=20`, `batch_size=16`, `learning_rate=2e-4`, `seed=42`
  - fixed sample seed/count matching the baseline bundle unless the helper derives the exact baseline ids directly
- [ ] Run in `tmux`, activate `ptycho311`, track the exact launched PID, and wait on that PID rather than polling broad process names.
- [ ] If the first long run fails, diagnose the narrow cause, repair it, and rerun or resume once before considering the item unrecoverable.

Verification for Task 3:

- Blocking:
  - tracked PID exits `0`
  - the corrected root contains the mandatory top-level and `rows/ffno/` outputs named in this plan
  - emitted corrected metadata names backlog item `2026-05-06-brdt-corrected-ffno-row-rerun`
  - emitted corrected profile/manifest surfaces record the no-refiner architecture contract and `parameter_count=27394`
- Supporting:
  - live tmux log
  - training/eval runtime sanity versus the historical `20`-epoch FFNO proxy row

### Task 4: Audit No-Refiner Purity And Same-Contract Fairness

- [ ] Write a same-contract audit comparing the corrected rerun against the read-only historical FFNO extension and the baseline four-row preflight.
- [ ] Prove the corrected row differs from the historical FFNO proxy only where expected:
  - no `cnn_blocks`
  - no post-bottleneck local refiners
  - corrected parameter count `27394` instead of `36674`
  - corrected backlog id, artifact root, and row labeling
- [ ] Prove all fairness-relevant fields match the baseline contract:
  - dataset id
  - split counts
  - normalization
  - input mode
  - fixed sample ids
  - operator pointer/geometry
  - training contract
- [ ] If any non-allowed drift appears, treat it as a correctness failure and fix/rerun rather than accepting an unequal-footing comparison.

Verification for Task 4:

- Blocking:
  - same-contract audit shows no non-allowed drift
  - no-refiner audit proves the executed corrected row contains zero post-bottleneck local refiner modules and rejects `cnn_blocks`
  - combined five-row lineage output uses the read-only baseline rows plus the corrected FFNO row, not the historical proxy row
- Supporting:
  - a human-readable metric delta note comparing corrected-vs-historical FFNO for context only

### Task 5: Write Durable Summary And Refresh Discoverability

- [ ] Write `docs/plans/NEURIPS-HYBRID-RESNET-2026/brdt_corrected_ffno_row_rerun_summary.md` with:
  - fixed contract
  - baseline lineage roots
  - corrected no-refiner architecture contract
  - corrected artifact root
  - key metric and parameter deltas versus the historical proxy
  - claim boundary `decision_support_append_only`
  - explicit statement that BRDT remains a deferred candidate lane, not a required pillar
- [ ] Update `brdt_ffno_row_extension_summary.md` so it clearly points readers to the corrected summary for pure-FFNO use while keeping the historical proxy row preserved as lineage-only context.
- [ ] Update `docs/index.md` so the corrected summary is discoverable and the historical entry still advertises itself as proxy context.
- [ ] Update `evidence_matrix.md`, `model_variant_index.json`, `ablation_index.json`, and `paper_evidence_index.md` so:
  - the corrected no-refiner row is discoverable as the pure-BRDT-FFNO `20`-epoch authority
  - the historical `2026-05-04` row remains explicitly labeled `FFNO-local proxy` or equivalent caveated wording
  - any historical entries continue to point at `2026-05-06-brdt-corrected-ffno-row-rerun` as the pure-FFNO successor

Verification for Task 5:

- Blocking:
  - corrected summary exists and names the corrected root plus the preserved historical proxy root
  - `docs/index.md` includes the corrected summary and preserves the historical proxy caveat
  - evidence indexes distinguish corrected no-refiner authority from preserved proxy lineage
- Supporting:
  - targeted grep / structured JSON spot checks over the touched summary and index files

## Deterministic Verification Gate

Run these as the required deterministic checks for this backlog item. The selected backlog item’s `check_commands` remain mandatory; the added adapter-contract inspection is an extra blocking preflight, not a replacement.

- Blocking before any expensive rerun:
  - `python - <<'PY'
from pathlib import Path
required = [
    Path("docs/plans/NEURIPS-HYBRID-RESNET-2026/brdt_ffno_row_extension_summary.md"),
    Path("docs/plans/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-06-brdt-corrected-ffno-row-rerun/execution_plan.md"),
    Path(".artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-brdt-four-row-preflight/preflight_manifest.json"),
    Path(".artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-brdt-four-row-preflight/metrics.json"),
]
missing = [str(path) for path in required if not path.exists()]
if missing:
    raise SystemExit(f"missing corrected BRDT FFNO rerun inputs: {missing}")
print("corrected BRDT FFNO rerun inputs present")
PY`
  - `python - <<'PY'
from scripts.studies.born_rytov_dt.models import build_neural_adapter
adapter = build_neural_adapter("ffno", in_channels=1)
info = adapter.info()
assert info.parameter_count == 27394, info.parameter_count
assert "cnn_blocks" not in info.arch_kwargs, info.arch_kwargs
print("corrected BRDT FFNO parameter_count:", info.parameter_count)
PY`
  - `pytest -q tests/studies/test_born_rytov_dt_adapters.py tests/studies/test_born_rytov_dt_preflight.py`
  - `python -m compileall -q scripts/studies/born_rytov_dt ptycho_torch`

- Blocking after the rerun:
  - verify the tracked PID exited `0`
  - verify the corrected artifact root contains the mandatory manifest, metrics, combined-lineage, and `rows/ffno/` artifacts named in this plan
  - verify the corrected summary and discoverability updates are written

- Supporting after the rerun:
  - any audit grep/JSON validation proving the historical proxy entries now point to the corrected no-refiner successor

## Completion Gate

- A new corrected append-only BRDT root exists under `2026-05-06-brdt-corrected-ffno-row-rerun` and the historical `2026-05-04` root remains untouched.
- The corrected executed FFNO row is provably no-refiner by code path and emitted artifact audit, and its recorded parameter count is `27394` unless an intentional code change updated the contract and that change was explicitly documented.
- The same-contract audit proves the corrected rerun stayed on equal footing with the baseline bundle except for the intended FFNO architecture correction and corrected metadata lineage.
- Durable summary and evidence-index surfaces now make the corrected row the pure-BRDT-FFNO `20`-epoch authority while preserving the historical FFNO-local-refiner proxy as explicit lineage-only context.
- BRDT remains a deferred candidate lane with claim boundary `decision_support_append_only`; this item does not reopen roadmap scope or paper-grade promotion.
