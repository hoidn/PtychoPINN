# Corrected BRDT FFNO 40-Epoch Rerun Execution Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:executing-plans` or `superpowers:subagent-driven-development` to execute this plan task-by-task. Do not create worktrees. Use `tmux` plus the `ptycho311` conda environment for the live BRDT run, track the exact launched PID, and treat the run as complete only when that PID exits `0` and the required output artifacts are freshly written.

**Goal:** Produce a corrected BRDT `40`-epoch bundle under the locked supervised+Born `2048 / 256 / 256` contract, rerunning `hybrid_resnet` and the pure no-refiner `ffno` row in one immutable root with complete provenance, per-epoch history, convergence audit, sample-`255` visuals, and truthful downstream evidence-surface updates.

**Architecture:** Reuse the hardened BRDT `40`-epoch paper-evidence flow, but rebind it to the corrected `2026-05-06-brdt-corrected-ffno-row-rerun` authority instead of the historical FFNO-local-refiner proxy. Keep BRDT inside the `candidate-brdt-preflight` lane: the new bundle may refresh additive BRDT context only after its own gate result is known, and it must never blur or replace the required CDI `lines128` and PDEBench CNS pillars.

**Tech Stack:** PATH `python`, PyTorch, task-local BRDT runners under `scripts/studies/born_rytov_dt/`, JSON/CSV/PNG/NumPy artifacts, `pytest`, `compileall`, tmux, `ptycho311`.

---

## Selected Backlog Objective

- Produce a corrected `40`-epoch BRDT bundle using SRU-Net / Hybrid ResNet and the corrected no-refiner BRDT FFNO adapter.
- Keep the BRDT dataset, operator, input mode, split, normalization, loss weights, seed policy, and fixed-sample policy identical to the locked four-row BRDT contract.
- Regenerate runtime provenance, per-epoch histories, convergence audit, efficiency fields, sample-`255` source arrays/visuals, and any repo-local BRDT tables/figures that still consume the FFNO row.

## Scope

- In scope:
  - a new immutable artifact root at `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-06-brdt-corrected-ffno-40ep-rerun/`
  - a truthful corrected `40`-epoch runner surface that points at the corrected `20`-epoch FFNO authority, not the historical local-refiner proxy
  - fresh `40`-epoch reruns for `hybrid_resnet` and `ffno` in the same root whenever recoverably possible
  - regenerated bundle outputs: metrics, combined metrics, histories, model profiles, runtime provenance, run-exit status, convergence audit, gate payload, sample-`255` visuals, and source arrays
  - durable summary and discoverability/index updates reflecting the new corrected `40`-epoch lineage and the resulting gate status
- Explicit non-goals:
  - do not promote BRDT over CDI `lines128` or PDEBench CNS
  - do not write `/home/ollie/Documents/neurips/` outputs
  - do not change the BRDT dataset/operator/normalization/loss contract
  - do not consume the `2026-05-05` FFNO row as pure-FFNO evidence
  - do not mark this item `BLOCKED` for normal test failures, import issues, path issues, or recoverable launch problems; diagnose, patch, and rerun first

## Binding Constraints

- Steering is binding:
  - BRDT remains additive candidate work only; it cannot replace the approved CDI/CNS pillars.
  - Equal-footing comparisons remain mandatory; do not relax the locked BRDT contract to make the rerun easier.
- Roadmap is binding:
  - `2026-05-06-brdt-corrected-ffno-row-rerun` and `2026-05-06-brdt-corrected-ffno-40ep-rerun` are the authorized path for pure-BRDT-FFNO comparison claims.
  - The `40`-epoch rerun is downstream of the corrected `20`-epoch row and must preserve the append-only candidate-lane claim boundary until its own gate passes.
  - Candidate-lane paper evidence, if achieved, must be recorded through checked-in evidence-surface updates; Phase 5 paper-facing index work remains paused.
- Paper-evidence package design is binding:
  - if the new bundle advertises the promoted boundary, the checked-in package-design amendment must agree with the gate result
  - if provenance is incomplete, keep the bundle at decision-support status rather than fabricating missing host/git data
- Repository guardrails are binding:
  - no worktrees
  - use `tmux` for the live run
  - use PATH `python`
  - preserve the stable physics code boundary (`ptycho/model.py`, `ptycho/diffsim.py`, `ptycho/tf_helper.py` untouched)

## Prerequisite Status

- Required upstream authorities already exist and must be consumed as read-only inputs:
  - baseline same-contract bundle:
    `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-brdt-four-row-preflight/`
  - corrected pure-FFNO `20`-epoch authority:
    `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-06-brdt-corrected-ffno-row-rerun/`
  - historical `40`-epoch caveat authority:
    `docs/plans/NEURIPS-HYBRID-RESNET-2026/brdt_supervised_born_40ep_paper_evidence_summary.md`
- Progress-ledger context:
  - the main NeurIPS initiative tranches needed to authorize candidate-lane follow-up are already underway with no recorded roadmap blocker preventing this BRDT item
  - the corrected `20`-epoch BRDT FFNO row is already discoverable in `paper_evidence_index.md` and must be treated as the authoritative FFNO prerequisite

## Implementation Architecture

- **Runner surface:** keep the historical `40`-epoch runner logic, but isolate corrected-`40` metadata and corrected-`20` input validation so the new backlog item does not reuse `2026-05-05` identifiers, artifact-root tokens, or summary pointers.
- **Gate and artifact surface:** preserve the existing writer-lock, PID-tracked exit-status, same-contract lineage, sample-`255`, and runtime-provenance checks; extend them only where needed to accept the corrected `20`-epoch FFNO root without weakening the old integrity checks.
- **Discoverability surface:** write one new corrected `40`-epoch durable summary and then update the BRDT-related evidence indexes and manifests so pure-FFNO reads move off the historical proxy row while the SRU-Net row can remain secondary context when appropriate.

## Concrete File And Artifact Targets

### Mandatory code and test targets

- Modify or parameterize: `scripts/studies/born_rytov_dt/run_brdt_40ep_paper_evidence.py`
- Create or modify a corrected wrapper/entrypoint dedicated to this backlog item:
  `scripts/studies/born_rytov_dt/run_corrected_ffno_40ep_rerun.py`
- Modify if validation helpers are still hard-wired to the historical FFNO-extension root:
  `scripts/studies/born_rytov_dt/extension_bundle.py`
- Modify only as required by the corrected bundle outputs or gate checks:
  `scripts/studies/born_rytov_dt/convergence.py`
  `scripts/studies/born_rytov_dt/preflight_visuals.py`
  `scripts/studies/born_rytov_dt/comparison.py`
- Test authority:
  `tests/studies/test_born_rytov_dt_preflight.py`
  `tests/studies/test_born_rytov_dt_adapters.py`

### Mandatory contract outputs

- New immutable artifact root:
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-06-brdt-corrected-ffno-40ep-rerun/`
- Required top-level artifacts under that root:
  - `invocation.json`
  - `invocation.sh`
  - `runtime_provenance.json`
  - `run_exit_status.json`
  - `preflight_manifest.json`
  - `metrics.json`
  - `metrics.csv`
  - `combined_metrics.json`
  - `combined_metrics.csv`
  - `combined_manifest.json`
  - `metric_schema.json`
  - `convergence_audit.json`
  - `convergence_audit.csv`
  - `paper_evidence_gate.json`
  - `visual_manifest.json`
- Required per-row outputs:
  - `rows/hybrid_resnet/history.json`
  - `rows/hybrid_resnet/history.csv`
  - `rows/hybrid_resnet/model_profile.json`
  - `rows/hybrid_resnet/row_summary.json`
  - `rows/ffno/history.json`
  - `rows/ffno/history.csv`
  - `rows/ffno/model_profile.json`
  - `rows/ffno/row_summary.json`
- Required visual/source-array artifacts:
  - `visuals/sample_0255_compare_q.png`
  - `visuals/sample_0255_error_q.png`
  - `visuals/sample_0255_sinogram_residual.png`
  - `figures/source_arrays/sample_0255_*`

### Mandatory durable documentation and index targets

- Create:
  `docs/plans/NEURIPS-HYBRID-RESNET-2026/brdt_corrected_ffno_40ep_rerun_summary.md`
- Update:
  `docs/plans/NEURIPS-HYBRID-RESNET-2026/paper_evidence_index.md`
  `docs/plans/NEURIPS-HYBRID-RESNET-2026/paper_evidence_manifest.json`
  `docs/plans/NEURIPS-HYBRID-RESNET-2026/model_variant_index.json`
  `docs/plans/NEURIPS-HYBRID-RESNET-2026/ablation_index.json`
  `docs/index.md`

### Conditional packaging targets

- Update only if the corrected `40`-epoch result is now the authoritative source for an existing repo-local BRDT package surface:
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/tables/brdt_decision_support_metrics.{json,csv,tex}`
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/figures/brdt_decision_support_recon.png`
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/tables/paper_efficiency_table.{json,csv,tex}`
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/paper_efficiency_table_summary.md`
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/paper_evidence_package_design.md`

## Tasks

### Task 1: Rebind the `40`-Epoch Runner To The Corrected FFNO Authority

**Files:**
- Modify: `scripts/studies/born_rytov_dt/run_brdt_40ep_paper_evidence.py`
- Create or modify: `scripts/studies/born_rytov_dt/run_corrected_ffno_40ep_rerun.py`
- Modify if needed: `scripts/studies/born_rytov_dt/extension_bundle.py`
- Test: `tests/studies/test_born_rytov_dt_preflight.py`

- [ ] Replace the historical `2026-05-05` backlog identity, artifact-root token, and durable-summary wiring with a corrected `2026-05-06-brdt-corrected-ffno-40ep-rerun` surface without weakening the old runner’s integrity checks.
- [ ] Point the FFNO prerequisite validation at the corrected `20`-epoch root and require no-refiner proof (`cnn_blocks` absent, no post-bottleneck refiners, corrected parameter-count/profile lineage).
- [ ] Keep the live runner’s writer-lock, duplicate-writer refusal, PID-tracked exit-status recording, and explicit rebuild/reconstruct semantics intact.
- [ ] Make the default execution path rerun both `hybrid_resnet` and `ffno` in one root. If the implementation eventually has to fall back to lineage reuse for `hybrid_resnet`, require that fallback to be explicit in manifests and summaries and keep the result demoted from any fresh paired-evidence wording.

**Verification**

- [ ] **Blocking:** run the backlog input-presence check exactly as required:

```bash
python - <<'PY'
from pathlib import Path
required = [
    Path("docs/plans/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-06-brdt-corrected-ffno-40ep-rerun/execution_plan.md"),
    Path(".artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-brdt-four-row-preflight/preflight_manifest.json"),
]
missing = [str(path) for path in required if not path.exists()]
if missing:
    raise SystemExit(f"missing corrected BRDT 40-epoch inputs: {missing}")
print("corrected BRDT 40-epoch inputs present")
PY
```

- [ ] **Blocking:** run targeted runner/gate selectors before any expensive training:

```bash
pytest -q tests/studies/test_born_rytov_dt_preflight.py -k "run_corrected_ffno_rerun or run_brdt_40ep_paper_evidence or same_contract_lineage or evidence_surfaces or reconstruct_runtime_provenance"
```

- [ ] **Blocking:** run the required deterministic pytest command from the backlog item:

```bash
pytest -q tests/studies/test_born_rytov_dt_adapters.py tests/studies/test_born_rytov_dt_preflight.py
```

- [ ] **Blocking:** run the required deterministic compile check from the backlog item:

```bash
python -m compileall -q scripts/studies/born_rytov_dt ptycho_torch
```

### Task 2: Prove The Corrected Bundle Surface In Dry-Run Mode

**Files/artifacts:**
- Create or refresh: `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-06-brdt-corrected-ffno-40ep-rerun/`

- [ ] Run the corrected `40`-epoch entrypoint in `--dry-run` mode against:
  - baseline root:
    `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-brdt-four-row-preflight`
  - corrected FFNO prerequisite root:
    `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-06-brdt-corrected-ffno-row-rerun`
  - dataset manifest:
    `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-brdt-four-row-preflight/decision_support_dataset/dataset_manifest.json`
- [ ] Keep the locked run contract: `epochs=40`, `batch_size=16`, `learning_rate=2e-4`, `scheduler=ReduceLROnPlateau`, `plateau_factor=0.5`, `plateau_patience=2`, `plateau_threshold=0.0`, `plateau_min_lr=1e-5`, `seed=42`, `fixed_sample_seed=17`, `visual_sample_id=255`.
- [ ] Verify that the dry-run manifest, combined metrics payload, and visual manifest all advertise the corrected backlog item, corrected artifact root, corrected claim-boundary seed, and the `hybrid_resnet` + corrected `ffno` row roster.
- [ ] If any dry-run payload still references the historical `2026-05-05` bundle or the `2026-05-04` FFNO extension root, fix that before launching the live run.

**Verification**

- [ ] **Blocking:** inspect the dry-run `preflight_manifest.json`, `combined_metrics.json`, and `visual_manifest.json` for corrected backlog identity and claim-boundary consistency.
- [ ] **Supporting:** rerun the most relevant dry-run selector if Task 1 changed again:

```bash
pytest -q tests/studies/test_born_rytov_dt_preflight.py -k "dry_run and (corrected or paper_evidence)"
```

### Task 3: Launch And Complete The Live Corrected `40`-Epoch Bundle

**Files/artifacts:**
- Write the live run only under:
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-06-brdt-corrected-ffno-40ep-rerun/`

- [ ] Launch the live run in `tmux` under `ptycho311`. Use the checked-in corrected entrypoint and the same input roots as Task 2.
- [ ] Track the exact launched PID. Wait on that PID, not a broad process search.
- [ ] Do not launch a duplicate run if the writer lock or a live writer already exists for the output root.
- [ ] Treat `hybrid_resnet` rerun as the default clean paired-evidence path. If a narrow fix attempt still leaves only `ffno` recoverable, record that as explicit mixed-lineage context and keep the claim boundary demoted accordingly; do not silently call it a fresh two-row bundle.
- [ ] On successful completion, confirm the required top-level artifacts and both per-row history files are freshly written before doing any rebuild, packaging, or summary work.

**Expected command shape**

```bash
python -m scripts.studies.born_rytov_dt.run_corrected_ffno_40ep_rerun \
  --baseline-root .artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-brdt-four-row-preflight \
  --ffno-root .artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-06-brdt-corrected-ffno-row-rerun \
  --manifest .artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-brdt-four-row-preflight/decision_support_dataset/dataset_manifest.json \
  --output-root .artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-06-brdt-corrected-ffno-40ep-rerun \
  --epochs 40 \
  --batch-size 16 \
  --learning-rate 2e-4 \
  --scheduler ReduceLROnPlateau \
  --plateau-factor 0.5 \
  --plateau-patience 2 \
  --plateau-threshold 0.0 \
  --plateau-min-lr 1e-5 \
  --seed 42 \
  --fixed-sample-seed 17 \
  --visual-sample-id 255 \
  --device cuda
```

**Verification**

- [ ] **Blocking:** after the tracked PID exits `0`, verify fresh existence of:
  - `rows/hybrid_resnet/history.json`
  - `rows/hybrid_resnet/history.csv`
  - `rows/ffno/history.json`
  - `rows/ffno/history.csv`
  - `metrics.json`
  - `combined_metrics.json`
  - `convergence_audit.json`
  - `paper_evidence_gate.json`
  - `runtime_provenance.json`
  - `run_exit_status.json`
  - `visuals/sample_0255_compare_q.png`
  - `visuals/sample_0255_error_q.png`
  - `figures/source_arrays/sample_0255_*`
- [ ] **Blocking:** confirm `run_exit_status.json` records the tracked PID and exit code `0`.
- [ ] **Blocking:** confirm `runtime_provenance.json` contains real training-run git/host/python/torch fields. If those fields are missing because the live run failed to capture them, fix the capture path and rerun the live bundle; do not paper over the gap at summary time.

### Task 4: Recompute Gate Outputs And Apply Truthful Promotion Or Demotion

**Files:**
- Modify if needed: `scripts/studies/born_rytov_dt/run_brdt_40ep_paper_evidence.py`
- Modify if needed: `scripts/studies/born_rytov_dt/convergence.py`
- Modify if needed: `scripts/studies/born_rytov_dt/preflight_visuals.py`
- Create: `docs/plans/NEURIPS-HYBRID-RESNET-2026/brdt_corrected_ffno_40ep_rerun_summary.md`

- [ ] Read the live bundle’s `paper_evidence_gate.json` and use it as the single source of truth for claim boundary and promotion status.
- [ ] Use `--rebuild-meta-only` only for legitimate meta regeneration after a successful live run and only when the original training-run `runtime_provenance.json` is already present and valid.
- [ ] Use `--reconstruct-runtime-provenance-from-invocation` only if the live bundle somehow suffers the historical overwrite pattern and only to record honest demotion. Never fabricate missing git/host fields.
- [ ] Write the new corrected `40`-epoch durable summary with:
  - the corrected `20`-epoch prerequisite lineage
  - same-contract audit wording
  - explicit `hybrid_resnet` and `ffno` `20 -> 40` read
  - sample-`255` visual provenance
  - final gate result and failed checks, if any
  - explicit statement that BRDT remains additive candidate work and does not replace CDI/CNS
- [ ] If the gate passes, add the checked-in BRDT evidence-package amendment required by the package design and keep the wording precise about additive secondary evidence only.
- [ ] If the gate fails, keep the summary, manifest, and indexes at decision-support status and record the exact unrecovered gate checks.

**Verification**

- [ ] **Blocking:** confirm `paper_evidence_gate.json`, `metrics.json`, `combined_metrics.json`, and `metric_schema.json` all carry the same final claim-boundary string after any rebuild or reseed step.
- [ ] **Supporting:** rerun the gate/rebuild selectors if Task 4 changed code:

```bash
pytest -q tests/studies/test_born_rytov_dt_preflight.py -k "paper_evidence_gate or rebuild_meta_only or reconstruct_runtime_provenance or evidence_surfaces"
```

### Task 5: Refresh Discoverability And Dependent BRDT Packaging Surfaces

**Files:**
- Update: `docs/plans/NEURIPS-HYBRID-RESNET-2026/paper_evidence_index.md`
- Update: `docs/plans/NEURIPS-HYBRID-RESNET-2026/paper_evidence_manifest.json`
- Update: `docs/plans/NEURIPS-HYBRID-RESNET-2026/model_variant_index.json`
- Update: `docs/plans/NEURIPS-HYBRID-RESNET-2026/ablation_index.json`
- Update: `docs/index.md`
- Conditionally update:
  `docs/plans/NEURIPS-HYBRID-RESNET-2026/tables/brdt_decision_support_metrics.{json,csv,tex}`
  `docs/plans/NEURIPS-HYBRID-RESNET-2026/figures/brdt_decision_support_recon.png`
  `docs/plans/NEURIPS-HYBRID-RESNET-2026/tables/paper_efficiency_table.{json,csv,tex}`
  `docs/plans/NEURIPS-HYBRID-RESNET-2026/paper_efficiency_table_summary.md`
  `docs/plans/NEURIPS-HYBRID-RESNET-2026/paper_evidence_package_design.md`

- [ ] Add a new `paper_evidence_index.md` row for `2026-05-06-brdt-corrected-ffno-40ep-rerun` and keep the old `2026-05-05` bundle explicitly labeled as historical proxy context or superseded pure-FFNO context, as appropriate.
- [ ] Update `paper_evidence_manifest.json` so the corrected `40`-epoch source root, claim boundary, promotion status, and blocked-claim reasons agree with the new gate result.
- [ ] Update `model_variant_index.json` for the new `hybrid_resnet` and corrected `ffno` `40`-epoch rows, including parameter counts, runtime fields, training-contract wording, and pure-FFNO caveat removal.
- [ ] Update `ablation_index.json` so the historical proxy-based `40`-epoch family is no longer the current pure-FFNO discoverability target.
- [ ] Update `docs/index.md` to expose the new corrected `40`-epoch summary.
- [ ] Refresh repo-local BRDT table/figure/efficiency packaging only if those surfaces currently cite the BRDT FFNO row; otherwise leave them unchanged and say explicitly in the summary that no packaging surface needed refresh.

**Verification**

- [ ] **Blocking:** confirm the evidence-surface text is mutually consistent about backlog item, artifact root, and claim boundary.
- [ ] **Supporting:** validate machine-readable indexes after editing:

```bash
python -m json.tool docs/plans/NEURIPS-HYBRID-RESNET-2026/model_variant_index.json >/tmp/model_variant_index.json
python -m json.tool docs/plans/NEURIPS-HYBRID-RESNET-2026/ablation_index.json >/tmp/ablation_index.json
python -m json.tool docs/plans/NEURIPS-HYBRID-RESNET-2026/paper_evidence_manifest.json >/tmp/paper_evidence_manifest.json
```

### Task 6: Final Required Checks And Evidence Capture

- [ ] Archive the blocking pytest and compile logs as implementation evidence and reference them from the execution report.
- [ ] Re-run the backlog item’s required deterministic checks after the final code/doc state is in place.
- [ ] In the execution report, state whether the final bundle is:
  - fresh paired corrected `40`-epoch evidence for both `hybrid_resnet` and `ffno`, or
  - mixed-lineage context because `hybrid_resnet` had to remain reused after a documented unrecoverable blocker
- [ ] In the execution report, state whether any repo-local BRDT packaging surfaces were refreshed, and if not, why not.

**Required deterministic checks**

- [ ] **Blocking:** input-presence check:

```bash
python - <<'PY'
from pathlib import Path
required = [
    Path("docs/plans/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-06-brdt-corrected-ffno-40ep-rerun/execution_plan.md"),
    Path(".artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-brdt-four-row-preflight/preflight_manifest.json"),
]
missing = [str(path) for path in required if not path.exists()]
if missing:
    raise SystemExit(f"missing corrected BRDT 40-epoch inputs: {missing}")
print("corrected BRDT 40-epoch inputs present")
PY
```

- [ ] **Blocking:** pytest:

```bash
pytest -q tests/studies/test_born_rytov_dt_adapters.py tests/studies/test_born_rytov_dt_preflight.py
```

- [ ] **Blocking:** compile check:

```bash
python -m compileall -q scripts/studies/born_rytov_dt ptycho_torch
```

## Completion Gate

- The corrected `40`-epoch FFNO row is proven no-refiner by source/root/profile lineage, not just by prose.
- The live bundle records genuine training-run git/host/python/torch provenance and a PID-backed `run_exit_status.json`; if not, the bundle is demoted honestly and the missing checks are explicit.
- The sample-`255` visuals and source arrays come from the corrected `40`-epoch bundle, not the historical local-refiner proxy.
- Every updated evidence surface agrees on backlog item, artifact root, and claim boundary.
- BRDT remains clearly labeled as additive candidate evidence and does not overwrite the scope boundaries of CDI `lines128` or PDEBench CNS.
