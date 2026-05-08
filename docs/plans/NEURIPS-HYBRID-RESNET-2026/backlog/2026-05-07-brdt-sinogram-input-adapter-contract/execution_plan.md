# BRDT Sinogram-Input Adapter Contract Execution Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:executing-plans` or `superpowers:subagent-driven-development` to execute this plan task-by-task. Do not create worktrees. Use PATH `python`. Keep normal verification and smoke commands under implementation ownership until they either succeed or fail after a documented narrow fix attempt. Do not launch the successor `40`-epoch paper-evidence run from this item.

**Goal:** Implement and verify the BRDT learned-model input contract in which SRU-Net and FFNO consume the measured complex sinogram directly while the Born inverse remains a non-learned reference only.

**Architecture:** Preserve the historical `born_init_image` path for old BRDT preflight and rerun lineage, but add a separate `input_mode="sinogram"` contract that flows consistently through row selection, data layout, task-local adapters, training/evaluation/preflight input preparation, and a dedicated smoke runner. Keep BRDT in the additive `candidate-*` lane: this item hardens adapter/readiness surfaces only and prepares the downstream `2026-05-07-brdt-sinogram-input-40ep-paper-evidence` run without promoting BRDT over the required CDI `lines128` and PDEBench CNS pillars.

**Tech Stack:** PATH `python`, PyTorch, task-local BRDT code under `scripts/studies/born_rytov_dt/`, JSON/Markdown artifact summaries, `pytest`, `compileall`, optional `tmux` with `ptycho311` only if the smoke execution stops being short.

---

## Selected Backlog Objective

- Add `input_mode="sinogram"` as a supported BRDT learned-model contract.
- Preserve `input_mode="born_init_image"` for historical preflight/rerun reproducibility.
- Keep `direct_sinogram` rejected as a legacy alias.
- Prove that SRU-Net and FFNO accept measured complex sinograms shaped `(B, 2, 64, 128)` and emit `(B, 1, 128, 128)` target-grid predictions.
- Prove that the measured sinogram is used both as model input and as the Born-consistency target, without computing a fixed Born inverse in the learned-model path.

## Scope

- In scope:
  - BRDT row-schema and runner-selection support for `input_mode="sinogram"`
  - channels-first sinogram tensor conversion and exact shape validation
  - task-local sinogram-input adapters for `ffno` and the Hybrid-family `sru_net` row
  - training, evaluation, preflight, and smoke-path input preparation for the sinogram contract
  - a dedicated feasibility-only smoke artifact root for this backlog item
  - durable summary/index updates needed so the new contract is discoverable and does not get confused with historical Born-image-input lineage
- Explicit non-goals:
  - do not run the `40`-epoch paper-evidence experiment
  - do not refresh manuscript figures, tables, PDF, or package zips
  - do not promote historical Born-image-input bundles into sinogram-input evidence
  - do not rewrite old BRDT summaries as if they always used the new contract
  - do not modify the locked BRDT physics operator or dataset contract unless a blocker is traced to a concrete bug that cannot be resolved inside the adapter/runner surface
  - do not mark the item `BLOCKED` for ordinary test failures, import/path issues, or recoverable smoke-run problems; diagnose, patch, and rerun first

## Binding Constraints

- Steering is binding:
  - BRDT remains additive candidate work only; it cannot replace the approved CDI and CNS pillars.
  - Equal-footing and honest claim-boundary rules still apply; do not relax the BRDT contract to make the item easier.
- Roadmap and approved design are binding:
  - candidate inverse-wave work may proceed only as `candidate-*` backlog activity and must preserve additive claim boundaries until a later evidence-package amendment says otherwise.
  - the old BRDT `2026-05-05` and `2026-05-06` bundles remain valid Born-image-input lineage only; this item must not overwrite or relabel them.
  - this adapter-contract item is the prerequisite for `2026-05-07-brdt-sinogram-input-40ep-paper-evidence`.
  - Phase 5 `/home/ollie/Documents/neurips/` work remains paused.
- Repository guardrails are binding:
  - no worktrees
  - use PATH `python`
  - keep `ptycho/model.py`, `ptycho/diffsim.py`, and `ptycho/tf_helper.py` untouched
  - prefer narrow targeted tests before any smoke execution
- Consistency-pass rule:
  - treat admissibility as contract-based, not label-based. Historical roots stay discoverable as Born-image-input lineage, but the new sinogram-input contract must have its own summary, smoke artifacts, and discoverability surfaces.

## Prerequisite Status

- Selected-item frontmatter records no explicit backlog prerequisite, so implementation may start immediately.
- Progress-ledger context is favorable:
  - `phase-0-evidence-inventory` and `phase-1-pde-benchmark-selection` are already completed.
  - no active initiative-level blocked tranche forbids candidate BRDT adapter work.
- Read-only upstream authorities already exist and must be consumed rather than redefined:
  - operator authority:
    `docs/plans/NEURIPS-HYBRID-RESNET-2026/brdt_operator_validation_report.md`
  - smoke dataset authority:
    `docs/plans/NEURIPS-HYBRID-RESNET-2026/brdt_dataset_preflight.md`
  - historical adapter/preflight contract summary:
    `docs/plans/NEURIPS-HYBRID-RESNET-2026/brdt_task_adapters.md`
- Downstream dependency:
  - the successor `2026-05-07-brdt-sinogram-input-40ep-paper-evidence` item should not execute until this adapter-contract item lands and its smoke proof is available.

## Implementation Architecture

- **Contract and roster surface:** `run_config.py`, `data.py`, and the sinogram-input runner/dry-run path must agree on accepted input modes, visible row IDs, legacy alias rejection, and which rows belong to the new contract.
- **Adapter and input-routing surface:** `models.py` plus the train/eval/preflight input-preparation helpers must enforce the exact sinogram tensor contract and keep Born-inverse derivation out of the learned-model input path.
- **Execution and provenance surface:** the smoke runner must use a dedicated adapter-contract artifact root, persist per-row contract artifacts plus one top-level smoke summary, and avoid writing into the successor `40`-epoch evidence root.
- **Discoverability surface:** one new durable summary plus targeted index updates must make the contract change findable without rewriting historical Born-image-input summaries or paper-evidence indexes prematurely.

## Concrete File And Artifact Targets

### Mandatory code and test targets

- Modify:
  - `scripts/studies/born_rytov_dt/run_config.py`
  - `scripts/studies/born_rytov_dt/data.py`
  - `scripts/studies/born_rytov_dt/models.py`
  - `scripts/studies/born_rytov_dt/train.py`
  - `scripts/studies/born_rytov_dt/evaluate.py`
  - `scripts/studies/born_rytov_dt/run_preflight.py`
  - `scripts/studies/born_rytov_dt/run_sinogram_input_smoke.py`
- Modify if needed to keep the successor runner on the new roster without launching the full experiment here:
  - `scripts/studies/born_rytov_dt/run_sinogram_input_40ep.py`
- Test authority:
  - `tests/studies/test_born_rytov_dt_adapters.py`
  - `tests/studies/test_born_rytov_dt_preflight.py`

### Mandatory contract outputs

- Dedicated feasibility-only artifact root for this backlog item:
  - `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-07-brdt-sinogram-input-adapter-contract/`
- Required smoke outputs under that root:
  - `smoke_summary.json`
  - `smoke/ffno/adapter_contract.json`
  - `smoke/ffno/invocation.json`
  - `smoke/ffno/invocation.sh`
  - `smoke/sru_net/adapter_contract.json`
  - `smoke/sru_net/invocation.json`
  - `smoke/sru_net/invocation.sh`
- The top-level smoke summary must record:
  - `input_mode: "sinogram"`
  - the dataset manifest path consumed
  - the exact learned rows executed (`ffno`, `sru_net`)
  - per-row status
  - proof fields that the learned-model input source was the measured complex sinogram and that Born consistency still targeted the measured sinogram

### Mandatory durable documentation/index targets

- Create:
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/brdt_sinogram_input_adapter_contract.md`
- Update:
  - `docs/index.md`
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/evidence_matrix.md`
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/ablation_index.json`

### Preferred but non-blocking packaging

- If implementation needs an additional dry-run contract payload for local debugging, keep it under the adapter-contract artifact root or `tmp/`; do not write into the successor `2026-05-07-brdt-sinogram-input-40ep-paper-evidence` root.
- Do not update `paper_evidence_index.md`, `paper_evidence_manifest.json`, manuscript assets, or historical BRDT summary supersession notes yet; those belong to the successor `40`-epoch item after a successful sinogram-input evidence run.

## Tasks

### Task 1: Freeze The Input-Mode And Runner-Selection Contract

**Files:**
- Modify: `scripts/studies/born_rytov_dt/run_config.py`
- Modify: `scripts/studies/born_rytov_dt/data.py`
- Modify if needed: `scripts/studies/born_rytov_dt/run_sinogram_input_40ep.py`
- Test: `tests/studies/test_born_rytov_dt_adapters.py`
- Test: `tests/studies/test_born_rytov_dt_preflight.py`

- [ ] Keep `SUPPORTED_INPUT_MODES` limited to `("born_init_image", "sinogram")` and keep `direct_sinogram` rejected with an explicit migration message.
- [ ] Preserve the historical `default_row_roster()` as the Born-image-input roster and keep the new `sinogram_input_row_roster()` separate so historical four-row preflight lineage is not silently rewritten.
- [ ] Ensure the new sinogram-input paper runner or dry-run contract consumes only the sinogram roster (`classical_born_backprop`, `ffno`, `sru_net`) and does not select old Born-image-input learned rows.
- [ ] Keep `born_init_image` available for historical reproducibility instead of deleting it to force the new contract.

**Verification**

- [ ] **Blocking:** run the backlog-required collection gate before any smoke execution:

```bash
pytest --collect-only -q tests/studies/test_born_rytov_dt_adapters.py tests/studies/test_born_rytov_dt_preflight.py
```

- [ ] **Blocking:** keep or add tests proving:
  - `input_mode="sinogram"` is accepted,
  - `direct_sinogram` is rejected,
  - `run_sinogram_input_40ep(..., dry_run=True)` writes only sinogram rows.

### Task 2: Harden The Sinogram-Input Adapter Surface

**Files:**
- Modify: `scripts/studies/born_rytov_dt/data.py`
- Modify: `scripts/studies/born_rytov_dt/models.py`
- Test: `tests/studies/test_born_rytov_dt_adapters.py`

- [ ] Keep the learned-model input tensor contract explicit and stable:
  - dataset/batch layout may remain `(B, angles, detectors, 2)`,
  - model-facing layout must be `(B, 2, 64, 128)`.
- [ ] Build or harden task-local sinogram-input adapters for `ffno` and `hybrid_resnet`/`sru_net` that only perform tensor-layout adaptation, lift/projection, and object-grid resizing before the existing model body.
- [ ] Do not compute a fixed Born inverse anywhere inside the learned-model adapter or learned sinogram input path.
- [ ] Add or preserve negative tests that fail if the learned sinogram path silently routes through `derive_born_init_image(...)`.
- [ ] Keep the Born inverse available only for the classical reference path, optional visualization, and historical Born-image-input lineage.

**Verification**

- [ ] **Blocking:** run the backlog-required targeted adapter suite:

```bash
pytest -q tests/studies/test_born_rytov_dt_adapters.py -k "sinogram or input_mode or model"
```

- [ ] **Blocking:** the targeted adapter tests must prove:
  - `ffno`: `(B, 2, 64, 128) -> (B, 1, 128, 128)`
  - `sru_net` / Hybrid-family path: `(B, 2, 64, 128) -> (B, 1, 128, 128)`
  - wrong-shape image tensors are rejected for the sinogram adapter

### Task 3: Route The New Contract Through Train, Eval, Preflight, And Smoke

**Files:**
- Modify: `scripts/studies/born_rytov_dt/train.py`
- Modify: `scripts/studies/born_rytov_dt/evaluate.py`
- Modify: `scripts/studies/born_rytov_dt/run_preflight.py`
- Modify: `scripts/studies/born_rytov_dt/run_sinogram_input_smoke.py`
- Modify if needed: `scripts/studies/born_rytov_dt/run_sinogram_input_40ep.py`
- Test: `tests/studies/test_born_rytov_dt_preflight.py`

- [ ] Ensure the sinogram path uses the measured sinogram in two distinct roles:
  - direct learned-model input,
  - observed measurement target for Born-consistency loss.
- [ ] Keep the historical `born_init_image` branch intact and isolated so older preflight and rerun artifacts remain reproducible.
- [ ] Decouple the smoke runner from the successor `40`-epoch paper-evidence root. Its default output root must live under the adapter-contract backlog item, not under `2026-05-07-brdt-sinogram-input-40ep-paper-evidence/`.
- [ ] Make the smoke runner persist a top-level `smoke_summary.json` that records the contract proof fields listed above, plus pointers to the per-row artifacts.
- [ ] If the smoke command encounters a normal harness or path failure, fix and rerun it instead of marking the backlog item blocked. Reserve `BLOCKED` for missing resources, unavailable hardware, or an unrecoverable external dependency outside current authority.

**Verification**

- [ ] **Blocking:** run the backlog-required targeted preflight/runner suite:

```bash
pytest -q tests/studies/test_born_rytov_dt_preflight.py -k "sinogram_input_40ep or input_mode"
```

- [ ] **Blocking:** run the dedicated smoke command against an explicit adapter-contract root:

```bash
python -m scripts.studies.born_rytov_dt.run_sinogram_input_smoke \
  --device cpu \
  --output-root .artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-07-brdt-sinogram-input-adapter-contract/smoke
```

- [ ] **Blocking:** confirm the smoke output records `input_mode="sinogram"` for both learned rows and does not mix in any historical Born-image-input learned row.
- [ ] **Supporting:** if the smoke command stops being short, move it into `tmux` under `ptycho311`, track the exact launched PID, and still require fresh artifact writes before calling it complete.

### Task 4: Write The Durable Contract Summary And Discoverability Updates

**Files:**
- Create: `docs/plans/NEURIPS-HYBRID-RESNET-2026/brdt_sinogram_input_adapter_contract.md`
- Update: `docs/index.md`
- Update: `docs/plans/NEURIPS-HYBRID-RESNET-2026/evidence_matrix.md`
- Update: `docs/plans/NEURIPS-HYBRID-RESNET-2026/ablation_index.json`

- [ ] Write a new durable summary that states:
  - the learned-model input contract is now measured complex sinogram,
  - the Born inverse remains non-learned reference only,
  - the historical Born-image-input bundles remain valid lineage for that old contract,
  - this item is feasibility-only and hands off to `2026-05-07-brdt-sinogram-input-40ep-paper-evidence`.
- [ ] Update `docs/index.md` so future planning/execution can find the new summary without reading backlog prose.
- [ ] Update `evidence_matrix.md` to point at the new adapter-contract summary as the current BRDT input-contract authority and to keep the old Born-image-input summaries discoverable as historical lineage.
- [ ] Update `ablation_index.json` only as a contract/harness/readiness entry. If no benchmark-performance row was added, do not pretend otherwise.
- [ ] Do not yet retarget manuscript-facing BRDT assets or the paper-evidence index; that requires the successor `40`-epoch item to complete.

**Verification**

- [ ] **Blocking:** verify the new summary and matrix/index updates exist and parse cleanly:

```bash
python - <<'PY'
import json
from pathlib import Path

required = [
    Path("docs/plans/NEURIPS-HYBRID-RESNET-2026/brdt_sinogram_input_adapter_contract.md"),
    Path("docs/index.md"),
    Path("docs/plans/NEURIPS-HYBRID-RESNET-2026/evidence_matrix.md"),
    Path("docs/plans/NEURIPS-HYBRID-RESNET-2026/ablation_index.json"),
]
missing = [str(path) for path in required if not path.exists()]
if missing:
    raise SystemExit(f"missing required BRDT contract docs: {missing}")
json.loads(Path("docs/plans/NEURIPS-HYBRID-RESNET-2026/ablation_index.json").read_text())
print("BRDT sinogram contract docs present and ablation index parses")
PY
```

- [ ] **Supporting:** use `rg -n "born_init_image|sinogram"` across the touched BRDT docs to confirm the historical/new contract split is explicit rather than contradictory.

### Task 5: Run The Required Final Deterministic Gate

**Files/artifacts:**
- No new file targets beyond the outputs above.

- [ ] Run every backlog-required deterministic check exactly as listed below; do not weaken or skip any of them.
- [ ] If one of these checks fails because of a normal code or harness issue, fix and rerun before deciding the item is unrecoverable.
- [ ] In the final execution report, explicitly state that `model_variant_index.json` was intentionally left unchanged if this item produced only feasibility-only adapter/smoke evidence and no benchmark row.

**Verification**

- [ ] **Blocking:** required deterministic commands from the selected backlog item:

```bash
pytest --collect-only -q tests/studies/test_born_rytov_dt_adapters.py tests/studies/test_born_rytov_dt_preflight.py
pytest -q tests/studies/test_born_rytov_dt_adapters.py -k "sinogram or input_mode or model"
pytest -q tests/studies/test_born_rytov_dt_preflight.py -k "sinogram_input_40ep or input_mode"
python -m compileall -q scripts/studies/born_rytov_dt ptycho_torch
```

- [ ] **Blocking:** ensure the dedicated smoke command from Task 3 has succeeded and its artifact root is populated.

## Completion Criteria

- The BRDT codebase accepts `input_mode="sinogram"` without dropping historical `born_init_image` support.
- SRU-Net and FFNO both accept measured complex sinograms and emit `(B, 1, 128, 128)` predictions.
- The learned-model input path does not compute a fixed Born inverse.
- The measured sinogram is proven to serve both as learned-model input and as the Born-consistency target.
- The dedicated adapter-contract smoke artifacts exist under the correct backlog-item root.
- The new durable summary and discoverability updates make the contract change findable while preserving historical Born-image-input lineage honestly.
