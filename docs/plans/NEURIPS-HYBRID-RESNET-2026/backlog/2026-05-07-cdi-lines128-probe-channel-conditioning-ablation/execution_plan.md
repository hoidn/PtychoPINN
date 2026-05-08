# Lines128 CDI Probe-Channel Conditioning Ablation Execution Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** Add two append-only `lines128` CDI ablation rows that concatenate the fixed Run1084 probe real/imag channels onto the learned-model input for `pinn_hybrid_resnet` and corrected pure `pinn_ffno`, then publish lineage-safe comparisons against the existing unconditioned rows.

**Architecture:** Keep the authoritative `lines128` contract frozen and route probe conditioning through explicit compare-wrapper -> torch-runner -> model-input configuration. Default rows must remain unchanged; only the two new probe-conditioned model IDs may opt into the extra channels. Long-running row execution stays under implementation ownership until the exact launched process exits `0` and the required row-local artifacts are freshly written.

**Tech Stack:** Python via PATH `python`, PyTorch/Lightning, `scripts/studies/grid_lines_compare_wrapper.py`, `scripts/studies/grid_lines_torch_runner.py`, existing Hybrid ResNet and FFNO generators, Markdown/JSON evidence indexes.

---

## Objective

- Add a controlled input-conditioning ablation for the locked `lines128` CDI benchmark by appending two probe channels:
  - probe real part
  - probe imaginary part
- Execute only:
  - `pinn_hybrid_resnet_probe_channels`
  - `pinn_ffno_probe_channels`
- Compare those fresh rows only against same-contract lineage authorities:
  - unconditioned `pinn_hybrid_resnet` from `docs/plans/NEURIPS-HYBRID-RESNET-2026/lines128_paper_benchmark_summary.md`
  - corrected pure `pinn_ffno` from `docs/plans/NEURIPS-HYBRID-RESNET-2026/cdi_lines128_ffno_no_refiner_row_rerun_summary.md`

## Scope

- Preserve the frozen `lines128` CDI contract:
  - `N=128`
  - `gridsize=1`
  - synthetic grid-lines with `set_phi=True`
  - Run1084 fixed probe with `probe_scale_mode=pad_extrapolate` and `probe_smoothing_sigma=0.5`
  - fixed split, seed, epoch budget, scheduler, loss, output mode, metric schema, and fixed visual sample policy
- Concatenate the same preprocessed probe used by the forward model into the learned input tensor for the two new rows only.
- Record the conditioning mode and probe lineage explicitly in invocation/config artifacts.
- Keep the physics loss probe, dataset, target reconstruction, visual scaling, and baseline rows unchanged.

## Explicit Non-Goals

- Do not change the locked six-row `lines128` authority or overwrite existing baseline rows.
- Do not promote probe conditioning to a new default model family or paper headline.
- Do not rerun unconditioned `pinn_hybrid_resnet` or `pinn_ffno` unless a narrow lineage audit proves an authoritative baseline is unusable after repair attempts.
- Do not expand into other CDI ablations, PDE work, or `/home/ollie/Documents/neurips/` paper-artifact assembly.
- Do not modify `ptycho/model.py`, `ptycho/diffsim.py`, or `ptycho/tf_helper.py`.

## Binding Constraints And Prerequisite Status

- Steering boundary: this work should strengthen core comparison evidence without relaxing equal-footing rules.
- Roadmap boundary: this is a Phase 3 CDI ablation only, specifically the allowed “fixed-probe channels” conditioning study under the existing `lines128` contract.
- Fairness rule: any conclusion must remain an input-conditioning ablation conclusion, not an architecture-family replacement claim.
- Progress-ledger status: `phase-0-evidence-inventory` and `phase-1-pde-benchmark-selection` are completed, and the ledger records no initiative-level blocker that prevents this CDI backlog item.
- Additional required prerequisites are already completed outside the ledger tranche list and must be consumed rather than recreated:
  - `docs/backlog/done/2026-04-29-cdi-lines128-paper-benchmark-execution.md`
  - `docs/backlog/done/2026-05-06-cdi-lines128-ffno-no-refiner-row-rerun.md`
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/lines128_paper_benchmark_summary.md`
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/cdi_lines128_ffno_no_refiner_row_rerun_summary.md`
- Interpreter and workflow policy:
  - invoke Python as `python`
  - use tmux for long-running launches
  - do not create worktrees
  - do not mark the item `BLOCKED` for ordinary import, path, environment, or test-harness failures; diagnose, fix narrowly, and rerun first
  - reserve `BLOCKED` for missing resources, unavailable hardware, roadmap conflict, external dependency outside current authority, user decision required, or a failure that remains unrecoverable after a documented narrow fix attempt

## Implementation Architecture

### 1. Row Contract Layer

- Extend the compare-wrapper row specification and metadata surfaces so the two new model IDs are explicit append-only variants with clear labels and lineage references.
- Serialize conditioning metadata in row payloads and invocation artifacts so later audits can prove the probe channels were enabled intentionally.

### 2. Runner And Model-Input Layer

- Add explicit torch-runner configuration that converts the already-preprocessed complex probe into two broadcast input channels and concatenates them to the learned-model input tensor for opt-in rows only.
- Keep default unconditioned behavior byte-for-byte compatible unless a targeted test proves a documented, harmless metadata addition.
- Prefer solving channel-count plumbing in the runner/factory path first; touch generator modules only if the existing constructor path cannot accept the conditioned input channel count cleanly.

### 3. Evidence And Packaging Layer

- Reuse the existing compare-wrapper bundle layout for row-local artifacts, reconstructions, visuals, and verification logs.
- Publish a durable summary plus index updates that append the new variants by lineage without mutating the baseline authority.

## File And Artifact Targets

### Mandatory Code And Test Surfaces Likely To Change

- `scripts/studies/grid_lines_compare_wrapper.py`
- `scripts/studies/grid_lines_torch_runner.py`
- `tests/test_grid_lines_compare_wrapper.py`
- `tests/torch/test_grid_lines_torch_runner.py`

### Conditional Code Surfaces

- `ptycho_torch/generators/hybrid_resnet.py`
- `ptycho_torch/generators/ffno.py`

Only modify these if the existing runner/factory path cannot pass a non-default input-channel count while keeping ordinary rows unchanged.

### Mandatory Contract Outputs

- Item root:
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-07-cdi-lines128-probe-channel-conditioning-ablation/`
- Fresh row-local artifacts for:
  - `pinn_hybrid_resnet_probe_channels`
  - `pinn_ffno_probe_channels`
- Durable summary:
  `docs/plans/NEURIPS-HYBRID-RESNET-2026/cdi_lines128_probe_channel_conditioning_ablation_summary.md`
- Required index updates:
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/evidence_matrix.md`
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/model_variant_index.json`
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/ablation_index.json`
  - `docs/studies/index.md`

### Preferred Packaging

- `verification/` logs for each deterministic check
- machine-readable contract/lineage diff summarizing allowed deltas versus the baseline rows
- an item-local comparison manifest or JSON bundle if the existing wrapper flow already emits one without widening scope

`docs/plans/NEURIPS-HYBRID-RESNET-2026/paper_evidence_index.md` is not a required output for this item unless implementation explicitly produces a new paper-facing bundle, which this plan does not authorize.

## Execution Checklist

### Tranche 1: Lock Baseline Lineage And Allowed Deltas

- [ ] Confirm the authoritative baseline roots and metric files for unconditioned `pinn_hybrid_resnet` and corrected pure `pinn_ffno`.
- [ ] Capture the immutable `lines128` contract fields that must stay fixed across the ablation.
- [ ] Define the only allowed row-level change as explicit probe-channel conditioning plus any metadata fields required to prove it.
- [ ] Confirm the two new row IDs and human-readable labels are append-only and do not collide with existing model IDs.

Blocking verification:

- [ ] Run the backlog prerequisite-presence check before coding or launching rows:

```bash
python - <<'PY'
from pathlib import Path
required = [
    Path("docs/backlog/done/2026-04-29-cdi-lines128-paper-benchmark-execution.md"),
    Path("docs/backlog/done/2026-05-06-cdi-lines128-ffno-no-refiner-row-rerun.md"),
    Path("docs/plans/NEURIPS-HYBRID-RESNET-2026/lines128_paper_benchmark_summary.md"),
    Path("docs/plans/NEURIPS-HYBRID-RESNET-2026/cdi_lines128_ffno_no_refiner_row_rerun_summary.md"),
    Path("scripts/studies/grid_lines_compare_wrapper.py"),
    Path("scripts/studies/grid_lines_torch_runner.py"),
    Path("ptycho_torch/generators/hybrid_resnet.py"),
    Path("ptycho_torch/generators/ffno.py"),
]
missing = [str(path) for path in required if not path.exists()]
if missing:
    raise SystemExit(f"missing probe-channel conditioning inputs: {missing}")
print("probe-channel conditioning inputs present")
PY
```

Supporting verification:

- [ ] Save a short contract note or JSON diff under the item root so later tranches can prove what was held fixed.

### Tranche 2: Implement Explicit Probe-Channel Plumbing With Default-Row Parity

- [ ] Add compare-wrapper row specs and payload fields for `pinn_hybrid_resnet_probe_channels` and `pinn_ffno_probe_channels`.
- [ ] Add torch-runner configuration that opt-in rows can use to request probe-channel conditioning explicitly.
- [ ] Build the conditioned learned input tensor by concatenating the already-preprocessed probe real/imag channels per sample.
- [ ] Ensure invocation/config/history artifacts record the conditioning mode and probe lineage clearly.
- [ ] Keep unconditioned rows unchanged by default.
- [ ] If required, make the smallest generator/factory change needed to accept the conditioned input channel count while preserving the default `1`-channel path.

Blocking verification:

- [ ] Run the required runner-focused selector before any expensive row launch:

```bash
pytest -q tests/torch/test_grid_lines_torch_runner.py -k "hybrid_resnet or ffno or probe"
```

Supporting verification:

- [ ] Add and run narrower node IDs during iteration to prove:
  - conditioned rows request extra channels
  - unconditioned rows keep prior channel counts and model IDs
  - invocation/config payloads record the conditioning mode and probe preprocessing lineage

### Tranche 3: Preflight Row Routing, Contract Preservation, And Output Layout

- [ ] Use wrapper preflight-only execution to confirm only the two new rows are selected for launch.
- [ ] Confirm preflight payloads resolve the same fixed-probe dataset/preprocessing lineage as the baseline rows.
- [ ] Confirm the wrapper still compares by lineage against the authoritative unconditioned Hybrid ResNet and corrected pure FFNO roots rather than silently relaunching them.
- [ ] Persist an item-local audit showing that the contract drift versus the authoritative rows is limited to the allowed conditioning fields.

Blocking verification:

- [ ] Run the required compare-wrapper selector before launching training:

```bash
pytest -q tests/test_grid_lines_compare_wrapper.py -k "hybrid_resnet or ffno or probe"
```

Supporting verification:

- [ ] Save the wrapper preflight payload or equivalent JSON audit under the item root.

### Tranche 4: Launch The Two Fresh Rows And Wait For Terminal Success

- [ ] Launch exactly one item-local run bundle rooted under:
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-07-cdi-lines128-probe-channel-conditioning-ablation/`
- [ ] Launch only:
  - `pinn_hybrid_resnet_probe_channels`
  - `pinn_ffno_probe_channels`
- [ ] Use tmux for the long-running command path, activate `ptycho311`, track the exact launched PID, and wait on that PID until completion.
- [ ] Do not start a duplicate run if another process is already writing to the same output root.
- [ ] Consider a row complete only when the tracked process exits `0` and the required row-local artifacts are freshly written.
- [ ] If a normal failure occurs, diagnose, fix narrowly, and rerun in the same scope before considering escalation.

Blocking verification:

- [ ] Confirm each fresh row produces row-local artifacts consistent with existing `lines128` compare-wrapper bundles, including invocation/config/history/metrics/model outputs plus reconstruction and visual artifacts.
- [ ] Run the required compile gate after code changes and before closing the item:

```bash
python -m compileall -q ptycho_torch scripts/studies
```

Supporting verification:

- [ ] Save launch logs, tmux logs, and any completion-proof repair artifacts under `verification/` if the shared wrapper path still requires them.

### Tranche 5: Summarize Metric Deltas And Update Durable Evidence Indexes

- [ ] Compute metric deltas from:
  - baseline unconditioned `pinn_hybrid_resnet`
  - corrected pure unconditioned `pinn_ffno`
- [ ] Write `docs/plans/NEURIPS-HYBRID-RESNET-2026/cdi_lines128_probe_channel_conditioning_ablation_summary.md` with:
  - exact row roots
  - conditioning proof
  - contract-preservation statement
  - metric deltas versus both unconditioned baselines
  - claim boundary language that this is append-only input-conditioning evidence
- [ ] Update `evidence_matrix.md`, `model_variant_index.json`, `ablation_index.json`, and `docs/studies/index.md`.
- [ ] If the implementation discovers durable project knowledge beyond item-local reporting, update the most relevant canonical doc in the same pass rather than leaving the knowledge implicit in code or artifacts alone.

Blocking verification:

- [ ] Rerun the full backlog deterministic gate before handing off:

```bash
python - <<'PY'
from pathlib import Path
required = [
    Path("docs/backlog/done/2026-04-29-cdi-lines128-paper-benchmark-execution.md"),
    Path("docs/backlog/done/2026-05-06-cdi-lines128-ffno-no-refiner-row-rerun.md"),
    Path("docs/plans/NEURIPS-HYBRID-RESNET-2026/lines128_paper_benchmark_summary.md"),
    Path("docs/plans/NEURIPS-HYBRID-RESNET-2026/cdi_lines128_ffno_no_refiner_row_rerun_summary.md"),
    Path("scripts/studies/grid_lines_compare_wrapper.py"),
    Path("scripts/studies/grid_lines_torch_runner.py"),
    Path("ptycho_torch/generators/hybrid_resnet.py"),
    Path("ptycho_torch/generators/ffno.py"),
]
missing = [str(path) for path in required if not path.exists()]
if missing:
    raise SystemExit(f"missing probe-channel conditioning inputs: {missing}")
print("probe-channel conditioning inputs present")
PY

pytest -q tests/torch/test_grid_lines_torch_runner.py -k "hybrid_resnet or ffno or probe"
pytest -q tests/test_grid_lines_compare_wrapper.py -k "hybrid_resnet or ffno or probe"
python -m compileall -q ptycho_torch scripts/studies
```

Supporting verification:

- [ ] Add a narrow summary-audit script or note proving the updated indexes point to the new summary and row roots.

## Completion Gate

- The two fresh row configs prove probe-channel concatenation explicitly and show the same Run1084 probe preprocessing lineage as the forward-model probe.
- Only the two new probe-conditioned rows were launched fresh.
- The summary reports metric deltas against the authoritative unconditioned Hybrid ResNet and corrected pure FFNO rows under the same `lines128` contract.
- The resulting evidence is framed strictly as append-only input-conditioning ablation evidence.
- Required evidence indexes are updated so later manuscript or planning work can discover the new rows without rereading raw run directories.
