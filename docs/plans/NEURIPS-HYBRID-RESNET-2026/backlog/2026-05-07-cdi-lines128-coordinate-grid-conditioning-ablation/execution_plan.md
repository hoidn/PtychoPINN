# Lines128 CDI Coordinate-Grid Conditioning Ablation Execution Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** Add two append-only `lines128` CDI ablation rows that concatenate deterministic unit-interval spatial coordinate channels onto the learned input for the Hybrid ResNet/SRU-Net CDI row and the corrected pure-FFNO CDI row, then publish lineage-safe comparisons against the existing unconditioned authorities.

**Architecture:** Keep the authoritative `lines128` contract frozen and route coordinate conditioning through explicit compare-wrapper -> torch-runner -> learned-input configuration. Default rows must remain unchanged; only the two new model IDs may opt into the extra channels. Long-running row execution stays under implementation ownership until the exact launched process exits `0` and the required row-local artifacts are freshly written, or a documented narrow fix attempt proves an unrecoverable blocker outside current authority.

**Tech Stack:** Python via PATH `python`, PyTorch/Lightning, `scripts/studies/grid_lines_compare_wrapper.py`, `scripts/studies/grid_lines_torch_runner.py`, existing Hybrid ResNet and FFNO generators, Markdown/JSON evidence indexes.

---

## Objective

- Add a controlled CDI `lines128` input-conditioning ablation that appends two deterministic coordinate channels to the learned input tensor:
  - `y` coordinate on `[0, 1]`
  - `x` coordinate on `[0, 1]`
- Execute only:
  - `pinn_hybrid_resnet_grid_channels`
  - `pinn_ffno_grid_channels`
- Compare those fresh rows only against same-contract lineage authorities:
  - unconditioned `pinn_hybrid_resnet` from `docs/plans/NEURIPS-HYBRID-RESNET-2026/lines128_paper_benchmark_summary.md`
  - corrected pure `pinn_ffno` from `docs/plans/NEURIPS-HYBRID-RESNET-2026/cdi_lines128_ffno_no_refiner_row_rerun_summary.md`

## Scope

- Preserve the frozen `lines128` CDI contract:
  - `N=128`
  - `gridsize=1`
  - synthetic grid-lines with `set_phi=True`
  - Run1084 fixed probe with `probe_scale_mode=pad_extrapolate` and `probe_smoothing_sigma=0.5`
  - fixed train/test split, seed, epoch budget, scheduler, loss, output mode, metric schema, and fixed visual sample policy
- Reuse the same coordinate convention as the PDEBench authored-FFNO adapter:
  - `torch.linspace(0.0, 1.0, steps=height)` for `y`
  - `torch.linspace(0.0, 1.0, steps=width)` for `x`
  - `torch.meshgrid(y, x, indexing="ij")`
  - channel order `[y, x]`
- Record the conditioning contract explicitly in invocation/config/history artifacts, including:
  - conditioning mode
  - `base_input_channels`
  - `learned_input_channels`
  - coordinate channel order
  - coordinate value range
  - meshgrid indexing
  - spatial shape used to build the grid
- Keep the observed diffraction input, probe preprocessing, physics target, baseline rows, and visual scaling unchanged.

## Explicit Non-Goals

- Do not overwrite or relabel the authoritative six-row `lines128` CDI bundle.
- Do not promote coordinate conditioning into a new default CDI path or unrelated model family.
- Do not combine coordinate channels with probe-channel conditioning in the same row.
- Do not rerun unconditioned `pinn_hybrid_resnet` or corrected pure `pinn_ffno` unless a narrow lineage audit proves an authoritative baseline is unusable after recoverable repair attempts.
- Do not expand into later roadmap phases, PDEBench work, additional CDI ablations, or `/home/ollie/Documents/neurips/` paper-artifact assembly.
- Do not modify `ptycho/model.py`, `ptycho/diffsim.py`, or `ptycho/tf_helper.py`.

## Binding Constraints And Prerequisite Status

- Steering boundary: this item is allowed only as a Phase 3 CDI evidence-strengthening ablation. It must preserve equal-footing comparison rules and must not relax fairness to make the ablation look better.
- Roadmap boundary: this is one of the explicitly allowed focused CDI input-conditioning ablations under Phase 3. It is append-only evidence and cannot replace the complete `lines128` table authority.
- Design boundary: the CDI headline remains `128x128`, and compact ablations must not silently mutate the benchmark contract or widen the claim beyond the two fresh rows.
- Progress-ledger status: `phase-0-evidence-inventory` and `phase-1-pde-benchmark-selection` are completed, and the ledger records no initiative-level blocker that prevents this CDI backlog item from proceeding.
- Required lineage prerequisites already exist and must be consumed rather than recreated:
  - `docs/backlog/done/2026-04-29-cdi-lines128-paper-benchmark-execution.md`
  - `docs/backlog/done/2026-05-06-cdi-lines128-ffno-no-refiner-row-rerun.md`
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/lines128_paper_benchmark_summary.md`
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/cdi_lines128_ffno_no_refiner_row_rerun_summary.md`
- Implementation precedent: probe-channel conditioning is already implemented as an append-only learned-input path in the same wrapper/runner stack. Coordinate conditioning should extend that same surface rather than inventing a second incompatible conditioning mechanism.
- Interpreter and workflow policy:
  - invoke Python as `python`
  - use tmux for long-running launches
  - do not create worktrees
  - do not mark the item `BLOCKED` for ordinary import, path, environment, or test-harness failures; diagnose, fix narrowly, and rerun first
  - reserve `BLOCKED` for missing resources, unavailable hardware, roadmap conflict, external dependency outside current authority, user decision required, or a failure that remains unrecoverable after a documented narrow fix attempt

## Implementation Architecture

### 1. Row Contract Layer

- Extend the compare-wrapper row specification and labels so the two new model IDs are explicit append-only variants with clear lineage anchors and no overlap with probe-conditioned rows.
- Serialize enough metadata to prove the conditioned rows used unit-interval `y/x` grids and nothing else changed in the row contract except the allowed input-conditioning fields.

### 2. Runner Conditioning Layer

- Extend the torch-runner learned-input conditioning surface with a coordinate-grid mode that appends broadcast `y` and `x` channels to the diffraction input.
- Keep default unconditioned behavior unchanged.
- Reuse the existing `learned_input_channels` plumbing rather than modifying model bodies unless runtime proof shows the current generator constructors cannot accept the widened input channel count.

### 3. Evidence And Packaging Layer

- Reuse the existing compare-wrapper bundle layout for row-local artifacts, verification logs, reconstructions, visuals, and summary packaging.
- Publish a durable summary plus evidence-index updates that append the new variants by lineage without mutating the baseline authorities.

## File And Artifact Targets

### Mandatory Code And Test Surfaces Likely To Change

- `scripts/studies/grid_lines_compare_wrapper.py`
- `scripts/studies/grid_lines_torch_runner.py`
- `tests/test_grid_lines_compare_wrapper.py`
- `tests/torch/test_grid_lines_torch_runner.py`

### Conditional Code Surfaces

- `ptycho_torch/generators/hybrid_resnet.py`
- `ptycho_torch/generators/ffno.py`

Only touch these if the existing `learned_input_channels` path fails to instantiate or execute the widened input cleanly for the conditioned rows. A generator edit is not expected and should be justified in the implementation report if it becomes necessary.

### Mandatory Contract Outputs

- Item root:
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-07-cdi-lines128-coordinate-grid-conditioning-ablation/`
- Fresh row-local artifacts for:
  - `pinn_hybrid_resnet_grid_channels`
  - `pinn_ffno_grid_channels`
- Durable summary:
  `docs/plans/NEURIPS-HYBRID-RESNET-2026/cdi_lines128_coordinate_grid_conditioning_ablation_summary.md`
- Required index updates:
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/evidence_matrix.md`
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/model_variant_index.json`
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/ablation_index.json`
  - `docs/studies/index.md`

### Preferred Packaging

- `verification/` logs for each deterministic check
- an item-local contract note or JSON diff that proves the allowed deltas versus the unconditioned lineage rows
- a preflight payload or row-plan JSON proving only the two new rows were selected

`docs/plans/NEURIPS-HYBRID-RESNET-2026/paper_evidence_index.md` is not a required output for this item unless implementation explicitly creates a new paper-facing bundle, which this plan does not authorize.

## Execution Checklist

### Tranche 1: Lock Lineage Anchors And Allowed Contract Drift

- [ ] Confirm the authoritative baseline roots and metric files for unconditioned `pinn_hybrid_resnet` and corrected pure `pinn_ffno`.
- [ ] Capture the immutable `lines128` fields that must stay fixed across the ablation.
- [ ] Define the allowed row-level drift as coordinate-grid conditioning only, with no probe-conditioning, architecture, scheduler, or dataset changes.
- [ ] Lock the coordinate convention for this item as unit-interval `[y, x]` channels with `ij` indexing and record it in an item-local contract note.
- [ ] Confirm the two new row IDs and labels are append-only and do not collide with existing row IDs.

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
    raise SystemExit(f"missing coordinate-grid conditioning inputs: {missing}")
print("coordinate-grid conditioning inputs present")
PY
```

Supporting verification:

- [ ] Save a short contract note or JSON diff under the item root so later tranches can prove what was held fixed.

### Tranche 2: Add Explicit Coordinate-Grid Conditioning With Default-Row Parity

- [ ] Extend `TorchRunnerConfig.input_conditioning_mode` handling with an explicit coordinate-grid mode for this ablation.
- [ ] Implement a deterministic grid builder that creates `y/x` unit-interval channels from the learned-input spatial shape using `ij` indexing.
- [ ] Concatenate those channels onto the learned input tensor only for the conditioned rows and keep the observed diffraction tensor unchanged for losses and metrics.
- [ ] Serialize a conditioning contract that records:
  - `mode`
  - `enabled`
  - `base_input_channels`
  - `learned_input_channels`
  - `coordinate_channel_count`
  - `coordinate_channels`
  - `coordinate_value_range`
  - `coordinate_meshgrid_indexing`
  - `coordinate_spatial_shape`
- [ ] Keep unconditioned rows byte-for-byte unchanged except for any additive metadata that is harmless and intentionally documented.
- [ ] Only if required, make the smallest generator-side change needed to accept the widened input channel count while preserving the ordinary `1`-channel path.

Blocking verification:

- [ ] Run the required torch-runner selector before any expensive row launch:

```bash
pytest -q tests/torch/test_grid_lines_torch_runner.py -k "hybrid_resnet or ffno or coordinate or grid"
```

Supporting verification:

- [ ] Add and run narrower node IDs during iteration to prove:
  - the coordinate-grid mode appends exactly two channels in `[y, x]` order
  - the coordinate channels are normalized to `[0, 1]`
  - the existing probe-conditioned mode still behaves as before
  - unconditioned rows keep prior channel counts and model IDs
  - learned-input contracts are written into invocation/config artifacts

### Tranche 3: Route The Two New Rows Through The Compare Wrapper

- [ ] Add explicit row specs, labels, and overrides for:
  - `pinn_hybrid_resnet_grid_channels`
  - `pinn_ffno_grid_channels`
- [ ] Keep the FFNO conditioned row aligned with the corrected pure-FFNO lineage by preserving `fno_cnn_blocks=0`.
- [ ] Ensure the two new rows use the coordinate-grid conditioning mode and do not inherit probe-channel conditioning flags.
- [ ] Use compare-wrapper preflight-only execution to confirm that only the two new rows are selected and that their payloads advertise the intended conditioning contract.
- [ ] Persist an item-local audit showing that the drift versus the authoritative baseline rows is limited to the allowed conditioning metadata.

Blocking verification:

- [ ] Run the required compare-wrapper selector before launching training:

```bash
pytest -q tests/test_grid_lines_compare_wrapper.py -k "hybrid_resnet or ffno or coordinate or grid"
```

Supporting verification:

- [ ] Save the wrapper preflight payload or equivalent row-plan JSON under the item root.

### Tranche 4: Launch The Two Fresh Rows And Wait For Terminal Success

- [ ] Launch exactly one item-local run bundle rooted under:
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-07-cdi-lines128-coordinate-grid-conditioning-ablation/`
- [ ] Launch only:
  - `pinn_hybrid_resnet_grid_channels`
  - `pinn_ffno_grid_channels`
- [ ] Use tmux for the long-running command path, activate `ptycho311`, track the exact launched PID, and wait on that PID until completion.
- [ ] Do not start a duplicate run if another process is already writing to the same output root.
- [ ] Consider the run complete only when the tracked process exits `0` and the required row-local artifacts are freshly written.
- [ ] If normal failures occur, diagnose, fix narrowly, and rerun in the same scope before escalation.

Blocking verification:

- [ ] Confirm each fresh row produces row-local artifacts consistent with existing `lines128` compare-wrapper bundles, including invocation/config/history/metrics/model outputs plus reconstruction and visual artifacts.
- [ ] Run the required compile gate after code changes and before closing the item:

```bash
python -m compileall -q ptycho_torch scripts/studies
```

Supporting verification:

- [ ] Save launch logs, tmux logs, and any completion-proof repair artifacts under `verification/` if the shared wrapper path still requires them.

### Tranche 5: Summarize Deltas And Update Durable Evidence Indexes

- [ ] Compute metric deltas from:
  - baseline unconditioned `pinn_hybrid_resnet`
  - corrected pure unconditioned `pinn_ffno`
- [ ] Write `docs/plans/NEURIPS-HYBRID-RESNET-2026/cdi_lines128_coordinate_grid_conditioning_ablation_summary.md` with:
  - exact row roots
  - coordinate-conditioning proof
  - channel ordering and normalization statement
  - contract-preservation statement
  - metric deltas versus both unconditioned baselines
  - an explicit conclusion stating whether coordinate channels helped Hybrid ResNet/SRU-Net, FFNO, both, or neither, and whether the effect is strong enough to justify a later table-refresh item
- [ ] Update `evidence_matrix.md`, `model_variant_index.json`, `ablation_index.json`, and `docs/studies/index.md`.
- [ ] Keep the checked-in summary and all required discoverability surfaces internally consistent on:
  - backlog item id `2026-05-07-cdi-lines128-coordinate-grid-conditioning-ablation`
  - summary authority path `docs/plans/NEURIPS-HYBRID-RESNET-2026/cdi_lines128_coordinate_grid_conditioning_ablation_summary.md`
  - item root `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-07-cdi-lines128-coordinate-grid-conditioning-ablation/`
  - fresh row ids `pinn_hybrid_resnet_grid_channels` and `pinn_ffno_grid_channels`
  - append-only claim boundary `cdi_lines128_coordinate_grid_conditioning_ablation_only`
- [ ] Update `docs/index.md` only if implementation adds a reusable conditioning surface or durable study entry that should be discoverable outside the usual NeurIPS indexes; otherwise state explicitly in the execution report that no index-hub change was needed.

Blocking verification:

- [ ] Rerun the full backlog deterministic gate before handoff:

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
    raise SystemExit(f"missing coordinate-grid conditioning inputs: {missing}")
print("coordinate-grid conditioning inputs present")
PY
```

- [ ] Run the backlog-required pytest selectors exactly:

```bash
pytest -q tests/torch/test_grid_lines_torch_runner.py -k "hybrid_resnet or ffno or coordinate or grid"
pytest -q tests/test_grid_lines_compare_wrapper.py -k "hybrid_resnet or ffno or coordinate or grid"
```

- [ ] Run the backlog-required compile gate exactly:

```bash
python -m compileall -q ptycho_torch scripts/studies
```

- [ ] Run this closeout artifact-validation gate after the summary and index updates are written:

```bash
python - <<'PY'
import json
from pathlib import Path

item_id = "2026-05-07-cdi-lines128-coordinate-grid-conditioning-ablation"
summary_path = Path(
    "docs/plans/NEURIPS-HYBRID-RESNET-2026/"
    "cdi_lines128_coordinate_grid_conditioning_ablation_summary.md"
)
evidence_matrix_path = Path(
    "docs/plans/NEURIPS-HYBRID-RESNET-2026/evidence_matrix.md"
)
model_variant_index_path = Path(
    "docs/plans/NEURIPS-HYBRID-RESNET-2026/model_variant_index.json"
)
ablation_index_path = Path(
    "docs/plans/NEURIPS-HYBRID-RESNET-2026/ablation_index.json"
)
studies_index_path = Path("docs/studies/index.md")
item_root = (
    ".artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/"
    "2026-05-07-cdi-lines128-coordinate-grid-conditioning-ablation/"
)
summary_rel = (
    "docs/plans/NEURIPS-HYBRID-RESNET-2026/"
    "cdi_lines128_coordinate_grid_conditioning_ablation_summary.md"
)
claim_boundary = "cdi_lines128_coordinate_grid_conditioning_ablation_only"
row_ids = {
    "pinn_hybrid_resnet_grid_channels",
    "pinn_ffno_grid_channels",
}

required_paths = [
    summary_path,
    evidence_matrix_path,
    model_variant_index_path,
    ablation_index_path,
    studies_index_path,
]
missing = [str(path) for path in required_paths if not path.exists()]
if missing:
    raise SystemExit(f"missing durable outputs: {missing}")

summary_text = summary_path.read_text(encoding="utf-8")
for needle in [item_id, item_root, claim_boundary, *sorted(row_ids)]:
    if needle not in summary_text:
        raise SystemExit(f"summary missing required text: {needle}")

evidence_matrix_text = evidence_matrix_path.read_text(encoding="utf-8")
for needle in [summary_path.name, claim_boundary, *sorted(row_ids)]:
    if needle not in evidence_matrix_text:
        raise SystemExit(f"evidence_matrix missing required text: {needle}")

studies_index_text = studies_index_path.read_text(encoding="utf-8")
for needle in [item_id, summary_rel, item_root, *sorted(row_ids)]:
    if needle not in studies_index_text:
        raise SystemExit(f"studies index missing required text: {needle}")

model_variant_index = json.loads(model_variant_index_path.read_text(encoding="utf-8"))
variant_rows = [
    row for row in model_variant_index["model_variants"]
    if row.get("row_id") in row_ids
]
if {row.get("row_id") for row in variant_rows} != row_ids:
    raise SystemExit("model_variant_index missing one or both coordinate-grid rows")
for row in variant_rows:
    if row.get("source_backlog_item") != item_id:
        raise SystemExit(
            f"model_variant_index row {row.get('row_id')} has wrong source_backlog_item"
        )
    if row.get("source_summary") != summary_rel:
        raise SystemExit(
            f"model_variant_index row {row.get('row_id')} has wrong source_summary"
        )
    artifact_root = row.get("artifact_root", "")
    if item_root.rstrip("/") not in artifact_root:
        raise SystemExit(
            f"model_variant_index row {row.get('row_id')} has wrong artifact_root"
        )

ablation_index = json.loads(ablation_index_path.read_text(encoding="utf-8"))
matching_families = [
    family for family in ablation_index["ablation_families"]
    if item_id in family.get("completed_items", [])
]
if not matching_families:
    raise SystemExit("ablation_index missing coordinate-grid ablation family entry")

summary_ok = False
root_ok = False
row_ok = False
for family in matching_families:
    if summary_rel in family.get("summary_authorities", []):
        summary_ok = True
    if item_root in family.get("artifact_roots", []):
        root_ok = True
    family_rows = {
        row.get("row_id")
        for row in family.get("rows", [])
        if isinstance(row, dict)
    }
    if row_ids.issubset(family_rows):
        row_ok = True
if not summary_ok:
    raise SystemExit("ablation_index family missing summary authority")
if not root_ok:
    raise SystemExit("ablation_index family missing artifact root")
if not row_ok:
    raise SystemExit("ablation_index family missing one or both coordinate-grid rows")

print("coordinate-grid conditioning durable outputs validated")
PY
```

Supporting verification:

- [ ] Archive the verification logs under the item root and reference them from the durable summary.

## Completion Criteria

- Both fresh row configs prove that coordinate channels were appended to the learned input with recorded `[y, x]` ordering and `[0, 1]` normalization.
- The summary compares grid-conditioned Hybrid ResNet/SRU-Net and corrected pure-FFNO rows against the corresponding unconditioned same-contract lineage anchors.
- The summary states clearly whether coordinate-grid conditioning helped Hybrid ResNet/SRU-Net, FFNO, both, or neither, and whether any improvement is large enough to justify a later table-refresh item.
- The checked-in summary, `evidence_matrix.md`, `model_variant_index.json`, `ablation_index.json`, and `docs/studies/index.md` all agree on the new row ids, summary authority, claim boundary, and item root so later manuscript or planning work can discover the new rows without rereading raw backlog items.
