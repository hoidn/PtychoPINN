# Lines128 NeuralOperator U-NO Table Extension Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:executing-plans` or `superpowers:subagent-driven-development` to implement this plan task-by-task. Keep this file as the execution authority for the selected backlog item.

**Goal:** Append fresh `pinn_neuralop_uno` and `supervised_neuralop_uno` rows to the locked `lines128` CDI paper table by reusing the authoritative six-row base bundle through append-only lineage, launching only the two U-NO rows under the unchanged contract, and publishing a new extended bundle plus durable discoverability updates.

**Architecture:** Treat the existing six-row `complete_table_20260430T150757Z_repair_tmux` root as immutable paper authority and build one new derived root around it. Extend the paper-benchmark / compare-wrapper surfaces just enough to promote the base rows by lineage, launch the two fresh U-NO rows through the existing Torch runner path, then collate an eight-row bundle with merged tables, fixed-sample visuals, row-local launcher proof, and explicit claim-boundary metadata.

**Tech Stack:** PATH `python`, `ptycho311` for long-running row launches, PyTorch/Lightning, external `neuraloperator==2.0.0` via the already integrated `neuralop_uno` generator, `scripts/studies/lines128_paper_benchmark.py`, `scripts/studies/grid_lines_compare_wrapper.py`, pytest, `compileall`, Markdown/JSON/CSV/TeX artifacts, repo-local `.artifacts/` verification logs.

---

## Selected Backlog Objective

- Implement backlog item `2026-04-30-cdi-lines128-uno-table-extension`.
- Run exactly two fresh rows under the locked `lines128` CDI contract:
  - `pinn_neuralop_uno`
  - `supervised_neuralop_uno`
- Reuse the completed six-row base root only by lineage/promotion into a new extended bundle.
- Publish a new claim boundary:
  `complete_lines128_cdi_benchmark_plus_uno_extension`.

## Scope And Explicit Non-Goals

In scope:

- consume the authoritative six-row base root:
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-paper-benchmark-execution/runs/complete_table_20260430T150757Z_repair_tmux`
- consume the completed `neuralop_uno` runtime support from the prior integration item
- extend the compare-wrapper / paper-benchmark bundle logic so the base six rows can be promoted by lineage while only the two U-NO rows are launched fresh
- preserve the fixed `lines128` contract:
  dataset identity, split, probe, `seed=3`, `40` epochs, scheduler, `mae`, `real_imag`, fixed visual sample ids `0` and `1`, selected FNO comparator `fno_vanilla`, and shared visual-scale policy
- produce a new derived root such as
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-30-cdi-lines128-uno-table-extension/runs/complete_table_plus_uno_<timestamp>`
- emit merged metrics/manifests/visuals plus row-local provenance for both fresh U-NO rows
- write a durable summary and update the NeurIPS evidence indexes so the new extension is discoverable

Explicit non-goals:

- do not rerun the six completed base rows
- do not overwrite or mutate `complete_table_20260430T150757Z_repair_tmux`
- do not change the locked `lines128` contract to make U-NO easier
- do not tune `neuralop_uno` hyperparameters beyond the preflight-frozen settings already implemented
- do not broaden `neuralop_uno` beyond the locked CDI lane (`N=128`, `gridsize=1`, `C=1`, `real_imag`) unless a narrow execution bug requires a clearly documented follow-up item
- do not reopen Phase 2 PDEBench work, WaveBench candidate work, BRDT candidate work, `256x256` CDI scaling, or manuscript-output assembly under `/home/ollie/Documents/neurips/`
- do not modify `ptycho/model.py`, `ptycho/diffsim.py`, or `ptycho/tf_helper.py`

## Binding Constraints And Prerequisite Status

Strategic and roadmap constraints:

- `docs/steering.md` requires apples-to-apples comparisons and forbids silently relaxing fairness constraints. This item must therefore keep the exact locked `lines128` contract and report any incompatibility explicitly rather than drifting the protocol.
- `docs/plans/2026-04-20-neurips-hybrid-resnet-submission-roadmap.md` places optional new CDI comparator extensions under Phase `3.3f` only after the authoritative six-row bundle is locked. This item is append-only follow-up evidence, not a rewrite of the current CDI headline authority.
- `docs/plans/2026-04-20-neurips-hybrid-resnet-submission-design.md` keeps the CDI headline at `128x128` and requires provenance-heavy artifact handling for any paper-facing comparator extension.
- The selected backlog item explicitly requires a new derived bundle, distinct claim boundary, row-local launcher proof, and durable index updates.

Prerequisite status that matters here:

- `state/NEURIPS-HYBRID-RESNET-2026/progress_ledger.json` confirms the initiative’s early phases are complete and shows no globally blocked tranche preventing current Phase 3 CDI work, but it does not yet reflect later microstate for the completed CDI backlog chain.
- Treat the checked-in backlog outcomes below as the binding prerequisite evidence:
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/lines128_paper_benchmark_summary.md`
    confirms the authoritative six-row base bundle is complete.
  - `docs/backlog/done/2026-04-30-cdi-lines128-uno-generator-integration.md`
    plus
    `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-30-cdi-lines128-uno-generator-integration/execution_report.md`
    confirm `neuralop_uno` is a real Torch generator in both PINN and supervised paths.
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/lines128_uno_preflight_summary.md`
    remains the environment/API authority for the frozen U-NO settings.
  - `docs/backlog/done/2026-04-29-wavebench-inverse-source-preflight.md`
    satisfies the backlog signal that optional WaveBench candidate work should be attempted before this optional U-NO table-extension lane.

Frozen U-NO settings and contract facts that must remain unchanged:

- architecture id: `neuralop_uno`
- training procedures:
  - `pinn_neuralop_uno` -> `pinn`
  - `supervised_neuralop_uno` -> `supervised`
- `in_channels=1`
- `out_channels=2`
- `hidden_channels=32`
- `lifting_channels=128`
- `projection_channels=128`
- `n_layers=4`
- `uno_out_channels=[32, 64, 64, 32]`
- `uno_n_modes=[[12, 12], [12, 12], [12, 12], [12, 12]]`
- `uno_scalings=[[1.0, 1.0], [0.5, 0.5], [1, 1], [2, 2]]`
- `positional_embedding="grid"`
- `channel_mlp_skip="linear"`
- `generator_output_mode="real_imag"`

Failure-handling policy:

- Do not mark the item `BLOCKED` for ordinary import, path, test-harness, or verification failures. Diagnose, patch narrowly, and rerun first.
- Reserve `BLOCKED` for:
  - missing or corrupted authoritative base-bundle inputs
  - unresolved external `neuraloperator` / CUDA / hardware failure after one narrow fix attempt
  - a duplicate active writer on the intended extension output root that cannot be cleanly resolved
  - an unrecoverable mismatch between the locked U-NO contract and the already integrated `neuralop_uno` runtime
  - a roadmap or user-authority conflict outside this item’s scope

Long-run execution rules for this item:

- No fresh U-NO row launch may start until the blocking pre-launch checks below are green.
- Launch long-running commands in tmux, activate `ptycho311`, and keep execution ownership until terminal success or recoverable failure handling is complete.
- Track the exact launched PID (`cmd ... & pid=$!; wait "$pid"`); do not use `pgrep -f` as the primary completion check.
- Do not start a second run against the same `complete_table_plus_uno_<timestamp>` output root while another writer is active.
- Consider a U-NO row or merged-bundle launch complete only when:
  - the tracked PID exits `0`, and
  - required row/bundle artifacts for that step exist and are freshly written

## Implementation Architecture

- **Append-only benchmark orchestration**
  - `scripts/studies/lines128_paper_benchmark.py` should own the extension mode, row roster, base-root promotion rules, execution manifest, and final bundle validation.
- **Shared runner/collation plumbing**
  - `scripts/studies/grid_lines_compare_wrapper.py` should own shared dataset reuse, child-run routing, merged metrics, fixed-sample visuals, and any current-root row recovery needed for the fresh U-NO rows.
- **Provenance and completion-proof support**
  - `scripts/studies/paper_provenance.py` should be updated only if existing completion-proof or lineage helpers are too narrow for append-only promoted-plus-fresh bundles.
- **Proof and publication**
  - Focused tests must prove immutability of the base root, exact row roster, same-contract launch flags, row-local proof requirements, and final bundle discoverability before and after the expensive row launches.

## Concrete File And Artifact Targets

Core implementation targets:

- Modify: `scripts/studies/lines128_paper_benchmark.py`
- Modify: `scripts/studies/grid_lines_compare_wrapper.py`
- Modify only if lineage/completion-proof reuse needs it:
  `scripts/studies/paper_provenance.py`
- Modify only if fresh row-local labels/provenance require a narrow fix:
  `scripts/studies/grid_lines_torch_runner.py`

Likely test targets:

- Modify: `tests/studies/test_lines128_paper_benchmark.py`
- Modify: `tests/test_grid_lines_compare_wrapper.py`
- Modify only if provenance helpers change:
  `tests/studies/test_paper_provenance.py`
- Keep the existing required runner gate in:
  `tests/torch/test_grid_lines_torch_runner.py`

Mandatory contract outputs:

- new derived bundle root:
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-30-cdi-lines128-uno-table-extension/runs/complete_table_plus_uno_<timestamp>`
- merged bundle payload in that root:
  - `metrics.json`
  - `metrics_table.csv`
  - `metrics_table.tex`
  - `metrics_table_best.tex`
  - `model_manifest.json`
  - `paper_benchmark_manifest.json`
  - explicit claim boundary
    `complete_lines128_cdi_benchmark_plus_uno_extension`
  - explicit base-row lineage plus fresh U-NO row provenance
  - fixed-sample visual bundle under the same sample ids and shared-scale policy
- fresh row-local directories for:
  - `pinn_neuralop_uno`
  - `supervised_neuralop_uno`
  Each row must retain invocation/config/history/metrics/reconstruction artifacts, launcher completion proof, and environment/package provenance.
- item-local verification logs under:
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-30-cdi-lines128-uno-table-extension/verification/`
- durable summary:
  `docs/plans/NEURIPS-HYBRID-RESNET-2026/lines128_uno_table_extension_summary.md`

Preferred packaging, not a substitute for the mandatory outputs:

- dedicated machine-readable lineage audit such as `base_row_lineage.json`
- explicit run/execution manifest for the append-only launch
- fixed-sample source arrays or audit payloads if the current `lines128` harness already emits them cleanly
- convenience audit comparing promoted base-row metrics in the extended bundle against the immutable six-row authority

Durable docs and index targets:

- Create: `docs/plans/NEURIPS-HYBRID-RESNET-2026/lines128_uno_table_extension_summary.md`
- Modify: `docs/plans/NEURIPS-HYBRID-RESNET-2026/paper_evidence_index.md`
- Modify: `docs/plans/NEURIPS-HYBRID-RESNET-2026/evidence_matrix.md`
- Modify: `docs/plans/NEURIPS-HYBRID-RESNET-2026/model_variant_index.json`
- Modify: `docs/plans/NEURIPS-HYBRID-RESNET-2026/ablation_index.json`
- Modify: `docs/studies/index.md`
- Modify: `docs/index.md`
- Update `docs/findings.md` only if execution uncovers a durable repo-level trap that is not already covered by the existing U-NO preflight and integration docs

## Execution Checklist

### Task 1: Lock The Append-Only Contract And Add Focused Red Tests

**Files:**
- Modify: `tests/studies/test_lines128_paper_benchmark.py`
- Modify: `tests/test_grid_lines_compare_wrapper.py`
- Optional modify: `tests/studies/test_paper_provenance.py`

- [ ] Add or update tests that prove the extension path consumes the immutable six-row base root and refuses any mode that would relaunch or overwrite those rows.
- [ ] Add or update tests that the extension roster is exactly eight rows after collation:
  - six promoted base rows
  - `pinn_neuralop_uno`
  - `supervised_neuralop_uno`
- [ ] Add or update tests that the U-NO extension row keys map to architecture id `neuralop_uno`, paper labels `U-NO + PINN` / `U-NO + supervised`, and the correct training procedures.
- [ ] Add or update tests that the final bundle manifest records:
  - base-root lineage
  - fresh U-NO row provenance
  - claim boundary `complete_lines128_cdi_benchmark_plus_uno_extension`
- [ ] Add or update tests that missing base metrics, missing row-local launcher proof, or missing fresh-row provenance prevent the merged bundle from being overstated as complete.
- [ ] Keep these tests bounded to append-only orchestration and collation. Do not reopen generator-registry behavior already proven by the prior U-NO integration item unless a narrow execution bug requires it.

**Blocking verification before moving on:**

- [ ] Run the new focused selectors and confirm they fail for the expected append-only extension gaps rather than unrelated infrastructure noise.
- [ ] Record red-first evidence under the item-local `verification/` directory.

### Task 2: Implement Append-Only Promotion And U-NO Row Routing

**Files:**
- Modify: `scripts/studies/lines128_paper_benchmark.py`
- Modify: `scripts/studies/grid_lines_compare_wrapper.py`
- Conditional modify: `scripts/studies/paper_provenance.py`
- Conditional modify: `scripts/studies/grid_lines_torch_runner.py`

- [ ] Add or extend an explicit append-only execution path in `lines128_paper_benchmark.py` that accepts:
  - the authoritative base root
  - a fresh extension output root
  - the exact two new U-NO rows to launch
- [ ] Keep the extension path deterministic:
  - same dataset identity
  - same split
  - same probe and preprocessing
  - same seed / epoch / loss / scheduler / output-mode contract
  - same fixed visual sample ids and shared scales
- [ ] Promote the six base rows into the new root by lineage reference and bundle collation only. Do not relaunch, retrain, or mutate the original row roots.
- [ ] Route only `pinn_neuralop_uno` and `supervised_neuralop_uno` through fresh child execution.
- [ ] Ensure the fresh U-NO rows retain row-local invocation/config/history/metrics/reconstruction artifacts and launcher completion proof in the same style expected by the existing paper-benchmark bundle.
- [ ] If promoted-plus-fresh bundle collation needs helper work, widen `paper_provenance.py` narrowly so the base lineage and current-root U-NO proofs are both represented cleanly.
- [ ] Preserve existing six-row bundle logic and existing FFNO/spectral/CNN handling. No unrelated comparator behavior should drift.

**Blocking verification before moving on:**

- [ ] Run the focused benchmark / wrapper selectors until the append-only logic is green.
- [ ] Re-run the selected required runner surface if any direct-runner code path changed.

### Task 3: Pass The Pre-Launch Deterministic Gates

**Files:**
- No new durable code targets; this task archives pre-launch verification evidence under the item-local artifact root.

- [ ] Run the selected backlog item’s required input-presence check unchanged:

```bash
python - <<'PY'
from pathlib import Path
required = [
    Path("docs/plans/NEURIPS-HYBRID-RESNET-2026/lines128_uno_table_extension_design.md"),
    Path(".artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-paper-benchmark-execution/runs/complete_table_20260430T150757Z_repair_tmux/metrics.json"),
]
missing = [str(p) for p in required if not p.exists()]
if missing:
    raise SystemExit(f"missing U-NO table-extension inputs: {missing}")
print("U-NO table-extension inputs present")
PY
```

- [ ] Run the selected backlog item’s required deterministic pytest gate unchanged:

```bash
pytest -q tests/test_grid_lines_compare_wrapper.py tests/torch/test_grid_lines_torch_runner.py
```

- [ ] Run the selected backlog item’s required compile gate unchanged:

```bash
python -m compileall -q ptycho_torch scripts/studies
```

- [ ] Run a blocking live `ptycho311` U-NO dependency/API gate before any tmux launch so the external package contract is validated without rediscovery:

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ptycho311
python -m pip show neuraloperator
python - <<'PY'
import inspect
import json
import sys

import neuralop
from neuralop.models import UNO

if neuralop.__version__ != "2.0.0":
    raise SystemExit(
        f"expected neuralop 2.0.0 in ptycho311, found {neuralop.__version__}"
    )

signature = str(inspect.signature(UNO))
required_terms = ["uno_out_channels", "uno_n_modes", "uno_scalings"]
missing = [term for term in required_terms if term not in signature]
if missing:
    raise SystemExit(f"UNO signature missing expected terms: {missing}")

print(
    json.dumps(
        {
            "python_executable": sys.executable,
            "python_version": sys.version.split()[0],
            "neuralop_version": neuralop.__version__,
            "uno_signature": signature,
        },
        indent=2,
        sort_keys=True,
    )
)
PY
```

- [ ] If the live gate fails because `neuraloperator` is missing or incompatible in `ptycho311`, do one narrow repair attempt with the pinned package, rerun the live gate, and archive both logs before deciding whether a row-level environment blocker remains:

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ptycho311
python -m pip install "neuraloperator==2.0.0"
python -m pip show neuraloperator
```

- [ ] Run stronger extension-focused selectors as a supplemental blocking gate before any expensive U-NO launch:

```bash
pytest -q tests/studies/test_lines128_paper_benchmark.py
```

- [ ] If `scripts/studies/paper_provenance.py` changed, add this supplemental blocking gate:

```bash
pytest -q tests/studies/test_paper_provenance.py
```

- [ ] Archive all pre-launch logs under the item-local `verification/` directory.

**Blocking / supporting labels:**

- [ ] `required input check` -> blocking
- [ ] `tests/test_grid_lines_compare_wrapper.py tests/torch/test_grid_lines_torch_runner.py` -> blocking
- [ ] `compileall` -> blocking
- [ ] live `ptycho311` `neuraloperator==2.0.0` / `neuralop.models.UNO` gate -> blocking
- [ ] pinned `neuraloperator==2.0.0` repair + rerun -> blocking when the first live gate fails
- [ ] `tests/studies/test_lines128_paper_benchmark.py` -> blocking
- [ ] `tests/studies/test_paper_provenance.py` -> blocking only if that helper changed; otherwise omit

### Task 4: Launch Only The Fresh U-NO Rows And Build The Extended Bundle

**Files / artifact targets:**
- Fresh run root under:
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-30-cdi-lines128-uno-table-extension/runs/`
- Long-run command owner:
  `scripts/studies/lines128_paper_benchmark.py`

- [ ] Create one fresh extension output root, for example:
  `complete_table_plus_uno_<timestamp>`.
- [ ] Before launching, confirm no other process is already writing to that root.
- [ ] Launch in tmux using `ptycho311`; the extension path must launch only:
  - `pinn_neuralop_uno`
  - `supervised_neuralop_uno`
- [ ] Track the exact launched PID and wait for terminal success. Do not rely on broad process-name polling.
- [ ] If one fresh U-NO row fails because of ordinary implementation bugs or environment drift that can be fixed narrowly in scope, diagnose, patch, and rerun the row or extension path instead of declaring the backlog item blocked immediately.
- [ ] Treat the run as complete only after:
  - PID exit code `0`
  - both fresh U-NO row directories contain row-local launcher proof, metrics, reconstructions, and provenance
  - merged bundle files and visuals are freshly written
- [ ] Keep the base six-row metrics and lineage stable in the new bundle. The extension may add rows and manifests, but it must not rewrite historical base-row values.

**Blocking verification after the launch:**

- [ ] Confirm the final bundle contains the mandatory merged outputs and exactly eight rows.
- [ ] Confirm the base authority root is unchanged and still separately discoverable.
- [ ] Confirm the fresh U-NO rows have explicit environment/package provenance, not just inherited wrapper labels.

**Supporting verification after the launch:**

- [ ] If convenient, run an audit comparing promoted base-row metrics in the extended bundle against the immutable six-row authority and archive the result.
- [ ] If the harness already emits fixed-sample source arrays cleanly, verify they remain aligned with sample ids `0` and `1`.

### Task 5: Publish The Durable Summary And Refresh Discoverability

**Files:**
- Create: `docs/plans/NEURIPS-HYBRID-RESNET-2026/lines128_uno_table_extension_summary.md`
- Modify: `docs/plans/NEURIPS-HYBRID-RESNET-2026/paper_evidence_index.md`
- Modify: `docs/plans/NEURIPS-HYBRID-RESNET-2026/evidence_matrix.md`
- Modify: `docs/plans/NEURIPS-HYBRID-RESNET-2026/model_variant_index.json`
- Modify: `docs/plans/NEURIPS-HYBRID-RESNET-2026/ablation_index.json`
- Modify: `docs/studies/index.md`
- Modify: `docs/index.md`
- Conditional: `docs/findings.md`

- [ ] Write the durable summary with:
  - authoritative extended bundle root
  - exact eight-row roster
  - preserved claim boundary
  - base-root lineage
  - fresh U-NO row provenance
  - final verification commands and archived log paths
  - outcome wording that keeps this as an append-only extension, not a rewrite of the six-row authority
- [ ] Add a new paper-evidence-index row for this backlog item with the correct phase, tier, protocol, summary authority, artifact root, and downstream-use note.
- [ ] Update the evidence matrix so the CDI lines128 row table and ablation/extension families reflect the new U-NO rows and their boundary.
- [ ] Add or update `model_variant_index.json` entries for both new U-NO rows under the fixed `cdi_lines128_seed3` contract.
- [ ] Add or update `ablation_index.json` so the U-NO extension is discoverable as an append-only architecture-table extension rather than an unrelated fresh benchmark.
- [ ] Update `docs/studies/index.md` and `docs/index.md` so later planning can discover the new summary and extended bundle from the standard entry points.
- [ ] Update `docs/findings.md` only if execution surfaces a durable rule or trap not already documented by the existing U-NO preflight/integration materials.

**Blocking verification before closing the item:**

- [ ] The summary, evidence indexes, and discoverability docs all point to the same authoritative extended root and claim boundary.
- [ ] No doc describes the new extension as replacing the immutable six-row authority.

## Verification Matrix

Blocking checks:

- `python` required-input presence check from the selected backlog item
- `pytest -q tests/test_grid_lines_compare_wrapper.py tests/torch/test_grid_lines_torch_runner.py`
- `python -m compileall -q ptycho_torch scripts/studies`
- live `ptycho311` gate proving `neuraloperator==2.0.0` is installed and `neuralop.models.UNO` exposes the expected constructor surface
- one narrow `neuraloperator==2.0.0` repair attempt plus a rerun of the live gate if the first environment/API check fails
- `pytest -q tests/studies/test_lines128_paper_benchmark.py`
- `pytest -q tests/studies/test_paper_provenance.py` only if that helper changed
- fresh U-NO launch exit code and mandatory artifact-presence validation
- final summary/index consistency check

Supporting checks:

- red-first focused selector logs for append-only extension behavior
- base-row metric-stability audit against the immutable six-row root
- fixed-sample source-array or visual-audit verification if already emitted by the harness

## Completion Gate

- The authoritative six-row base root remains untouched and still stands as the original CDI headline authority.
- The new extension root is the only place where U-NO rows appear.
- Only `pinn_neuralop_uno` and `supervised_neuralop_uno` were freshly launched.
- The final extended bundle is complete only if both U-NO rows carry row-local launcher proof, metrics, reconstructions, and environment provenance.
- All required deterministic checks have passing archived evidence.
- The durable summary and evidence indexes all reference the same extended root and claim boundary:
  `complete_lines128_cdi_benchmark_plus_uno_extension`.
