# CDI Lines128 FFNO Depth-24 Ablation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add one append-only CDI `lines128` FFNO row with `fno_blocks=24`, compare it against the corrected no-refiner four-block `pinn_ffno` authority under the same locked contract, and publish the result as ablation evidence without changing the active paper-facing FFNO rows.

**Architecture:** Reuse the corrected four-block no-refiner `pinn_ffno` row by lineage and add only the minimal row-spec or wrapper plumbing required for a distinct `pinn_ffno_depth24` row. Keep execution isolated under an item-local artifact root, prove that only `fno_blocks` changed, and update the NeurIPS summary/index surfaces as append-only depth-ablation evidence rather than as headline manuscript authority.

**Tech Stack:** Python 3.11 in `ptycho311`, PyTorch/Lightning, `scripts/studies/grid_lines_compare_wrapper.py`, `scripts/studies/grid_lines_torch_runner.py`, FFNO generator surfaces in `ptycho_torch/`, Markdown/JSON evidence indexes under `docs/plans/NEURIPS-HYBRID-RESNET-2026/`.

---

## Selected Backlog Objective

- Add one new CDI `lines128` row id, preferably `pinn_ffno_depth24`, whose only intentional architectural change versus the corrected default `pinn_ffno` row is `fno_blocks=24` instead of `4`.
- Compare the fresh depth-24 row directly against the corrected no-refiner four-block `pinn_ffno` authority from `2026-05-06-cdi-lines128-ffno-no-refiner-row-rerun`.
- Record amplitude and phase reconstruction metrics, parameter count, runtime, provenance, and a concise interpretation of whether deeper pure FFNO helps or hurts on the locked CDI contract.

## Scope

- Fixed CDI contract for both rows:
  - `N=128`, `gridsize=1`, synthetic grid-lines, `set_phi=True`
  - Run1084 fixed-probe lineage with `probe_scale_mode=pad_extrapolate` and `probe_smoothing_sigma=0.5`
  - `seed=3`, `nimgs_train=2`, `nimgs_test=2`, fixed sample ids
  - `nphotons=1e9`, `probe_mask=off`
  - `40` epochs, batch `16`, Adam `2e-4`
  - `ReduceLROnPlateau(factor=0.5, patience=2, threshold=0.0, min_lr=1e-4)`
  - `torch_loss_mode=mae`, `torch_mae_pred_l2_match_target=off`, `torch_output_mode=real_imag`
  - `fno_modes=12`, `fno_width=32`, `fno_cnn_blocks=0`
- Allowed row difference:
  - baseline `pinn_ffno`: `fno_blocks=4`
  - fresh `pinn_ffno_depth24`: `fno_blocks=24`
- Required durable outputs:
  - item-local artifact root with the fresh row, row-local provenance, verification logs, and derived two-row comparison payloads
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/cdi_lines128_ffno_depth24_ablation_summary.md`
  - updates to `docs/plans/NEURIPS-HYBRID-RESNET-2026/evidence_matrix.md`
  - updates to `docs/plans/NEURIPS-HYBRID-RESNET-2026/model_variant_index.json`
  - updates to `docs/plans/NEURIPS-HYBRID-RESNET-2026/ablation_index.json`
  - update to `docs/studies/index.md`

## Explicit Non-Goals

- Do not rerun or relabel the corrected four-block `pinn_ffno` row.
- Do not use the historical `fno_cnn_blocks=2` FFNO-local-refiner proxy as the comparison baseline.
- Do not add the supervised depth-24 companion row here; that belongs to `2026-05-06-cdi-lines128-supervised-ffno-depth24-no-refiner-rerun`.
- Do not refresh paper-facing CDI tables, phase-zoom figures, model-config tables, or efficiency tables here; that promotion decision belongs to `2026-05-06-cdi-lines128-ffno-depth24-final-paper-refresh`.
- Do not change the dataset, probe, loss, width, modes, scheduler, seed, sample ids, or metric definitions while claiming this isolates FFNO depth.
- Do not broaden this item into CNS authored-FFNO work, SRU-Net encoder work, multi-seed CDI work, or a broader FFNO sweep.
- Do not modify `ptycho/model.py`, `ptycho/diffsim.py`, or `ptycho/tf_helper.py`.

## Steering And Roadmap Constraints

- This remains a Phase 3 CDI item under `phase-3-cdi-anchor-regeneration`; it must not silently expand into later roadmap items or rewrite the roadmap.
- Steering requires equal-footing comparisons and forbids silent fairness drift. If the new row cannot be kept on the same `lines128` contract, record the incompatibility precisely instead of weakening the comparator.
- The result is append-only ablation evidence. It does not automatically replace the current active four-block no-refiner FFNO manuscript rows.
- Long-running commands stay under implementation ownership until tracked terminal success or a recoverable failure is diagnosed, fixed, and rerun. Use tmux for long runs, activate `ptycho311`, track the exact launched PID, and do not launch a duplicate writer into the same output root.
- Normal test, import, path, or harness failures are not automatic `BLOCKED` outcomes. Diagnose, patch narrowly, and rerun first. Reserve `BLOCKED` for genuinely missing authoritative prerequisite artifacts, unavailable hardware, external dependency issues outside current authority, roadmap conflict, or an unrecoverable failure after a documented narrow fix attempt.

## Prerequisite Status

- Consumed `progress_ledger.json` confirms the initiative already completed its earlier roadmap setup tranches (`phase-0-evidence-inventory`, `phase-1-pde-benchmark-selection`, and the recorded early Phase 2 readiness tranches), so this selected item is operating inside an already active NeurIPS campaign rather than creating a new phase.
- Direct item prerequisites are already satisfied in repo-local authorities and must be reused, not rerun, unless a narrow artifact-integrity audit proves otherwise:
  - `2026-04-29-cdi-lines128-paper-benchmark-execution` is complete; authoritative six-row bundle root:
    `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-paper-benchmark-execution/runs/complete_table_20260430T150757Z_repair_tmux`
  - `2026-05-06-cdi-lines128-ffno-no-refiner-row-rerun` is complete; corrected four-block no-refiner source root:
    `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-06-cdi-lines128-ffno-no-refiner-row-rerun/runs/ffno_no_refiner_20260506T223454Z`
  - `2026-05-06-cdi-lines128-no-refiner-ffno-table-refresh` is complete; active paper-local FFNO refresh root:
    `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-06-cdi-lines128-no-refiner-ffno-table-refresh/`
- Implementation should treat those completed authorities as locked lineage inputs. Only the new depth-24 row should run in this item.

## Implementation Architecture

### 1. Row-Spec And Contract Plumbing

- Keep `pinn_ffno` permanently bound to the corrected four-block no-refiner contract.
- Add a distinct row surface for `pinn_ffno_depth24` with display text such as `FFNO-24 + PINN`.
- Prefer the smallest possible change surface:
  - likely primary surface: `scripts/studies/grid_lines_compare_wrapper.py`
  - conditional support surface: `scripts/studies/grid_lines_torch_runner.py` only if row-id override, config capture, or artifact naming is insufficient
  - avoid mutating `scripts/studies/lines128_paper_benchmark.py` unless shared serialization logic is required and can be reused without changing the complete-table roster assumptions

### 2. Execution And Audit Surface

- Launch exactly one item-local long run for the fresh depth-24 row.
- Preserve baseline comparison by lineage using the corrected four-block root rather than rerunning it.
- Emit a machine-readable contract audit proving:
  - baseline root reports `fno_blocks=4` and `fno_cnn_blocks=0`
  - fresh root reports `fno_blocks=24` and `fno_cnn_blocks=0`
  - no other locked CDI contract field drifted

### 3. Durable Summary And Discoverability

- Publish a concise summary that states the claim boundary, baseline lineage source, depth-only comparison, and compute context.
- Update only the study/evidence indexes that should discover this append-only ablation.
- Keep manuscript-facing table/figure promotion explicitly deferred.

## File And Artifact Targets

### Mandatory Contract Outputs

- Fresh plan authority:
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-06-cdi-lines128-ffno-depth24-ablation/execution_plan.md`
- Durable summary:
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/cdi_lines128_ffno_depth24_ablation_summary.md`
- Discoverability updates:
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/evidence_matrix.md`
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/model_variant_index.json`
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/ablation_index.json`
  - `docs/studies/index.md`
- Required item-local evidence root:
  - preferred concrete root:
    `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-06-cdi-lines128-ffno-depth24-ablation/`
  - required contents:
    - fresh run root for `pinn_ffno_depth24`
    - verification logs
    - contract audit artifacts
    - two-row comparison payloads referencing both the reused four-block root and the fresh depth-24 root

### Likely Code Targets

- Likely mandatory if the current wrapper cannot express a second FFNO row id cleanly:
  - `scripts/studies/grid_lines_compare_wrapper.py`
  - `tests/test_grid_lines_compare_wrapper.py`
- Conditional only if the wrapper path cannot preserve row-local artifact naming or config capture:
  - `scripts/studies/grid_lines_torch_runner.py`
  - `tests/torch/test_grid_lines_torch_runner.py`
- Conditional only if shared paper-benchmark helpers are reused for comparison serialization or contract auditing:
  - `scripts/studies/lines128_paper_benchmark.py`
  - `tests/studies/test_lines128_paper_benchmark.py`
- Check surfaces, not planned edit surfaces:
  - `ptycho_torch/generators/ffno.py`
  - `ptycho_torch/generators/ffno_bottleneck.py`
  - `tests/torch/test_generator_registry.py`

### Preferred Packaging

- Keep row-local artifacts under a fresh timestamped run directory such as:
  - `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-06-cdi-lines128-ffno-depth24-ablation/runs/ffno_depth24_<timestamp>/`
- Keep verification logs under:
  - `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-06-cdi-lines128-ffno-depth24-ablation/verification/`
- Prefer derived comparison payload names such as:
  - `comparison_depth24_vs_depth4.json`
  - `comparison_depth24_vs_depth4.csv`
  - optional `comparison_depth24_vs_depth4.tex`

## Task Checklist

### Task 1: Lock The Baseline And Preflight The Item

**Files:**
- Read/consume:
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/lines128_paper_benchmark_preflight.md`
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/lines128_paper_benchmark_summary.md`
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/cdi_lines128_ffno_no_refiner_row_rerun_summary.md`
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/cdi_lines128_no_refiner_ffno_table_refresh_summary.md`
- Audit inputs:
  - `docs/backlog/done/2026-04-29-cdi-lines128-paper-benchmark-execution.md`
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/tables/cdi_lines128_metrics_extended.json`

- [ ] Run the authoritative input-presence gate from the selected item before editing or launching anything expensive:

```bash
python - <<'PY'
from pathlib import Path
required = [
    Path("docs/backlog/done/2026-04-29-cdi-lines128-paper-benchmark-execution.md"),
    Path("docs/plans/NEURIPS-HYBRID-RESNET-2026/lines128_paper_benchmark_preflight.md"),
    Path("docs/plans/NEURIPS-HYBRID-RESNET-2026/lines128_paper_benchmark_summary.md"),
    Path("docs/plans/NEURIPS-HYBRID-RESNET-2026/tables/cdi_lines128_metrics_extended.json"),
    Path("ptycho_torch/generators/ffno.py"),
    Path("ptycho_torch/generators/ffno_bottleneck.py"),
    Path("scripts/studies/grid_lines_torch_runner.py"),
    Path("scripts/studies/grid_lines_compare_wrapper.py"),
    Path("scripts/studies/lines128_paper_benchmark.py"),
]
missing = [str(path) for path in required if not path.exists()]
if missing:
    raise SystemExit(f"missing CDI FFNO depth-24 ablation inputs: {missing}")
print("CDI FFNO depth-24 ablation inputs present")
PY
```

- [ ] Record the corrected four-block baseline lineage root and the active paper-local refresh root in the new item-local notes or summary draft before any code changes.
- [ ] Audit whether the current wrapper already supports an explicit `pinn_ffno_depth24` row id with row-local `fno_blocks=24` while preserving `pinn_ffno` at `4`. If yes, preserve that evidence and skip unnecessary production edits. If no, proceed to Task 2 with the narrowest wrapper/plumbing change.

**Verification:**
- Blocking:
  - the required presence gate above must pass before the implementation assumes prerequisite authority
  - the corrected no-refiner baseline root must be confirmed as the comparator source; do not fall back to the historical proxy root
- Supporting:
  - inspect the corrected baseline `invocation.json` or `config.json` and note `fno_blocks=4`, `fno_cnn_blocks=0` for later contract diff use

### Task 2: Add Minimal Depth-24 Row Plumbing

**Files:**
- Likely modify:
  - `scripts/studies/grid_lines_compare_wrapper.py`
  - `tests/test_grid_lines_compare_wrapper.py`
- Modify only if needed:
  - `scripts/studies/grid_lines_torch_runner.py`
  - `tests/torch/test_grid_lines_torch_runner.py`
  - `scripts/studies/lines128_paper_benchmark.py`
  - `tests/studies/test_lines128_paper_benchmark.py`
- Verify only:
  - `ptycho_torch/generators/ffno.py`
  - `ptycho_torch/generators/ffno_bottleneck.py`
  - `tests/torch/test_generator_registry.py`

- [ ] Add or expose a row spec for `pinn_ffno_depth24` that keeps architecture `ffno`, training mode `PINN`, and all locked CDI fields unchanged except `fno_blocks=24`.
- [ ] Preserve the existing meaning of `pinn_ffno` as the corrected four-block no-refiner authority. Do not relabel or mutate that row.
- [ ] Ensure invocation/config artifacts for the new row persist:
  - `model_id = pinn_ffno_depth24`
  - `fno_blocks = 24`
  - `fno_cnn_blocks = 0`
  - display label such as `FFNO-24 + PINN`
- [ ] Add focused regression coverage proving:
  - default `pinn_ffno` remains four blocks
  - the new row emits the depth-24 override
  - artifact naming stays row-local under `pinn_ffno_depth24`
- [ ] If shared comparison serialization is reused, keep it item-local or append-only; do not change the complete-table roster assumptions in `lines128_paper_benchmark.py`.

**Verification:**
- Blocking:
  - `pytest -q tests/torch/test_generator_registry.py -k "ffno"`
  - `pytest -q tests/torch/test_grid_lines_torch_runner.py -k "ffno"`
  - `pytest -q tests/test_grid_lines_compare_wrapper.py -k "ffno"`
  - `python -m compileall -q ptycho_torch scripts/studies`
- Supporting:
  - if `scripts/studies/lines128_paper_benchmark.py` changes, run a narrow selector such as `pytest -q tests/studies/test_lines128_paper_benchmark.py -k "ffno"` and archive that log under the item-local `verification/` directory

### Task 3: Run Only The Fresh Depth-24 Row

**Files / artifacts:**
- Fresh item root:
  - `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-06-cdi-lines128-ffno-depth24-ablation/`
- Fresh run root:
  - `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-06-cdi-lines128-ffno-depth24-ablation/runs/ffno_depth24_<timestamp>/`
- Expected fresh row outputs under the run root:
  - `runs/pinn_ffno_depth24/{invocation.json,config.json,history.json,metrics.json,model.pt,stdout.log,stderr.log}`
  - `recons/pinn_ffno_depth24/recon.npz`
  - row-local visuals for amplitude/phase prediction and error

- [ ] Launch the long-running row in tmux from the repo root with `ptycho311` active, a unique output root, and tracked PID ownership. Do not launch another writer into that same output root.
- [ ] Derive the command from the corrected no-refiner four-block rerun, changing only what is required for the new row:
  - explicit row id / model selection for `pinn_ffno_depth24`
  - item-local output root
  - `--fno-blocks 24`
  - `--fno-cnn-blocks 0`
- [ ] Wait on the exact launched PID until exit `0`, then verify fresh required artifacts exist before accepting the run as complete.
- [ ] If the run fails because of code, harness, or environment drift, diagnose and patch narrowly, then relaunch the same item-local row. Do not mark the item `BLOCKED` until a documented narrow fix attempt fails for a reason outside current authority.
- [ ] Produce an item-local contract audit proving the only allowed contract delta versus the corrected baseline is `fno_blocks: 4 -> 24`.

**Verification:**
- Blocking:
  - tracked tmux-owned command exits `0`
  - fresh `invocation.json` / `config.json` for `pinn_ffno_depth24` record `fno_blocks=24` and `fno_cnn_blocks=0`
  - baseline lineage audit records `fno_blocks=4` and `fno_cnn_blocks=0`
  - required row outputs are present and freshly written
  - contract audit reports no drift outside the allowed `fno_blocks` change
- Supporting:
  - capture parameter count, train wall time, and inference time in a machine-readable comparison payload so metric deltas are not interpreted without compute context
  - compare the fresh saved state dict against the corrected baseline to note parameter growth from depth `4` to `24`

### Task 4: Publish The Append-Only Comparison And Update Discoverability

**Files:**
- Create:
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/cdi_lines128_ffno_depth24_ablation_summary.md`
- Update:
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/evidence_matrix.md`
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/model_variant_index.json`
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/ablation_index.json`
  - `docs/studies/index.md`
- Preferred item-local comparison payloads:
  - `comparison_depth24_vs_depth4.json`
  - `comparison_depth24_vs_depth4.csv`
  - optional `comparison_depth24_vs_depth4.tex`

- [ ] Build a two-row comparison bundle containing:
  - corrected baseline `pinn_ffno` by lineage, with source root recorded explicitly
  - fresh `pinn_ffno_depth24`
- [ ] Require side-by-side amplitude and phase values for:
  - MAE
  - MSE
  - PSNR
  - SSIM
  - MS-SSIM
  - FRC50
  - parameter count
  - runtime / provenance fields where available
- [ ] Write the durable summary with:
  - exact baseline source path
  - exact fresh depth-24 root
  - explicit statement that the baseline was reused by lineage unless a justified rerun was truly necessary
  - claim boundary that this is a CDI FFNO depth ablation only
  - explicit note that manuscript-facing promotion is deferred to `2026-05-06-cdi-lines128-ffno-depth24-final-paper-refresh`
- [ ] Update the evidence and study indexes so later paper/planning tasks can discover this ablation without rereading raw backlog docs.
- [ ] Keep paper-local FFNO tables and figures unchanged in this item. If any implementation step would force direct paper-table edits, stop and treat that as scope drift into the final refresh item.

**Verification:**
- Blocking:
  - the summary includes side-by-side baseline and depth-24 metrics for the required amplitude/phase metric family
  - the summary states whether the baseline was reused by lineage or rerun for a justified reason
  - `evidence_matrix.md`, `model_variant_index.json`, `ablation_index.json`, and `docs/studies/index.md` all point to the new summary or fresh root consistently
- Supporting:
  - `paper_evidence_index.md` is not a mandatory update here because this item does not itself promote new manuscript-facing assets; if implementation updates it anyway, keep the entry explicitly append-only and non-promoted

## Required Deterministic Checks

Run these as the required deterministic closeout checks for this item. Unless a stronger replacement is explicitly documented, they remain mandatory.

- Blocking before long-run launch and again at closeout if code changed:

```bash
python - <<'PY'
from pathlib import Path
required = [
    Path("docs/backlog/done/2026-04-29-cdi-lines128-paper-benchmark-execution.md"),
    Path("docs/plans/NEURIPS-HYBRID-RESNET-2026/lines128_paper_benchmark_preflight.md"),
    Path("docs/plans/NEURIPS-HYBRID-RESNET-2026/lines128_paper_benchmark_summary.md"),
    Path("docs/plans/NEURIPS-HYBRID-RESNET-2026/tables/cdi_lines128_metrics_extended.json"),
    Path("ptycho_torch/generators/ffno.py"),
    Path("ptycho_torch/generators/ffno_bottleneck.py"),
    Path("scripts/studies/grid_lines_torch_runner.py"),
    Path("scripts/studies/grid_lines_compare_wrapper.py"),
    Path("scripts/studies/lines128_paper_benchmark.py"),
]
missing = [str(path) for path in required if not path.exists()]
if missing:
    raise SystemExit(f"missing CDI FFNO depth-24 ablation inputs: {missing}")
print("CDI FFNO depth-24 ablation inputs present")
PY
```

- Blocking before long-run launch if code changed:

```bash
pytest -q tests/torch/test_generator_registry.py -k "ffno"
pytest -q tests/torch/test_grid_lines_torch_runner.py -k "ffno"
pytest -q tests/test_grid_lines_compare_wrapper.py -k "ffno"
python -m compileall -q ptycho_torch scripts/studies
```

- Blocking post-run item-local audit:
  - verify `pinn_ffno_depth24` reports `fno_blocks=24`
  - verify the reused corrected `pinn_ffno` baseline reports `fno_blocks=4` and `fno_cnn_blocks=0`
  - verify no fixed CDI contract field changed outside the allowed depth delta

## Completion Gate

- The fresh depth-24 row completes under the same corrected no-refiner `lines128` CDI contract as `pinn_ffno`, except for `fno_blocks=24`.
- The baseline comparator is the corrected four-block no-refiner `pinn_ffno` row, reused by lineage unless a justified artifact-integrity issue forces a rerun.
- The summary reports side-by-side amplitude and phase MAE, MSE, PSNR, SSIM, MS-SSIM, and FRC50 for the corrected four-block baseline and fresh depth-24 row.
- Parameter count and runtime/provenance are recorded so metric movement is not read without compute context.
- The result remains append-only ablation evidence and does not silently change the active paper-facing FFNO rows.
