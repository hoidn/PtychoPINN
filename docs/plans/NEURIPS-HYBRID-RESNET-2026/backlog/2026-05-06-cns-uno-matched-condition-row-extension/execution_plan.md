# CNS U-NO Matched-Condition Row Extension Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. Do not create worktrees.

**Goal:** Append one NeuralOperator U-NO comparator row to the existing PDEBench CNS matched-condition result set under the locked `history_len=5`, `512 / 64 / 64`, `40`-epoch capped contract, then republish a derived plus-U-NO table bundle by lineage without rerunning the existing four rows.

**Architecture:** Reuse the current PDEBench image-suite CNS runner and the existing external U-NO body contract, add only the minimum profile and adapter surface needed to instantiate a supervised CNS U-NO row, run exactly that row under the current matched-condition recipe, and extend the table-packaging path so the four existing rows remain lineage-owned while the new U-NO row is appended as bounded capped context. Preserve equal-footing and claim-boundary rules from the steering document, roadmap, and current matched-condition summary: same task, same split counts, same `history_len`, same epochs, same batch size, same MSE training loss, same normalization contract, and no CDI cross-comparison.

**Tech Stack:** Python, PyTorch, NeuralOperator `UNO`, `scripts/studies/pdebench_image128/*`, `scripts/studies/paper_results_refresh.py`, pytest, compileall, tmux in the `ptycho311` environment for the long CNS row run.

---

## Objective

- Add one CNS U-NO row, preferably `neuralop_uno_cns_base`, to the current matched-condition PDEBench `2d_cfd_cns` table.
- Publish a derived plus-U-NO bundle that keeps the current four matched-condition rows by lineage and appends the new U-NO row.
- State explicitly whether the derived table should replace the current manuscript CNS table or remain adjacent append-only context.

## Scope

- Fixed benchmark contract:
  - task: `2d_cfd_cns`
  - spatial size: `128x128`
  - `history_len=5`
  - split counts: `train=512`, `val=64`, `test=64`
  - `max_windows_per_trajectory=8`
  - `epochs=40`
  - `batch_size=4`
  - training loss: `mse`
  - metric family: `err_RMSE`, `err_nRMSE`, `relative_l2`, `fRMSE_low`, `fRMSE_mid`, `fRMSE_high`
  - claim boundary: `bounded_capped_decision_support_only`
- Reuse the current matched-condition authority from `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_cns_matched_condition_table_refresh_summary.md` and `docs/plans/NEURIPS-HYBRID-RESNET-2026/tables/pdebench_cns_matched_condition_metrics.json`.
- Use the existing NeuralOperator U-NO body as the architectural source of truth. A narrow PDEBench task-local wrapper is allowed only for channel mapping, shape adaptation, padding/cropping, or dependency/provenance capture.
- Run only the new U-NO row. Existing `author_ffno_cns_base`, `spectral_resnet_bottleneck_base`, `fno_base`, and `unet_strong` rows remain source-owned by lineage.

## Explicit Non-Goals

- Do not rerun FFNO, SRU-Net, FNO, or U-Net just to append U-NO.
- Do not reopen the CNS headline-table selection policy or mix contracts.
- Do not promote this item to full-training PDEBench evidence.
- Do not compare CNS U-NO results against CDI U-NO rows.
- Do not modify `ptycho/model.py`, `ptycho/diffsim.py`, or `ptycho/tf_helper.py`.

## Binding Constraints

- Steering:
  - keep equal-footing comparisons explicit
  - preserve approved metric, split, and protocol boundaries
  - do not relax fairness constraints for convenience
- Roadmap and design:
  - this is Phase 2 bounded capped CNS work, not a later-paper rewrite
  - the current matched-condition `history_len=5`, `512 / 64 / 64`, `40`-epoch table remains the baseline authority until the derived plus-U-NO bundle is published
  - any manuscript-facing recommendation must keep the claim boundary at `bounded_capped_decision_support_only`
- Repo workflow:
  - use tmux for the long run and activate `ptycho311`
  - keep long-running commands under implementation ownership until the tracked PID exits `0` or a recoverable failure has been diagnosed and rerun
  - never mark the item `BLOCKED` for a normal import failure, test failure, path issue, or harness bug before a narrow fix attempt and rerun
  - reserve `BLOCKED` for an unrecoverable NeuralOperator/U-NO compatibility failure, missing required external dependency that cannot be installed under current authority, missing required CNS data/artifacts, unavailable hardware, or a benchmark-contract change that would violate roadmap authority

## Prerequisite Status

- Progress ledger state:
  - initiative tranches `phase-0-evidence-inventory` and `phase-1-pde-benchmark-selection` are complete
  - no initiative tranches are currently marked blocked in `state/NEURIPS-HYBRID-RESNET-2026/progress_ledger.json`
- Item-specific prerequisites already expected to exist before implementation closes:
  - `docs/backlog/done/2026-04-30-cdi-lines128-uno-generator-integration.md`
  - `docs/backlog/done/2026-05-04-cns-matched-condition-table-refresh.md`
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_cns_matched_condition_table_refresh_summary.md`
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/tables/pdebench_cns_matched_condition_metrics.json`
  - `ptycho_torch/generators/neuralop_uno.py`
  - `scripts/studies/pdebench_image128/run_config.py`
  - `scripts/studies/pdebench_image128/models.py`
  - `scripts/studies/run_pdebench_image128_suite.py`

## File And Artifact Targets

### Mandatory contract outputs

- Row-local artifact root:
  - `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-06-cns-uno-matched-condition-row-extension/`
- Summary authority:
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_cns_uno_row_extension_summary.md`
- Derived paper-local table assets:
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/tables/pdebench_cns_matched_condition_metrics_plus_uno.json`
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/tables/pdebench_cns_matched_condition_metrics_plus_uno.csv`
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/tables/pdebench_cns_matched_condition_metrics_plus_uno.tex`
- Durable discoverability updates:
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/evidence_matrix.md`
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/paper_evidence_index.md`
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/model_variant_index.json`
  - `docs/studies/index.md`

### Likely code and test targets

- `scripts/studies/pdebench_image128/run_config.py`
- `scripts/studies/pdebench_image128/models.py`
- `scripts/studies/run_pdebench_image128_suite.py`
- `scripts/studies/paper_results_refresh.py`
- `tests/studies/test_pdebench_image128_models.py`
- `tests/studies/test_pdebench_image128_runner.py`
- `tests/studies/test_paper_results_refresh.py`

### Preferred packaging, not mandatory if the same contract is met more directly

- A narrow PDEBench-side U-NO helper module under `scripts/studies/pdebench_image128/` if `models.py` would otherwise become hard to maintain.
- Item-local lineage payloads such as:
  - `plus_uno_lineage.json`
  - `plus_uno_decision.json`
  - `plus_uno_row_manifest.json`
- If the actual run is emitted under an existing `.artifacts/work/...` runner convention, keep a stable pointer or lineage manifest under the mandatory item root instead of copying bulky artifacts.

## Implementation Architecture

- Unit 1: profile and builder contract
  - add or reuse a CNS-specific U-NO profile and make model construction deterministic, shape-safe, and provenance-rich
- Unit 2: single-row execution and row-local provenance
  - run only the U-NO row under the locked matched-condition CNS contract and emit the row-local artifacts needed for later lineage and review
- Unit 3: derived table packaging and durable indexes
  - append the U-NO row to the existing matched-condition table assets by lineage, then update the human and machine indexes that advertise the new bundle

## Task Checklist

### Task 1: Freeze The Matched-Condition U-NO Contract

**Files:**
- Read/confirm: `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_cns_matched_condition_table_refresh_summary.md`
- Read/confirm: `docs/plans/NEURIPS-HYBRID-RESNET-2026/tables/pdebench_cns_matched_condition_metrics.json`
- Read/confirm: `ptycho_torch/generators/neuralop_uno.py`
- Read/confirm: `scripts/studies/pdebench_image128/run_config.py`
- Read/confirm: `scripts/studies/pdebench_image128/models.py`

- [ ] Confirm the current matched-condition table has exactly four rows and that their fixed contract is the binding baseline for the plus-U-NO extension.
- [ ] Confirm whether the current PDEBench runner already has a U-NO-compatible profile or whether a new `neuralop_uno_cns_base` profile is required.
- [ ] Freeze the row-local manifest requirements before coding: input/output channel mapping, spatial divisibility/pad-crop policy, UNO mode counts, hidden width, layer/depth settings, dependency identity, and the exact row label to publish.
- [ ] If the existing CDI-side `ptycho_torch/generators/neuralop_uno.py` cannot be reused directly because its contract is locked to CDI `C=1 -> 2` real/imag output, record that the PDEBench implementation must wrap the same underlying UNO body rather than widening the CDI file’s contract.

**Blocking verification before code changes:**
- [ ] Run the required deterministic input-presence check unchanged.

### Task 2: Add Or Reuse A CNS U-NO Profile And Builder

**Files:**
- Modify: `scripts/studies/pdebench_image128/run_config.py`
- Modify: `scripts/studies/pdebench_image128/models.py`
- Modify if needed: `scripts/studies/run_pdebench_image128_suite.py`
- Test: `tests/studies/test_pdebench_image128_models.py`
- Test: `tests/studies/test_pdebench_image128_runner.py`
- Optional new helper: `scripts/studies/pdebench_image128/<uno_helper>.py`

- [ ] Add `neuralop_uno_cns_base` only if no existing profile already expresses the matched-condition CNS U-NO row cleanly.
- [ ] Keep the profile explicit about the bounded capped CNS contract and about every U-NO architectural setting needed for reproducibility.
- [ ] Build the model through the existing external U-NO body or a narrow PDEBench task-local wrapper around that same body.
- [ ] Fail closed with a structured blocker if the U-NO dependency surface is unavailable or if shape compatibility would require changing task semantics, target semantics, loss, or the locked matched-condition contract.
- [ ] Ensure `describe_model()` or the emitted profile manifest captures external-source provenance and the row-local configuration fields needed by the backlog item.

**Blocking verification before any long run:**
- [ ] `pytest -q tests/studies/test_pdebench_image128_models.py -k "uno or profile or cns"`
- [ ] `pytest -q tests/studies/test_pdebench_image128_runner.py -k "matched_condition or pdebench_cns or uno"`
- [ ] If `scripts/studies/paper_results_refresh.py` is touched in this tranche, also run `pytest -q tests/studies/test_paper_results_refresh.py -k "cns_matched_condition"` before launching the U-NO row.

### Task 3: Launch Exactly One CNS U-NO Row

**Files and artifacts:**
- Use: `scripts/studies/run_pdebench_image128_suite.py`
- Emit under the item root:
  - invocation/config/provenance files
  - row-local `metrics_*.json`
  - row-local `model_profile_*.json`
  - checkpoint/history files if the runner already writes them
  - a compact row manifest that names the fixed contract and dependency version

- [ ] Launch only the U-NO row under the locked `history_len=5`, `512 / 64 / 64`, `40`-epoch, batch-`4`, MSE matched-condition CNS contract.
- [ ] Use tmux and `ptycho311`, and track the exact launched PID until completion. Do not leave the run in an indeterminate background state.
- [ ] Reuse the existing CNS dataset/split/normalization contract exactly. Do not generate new splits and do not drift to a different cap, history length, or loss.
- [ ] If the first run fails for a narrow fixable reason, diagnose, patch narrowly, and rerun the same single-row launch. Do not close as blocked before that attempt.
- [ ] If U-NO is unrecoverably incompatible with the locked CNS contract, stop with a precise blocker summary and do not substitute another model family.

**Blocking verification before packaging:**
- [ ] Confirm the U-NO row finished with a clean tracked-run outcome and produced fresh metrics/provenance artifacts under the item-local lineage.
- [ ] Confirm the row manifest records the exact U-NO dependency identity and shape-adaptation policy.

### Task 4: Publish The Derived Plus-U-NO Table Bundle By Lineage

**Files:**
- Modify: `scripts/studies/paper_results_refresh.py`
- Test: `tests/studies/test_paper_results_refresh.py`
- Write item-local derived assets under the item root
- Write paper-local assets:
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/tables/pdebench_cns_matched_condition_metrics_plus_uno.json`
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/tables/pdebench_cns_matched_condition_metrics_plus_uno.csv`
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/tables/pdebench_cns_matched_condition_metrics_plus_uno.tex`

- [ ] Extend or parameterize the current matched-condition asset writer so it can emit a derived plus-U-NO bundle without overwriting the current four-row authority files by default.
- [ ] Preserve the four existing rows exactly by lineage, including their row IDs, labels, source run roots, row roles, and claim boundary.
- [ ] Append one U-NO row with manuscript label `U-NO` or equally explicit wording, and mark it as an appended comparator row rather than as a silently promoted authority row.
- [ ] Emit JSON/CSV/TeX plus lineage payloads that show the derived bundle contains four reused rows plus one fresh U-NO row under the same fixed contract.
- [ ] Ensure the summary states whether the manuscript should switch to the plus-U-NO table or keep it as adjacent context. The default assumption is adjacent append-only context unless the new summary explicitly justifies replacement.

**Blocking verification for packaging:**
- [ ] `pytest -q tests/studies/test_paper_results_refresh.py -k "cns_matched_condition"`
- [ ] Validate the derived JSON has exactly five rows: the original four row IDs plus one U-NO row under the same fixed contract.

### Task 5: Update Durable Discoverability Surfaces

**Files:**
- Write: `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_cns_uno_row_extension_summary.md`
- Modify: `docs/plans/NEURIPS-HYBRID-RESNET-2026/evidence_matrix.md`
- Modify: `docs/plans/NEURIPS-HYBRID-RESNET-2026/paper_evidence_index.md`
- Modify: `docs/plans/NEURIPS-HYBRID-RESNET-2026/model_variant_index.json`
- Modify: `docs/studies/index.md`

- [ ] Write the summary as the new execution authority outcome for this backlog item, including contract, emitted assets, row lineage, U-NO dependency/version, and the recommendation on manuscript use.
- [ ] Update `evidence_matrix.md` so the CNS matched-condition section advertises the plus-U-NO derivative as either the new manuscript-facing table or append-only adjacent context, whichever the summary concludes.
- [ ] Update `paper_evidence_index.md` with a new backlog row for this completed item and its tier/claim boundary.
- [ ] Add the U-NO row and derived table references to `model_variant_index.json` with clear source-summary and source-run-root lineage.
- [ ] Update `docs/studies/index.md` so the PDEBench image-suite study listing can discover the new summary and plus-U-NO table assets.

**Supporting verification:**
- [ ] Review the updated indexes for consistency of backlog item ID, summary path, artifact root, claim boundary, and current-authority wording.

## Required Deterministic Checks

These are mandatory final checks for this item unless a later approved plan replaces them with a strictly stronger contract. Keep them unchanged here.

- [ ] `python - <<'PY'
from pathlib import Path
required = [
    Path("docs/backlog/done/2026-04-30-cdi-lines128-uno-generator-integration.md"),
    Path("docs/backlog/done/2026-05-04-cns-matched-condition-table-refresh.md"),
    Path("docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_cns_matched_condition_table_refresh_summary.md"),
    Path("docs/plans/NEURIPS-HYBRID-RESNET-2026/tables/pdebench_cns_matched_condition_metrics.json"),
    Path("ptycho_torch/generators/neuralop_uno.py"),
    Path("scripts/studies/pdebench_image128/run_config.py"),
    Path("scripts/studies/pdebench_image128/models.py"),
    Path("scripts/studies/run_pdebench_image128_suite.py"),
]
missing = [str(path) for path in required if not path.exists()]
if missing:
    raise SystemExit(f"missing CNS U-NO row-extension inputs: {missing}")
print("CNS U-NO row-extension inputs present")
PY`
- [ ] `pytest -q tests/studies/test_pdebench_image128_models.py -k "uno or profile or cns"`
- [ ] `pytest -q tests/studies/test_pdebench_image128_runner.py -k "matched_condition or pdebench_cns or uno"`
- [ ] `python -m compileall -q ptycho_torch scripts/studies`

## Additional Final Checks

- [ ] If the packaging code changed, rerun `pytest -q tests/studies/test_paper_results_refresh.py -k "cns_matched_condition"`.
- [ ] Validate that the plus-U-NO JSON/CSV/TeX files agree on row order and row count.
- [ ] Validate that the derived bundle preserves the original four source run roots unchanged and adds exactly one U-NO source run root.
- [ ] Validate that the summary, evidence matrix, paper evidence index, studies index, and model variant index all use the same backlog item ID, summary path, claim boundary, and current-authority wording.

## Completion Criteria

- One U-NO CNS row exists under the same matched-condition contract as the current four-row table, or the item closes with a precise unrecoverable blocker explaining why that contract could not be preserved.
- The derived plus-U-NO bundle exists with five rows total and explicit lineage for the four existing rows.
- The current claim boundary remains bounded capped decision-support only.
- The summary explicitly answers whether the manuscript should replace `pdebench_cns_matched_condition_metrics.tex` with the plus-U-NO table or keep the plus-U-NO table as adjacent context.
- All required deterministic checks pass, plus any targeted packaging test needed for the touched code path.
