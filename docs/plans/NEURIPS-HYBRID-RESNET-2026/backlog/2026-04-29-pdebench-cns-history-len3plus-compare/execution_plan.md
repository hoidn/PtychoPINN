# PDEBench CNS History Length 3+ Compare Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Determine whether increasing the capped PDEBench `2d_cfd_cns` temporal-context contract from `history_len=2` to `history_len=3` improves the local four-row CNS comparison, and evaluate `history_len=4` only if a predeclared gate opens after the `history_len=3` evidence exists.

**Architecture:** Treat this as an audit-first, capped Phase 2 context-ablation lane. Reuse the existing CNS runner and history-delta reporting support, freeze the audited `history_len=2` anchors into an item-local manifest, inspect the derived `history_len=3/4` contracts, run the required four-row `history_len=3` pilots at `10` and `40` epochs, emit cross-history sidecars against the frozen anchors, and only then decide whether the optional `history_len=4` branch is scientifically justified.

**Tech Stack:** PATH `python`, PyTorch (POLICY-001), `scripts/studies/pdebench_image128/`, pytest, compileall, tmux with `ptycho311` for long runs, Markdown/JSON/CSV/PNG artifacts under `.artifacts/`

---

## Initiative

- ID: `NEURIPS-HYBRID-RESNET-2026`
- Backlog item: `2026-04-29-pdebench-cns-history-len3plus-compare`
- Plan authority date: `2026-04-29`
- Scope owner: Roadmap Phase 2 capped CNS follow-up lane
- Selected-item context: `state/NEURIPS-HYBRID-RESNET-2026/backlog_drain/iterations/4/items/2026-04-29-pdebench-cns-history-len3plus-compare/selected-item-context.md`
- Previous plan path from selected-item context: `docs/plans/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-pdebench-cns-history-len3plus-compare/execution_plan.md`
- Artifact root: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-history-len3plus-compare/`
- Durable summary target: `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_cns_history_len3plus_compare_summary.md`
- CNS summary target: `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_2d_cfd_cns_summary.md`

This document supersedes prior seed content for this backlog item and is the
execution authority for implementation.

## Inputs Read

- `AGENTS.md`
- `docs/index.md`
- `docs/findings.md`
- `docs/INITIATIVE_WORKFLOW_GUIDE.md`
- `docs/TESTING_GUIDE.md`
- `docs/steering.md`
- `docs/studies/index.md`
- `docs/plans/2026-04-20-neurips-hybrid-resnet-submission-design.md`
- `docs/plans/2026-04-20-neurips-hybrid-resnet-submission-roadmap.md`
- `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_128x128_image_suite_plan.md`
- `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_2d_cfd_cns_summary.md`
- `state/NEURIPS-HYBRID-RESNET-2026/progress_ledger.json`
- `state/NEURIPS-HYBRID-RESNET-2026/backlog_drain/iterations/4/items/2026-04-29-pdebench-cns-history-len3plus-compare/selected-item-context.md`
- `state/NEURIPS-HYBRID-RESNET-2026/backlog_drain/iterations/4/items/2026-04-29-pdebench-cns-history-len3plus-compare/plan-phase/plan_path.txt`
- `docs/plans/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-pdebench-cns-history-len3plus-compare/execution_plan.md` (background only)

## Selected Objective

- Answer the bounded scientific question: does increasing temporal context from
  `history_len=2` to `history_len=3` help the capped local CNS compare?
- Preserve the same equal-footing four-row compare surface used by the
  completed `history_len=1` ablation:
  - `spectral_resnet_bottleneck_base`
  - `hybrid_resnet_cns`
  - `fno_base`
  - `unet_strong`
- Compare fresh longer-context rows against frozen audited `history_len=2`
  anchors at matching `10`- and `40`-epoch budgets.
- Evaluate `history_len=4` only after the `history_len=3` compare payloads
  exist and only if the gate in this plan explicitly opens.

## Scope

- Keep the official CNS dataset, capped split family, MSE loss, normalization,
  batch-size policy, epoch budgets, and metric family fixed.
- Treat temporal context as the only intended task-contract delta:
  `history_len=2 -> history_len=3`, with derived input-channel and valid-window
  counts recorded explicitly from emitted artifacts.
- Reuse the existing runner and history-delta reporting support if they pass
  audit; do not churn code just because the plan is fresh.
- Reuse the already audited `history_len=2` anchors from the completed
  `history_len=1` compare after item-local manifest validation.
- Run required fresh `history_len=3` four-row pilots at `10` and `40` epochs.
- Run `history_len=4` only as a contingent branch behind a written gate.

## Explicit Non-Goals

- Do not widen this work into rollout or autoregressive evaluation.
- Do not widen this work into full-training benchmark rows, suite-level PDE
  claims, manuscript artifacts, or `/home/ollie/Documents/neurips/` outputs.
- Do not change dataset path, split counts, `max_windows_per_trajectory`,
  optimizer family, scheduler family, training loss, batch size, or metric
  family to make longer-history rows look better.
- Do not rerun `history_len=1`; it is prerequisite context only.
- Do not turn the optional `history_len=4` branch into an ungated automatic
  second study point.
- Do not widen into FFNO, GNOT, Darcy, SWE, hybrid-spectral architecture,
  spectral-modes work, physics regularization, or any unrelated backlog lane.
- Do not touch `ptycho/model.py`, `ptycho/diffsim.py`, or `ptycho/tf_helper.py`.
- Do not create worktrees.

## Steering, Roadmap, And Fairness Constraints

- Steering requires explicit equal-footing comparisons and forbids silently
  relaxing fairness constraints.
- The roadmap allows bounded Phase 2 capped CNS follow-ups, but they remain
  decision-support-only until full-training PDE benchmark gates are satisfied.
- The design and PDEBench suite plan keep this lane on the official
  `2d_cfd_cns` file and the fixed denormalized metric family; this item must
  not drift into a new dataset, baseline family, or evaluation protocol.
- The roadmap explicitly names this item as an allowed bounded CNS follow-up
  and requires the full four-row shell when the scope is a history-contract
  compare.
- The selected backlog item makes `history_len=3` mandatory and allows
  `history_len=4` only behind an explicit gate.
- The fixed equal-footing contract across fresh and reference rows is:
  - dataset:
    `/home/ollie/Documents/pdebench-data/2d_cfd_cns/2D_CFD_Rand_M1.0_Eta0.01_Zeta0.01_periodic_128_Train.hdf5`
  - split: `512 / 64 / 64` trajectories
  - `max_windows_per_trajectory=8`
  - batch size: `4`
  - training loss: `mse`
  - metric family:
    `err_RMSE`, `err_nRMSE`, `relative_l2`, `fRMSE_low`, `fRMSE_mid`,
    `fRMSE_high`
  - profiles:
    `spectral_resnet_bottleneck_base`, `hybrid_resnet_cns`, `fno_base`,
    `unet_strong`
- The only allowed scientific delta for the base compare is temporal-context
  length and its derived sample/input-channel contract:
  - reference: `history_len=2`, `concat u[t-2:t] -> u[t]`
  - fresh mandatory run: `history_len=3`, `concat u[t-3:t] -> u[t]`
  - optional gated run: `history_len=4`, `concat u[t-4:t] -> u[t]`
- Because longer history reduces eligible windows per trajectory, the durable
  summary must label the result as capped context-ablation evidence and must
  state the exact window-count deltas before making any ranking statement.

## Prerequisite Status

- Satisfied from current durable state:
  - the official `2d_cfd_cns` file is staged and checksum-verified
  - the CNS runner already supports positive `history_len` values and
    `inspect` / `pilot` modes
  - the canonical CNS Hybrid shell is fixed to `hybrid_resnet_cns` with
    skip-add plus `pixelshuffle`
  - the completed `history_len=1` compare already established the fixed rule
    that only `history_len` and its derived sample/input-channel contract may
    differ in a cross-history compare
  - the exact `history_len=2` anchor family already exists for all four rows at
    both `10` and `40` epochs, including the backfilled `40`-epoch
    `hybrid_resnet_cns` anchor
- Reusable frozen reference roots:
  - `10ep` spectral + hybrid:
    `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-ffno-bottleneck-cns-compare/readiness-cap-20260421T203200Z-ffno-cns-10ep`
  - `10ep` FNO + U-Net:
    `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-2d-cfd-cns/readiness-cap-20260421T183717Z-10ep-mse`
  - `40ep` spectral:
    `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-ffno-bottleneck-cns-compare/readiness-cap-20260421T221008Z-spectral40ep`
  - `40ep` hybrid:
    `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-markov-history1-compare/history2-hybrid-cns-pilot-40ep-20260423T223143Z`
  - `40ep` FNO + U-Net:
    `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-2d-cfd-cns/readiness-cap-20260421T185926Z-40ep-mse`
- Not prerequisites for this item:
  - any new external baseline
  - full-training benchmark-complete CNS rows
  - manuscript-assembly work under `/home/ollie/Documents/neurips/`

## Implementation Architecture

- **Compare-contract unit:** `scripts/studies/pdebench_image128/reporting.py`
  and `tests/studies/test_pdebench_image128_runner.py` own the reference-manifest
  format, the history-delta compare payload, row-family labels, and the
  invariant that fixed-contract fields remain equal while history-derived
  fields may vary.
- **Execution unit:** `scripts/studies/pdebench_image128/cfd_cns.py` owns
  emitted history metadata, split manifests, inspect mode, and the four-row
  pilot runs. Reuse the current parameterized path rather than adding a new
  runner.
- **Interpretation unit:** the new longer-history summary, the CNS summary
  update, `docs/index.md`, `docs/studies/index.md`, and the progress ledger
  own the durable claim boundary. They must separate “more context changed
  metrics” from “more context also changed eligible-window counts.”

## Concrete File And Artifact Targets

### Code And Test Surfaces

- Audit and modify only if the pre-run checks expose a real contract gap:
  - `scripts/studies/pdebench_image128/reporting.py`
  - `tests/studies/test_pdebench_image128_runner.py`
- Modify only if inspect or pilot artifacts show missing history metadata:
  - `scripts/studies/pdebench_image128/cfd_cns.py`
  - `tests/studies/test_pdebench_cfd_cns_data.py`
- Verification-only reuse:
  - `tests/studies/test_pdebench_cfd_cns_metrics.py`

### Durable Docs And State

- Create: `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_cns_history_len3plus_compare_summary.md`
- Modify: `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_2d_cfd_cns_summary.md`
- Modify: `docs/index.md`
- Modify: `docs/studies/index.md`
- Modify: `state/NEURIPS-HYBRID-RESNET-2026/progress_ledger.json`
- Modify only if the result becomes a stable reusable rule rather than a
  summary-local conclusion: `docs/findings.md`

### Required Artifacts

- Create study root:
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-history-len3plus-compare/`
- Create item-local frozen manifest:
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-history-len3plus-compare/history2_reference_runs.json`
- Create inspect roots:
  - `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-history-len3plus-compare/history3-inspect-<timestamp>/`
  - `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-history-len3plus-compare/history4-inspect-<timestamp>/`
- Create required fresh run roots:
  - `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-history-len3plus-compare/history3-pilot-10ep-<timestamp>/`
  - `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-history-len3plus-compare/history3-pilot-40ep-<timestamp>/`
- Create required compare sidecars:
  - `compare_10ep_history3_against_history2.json`
  - `compare_10ep_history3_against_history2.csv`
  - `compare_40ep_history3_against_history2.json`
  - `compare_40ep_history3_against_history2.csv`
  - sample gallery PNGs only if targets align
- Create gate record:
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-history-len3plus-compare/history4_gate_decision.json`
- Create optional gated artifacts only if the gate opens:
  - `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-history-len3plus-compare/history4-pilot-10ep-<timestamp>/`
  - `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-history-len3plus-compare/history4-pilot-40ep-<timestamp>/`
  - `compare_10ep_history4_against_history2.json`
  - `compare_10ep_history4_against_history2.csv`
  - `compare_40ep_history4_against_history2.json`
  - `compare_40ep_history4_against_history2.csv`

## Execution Guardrails

- The selected backlog item’s required deterministic checks stay mandatory.
  Any focused selector in this plan is additive, not a replacement.
- Expensive training must wait for a green focused runner/reporting selector and
  then a green run of the required backlog-item checks.
- If a test, import, path, or harness failure occurs, diagnose, fix, and rerun
  before considering the item blocked.
- Reserve `BLOCKED` for missing resources, unavailable hardware, roadmap
  conflict, user decision required, external dependency outside current
  authority, or an unrecoverable failure after a documented narrow fix attempt.
- For long runs, use tmux, activate `ptycho311`, track the exact launched PID,
  avoid duplicate launches into the same output root, and require both exit
  code `0` and fresh required artifacts before treating a run as complete.
- This lane is capped decision-support evidence only. No result from this plan
  may be promoted to benchmark-complete PDE evidence.

## Required Deterministic Checks

The selected backlog item requires these unchanged checks:

```bash
pytest -q tests/studies/test_pdebench_image128_runner.py tests/studies/test_pdebench_cfd_cns_data.py tests/studies/test_pdebench_cfd_cns_metrics.py
python -m compileall -q scripts/studies/pdebench_image128 scripts/studies/run_pdebench_image128_suite.py
```

Run this focused selector before any expensive pilot:

```bash
pytest tests/studies/test_pdebench_image128_runner.py -k 'history or reference_run_manifest or cross_run_compare' -v
```

Archive the passing pytest log used to justify completion under the active
artifact root or a linked evidence location, per `docs/TESTING_GUIDE.md`.

## Task 1: Audit Current Cross-History Support And Patch Only If Needed

**Files:**
- Audit: `scripts/studies/pdebench_image128/reporting.py`
- Audit: `tests/studies/test_pdebench_image128_runner.py`
- Modify only if the audit shows a real gap

- [ ] Confirm the current helper and tests already support the required
  longer-context behavior:
  - fixed-equality fields still match exactly
  - allowed differences remain limited to `history_len`, derived
    `sample_contract`, and derived `input_channels`
  - fresh rows may have either smaller or larger `history_len` than the
    reference rows
  - row-family labels derive from actual history lengths rather than hard-coded
    `history1` labels
  - `hybrid_resnet_base` is still rejected as a proxy anchor for
    `hybrid_resnet_cns`
- [ ] If the focused selector already passes and the surfaces are adequate,
  record this task as audit-only and do not edit production files.
- [ ] If the focused selector fails or the audit finds a gap, tighten the tests
  first where needed, make the smallest justified patch, and rerun until green.

**Verification for Task 1**

```bash
pytest tests/studies/test_pdebench_image128_runner.py -k 'history or reference_run_manifest or cross_run_compare' -v
python -m compileall -q scripts/studies/pdebench_image128 scripts/studies/run_pdebench_image128_suite.py
pytest -q tests/studies/test_pdebench_image128_runner.py tests/studies/test_pdebench_cfd_cns_data.py tests/studies/test_pdebench_cfd_cns_metrics.py
```

## Task 2: Freeze The `history_len=2` Anchors And Inspect The Longer-Context Contracts

**Files:**
- Artifact-only work unless Task 1 or the inspect outputs expose a metadata gap

- [ ] Build `history2_reference_runs.json` under the item artifact root using
  the audited `10`- and `40`-epoch history-2 anchor rows for all four required
  profiles.
- [ ] Confirm every reused reference row contains the required artifacts:
  `invocation.json`, `dataset_manifest.json`, `split_manifest.json`,
  `comparison_summary.json`, `metrics_<profile>.json`,
  `model_profile_<profile>.json`.
- [ ] Run `inspect` mode for `history_len=3` and `history_len=4` into fresh
  output roots so the plan records the derived contract before training.
- [ ] Verify the inspect artifacts show the expected input-channel and
  supervision-window deltas:
  - `history_len=3` -> `input_channels=12`
  - `history_len=4` -> `input_channels=16`
  - valid-window counts shrink relative to `history_len=2`

**Execution notes for Task 2**

Reference-manifest creation should use the existing reporting helper rather than
handwritten JSON. Inspect commands:

```bash
python scripts/studies/pdebench_image128/cfd_cns.py \
  --mode inspect \
  --data-root /home/ollie/Documents/pdebench-data \
  --output-root .artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-history-len3plus-compare/history3-inspect-<timestamp> \
  --history-len 3

python scripts/studies/pdebench_image128/cfd_cns.py \
  --mode inspect \
  --data-root /home/ollie/Documents/pdebench-data \
  --output-root .artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-history-len3plus-compare/history4-inspect-<timestamp> \
  --history-len 4
```

**Verification for Task 2**

```bash
python - <<'PY'
import json
from pathlib import Path

root = Path(".artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-history-len3plus-compare")
manifest = json.loads((root / "history2_reference_runs.json").read_text())
assert manifest["history_len"] == 2
assert sorted(manifest["required_rows"]) == ["10ep", "40ep"]
for stem, expected_history, expected_channels in [
    ("history3-inspect-", 3, 12),
    ("history4-inspect-", 4, 16),
]:
    latest = sorted(root.glob(f"{stem}*/dataset_manifest.json"))[-1]
    payload = json.loads(latest.read_text())
    assert int(payload["history_len"]) == expected_history
    assert int(payload["input_channels"]) == expected_channels
print("reference manifest and inspect contracts verified")
PY
```

## Task 3: Run The Mandatory `history_len=3` Four-Row Pilots And Emit Cross-History Sidecars

**Files:**
- No code changes unless a real runner/reporting defect appears

- [ ] Only after Task 1 is green, launch the required four-row `history_len=3`
  `10`-epoch pilot with:
  - profiles:
    `spectral_resnet_bottleneck_base,hybrid_resnet_cns,fno_base,unet_strong`
  - `512 / 64 / 64` trajectories
  - `max_windows_per_trajectory=8`
  - `batch_size=4`
  - `training_loss=mse`
- [ ] Launch the matching `40`-epoch pilot on the same contract.
- [ ] After each run completes cleanly, write the history-delta compare payload
  against the frozen `history_len=2` anchors using the existing reporting
  helper, not ad hoc spreadsheets.
- [ ] If any pilot fails because of a normal harness or code issue, diagnose,
  fix, rerun, and document the narrow fix attempt before considering a blocker.

**Execution notes for Task 3**

Use tmux plus `ptycho311` for these runs. Example commands:

```bash
python scripts/studies/pdebench_image128/cfd_cns.py \
  --mode pilot \
  --data-root /home/ollie/Documents/pdebench-data \
  --output-root .artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-history-len3plus-compare/history3-pilot-10ep-<timestamp> \
  --profiles spectral_resnet_bottleneck_base,hybrid_resnet_cns,fno_base,unet_strong \
  --history-len 3 \
  --epochs 10 \
  --batch-size 4 \
  --max-train-trajectories 512 \
  --max-val-trajectories 64 \
  --max-test-trajectories 64 \
  --max-windows-per-trajectory 8

python scripts/studies/pdebench_image128/cfd_cns.py \
  --mode pilot \
  --data-root /home/ollie/Documents/pdebench-data \
  --output-root .artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-history-len3plus-compare/history3-pilot-40ep-<timestamp> \
  --profiles spectral_resnet_bottleneck_base,hybrid_resnet_cns,fno_base,unet_strong \
  --history-len 3 \
  --epochs 40 \
  --batch-size 4 \
  --max-train-trajectories 512 \
  --max-val-trajectories 64 \
  --max-test-trajectories 64 \
  --max-windows-per-trajectory 8
```

Then emit `compare_10ep_history3_against_history2.*` and
`compare_40ep_history3_against_history2.*` via
`write_history_delta_compare(...)`.

**Verification for Task 3**

```bash
python - <<'PY'
import json
from pathlib import Path

root = Path(".artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-history-len3plus-compare")
for name in [
    "compare_10ep_history3_against_history2.json",
    "compare_40ep_history3_against_history2.json",
]:
    payload = json.loads((root / name).read_text())
    assert payload["evidence_scope"] == "capped_decision_support_only"
    assert payload["allowed_contract_delta"]["fresh_history_len"] == 3
    assert payload["allowed_contract_delta"]["reference_history_len"] == 2
    assert payload["row_family_labels"]["fresh"] == "fresh_history3"
    assert payload["row_family_labels"]["reference"] == "reference_history2"
print("history3 compare payloads verified")
PY
```

## Task 4: Open Or Close The Optional `history_len=4` Gate

**Files:**
- Artifact-only unless the gate exposes a reporting defect

- [ ] Write `history4_gate_decision.json` after inspecting the `history_len=3`
  compare payloads.
- [ ] Open the gate only if one of these holds:
  - the `history_len=3` spectral row improves aggregate error without
    worsening `fRMSE_high` against the matching `history_len=2` anchor, or
  - implementation records a concrete scientific reason why a second
    longer-context point is necessary before interpreting the lane
- [ ] If the gate stays closed, record the exact reason and stop the study at
  `history_len=3` without marking the item blocked.
- [ ] If the gate opens, run the same `10`- and `40`-epoch four-row pilots for
  `history_len=4` on the unchanged capped contract and emit the matching
  history-delta compare sidecars against the same frozen `history_len=2`
  anchors.

**Verification for Task 4**

```bash
python - <<'PY'
import json
from pathlib import Path

root = Path(".artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-history-len3plus-compare")
gate = json.loads((root / "history4_gate_decision.json").read_text())
assert gate["decision"] in {"open", "closed"}
if gate["decision"] == "open":
    for name in [
        "compare_10ep_history4_against_history2.json",
        "compare_40ep_history4_against_history2.json",
    ]:
        payload = json.loads((root / name).read_text())
        assert payload["allowed_contract_delta"]["fresh_history_len"] == 4
print("history4 gate artifacts verified")
PY
```

## Task 5: Write Durable Interpretation And Update Discoverability

**Files:**
- Create: `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_cns_history_len3plus_compare_summary.md`
- Modify: `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_2d_cfd_cns_summary.md`
- Modify: `docs/index.md`
- Modify: `docs/studies/index.md`
- Modify: `state/NEURIPS-HYBRID-RESNET-2026/progress_ledger.json`
- Modify `docs/findings.md` only if the result becomes a reusable rule rather
  than a study-local interpretation

- [ ] Write the new summary as the execution lane’s durable authority.
- [ ] State the exact sample/window contract change caused by longer history
  before any ranking statement.
- [ ] Label the result as capped context-ablation evidence, not benchmark
  performance.
- [ ] Update the CNS summary and study indexes so the new summary is the
  discoverable source for this lane.
- [ ] Update the progress ledger with:
  - completion state for this backlog item
  - run roots and compare payload paths
  - `history4` gate outcome
  - evidence scope `capped_decision_support_only`
  - any blocker or follow-on note only if one remains after the narrow fix
    attempts required above

**Verification for Task 5**

```bash
python - <<'PY'
from pathlib import Path

required = [
    Path("docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_cns_history_len3plus_compare_summary.md"),
    Path("docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_2d_cfd_cns_summary.md"),
    Path("docs/index.md"),
    Path("docs/studies/index.md"),
    Path("state/NEURIPS-HYBRID-RESNET-2026/progress_ledger.json"),
]
missing = [str(path) for path in required if not path.exists()]
if missing:
    raise SystemExit(f"missing durable outputs: {missing}")
print("durable docs/state outputs present")
PY
```

## Completion Standard

- [ ] Required deterministic checks are green and archived.
- [ ] `history_len=2` anchors are frozen in an item-local manifest.
- [ ] `history_len=3` inspect and both required pilots completed with fresh
  artifacts and history-delta compare sidecars.
- [ ] `history4` is either explicitly closed with a written gate record or
  completed under the unchanged capped contract with matching compare sidecars.
- [ ] The durable summary and discovery surfaces clearly state the equal-footing
  boundaries, the window-count delta, and the decision-support-only evidence
  scope.
