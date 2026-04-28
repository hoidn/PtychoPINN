# PDEBench CNS FFNO Convolutional Features Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add one bounded FFNO-family CNS variant with explicit local convolutional features, compare it fairly against the authored FFNO row, the local FFNO-close row, and the pinned Hybrid-spectral anchor on the fixed capped `2d_cfd_cns` contract, then publish durable repo-local interpretation without widening into a benchmark-complete claim.

**Architecture:** Reuse the existing supervised CNS runner and canonical capped contract. Implement exactly one new FFNO-close derivative that changes only the bottleneck internals by adding a per-block local convolutional branch, keep the outer shell fixed, backfill the missing local FFNO-close `40`-epoch anchor if it is still absent, and collate fresh `10`/`40`-epoch cross-run compares plus durable summary/state updates. All outputs remain capped decision-support evidence only.

**Tech Stack:** PATH `python`; long runs in tmux with `ptycho311`; PyTorch/Lightning; `ptycho_torch/generators/`; `scripts/studies/pdebench_image128/`; pytest; compileall; Markdown/JSON/CSV/PNG artifacts under `.artifacts/`.

---

## Initiative

- ID: `NEURIPS-HYBRID-RESNET-2026`
- Backlog item: `2026-04-27-pdebench-ffno-convolutional-features-cns`
- Status: pending
- Date: `2026-04-28`
- Selected-item authority: `state/NEURIPS-HYBRID-RESNET-2026/backlog_drain/iterations/16/items/2026-04-27-pdebench-ffno-convolutional-features-cns/selected-item-context.md`
- Plan path authority: `docs/plans/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-27-pdebench-ffno-convolutional-features-cns/execution_plan.md`
- Durable summary target: `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_ffno_convolutional_features_cns_summary.md`
- CNS summary sync target: `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_2d_cfd_cns_summary.md`
- Artifact root: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-ffno-convolutional-features-cns/`

This document supersedes earlier revisions at this path and is the new execution authority for this backlog item. Implementation should rely on this plan plus the approved design, not on older queue notes or the raw backlog file.

## Inputs Read

- Consumed steering: `docs/steering.md`
- Consumed design: `docs/plans/2026-04-20-neurips-hybrid-resnet-submission-design.md`
- Consumed roadmap: `docs/plans/2026-04-20-neurips-hybrid-resnet-submission-roadmap.md`
- Consumed selected-item context: `state/NEURIPS-HYBRID-RESNET-2026/backlog_drain/iterations/16/items/2026-04-27-pdebench-ffno-convolutional-features-cns/selected-item-context.md`
- Consumed progress ledger: `state/NEURIPS-HYBRID-RESNET-2026/progress_ledger.json`
- `AGENTS.md`
- `docs/index.md`
- `docs/findings.md`
- `docs/INITIATIVE_WORKFLOW_GUIDE.md`
- `docs/DEVELOPER_GUIDE.md`
- `docs/TESTING_GUIDE.md`
- `docs/COMMANDS_REFERENCE.md`
- `docs/model_baselines.md`
- `docs/studies/index.md`
- `docs/backlog/index.md`
- `docs/backlog/in_progress/2026-04-27-pdebench-ffno-convolutional-features-cns.md`
- `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_128x128_image_suite_plan.md`
- Background only:
  - prior plan revision at this same path
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_ffno_bottleneck_summary.md`
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_author_ffno_equal_footing_summary.md`
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_gnot_cns_compare_summary.md`

## Selected Objective

- Add one explicit local-convolution FFNO-family CNS row that stays inside the existing FFNO-close shell and answers a narrow question: does adding local convolutional features inside the FFNO bottleneck materially improve capped `2d_cfd_cns` performance?
- Compare the new row against the required same-contract references:
  - `author_ffno_cns_base`
  - `ffno_bottleneck_base`
  - `spectral_resnet_bottleneck_base`
- Record the result in terms of both aggregate denormalized error and high-frequency behavior, with `relative_l2` and `fRMSE_high` called out explicitly in the durable summary.
- Repair the only missing fairness prerequisite discovered during planning: there is no authoritative local `ffno_bottleneck_base` `40`-epoch anchor yet, so this plan includes a bounded `40`-epoch backfill for that row before the `40`-epoch comparison.

## Scope

- Task remains **CNS only**:
  - task: `2d_cfd_cns`
  - dataset: `/home/ollie/Documents/pdebench-data/2d_cfd_cns/2D_CFD_Rand_M1.0_Eta0.01_Zeta0.01_periodic_128_Train.hdf5`
  - resolution: `128x128`
- Fixed capped contract for all new runs in this item:
  - split caps: `512 / 64 / 64` trajectories
  - `max_windows_per_trajectory=8`
  - `history_len=2`
  - batch size `4`
  - training loss `mse`
  - metrics: `err_RMSE`, `err_nRMSE`, `relative_l2`, `fRMSE_low`, `fRMSE_mid`, `fRMSE_high`
- Fresh execution sequence:
  - inspect snapshot under this item’s artifact root
  - fresh `10`-epoch run for the new local-conv FFNO variant
  - fresh `40`-epoch backfill for `ffno_bottleneck_base` if no valid same-contract anchor is already present
  - fresh `40`-epoch run for the new local-conv FFNO variant
  - cross-run compare sidecars for `10` and `40` epochs
- Chosen architecture for this backlog item:
  - add exactly one new profile: `ffno_bottleneck_localconv_base`
  - keep the same supervised shell as `ffno_bottleneck_base`
  - keep the same FFNO spectral path, block count, hidden width, downsampling depth, skip style, and output head
  - change only the bottleneck block internals by adding an explicit local `3x3` convolutional residual branch per block before the final projection/residual merge
  - keep the local-conv addition explicit in profile naming and model-profile provenance

## Explicit Non-Goals

- Do not widen this item into CDI, ptychography, SWE, Darcy, OpenFWI, or `/home/ollie/Documents/neurips/` artifact assembly.
- Do not turn this into a broad FFNO architecture sweep. This plan does **not** include:
  - convolutional stems
  - decoder-side refinement heads
  - multiple local-feature variants
  - hyperparameter sweeps over local-conv kernel size, gating, or norm style
- Do not reopen separate CNS lanes:
  - `history_len=1`
  - spectral modes `32/32`
  - Hybrid-spectral sharing/depth ablations
  - GNOT
  - physics regularization
- Do not change split, normalization, epoch budgets, batch size, metric family, or training loss to make the new row look better.
- Do not relabel capped results as full-training benchmark evidence, suite-complete evidence, or paper-facing competitiveness evidence.
- Do not create worktrees.
- Do not touch `ptycho/model.py`, `ptycho/diffsim.py`, or `ptycho/tf_helper.py`.
- Do not refactor the runner broadly. Patch only the smallest blocker required by the deterministic checks or the compare helper.

## Steering, Roadmap, And Fairness Constraints

- Preserve the approved roadmap phase order and evidence boundaries. This item strengthens the Phase 2 CNS evidence story only; it does not satisfy the roadmap’s full-training benchmark gate.
- Equal footing is mandatory. Within each compare bucket, dataset file, split caps, `history_len`, `max_windows_per_trajectory`, batch size, epoch budget, training loss, and metric family must stay fixed across compared rows.
- The steering document requires the approved comparison standards to remain explicit. If the `40`-epoch compare cannot include a same-contract local FFNO-close anchor, record that as a blocker and backfill it; do not silently compare the new `40`-epoch row against the old `10`-epoch FFNO-close row.
- The selected-item notes are binding:
  - this is an FFNO-family extension, not a Hybrid-spectral ablation
  - do not change split, loss, normalization, epoch budget, or metrics
  - report both aggregate `relative_l2` and high-frequency behavior
- The design and roadmap require capped/pilot outputs to remain decision-support only. This item may rank rows only inside this bounded capped lane for local interpretation.
- Long-running commands must run in tmux with `ptycho311` active, track the exact launched PID, and count as complete only when:
  - the tracked PID exits `0`
  - required output artifacts for that step exist and are freshly written
- Follow `REPORTING-ARTIFACT-BOUNDARY-001`: launcher exit status plus required metrics/manifests decide success. Optional galleries may warn, but they must not by themselves flip a successful run into failure.
- Use PATH `python` per interpreter policy.

## Prerequisite Status

### Satisfied

- The selected-item context marks this item as having no active backlog prerequisite.
- The official CNS file is already staged and verified.
- The supervised CNS runner, normalization path, metric path, and compare helper already exist.
- The required existing reference evidence is available for `10` epochs:
  - `author_ffno_cns_base`: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-author-ffno-equal-footing-cns/author-ffno-10ep-20260422T232119Z`
  - `ffno_bottleneck_base`: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-ffno-bottleneck-cns-compare/readiness-cap-20260421T203200Z-ffno-cns-10ep`
  - `spectral_resnet_bottleneck_base`: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-ffno-bottleneck-cns-compare/readiness-cap-20260421T203200Z-ffno-cns-10ep`
- The required existing reference evidence is available for `40` epochs for:
  - `author_ffno_cns_base`: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-author-ffno-equal-footing-cns/author-ffno-40ep-20260422T234340Z`
  - `spectral_resnet_bottleneck_base`: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-ffno-bottleneck-cns-compare/readiness-cap-20260421T221008Z-spectral40ep`
- Existing study tests already cover the model-profile and runner surfaces named in the backlog item’s required check commands.

### Open But Not Blocking

- No authoritative fresh run roots exist yet for `ffno_bottleneck_localconv_base`.
- There is no authoritative same-contract `40`-epoch `ffno_bottleneck_base` run root yet. This is a fairness prerequisite for the `40`-epoch compare and must be created inside this item unless an existing valid root is discovered during execution.
- No durable summary, CNS summary sync, or progress-ledger completion entry exists yet for this backlog item.

## Implementation Architecture

- **Model/Profile Unit:** `ptycho_torch/generators/ffno_bottleneck.py`, `scripts/studies/pdebench_image128/models.py`, and `scripts/studies/pdebench_image128/run_config.py` own the new local-conv variant. The shell must stay identical to `ffno_bottleneck_base`; only the bottleneck internals and provenance fields may change.
- **Verification/Compare Unit:** `tests/torch/test_ffno_bottleneck.py`, `tests/studies/test_pdebench_image128_models.py`, `tests/studies/test_pdebench_image128_runner.py`, and optionally `scripts/studies/pdebench_image128/reporting.py` own the regression surface. The compare helper should remain generic; patch it only if the new lane exposes a real blocker.
- **Execution/Docs/State Unit:** `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-ffno-convolutional-features-cns/`, `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_ffno_convolutional_features_cns_summary.md`, `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_2d_cfd_cns_summary.md`, `docs/studies/index.md`, `docs/index.md`, `state/NEURIPS-HYBRID-RESNET-2026/progress_ledger.json`, and optionally `docs/findings.md` own the durable interpretation and queue state.

## Concrete File And Artifact Targets

### Repo Surfaces Likely To Change

- `ptycho_torch/generators/ffno_bottleneck.py`
- `scripts/studies/pdebench_image128/models.py`
- `scripts/studies/pdebench_image128/run_config.py`
- `tests/torch/test_ffno_bottleneck.py`
- `tests/studies/test_pdebench_image128_models.py`
- `tests/studies/test_pdebench_image128_runner.py`
- Only if a real compare-surface blocker appears:
  - `scripts/studies/pdebench_image128/reporting.py`
- Expected durable docs/state updates at close:
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_ffno_convolutional_features_cns_summary.md`
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_2d_cfd_cns_summary.md`
  - `docs/studies/index.md`
  - `docs/index.md`
  - `state/NEURIPS-HYBRID-RESNET-2026/progress_ledger.json`
  - `docs/findings.md` only if the result creates a reusable rule worth promoting

### Required Generated Artifacts

- Study root: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-ffno-convolutional-features-cns/`
- Required run roots:
  - `inspect-<timestamp>/`
  - `cns-ffno-localconv-10ep-<timestamp>/`
  - `cns-ffno-close-backfill-40ep-<timestamp>/` only if no valid existing `40`-epoch FFNO-close root is found
  - `cns-ffno-localconv-40ep-<timestamp>/`
- Required manifests and compares:
  - `reference_runs_10ep.json`
  - `reference_runs_40ep.json`
  - `compare_10ep_against_existing.json`
  - `compare_10ep_against_existing.csv`
  - `compare_40ep_against_existing.json`
  - `compare_40ep_against_existing.csv`
- Optional but expected when target alignment succeeds:
  - `compare_10ep_sample0.png`
  - `compare_10ep_sample0_error.png`
  - `compare_40ep_sample0.png`
  - `compare_40ep_sample0_error.png`

## Fixed Existing Reference Rows

### `10`-Epoch Required References

- `author_ffno_cns_base` from `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-author-ffno-equal-footing-cns/author-ffno-10ep-20260422T232119Z`
- `ffno_bottleneck_base` from `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-ffno-bottleneck-cns-compare/readiness-cap-20260421T203200Z-ffno-cns-10ep`
- `spectral_resnet_bottleneck_base` from `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-ffno-bottleneck-cns-compare/readiness-cap-20260421T203200Z-ffno-cns-10ep`

### `10`-Epoch Optional Continuity Rows

- `hybrid_resnet_cns` from `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-ffno-bottleneck-cns-compare/readiness-cap-20260421T203200Z-ffno-cns-10ep`
- `fno_base`, `unet_strong`, and optionally `hybrid_resnet_base` from `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-2d-cfd-cns/readiness-cap-20260421T183717Z-10ep-mse`

### `40`-Epoch Required References

- `author_ffno_cns_base` from `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-author-ffno-equal-footing-cns/author-ffno-40ep-20260422T234340Z`
- `spectral_resnet_bottleneck_base` from `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-ffno-bottleneck-cns-compare/readiness-cap-20260421T221008Z-spectral40ep`
- `ffno_bottleneck_base` from a fresh same-contract backfill produced by this item unless a valid existing `40`-epoch root is discovered during execution

### `40`-Epoch Optional Continuity Rows

- `fno_base`, `unet_strong`, and optionally `hybrid_resnet_base` from `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-2d-cfd-cns/readiness-cap-20260421T185926Z-40ep-mse`

## Task 1: Add The Bounded Local-Convolution FFNO Variant

**Files:**
- Modify: `ptycho_torch/generators/ffno_bottleneck.py`
- Modify: `scripts/studies/pdebench_image128/models.py`
- Modify: `scripts/studies/pdebench_image128/run_config.py`
- Modify: `tests/torch/test_ffno_bottleneck.py`
- Modify: `tests/studies/test_pdebench_image128_models.py`

- [ ] Add failing tests first for the new profile and bottleneck behavior:
  - the new profile resolves through `get_model_profile`
  - the new model builds through `build_model_from_profile`
  - the local-conv bottleneck preserves shape
  - the local-conv path is explicit in the module structure and model-profile JSON
- [ ] Register `ffno_bottleneck_localconv_base` in `run_config.py` with the same shell fields as `ffno_bottleneck_base` and explicit provenance markers for the local-conv addition.
- [ ] Implement the new bottleneck path in `ffno_bottleneck.py` and wire it through `models.py`.
- [ ] Keep all non-bottleneck shell settings fixed:
  - `hidden_channels=32`
  - `fno_modes=12`
  - `fno_blocks=4`
  - `hybrid_downsample_steps=2`
  - `hybrid_skip_connections=True`
  - `hybrid_skip_style="add"`
  - `ffno_bottleneck_blocks=6`
  - `ffno_bottleneck_modes=12`
  - `ffno_bottleneck_share_weights=True`
  - `ffno_bottleneck_mlp_ratio=2.0`
  - `ffno_bottleneck_gate_init=0.1`
  - `ffno_bottleneck_norm="instance"`
- [ ] Do not introduce a second variant in this backlog item unless the chosen local-conv path is provably impossible without breaking the fixed shell; if that happens, stop and update the plan rather than improvising a wider sweep.

**Verification for Task 1**

- [ ] `pytest -q tests/torch/test_ffno_bottleneck.py`
- [ ] `pytest -q tests/studies/test_pdebench_image128_models.py`

## Task 2: Pass The Required Deterministic Checks Before Any Fresh Runs

**Files:**
- Modify only as needed from Task 1
- Modify only if runner/reporting tests expose a blocker: `tests/studies/test_pdebench_image128_runner.py`, `scripts/studies/pdebench_image128/reporting.py`

- [ ] Extend runner tests only as far as needed to prove the new lane can be collated by the generic cross-run compare helper and that its `model_profile_*.json` records the local-conv provenance fields.
- [ ] Run the backlog item’s required deterministic checks unchanged:

```bash
pytest -q tests/studies/test_pdebench_image128_models.py tests/studies/test_pdebench_image128_runner.py
python -m compileall -q scripts/studies/pdebench_image128
```

- [ ] Run the stronger local generator compile gate after code changes:

```bash
python -m compileall -q ptycho_torch/generators/ffno_bottleneck.py
```

- [ ] Do not launch new study runs until these checks pass.

## Task 3: Freeze The Local Contract Snapshot And Run The Fresh `10`-Epoch Variant

**Files:**
- Output only: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-ffno-convolutional-features-cns/`

- [ ] Create a fresh inspect snapshot for this item’s artifact root:

```bash
python scripts/studies/run_pdebench_image128_suite.py \
  --task 2d_cfd_cns \
  --mode inspect \
  --data-root /home/ollie/Documents/pdebench-data \
  --output-root .artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-ffno-convolutional-features-cns/inspect-<timestamp> \
  --history-len 2 \
  --max-train-trajectories 512 \
  --max-val-trajectories 64 \
  --max-test-trajectories 64 \
  --max-windows-per-trajectory 8
```

- [ ] Verify the inspect root contains at least:
  - `hdf5_metadata.json`
  - `dataset_manifest.json`
  - `trajectory_split_manifest.json`
  - `invocation.json`
  - `invocation.sh`
- [ ] Write `reference_runs_10ep.json` under the item artifact root, freezing the three required `10`-epoch references and any optional continuity rows that pass the compare helper’s contract checks.
- [ ] Launch the fresh `10`-epoch variant run in tmux on the fixed capped contract:

```bash
python scripts/studies/run_pdebench_image128_suite.py \
  --task 2d_cfd_cns \
  --mode pilot \
  --data-root /home/ollie/Documents/pdebench-data \
  --output-root .artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-ffno-convolutional-features-cns/cns-ffno-localconv-10ep-<timestamp> \
  --profiles ffno_bottleneck_localconv_base \
  --history-len 2 \
  --epochs 10 \
  --batch-size 4 \
  --max-train-trajectories 512 \
  --max-val-trajectories 64 \
  --max-test-trajectories 64 \
  --max-windows-per-trajectory 8 \
  --device cuda \
  --num-workers 0
```

- [ ] Collate the fresh run with `write_cross_run_compare(...)` into `compare_10ep_against_existing.json` and `.csv`, using:
  - `fresh_run_root=<new 10ep run root>`
  - `fresh_profile_id="ffno_bottleneck_localconv_base"`
  - the frozen required `10`-epoch reference rows above
  - optional continuity rows only if their contracts and sample targets still align
- [ ] Render `compare_10ep_sample0.png` and `compare_10ep_sample0_error.png` only if target alignment succeeds. If it does not, keep the JSON/CSV compare, record `cross_run_gallery_blocked`, and do not rerun old anchors.

**Verification for Task 3**

- [ ] The tracked `10`-epoch process exits `0`.
- [ ] The fresh run root contains `comparison_summary.json`, `comparison_summary.csv`, `metrics_ffno_bottleneck_localconv_base.json`, and `model_profile_ffno_bottleneck_localconv_base.json`.
- [ ] `compare_10ep_against_existing.json` and `.csv` are fresh.
- [ ] The required `10`-epoch compare includes `author_ffno_cns_base`, `ffno_bottleneck_base`, and `spectral_resnet_bottleneck_base`.
- [ ] No old anchor is silently rerun just to make the merged compare.

## Task 4: Backfill The Missing `40`-Epoch Local FFNO-Close Anchor And Run The Fresh `40`-Epoch Variant

**Files:**
- Output only: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-ffno-convolutional-features-cns/`

- [ ] Before launching the new `40`-epoch variant, confirm whether a valid same-contract `ffno_bottleneck_base` `40`-epoch run already exists.
- [ ] If no such root exists, launch the bounded backfill in tmux:

```bash
python scripts/studies/run_pdebench_image128_suite.py \
  --task 2d_cfd_cns \
  --mode pilot \
  --data-root /home/ollie/Documents/pdebench-data \
  --output-root .artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-ffno-convolutional-features-cns/cns-ffno-close-backfill-40ep-<timestamp> \
  --profiles ffno_bottleneck_base \
  --history-len 2 \
  --epochs 40 \
  --batch-size 4 \
  --max-train-trajectories 512 \
  --max-val-trajectories 64 \
  --max-test-trajectories 64 \
  --max-windows-per-trajectory 8 \
  --device cuda \
  --num-workers 0
```

- [ ] Freeze `reference_runs_40ep.json` with the required rows:
  - `author_ffno_cns_base`
  - `ffno_bottleneck_base` from the discovered or freshly backfilled `40`-epoch root
  - `spectral_resnet_bottleneck_base`
- [ ] Launch the fresh `40`-epoch local-conv variant run:

```bash
python scripts/studies/run_pdebench_image128_suite.py \
  --task 2d_cfd_cns \
  --mode pilot \
  --data-root /home/ollie/Documents/pdebench-data \
  --output-root .artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-ffno-convolutional-features-cns/cns-ffno-localconv-40ep-<timestamp> \
  --profiles ffno_bottleneck_localconv_base \
  --history-len 2 \
  --epochs 40 \
  --batch-size 4 \
  --max-train-trajectories 512 \
  --max-val-trajectories 64 \
  --max-test-trajectories 64 \
  --max-windows-per-trajectory 8 \
  --device cuda \
  --num-workers 0
```

- [ ] Collate `compare_40ep_against_existing.json` and `.csv` with `write_cross_run_compare(...)`, requiring the `40`-epoch author, local FFNO-close, and spectral rows.
- [ ] Render `compare_40ep_sample0.png` and `compare_40ep_sample0_error.png` only if target alignment succeeds.
- [ ] Treat any missing same-contract `40`-epoch FFNO-close reference as a blocker, not a reason to shrink the comparison standard.

**Verification for Task 4**

- [ ] The `40`-epoch backfill process exits `0` if it was needed, and its run root contains `metrics_ffno_bottleneck_base.json` plus `model_profile_ffno_bottleneck_base.json`.
- [ ] The fresh `40`-epoch local-conv process exits `0`.
- [ ] `reference_runs_40ep.json`, `compare_40ep_against_existing.json`, and `compare_40ep_against_existing.csv` are fresh.
- [ ] The required `40`-epoch compare includes `author_ffno_cns_base`, `ffno_bottleneck_base`, and `spectral_resnet_bottleneck_base`.
- [ ] The durable summary can truthfully state that the `40`-epoch compare did not mix `10`-epoch and `40`-epoch FFNO-family rows.

## Task 5: Write Durable Summary, Update Discoverability, And Record Queue State

**Files:**
- Create: `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_ffno_convolutional_features_cns_summary.md`
- Modify: `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_2d_cfd_cns_summary.md`
- Modify: `docs/studies/index.md`
- Modify: `docs/index.md`
- Modify: `state/NEURIPS-HYBRID-RESNET-2026/progress_ledger.json`
- Modify only if justified: `docs/findings.md`

- [ ] Write the durable summary with:
  - implementation surfaces touched
  - the exact chosen local-conv variant and why the scope stayed at one variant
  - the exact `10`-epoch and `40`-epoch reference run roots used
  - whether the `40`-epoch FFNO-close backfill was required
  - aggregate results with `relative_l2`
  - high-frequency results with `fRMSE_high`
  - runtime / parameter-count tradeoffs
  - an explicit capped-lane / benchmark-incomplete disclaimer
- [ ] Sync the CNS summary with a short subsection or bullet that points to the new durable summary and states whether local convolution materially helped FFNO on this capped contract.
- [ ] If `docs/studies/index.md` and `docs/index.md` do not already expose the durable summary as a discoverable source, add concise entries.
- [ ] Append a `post_completion_updates` entry to `state/NEURIPS-HYBRID-RESNET-2026/progress_ledger.json` using `tranche_id: "2026-04-27-pdebench-ffno-convolutional-features-cns"`, recording at minimum:
  - `design_path`
  - `plan_path`
  - `summary_path`
  - `cns_summary_path`
  - `artifact_root`
  - `reference_manifest_paths` for `10ep` and `40ep`
  - `fresh_variant_runs` for `10ep` and `40ep`
  - `ffno_close_40ep_backfill_run_root` if one was created
  - `merged_compare_artifacts` for `10ep` and `40ep`
  - `local_equal_footing_contract`
  - `performance_assessment_complete: false`
  - an explicit evidence-scope label that keeps the item capped and decision-support-only
  - the required deterministic verification commands and fresh run commands that succeeded
- [ ] Update `docs/findings.md` only if the result establishes a reusable FFNO/local-feature rule worth carrying forward beyond this single backlog lane.

## Completion Criteria

- [ ] `ffno_bottleneck_localconv_base` builds and runs without changing the fixed capped CNS contract.
- [ ] The backlog item’s required deterministic checks pass:
  - `pytest -q tests/studies/test_pdebench_image128_models.py tests/studies/test_pdebench_image128_runner.py`
  - `python -m compileall -q scripts/studies/pdebench_image128`
- [ ] Fresh `10`-epoch compare artifacts exist and include the authored FFNO, local FFNO-close, and spectral anchors.
- [ ] Fresh `40`-epoch compare artifacts exist and include a same-contract `40`-epoch local FFNO-close anchor, whether discovered or backfilled.
- [ ] The durable summary states clearly whether local convolution improved or hurt FFNO on `relative_l2` and `fRMSE_high`, and whether any gain looks worth the extra complexity.
- [ ] All outputs remain labeled as capped decision-support evidence, not benchmark-complete CNS evidence.
