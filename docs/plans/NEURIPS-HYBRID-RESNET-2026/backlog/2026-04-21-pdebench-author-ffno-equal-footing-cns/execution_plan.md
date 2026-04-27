# PDEBench Author FFNO Equal-Footing CNS Execution Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Integrate the official-author FFNO model into the PDEBench `2d_cfd_cns` runner, prove whether it can honor the existing capped local CNS contract, and, if it can, produce fresh `10`- and `40`-epoch author-FFNO rows plus durable cross-run comparisons against the already-recorded local CNS reference rows without changing the fixed fairness contract.

**Architecture:** Keep `scripts/studies/pdebench_image128/cfd_cns.py` authoritative for the CNS data split, `history_len=2` supervision contract, MSE training loss, metric family, and reporting surfaces. Add one optional external-baseline profile, `author_ffno_cns_base`, behind a thin adapter plus provenance record, stage the work as source gate -> adapter/profile wiring -> bounded smoke gate -> fresh author-only `10`/`40`-epoch runs -> metadata-based cross-run collation against fixed existing reference rows -> durable summary/index/ledger updates, and stop with an explicit incompatibility or contract-mismatch record instead of drifting the contract if the author code cannot fit. No worktree is used because repo policy explicitly forbids worktrees here.

**Tech Stack:** PATH `python`, PyTorch (POLICY-001), existing PDEBench image-suite runner, pytest, compileall, matplotlib, tmux with PATH `python` in the documented host environment for long runs (start with `ptycho311`; use a separately documented compatible local environment if the pinned author FFNO source requires it), repo-local external source staged under `.artifacts/external/`

---

## Initiative

- ID: `NEURIPS-HYBRID-RESNET-2026`
- Backlog item: `2026-04-21-pdebench-author-ffno-equal-footing-cns`
- Status: pending
- Date: 2026-04-22
- Scope owner: Roadmap Phase 2 capped external-baseline lane
- Previous backlog-only plan: `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_author_ffno_equal_footing_plan.md`
- Artifact root: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-author-ffno-equal-footing-cns/`
- Durable summary target: `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_author_ffno_equal_footing_summary.md`

## Inputs Read

- `AGENTS.md`
- `docs/index.md`
- `docs/findings.md`
- `docs/INITIATIVE_WORKFLOW_GUIDE.md`
- `docs/DEVELOPER_GUIDE.md`
- `docs/TESTING_GUIDE.md`
- `docs/workflows/pytorch.md`
- `docs/model_baselines.md`
- `docs/studies/index.md`
- `docs/COMMANDS_REFERENCE.md`
- `docs/steering.md`
- `docs/plans/2026-04-20-neurips-hybrid-resnet-submission-design.md`
- `docs/plans/2026-04-20-neurips-hybrid-resnet-submission-roadmap.md`
- `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_128x128_image_suite_plan.md`
- `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_2d_cfd_cns_summary.md`
- `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_ffno_bottleneck_summary.md`
- `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_gnot_cns_compare_summary.md`
- `docs/backlog/index.md`
- `state/NEURIPS-HYBRID-RESNET-2026/progress_ledger.json`
- `state/NEURIPS-HYBRID-RESNET-2026/backlog_drain/iterations/0/items/2026-04-21-pdebench-author-ffno-equal-footing-cns/selected-item-context.md`
- `scripts/studies/pdebench_image128/{models.py,run_config.py,cfd_cns.py,gnot_adapter.py,render_hybrid_upsampler_gallery.py}`
- `tests/studies/{test_pdebench_image128_models.py,test_pdebench_image128_runner.py}`

## Objective

- Run the authors' actual FFNO model, not the repo's existing FFNO-close bottleneck proxy, on the local PDEBench `2d_cfd_cns` capped compare contract.
- Keep the equal-footing protocol fixed across the new author row and the existing local reference rows.
- Produce one fresh `10`-epoch author row and one fresh `40`-epoch author row, then compare them against the existing local `spectral_resnet_bottleneck_base`, `fno_base`, and `unet_strong` rows with the same reported metric family; use an already-existing hybrid row only as optional continuity context.

## Scope

- Select and pin the authoritative author FFNO source and record why it qualifies as the real author model for this backlog item.
- Add one optional `author_ffno_cns_base` integration path to the existing CNS runner without changing the shared PDEBench data, split, normalization, or metric contracts.
- Run a bounded smoke gate to prove the author model either fits the local contract or blocks cleanly with an explicit reason.
- If the smoke gate succeeds, run fresh author-only `10`-epoch and `40`-epoch jobs under the fixed local contract, then collate those new rows against the already-recorded local CNS reference rows.
- Write durable summary, update the CNS summary, and update discoverability/ledger surfaces.

## Explicit Non-Goals

- Do not replace `fno_base` or promote author FFNO into the default primary bundle lists.
- Do not reuse `ffno_bottleneck_base` as if it were the author model.
- Do not change the PDEBench CNS benchmark gate, dataset slice, `history_len`, training loss, or metric family to suit imported author code.
- Do not create `/home/ollie/Documents/neurips/` artifacts or manuscript prose; this remains a Phase 2 decision-support scope only.
- Do not touch `ptycho/model.py`, `ptycho/diffsim.py`, or `ptycho/tf_helper.py`.
- Do not let runtime-hosting convenience rewrite the benchmark contract. If the official code needs a separately documented compatible local environment, treat that as external-baseline setup cost, keep commands on PATH `python` after activation, and capture exact environment/package provenance instead of changing the data, split, loss, or metric contract.

## Steering And Roadmap Constraints

- Steering makes author FFNO the preferred current external-baseline attempt before the paper-default GNOT rerun unless FFNO is explicitly blocked.
- The roadmap allows this item as a bounded CNS follow-up because the verified `history_len=2` MSE anchor family already exists, but it must stay capped and decision-support-only.
- The backlog dependency index marks this item as a parallel external-baseline lane with no active backlog prerequisite. This plan therefore reuses the existing capped CNS anchors and must not widen itself into mandatory shell reruns unless a later approved backlog item or roadmap update explicitly changes scope.
- Environment-hosting cost is part of this external-baseline lane, not a scope violation. The existing GNOT summary already established that an official external baseline may need a separately documented compatible local environment; author FFNO may use the same pattern if needed, but the compare contract and reporting surfaces must stay fixed.
- Equal-footing means the following stay fixed unless the plan records an incompatibility and stops:
  - dataset: `/home/ollie/Documents/pdebench-data/2d_cfd_cns/2D_CFD_Rand_M1.0_Eta0.01_Zeta0.01_periodic_128_Train.hdf5`
  - split: `512 / 64 / 64` trajectories
  - `max_windows_per_trajectory=8`
  - supervision contract: `history_len=2`, `concat u[t-2:t] -> u[t]`
  - training loss: `mse`
  - optimizer/scheduler recipe for this local equal-footing lane: Adam, `lr=2e-4`, `ReduceLROnPlateau(factor=0.5, patience=2, min_lr=1e-5, threshold=0.0)`
  - batch size: `4`
  - metrics/reporting: `err_RMSE`, `err_nRMSE`, `relative_l2`, `fRMSE_low`, `fRMSE_mid`, `fRMSE_high`, per-profile prediction PNG/NPZ artifacts, and comparison galleries
- If the selected author implementation cannot satisfy that contract without material protocol drift, the correct outcome is a blocker/incompatibility summary, not a silent recipe or data-contract rewrite.

## Prerequisite Status

- Satisfied from the progress ledger:
  - official `2d_cfd_cns` file is staged and checksum-verified
  - CNS adapter/data/metric path is implemented and tested
  - current local CNS loss contract is MSE
  - canonical CNS Hybrid shell is fixed to `hybrid_resnet_cns` with skip-add and `pixelshuffle`
  - capped decision-support anchor family already exists for `spectral_resnet_bottleneck_base`, `hybrid_resnet_cns`, `fno_base`, and `unet_strong`
- Relevant execution precedent:
  - the official GNOT external-baseline lane already proved that a separately documented compatible local environment can be a valid hosting path for an equal-footing CNS compare when the data/history/metric contract stays fixed
- Not a prerequisite for this item:
  - the paper-default GNOT rerun
  - full-training benchmark-complete CNS rows
- Still true after this item:
  - any result remains capped decision-support evidence only until the full-training Phase 2 benchmark gates are satisfied

## Fixed Existing Reference Rows

These rows already exist and are the authoritative local comparison targets for
this backlog item. Do not rerun them in this plan. The implementation must
reuse these exact run roots unless one is missing required provenance or fails
the explicit contract-match check.

### Required `10`-Epoch Reference Rows

- `spectral_resnet_bottleneck_base`:
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-ffno-bottleneck-cns-compare/readiness-cap-20260421T203200Z-ffno-cns-10ep`
- `fno_base`:
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-2d-cfd-cns/readiness-cap-20260421T183717Z-10ep-mse`
- `unet_strong`:
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-2d-cfd-cns/readiness-cap-20260421T183717Z-10ep-mse`

### Optional `10`-Epoch Continuity Rows

- `hybrid_resnet_cns`:
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-ffno-bottleneck-cns-compare/readiness-cap-20260421T203200Z-ffno-cns-10ep`
- `hybrid_resnet_base`:
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-2d-cfd-cns/readiness-cap-20260421T183717Z-10ep-mse`

### Required `40`-Epoch Reference Rows

- `spectral_resnet_bottleneck_base`:
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-ffno-bottleneck-cns-compare/readiness-cap-20260421T221008Z-spectral40ep`
- `fno_base`:
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-2d-cfd-cns/readiness-cap-20260421T185926Z-40ep-mse`
- `unet_strong`:
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-2d-cfd-cns/readiness-cap-20260421T185926Z-40ep-mse`

### Optional `40`-Epoch Continuity Row

- `hybrid_resnet_base`:
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-2d-cfd-cns/readiness-cap-20260421T185926Z-40ep-mse`

## Implementation Architecture

- **External Source Gate:** add a dedicated author-FFNO adapter that resolves the pinned official repo from `AUTHOR_FFNO_ROOT` or `.artifacts/external/<author-ffno-repo-slug>/`, emits `author_ffno_source.json`, and converts missing source, missing dependencies in the selected host environment, or irreconcilable contract mismatches into `ModelBuildBlocker` payloads.
- **Runtime Host Gate:** start with `ptycho311`. If the pinned author source cannot import or execute there for environment-hosting reasons, switch to the simplest compatible local environment that can still run the repo's PDEBench suite commands on PATH `python` and record that choice in `author_ffno_source.json` plus the durable summary. Prefer an already working local env when one exists; if a new env is unavoidable, keep it minimal and document the exact package versions and activation step. Block only if no compatible host can preserve the fixed local compare contract.
- **Runner/Profile Integration:** add `author_ffno_cns_base` / `author_ffno_cns_net` as a manual opt-in profile wired through `run_config.py`, `models.py`, and `cfd_cns.py`. The adapter may perform only model-internal reshaping/tokenization/coordinate construction; the existing CNS dataset, normalization, split, loss, and metric code remain authoritative.
- **Evidence And Reporting:** require one bounded smoke gate before any expensive run. Only after a passing smoke gate may the implementation launch fresh author-only `10`-epoch and `40`-epoch runs, verify metadata parity against the fixed existing reference rows, write cross-run comparison JSON/CSV artifacts, optionally render cross-run galleries when sample targets align, and update the durable summary, index, and progress ledger. Missing or mismatched reference artifacts are blockers for comparison collation, not reasons to rerun the local shell.

## Concrete File And Artifact Targets

### Code

- Create: `scripts/studies/pdebench_image128/author_ffno_adapter.py`
- Modify: `scripts/studies/pdebench_image128/models.py`
- Modify: `scripts/studies/pdebench_image128/run_config.py`
- Modify: `scripts/studies/pdebench_image128/cfd_cns.py` only if the author profile needs explicit local-recipe assertions or extra provenance wiring

### Tests

- Modify: `tests/studies/test_pdebench_image128_models.py`
- Modify: `tests/studies/test_pdebench_image128_runner.py`

### Durable Docs

- Create: `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_author_ffno_equal_footing_summary.md`
- Modify: `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_2d_cfd_cns_summary.md`
- Modify: `docs/studies/index.md`
- Modify: `docs/index.md`
- Modify: `state/NEURIPS-HYBRID-RESNET-2026/progress_ledger.json`

### Artifacts

- Create/update: `.artifacts/external/<author-ffno-repo-slug>/`
- Create: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-author-ffno-equal-footing-cns/author_ffno_source.json`
- Create: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-author-ffno-equal-footing-cns/reference_runs.json`
- Create: smoke run root under `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-author-ffno-equal-footing-cns/`
- Create: fresh author-only `10`-epoch run root under the same artifact root
- Create: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-author-ffno-equal-footing-cns/compare_10ep_against_existing.json`
- Create: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-author-ffno-equal-footing-cns/compare_10ep_against_existing.csv`
- Create if sample targets align: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-author-ffno-equal-footing-cns/compare_10ep_sample0.png`
- Create if sample targets align: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-author-ffno-equal-footing-cns/compare_10ep_sample0_error.png`
- Create: fresh author-only `40`-epoch run root under the same artifact root
- Create: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-author-ffno-equal-footing-cns/compare_40ep_against_existing.json`
- Create: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-author-ffno-equal-footing-cns/compare_40ep_against_existing.csv`
- Create if sample targets align: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-author-ffno-equal-footing-cns/compare_40ep_sample0.png`
- Create if sample targets align: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-author-ffno-equal-footing-cns/compare_40ep_sample0_error.png`

## Task 1: Lock The Author Source And Red-Test The Blocker Contract

**Files:**
- Create: `scripts/studies/pdebench_image128/author_ffno_adapter.py`
- Modify: `tests/studies/test_pdebench_image128_models.py`
- Modify: `tests/studies/test_pdebench_image128_runner.py`

- [ ] **Step 1: Pin the authoritative author source**

Record a single chosen source under `.artifacts/external/<author-ffno-repo-slug>/` and write `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-author-ffno-equal-footing-cns/author_ffno_source.json` with:

- repo URL
- pinned commit SHA
- local root
- exact model entrypoint/module/class used
- why this is the real author FFNO implementation rather than a proxy
- dependency notes
- chosen host environment and activation notes, including whether default `ptycho311` worked or which alternate compatible local environment was selected instead
- critical runtime package versions (`python`, `torch`, and any author-required extras)
- explicit note on whether the source natively supports the local `2d_cfd_cns` one-step contract or requires only wrapper-local tensor adaptation

- [ ] **Step 2: Add failing model/profile tests before implementation**

Add tests analogous to the existing GNOT path for:

- `author_ffno_cns_base` exists and stays out of `PRIMARY_*` and `READINESS_*` bundle lists
- missing `AUTHOR_FFNO_ROOT` / missing default clone path blocks cleanly through `ModelBuildBlocker`
- missing required author-side dependencies in the selected host environment block cleanly with `reason="model_dependency_unavailable"`
- `describe_model()` / `model_profile_author_ffno_cns_base.json` carries external-source provenance rich enough to identify the chosen host environment
- the runner uses the local equal-footing recipe for this profile (`mse`, Adam, `2e-4`, `ReduceLROnPlateau`) rather than a paper-default override

- [ ] **Step 3: Run the focused red slice**

Run:

```bash
pytest tests/studies/test_pdebench_image128_models.py -k 'author_ffno' -v
pytest tests/studies/test_pdebench_image128_runner.py -k 'author_ffno' -v
```

Expected: FAIL because the new author FFNO path does not exist yet.

## Task 2: Add The Optional `author_ffno_cns_base` Integration Path

**Files:**
- Create: `scripts/studies/pdebench_image128/author_ffno_adapter.py`
- Modify: `scripts/studies/pdebench_image128/models.py`
- Modify: `scripts/studies/pdebench_image128/run_config.py`
- Modify: `scripts/studies/pdebench_image128/cfd_cns.py` only if needed for recipe/provenance assertions
- Test: `tests/studies/test_pdebench_image128_models.py`
- Test: `tests/studies/test_pdebench_image128_runner.py`

- [ ] **Step 1: Add the manual profile**

Add `author_ffno_cns_base` with:

- `base_model="author_ffno_cns_net"`
- `evidence_scope="readiness-only"`
- manual opt-in status only
- hyperparameters sourced from the pinned author repo/config, not copied from `ffno_bottleneck_base`
- no optimizer/loss overrides unless Task 1 proves that the local equal-footing recipe is impossible, in which case the task stops as blocked instead of changing the contract

- [ ] **Step 2: Implement the thin adapter**

The adapter must:

- resolve the pinned source from `AUTHOR_FFNO_ROOT` first, then the repo-local default clone
- expose `external_source_provenance` so `describe_model()` writes it into `model_profile_author_ffno_cns_base.json`, including the chosen host environment metadata captured in Task 1
- accept the runner's `task_metadata`
- translate the existing `B,C,H,W -> B,C,H,W` CNS tensors into whatever the author model needs internally
- keep all tensor adaptation inside the wrapper; do not change split logic, dataset normalization, or reporting

- [ ] **Step 3: Wire the adapter into the builder and blocker flow**

Extend `build_model_from_profile()` so that:

- `author_ffno_cns_net` instantiates the new adapter
- missing source/dependency/incompatibility cases become `ModelBuildBlocker` payloads
- successful builds preserve `external_source_provenance`

- [ ] **Step 4: Re-run the focused tests and then the required deterministic checks**

Run:

```bash
pytest tests/studies/test_pdebench_image128_models.py -k 'author_ffno' -v
pytest tests/studies/test_pdebench_image128_runner.py -k 'author_ffno' -v
pytest -q tests/studies/test_pdebench_image128_models.py tests/studies/test_pdebench_image128_runner.py
python -m compileall -q scripts/studies/pdebench_image128
```

Expected: PASS

## Task 3: Prove The Author Model Fits The Local Contract With A Bounded Smoke Gate

**Files:**
- Output only: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-author-ffno-equal-footing-cns/`

- [ ] **Step 1: Launch a bounded one-profile smoke gate**

Use a fresh timestamped output root, for example:

```bash
python scripts/studies/run_pdebench_image128_suite.py \
  --task 2d_cfd_cns \
  --mode readiness \
  --data-root /home/ollie/Documents/pdebench-data \
  --output-root .artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-author-ffno-equal-footing-cns/smoke-author-ffno-<timestamp> \
  --profiles author_ffno_cns_base \
  --history-len 2 \
  --epochs 1 \
  --batch-size 4 \
  --max-train-trajectories 8 \
  --max-val-trajectories 2 \
  --max-test-trajectories 2 \
  --max-windows-per-trajectory 2 \
  --device cuda \
  --num-workers 0
```

- [ ] **Step 2: Run it through tmux and enforce the long-run guardrail**

- use the tmux skill
- activate the documented host environment for author FFNO runs: start with `ptycho311`, but if Task 1 selected a compatible alternate local environment, activate that environment instead and keep commands on PATH `python`
- track the exact launched process, not a broad `pgrep`
- declare completion only when the tracked process exits `0` and required artifacts are freshly written

- [ ] **Step 3: Verify the smoke-gate outputs**

Required fresh outputs:

- `invocation.json`
- `invocation.sh`
- `dataset_manifest.json`
- `split_manifest.json`
- `model_profile_author_ffno_cns_base.json`
- `metrics_author_ffno_cns_base.json`
- `comparison_summary.json`
- `comparison_summary.csv`
- `comparison_author_ffno_cns_base_sample0.png`
- `comparison_author_ffno_cns_base_sample0.npz`

- [ ] **Step 4: Stop early on incompatibility**

If the smoke gate yields a blocker payload or otherwise proves the author model cannot run on the fixed local contract, do all of the following and stop:

- if the only failure in `ptycho311` is environment-hosting or dependency mismatch, switch once to the Task 1 documented compatible local environment and rerun the smoke gate before declaring the item blocked
- write the blocker and exact reason into the durable summary
- update `state/NEURIPS-HYBRID-RESNET-2026/progress_ledger.json`
- update `docs/index.md` / `docs/studies/index.md` if the blocker changes durable operator knowledge
- do **not** launch the `10`-epoch or `40`-epoch compare runs

## Task 4: Run The Fresh Equal-Footing `10`-Epoch Author Row And Compare It Against Existing Local Rows

**Files:**
- Output only: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-author-ffno-equal-footing-cns/`

- [ ] **Step 1: Freeze the reference manifest for both epoch budgets**

Write `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-author-ffno-equal-footing-cns/reference_runs.json` with:

- the required `10`-epoch and `40`-epoch reference rows listed in `Fixed Existing Reference Rows`
- any optional continuity rows chosen for context at either epoch budget
- for each row:
  - `run_root`
  - `profile_id`
  - expected `epochs`
  - expected dataset path
  - expected split counts
  - expected `max_windows_per_trajectory`
  - expected `history_len`
  - expected training loss
  - source document that established the row

- [ ] **Step 2: Keep the author run contract fixed**

- task: `2d_cfd_cns`
- mode: `readiness`
- `history_len=2`
- split: `512 / 64 / 64`
- `max_windows_per_trajectory=8`
- epochs: `10`
- batch size: `4`
- device: `cuda`
- local equal-footing recipe for the new author row: `mse`, Adam, `2e-4`, `ReduceLROnPlateau`
- host environment: the same documented environment that passed the smoke gate; do not change environments between the smoke result and the fresh author run without recording why

- [ ] **Step 3: Launch the fresh author-only `10`-epoch run**

```bash
python scripts/studies/run_pdebench_image128_suite.py \
  --task 2d_cfd_cns \
  --mode readiness \
  --data-root /home/ollie/Documents/pdebench-data \
  --output-root .artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-author-ffno-equal-footing-cns/author-ffno-10ep-<timestamp> \
  --profiles author_ffno_cns_base \
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

- [ ] **Step 4: Collate the fresh author row against the existing `10`-epoch reference rows**

Build `compare_10ep_against_existing.json` and `compare_10ep_against_existing.csv`
under the author artifact root by loading the fresh author run plus the fixed
existing reference run roots and doing all of the following before writing the
merged compare artifact:

- assert the required reference artifacts exist:
  - `invocation.json`
  - `dataset_manifest.json`
  - `split_manifest.json`
  - `comparison_summary.json`
  - `metrics_<profile>.json`
  - `model_profile_<profile>.json`
- assert the fixed equal-footing contract matches across the author row and each
  required reference row:
  - same dataset file
  - same `512 / 64 / 64` trajectory split
  - same `max_windows_per_trajectory=8`
  - same `history_len=2`
  - same `epochs=10`
  - same batch size `4`
  - same training loss `mse`
  - same metric family
- merge at least these rows into one table:
  - `author_ffno_cns_base`
  - `spectral_resnet_bottleneck_base`
  - `fno_base`
  - `unet_strong`
- append optional continuity rows only if they already exist and pass the same
  contract-match checks

- [ ] **Step 5: Render a cross-run gallery only if the saved sample targets align**

Use the saved `comparison_<profile>_sample0.npz` artifacts from the author run
and the selected reference rows to render
`compare_10ep_sample0.png` / `compare_10ep_sample0_error.png` only if:

- `field_order` matches across all selected rows
- `target` arrays match across all selected rows to within a strict tolerance

If the targets do not align, do **not** rerun the local rows just to make a
gallery. Instead:

- record `cross_run_gallery_blocked` with the mismatch reason in
  `compare_10ep_against_existing.json`
- note the limitation in the durable summary
- keep the merged JSON/CSV compare as the required comparison artifact

- [ ] **Step 6: Verify the `10`-epoch compare is complete**

Required conditions:

- tracked process exit code `0`
- the fresh author run root contains `comparison_summary.json`,
  `comparison_summary.csv`, `metrics_author_ffno_cns_base.json`, and
  `model_profile_author_ffno_cns_base.json`
- `compare_10ep_against_existing.json` and `compare_10ep_against_existing.csv`
  are fresh
- no fresh rerun root was created for `spectral_resnet_bottleneck_base`,
  `fno_base`, `unet_strong`, or any optional continuity row
- cross-run gallery PNGs are present if target alignment succeeded, or the
  merged compare JSON explicitly records why they were skipped

## Task 5: Run The Fresh Equal-Footing `40`-Epoch Author Row And Compare It Against Existing Local Rows

**Files:**
- Output only: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-author-ffno-equal-footing-cns/`

- [ ] **Step 1: Reuse the same fixed contract and switch only the epoch budget**

The fresh author `40`-epoch run must reuse the exact same:

- dataset file
- split counts
- `history_len=2`
- `max_windows_per_trajectory=8`
- batch size `4`
- metric family
- documented host environment that passed the smoke gate unless the durable summary records a forced host change and why it still preserves fairness

Only `epochs` changes from `10` to `40`. The existing reference rows remain the
fixed run roots listed in `Fixed Existing Reference Rows`; do not rerun them.

- [ ] **Step 2: Launch the fresh author-only `40`-epoch run**

```bash
python scripts/studies/run_pdebench_image128_suite.py \
  --task 2d_cfd_cns \
  --mode readiness \
  --data-root /home/ollie/Documents/pdebench-data \
  --output-root .artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-author-ffno-equal-footing-cns/author-ffno-40ep-<timestamp> \
  --profiles author_ffno_cns_base \
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

- [ ] **Step 3: Collate the fresh author row against the existing `40`-epoch reference rows**

Build `compare_40ep_against_existing.json` and `compare_40ep_against_existing.csv`
under the author artifact root by loading the fresh author run plus the fixed
existing `40`-epoch reference rows and asserting the same equal-footing
contract fields as Task 4, except `epochs=40`.

The merged `40`-epoch table must include at least:

- `author_ffno_cns_base`
- `spectral_resnet_bottleneck_base`
- `fno_base`
- `unet_strong`

The older `hybrid_resnet_base` row is optional continuity only.

- [ ] **Step 4: Render a cross-run `40`-epoch gallery only if the saved sample targets align**

Use the same target-alignment rule as Task 4. If the author and reference
sample-0 NPZs do not share the same target tensor, record the blocker in
`compare_40ep_against_existing.json` and the durable summary instead of rerunning
the local shell.

- [ ] **Step 5: Verify the `40`-epoch compare is complete**

Required conditions:

- tracked process exit code `0`
- the fresh author run root contains `comparison_summary.json`,
  `comparison_summary.csv`, `metrics_author_ffno_cns_base.json`, and
  `model_profile_author_ffno_cns_base.json`
- `compare_40ep_against_existing.json` and `compare_40ep_against_existing.csv`
  are fresh
- no fresh rerun root was created for `spectral_resnet_bottleneck_base`,
  `fno_base`, `unet_strong`, or the optional `hybrid_resnet_base` continuity row
- cross-run gallery PNGs are present if target alignment succeeded, or the
  merged compare JSON explicitly records why they were skipped

- [ ] **Step 6: Treat infeasibility as a blocker, not a silent scope shrink**

If the author row OOMs, diverges irreparably, or otherwise cannot complete the `40`-epoch fixed contract after the `10`-epoch path succeeded, write that as an explicit blocker/caveat in the summary and ledger. Do not silently drop to fewer epochs or fewer comparison rows.

## Task 6: Write Durable Summary, Update Discoverability, And Record The Queue State

**Files:**
- Create: `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_author_ffno_equal_footing_summary.md`
- Modify: `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_2d_cfd_cns_summary.md`
- Modify: `docs/studies/index.md`
- Modify: `docs/index.md`
- Modify: `state/NEURIPS-HYBRID-RESNET-2026/progress_ledger.json`

- [ ] **Step 1: Write the durable summary**

The summary must include:

- pinned author source and why it is authoritative
- environment/dependency notes
- implementation surfaces touched
- smoke-gate outcome
- the exact existing `10`-epoch and `40`-epoch reference run roots used for
  comparison
- `10`-epoch author-run contract and results
- `10`-epoch cross-run comparison results, including whether a gallery was
  rendered or blocked by target mismatch
- `40`-epoch author-run contract and results or explicit blocker
- `40`-epoch cross-run comparison results, including whether a gallery was
  rendered or blocked by target mismatch
- fairness caveats
- explicit distinction between `author_ffno_cns_base` and `ffno_bottleneck_base`
- artifact paths for source JSON, reference manifest, author run roots, merged
  comparison JSON/CSV files, and any galleries

- [ ] **Step 2: Update the CNS summary**

Add a section to `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_2d_cfd_cns_summary.md` that:

- links the new author FFNO summary
- records the new author row(s) and the existing local rows they were compared against
- keeps the existing FFNO-close bottleneck result separate
- preserves the capped-readiness, decision-support-only claim boundary

- [ ] **Step 3: Update discoverability docs**

Update:

- `docs/studies/index.md`
- `docs/index.md`

so the actual author FFNO baseline is discoverable separately from the FFNO-close bottleneck proxy experiment.

- [ ] **Step 4: Update the progress ledger**

Record one new tranche entry, or a blocker entry if the smoke gate failed, with:

- decision and decision scope
- `author_ffno_source.json` path
- `reference_runs.json` path
- chosen host environment and critical package versions
- smoke, author-only `10`-epoch, and author-only `40`-epoch run roots as applicable
- merged `10`-epoch and `40`-epoch compare artifact paths
- existing reference run roots used for each compare
- summary path
- verification commands and outcomes
- affected surfaces

No backlog dependency index change is required unless the implementation discovers a real new backlog prerequisite; otherwise keep the lane parallel.

## Required Deterministic Checks

These are mandatory end-state checks from the selected backlog item and must be run even if stronger focused tests are also used:

```bash
pytest -q tests/studies/test_pdebench_image128_models.py tests/studies/test_pdebench_image128_runner.py
python -m compileall -q scripts/studies/pdebench_image128
```

## Additional Verification

- [ ] Smoke gate either completed with fresh artifacts or produced a clean blocker payload and stopped the expensive author runs.
- [ ] The fresh `10`-epoch author run root exists and `compare_10ep_against_existing.json` / `.csv` were written without rerunning the existing local rows.
- [ ] The fresh `40`-epoch author run root exists and `compare_40ep_against_existing.json` / `.csv` were written without rerunning the existing local rows, or the durable summary/ledger explicitly records the blocker.
- [ ] `model_profile_author_ffno_cns_base.json` includes `external_source_provenance`.
- [ ] `author_ffno_source.json` and the durable summary record the chosen host environment and enough package provenance to reproduce the run host.
- [ ] `reference_runs.json` records the exact run-root/profile mapping for the reused local rows.
- [ ] `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_author_ffno_equal_footing_summary.md` exists and is linked from `docs/index.md` and `docs/studies/index.md`.
- [ ] `state/NEURIPS-HYBRID-RESNET-2026/progress_ledger.json` records the final outcome.

## Stop Conditions

- Stop and write a blocker if no pinned official-author FFNO source can be justified.
- Stop and write a blocker if the author implementation cannot fit the local `2d_cfd_cns` contract without changing the fixed split, `history_len`, MSE loss, or metric family.
- Stop and write a blocker if any required existing reference row is missing required provenance artifacts or fails the explicit contract-match checks; do not rerun the local shell to paper over that mismatch from this backlog item.
- Stop and write a blocker only if no compatible documented local host environment can run the pinned author code cleanly while preserving the fixed local compare contract and PATH `python` workflow.
- Do not convert any capped result from this item into a benchmark-complete or manuscript-facing claim.
