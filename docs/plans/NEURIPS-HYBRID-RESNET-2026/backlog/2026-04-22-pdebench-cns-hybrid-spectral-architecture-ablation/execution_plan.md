# PDEBench CNS Hybrid-Spectral Architecture Ablation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce fresh capped PDEBench `2d_cfd_cns` evidence that isolates Hybrid-spectral weight sharing and shared-shell spectral bottleneck depth under the fixed canonical CNS shell, then publish durable repo-local interpretation without widening into benchmark-complete or cross-domain claims.

**Architecture:** Reuse the existing CNS runner, profile registry, and reporting path unless the binding deterministic checks expose a real blocker. Freeze the shell and capped contract, run only the approved fresh `pilot` tranches, confirm only the unique finalists on the larger `1024 / 128 / 128` cap, then sync the durable summary, CNS summary, docs discoverability, and initiative ledger while keeping every result labeled as capped decision-support evidence only.

**Tech Stack:** PATH `python`; long runs in tmux with `ptycho311`; PyTorch/Lightning; `scripts/studies/pdebench_image128/`; pytest; compileall; Markdown/JSON/CSV/PNG artifacts under `.artifacts/`.

---

## Initiative

- ID: `NEURIPS-HYBRID-RESNET-2026`
- Backlog item: `2026-04-22-pdebench-cns-hybrid-spectral-architecture-ablation`
- Status: pending
- Date: `2026-04-28`
- Selected-item authority: `state/NEURIPS-HYBRID-RESNET-2026/backlog_drain/iterations/15/items/2026-04-22-pdebench-cns-hybrid-spectral-architecture-ablation/selected-item-context.md`
- Plan path authority: `docs/plans/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-22-pdebench-cns-hybrid-spectral-architecture-ablation/execution_plan.md`
- Durable summary target: `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_cns_hybrid_spectral_arch_ablation_summary.md`
- CNS summary sync target: `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_2d_cfd_cns_summary.md`
- Artifact root: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-hybrid-spectral-architecture-ablation/`

This document supersedes earlier revisions at this path and is the new execution authority for this backlog item. Implementation should rely on this plan plus the approved design, not on older queue notes or raw backlog prose.

## Inputs Read

- Consumed steering: `docs/steering.md`
- Consumed design: `docs/plans/2026-04-20-neurips-hybrid-resnet-submission-design.md`
- Consumed roadmap: `docs/plans/2026-04-20-neurips-hybrid-resnet-submission-roadmap.md`
- Consumed selected-item context: `state/NEURIPS-HYBRID-RESNET-2026/backlog_drain/iterations/15/items/2026-04-22-pdebench-cns-hybrid-spectral-architecture-ablation/selected-item-context.md`
- Consumed progress ledger: `state/NEURIPS-HYBRID-RESNET-2026/progress_ledger.json`
- Consumed plan review report: `.artifacts/review/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-22-pdebench-cns-hybrid-spectral-architecture-ablation-plan-review.json`
- `AGENTS.md`
- `docs/index.md`
- `docs/findings.md`
- `docs/INITIATIVE_WORKFLOW_GUIDE.md`
- `docs/DEVELOPER_GUIDE.md`
- `docs/TESTING_GUIDE.md`
- `docs/COMMANDS_REFERENCE.md`
- `docs/model_baselines.md`
- `docs/studies/index.md`
- `docs/backlog/in_progress/2026-04-22-pdebench-cns-hybrid-spectral-architecture-ablation.md`
- `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_128x128_image_suite_plan.md`
- `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_2d_cfd_cns_summary.md`
- Background only:
  - prior plan revision at this same path
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_spectral_weight_sharing_summary.md`
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_hybrid_spectral_cns_architecture_ablation_design.md`
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_cns_hybrid_spectral_arch_ablation_plan.md`

## Selected Objective

- Run a focused PDEBench `2d_cfd_cns` Hybrid-spectral ablation that answers only the approved spectral-family questions:
  - whether disabling spectral weight sharing still helps under the fixed canonical CNS shell
  - whether deeper shared spectral bottlenecks (`8`, `10` blocks versus `6`) materially improve the same-shell capped CNS row
- Produce fresh auditable `pilot` run roots for the approved rows instead of relying on older sibling-study artifacts alone.
- Publish a durable repo-local summary and state update that keep the result discoverable and explicitly bounded as capped decision-support evidence, not benchmark-complete PDEBench evidence.

## Scope

- Task remains **CNS only**:
  - task: `2d_cfd_cns`
  - dataset: `/home/ollie/Documents/pdebench-data/2d_cfd_cns/2D_CFD_Rand_M1.0_Eta0.01_Zeta0.01_periodic_128_Train.hdf5`
  - resolution: `128x128`
- Fixed ranked `pilot` contract:
  - split caps: `512 / 64 / 64` trajectories
  - `max_windows_per_trajectory=8`
  - `history_len=2`
  - batch size `4`
  - training loss `mse`
  - metrics: `err_RMSE`, `err_nRMSE`, `relative_l2`, `fRMSE_low`, `fRMSE_mid`, `fRMSE_high`
- Fixed larger-cap finalist confirmation:
  - split caps: `1024 / 128 / 128`
  - `max_windows_per_trajectory=8`
  - `history_len=2`
  - batch size `4`
  - training loss `mse`
  - epochs `40`
- Fixed canonical CNS shell on every ablation row:
  - `hidden_channels=32`
  - `fno_modes=12`
  - `fno_blocks=4`
  - `hybrid_downsample_steps=2`
  - `hybrid_resnet_blocks=6`
  - `hybrid_skip_connections=True`
  - `hybrid_skip_style="add"`
  - `hybrid_upsampler="pixelshuffle"`
  - `spectral_bottleneck_modes=12`
  - `spectral_bottleneck_gate_init=0.1`
  - `spectral_bottleneck_gate_mode="shared"`
- Approved fresh tranches only:
  - sharing `10`-epoch pilot: `spectral_resnet_bottleneck_base`, `spectral_resnet_bottleneck_noshare`
  - sharing `40`-epoch pilot: `spectral_resnet_bottleneck_base`, `spectral_resnet_bottleneck_noshare`
  - shared-depth `40`-epoch pilot: `spectral_resnet_bottleneck_base`, `spectral_resnet_bottleneck_shared_blocks8`, `spectral_resnet_bottleneck_shared_blocks10`
  - larger-cap confirmation: rerun only the unique finalists from the two fresh `40`-epoch pilot tranches

## Explicit Non-Goals

- Do not widen this item into CDI, ptychography, SWE, Darcy, OpenFWI, or `/home/ollie/Documents/neurips/` artifact assembly.
- Do not reopen shell axes as primary questions:
  - skip routing
  - upsampler choice
  - local-vs-spectral family compare
- Do not mix in separate CNS ablation families:
  - `history_len=1`
  - physics regularization
  - higher-mode `32/32`
  - authored FFNO
  - GNOT
  - VCNeF
- Do not run non-shared deeper rows here, even though profiles may already exist.
- Do not change the fixed CNS contract to MAE, a different batch size, different split caps, different history length, or a different metric family.
- Do not relabel capped or larger-cap pilot runs as benchmark rows, suite-complete evidence, or paper-facing competitiveness evidence.
- Do not create worktrees.
- Do not touch `ptycho/model.py`, `ptycho/diffsim.py`, or `ptycho/tf_helper.py`.
- Do not turn this into a broad runner refactor. Patch only the smallest blocker if the required deterministic checks fail.

## Steering, Roadmap, And Fairness Constraints

- Preserve the approved roadmap phase order and evidence boundaries. This item strengthens the Phase 2 CNS comparison story only; it does not satisfy the roadmap’s full-training benchmark gate.
- Equal footing is mandatory. Within each compare tranche, dataset file, split caps, `history_len`, `max_windows_per_trajectory`, batch size, epoch budget, training-loss contract, metric family, and fixed shell must stay identical across compared rows.
- This remains a CNS-only Hybrid-spectral ablation. Do not infer CDI or broader PDE family conclusions from it.
- The design and roadmap require capped, smoke, and pilot outputs to remain decision-support only. This item may rank rows only inside this bounded capped lane for internal finalist selection.
- The backlog item’s fixed `mse` CNS contract is a binding task-local override to the broader PDE Hybrid ResNet MAE baseline in `docs/model_baselines.md`. Do not “correct” it back to MAE here.
- The shared-depth tranche must stay on the shared shell. Non-shared deeper rows are out of scope.
- Steering already treats authored FFNO and GNOT as separate valid comparison lanes. Those prerequisites have been handled elsewhere; do not reopen their ordering or widen this item into external-baseline work.
- Long-running commands must run in tmux with `ptycho311` active, track the exact launched PID, and count as complete only when:
  - the tracked PID exits `0`
  - required output artifacts for that step exist and are freshly written
- Follow `REPORTING-ARTIFACT-BOUNDARY-001`: launcher exit status plus required metrics and manifest artifacts decide core success. Optional gallery rendering may warn, but by itself must not turn a completed pilot run into a failure.
- Use PATH `python` per interpreter policy.

## Prerequisite Status

### Satisfied

- The selected-item context marks this item independently runnable with no active backlog prerequisite.
- The official CNS file is already staged and verified:
  - bytes: `55,050,245,208`
  - MD5: `21969082d0e9524bcc4708e216148e60`
- The supervised CNS adapter, denormalized metric path, and artifact-writing path already exist.
- The canonical CNS shell is already fixed at skip-add plus pixelshuffle.
- The current CNS loss contract is already corrected to `mse` for this task family.
- The required profile IDs already exist in `scripts/studies/pdebench_image128/run_config.py`:
  - `spectral_resnet_bottleneck_base`
  - `spectral_resnet_bottleneck_noshare`
  - `spectral_resnet_bottleneck_shared_blocks8`
  - `spectral_resnet_bottleneck_shared_blocks10`
  - `spectral_resnet_bottleneck_noshare_blocks8`
  - `spectral_resnet_bottleneck_noshare_blocks10`
- `scripts/studies/pdebench_image128/cfd_cns.py` already supports `inspect`, `readiness`, `pilot`, and `benchmark`.
- Existing study tests already cover the CNS model-profile and runner surfaces named in the backlog-item check commands.
- The broader CNS prerequisites recorded in the progress ledger are already complete enough for this bounded lane:
  - official CNS data staged and verified
  - canonical skip-add plus pixelshuffle shell promoted
  - authored FFNO lane completed separately
  - GNOT lane completed separately
  - sibling `history_len=1` and modes-32 CNS lanes completed separately

### Open But Not Blocking

- No authoritative fresh architecture-ablation run roots exist yet under this item’s artifact root.
- The durable study summary, CNS summary sync, docs discoverability updates, and progress-ledger completion entry for this item do not exist yet.
- A prior capped `10`-epoch sharing compare exists in `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_spectral_weight_sharing_summary.md`, but this backlog item still needs fresh execution authority for the full approved sequence: fresh `10`-epoch sharing, fresh `40`-epoch sharing, fresh `40`-epoch shared-depth, and larger-cap finalist confirmation.
- Full-training CNS benchmark completeness remains open because this backlog item intentionally stays capped and `pilot`-mode only.

## Implementation Architecture

- **Contract Validation Unit:** `scripts/studies/pdebench_image128/{run_config.py,cfd_cns.py,reporting.py}`, `scripts/studies/run_pdebench_image128_suite.py`, and `tests/studies/test_pdebench_image128_{models,runner}.py` own the profile surface, mode behavior, and reporting contract. Default expectation: no code changes. Patch only a minimal blocker if the required deterministic checks fail.
- **Execution + Audit Unit:** fresh run roots under `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-hybrid-spectral-architecture-ablation/` own the inspect snapshot, frozen reference manifests, fresh `pilot` reruns, compare sidecars, ranking payloads, finalist-selection JSON, and larger-cap delta payload.
- **Interpretation + State Unit:** `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_cns_hybrid_spectral_arch_ablation_summary.md`, `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_2d_cfd_cns_summary.md`, `docs/studies/index.md`, `docs/index.md`, `state/NEURIPS-HYBRID-RESNET-2026/progress_ledger.json`, and optionally `docs/findings.md` own the durable interpretation and must preserve the capped-lane / benchmark-incomplete boundary.

## Concrete File And Artifact Targets

### Repo Surfaces That May Change

- Default expectation: no production code changes are required.
- Modify only if Task 1 exposes a real blocker:
  - `scripts/studies/pdebench_image128/run_config.py`
  - `scripts/studies/pdebench_image128/cfd_cns.py`
  - `scripts/studies/pdebench_image128/reporting.py`
  - `scripts/studies/run_pdebench_image128_suite.py`
  - `tests/studies/test_pdebench_image128_models.py`
  - `tests/studies/test_pdebench_image128_runner.py`
- Expected durable documentation/state updates at close:
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_cns_hybrid_spectral_arch_ablation_summary.md`
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_2d_cfd_cns_summary.md`
  - `docs/studies/index.md`
  - `docs/index.md`
  - `state/NEURIPS-HYBRID-RESNET-2026/progress_ledger.json`
  - `docs/findings.md` only if the fresh result establishes a reusable rule that materially changes or strengthens current CNS guidance

### Required Generated Artifacts

- Study root: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-hybrid-spectral-architecture-ablation/`
- Required run roots:
  - `inspect-<timestamp>/`
  - `cns-hybrid-spectral-sharing-10ep-<timestamp>/`
  - `cns-hybrid-spectral-sharing-40ep-<timestamp>/`
  - `cns-hybrid-spectral-depth-shared-40ep-<timestamp>/`
  - `cns-hybrid-spectral-finalists-1024cap-40ep-<timestamp>/`
- Frozen reference manifests:
  - `reference_runs_10ep.json`
  - `reference_runs_40ep.json`
- Required sidecars:
  - `compare_sharing_10ep_against_existing.json`
  - `compare_sharing_10ep_against_existing.csv`
  - `compare_sharing_40ep_against_existing.json`
  - `compare_sharing_40ep_against_existing.csv`
  - `compare_depth_40ep_against_existing.json`
  - `compare_depth_40ep_against_existing.csv`
  - `sharing_10ep_ranking.json`
  - `sharing_40ep_ranking.json`
  - `depth_40ep_ranking.json`
  - `selected_finalists_1024cap.json`
  - `finalist_delta_1024cap.json`
- Conditionally required if `unique_finalist_count == 2`:
  - `compare_finalists_1024cap_40ep_within_run.json`
  - `compare_finalists_1024cap_40ep_within_run.csv`
- Required pointer files:
  - `inspect_run_root.txt`
  - `stage1_sharing_10ep_run_root.txt`
  - `stage2_sharing_40ep_run_root.txt`
  - `stage3_depth_40ep_run_root.txt`
  - `stage4_finalists_1024cap_run_root.txt`
- Required preflight verification files:
  - `verification/preflight_pytest.log`
  - `verification/preflight_compileall.log`
- Conditionally required final verification files, only if any repo surface changes during execution:
  - `verification/final_pytest.log`
  - `verification/final_compileall.log`

## Required Deterministic Checks

These are the binding backlog-item `check_commands`. No narrower replacement is justified because they already cover the exact model/runner surfaces this plan may touch.

```bash
pytest -q tests/studies/test_pdebench_image128_models.py tests/studies/test_pdebench_image128_runner.py
python -m compileall -q scripts/studies/pdebench_image128 scripts/studies/run_pdebench_image128_suite.py
```

Archive both outputs under the study-root `verification/` directory before long runs and again at close if any repo surface changed.

## Frozen Existing Reference Rows

- Required `10`-epoch quantitative context rows:
  - `fno_base`: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-2d-cfd-cns/readiness-cap-20260421T183717Z-10ep-mse`
  - `unet_strong`: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-2d-cfd-cns/readiness-cap-20260421T183717Z-10ep-mse`
- Optional `10`-epoch context-only hybrid provenance row:
  - `hybrid_resnet_cns`: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-ffno-bottleneck-cns-compare/readiness-cap-20260421T203200Z-ffno-cns-10ep`
  - mark `quantitative_eligibility="context_only_not_compared"`
  - reason: the older artifact is useful shell context but is not the fresh study row being generated here
- Required `40`-epoch context rows:
  - `hybrid_resnet_cns`: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-markov-history1-compare/history2-hybrid-cns-pilot-40ep-20260423T223143Z`
  - `fno_base`: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-2d-cfd-cns/readiness-cap-20260421T185926Z-40ep-mse`
  - `unet_strong`: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-2d-cfd-cns/readiness-cap-20260421T185926Z-40ep-mse`

## Execution Tasks

### Task 1: Preflight The Existing Contract And Freeze Reference Manifests

**Files:**
- Modify only if checks fail: `scripts/studies/pdebench_image128/run_config.py`, `scripts/studies/pdebench_image128/cfd_cns.py`, `scripts/studies/pdebench_image128/reporting.py`, `scripts/studies/run_pdebench_image128_suite.py`
- Test: `tests/studies/test_pdebench_image128_models.py`, `tests/studies/test_pdebench_image128_runner.py`
- Generate: study-root verification logs, `inspect-<timestamp>/`, `reference_runs_10ep.json`, `reference_runs_40ep.json`

- [ ] Run the required deterministic checks before any edits or long runs and archive their output under `verification/preflight_*.log`.
- [ ] Launch one fresh `inspect` run to snapshot `hdf5_metadata.json`, `dataset_manifest.json`, `split_manifest.json`, and the effective history-window contract for `history_len=2`.

```bash
python scripts/studies/run_pdebench_image128_suite.py \
  --task 2d_cfd_cns \
  --mode inspect \
  --data-root /home/ollie/Documents/pdebench-data \
  --output-root .artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-hybrid-spectral-architecture-ablation/inspect-<timestamp> \
  --history-len 2 \
  --max-train-trajectories 512 \
  --max-val-trajectories 64 \
  --max-test-trajectories 64 \
  --max-windows-per-trajectory 8 \
  --device cuda \
  --num-workers 0
```

- [ ] Freeze `reference_runs_10ep.json` and `reference_runs_40ep.json` using the exact reference roots listed above. Do not silently substitute newer or prettier roots.
- [ ] If the deterministic checks fail, patch only the smallest blocker needed for this item, then rerun the same checks before continuing.

**Verification**

- [ ] `verification/preflight_pytest.log` and `verification/preflight_compileall.log` exist.
- [ ] `inspect_run_root.txt` points at a fresh inspect root whose `hdf5_metadata.json`, `dataset_manifest.json`, and `split_manifest.json` all parse successfully.
- [ ] Both reference manifests parse and preserve the exact roots and eligibility labels above.

### Task 2: Run Fresh `10`-Epoch Sharing Pilot

**Files:**
- Generate: `cns-hybrid-spectral-sharing-10ep-<timestamp>/`
- Generate: `compare_sharing_10ep_against_existing.json/csv`, `sharing_10ep_ranking.json`, `stage1_sharing_10ep_run_root.txt`

- [ ] Run `pilot` mode in tmux with `ptycho311` active for `spectral_resnet_bottleneck_base,spectral_resnet_bottleneck_noshare` under the fixed capped `512 / 64 / 64`, `history_len=2`, `mse`, batch `4`, `epochs=10` contract.

```bash
python scripts/studies/run_pdebench_image128_suite.py \
  --task 2d_cfd_cns \
  --mode pilot \
  --data-root /home/ollie/Documents/pdebench-data \
  --output-root .artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-hybrid-spectral-architecture-ablation/cns-hybrid-spectral-sharing-10ep-<timestamp> \
  --profiles spectral_resnet_bottleneck_base,spectral_resnet_bottleneck_noshare \
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

- [ ] After PID exit `0`, verify the fresh run root contains `comparison_summary.json`, `comparison_summary.csv`, `metrics_*.json`, `model_profile_*.json`, `dataset_manifest.json`, and `split_manifest.json`.
- [ ] Write `compare_sharing_10ep_against_existing.json/csv` by comparing the fresh spectral rows against the frozen `10`-epoch context rows from `reference_runs_10ep.json`.
- [ ] Write `sharing_10ep_ranking.json` ranking only the two fresh spectral rows by `relative_l2`, then `err_nRMSE`, then `fRMSE_high`.

**Verification**

- [ ] The fresh run root preserves the fixed shell and capped contract in emitted `model_profile_*.json` and manifests.
- [ ] The compare sidecar labels the result as capped decision-support evidence only.
- [ ] `sharing_10ep_ranking.json` references only the two fresh spectral rows.

### Task 3: Run Fresh `40`-Epoch Sharing Pilot

**Files:**
- Generate: `cns-hybrid-spectral-sharing-40ep-<timestamp>/`
- Generate: `compare_sharing_40ep_against_existing.json/csv`, `sharing_40ep_ranking.json`, `stage2_sharing_40ep_run_root.txt`

- [ ] Run `pilot` mode in tmux with the same fixed contract as Task 2, changing only `epochs=40`.

```bash
python scripts/studies/run_pdebench_image128_suite.py \
  --task 2d_cfd_cns \
  --mode pilot \
  --data-root /home/ollie/Documents/pdebench-data \
  --output-root .artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-hybrid-spectral-architecture-ablation/cns-hybrid-spectral-sharing-40ep-<timestamp> \
  --profiles spectral_resnet_bottleneck_base,spectral_resnet_bottleneck_noshare \
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

- [ ] Verify the fresh run root is complete and fresh by PID exit `0` plus required artifacts present.
- [ ] Write `compare_sharing_40ep_against_existing.json/csv` against the frozen `40`-epoch reference manifest.
- [ ] Write `sharing_40ep_ranking.json` using the same ordering rule: `relative_l2`, then `err_nRMSE`, then `fRMSE_high`.

**Verification**

- [ ] The fresh run root preserves the fixed shell and capped contract.
- [ ] The compare sidecar keeps `hybrid_resnet_cns`, `fno_base`, and `unet_strong` as context rows only, not as newly rerun study rows.
- [ ] `sharing_40ep_ranking.json` clearly identifies the fresh sharing-tranche winner.

### Task 4: Run Fresh `40`-Epoch Shared-Depth Pilot

**Files:**
- Generate: `cns-hybrid-spectral-depth-shared-40ep-<timestamp>/`
- Generate: `compare_depth_40ep_against_existing.json/csv`, `depth_40ep_ranking.json`, `stage3_depth_40ep_run_root.txt`

- [ ] Run `pilot` mode in tmux for `spectral_resnet_bottleneck_base,spectral_resnet_bottleneck_shared_blocks8,spectral_resnet_bottleneck_shared_blocks10` under the same fixed capped contract as Task 3.

```bash
python scripts/studies/run_pdebench_image128_suite.py \
  --task 2d_cfd_cns \
  --mode pilot \
  --data-root /home/ollie/Documents/pdebench-data \
  --output-root .artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-hybrid-spectral-architecture-ablation/cns-hybrid-spectral-depth-shared-40ep-<timestamp> \
  --profiles spectral_resnet_bottleneck_base,spectral_resnet_bottleneck_shared_blocks8,spectral_resnet_bottleneck_shared_blocks10 \
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

- [ ] Do not include `spectral_resnet_bottleneck_noshare_blocks8` or `spectral_resnet_bottleneck_noshare_blocks10`.
- [ ] After successful completion, write `compare_depth_40ep_against_existing.json/csv` against the frozen `40`-epoch context rows.
- [ ] Write `depth_40ep_ranking.json` using the same ordering rule: `relative_l2`, then `err_nRMSE`, then `fRMSE_high`.

**Verification**

- [ ] The fresh run root preserves the fixed shell and capped contract for all three shared-depth rows.
- [ ] The compare sidecar remains capped decision-support evidence only.
- [ ] `depth_40ep_ranking.json` clearly identifies the fresh shared-depth finalist.

### Task 5: Select Unique Finalists And Run Larger-Cap Confirmation

**Files:**
- Generate: `selected_finalists_1024cap.json`
- Generate: `cns-hybrid-spectral-finalists-1024cap-40ep-<timestamp>/`
- Generate: `finalist_delta_1024cap.json`, `stage4_finalists_1024cap_run_root.txt`
- Conditionally generate: `compare_finalists_1024cap_40ep_within_run.json/csv`

- [ ] Select one finalist from `sharing_40ep_ranking.json` and one finalist from `depth_40ep_ranking.json` using the already-recorded ordering rule.
- [ ] If both tranches nominate the same profile, record `unique_finalist_count=1` and rerun only that one row on the `1024 / 128 / 128` cap.
- [ ] If the finalists differ, rerun exactly those two rows together on the `1024 / 128 / 128`, `history_len=2`, `mse`, batch `4`, `epochs=40`, `max_windows_per_trajectory=8` contract.

```bash
python scripts/studies/run_pdebench_image128_suite.py \
  --task 2d_cfd_cns \
  --mode pilot \
  --data-root /home/ollie/Documents/pdebench-data \
  --output-root .artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-hybrid-spectral-architecture-ablation/cns-hybrid-spectral-finalists-1024cap-40ep-<timestamp> \
  --profiles <finalist_csv_from_selection_json> \
  --history-len 2 \
  --epochs 40 \
  --batch-size 4 \
  --max-train-trajectories 1024 \
  --max-val-trajectories 128 \
  --max-test-trajectories 128 \
  --max-windows-per-trajectory 8 \
  --device cuda \
  --num-workers 0
```

- [ ] Write `selected_finalists_1024cap.json` with the source tranche winner, unique-finalist count, and selection rationale.
- [ ] Write `finalist_delta_1024cap.json` summarizing how each finalist changed from its `512 / 64 / 64` pilot row to its `1024 / 128 / 128` confirmation row.
- [ ] If two unique finalists ran, also write `compare_finalists_1024cap_40ep_within_run.json/csv`.

**Verification**

- [ ] No extra profiles appear in the larger-cap run beyond the unique finalists.
- [ ] `selected_finalists_1024cap.json` and `finalist_delta_1024cap.json` parse and cite the exact source ranking files.
- [ ] If two finalists ran, the within-run compare sidecar exists; if one finalist ran, the absence is explicitly explained in `selected_finalists_1024cap.json`.

### Task 6: Publish Durable Interpretation And Initiative State

**Files:**
- Create: `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_cns_hybrid_spectral_arch_ablation_summary.md`
- Modify: `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_2d_cfd_cns_summary.md`
- Modify: `docs/studies/index.md`
- Modify: `docs/index.md`
- Modify: `state/NEURIPS-HYBRID-RESNET-2026/progress_ledger.json`
- Optional modify: `docs/findings.md`

- [ ] Write the durable study summary with:
  - the fixed shell and fixed capped contract
  - fresh `10`-epoch and `40`-epoch sharing outcomes
  - fresh `40`-epoch shared-depth outcome
  - larger-cap finalist confirmation outcome
  - explicit statement that all results remain capped decision-support evidence only
- [ ] Sync the broader CNS summary so the canonical CNS story references this bounded Hybrid-spectral lane without promoting it to benchmark-complete evidence.
- [ ] Update `docs/studies/index.md` and `docs/index.md` so the new summary is discoverable.
- [ ] Add a new `progress_ledger.json` completion/update entry for this backlog item, including plan path, artifact root, summary path, evidence scope, metric interpretation, verification commands, and affected surfaces.
- [ ] Update `docs/findings.md` only if the fresh result changes or materially strengthens a durable CNS rule; otherwise leave findings unchanged.
- [ ] If any repo surface changed, rerun the required deterministic checks and archive their output under `verification/final_*.log`.

**Verification**

- [ ] Summary and index docs mention the fixed-shell, capped-only boundary explicitly.
- [ ] `progress_ledger.json` parses after the update.
- [ ] If repo files changed, `verification/final_pytest.log` and `verification/final_compileall.log` exist and correspond to the required deterministic checks.

## Completion Criteria

- Fresh `pilot` run roots exist for the approved `10`-epoch sharing, `40`-epoch sharing, `40`-epoch shared-depth, and `1024 / 128 / 128` finalist-confirmation tranches.
- Every compare tranche preserves equal footing and the fixed canonical CNS shell.
- The final durable summary, CNS summary sync, docs discoverability updates, and progress-ledger entry all exist and keep the result labeled as capped decision-support evidence only.
- The plan’s required deterministic checks are archived preflight, and the final `verification/final_*.log` artifacts are required only if repo surfaces changed and the checks were rerun at close.
