# CNS Hybrid-Spectral To FFNO Parameter-Space Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run the CNS-only half of the Hybrid-spectral to FFNO parameter-space study under the capped PDEBench `2d_cfd_cns` decision-support contract, isolating which one-axis shell changes matter between the current Hybrid-spectral lane and the repo-local FFNO-family lane.

**Architecture:** Reuse the fixed `history_len=2` CNS image-suite contract and the completed Hybrid-spectral architecture-ablation plus FFNO local-conv lanes as pinned reference authorities. Add only the minimum manual-only profile support needed for the missing encoder/downsampling and decoder bridge rows, freeze inspect and reference manifests before training, then execute a staged `green checks -> inspect -> 10 epoch compare -> selective 40 epoch follow-up -> durable summary` flow. Every fresh row must be attributable to one declared shell-axis delta against the same `spectral_resnet_bottleneck_base` anchor.

**Tech Stack:** PATH `python`, PyTorch/Lightning, `scripts/studies/pdebench_image128/`, `scripts/studies/run_pdebench_image128_suite.py`, tmux plus `ptycho311` for long CNS runs, Markdown/JSON/CSV/PNG artifacts under `.artifacts/NEURIPS-HYBRID-RESNET-2026/`.

---

## Selected Backlog Objective

- Execute the PDEBench CNS portion of the Hybrid-spectral to FFNO parameter-space study under the capped `2d_cfd_cns` contract.
- Separate encoder/downsampling, decoder, and bottleneck-family interpretation with one-axis-at-a-time evidence.
- Produce a CNS-only summary that says which intermediate rows, if any, deserve later full-training CNS or Phase 3 CDI follow-up.

## Scope

- Task: `2d_cfd_cns` only.
- Dataset: `/home/ollie/Documents/pdebench-data/2d_cfd_cns/2D_CFD_Rand_M1.0_Eta0.01_Zeta0.01_periodic_128_Train.hdf5`.
- Fixed capped contract unless a narrow harness repair is required:
  - `history_len=2`
  - training loss `mse`
  - batch size `4`
  - `max_windows_per_trajectory=8`
  - main capped split family `512 / 64 / 64`
  - optional tie-break confirmation split `1024 / 128 / 128` only if the axis story remains unresolved after `40` epochs
  - metrics: `err_RMSE`, `err_nRMSE`, `relative_l2`, `fRMSE_low`, `fRMSE_mid`, `fRMSE_high`
- Reuse same-contract historical anchors where the pinned roots and compare surfaces already exist and parse cleanly.

## Explicit Non-Goals

- Do not run CDI, ptycho, `lines128`, or any Phase 3 work from this item.
- Do not turn capped CNS results into full-training benchmark or paper-grade claims.
- Do not reopen the roadmap decision that this lane is capped decision-support evidence only.
- Do not promote a manual-only profile into a default bundle from this item alone.
- Do not expand into a Cartesian sweep, a broader FFNO study, a broader Hybrid-spectral study, `/home/ollie/Documents/neurips/` artifact work, or unrelated backlog items.

## Binding Constraints And Prerequisite Status

- Steering, design, and roadmap are binding:
  - remain inside the Phase 2 PDEBench lane
  - keep equal-footing comparisons explicit
  - preserve the current CNS metric, split, and protocol boundaries
  - keep all conclusions labeled as capped decision-support evidence
- Required prerequisites already completed in the progress ledger:
  - `2026-04-22-pdebench-cns-hybrid-spectral-architecture-ablation` completed on `2026-04-28`
  - `2026-04-27-pdebench-ffno-convolutional-features-cns` completed on `2026-04-28`
- The authoritative prerequisite summaries for this plan are:
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_cns_hybrid_spectral_arch_ablation_summary.md`
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_ffno_convolutional_features_cns_summary.md`
- The larger-budget `modes24` convergence follow-up may remain in progress elsewhere, but it is not a prerequisite and must not serialize this backlog item.
- Ordinary import, path, environment, or test-harness failures are not grounds to mark this item `BLOCKED`. Diagnose, patch narrowly, and rerun first. Reserve `BLOCKED` for missing resources, unavailable hardware, unusable pinned authorities after a narrow repair attempt, roadmap conflict, external dependency outside current authority, or a required user decision.

## Fixed Fairness Contract

- Task and data identity must stay fixed across all fresh rows and reused anchors.
- The shell contract for fresh rows must stay anchored to `spectral_resnet_bottleneck_base` except for the explicitly approved one-axis delta under test.
- The ordering keys for promotion and interpretation are:
  - `relative_l2`
  - then `err_nRMSE`
  - then `fRMSE_high`
- All fresh and reused results in this plan must remain labeled `capped_decision_support_only`.

## Pinned Reused Reference Authorities

Implementation must copy from these authorities, not rediscover substitutes by scanning the artifact tree.

- Hybrid-spectral prerequisite lane:
  - root: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-hybrid-spectral-architecture-ablation`
  - pinned manifest authorities:
    - `reference_runs_10ep.json`
    - `reference_runs_40ep.json`
  - pinned fresh-result authorities:
    - `cns-hybrid-spectral-sharing-10ep-20260428T032825Z`
    - `cns-hybrid-spectral-sharing-40ep-20260428T035154Z`
    - `cns-hybrid-spectral-depth-shared-40ep-20260428T043715Z`
    - `cns-hybrid-spectral-finalists-1024cap-40ep-20260428T054559Z` only as an optional tie-break reference
- FFNO local-conv prerequisite lane:
  - root: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-ffno-convolutional-features-cns`
  - pinned manifest authorities:
    - `reference_runs_10ep.json`
    - `reference_runs_40ep.json`
  - pinned fresh-result authorities:
    - `cns-ffno-localconv-10ep-20260428T082501Z`
    - `cns-ffno-close-backfill-40ep-20260428T084852Z`
    - `cns-ffno-localconv-40ep-20260428T090626Z`
- Reused external same-contract context already pinned by those manifests:
  - `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-author-ffno-equal-footing-cns/author-ffno-10ep-20260422T232119Z`
  - `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-author-ffno-equal-footing-cns/author-ffno-40ep-20260422T234340Z`
- Reused local same-contract context already pinned by those manifests:
  - `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-ffno-bottleneck-cns-compare/readiness-cap-20260421T203200Z-ffno-cns-10ep`
  - `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-ffno-bottleneck-cns-compare/readiness-cap-20260421T221008Z-spectral40ep`
  - `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-2d-cfd-cns/readiness-cap-20260421T183717Z-10ep-mse`
  - `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-2d-cfd-cns/readiness-cap-20260421T185926Z-40ep-mse`

## Approved Study Matrix

This matrix is the execution boundary. Do not add rows, swap anchors, or convert reused context into fresh reruns unless a narrow support backfill is required because a pinned authority is unusable.

### Reused quantitative anchors

| Role | Profile ID | Budget | Source authority | Allowed use |
| --- | --- | --- | --- | --- |
| Hybrid-spectral aggregate anchor | `spectral_resnet_bottleneck_base` | `10` and `40` epochs | pinned spectral/FFNO manifest authorities | direct anchor for fresh encoder and decoder probes |
| Hybrid-spectral higher-frequency context | `spectral_resnet_bottleneck_shared_blocks10` | `40` epochs | `.../cns-hybrid-spectral-depth-shared-40ep-20260428T043715Z` | optional deeper-spectral context only |
| Repo-local FFNO-close anchor | `ffno_bottleneck_base` | `10` and `40` epochs | pinned FFNO manifests plus `40`-epoch backfill | fixed bottleneck-axis endpoint |
| Repo-local FFNO with local branch | `ffno_bottleneck_localconv_base` | `10` and `40` epochs | pinned FFNO local-conv roots | fixed stronger repo-local FFNO-family endpoint |
| Local pure-Hybrid context | `hybrid_resnet_cns` | `10` epochs reused, `40` epochs optional only | pinned existing reference manifests | optional local-only bottleneck context |
| Authored FFNO context | `author_ffno_cns_base` | `10` and `40` epochs | pinned authored-FFNO roots | external same-contract context only |
| Local benchmark context | `fno_base`, `unet_strong` | `10` and `40` epochs | pinned local benchmark roots | optional comparison context only |

### Fresh rows approved for `10` epochs

| Axis | Fresh profile ID | Direct anchor | Exact knob delta | Required implementation surface |
| --- | --- | --- | --- | --- |
| encoder/downsampling | `spectral_resnet_bottleneck_base_down1` | `spectral_resnet_bottleneck_base` | only `hybrid_downsample_steps: 2 -> 1` | manual-only profile support in `scripts/studies/pdebench_image128/run_config.py`; existing model builder should already accept the knob |
| decoder | `spectral_resnet_bottleneck_base_transpose` | `spectral_resnet_bottleneck_base` | only `hybrid_upsampler: pixelshuffle -> cyclegan_transpose` | manual-only profile support in `scripts/studies/pdebench_image128/run_config.py`; existing model builder should already accept the knob |

The bottleneck axis is already defined by the completed same-contract rows. This plan does not authorize inventing a new bottleneck-family experiment.

### Promotion rules for `40` epochs

- Promote `spectral_resnet_bottleneck_base_down1` only if the `10`-epoch compare shows a real encoder/downsampling trade-off against `spectral_resnet_bottleneck_base` on the ordering keys, or leaves the axis genuinely ambiguous.
- Promote `spectral_resnet_bottleneck_base_transpose` only if the `10`-epoch compare shows a real decoder trade-off against `spectral_resnet_bottleneck_base` on the same ordering keys, or leaves the axis genuinely ambiguous.
- Do not rerun `ffno_bottleneck_base`, `ffno_bottleneck_localconv_base`, `author_ffno_cns_base`, `fno_base`, `unet_strong`, or `spectral_resnet_bottleneck_shared_blocks10` from this item unless a pinned authority is unusable and the narrow replacement is recorded as support work rather than as a new study row.
- If the axis story is still unresolved after promoted `40`-epoch rows, allow at most one `1024 / 128 / 128` tie-break confirmation pass and record why it was necessary.

## Implementation Architecture

- **Profile matrix unit:** `scripts/studies/pdebench_image128/run_config.py` owns the two approved fresh profile IDs, their manual-only status, and the guarantee that they do not join primary/readiness profile bundles.
- **Model wiring unit:** `scripts/studies/pdebench_image128/models.py` owns execution of the approved knob deltas through existing `hybrid_downsample_steps` and `hybrid_upsampler` surfaces. `ptycho_torch/generators/*` stays untouched unless a proven builder gap blocks one of those existing knobs.
- **Runner and reporting unit:** `scripts/studies/pdebench_image128/cfd_cns.py`, `scripts/studies/pdebench_image128/reporting.py`, and `scripts/studies/run_pdebench_image128_suite.py` own inspect proof, reference-manifest assembly, same-contract execution, and compare collation.
- **Evidence unit:** `tests/studies/test_pdebench_image128_models.py`, `tests/studies/test_pdebench_image128_runner.py`, the durable CNS summaries, and `state/NEURIPS-HYBRID-RESNET-2026/progress_ledger.json` own verification and durable knowledge.

## Concrete File And Artifact Targets

**Code likely to change**

- `scripts/studies/pdebench_image128/run_config.py`
- `scripts/studies/pdebench_image128/cfd_cns.py`
- `scripts/studies/pdebench_image128/reporting.py`
- `scripts/studies/run_pdebench_image128_suite.py`
- `scripts/studies/pdebench_image128/models.py` only if existing support for `hybrid_downsample_steps` or `hybrid_upsampler` proves incomplete
- `ptycho_torch/generators/hybrid_resnet.py` only if an unexpected builder gap forces a narrow generator-side fix

**Tests likely to change**

- `tests/studies/test_pdebench_image128_models.py`
- `tests/studies/test_pdebench_image128_runner.py`
- `tests/torch/test_ffno_bottleneck.py` only if generator behavior changes

**Durable docs and state likely to change**

- `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_cns_hybrid_spectral_ffno_parameter_space_summary.md`
- `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_2d_cfd_cns_summary.md`
- `state/NEURIPS-HYBRID-RESNET-2026/progress_ledger.json`
- `docs/index.md` and `docs/studies/index.md` only if the new durable summary needs discoverability updates

**Execution artifacts**

- `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-hybrid-spectral-ffno-parameter-space/`
- `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-28-pdebench-cns-hybrid-spectral-ffno-parameter-space/execution_report.md`

## Mandatory Deterministic Gate

These backlog check commands are mandatory and must pass before any expensive inspect or training step:

```bash
pytest -q tests/studies/test_pdebench_image128_models.py tests/studies/test_pdebench_image128_runner.py
python -m compileall -q scripts/studies/pdebench_image128 scripts/studies/run_pdebench_image128_suite.py
```

Rules:

- If either command fails, diagnose and fix the narrow cause first, then rerun until green.
- Do not launch inspect, `10`-epoch, or `40`-epoch CNS runs while either mandatory check is red.
- If generator code changes, add this narrow selector before rerunning the gate:

```bash
pytest -q tests/torch/test_ffno_bottleneck.py
```

## Task Checklist

### Task 1: Reconfirm Authorities And Freeze The Study Contract

- [ ] Re-read the two prerequisite summaries and confirm that the pinned reused anchors in this plan still match the authoritative ledger paths and intended uses.
- [ ] Run the mandatory deterministic gate immediately from repo root. If it fails, fix the narrow cause before continuing.
- [ ] Write `study_matrix.json` under this item's artifact root as a machine-readable copy of this plan's approved matrix. It must enumerate:
  - the pinned reused anchors
  - the two approved fresh `10`-epoch rows
  - each row's axis label, direct anchor, and evidence role
  - whether each row is reused context, fresh execution, or conditional `40`-epoch promotion
- [ ] Keep the row count bounded to the approved matrix unless a narrow support backfill is required because a pinned authority is unusable.
- [ ] Do not add a fresh bottleneck-family row in Task 1 or later tasks.

**Verification**

- Mandatory deterministic gate passes.
- `study_matrix.json` parses and matches the fixed CNS contract.
- Implementation does not proceed to inspect or training until the gate is green.

### Task 2: Add Only The Missing Manual-Only Profile Support

- [ ] Inspect `scripts/studies/pdebench_image128/run_config.py` and `scripts/studies/pdebench_image128/models.py` against the two approved fresh rows.
- [ ] Add only the minimum support needed for:
  - `spectral_resnet_bottleneck_base_down1`
  - `spectral_resnet_bottleneck_base_transpose`
- [ ] Keep each profile attributable to one axis only:
  - encoder/downsampling probe changes only `hybrid_downsample_steps`
  - decoder probe changes only `hybrid_upsampler`
- [ ] Mark both rows `manual-only`; do not add them to `PRIMARY_CFD_CNS_PROFILE_IDS` or `READINESS_CFD_CNS_PROFILE_IDS`.
- [ ] Preserve existing primary CNS bundle membership and the canonical full-training row `hybrid_resnet_cns`.
- [ ] Ensure `describe_model(...)` and emitted model-profile JSON still capture parameter count and full profile config for each fresh row.

**Verification**

- Add or update focused tests in `tests/studies/test_pdebench_image128_models.py` so each profile builds, preserves the intended shell invariants, and cannot be silently promoted into a primary bundle.
- Add explicit config-delta tests proving:
  - `spectral_resnet_bottleneck_base_down1` changes only `hybrid_downsample_steps`
  - `spectral_resnet_bottleneck_base_transpose` changes only `hybrid_upsampler`
- If generator changes were required, run `pytest -q tests/torch/test_ffno_bottleneck.py`.
- Rerun the mandatory deterministic gate and require green results before Task 3.

### Task 3: Produce Inspect Proof And Freeze Reference Manifests

- [ ] Run the CNS inspect path for this backlog item before any training so the artifact root contains:
  - `inspection_manifest.json`
  - `dataset_manifest.json`
  - `split_manifest.json`
  - required schema or HDF5 metadata
- [ ] Build frozen `reference_runs_10ep.json` and `reference_runs_40ep.json` for this item by copying from the pinned prerequisite authorities, not by ad hoc artifact discovery.
- [ ] Seed `reference_runs_10ep.json` from the FFNO local-conv prerequisite manifest so the required rows remain `author_ffno_cns_base`, `ffno_bottleneck_base`, and `spectral_resnet_bottleneck_base`, with `hybrid_resnet_cns`, `fno_base`, and `unet_strong` preserved only as optional context.
- [ ] Seed `reference_runs_40ep.json` from the FFNO local-conv prerequisite manifest and add `spectral_resnet_bottleneck_shared_blocks10` only as optional deeper-spectral context if needed for `40`-epoch interpretation.
- [ ] Validate that every referenced root uses the same task, dataset file, split family, `history_len=2`, `max_windows_per_trajectory=8`, training loss `mse`, and metric family.
- [ ] If a fairness-critical same-contract anchor is missing a required compare surface, backfill that anchor first and record it as support work rather than a new study row.

**Verification**

- Parse the inspect artifacts and both reference manifests with a small JSON validation command.
- Confirm inspect proof exists before any expensive compare launch is treated as valid.
- Keep the mandatory deterministic gate green if any code changed while wiring inspect or manifest logic.

### Task 4: Run The Bounded `10`-Epoch Compare

- [ ] Launch only the two approved fresh rows at `10` epochs:
  - `spectral_resnet_bottleneck_base_down1`
  - `spectral_resnet_bottleneck_base_transpose`
- [ ] Use tmux plus `ptycho311` for long runs.
- [ ] Follow the long-run guardrail exactly:
  - track the exact launched PID
  - wait on that PID rather than using broad `pgrep` loops
  - do not launch a duplicate run writing to the same output root
  - require exit code `0` plus freshly written required outputs before treating the run as complete
- [ ] Collate cross-run compare outputs against the frozen `10`-epoch reference manifest:
  - `compare_10ep_against_existing.json`
  - `compare_10ep_against_existing.csv`
  - fixed-sample prediction gallery
  - fixed-sample error gallery
- [ ] Record parameter-count and runtime context for each fresh row.
- [ ] Interpret the bottleneck continuum using the pinned reused rows already present in the reference manifest; do not treat that as permission to add a fresh bottleneck experiment.

**Verification**

- Confirm tracked launcher exit-code files, metrics JSON, model-profile JSON, and compare outputs exist for every fresh row.
- Parse the `10`-epoch compare JSON and CSV outputs.
- Keep the evidence label explicit as capped decision-support only.

### Task 5: Promote Only The Necessary `40`-Epoch Follow-Up Rows

- [ ] Apply the predeclared promotion rules and choose the smallest necessary `40`-epoch follow-up set.
- [ ] Promote `spectral_resnet_bottleneck_base_down1` only if the encoder/downsampling axis is genuinely competitive or ambiguous at `10` epochs.
- [ ] Promote `spectral_resnet_bottleneck_base_transpose` only if the decoder axis is genuinely competitive or ambiguous at `10` epochs.
- [ ] Do not rerun every exploratory row at `40` epochs by default. Promote only direct per-axis winners, genuine ties, or rows made ambiguous by a required support backfill.
- [ ] Keep the contract identical to Task 4 unless a documented harness bug forces a narrow fix and rerun.
- [ ] Write the `40`-epoch cross-run compare outputs:
  - `compare_40ep_against_existing.json`
  - `compare_40ep_against_existing.csv`
  - fixed-sample prediction gallery
  - fixed-sample error gallery
- [ ] If the axis story is still unresolved after promoted `40`-epoch rows, use at most one `1024 / 128 / 128` tie-break confirmation pass and record why it was needed.
- [ ] Use `spectral_resnet_bottleneck_shared_blocks10` only as optional context when deciding whether a deeper shared spectral explanation is stronger than either fresh shell probe.

**Verification**

- Require tracked exit code `0`, fresh metrics/model-profile artifacts, and parseable compare JSON/CSV outputs for every promoted row.
- Confirm every `40`-epoch row still matches the frozen fairness contract.
- If a larger-cap tie-break was required, verify the rationale and contract are recorded in the durable summary and ledger.

### Task 6: Write Durable Reporting And Update Initiative State

- [ ] Write `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_cns_hybrid_spectral_ffno_parameter_space_summary.md` with:
  - the fixed fairness contract
  - the exact profile matrix
  - per-axis interpretation for encoder/downsampling, decoder, and bottleneck
  - runtime and parameter-count context
  - any carry-forward recommendation for later full-training CNS or CDI planning
  - an explicit claim boundary that all conclusions remain capped decision-support evidence only
- [ ] Update `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_2d_cfd_cns_summary.md` with the new lane outcome.
- [ ] Update `state/NEURIPS-HYBRID-RESNET-2026/progress_ledger.json` with:
  - completion decision
  - artifact root
  - reference manifest paths
  - fresh run roots
  - compare artifact paths
  - verification evidence
- [ ] Update `docs/index.md` and `docs/studies/index.md` only if the new durable summary should become discoverable from those maps.
- [ ] Keep the execution report path current at `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-28-pdebench-cns-hybrid-spectral-ffno-parameter-space/execution_report.md`.

**Verification**

- Confirm the new summary, updated CNS summary, updated ledger entry, and execution report all exist.
- Run a required-output validation command that parses the summary-linked JSON artifacts and checks the durable summary path.
- Rerun the mandatory deterministic gate if any code changed during reporting support work.

## Completion Criteria

- The final summary distinguishes encoder/downsampling, decoder, and bottleneck effects on CNS without drifting into CDI, broader FFNO scope, or full-training claims.
- Every reported fresh row records its direct anchor, changed axis, capped contract, parameter count, runtime, and evidence scope.
- The item leaves behind reusable same-contract `10`-epoch and `40`-epoch reference manifests for later CNS follow-ups.
- Any carry-forward recommendation is framed only as a candidate for later full-training or CDI planning, not as a promoted default or paper row.

## Execution Notes

- Use PATH `python`; do not introduce repo-specific interpreter wrappers.
- Use tmux for long runs and activate `ptycho311` there.
- Do not create worktrees.
- Do not modify `ptycho/model.py`, `ptycho/diffsim.py`, or `ptycho/tf_helper.py`.
