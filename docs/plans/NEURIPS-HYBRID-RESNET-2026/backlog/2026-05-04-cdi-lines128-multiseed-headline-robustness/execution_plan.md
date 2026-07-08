# Lines128 Multi-Seed Headline Robustness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:executing-plans` or `superpowers:subagent-driven-development` to implement this plan task-by-task. Keep this file as the execution authority for the selected backlog item.

**Goal:** Strengthen the headline `lines128` CDI evidence by reusing the accepted `seed=3` rows where their audits pass, adding only the missing seeds needed for a fixed three-seed aggregate on the current headline row roster, and publishing an append-only training-seed-robustness bundle with per-seed values plus mean and standard deviation.

**Architecture:** Extend the existing `lines128` paper-benchmark orchestration so it can audit previously accepted single-seed roots, determine the missing seed-row matrix for a pinned seed set, launch only the missing rows under the unchanged CDI contract, and collate one additive multiseed bundle with explicit lineage, row-level blocker handling, and aggregate table payloads. Keep current single-seed authorities and paper-refresh outputs intact; any multiseed manuscript-facing tables must be new additive outputs with explicit source lineage.

**Tech Stack:** PATH `python`, `ptycho311` for long-running launches, PyTorch/Lightning, `scripts/studies/lines128_paper_benchmark.py`, `scripts/studies/grid_lines_compare_wrapper.py`, `scripts/studies/grid_lines_torch_runner.py`, `scripts/studies/paper_results_refresh.py`, pytest, `compileall`, Markdown/JSON/CSV/TeX artifacts, repo-local `.artifacts/` verification logs.

---

## Selected Backlog Objective

- Implement backlog item `2026-05-04-cdi-lines128-multiseed-headline-robustness`.
- Reuse accepted `seed=3` rows by lineage after contract and provenance audit.
- Produce a fixed three-seed robustness result for exactly these headline rows:
  - `baseline` (`cnn`, supervised)
  - `pinn` (`cnn`, PINN)
  - `pinn_hybrid_resnet`
  - `pinn_fno_vanilla`
  - `pinn_ffno`
  - `pinn_neuralop_uno`
- Report this as training-seed robustness on the existing `lines128` split only.
- Publish an append-only robustness authority with per-seed metrics plus mean/std, without rewriting any existing single-seed claim boundary.

## Scope And Explicit Non-Goals

In scope:

- pin the multiseed set to `{3, 11, 17}` so the roadmap’s “exact seed set” requirement is satisfied while preserving the accepted `seed=3` lineage
- audit the accepted `seed=3` roots before launching anything new
- audit for already completed `seed=11` or `seed=17` rows on the same locked contract and reuse them if they pass, instead of rerunning automatically
- launch only the missing seed-row pairs needed to complete the six-row headline roster under the unchanged `lines128` CDI contract
- keep the fresh-launch budget inside the explicit runtime authority in this plan: at most the six-row by two-missing-seed primary matrix, plus bounded contingency under the hard cap below
- keep fixed across seeds:
  - `N=128`
  - `gridsize=1`
  - synthetic grid-lines data
  - `set_phi=True`
  - custom Run1084 probe
  - `probe_scale_mode=pad_extrapolate`
  - `probe_smoothing_sigma=0.5`
  - `nimgs_train=2`
  - `nimgs_test=2`
  - `nphotons=1e9`
  - `torch_epochs=40`
  - `torch_learning_rate=2e-4`
  - `ReduceLROnPlateau(factor=0.5, patience=2, min_lr=1e-4, threshold=0.0)`
  - `torch_loss_mode=mae`
  - `torch_mae_pred_l2_match_target=off`
  - `torch_output_mode=real_imag`
  - `probe_mask=off`
  - fixed sample ids `0` and `1`
  - shared visual-scale policy
- produce one additive robustness bundle under:
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-04-cdi-lines128-multiseed-headline-robustness/`
- write the durable summary:
  `docs/plans/NEURIPS-HYBRID-RESNET-2026/lines128_multiseed_headline_robustness_summary.md`
- update the required discoverability surfaces:
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/evidence_matrix.md`
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/model_variant_index.json`
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/paper_evidence_index.md`
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/ablation_index.json`
  - `docs/studies/index.md`

Explicit non-goals:

- do not broaden this into expanded-object robustness, object-family robustness, noise sweeps, probe variations, architecture ablations, new baselines, or `256x256` CDI work
- do not rerun accepted `seed=3` rows before an audit shows an unrecoverable contract or provenance gap
- do not add `supervised_ffno`, `supervised_neuralop_uno`, `pinn_spectral_resnet_bottleneck_net`, classical CDI rows, or any PDEBench work to this item
- do not overwrite:
  - `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-paper-benchmark-execution/runs/complete_table_20260430T150757Z_repair_tmux`
  - `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-30-cdi-lines128-uno-table-extension/runs/complete_table_plus_uno_20260504T100347Z`
  - existing single-seed paper-refresh payloads such as `docs/plans/NEURIPS-HYBRID-RESNET-2026/tables/cdi_lines128_metrics_extended.*`
- do not modify `ptycho/model.py`, `ptycho/diffsim.py`, or `ptycho/tf_helper.py`
- do not create worktrees

## Binding Constraints And Prerequisite Status

Strategic and roadmap constraints:

- `docs/steering.md` requires apples-to-apples comparisons and forbids silently relaxing fairness constraints. If a row cannot keep the fixed contract, record a row-level incompatibility or blocker instead of drifting the protocol.
- `docs/plans/2026-04-20-neurips-hybrid-resnet-submission-roadmap.md` Phase `3.3g` authorizes this exact lane only after the complete `lines128` table and the U-NO extension exist. This is an append-only robustness pass, not a new headline-contract rewrite.
- `docs/plans/2026-04-20-neurips-hybrid-resnet-submission-design.md` keeps CDI `128x128` as the headline and requires provenance-heavy handling for paper-facing evidence.
- `docs/plans/NEURIPS-HYBRID-RESNET-2026/lines128_paper_benchmark_design.md` says any multi-seed paper claim must pin the seed list, aggregation rules, and extra runtime budget before execution. This plan is that authority.
- `docs/plans/2026-04-20-neurips-hybrid-resnet-submission-roadmap.md` requires multi-seed results used for headline claims to report the exact seed set, per-seed values, mean, and standard deviation while keeping dataset, split, probe, metric schema, and visual policy fixed.

Prerequisite status that matters here:

- `state/NEURIPS-HYBRID-RESNET-2026/progress_ledger.json` shows the initiative is not globally blocked and that Phase 0 and early Phase 2 planning work is complete; it does not supersede the later checked-in CDI authorities below.
- Treat these checked-in outputs as the binding phase-local prerequisites:
  - `docs/backlog/done/2026-04-29-cdi-lines128-paper-benchmark-execution.md`
  - `docs/backlog/done/2026-04-30-cdi-lines128-uno-table-extension.md`
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/lines128_paper_benchmark_summary.md`
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/lines128_uno_table_extension_summary.md`
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/tables/cdi_lines128_metrics_extended.json`
- Authoritative seed=3 lineage to preserve:
  - `baseline`, `pinn`, `pinn_hybrid_resnet`, `pinn_fno_vanilla` inherit their primary seed=3 paper lineage from the complete-table authority
  - historical `pinn_ffno` inherits seed=3 lineage from the complete-table authority only as `FFNO-local proxy`; canonical no-refiner FFNO robustness must wait for corrected `fno_cnn_blocks=0` lineage
  - `pinn_neuralop_uno` inherits its seed=3 paper lineage from the U-NO extension authority
- Existing single-seed paper-refresh payloads remain valid historical lineage and must not be silently reinterpreted as multiseed aggregates.

Aggregation and exclusion rules:

- Aggregate only rows with a complete audited seed set `{3, 11, 17}` under the locked contract.
- If a row cannot produce a complete seed set after one documented narrow fix attempt, record a row-level blocker and exclude that row from the aggregate mean/std table.
- Keep incomplete rows visible in the summary and manifests, but never mix them into the completed aggregate table by prose alone.
- Use an explicit new claim boundary such as `cdi_lines128_headline_training_seed_robustness`; do not reuse `complete_lines128_cdi_benchmark` or `complete_lines128_cdi_benchmark_plus_uno_extension`.

Authorized extra runtime budget:

- Worst-case fresh matrix if only the accepted `seed=3` lineage is reusable: `12` launches total = `6` headline rows x missing seeds `{11, 17}`.
- Use these pre-authorized planning caps when converting the audited missing-row matrix into launch budget:
  - `baseline`: `615` seconds from `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-minimum-paper-table/runs/minimum_subset_20260430T035104Z/model_manifest.json` (`train_wall_time_sec=611.9325758698396`, rounded up)
  - `pinn`: `535` seconds from `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-minimum-paper-table/runs/minimum_subset_20260430T035104Z/model_manifest.json` (`train_wall_time_sec=530.9505021998193`, rounded up)
  - `pinn_hybrid_resnet`: `1085` seconds from `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-paper-benchmark-execution/runs/complete_table_20260430T150757Z_repair_tmux/model_manifest.json` (`command_wall_time_sec=1081.543966`, rounded up)
  - `pinn_fno_vanilla`: `515` seconds from `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-paper-benchmark-execution/runs/complete_table_20260430T150757Z_repair_tmux/model_manifest.json` (`command_wall_time_sec=511.236727`, rounded up)
  - corrected no-refiner `pinn_ffno`: `1400` seconds as the conservative cap for the only remaining headline row whose preserved prerequisite lineage does not retain standalone runtime; use the slowest prior fresh Torch headline-row runtime as the ceiling instead of inventing a lower number
  - `pinn_neuralop_uno`: `1095` seconds from `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-30-cdi-lines128-uno-table-extension/runs/complete_table_plus_uno_20260504T100347Z/model_manifest.json` (`command_wall_time_sec=1094.941273`, rounded up)
- Authorized primary fresh-launch budget for the full `12`-launch matrix: `10490` seconds (`2.92` GPU-hours) = `2 * (615 + 535 + 1085 + 515 + 1400 + 1095)`.
- Authorized contingency budget: `7510` additional seconds (`2.09` GPU-hours), usable only for tmux start/stop overhead and one documented narrow-fix relaunch per blocked row.
- Hard cap for this backlog item: `18000` seconds (`5.0` GPU-hours) of fresh launch time. Reused audited roots consume zero fresh-runtime budget.
- If the audited missing-seed matrix plus the documented contingency need would exceed the `18000` second cap, or if satisfying the cap would require changing the six-row roster or seed set, stop before launch and record the item as blocked on runtime authority instead of widening scope silently.

Failure-handling policy:

- Do not mark the item `BLOCKED` for ordinary import, path, environment, or test-harness failures. Diagnose, patch narrowly, and rerun first.
- Reserve item-level `BLOCKED` for:
  - missing or corrupted prerequisite roots that cannot be repaired from current authority documents
  - unavailable hardware or `ptycho311` runtime needed for the missing-seed launches
  - an unrecoverable contract mismatch that would require a roadmap or design change outside this item’s authority
  - a duplicate active writer on the intended multiseed output root that cannot be safely resolved

Long-run execution rules:

- No fresh seed-row launch may start until the blocking pre-launch checks in Task 3 are green.
- Launch long-running commands in tmux, activate `ptycho311`, and keep ownership until terminal success or recoverable failure handling is complete.
- Track the exact launched PID with `cmd ... & pid=$!; wait "$pid"`.
- Do not launch a duplicate writer against the same multiseed output root.
- Consider a fresh seed-row launch complete only when:
  - the tracked PID exits `0`
  - the row-local invocation, config, history, metrics, reconstruction, visuals, randomness contract, and launcher-completion proof are freshly written

## Implementation Architecture

- **Multiseed orchestration and collation**
  - `scripts/studies/lines128_paper_benchmark.py` should own the multiseed mode, the pinned seed set, seed-audit manifest generation, row inclusion/exclusion rules, and final aggregate bundle collation.
- **Seed-local row execution and reuse**
  - `scripts/studies/grid_lines_compare_wrapper.py` should own any additive routing or reuse logic needed to launch missing seed-row pairs cleanly and to preserve row-local provenance in a multiseed bundle.
  - `scripts/studies/grid_lines_torch_runner.py` should change only if current seed/provenance artifacts are insufficient for row-level multiseed audit or for U-NO deterministic-carve-out continuity.
- **Additive paper-refresh packaging**
  - `scripts/studies/paper_results_refresh.py` should emit new multiseed payloads only if manuscript-facing table refresh is updated. Existing single-seed tables remain untouched and explicitly sourced from their historical roots.
- **Evidence and discoverability**
  - the summary plus evidence indexes must distinguish:
    - preserved single-seed lineage
    - completed multiseed headline rows
    - row-level blockers
    - the narrow training-seed-only robustness claim boundary

## Concrete File And Artifact Targets

Core implementation targets:

- Modify: `scripts/studies/lines128_paper_benchmark.py`
- Modify: `scripts/studies/grid_lines_compare_wrapper.py`
- Modify only if seed-local provenance or randomness-contract capture needs a narrow fix:
  `scripts/studies/grid_lines_torch_runner.py`
- Modify: `scripts/studies/paper_results_refresh.py`
- Modify only if generic table rendering becomes reusable enough to justify it:
  `scripts/studies/metrics_tables.py`

Likely test targets:

- Modify: `tests/studies/test_lines128_paper_benchmark.py`
- Modify: `tests/studies/test_paper_results_refresh.py`
- Modify: `tests/test_grid_lines_compare_wrapper.py`
- Modify only if runner seed/provenance output changes:
  `tests/torch/test_grid_lines_torch_runner.py`

Mandatory contract outputs:

- item artifact root:
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-04-cdi-lines128-multiseed-headline-robustness/`
- one multiseed bundle root under that item root, for example:
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-04-cdi-lines128-multiseed-headline-robustness/runs/headline_multiseed_<timestamp>/`
- within the multiseed bundle root:
  - an audit manifest for seed reuse and launch decisions
  - a launch-budget manifest that records the missing-row matrix, per-row planning caps, planned fresh-runtime total, contingency reserved, and actual post-run runtime consumption versus the `18000` second hard cap
  - a row-status manifest that names completed rows and blocked rows
  - per-seed metrics plus aggregate mean/std in JSON, CSV, and TeX
  - explicit seed set `{3, 11, 17}`
  - explicit claim boundary for training-seed robustness only
  - explicit lineage back to the accepted seed=3 authorities and any reused seed=11/17 roots
  - seed-local row artifact directories or lineage pointers for every included row
- item-local verification logs under:
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-04-cdi-lines128-multiseed-headline-robustness/verification/`
- durable summary:
  `docs/plans/NEURIPS-HYBRID-RESNET-2026/lines128_multiseed_headline_robustness_summary.md`

Preferred packaging, not a substitute for the mandatory outputs:

- additive paper-local table payloads under `docs/plans/NEURIPS-HYBRID-RESNET-2026/tables/`, for example:
  - `cdi_lines128_headline_multiseed_metrics.json`
  - `cdi_lines128_headline_multiseed_metrics.csv`
  - `cdi_lines128_headline_multiseed_metrics.tex`
- a convenience seed-lineage payload that lists the exact source root for every row and seed
- a seed-local visual audit or compact stability table if it can be emitted cleanly without widening scope

Durable docs and index targets:

- Create: `docs/plans/NEURIPS-HYBRID-RESNET-2026/lines128_multiseed_headline_robustness_summary.md`
- Modify: `docs/plans/NEURIPS-HYBRID-RESNET-2026/evidence_matrix.md`
- Modify: `docs/plans/NEURIPS-HYBRID-RESNET-2026/model_variant_index.json`
- Modify: `docs/plans/NEURIPS-HYBRID-RESNET-2026/paper_evidence_index.md`
- Modify: `docs/plans/NEURIPS-HYBRID-RESNET-2026/ablation_index.json`
- Modify: `docs/studies/index.md`
- Update `docs/index.md` only if the new multiseed summary or helper becomes a first-stop discoverability surface instead of remaining adequately covered by the study index and evidence indexes
- Update `docs/findings.md` only if implementation uncovers a durable repo-level trap not already covered by the existing CDI or U-NO authorities

## Execution Checklist

### Task 1: Audit Existing Seed Lineage And Add Red Tests

**Files:**

- Modify: `tests/studies/test_lines128_paper_benchmark.py`
- Modify: `tests/test_grid_lines_compare_wrapper.py`
- Modify: `tests/studies/test_paper_results_refresh.py`
- Optional modify: `tests/torch/test_grid_lines_torch_runner.py`

- [ ] Add tests for a new multiseed mode in `lines128_paper_benchmark.py` that consumes the complete-table and U-NO extension authorities without mutating them.
- [ ] Add tests that the fixed seed set is exactly `{3, 11, 17}` and that `seed=3` is reused by lineage instead of relaunched when its audit passes.
- [ ] Add tests that any discovered `seed=11` or `seed=17` exact-contract root can also be reused by lineage, and that only genuinely missing seed-row pairs remain in the launch matrix.
- [ ] Add tests that the included aggregate roster is exactly:
  `baseline`, `pinn`, `pinn_hybrid_resnet`, `pinn_fno_vanilla`, `pinn_ffno`, `pinn_neuralop_uno`.
- [ ] Add tests that incomplete rows are kept out of the mean/std aggregate table and surfaced instead through explicit row-level blocker metadata.
- [ ] Add tests that any new paper-refresh output is additive and leaves current single-seed payloads and their source lineage unchanged.
- [ ] Add tests that the summary payload and machine-readable aggregate bundle report:
  - exact seed set
  - per-seed values
  - mean
  - standard deviation
  - claim boundary
  - source-lineage pointers

**Verification for this tranche:**

- [ ] Supporting: run the new focused selectors once to confirm they fail for the intended multiseed gaps rather than unrelated infrastructure noise.
- [ ] Supporting: archive the red-first log under the item-local `verification/` directory.

### Task 2: Implement Multiseed Audit, Launch-Matrix, And Aggregate Packaging

**Files:**

- Modify: `scripts/studies/lines128_paper_benchmark.py`
- Modify: `scripts/studies/grid_lines_compare_wrapper.py`
- Modify: `scripts/studies/paper_results_refresh.py`
- Conditional modify: `scripts/studies/grid_lines_torch_runner.py`
- Conditional modify: `scripts/studies/metrics_tables.py`

- [ ] Add a new multiseed execution path in `lines128_paper_benchmark.py` that:
  - reads the accepted seed=3 authorities
  - pins seeds `{3, 11, 17}`
  - audits reuse candidates before launch
  - derives the missing seed-row launch matrix
  - collates one additive robustness bundle
- [ ] Encode the aggregate inclusion rule directly in the implementation: a row reaches the mean/std table only when all three seeds are present under the locked contract.
- [ ] Preserve row-local audit tolerance where it is already part of accepted single-seed lineage, including the U-NO deterministic `warn` carve-out for `pinn_neuralop_uno`, but do not broaden that carve-out to other rows.
- [ ] Keep existing single-seed claim boundaries intact:
  - `complete_lines128_cdi_benchmark`
  - `complete_lines128_cdi_benchmark_plus_uno_extension`
- [ ] Emit a new additive claim boundary for the multiseed bundle only.
- [ ] If manuscript-facing tables are refreshed, emit new multiseed payload filenames rather than overwriting `cdi_lines128_metrics_extended.*`.
- [ ] Preserve the existing single-seed source-lineage note in any refreshed tables or summaries.

**Verification for this tranche:**

- [ ] Blocking before launch: `pytest -q tests/studies/test_lines128_paper_benchmark.py`
- [ ] Blocking before launch: `pytest -q tests/studies/test_paper_results_refresh.py tests/test_grid_lines_compare_wrapper.py`
- [ ] Blocking before launch only if runner code changed: `pytest -q tests/torch/test_grid_lines_torch_runner.py`

### Task 3: Clear The Pre-Launch Deterministic Gates

**Files:**

- No new durable code targets. This tranche archives verification evidence and refuses long runs until the gates are green.

- [ ] Run the selected backlog item’s required input check unchanged:

```bash
python - <<'PY'
from pathlib import Path
required = [
    Path("docs/backlog/done/2026-04-29-cdi-lines128-paper-benchmark-execution.md"),
    Path("docs/backlog/done/2026-04-30-cdi-lines128-uno-table-extension.md"),
    Path("docs/plans/NEURIPS-HYBRID-RESNET-2026/lines128_paper_benchmark_summary.md"),
    Path("docs/plans/NEURIPS-HYBRID-RESNET-2026/tables/cdi_lines128_metrics_extended.json"),
    Path("scripts/studies/grid_lines_compare_wrapper.py"),
    Path("scripts/studies/grid_lines_torch_runner.py"),
]
missing = [str(path) for path in required if not path.exists()]
if missing:
    raise SystemExit(f"missing CDI multiseed robustness inputs: {missing}")
print("CDI multiseed robustness inputs present")
PY
```

- [ ] Run the backlog item’s required deterministic pytest gate unchanged:

```bash
pytest -q tests/studies/test_paper_results_refresh.py tests/test_grid_lines_compare_wrapper.py tests/torch/test_grid_lines_torch_runner.py
```

- [ ] Run the backlog item’s required compile gate unchanged:

```bash
python -m compileall -q scripts/studies ptycho_torch
```

- [ ] Run the stronger focused benchmark-surface gate because this item is expected to modify `lines128_paper_benchmark.py` and the backlog check list does not cover that surface directly:

```bash
pytest -q tests/studies/test_lines128_paper_benchmark.py
```

- [ ] Archive every pre-launch log under the item-local `verification/` directory before any tmux launch.

**Verification for this tranche:**

- [ ] Blocking: all four commands above must pass before any fresh seed-row launch begins.

### Task 4: Launch Only Missing Seed-Row Pairs And Collate The Multiseed Bundle

**Files:**

- Primary implementation paths from Task 2
- No new durable code should be introduced here unless a narrow recoverable launch bug is discovered

- [ ] Run the audit first and record, per row and per seed, whether the source is:
  - reused by lineage
  - freshly launched
  - blocked with an explicit reason
- [ ] Convert the audited missing seed-row matrix into a launch-budget manifest before any tmux launch. Use only the row caps in the “Authorized extra runtime budget” section, reserve contingency explicitly, and refuse launch if the planned total would exceed the `18000` second hard cap.
- [ ] Launch only the missing seed-row pairs under tmux in `ptycho311`, using a unique multiseed output root and exact PID tracking.
- [ ] Do not launch a second run for a row and seed that already has a passing audited artifact on the locked contract.
- [ ] Require every fresh seed-row artifact to include the same core evidence surfaces expected of the accepted seed=3 rows:
  - invocation
  - config
  - history
  - metrics
  - reconstruction
  - visuals
  - randomness contract
  - launcher completion proof
- [ ] After the fresh launches finish, collate the aggregate bundle with:
  - per-row per-seed metrics
  - row mean/std
  - explicit exact seed set
  - row-level completed or blocked status
  - claim-boundary metadata
  - lineage back to the single-seed authorities and any reused missing-seed roots
- [ ] Exclude incomplete rows from the aggregate mean/std table, but keep them visible in the manifests and summary.

**Verification for this tranche:**

- [ ] Blocking for each fresh launch: tracked PID exits `0` and all required row-local artifacts are freshly written.
- [ ] Blocking before the first fresh launch: the launch-budget manifest shows planned fresh runtime plus reserved contingency `<= 18000` seconds.
- [ ] Blocking for bundle acceptance: the collated multiseed root contains the aggregate JSON, CSV, and TeX payloads plus the audit and row-status manifests.
- [ ] Supporting: if a row fails after one documented narrow fix attempt, record the blocker and continue collating the remaining rows instead of relaunching blindly.

### Task 5: Publish The Durable Summary And Refresh Discoverability

**Files:**

- Create: `docs/plans/NEURIPS-HYBRID-RESNET-2026/lines128_multiseed_headline_robustness_summary.md`
- Modify: `docs/plans/NEURIPS-HYBRID-RESNET-2026/evidence_matrix.md`
- Modify: `docs/plans/NEURIPS-HYBRID-RESNET-2026/model_variant_index.json`
- Modify: `docs/plans/NEURIPS-HYBRID-RESNET-2026/paper_evidence_index.md`
- Modify: `docs/plans/NEURIPS-HYBRID-RESNET-2026/ablation_index.json`
- Modify: `docs/studies/index.md`
- Conditional modify: `docs/index.md`

- [ ] Write the durable summary with:
  - exact seed set `{3, 11, 17}`
  - preserved single-seed authorities
  - which rows were fully aggregated
  - which rows were blocked and why
  - planned versus actual fresh-runtime consumption, including the `18000` second hard cap and any used contingency
  - per-row per-seed metrics plus mean/std
  - the explicit statement that this is training-seed robustness on the existing `lines128` split, not broader object-distribution robustness
- [ ] Update `evidence_matrix.md` with a multiseed headline-robustness section that references the new bundle without erasing the accepted single-seed tables.
- [ ] Add a new multiseed dataset-contract or aggregate-contract entry in `model_variant_index.json`, keeping the original `cdi_lines128_seed3` entries intact.
- [ ] Add the new backlog outcome to `paper_evidence_index.md` as append-only paper-supporting robustness evidence with its own narrow claim boundary.
- [ ] Update `ablation_index.json` because this is a fixed-contract robustness study rather than a new baseline or architecture family.
- [ ] Update `docs/studies/index.md` so future planning can discover the multiseed authority directly.
- [ ] If new paper-local multiseed tables were emitted, note clearly that they are additive multiseed payloads and not replacements for the existing single-seed table files.

**Verification for this tranche:**

- [ ] Blocking final deterministic gates:

```bash
pytest -q tests/studies/test_lines128_paper_benchmark.py tests/studies/test_paper_results_refresh.py tests/test_grid_lines_compare_wrapper.py tests/torch/test_grid_lines_torch_runner.py
```

- [ ] Blocking final required backlog gates:

```bash
pytest -q tests/studies/test_paper_results_refresh.py tests/test_grid_lines_compare_wrapper.py tests/torch/test_grid_lines_torch_runner.py
python -m compileall -q scripts/studies ptycho_torch
```

- [ ] Supporting final workflow evidence: run `pytest -q -m integration` only if implementation touched generic train/infer/persistence plumbing beyond the study-local surfaces. If it is run, archive the log; if it is not run, the final summary must state why the narrower evidence was sufficient.
