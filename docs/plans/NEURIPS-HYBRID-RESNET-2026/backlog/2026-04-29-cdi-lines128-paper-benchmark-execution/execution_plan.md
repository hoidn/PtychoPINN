# Lines128 Paper-Quality CDI Benchmark Execution Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:executing-plans` or `superpowers:subagent-driven-development` to implement this plan task-by-task. Keep this file as the execution authority for the selected backlog item.

**Goal:** Extend the paper-complete `lines128` minimum CDI subset into the complete fixed-contract six-row benchmark by adding `pinn_spectral_resnet_bottleneck_net` and `pinn_ffno`, then publish validated merged tables, visuals, manifests, and a durable summary.

**Architecture:** Reuse the frozen `N=128`, `seed=3`, `fno_vanilla` contract from the checked-in preflight/harness surfaces, keep the current minimum-subset root immutable as prerequisite evidence, and launch or promote only the missing complete-table rows into a new unique complete-benchmark root. Preserve `grid_lines_compare_wrapper.py` as the shared dataset/collation authority, `grid_lines_torch_runner.py` as the per-row authority, and extend `lines128_paper_benchmark.py` only enough to own complete-table orchestration, row-audit acceptance, and final paper-status validation.

**Tech Stack:** Python 3.11, `ptycho311`, tmux-managed long runs, PyTorch/Lightning, `scripts/studies/grid_lines_compare_wrapper.py`, `scripts/studies/grid_lines_torch_runner.py`, `scripts/studies/lines128_paper_benchmark.py`, Markdown/JSON/CSV/TeX evidence artifacts.

---

## Selected Backlog Objective

- Complete the `lines128` paper-quality CDI package by taking the already accepted four-row minimum subset and adding the required paper rows:
  - `pinn_spectral_resnet_bottleneck_net`
  - `pinn_ffno`
- Publish one authoritative complete-benchmark root with:
  - merged machine-readable metrics
  - paper-facing CSV/TeX tables
  - fixed-sample reconstruction visuals
  - row/root provenance manifests
  - a durable NeurIPS summary

## Scope

- Consume the fixed contract from:
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/lines128_paper_benchmark_preflight.md`
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/lines128_paper_benchmark_harness_preflight.md`
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/lines128_minimum_paper_table_execution_authority.md`
- Keep the selected FNO comparator frozen at `fno_vanilla` and the seed policy frozen at `seed=3`.
- Audit existing candidate roots first, promote them only when the paper contract is satisfied directly or via deterministic provenance recovery, and rerun only the rows that still fail the contract after that audit.
- Launch any required long-running execution in tmux from the repo root with `ptycho311`, a unique output root, exact PID tracking, and no duplicate writer to the same output root.
- Accept the final bundle as `paper_complete` only if every required row and required metric is present, or explicitly `not_applicable` under the metric schema; otherwise emit `benchmark_incomplete` with missing-field reasons.

## Explicit Non-Goals

- Do not change the recovered `grid-lines-n128-hybrid-resnet-legacy-best-e40-seed3` contract to fit a row family.
- Do not change the selected FNO comparator, seed policy, or visual/sample policy after seeing metrics.
- Do not reopen PDEBench, `256x256` CDI scaling, manuscript prose, or `/home/ollie/Documents/neurips/` evidence-bundle work.
- Do not broaden into multi-seed claims, broad FFNO/spectral sweeps, or classical CDI comparator expansion.
- Do not rewrite prerequisite roots in place. The minimum-subset authoritative root and the FFNO-vs-Hybrid prerequisite root remain preserved inputs.

## Binding Constraints And Prerequisite Status

- Steering/roadmap boundary:
  - This is Roadmap Phase 3 CDI packaging work only.
  - Preserve apples-to-apples comparison standards and the fixed paper claim boundary.
- Long-run guardrails:
  - Use tmux for long-running commands.
  - Activate `ptycho311`.
  - Track the launched PID exactly; do not rely on broad `pgrep -f` polling.
  - A launched run completes only when the tracked PID exits `0` and required artifacts are freshly written.
- Failure handling:
  - If a normal import/test/path/environment/harness failure occurs, diagnose, fix narrowly, and rerun.
  - Reserve `BLOCKED` only for missing resources, unavailable hardware, roadmap conflict, external dependency outside current authority, required user decision, or an unrecoverable failure after a documented narrow fix attempt.
- Prerequisites already satisfied:
  - `2026-04-27-cdi-ffno-generator-lines-best-config`: completed.
    - Stable prerequisite root:
      `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-27-cdi-ffno-generator-lines-best-config/lines128_ffno_vs_hybrid_resnet`
    - Important caveat: row-level `invocation.*` files were backfilled during repair, so any reuse must record that recovered-provenance fact explicitly.
  - `2026-04-29-cdi-lines128-paper-benchmark-harness`: completed as readiness-only.
    - Decision artifact:
      `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-paper-benchmark-harness/preflight/benchmark_decisions.json`
    - Current gap: `scripts/studies/lines128_paper_benchmark.py` currently exposes only `preflight` and `minimum_subset` modes, so this item must add a complete-table execution surface without regressing those existing modes.
  - `2026-04-29-cdi-lines128-minimum-paper-table`: completed as `paper_complete`.
    - Authoritative root:
      `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-minimum-paper-table/runs/minimum_subset_20260430T084339Z`
    - This root already satisfies the four-row minimum subset and should be treated as preserved paper-grade input evidence.
- Candidate-row status entering this item:
  - `pinn_ffno`: likely promotable from the fixed-contract prerequisite pair after audit.
  - `pinn_spectral_resnet_bottleneck_net`: current checked-in summary is only a `1`-epoch `N=128` plumbing smoke on `nimgs_test=1`, so assume rerun is required unless an audit finds a separate contract-complete spectral root.

## Implementation Architecture

- **Authority and row-audit surface**
  - Freeze the six-row complete-table roster, selected comparator, fixed seed, fixed sample IDs, shared visual-scale policy, and reuse-versus-rerun decision in one machine-readable execution artifact before launch.
  - That artifact must distinguish promoted existing roots from freshly launched rows and record any recovered provenance fields.
- **Benchmark execution surface**
  - Extend `scripts/studies/lines128_paper_benchmark.py` with a complete-table execution mode, or add an equally thin adjacent launcher if keeping the existing script smaller is materially cleaner.
  - The execution surface must continue delegating shared dataset generation/collation to `grid_lines_compare_wrapper.py` and per-row work to `grid_lines_torch_runner.py`.
- **Acceptance and publication surface**
  - Finalization must compute `paper_complete` versus `benchmark_incomplete`, populate merged manifests consistently, and publish the durable summary plus `docs/studies/index.md` entry from the same accepted root.

## Concrete File And Artifact Targets

- Likely code targets:
  - `scripts/studies/lines128_paper_benchmark.py`
  - `scripts/studies/grid_lines_compare_wrapper.py`
  - `scripts/studies/grid_lines_torch_runner.py`
  - `scripts/studies/paper_provenance.py`
- Likely test targets:
  - `tests/studies/test_lines128_paper_benchmark.py`
  - `tests/studies/test_paper_provenance.py`
  - `tests/test_grid_lines_compare_wrapper.py`
  - `tests/torch/test_grid_lines_torch_runner.py`
- Durable doc targets:
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/lines128_paper_benchmark_summary.md` (new authoritative summary for this item)
  - `docs/studies/index.md`
  - `docs/index.md` only if the new summary becomes a primary discoverable NeurIPS reference and is not already surfaced adequately
- Execution artifacts:
  - `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-paper-benchmark-execution/`
  - new unique benchmark root under `.../runs/`
  - row-audit / execution-authority / verification logs under the same backlog artifact root

## Execution Checklist

### Tranche 1: Freeze Complete-Table Authority And Audit Reusable Rows

- [ ] Create a machine-readable complete-table execution artifact under the selected item artifact root that records:
  - six required rows:
    - `baseline`
    - `pinn`
    - `pinn_hybrid_resnet`
    - `pinn_fno_vanilla`
    - `pinn_spectral_resnet_bottleneck_net`
    - `pinn_ffno`
  - selected comparator `fno_vanilla`
  - fixed seed `3`
  - fixed sample IDs `0,1`
  - shared visual-scale policy inherited from the minimum-subset authority
  - row-by-row decision: `promote_existing`, `rerun_required`, or `blocked`
- [ ] Audit candidate existing roots for `pinn_ffno` and `pinn_spectral_resnet_bottleneck_net` against the frozen contract:
  - dataset/split identity
  - probe preprocessing
  - epoch/loss/scheduler/output-mode contract
  - metrics payload completeness
  - reconstruction arrays
  - invocation/config/randomness provenance
  - launcher completion / exit evidence
- [ ] Record every recovered field, caveat, and rejection reason in a deterministic audit manifest before launch.
- [ ] If a row has only a narrow provenance gap that can be repaired without relaunching the underlying finished work, repair it deterministically and rerun the audit.
- [ ] If a row still fails the contract after the repair attempt, mark it `rerun_required`; do not quietly weaken the contract.

Verification before moving on:

- [ ] Audit manifest exists and names all six rows with final pre-launch status.
- [ ] The plan-authorized comparator/seed/sample policy in the execution artifact matches:
  - `benchmark_decisions.json`
  - `lines128_minimum_paper_table_execution_authority.md`
  - the fixed-contract preflight note

### Tranche 2: Close Harness/Wrapper Gaps For Full Six-Row Execution

- [ ] Extend the execution surface so a complete-table run can be launched from one authoritative command while preserving existing `preflight` and `minimum_subset` behavior.
- [ ] Keep minimum-subset behavior stable; do not regress the already accepted `paper_complete` minimum bundle logic.
- [ ] Add only the missing complete-table functionality needed for:
  - six-row roster handling
  - promoted existing-row ingestion
  - fresh rerun support for missing rows
  - merged manifest/model-manifest consistency
  - fixed-sample visual collation across all accepted rows
  - complete-table `paper_complete` versus `benchmark_incomplete` status computation
- [ ] If promoted rows rely on recovered Torch completion evidence, ensure finalization reattaches `launcher_completion.json` consistently inside row outputs and top-level manifests.

Verification for this tranche:

- [ ] Run focused regression coverage for the full-table execution path and provenance handling, for example:
  - `pytest -q tests/studies/test_lines128_paper_benchmark.py tests/studies/test_paper_provenance.py tests/test_grid_lines_compare_wrapper.py`
- [ ] Run the backlog item’s required deterministic checks before any expensive launch. The full benchmark must wait for green results on:
  - `pytest -q tests/torch/test_grid_lines_hybrid_resnet_integration.py tests/torch/test_grid_lines_torch_runner.py tests/test_grid_lines_compare_wrapper.py`
  - `python -m compileall -q ptycho_torch scripts/studies`

### Tranche 3: Launch Or Reconstruct The Complete Benchmark Root

- [ ] Create a new unique complete-table root under:
  - `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-paper-benchmark-execution/runs/`
- [ ] Consume the frozen execution artifact from Tranche 1 plus the checked-in prerequisite decision surfaces.
- [ ] Reuse the accepted minimum-subset evidence without rewriting its authoritative root in place.
- [ ] Promote the audited FFNO row if and only if the audit marked it contract-complete.
- [ ] Rerun the spectral row unless the audit found a separate contract-complete spectral root; do not treat the current one-epoch smoke as paper evidence.
- [ ] If any rerun is required:
  - launch in tmux from repo root
  - activate `ptycho311`
  - write to the new unique output root only
  - track the exact launcher PID and wait on that PID
  - do not start a duplicate writer if another process is already writing the same root
- [ ] If the execution surface supports reuse of existing row artifacts, ensure the final complete-table root still contains a coherent, self-describing paper bundle and references promoted source roots explicitly in its audit/manifest surfaces.
- [ ] Treat a launch as accepted only when:
  - tracked PID exits `0`
  - all required row artifacts exist for the six-row roster
  - merged bundle artifacts are freshly written
  - manifest status and row-status surfaces agree

Verification for this tranche:

- [ ] Validate the final root contains, at minimum:
  - `metrics.json`
  - `metric_schema.json`
  - `model_manifest.json`
  - `metrics_table.csv`
  - `metrics_table.tex`
  - `metrics_table_best.tex`
  - `paper_benchmark_manifest.json`
  - `visuals/amp_phase_gt.png`
  - `visuals/compare_amp_phase.png`
  - `visuals/frc_curves.png`
  - `recons/gt/recon.npz`
  - per-row `runs/<row>/invocation.json`
  - per-row `runs/<row>/metrics.json`
  - per-row `recons/<row>/recon.npz`
- [ ] Verify the accepted root reports:
  - all six required rows
  - `selected_fno_comparator = fno_vanilla`
  - `seed = 3`
  - `fixed_sample_ids = [0, 1]`
  - row-level provenance for every promoted or rerun row
- [ ] If any required field is absent, keep the result `benchmark_incomplete` and record explicit missing-field reasons rather than overstating completeness.

### Tranche 4: Publish Durable Summary And Registry Updates

- [ ] Write `docs/plans/NEURIPS-HYBRID-RESNET-2026/lines128_paper_benchmark_summary.md` with:
  - authoritative complete-benchmark root
  - whether each late row was promoted or rerun
  - accepted six-row roster and paper-facing labels
  - benchmark status and claim boundary
  - table/visual artifact pointers
  - any remaining missing-field or provenance caveats
- [ ] Update `docs/studies/index.md` with the complete Lines128 paper benchmark entry and point it at the new summary and authoritative root.
- [ ] Update `docs/index.md` only if needed for discoverability of the new durable summary.
- [ ] Do not write manuscript prose or mirror artifacts into `/home/ollie/Documents/neurips/`; that remains Phase 5 work.

Verification for this tranche:

- [ ] Confirm the new study-index entry matches the same authoritative root and status named in the durable summary.
- [ ] Confirm doc text distinguishes:
  - prerequisite FFNO pair evidence
  - minimum subset evidence
  - the new complete-table evidence

### Tranche 5: Final Deterministic Closeout

- [ ] Rerun the backlog-required deterministic checks and archive their logs under this item’s verification directory:
  - `pytest -q tests/torch/test_grid_lines_hybrid_resnet_integration.py tests/torch/test_grid_lines_torch_runner.py tests/test_grid_lines_compare_wrapper.py`
  - `python -m compileall -q ptycho_torch scripts/studies`
- [ ] Archive any focused full-table regression logs used to validate the new execution surface.
- [ ] Record the final accepted root, final benchmark status, and verification-log paths in the durable summary.

## Completion Criteria

- The accepted complete-table root contains one coherent six-row Lines128 paper bundle.
- The row roster is exactly:
  - `baseline`
  - `pinn`
  - `pinn_hybrid_resnet`
  - `pinn_fno_vanilla`
  - `pinn_spectral_resnet_bottleneck_net`
  - `pinn_ffno`
- The fixed contract remains aligned with the recovered `N=128` legacy-best paper benchmark contract, selected comparator `fno_vanilla`, and fixed seed `3`.
- `metrics.json`, `model_manifest.json`, and `paper_benchmark_manifest.json` agree on benchmark status and row artifacts.
- The durable summary and `docs/studies/index.md` point to the same authoritative root and same claim boundary.
- Required deterministic checks pass and their logs are archived.

## Required Deterministic Checks

Implementation may use narrower focused tests during development, but these backlog-item checks remain mandatory for completion and must be green before any expensive full benchmark launch:

```bash
pytest -q tests/torch/test_grid_lines_hybrid_resnet_integration.py tests/torch/test_grid_lines_torch_runner.py tests/test_grid_lines_compare_wrapper.py
python -m compileall -q ptycho_torch scripts/studies
```
