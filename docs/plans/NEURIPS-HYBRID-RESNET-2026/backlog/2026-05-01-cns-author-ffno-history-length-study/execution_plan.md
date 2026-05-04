# CNS Authored FFNO History-Length Study Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Determine whether the authored PDEBench CNS FFNO row benefits from longer temporal context under the fixed local capped history-ablation contract, without reopening the locked CNS headline table or changing the current paper authority.

**Architecture:** Reuse the existing authored-FFNO CNS adapter and capped `2d_cfd_cns` runner path, first proving that `history_len` only changes the input/sample contract and not the model family, split family, normalization, target horizon, or metric surface. Freeze the accepted `history_len=2`, `40`-epoch authored FFNO row as the comparison anchor, run a fresh `history_len=3` matched-budget row, and open `history_len=4` then `history_len=5` only through a predeclared metric gate. Publish one durable summary plus evidence-index updates that classify every new row as adjacent capped decision-support evidence only.

**Tech Stack:** Python 3.11, `ptycho311`, PyTorch, `scripts/studies/pdebench_image128/*`, `scripts/studies/run_pdebench_image128_suite.py`, Markdown/JSON evidence indexes, `.artifacts/NEURIPS-HYBRID-RESNET-2026/`.

---

## Selected Objective

- Answer the narrow within-model question: does `author_ffno_cns_base` improve, saturate, or degrade when the local CNS adapter sees more than `history_len=2` input frames under the same capped study contract?

## Scope

- Freeze the accepted authored-FFNO anchor at:
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-author-ffno-equal-footing-cns/author-ffno-40ep-20260422T234340Z`
- Keep the local CNS history-ablation contract fixed to:
  - official file:
    `/home/ollie/Documents/pdebench-data/2d_cfd_cns/2D_CFD_Rand_M1.0_Eta0.01_Zeta0.01_periodic_128_Train.hdf5`
  - trajectories: `512 / 64 / 64`
  - `max_windows_per_trajectory=8`
  - train-only per-field normalization
  - training loss: `mse`
  - optimizer / scheduler family: `Adam`, `lr=2e-4`, `ReduceLROnPlateau(factor=0.5, patience=2, threshold=0.0, min_lr=1e-5)`
  - batch size: `4`
  - metric family:
    `err_RMSE`, `err_nRMSE`, `relative_l2`, `fRMSE_low`, `fRMSE_mid`, `fRMSE_high`
- Run `history_len=3` first at the matched `40`-epoch budget.
- Open `history_len=4` only if the `history_len=3` gate passes.
- Open `history_len=5` only if the `history_len=4` gate passes.
- Record absolute metrics and deltas, valid-window counts, split counts, parameter count, runtime, and peak memory for each fresh row that is actually evaluated.

## Explicit Non-Goals

- Do not rerun SRU-Net / spectral-bottleneck, Hybrid, FNO, U-Net, repo-local FFNO proxy rows, or GNOT.
- Do not silently migrate this ablation onto the newer `2048 / 256 / 256` capped evidence-strengthening follow-up lane. This item stays on the locked `512 / 64 / 64`, `history_len=2` history-ablation lane because the selected backlog item made that lane authoritative for this question.
- Do not mix any longer-history authored-FFNO row into the locked `history_len=2` CNS headline table.
- Do not convert this into an autoregressive rollout study.
- Do not claim full-training or paper-grade superiority from these capped runs.
- Do not create `/home/ollie/Documents/neurips/` outputs.

## Binding Constraints

- Steering authority:
  - preserve equal-footing comparisons and fairness constraints;
  - do not relax protocol boundaries to make the item easier;
  - authored FFNO is valid comparison work, but this item remains an adjacent ablation rather than headline-table authority.
- Roadmap / design authority:
  - Phase 2 PDEBench CNS work may produce capped decision-support evidence, but smoke or capped rows do not become benchmark-performance claims;
  - the locked CNS paper lane remains `history_len=2`;
  - longer-history results stay separate unless a later roadmap-level decision reopens the paper contract.
- Progress / prerequisite status:
  - `2026-04-21-pdebench-author-ffno-equal-footing-cns` is complete and provides the frozen authored-FFNO anchor plus the existing author adapter path.
  - `2026-04-29-cns-spectral-history-len4plus-compare` is complete and provides precedent for multi-history compare artifacts and gate recording.
- Failure-handling rule:
  - do not mark the item `BLOCKED` for normal test failures, import issues, path issues, or harness bugs; diagnose, narrow-fix, and rerun first.
  - reserve `BLOCKED` for missing dataset access, unavailable GPU/runtime resources, external authored dependency failure that remains unrecoverable after a narrow fix attempt, roadmap/user-scope conflict, or another unrecoverable external dependency.

## Implementation Architecture

### Unit A: Contract And Adapter Audit

- Purpose: prove that `history_len` changes only the input/sample contract for `author_ffno_cns_base`.
- Primary surfaces:
  - `scripts/studies/pdebench_image128/author_ffno_adapter.py`
  - `scripts/studies/pdebench_image128/models.py`
  - `scripts/studies/pdebench_image128/cfd_cns.py`
  - `tests/studies/test_pdebench_image128_models.py`
  - `tests/studies/test_pdebench_cfd_cns_data.py`
- Responsibility:
  - confirm `in_channels = history_len * state_channels`;
  - confirm the authored FFNO body, coordinate-feature behavior, output head width, target horizon, and field order stay unchanged except for the widened input tensor.

### Unit B: Run Execution And Compare Artifacts

- Purpose: own inspect/provenance artifacts, long-run launch discipline, gate decisions, and cross-history compare payloads.
- Primary surfaces:
  - `scripts/studies/pdebench_image128/cfd_cns.py`
  - `scripts/studies/pdebench_image128/reporting.py`
  - `tests/studies/test_pdebench_image128_runner.py`
- Responsibility:
  - emit history-specific inspect manifests and fresh run roots;
  - write reference manifests and compare sidecars through the existing reporting helpers;
  - record explicit gate decisions before each optional longer-history launch.

### Unit C: Durable Interpretation And Discoverability

- Purpose: turn any evaluated rows into durable project knowledge without changing the locked paper lane.
- Primary surfaces:
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_author_ffno_history_length_summary.md`
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/evidence_matrix.md`
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/model_variant_index.json`
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/ablation_index.json`
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/paper_evidence_index.md`
  - `docs/studies/index.md`
- Responsibility:
  - state the claim boundary explicitly;
  - add machine-readable row/index discoverability for every new evaluated history row;
  - keep the paper-authority row lock unchanged.

## File And Artifact Targets

### Mandatory Contract Outputs

- Durable summary:
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_author_ffno_history_length_summary.md`
- Artifact root for this backlog item:
  - `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-author-ffno-history-length-compare/`
- Frozen reference manifest for the authored-FFNO anchor:
  - `history2_reference_runs.json` under the artifact root
- Inspect roots:
  - `history3-inspect-<timestamp>`
  - `history4-inspect-<timestamp>` only if the `history_len=4` gate is being evaluated
  - `history5-inspect-<timestamp>` only if the `history_len=5` gate is being evaluated
- Fresh pilot roots:
  - `history3-pilot-40ep-<timestamp>`
  - `history4-pilot-40ep-<timestamp>` only if gated open
  - `history5-pilot-40ep-<timestamp>` only if gated open
- Launch-proof directories with tracked `exit_code` for each fresh pilot root.
- Compare payloads:
  - `compare_40ep_history3_against_history2.json` and `.csv`
  - `compare_40ep_history4_against_history2_history3.json` and `.csv` if `history_len=4` runs
  - `compare_40ep_history5_against_history2_history3_history4.json` and `.csv` if `history_len=5` runs
- Gate records:
  - `history4_gate_decision.json`
  - `history5_gate_decision.json` if `history_len=4` ran
- Evidence index updates for any evaluated row:
  - `evidence_matrix.md`
  - `model_variant_index.json`
  - `ablation_index.json`
  - `paper_evidence_index.md`
  - `docs/studies/index.md`

### Conditional Code Targets

- Only touch these if the audit proves existing behavior is insufficient:
  - `scripts/studies/pdebench_image128/author_ffno_adapter.py`
  - `scripts/studies/pdebench_image128/models.py`
  - `scripts/studies/pdebench_image128/cfd_cns.py`
  - `scripts/studies/pdebench_image128/reporting.py`
  - `tests/studies/test_pdebench_image128_models.py`
  - `tests/studies/test_pdebench_image128_runner.py`
  - `tests/studies/test_pdebench_cfd_cns_data.py`
  - `tests/studies/test_pdebench_cfd_cns_metrics.py`

### Preferred But Non-Blocking Packaging

- Cross-history gallery PNGs for prediction and error comparisons when the saved target arrays align exactly enough for reuse.
- If gallery rendering is impossible because targets differ, keep the compare JSON/CSV authoritative and record the gallery blocker in the compare payload and summary instead of failing the item.

## Execution Checklist

### Task 1: Freeze The Comparison Contract

- [ ] Re-read the frozen author-FFNO anchor summary and record the accepted reference numbers in the working notes for direct comparison:
  - `err_nRMSE = relative_l2 = 0.0281477310`
  - `err_RMSE = 0.6802443266`
  - `fRMSE_low = 1.6124732494`
  - `fRMSE_mid = 0.0759288296`
  - `fRMSE_high = 0.1210141182`
  - `parameter_count = 1,073,672`
  - `runtime_sec = 4725.5117`
- [ ] Create the new backlog-item artifact root:
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-author-ffno-history-length-compare/`
- [ ] Write `history2_reference_runs.json` with `build_reference_run_manifest(...)` / `write_reference_run_manifest(...)`, using the frozen authored-FFNO `history_len=2`, `40`-epoch run as the required reference row.
- [ ] Record in the summary draft that this backlog item is adjacent capped context only, does not modify the locked `512 / 64 / 64`, `history_len=2` CNS paper lane for this question, and does not promote any later `2048 / 256 / 256` follow-up lane into current authority by implication.

Blocking verification:

- [ ] Confirm the frozen run root still contains `invocation.json`, `dataset_manifest.json`, `split_manifest.json`, `comparison_summary.json`, `model_profile_author_ffno_cns_base.json`, and `metrics_author_ffno_cns_base.json`.
- [ ] Archive the verification output under the backlog-item artifact root.

### Task 2: Audit And Patch The Authored-FFNO History Path Only If Needed

- [ ] Inspect the current authored-FFNO model construction path and confirm:
  - `AuthorFfnoCnsModel` still uses the authored factorized-FFNO body;
  - `history_len` widens only `in_channels`;
  - task metadata, field order, target channels, coordinate features, and output head contract stay fixed.
- [ ] Inspect the CNS dataset/runner path and confirm `history_len=3` yields:
  - `sample_contract = concat u[t-3:t] -> u[t]`
  - `input_channels = 12`
  - same `512 / 64 / 64` trajectory caps
  - same emitted `4096 / 512 / 512` window counts when `max_windows_per_trajectory=8`
- [ ] If the audit fails, make the smallest code/test patch necessary in the conditional code targets above.
- [ ] If the audit passes with no code changes, skip directly to Task 3.

Blocking verification:

- [ ] Run the required deterministic checks and archive their logs:
  ```bash
  pytest -q tests/studies/test_pdebench_image128_models.py -k 'author_ffno'
  pytest -q tests/studies/test_pdebench_image128_runner.py tests/studies/test_pdebench_cfd_cns_data.py tests/studies/test_pdebench_cfd_cns_metrics.py
  python -m compileall -q scripts/studies/pdebench_image128 scripts/studies/run_pdebench_image128_suite.py
  ```
- [ ] If `reporting.py` or the history-compare path changes, also run a focused supporting selector such as:
  ```bash
  pytest -q tests/studies/test_pdebench_image128_runner.py -k 'author_ffno or history3 or cross_run_compare'
  ```

### Task 3: Produce The `history_len=3` Contract Proof And Matched-Budget Row

- [ ] Run a fresh `history_len=3` inspect pass before any expensive training:
  ```bash
  python scripts/studies/run_pdebench_image128_suite.py \
    --task 2d_cfd_cns \
    --mode inspect \
    --data-root /home/ollie/Documents/pdebench-data \
    --output-root .artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-author-ffno-history-length-compare/history3-inspect-<timestamp> \
    --history-len 3 \
    --max-train-trajectories 512 \
    --max-val-trajectories 64 \
    --max-test-trajectories 64 \
    --max-windows-per-trajectory 8
  ```
- [ ] Confirm the inspect root proves the allowed delta only:
  - `history_len=3`
  - `input_channels=12`
  - expected sample contract string
  - same dataset file, split caps, and metric family
- [ ] If Task 2 changed model-building or runner code materially, optionally run a one-epoch single-profile readiness smoke under a fresh output root before the 40-epoch launch. Treat this as supporting feasibility evidence only.
- [ ] Launch the fresh `history_len=3`, `40`-epoch authored-FFNO pilot in `tmux` inside the `ptycho311` environment, with a unique output root and tracked PID ownership:
  ```bash
  source ~/miniconda3/etc/profile.d/conda.sh
  conda activate ptycho311
  python scripts/studies/run_pdebench_image128_suite.py \
    --task 2d_cfd_cns \
    --mode pilot \
    --profiles author_ffno_cns_base \
    --data-root /home/ollie/Documents/pdebench-data \
    --output-root .artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-author-ffno-history-length-compare/history3-pilot-40ep-<timestamp> \
    --history-len 3 \
    --epochs 40 \
    --batch-size 4 \
    --max-train-trajectories 512 \
    --max-val-trajectories 64 \
    --max-test-trajectories 64 \
    --max-windows-per-trajectory 8
  ```
- [ ] Keep the run under implementation ownership until either:
  - the exact launched PID exits `0`, and
  - the run root contains fresh `invocation.json`, `dataset_manifest.json`, `split_manifest.json`, `normalization_stats_state.json`, `comparison_summary.json`, `model_profile_author_ffno_cns_base.json`, `metrics_author_ffno_cns_base.json`, and tracked launch `exit_code`.
- [ ] If the run fails, diagnose and narrow-fix the cause, then relaunch under a new output root. Do not mark `BLOCKED` unless the failure is unrecoverable after a documented narrow fix attempt.

Blocking verification:

- [ ] The three required deterministic checks from Task 2 must pass before the 40-epoch launch.
- [ ] The `history3` inspect root must exist and show the expected contract before the 40-epoch launch.

Supporting verification:

- [ ] Capture runtime, peak memory, parameter count, valid windows, and split counts from the fresh `history3` run root for later comparison.
- [ ] Archive the tmux launch transcript or equivalent launch log under the backlog-item artifact root.

### Task 4: Compare `history_len=3` Against The Frozen Anchor And Decide The `history_len=4` Gate

- [ ] Use `write_cross_run_compare(...)` to produce `compare_40ep_history3_against_history2.json` and `.csv`.
- [ ] Ensure the compare payload records:
  - `delta_kind = history_len_only`
  - frozen reference `history_len=2`
  - fresh row `history_len=3`
  - absolute metrics and deltas for all required fields
  - valid windows and split counts
  - parameter count, runtime, and peak memory
- [ ] Write `history4_gate_decision.json` before any `history_len=4` launch.
- [ ] Apply this exact gate:
  - open `history_len=4` only if the fresh `history_len=3` row improves all three aggregate metrics versus the frozen `history_len=2` authored-FFNO anchor:
    `err_nRMSE`, `err_RMSE`, and `relative_l2`;
  - and `fRMSE_high` does not worsen beyond floating-point tolerance (`fresh <= reference + 1e-6`);
  - otherwise keep the gate closed.
- [ ] Record the closed-gate reason explicitly if any condition fails.
- [ ] Do not use the spectral-only `history_len=4/5` precedent as an automatic scientific override for authored FFNO. This plan does not grant a pre-run override; observed authored-FFNO results must open the gate.

Blocking verification:

- [ ] Confirm the compare JSON/CSV exist and reference the correct frozen anchor root.
- [ ] Confirm `history4_gate_decision.json` exists before any `history_len=4` training starts.

### Task 5: Run `history_len=4` Only If The Gate Opens

- [ ] If `history4_gate_decision.json` is `closed`, skip this task and proceed to Task 7.
- [ ] If the gate is `open`, run a fresh `history_len=4` inspect pass under a unique output root and confirm:
  - `sample_contract = concat u[t-4:t] -> u[t]`
  - `input_channels = 16`
  - same dataset file, split caps, and metric family
- [ ] Launch the fresh `history_len=4`, `40`-epoch authored-FFNO pilot under the same capped contract and long-run ownership rules as Task 3.
- [ ] Produce `compare_40ep_history4_against_history2_history3.json` and `.csv` so the result can be read both against the frozen paper-lane anchor and the immediately prior longer-history row.
- [ ] Capture the first direct answer for `history_len=4`: improvement, saturation, or degradation relative to `history_len=3`.

Blocking verification:

- [ ] Required deterministic checks remain the same as Task 2 if any code changed after Task 4.
- [ ] The `history4` inspect root and gate-open record must exist before the 40-epoch `history4` launch.

### Task 6: Run `history_len=5` Only If The `history_len=4` Gate Opens

- [ ] Write `history5_gate_decision.json` before any `history_len=5` launch.
- [ ] Open `history_len=5` only if the fresh `history_len=4` row, under the same capped contract, improves `err_nRMSE`, `err_RMSE`, and `relative_l2` versus the freshest prior authored-FFNO row and does not worsen `fRMSE_high` beyond `1e-6`.
- [ ] If the gate is closed, record the reason and skip the launch.
- [ ] If the gate is open:
  - run a fresh `history_len=5` inspect pass and confirm `input_channels = 20`;
  - launch the fresh `history_len=5`, `40`-epoch authored-FFNO pilot under the same long-run rules;
  - write `compare_40ep_history5_against_history2_history3_history4.json` and `.csv`.

Blocking verification:

- [ ] `history5_gate_decision.json` must exist before any `history_len=5` training starts.
- [ ] If a `history_len=5` run happens, the same completion artifacts required in Task 3 must exist for the fresh run root.

### Task 7: Publish The Durable Interpretation And Update Discoverability

- [ ] Write `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_author_ffno_history_length_summary.md`.
- [ ] The summary must state:
  - frozen anchor root and reference metrics;
  - exact capped contract and its allowed delta (`history_len` only);
  - which fresh history rows actually ran;
  - whether longer context improved, saturated, or degraded authored FFNO;
  - whether raw eligible windows shrank and whether emitted capped split counts stayed fixed;
  - the exact gate decisions for `history_len=4` and `history_len=5`;
  - that this evidence is `decision_support` / adjacent capped context only and does not reopen the locked paper lane.
- [ ] Update `docs/plans/NEURIPS-HYBRID-RESNET-2026/evidence_matrix.md` with the completed backlog output and the bounded authored-FFNO history-length read.
- [ ] Add one `model_variant_index.json` entry for each new evaluated authored-FFNO row:
  - `cns_history3_cap512_40ep__author_ffno_cns_base__supervised`
  - `cns_history4_cap512_40ep__author_ffno_cns_base__supervised` if run
  - `cns_history5_cap512_40ep__author_ffno_cns_base__supervised` if run
- [ ] Extend `ablation_index.json` under the CNS history-length family so authored-FFNO longer-history outputs become discoverable alongside the prior spectral / hybrid / FNO history studies.
- [ ] Add a `paper_evidence_index.md` row for this completed backlog item as `decision_support`, making clear that it is a within-model temporal-context ablation, not a headline-row change.
- [ ] Update `docs/studies/index.md` to add the new authored-FFNO history-length summary under the PDEBench CNS study cluster.

Blocking verification:

- [ ] Confirm every evaluated fresh row has a summary mention and the relevant machine-readable index entry.
- [ ] Confirm the locked paper-authority rows and claim boundary language remain unchanged.

## Verification Matrix

### Blocking Before Any 40-Epoch Launch

- Required deterministic checks from the backlog item:
  ```bash
  pytest -q tests/studies/test_pdebench_image128_models.py -k 'author_ffno'
  pytest -q tests/studies/test_pdebench_image128_runner.py tests/studies/test_pdebench_cfd_cns_data.py tests/studies/test_pdebench_cfd_cns_metrics.py
  python -m compileall -q scripts/studies/pdebench_image128 scripts/studies/run_pdebench_image128_suite.py
  ```
- Required contract proof:
  - the matching inspect root for the target `history_len` exists and shows the expected history-derived input channel count and sample contract.

### Supporting Checks

- If code changes touched cross-history compare logic, run a narrower selector that exercises `author_ffno`, `history3`, and `cross_run_compare`.
- If model-building or runner plumbing changed materially, a one-epoch authored-FFNO readiness smoke is allowed before the `40`-epoch run, but it is supporting feasibility evidence only.
- Gallery rendering is supporting only; lack of exact saved-target alignment must be recorded, not treated as item failure.

### Completion Evidence

- Every fresh long run must have:
  - tracked launch ownership with exact `exit_code`;
  - fresh invocation and dataset/split manifests;
  - fresh metrics and model-profile payloads;
  - archived verification logs for the required pytest / compileall commands.

## Long-Run Ownership Rules

- Use `tmux` for any long `history_len=3/4/5` training run and activate `ptycho311`.
- Track the exact launched PID and wait on that PID; do not use broad `pgrep -f` polling as the main completion check.
- Do not launch a duplicate run if another process is already writing to the same `--output-root`.
- A long run is complete only when:
  - the tracked PID exits with code `0`, and
  - the required run-root artifacts for that step exist and are freshly written.
- If a run must be relaunched after a narrow fix, use a new output root and preserve the failed root for provenance.

## Documentation Update Rules

- `docs/index.md` does not need a new top-level entry unless implementation concludes this summary became a first-stop authority beyond the existing `docs/studies/index.md` and evidence indexes.
- If no fresh row is evaluated because the item closes at the `history_len=3` or `history_len=4` gate, still publish the durable summary and evidence-index updates for the actually completed comparison work.
