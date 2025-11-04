# Phase G — Comparison & Analysis (Planning)

## Context
- **Initiative:** STUDY-SYNTH-FLY64-DOSE-OVERLAP-001
- **Phase Goal:** Produce three-way comparison artifacts (PINN vs baseline vs pty-chi) for every dose/view/split combination using shared metrics (MS-SSIM phase/amplitude, MAE, MSE, PSNR, FRC) with deterministic alignment/registration.
- **Dependencies:** Phase C datasets, Phase E TensorFlow runs (PINN + baseline checkpoints), Phase F LSQML manifests/logs, CONFIG-001 config bridge, DATA-001 NPZ contract, POLICY-001 torch availability, OVERSAMPLING-001 sparse acceptance thresholds.
- **Artifact Hub:** `plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-05T140500Z/phase_g_comparison_plan/{plan,red,green,collect,cli,analysis,docs}/`

### G0 — Evidence Inventory & Harness Prep
Goal: Confirm prior phases expose the required inputs and define RED harness for new comparison orchestrator.
Prereqs: Phase F3 manifests copied (sparse train/test), Phase E outputs reachable, dataset registry up to date.
Exit Criteria: Inventory doc lists per-dose artifacts, test strategy section updated with Phase G selectors, RED pytest stub recorded.

| ID | Task Description | State | How/Why & Guidance |
| --- | --- | --- | --- |
| G0.1 | Catalog required inputs (PINN checkpoints, baseline checkpoints, Phase F manifests, Phase C/D NPZ paths) and record under `reports/2025-11-05T162500Z/phase_g_inventory/analysis/inventory.md` | [x] | Attempt #91 inventory captured canonical paths, acceptance stats, and gaps (dose_1000 sparse/test + higher doses) in `analysis/inventory.md`. |
| G0.2 | Update `test_strategy.md` Phase G section with active selectors, execution proof policy, and artifact destinations | [x] | Phase G section refreshed (Attempt #91) with active selectors, collect-proof log references, and G2 execution guardrails; pending selectors will be appended after real runs. |
| G0.3 | Land deterministic pytest in `tests/study/test_dose_overlap_comparison.py` covering all dose/view/split combinations | [x] | Attempt #90 (2025-11-05T140500Z) — see `green/pytest_phase_g_target_green.log`; test now asserts 12 jobs and validates metric config. |

### G1 — Comparison Job Orchestration
Goal: Implement deterministic job builder + CLI to invoke `scripts/compare_models.py` across all conditions with metadata mirroring Phase F manifests.
Prereqs: G0 RED harness in place; plan references this section.
Exit Criteria: `build_comparison_jobs` returns 18 jobs (3 doses × 3 views × 2 splits), CLI filters confirmed via tests, manifest/summary emitted with metrics placeholders.

| ID | Task Description | State | How/Why & Guidance |
| --- | --- | --- | --- |
| G1.1 | Implement `studies/fly64_dose_overlap/comparison.py::build_comparison_jobs` producing dataclasses with pointers to PINN, baseline, Phase F pty-chi outputs | [x] | Attempt #90 delivered job builder (12 jobs: 3 doses × 2 views × 2 splits) with fail-fast path validation. Evidence: `green/pytest_phase_g_target_green.log`. |
| G1.2 | Extend tests to assert 12 jobs, deterministic ordering, and per-job metric config (ms_ssim_sigma=1.0, registration flags) | [x] | Attempt #90 tightened assertions on ordering + metric config. Collect proof: `collect/pytest_phase_g_collect.log`. |
| G1.3 | Add CLI entry `python -m studies.fly64_dose_overlap.comparison` supporting filters (`--dose`, `--view`, `--split`, `--dry-run`) and manifest/summary emission | [x] | Attempt #90 implemented CLI dry-run (manifest + summary). Evidence: `cli/phase_g_cli_dry_run.log`. |

### G2 — Deterministic Comparison Runs
Goal: Execute comparisons for dense/sparse views and capture aligned NPZs, CSV metrics, plots.
Prereqs: G1 CLI implemented; ensures dataset outputs exist.
Exit Criteria: Each dose/view/split run produces metrics CSV, aligned NPZs, summary updates; sparse runs document acceptance context.

| ID | Task Description | State | How/Why & Guidance |
| --- | --- | --- | --- |
| G2.1 | Run dense/train + dense/test comparisons, capture CLI logs, metrics CSV, plots | [ ] | Store under `cli/dense_{split}.log`, `analysis/dense/{split}/metrics.csv`, etc. Note MS-SSIM obs in summary. |
| G2.2 | Run sparse/train + sparse/test comparisons; include handling for singular LSQML results (expected) | [ ] | Accept non-zero return codes, ensure manifest records `selection_strategy="greedy"`. |
| G2.3 | Execute gs1 baseline comparisons (train/test) for parity with Phase E outputs | [ ] | Document that iterative input absent (two-way comparison) and note differences. |

### G3 — Analysis & Documentation
Goal: Summarize results, update documentation, and register selectors/commands.
Prereqs: G2 artifacts ready; tests GREEN.
Exit Criteria: `docs/TESTING_GUIDE.md` and `docs/development/TEST_SUITE_INDEX.md` updated; summary.md synthesizes metrics; docs/fix_plan attempt recorded.

| ID | Task Description | State | How/Why & Guidance |
| --- | --- | --- | --- |
| G3.1 | Draft `analysis/summary.md` capturing MS-SSIM trends per dose/view/split | [ ] | Align findings with OVERSAMPLING-001 (sparse acceptance) and CONFIG-001 (pure orchestrator). |
| G3.2 | Update documentation registries with Phase G selectors + commands; include collect-only proof logs | [ ] | Provide references to `collect/pytest_phase_g_collect.log` and CLI run logs. |
| G3.3 | Log Attempt history + findings updates (if any new lessons) in `docs/fix_plan.md` and `docs/findings.md` | [ ] | Note differences vs Phase F and highlight any new policies. |

## References
- `docs/TESTING_GUIDE.md` §§2,4 — authoritative test + CLI commands (set `AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md` before execution).
- `docs/development/TEST_SUITE_INDEX.md` — register new Phase G selectors.
- `docs/findings.md` CONFIG-001, DATA-001, POLICY-001, OVERSAMPLING-001 — ensure compliance across comparison pipeline.
- `plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T094500Z/phase_f_ptychi_baseline/` — manifests + logs for LSQML outputs consumed during Phase G.
