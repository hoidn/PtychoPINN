# Extension Recovery Plan — post-bisect fix wave + E5 gate

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to execute task-by-task. This plan is the recovery sequence to run AFTER the hybrid_resnet regression bisect/debugging completes and releases the working tree. Do NOT start any task before Gate R0 passes.

**Goal:** Restore a verified working tree, fix all audited deficiencies of the hybrid-resnet/representation/scaling extension (E0–E4), disposition the pre-existing hybrid_resnet integration-test regression per the bisect verdict, and close the initiative through the held E5 final review gate.

**Requirement sources (authoritative, in `.superpowers/sdd/ext/`):** `audit-summary.md` (consolidated deficiency list C1/I1–I5 + minors), `audit-code.md`, `audit-e3.md`, `audit-e4.md`, `audit-numerics.md`, `bisect-report.md` (written by the bisect agent on completion). Parent plan: `docs/plans/2026-07-01-hybrid-resnet-varpro-probe-extension.md`; extension base c0798151, pre-fix tip a1d4d368.

## Global Constraints
- Branch `fno-stable` only. No worktrees. Conda env `ptycho311`; invoke Python via PATH `python` (PYTHON-ENV-001).
- Read-only core: `ptycho/model.py`, `ptycho/diffsim.py`, `ptycho/tf_helper.py`, `ptycho/config/config.py`, `ptycho_torch/config_params.py`. No physics/config-semantics changes — plumbing, metrics persistence, figures, docs only.
- No artifacts committed; everything bulky stays under git-ignored `.artifacts/`.
- No "claude" in commit messages. Never `--no-verify`. No push/PR without explicit user approval (origin is public).
- Long GPU/eval runs: tmux + tracked PID (`cmd & pid=$!; wait "$pid"`), no pgrep polling.
- Datasets are frozen: `.artifacts/varpro_ablation/datasets/*` — never regenerate.

---

### Task R0: Tree recovery gate (controller-run, blocking)

- [ ] Verify the bisect released the tree: `git rev-parse --abbrev-ref HEAD` == `fno-stable` AND `git rev-parse --short HEAD` == `a1d4d368`. If not: run `git bisect reset fno-stable` ONLY if `.git/BISECT_LOG` exists; else `git checkout fno-stable`; re-verify.
- [ ] `git status --porcelain` shows only known untracked paths (`.superpowers/`, this plan file, `notebooks/archive/ePIE_recon_simulation`, `scripts/orchestration`, `tmp/`). Any tracked-file modification = STOP, investigate before proceeding (the bisect was forbidden from modifying tracked files).
- [ ] `git submodule update --init --recursive` and confirm `ls ptycho/FRC/` is non-empty.
- [ ] Clear bisect scratch: `rm -rf .artifacts/integration/grid_lines_hybrid_resnet` (each consumer regenerates; avoids stale cross-commit memory-maps).
- [ ] Confidence check that the environment is back to extension-era behavior: `python -m pytest tests/torch/test_hybres_extension_preconditions.py -q` → 5 passed.

### Task R1: Bisect-verdict disposition (controller + USER DECISION)

- [ ] Read `.superpowers/sdd/ext/bisect-report.md`. Classify per its own a/b/c/d taxonomy: (a) genuine training/model regression, (b) data-pipeline regression, (c) baseline/fixture tightening, (d) combination.
- [ ] Present to user with the first-bad commit + mechanism, and ask ONE decision (D1): **fix the regression now** (new debugging/fix task appended to this plan), **rebaseline the fixture** (regenerate `tests/fixtures/grid_lines_hybrid_resnet_metrics.json` at current behavior + document why in the fixture commit), or **defer** (document as known-failing in `docs/bugs/` with the bisect evidence; test stays red). Do not pick silently — this changes what E5's "tests pass" means.
- [ ] Execute the chosen disposition as its own commit(s) before E5.

### Task R2: Audit fix wave — code (one implementer subagent)

**Files:** Modify `scripts/studies/compose_varpro_comparison_grid.py`, `scripts/studies/flux_sweep_eval.py`, `scripts/studies/varpro_probe_ablation_runner.py`; extend their tests (`tests/torch/test_compose_varpro_comparison_grid.py`, `tests/torch/test_flux_sweep_eval_cli.py`, `tests/torch/test_varpro_probe_ablation_runner.py`).
**Brief inputs:** `audit-summary.md` items C1, I1, I4 + minor rollup (stale comment, `parse_fidelity_table` warning). TDD per item.

- [ ] **C1 (Critical):** `combined_metrics` `amp_fidelity_ncc` must be single-methodology: route Axis A/C rows through the validated gate placement (`recon_quality_gate.py` path) or emit them in a separately-named, per-row-labeled column; never mix banned center-crop NCC with gate-path values under one header. Add a regression test that fails if a center-crop NCC lands in a gate-labeled column.
- [ ] **I1:** add measurement-domain (Fourier/diffraction) error to `flux_sweep_eval.py` per plan L52/L89 — predicted vs measured diffraction per (flux × varpro ON/OFF); test on synthetic input.
- [ ] **I4:** persist solved `s1, s2, c_A, c_phi` as machine-readable artifacts (metrics.json fields / eval NPZ), full precision — makes headline scalars independently verifiable.
- [ ] **Minors:** fix stale ">=2 variants" comment; `parse_fidelity_table` warns loudly on empty parse; tick the completed E1–E4 checkboxes in `docs/plans/2026-07-01-hybrid-resnet-varpro-probe-extension.md`; commit this recovery plan file.
- [ ] Run the four test files; commit incrementally (one logical change per commit).

### Task R3: Regeneration runs (controller-run; inference-only, no retraining)

- [ ] Re-run both flux evals with the I1/I4-extended eval: `python scripts/studies/flux_sweep_eval.py --out .artifacts/varpro_ablation/ext_fluxsweep/cnn_ri --label cnn_ri --skip-anchor` (and `hybres`) → tables now include measurement error + persisted scalars.
- [ ] **I2 anchor replacement:** validate the FIDELITY methodology on-branch — run `recon_quality_gate.py` against one fno-stable flux checkpoint and compare its amp fidelity to the eval's FIDELITY line; record agreement (or discrepancy → STOP, report) in the findings doc.
- [ ] Re-run `python scripts/studies/compose_varpro_comparison_grid.py` → regenerated `composite/` with single-methodology fidelity column + convention-labeled panels.
- [ ] Spot-verify: lines-dyad fidelity in the new table is ~0.97-consistent (gate path), not 0.28–0.32.

### Task R4: Findings-doc corrections (same or fresh subagent, doc-only commit)

- [ ] **I3:** fix the `canvas_amp_std` misattribution (0.0014 is `hybres_gs1_both`; `hybres_gs1_neither` is 1.33); keep the degenerate-recon caveat (gauged amp fidelity ~0.055, s1=131.65) attached to the `both` MAE win.
- [ ] **I5:** reword "AP concentrates information in one channel (near zero)" → attenuated, with the actual recorded ratio (weak channel ≈ 37% of strong).
- [ ] **I2 caveat:** wherever FIDELITY qualifiers are quoted, state the anchor-check replacement from R3 and its result.
- [ ] Refresh any numbers the R3 regeneration changed (new fidelity column, measurement-error values).

### Task R5: E5 final gate (held from the extension plan)

- [ ] Task-review the R2–R4 diff (reviewer subagent, brief = audit-summary items), fix wave if needed.
- [ ] Run and archive (per TESTING_GUIDE) the extension selectors: `tests/torch/test_hybres_extension_preconditions.py tests/torch/test_varpro_probe_ablation_runner.py tests/torch/test_flux_sweep_eval_cli.py tests/torch/test_compose_varpro_comparison_grid.py tests/torch/test_ptycho_dataset_normalized_amplitude.py` plus the parent verification-bundle selectors touched by harness changes (derive from parent plan `2026-07-01-varpro-probe-scaling-ablation-and-merge.md`) plus `tests/torch/test_grid_lines_hybrid_resnet_integration.py` (expected outcome per D1 disposition).
- [ ] Regenerate the whole-branch review package `scripts/review-package c0798151 HEAD` (now includes fix commits) and dispatch the final reviewer (most capable model) with the accumulated-minors list for triage.
- [ ] Present to user for decision (D2): merge/push/PR — never push without explicit approval.

## Contingencies
- **Bisect inconclusive / noisy** (many 125 skips, first-bad adjacent to skipped range): narrow manually around the reported range; the `b069bfaa` sys.path boundary is already environment-compensated (PYTHONPATH). If signal is training-stochastic, re-run the boundary commits 2–3× before trusting; if still ambiguous, present the narrowed range to the user under R1-D1 with "defer" recommended.
- **Bisect left tree dirty/detached and agent unreachable:** R0's manual reset path recovers it; the fno-stable ref itself is never moved by bisect.
- **R3 gate-vs-eval FIDELITY disagreement:** stop before R4; that would falsify the extension's fidelity claims and needs its own root-cause task, not a wording fix.
