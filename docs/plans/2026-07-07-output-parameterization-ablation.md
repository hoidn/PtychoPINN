# Output-Parameterization Ablation — does the hybres poisson failure survive an amp/phase head?

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development. Briefs/reports under `.superpowers/sdd/ext/`, ledger `.superpowers/sdd/progress.md`.

**Date:** 2026-07-07 · **Branch:** `fno-stable` · **Status:** COMPLETE 2026-07-07
**Parent findings:** POISSON-LADDER-001 (H1 objective underdetermination; universal poisson phase collapse; RCA fix-candidate #4 "bound hybres's output head" untested), POISSON-SCALE-001 (count-lift not outcome-preserving — NOT used here).

**Goal:** One controlled A/B answering: is hybrid_resnet's poisson failure (grainy amplitude ~0.284, phase collapsed to near-constant) specific to the `real_imag` output parameterization, or does it persist with an amplitude+phase head? Every ladder run to date used `--output-mode real_imag`, so this variable has never been isolated.

**Why it matters:** Track B proved the objective's high-q blindness is parameterization-independent (it lives in object space), but the *implicit prior* that picks a point in the flat set is the head/parameterization. A bounded amp head is the cheapest architecture-side mitigation candidate; phase recovery, if seen, would be the first poisson recipe to reconstruct phase at all.

## Global constraints

- NO code edits anywhere (run-only + docs). If the amp/phase mode is unsupported for hybrid_resnet, the initiative STOPs with a BLOCKED report — no enabling changes under this plan.
- Protected files untouchable as always (`ptycho/model.py`, `ptycho/diffsim.py`, `ptycho/tf_helper.py`, `ptycho/config/config.py`, `ptycho_torch/config_params.py`, `ptycho_torch/model.py`); gs2's no-touch set likewise (`ptycho_torch/{helper,data_container_bridge,patch_generator}.py`).
- Frozen evidence: all existing run dirs. New outputs ONLY under `.artifacts/varpro_ablation/poisson_ladder/diag/rung2_amp_phase_{poisson,mae}/`.
- Do NOT pass `--count-scale-mode` (default `off` = the frozen runs' loss units; POISSON-SCALE-001 forbids comparing lifted runs to these references).
- GPU discipline: double `nvidia-smi` free check ≥30 s apart before each launch; one run at a time; exact-PID tracking (`wait "$pid"`), never `pgrep -f` loops; 40-min busy-wait cap then BLOCKED (gs2 session shares the GPU).
- Commits: Task 3 docs only, ONE commit, explicit `git add` paths, staged-purity check (shared tree), no "claude" in messages, no trailers, never `--no-verify`, no pushes.
- Metrics arrays are [amplitude, phase]. Phase verdicts REQUIRE the visual check of `visuals/compare_amp_phase.png` — frc50 on near-constant phase maps is unreliable (ledgered lesson); the visual overrides the number.

### Task 1 — CPU pre-flight (read-only) ✅ gate for Task 2
1. Confirm `--output-mode amp_phase_logits` (argparse choices: `real_imag`, `amp_phase_logits`, `amp_phase`) is supported for `--architecture hybrid_resnet` — check runner dispatch + generator code for architecture gating; cite file:line.
2. Pin the semantics: which mode gives a BOUNDED amplitude head (what activation), and how phase is produced. Choose the bounded-amp variant (expected: `amp_phase_logits`); record the choice + citation.
3. Read both frozen invocations to be cloned (flag source of truth):
   `.artifacts/varpro_ablation/poisson_ladder/rung2_hybres_orig/runs/pinn_hybrid_resnet/invocation.json` (poisson recipe) and `.artifacts/varpro_ablation/ext_matrix_aligned/neither/runs/pinn_hybrid_resnet/invocation.json` (MAE recipe).
- Unsupported / no bounded variant ⇒ BLOCKED, stop the plan.

**Complete (2026-07-07).** Mode chosen: `amp_phase_logits` (bounded sigmoid amplitude / π·tanh phase, `ptycho_torch/model.py:94-104`, over the byte-identical real_imag core network, `ptycho_torch/generators/hybrid_resnet.py:693-694`); `amp_phase` rejected as a confound (adds dedicated conv heads). See `.superpowers/sdd/ext/task-outputmode-ablation-report.md`.

### Task 2 — GPU A/B (two 5-epoch runs, poisson first)
Clone each frozen invocation EXACTLY, changing ONLY `--output-mode` (→ the Task 1 choice) and `--output-root`:
1. **Test:** poisson recipe → `diag/rung2_amp_phase_poisson/`
2. **Control:** MAE recipe → `diag/rung2_amp_phase_mae/` (isolates any parameterization-intrinsic degradation under the known-good objective)

Readout per run: mae[0], mae[1], frc50[0], frc50[1], final live train loss, and the mandatory visual amp+phase description.

**Interpretation matrix (references: poisson+real_imag 0.2839/grainy/flat-phase; MAE+real_imag 0.0781/clean/structured-phase):**
- **Case A (parameterization-independent):** control healthy (amp mae ≈0.08–0.12, phase visually structured) AND test still grainy with visually flat phase → the NLL failure survives the amp/phase head; fix-candidate #4 demoted.
- **Case B (parameterization-dependent):** control healthy AND test materially improved — amp mae well below 0.28 (crossing the 0.1562 rung gate = strong signal) and/or phase VISUALLY structured → the bounded head is a real architecture-side prior; phase recovery would be the first phase-reconstructing poisson recipe: flag loudly.
- **Case C (confounded):** control itself degraded vs 0.0781 → amp/phase mode has an intrinsic problem on hybres; no poisson conclusion; record and stop.
- **Contingency (bounded):** if the test result is ambiguous between A and B (e.g. amplitude improves but phase stays flat), ONE optional extra run — cnn + poisson + same amp/phase mode (clone `rung1_cnn_orig` invocation) — to check whether the phase verdict generalizes across architectures. Controller decides; no other recipe iteration.

**Complete (2026-07-07).** Mechanical flag-name correction (approved by controller): the runner flag is `--output-dir`, not `--output-root` as drafted above. Results (mae[0]/mae[1], phase visual): poisson+real_imag ref 0.28393/0.17940 flat; poisson+amp_phase_logits test 0.22717/0.25054 flat (near-constant, pinned near −π; frc50[1] rise is the known flat-phase FRC artifact); MAE+real_imag ref 0.07806/0.13042 structured; MAE+amp_phase_logits control 0.08241/0.12745 structured. Control healthy + test still flat-phase/grainy ⇒ **Case A**: fix-candidate #4 demoted. Amp-mae anomaly (~20% relative improvement, no structural correlate) recorded, not overstated. Contingency (cnn-twin) declined — case verdict unambiguous. See `.superpowers/sdd/ext/task-outputmode-ablation-report.md`.

### Task 3 — Docs (ONE commit)
- `docs/findings.md` POISSON-LADDER-001: append the ablation result to the phase-caveat/remedies area — which case matched, the four mae numbers, visual phase verdicts, and the updated status of fix-candidate #4 (demoted / promoted / confounded). If Case B with phase recovery, note it as the current leading practical mitigation and cross-ref this plan.
- This plan: mark tasks complete with the result + report path.
- Reports: `.superpowers/sdd/ext/task-outputmode-ablation-report.md` (Tasks 1–2), `task-outputmode-docs-report.md` (Task 3).

**Complete (2026-07-07).** `docs/findings.md` POISSON-LADDER-001 appended with the "Output-parameterization ablation (2026-07-07)" addendum (Case A verdict, four mae pairs, fix-candidate #4 demoted, anomaly note, cross-ref to this plan). Report: `.superpowers/sdd/ext/task-outputmode-docs-report.md`.

## Non-goals
No hybrid poisson+MAE objective work (protected model.py — separate plan), no 20-epoch horizons, no seed sweeps, no count-lift interaction, no TF backend.
