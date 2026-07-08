# Hybrid-ResNet + Representation/Scaling Comparison Extension — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend the Phase-1 varpro/probe visual comparison along **three comparison axes** that together reproduce the PtychoPINN-CI manuscript's ablation ladder on `fno-stable`:
- **Axis A — decoder representation:** amplitude/phase vs real/imaginary output, matched otherwise (manuscript Fig 3 / Sec 3.2). This axis is NEW and is the reason the real/imag decoder exists: it is a *prerequisite* for the dynamic scaling (FT-linearity makes intensity quadratic in `s1,s2`).
- **Axis B — dynamic scaling across photon flux:** a photon-flux sweep showing the scaling constants absorb the incident-intensity scale (`c_A = √(s1²+s2²) ∝ √counts`, `c_φ = arctan(s2/s1)` flux-invariant), and that inference varpro ON keeps reconstructions measurement-consistent across ~4 orders of flux while OFF does not (manuscript Fig 3c). Reuses the Phase-1 flux-sweep scripts.
- **Axis C — probe-weighted stitching on a non-CNN generator:** the `hybrid_resnet` (srunet) generator under (neither / both) knobs — the first empirical exercise of the merged training physics with a natively-real/imag, non-CNN generator (the original purpose of this extension).

**Architecture:** Reuse the cherry-picked ablation harness and the exact Phase-1 datasets; add (1) an `--architecture` passthrough so arms can select `hybrid_resnet`, and (2) a `--cnn-output-mode` passthrough so the CNN generator can emit `amp_phase` vs `real_imag` for Axis A. Compose one comparison figure per axis (plus a merged overview) from per-arm artifacts.

**Tech Stack:** PyTorch/Lightning (`ptycho_torch`), `scripts/studies/varpro_probe_ablation_runner.py` (cherry-picked in parent-plan Task 2.8), `scripts/studies/grid_lines_torch_runner.py`, `scripts/studies/make_flux_sweep.py` + `flux_sweep_eval.py` (Phase-1, amendment #14), pytest, matplotlib, tmux.

**Execution branch:** `fno-stable` only.

**Parent plan:** `docs/plans/2026-07-01-varpro-probe-scaling-ablation-and-merge.md`. HARD PREREQUISITE: parent Tasks 2.1–2.8 complete — the knobs (`training_patch_weighting`, `physics_forward_mode`, `rect_s1s2_trainable`, **`cnn_output_mode`**), the merged helper, the C=4 regression, the harness, and the Phase-1 findings doc must all exist on fno-stable. Axis A specifically depends on parent Task 2.3 (CNN `cnn_output_mode` real/imag) and Task 2.4 (shared decoder). Do not start otherwise.

**Grounding reference (authoritative for intent):** PtychoPINN-CI manuscript, "Contrast-invariant deep ptychography neural networks" (Vong, Henke, Hoidn, Mehta, Shapiro, Hexemer, Schwarz). Eq 1 (O = s1·ã + j·s2·b̃), Eq 3 (I quadratic in s1,s2), Eq 5 (c_A, c_φ), Eq 6 (LS solve), Fig 3 (RI vs AP), Fig 3c (Fourier error, scaling on/off). Plan-amendments-pending.md **#14** is the verified physical/code summary this plan is built on.

## Global Constraints

- All parent-plan Global Constraints apply verbatim (no worktrees; identical knob names; defaults never change behavior; replace-semantics scale routing; artifacts under git-ignored `.artifacts/`; tmux + tracked-PID for long runs; PATH `python`; conda env `ptycho311` on fno-stable).
- **Datasets (CORRECTED — supersedes the original "fly64_p1e9" reference):** Phase 1 did NOT use `fly64_p1e9` — that subsample is broken (flat object, per-position diffraction correlation 1.0; amendment #14). Phase 1 pivoted to synthetic **lines** (amendment #13b). This extension uses:
  - **Axes A & C:** `.artifacts/varpro_ablation/datasets/lines_N64_{train,test}.npz` (proven-reconstructable, real complex ground truth, counts convention; amendment #13c). No regeneration.
  - **Axis B:** `.artifacts/varpro_ablation/datasets/fluxsweep_N64_{train,mean1_test,mean100_test,mean10000_test}.npz` (one frozen lines object + probe, counts at mean {1,100,10000}; `make_flux_sweep.py`). Regenerate on fno-stable only if absent.
- Training budget parity: same epochs (60), batch size, and seed policy as the Phase-1 matrix so rows are visually comparable.
- No new physics code. Axis A uses the parent-merged `cnn_output_mode`; Axes B/C use the parent-merged `RectangularScaledDiffraction` + varpro inference gate. If a physics change turns out to be needed, that is a parent-plan defect: stop and report, don't patch here.
- `neuralop_uno` remains excluded (gridsize-locked).
- **Metric convention (amendment #14 caveat):** on this pipeline the probe is normalized at inference, so varpro-ON reconstructions come out in **count-amplitude units** (`|O| ∝ √counts`), NOT the manuscript's ~1. All amplitude comparisons to truth MUST divide out the global amplitude gauge `c_A` (equivalently, gauge-quotient as in the recon-quality gate) before scoring, and figures must state which convention (physical vs count-unit) each panel is in.
- **Fidelity metric MUST route through the validated gate placement (Phase-1 lesson, 2026-07-02).** Scale-free amp/phase fidelity-vs-truth is computed with `recon_quality_gate.py`'s direct per-patch placement into the truth frame (lines gs1 baseline 0.97). Do NOT score fidelity by center-cropping the `reconstruct_image_barycentric` canvas against the padded `objectGuess`: the Phase-1 flux-sweep eval tried that and got an untrustworthy amp NCC (~0.25–0.31, contradicting the gate's 0.97 — a framing artifact, phase aligned fine while amplitude did not). Raw (phase-aligned, non-gauged) amp/phase MAE from the harness is fine for showing the count-unit scale effect, but any "does varpro change reconstruction quality" claim uses the gate path.

## Comparison Axes & Arms

**Retrain vs toggle (cost model — do not conflate):**
- **Axis A (amp/phase vs real/imag) requires TWO SEPARATE TRAINING RUNS.** The output parameterization is baked into the decoder architecture, so a checkpoint is mode-specific and CANNOT be toggled at inference. Each representation = its own training run.
- **Axis B (varpro ON/OFF) is a FREE single-checkpoint inference re-solve** — one trained model, both variants produced by `reconstruct_image_barycentric` at reconstruct time (no retrain).
- **Axis C** = one training run per arm (neither/both differ in training-time physics).

### Axis A — decoder representation (amp/phase vs real/imag) — NEW
CNN generator, matched training data (lines) and inference (varpro ON + probe-weighted), differing ONLY in output parameterization. Reproduces manuscript Fig 3 / Sec 3.2 ("share inference-time scaling and probe-weighted stitching but differ only in output parameterization").

| Arm | generator | cnn_output_mode | scaling (inference) | stitching | gridsize |
|---|---|---|---|---|---|
| `repr_ampphase` | cnn | amp_phase | varpro on | probe | 1 |
| `repr_realimag` | cnn | real_imag | varpro on | probe | 1 |

Expected finding (manuscript): AP concentrates information in one channel (scaling only adjusts relative real/imag magnitudes, cannot redistribute learned channel content); RI balances both channels → higher-fidelity amplitude. Report per-arm amp & phase panels PLUS the complex-pixel real-imag scatter overlaid on the unit circle (Fig 3 style).

### Axis B — dynamic scaling across photon flux (amendment #14)
One checkpoint per generator (train once at mean=100), evaluated at mean-count {1, 100, 10000} × inference varpro {ON, OFF}. Generators: `cnn` (real_imag) and `hybrid_resnet`. This is the Phase-1 flux-sweep methodology, re-run on fno-stable to show the scaling generalizes across generators.

Metrics per (generator, flux, varpro): solved `s1,s2`; `c_A=√(s1²+s2²)` (predict ∝ √mean-count); `c_φ=arctan(s2/s1)` (predict flux-invariant); reconstructed `|O|` level; Fourier/measurement error. Phase-1 result to reproduce (CNN, gs1_frozen): c_A ratios 0.098/1.0/10.019 vs √ 0.100/1.0/10.000; c_φ ≈ −65° at all fluxes; varpro-OFF `|O|` flux-invariant, varpro-ON `|O| ∝ √flux`.

### Axis C — probe-weighted stitching on a non-CNN generator (original purpose)
`hybrid_resnet` (emits real/imag natively), gridsize 1, on lines.

| Arm | generator | patch_weighting | physics_forward_mode | rect_s1s2_trainable | inference |
|---|---|---|---|---|---|
| `hybres_gs1_neither` | hybrid_resnet | central_mask | amplitude | (n/a) | uniform, varpro off |
| `hybres_gs1_both` | hybrid_resnet | probe | rectangular_scaled | True | probe, varpro on |

Rationale (updated): gridsize 1 per project default. At gs1, training-time probe weighting is inert (single-patch groups), so the `both` row's real training-side difference is the rectangular-scaled forward; probe weighting + canvas varpro act at inference stitching (same stage split the Phase-1 gs1 arms document). `rect_s1s2_trainable=True` is set in `both` for semantic completeness ("both knobs on") even though it's a training-time no-op at gs1 AND — per amendment #14 — inference re-solves s1/s2 regardless. `neither` = current fno-stable defaults, doubling as a no-regression baseline. This is the first `rectangular_scaled` + `hybrid_resnet` (C=1, native real/imag) training run anywhere.

---

### Task E1: Preconditions gate + registry/knob verification

**Files:** Test: `tests/torch/test_hybres_extension_preconditions.py` (new, small)

- [x] **Step 1: Verify on-branch facts** — `git branch --show-current` == `fno-stable`; `ModelConfig` has `training_patch_weighting`, `physics_forward_mode`, `rect_s1s2_trainable`, **`cnn_output_mode`** (Axis A dep, parent Task 2.3); the generator registry has a resnet-family key (record the EXACT key — `hybrid_resnet` / `srunet` / etc.); the resnet branch's `generator_output` is `real_imag` (read `_build_generator_module_from_config`/`_resolve_generator_from_config` in `model.py`). Confirm the flux-sweep scripts exist (`scripts/studies/make_flux_sweep.py`, `flux_sweep_eval.py`) or are portable from varpro-ablation. `ls scripts/studies/varpro_probe_ablation_runner.py tests/fixtures/varpro_parity`.
- [x] **Step 2: Write the preconditions test** — pin: the four knobs exist with safe defaults (`central_mask`/`amplitude`/`True`/`amp_phase`); the resnet key is registered; `cnn_output_mode` accepts both `amp_phase` and `real_imag`. Run → PASS.
- [x] **Step 3: Commit** — `git add tests/torch/test_hybres_extension_preconditions.py docs/plans/2026-07-01-hybrid-resnet-varpro-probe-extension.md && git commit -m "test: pin preconditions for representation/scaling comparison extension"` (also lands this plan doc if still untracked).

### Task E2: Harness passthroughs (`--architecture`, `--cnn-output-mode`) + arm definitions

**Files:** Modify `scripts/studies/varpro_probe_ablation_runner.py`; Test: extend `tests/torch/test_varpro_probe_ablation_runner.py`

**Interfaces:** consumes the harness arm-definition dict + per-branch condition→overrides adapter and `grid_lines_torch_runner.py`'s `--architecture`/output-mode flags (parent Task 2.7). Produces: `--architecture <key>` and `--cnn-output-mode {amp_phase,real_imag}` on the harness CLI (defaults preserve Phase-1 behavior); arm entries for all three axes.

- [x] **Step 1: Write failing arm-mapping tests** (no training): `repr_ampphase`→`{architecture=cnn, cnn_output_mode='amp_phase', gridsize=1}` with inference `[{'patch_weighting':'probe','varpro_scaling':True}]`; `repr_realimag` same but `cnn_output_mode='real_imag'`; `hybres_gs1_both`→`{architecture=<resnet key>, training_patch_weighting='probe', physics_forward_mode='rectangular_scaled', rect_s1s2_trainable=True, gridsize=1}` inference `[{'probe',True}]`; `hybres_gs1_neither`→ fno-stable defaults, inference `[{'uniform',False}]`. Run → FAIL.
- [x] **Step 2: Implement** the arm entries + both passthroughs into the overrides handed to the runner subprocess. No physics code. Run tests → PASS.
- [x] **Step 3: Smoke** one tiny run (1 epoch) of `repr_realimag` and `hybres_gs1_both` on `lines_N64`; assert checkpoint hparams round-trip `cnn_output_mode`/`physics_forward_mode`. Fresh scratch per amendment #13c.
- [x] **Step 4: Commit** (`feat: architecture + cnn-output-mode passthrough and comparison arms`).

### Task E3: Execute arms (controller-run, tmux)

- [x] **Axis A & C (lines, 60 epochs):** `repr_ampphase`, `repr_realimag`, `hybres_gs1_neither`, `hybres_gs1_both` on `lines_N64_{train,test}.npz` → `.artifacts/varpro_ablation/ext_matrix/`. Completion = exit 0 + fresh metrics.json/PNGs per arm.
- [x] **Axis B (flux sweep):** for generators `cnn`(real_imag) and `hybrid_resnet`: train once at mean=100 on `fluxsweep_N64_train.npz`, then `flux_sweep_eval.py` across mean {1,100,10000} × varpro {ON,OFF}; record s1/s2, c_A, c_φ, |O|, Fourier error → `.artifacts/varpro_ablation/ext_fluxsweep/`.
- [x] **Sanity:** `both`/`repr_realimag` exercised merged physics (checkpoint hparams show the modes); Axis B c_A tracks √flux and c_φ is flux-invariant for BOTH generators (else stop — a scaling regression). Record values.

### Task E4: Composite comparison figures + findings

**Files:** Create `scripts/studies/compose_varpro_comparison_grid.py`; Test `tests/torch/test_compose_varpro_comparison_grid.py`; Modify `docs/plans/2026-07-01-varpro-ablation-phase1-findings.md` (append "Representation/scaling extension" section).

**Interfaces:** consumes per-arm dirs from `.artifacts/varpro_ablation/matrix/` (Phase-1 CNN), `.../ext_matrix/`, `.../ext_fluxsweep/`. Produces one figure per axis + a merged metrics table (`combined_metrics.json` + markdown) under `.artifacts/varpro_ablation/composite/`.

- [x] **Step 1:** Unit test with synthetic arm dirs → asserts each axis figure + combined table exist, one row per arm.
- [x] **Step 2 (Axis A figure):** amp | phase | real-imag-scatter columns for `repr_ampphase` vs `repr_realimag`, shared truth row, shared per-column color scales (per-row scaling fakes differences). Amplitude panels gauge-quotiented (divide out c_A) per the metric convention.
- [x] **Step 3 (Axis B figure):** log-log `c_A` vs mean-count with √ reference line + flat `c_φ`, for both generators; a `|O|`-level and Fourier-error panel for varpro ON vs OFF across flux.
- [x] **Step 4 (Axis C figure):** `hybres_gs1_neither` vs `hybres_gs1_both` rows merged with the Phase-1 CNN dyads (shared truth row / color scales).
- [x] **Step 5:** Append findings: merged metric table, links, and 2–4 sentences per axis (RI-vs-AP channel balance; s1/s2 ∝ √flux + c_φ invariance across generators; hybrid-resnet neither-vs-both). Note the probe-normalization caveat and cite the manuscript. Commit code + tests + findings doc (never artifacts).

### Task E5: Final review gate

- [ ] Run the extension test files plus the parent verification bundle selectors touched by harness changes; archive logs per TESTING_GUIDE. Then whole-diff review per superpowers:requesting-code-review before merging.

## Non-goals / Deferred

- No gs2 hybrid-resnet rows (gridsize is 1 unless specified otherwise).
- No other architectures (FNO, hybrid UNO, spectral bottleneck).
- No changes to physics or config semantics; plumbing + execution + figure assembly only.
- No amp/phase output for `hybrid_resnet` — Axis A (representation) uses the CNN generator, the only one carrying `cnn_output_mode`; extending amp/phase to hybrid_resnet is out of scope.
- **RETRACTED framing (do not reintroduce):** the earlier hypothesis that trainable `rect_s1s2` helps by "extending the decoder box for strong-phase objects" is WRONG (amendment #14). s1/s2 are a global per-dataset photon/amplitude-scale factorization, distinct from the real_imag decoder box (#13). The scaling axis is a PHOTON-FLUX generalization test, not a phase-strength test.

## Open risks

- First `rectangular_scaled` + `hybrid_resnet` training (C=1): if it destabilizes (NaN loss), record the failure mode and report — do not tune physics here.
- **Probe-normalization convention (amendment #14):** varpro-ON `|O|` is in count-amplitude units on this pipeline (∝ √counts), unlike the manuscript's ~1. Axis A/B amplitude scoring MUST gauge-quotient; if a future parent task adopts the manuscript's inference-probe convention, the raw `|O|` numbers here shift by `c_A` (the comparison conclusions do not).
- Axis A depends on parent Task 2.3 (`cnn_output_mode`) being complete and the supervised path being unaffected by the knob (parent test). If `real_imag` silently alters the CNN default, stop — that's a parent-plan defect.
- Phase-1 harness may save only PNGs, not canvases; the compose step's PNG fallback loses colormap fidelity — prefer the harness's per-variant canvas NPZ (parent-plan Task 1.5 amendment #7).
