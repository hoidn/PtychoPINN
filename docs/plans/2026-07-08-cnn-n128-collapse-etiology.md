# CNN N=128 Collapse — Etiology RCA (post-mechanism factorization)

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development. Controller-owned GPU runs follow the PID-tracked/tmux conventions in CLAUDE.md.

**Goal:** identify WHY the cnn generator flat-collapses under the rectangular_scaled count-Poisson recipe at N=128, given that the same architecture at the same N trains fine under every other Poisson plumbing tested — including origin/main's reference.

**Status:** COMPLETE and SUPERSEDED (2026-07-08). The discriminating phase finished with "torch-port training-dynamics defect, prime suspect learnable intensity_scale" — that suspect was subsequently REFUTED and the root cause isolated (CBAM encoder attention) by the follow-on campaign: see `docs/plans/2026-07-08-cnn-n128-tf-parity.md` and the PARITY RESOLUTION in `docs/findings.md#TORCH-N128-FLAT-AMP-001`. This plan's data and eliminations all stand; only its final suspect attribution is superseded. *(Original status line: DRAFT; supersedes the causal framing, not the data, of the Task 9b mechanism note.)*

## Why this plan exists (user challenge, 2026-07-08)

The user observed that the origin/main reference cnn does not exhibit this failure mode at N=128. Verified — and the challenge generalizes:

| # | Configuration (all cnn arch unless noted) | Outcome | Evidence |
|---|---|---|---|
| 1 | fno-stable, rect-count Poisson, N=128, K-NN lines counts (108 cts/px, 1.77e6 ph/img), seeds 42/11/17, gs1+gs2 | **FLAT COLLAPSE** (objframe pearson ~0.05, canvas_std ~1e-3), converged, 50ep no escape | `c3_lines128/`, `task9_seed/`, Task 9 |
| 2 | fno-stable, SAME recipe/data, N=64 (4.4e5 ph/img) | works (0.916) | Task 9 |
| 3 | fno-stable, SAME recipe/data/N, hybrid_resnet | works (0.872–0.985) | Task 9, 3R |
| 4 | fno-stable, MAE/amplitude recipe, N=128 | works (corr 0.95 @5ep, 0.96 @20ep) | aligned ablation |
| 5 | fno-stable, **Poisson via amplitude forward** (`physics_scale`=328.7 internal count-lifting ⇒ effective ~1e9 ph/img), N=128 aligned grid-lines | **works** (mae 0.166, structured; phase weak @5ep — separate known caveat) | ladder Rung 1R |
| 6 | origin/main reference cnn CLI, N=128 grid-lines gs1/gs2 (its own amplitude-Poisson regime + double-division equilibrium) | works-ish (masked pearson 0.740/0.591, washed — RECT-PROBE-SCALE-DOUBLE-DIV-001) | mainref Lane A |
| 7 | origin/main encoder code | **has the IDENTICAL `N==128` 4-stage depth branch** as fno-stable | `origin/main:ptycho_torch/model.py:204` |

Rows 5–7 bound the claim: the 4-stage encoder depth trains to a non-collapsed solution at N=128 under at least two other Poisson regimes (one on our own stack). Therefore the Task 9b ReLU-dead-fraction cascade is the **anatomy of the failed basin** (how the collapsed network is dead — those measurements stand), **not the etiology** (why optimization lands there). Depth is a susceptibility factor at most. The un-factored differential is confined to what row 1 has that rows 4/5/6 don't: the **rectangular_scaled count-level Poisson recipe** and/or its **likelihood scale** (1.77e6 ph/img vs Rung 1R's effective ~1e9) and/or its **data statistics** (K-NN uint16 counts vs normalized float32).

## Hypotheses (ranked)

- **H-A — dose × depth interaction.** Poisson NLL sharpness scales with counts. At 1.77e6 ph/img the likelihood gradient is ~565× weaker than Rung 1R's effective 1e9; too weak to pull a 4-stage ReLU encoder out of the attriting basin, while N=64's 3-stage encoder (row 2) escapes at even lower dose and hybres (row 3) has no such basin. Consistent with EVERY row above, including the dose "irony" (below-bar N=64 works — shallow encoder needs less guidance). Prediction: a dose ladder at fixed cnn/N=128/rect-count unlocks reconstruction at some `target_mean_count`.
- **H-B — rectangular_scaled path defect at N=128 (cnn-conditional).** Something specific to the rect forward's constants/dynamics (`s1/s2` early trajectory, `output_scale` interplay, probe convention post-P1) mis-guides cnn at N=128. Task 9 refuted N-dependence of the *converged* constants but never compared rect vs amplitude forward at matched dose/data. Prediction: cnn+Poisson at the SAME 1.77e6 dose through the amplitude forward reconstructs while rect collapses.
- **H-C — training-time data statistics.** 9b's random-noise control exonerated input sparsity for the *trained* encoder (inference-time), not as a training-dynamics factor. K-NN lines counts (67.9% exact-zero px) might shape early optimization differently from dense-grid or normalized data. Prediction: cnn+rect-count on `gridgeom_N128` (dense-grid counts, same dose — data already exists) behaves differently from row 1.
- **H-D — CBAM / init / LR secondary factors.** Only if A–C all null: 9b's listed controls (CBAM monkeypatch on frozen ckpt; 3-stage-encoder retrain at N=128 via stride-2 pre-downsample).

## Tasks

### Task 0 (docs, immediate — lands with this plan): re-scope the findings row
Amend `TORCH-N128-FLAT-AMP-001`: (a) add rows 5–7 above to the evidence; (b) reframe the 9b paragraph as basin anatomy, explicitly NOT establishing depth causality (main + Rung 1R train the same depth to non-collapse); (c) point the open etiology question at this plan.

### Task 1 (CPU, one agent): pin the reference-regime facts
- Main's Lane A run: exact loss mode, `physics_scale`/normalization handling, effective photons/img of the grid-lines data it trained on (read the mainref evidence bundle + origin/main code — NO new runs; main cannot ingest count npz anyway, NORMALIZE-DATA-UINT16-001).
- Diff fno-stable vs origin/main cnn model code beyond the (already-verified, identical) encoder depth branch: CBAM defaults, decoder head, probe_big, activation choices. Output: a one-page differential table.
- Compute the effective per-image photon scale of every row in the evidence matrix (fill the dose column exactly).

### Task 2 (GPU, ~40 min, cheapest discriminator first): E3 geometry control
`gs1_trainable --N 128` (default cnn) on **existing** `gridgeom_N128_{train,test}.npz` (dense-grid counts, 108 cts/px). 25ep/batch-8 protocol, fresh root `.artifacts/varpro_ablation/etiology/e3_cnn_gridgeom_n128/`.
- Collapse (expected under H-A/H-B) ⇒ K-NN-geometry/statistics largely exonerated → H-C down-weighted.
- Reconstructs ⇒ H-C promoted: the differential is in the K-NN count data itself.

### Task 3 (GPU + dataset builds, ~2.5 h): E1 dose ladder (tests H-A)
Extend the lines builder invocation (existing `make_lines_datasets` machinery; NEW filenames + provenance per SYNTH-POISSON-DOSE-001 — measured dose recorded, nphotons omitted/annotated) at `target_mean_count ∈ {432, 1728, 6912}` cts/px ⇒ {7.1e6, 2.8e7, 1.1e8} ph/img at N=128. Run cnn/rect-count/N=128/gs1, 25ep each, T2 readout.
- Unlock at some rung ⇒ H-A confirmed with a measured threshold; findings row gains dose-conditionality; SYNTH-POISSON-DOSE-001 gains the cnn-recipe floor.
- All collapse up to 1.1e8 ⇒ H-A refuted → H-B primary.

### Task 4 (GPU + one twin build, ~1 h): E2 forward-path swap at matched dose (tests H-B)
Build a normalized-amplitude twin of `lines_N128` (respect NORMALIZATION-001: `diffraction` = normalized amplitude, per-image photon scale carried in metadata so the pipeline derives `physics_scale` ≈ sqrt(1.77e6-scale) internally — mirror the Rung 1R/ladder-R1 convention EXACTLY; the ladder's Rung-1 twin misdesign is the cautionary reference). Run cnn+Poisson through the **amplitude** forward at N=128.
- Reconstructs (at 1.77e6 effective dose, where rect collapses) ⇒ H-B implicated: rect-path audit follows (s1/s2 early-epoch trajectories, gradient probes; RECT-MAE-UNITS-001/RECT-B5-SCALE-001 lineage).
- Collapses ⇒ rect exonerated; combined with Task 3 this isolates likelihood scale (H-A) cleanly.

### Task 5 (conditional, from 9b's falsification list): H-D controls
Only if Tasks 2–4 fail to discriminate: CBAM `nn.Identity` monkeypatch on the frozen collapsed checkpoint (CPU, hours-cheap); then a 3-stage-encoder N=128 retrain (stride-2 pre-downsample; requires a scoped, authorized model-code change — user gate).

### Task 6 (docs close-out)
Findings amendment with the discriminated etiology; ledger; plan status COMPLETE.

## Sequencing & budget
Task 0 with this commit. Task 1 (CPU) parallel with Task 2 (GPU). Then 3 → 4 (GPU serial). Total GPU ≈ 4 h worst case; every run 25ep/gs1/batch-8, PID-tracked in tmux, headroom gate ≥10240 MiB, T2 metric basis, artifacts git-ignored with provenance JSONs.

## Constraints
Frozen files untouched (Task 5's encoder-depth control is explicitly user-gated as a plan-authorized exception if reached). Frozen datasets never regenerated; all new datasets get new filenames + measured-dose provenance. No pushes to origin. Seeds: default 42 (the collapse is seed-robust; do not spend GPU on seed replication here).

## Amendments (2026-07-08, post-Task-1 + E3)

1. **Evidence row 6 CORRECTED — main was never run at N=128.** Task 1 (regime pinning, `.superpowers/sdd/ext/etiology-task1-report.md`) verified the mainref Lane A numbers (0.740/0.591) came from `deadleaves_N64` data (`train.log` shows N=64; `grid_lines_torch_runner.py` does not exist on origin/main). Main's non-collapse evidence is N=64-only — where fno-stable's cnn also works (row 2). Row 7 (identical depth branch) stands but is moot for the main comparison. Rung 1R (row 5) is the ONLY valid N=128 cross-regime counter-example.
2. **NEW H-E — U-Net skip connections dropped in fno-stable's cnn (code-verified, controller-confirmed).** origin/main `Encoder.forward` returns `(x, skips)` with per-stage skip collection and decoder `merge_blocks`; fno-stable's `Encoder.forward` returns `x` only (no skips anywhere in the cnn path; decoder attention blocks constructed but never invoked). The skip-less decoder depends entirely on the bottleneck — exactly where 9b measured the signal death (CV 0.0037) while early stages remain healthy (block0 CV 0.473, food for a U-Net decoder). H-E composes with H-A: no-skips raises the likelihood-guidance threshold needed to escape the attriting basin (Rung 1R's effective ~1e9 clears it at N=128; 1.77e6 does not; N=64's 3-stage encoder needs less). Definitive H-E test = restore skips and rerun row 1 — requires a `ptycho_torch/model.py` change (FROZEN: user authorization required; queued as a Task 5 option).
3. **E3 verdict (Task 2): COLLAPSE on gridgeom** (0.053/0.019/0.0081) — H-C down-weighted as predicted.
4. **E1 ladder rungs amended to {432, 1728, 3456} cts/px** (silent uint16 wrap at the planned 6912: max px scales to ~117k > 65535; `to_counts` fail-fast added in `a7b80e65`). Matching Rung 1R's 1e9 ph/img via counts would need ~61k cts/px (~14× over ceiling) — E2's amplitude twin covers that regime instead.
5. **Dose column (Task 1 verified):** row 1 = 1.77e6; row 2 = 4.4e5; row 5 = ~1e9 (effective, physics_scale-lifted); row 6 = ~4.4e5 (deadleaves N=64). Task 1 concerns logged: RECT-PROBE-SCALE-DOUBLE-DIV-001's "main has the same double division" only partially verified on main; ~55× output_scale convention discrepancy main-vs-fno noted, unresolved.

## Results (2026-07-08, E1b–E1d window map + seed grids)

| dose (cts/px) | draws | escapes | pearson values |
|---|---|---|---|
| 108 (base) | 10 gs1 (+1 gs2) | **0** | 0.046–0.128 (all) |
| 216 | 1 | 1 | 0.619 |
| 432 | 5 | 2 | 0.51, 0.834 vs 0.069–0.093 |
| 648 | 1 | 1 | 0.550 |
| 864 | 1 | 0 | 0.081 |
| 1728 | 5 | 1 | 0.877 vs 0.061–0.107 |
| 3456 | 3 | 0 | 0.071–0.11 |

**Verdicts:** outcomes strictly bimodal (collapse ≤0.13 vs escape ≥0.51 — no middle ground). H-A in its simple form REFUTED (non-monotonic; no dose trend within 432–3456). H-B REFUTED by the scale audit (chain dose-adaptive to 5 s.f.; trained s1/s2 pinned ≈1.00 everywhere; the "drifting s1/s2" that motivated the mismatch-window reading was the inference-time VarPro solve — symptom, not cause). H-C down-weighted (E3). **Standing account: stochastic basin lottery** — near-certain collapse at base dose (0/10, 95% upper bound ≈24%), ~31% escape at elevated doses (5/16, Fisher vs base p≈0.066 — dose effect suggestive, unproven), on the H-E-susceptible skip-less architecture. **Next discriminators:** (1) H-E skip restoration at base dose, n≈5 seeds — USER GATE (frozen `ptycho_torch/model.py`); (2) E2 amplitude-path cell at matched 1.77e6 dose, n≈5 seeds (twin build + SDD; if amplitude escapes ≈5/5 where rect is 0/10, the recipe's likelihood plumbing matters beyond architecture; if similar lottery, path exonerated too). Task 4/E2 and Task 5/H-E both remain open; docs consolidated in TORCH-N128-FLAT-AMP-001.

## Amendment 2 (2026-07-08, user correction: reference = TF implementation)

The reference CNN is `ptycho/model.py` (TF, paper lineage) — verified SKIP-LESS with the same N-conditional depth (N=128 → 4 plain Conv-Pool stages, no `Concatenate`). Therefore: fno-stable's torch cnn is FAITHFUL to the reference; origin/main's torch port is the deviation (added U-Net skips). H-E reframed: skips are a candidate FIX (adopting main's-port deviation), not a fidelity restoration; the basin lottery characterizes the reference architecture family under rect-count Poisson at N=128. NEW OPTION **E4**: run the TF reference pipeline at N=128 on count data (fno-stable side — TF uint16 fix present; main's TF CLI cannot ingest counts per MAINREF-CLI-001/NORMALIZE-DATA-UINT16-001) to discriminate "reference lotteries too" vs "TF dynamics differ". Feasibility unverified (TF grid_lines/nongrid workflow N=128 count ingestion) — scope before commissioning. Task 5's user gate now reads: authorize EITHER skip-adoption in `ptycho_torch/model.py` (frozen) as a fix experiment, OR E4 TF-reference characterization, or both.

## E2 result (2026-07-08)

Amplitude forward at matched base dose (same lines_N128 counts, same cnn/N/protocol, only `--physics-forward-mode amplitude` via `b74ef65a`): **0/5 escapes** (objframe pearson −0.056…0.178, canvas_std ~1e-4 — flatter than the rect collapse). **H-B REFUTED in full: the forward path does not matter.** Pooled base-dose evidence across both paths: **0/15 escapes** vs elevated-dose 5/16 → Fisher p≈0.026 — the dose effect on escape probability is now statistically established. Rung 1R's amplitude-path escape at effective ~1e9 ph/img sits on the same dose axis (residual confound: different data/pipeline). Standing etiology: skip-less depth-4 CNN (reference-faithful architecture) × count-scale Poisson likelihood, collapse basin dominant at fly001 dose, escape lottery opening with dose. E4 (TF reference, same data/dose) in flight — discriminates whether TF training dynamics share the fate.

## E4 result + FINAL VERDICT (2026-07-08)

TF reference (skip-less, same data, same 1.77e6 ph/img dose, 25ep, ~1 min/run): **3/3 structured escapes** — pearson 0.610 / 0.683 / 0.237, canvas_std 0.140/0.136/0.077; patch-check PNGs confirm aligned scoring (not flat, not shifted). Torch 0/15 vs TF 3/3 at matched data/dose: Fisher p≈0.001.

**FINAL:** H-E demoted with H-A/H-B/H-C — the reference design does not carry the failure; the collapse is a **torch-port training-dynamics defect**, dose-modulated (p≈0.026), path-independent (E2), geometry-independent (E3), scale-constants-clean (audit). Prime suspect: TF's LEARNABLE `intensity_scale` vs torch's fixed constants + never-moving rect s1/s2 (caveat flagged by the E4 executor: this plumbing difference is not yet isolated from init/optimizer differences). 9b anatomy stands for the torch losing basin. Follow-up candidate (not commissioned): make the torch scale learnable / replicate TF's normalization in a controlled A/B — touches model/config surfaces, needs its own scoped plan. Plan status: discriminating phase COMPLETE.

**RESOLUTION (same day, follow-on campaign):** the A/B was commissioned and run (`docs/plans/2026-07-08-cnn-n128-tf-parity.md`) — the scale suspect was REFUTED from both sides (torch learnable/fixed-ln2 scale arms don't rescue, δ provably unmoved; TF's own log_scale is inert in 10/10 reference runs; TF with the scale frozen doesn't collapse). The root cause is the torch port's default-on **CBAM encoder attention** (cbam-off alone: 0/15 → 3/5, p≈0.009; full TF-parity preset: 6/10 vs the TF reference's own 4/10, p=0.656 — and TF's 3/3 above proved a small-sample fluke of a ~40%-escape lottery). Authoritative record: `docs/findings.md#TORCH-N128-FLAT-AMP-001` (PARITY RESOLUTION).
