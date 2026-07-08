# 20-Epoch Aligned Grid-Lines Ablation Rerun

**Goal:** Rerun the aligned hybrid_resnet (+ cnn baseline) ablation with `--epochs 20` (vs the integration recipe's 5) to test which 5-epoch conclusions are training-length artifacts and which are stable properties of the knobs.

**Baseline being extended:** `docs/findings.md` HYBRES-ALIGN-001 + REASSEMBLY-BRIDGE-001; driver `scripts/studies/aligned_hybres_ablation_driver.sh`; 5-epoch artifacts at `.artifacts/varpro_ablation/ext_matrix_aligned/` (montage + `variants_summary.md`). Runner cnn support landed in `299eabc4`.

## Hypotheses under test (5-epoch result → 20-epoch question)

| # | 5-epoch result | 20-epoch question |
|---|---|---|
| H1 | `rectangular_scaled`+MAE collapses (amp MAE ~11–21, corr ~0.1) | **Dropped — known outcome, not tested at 20ep.** The rect+MAE objective is broken by construction: `RectangularMAELoss` double-squares (`abs(I_pred^2 − A_meas)`, squared-intensity vs amplitude), so the collapse is structural, not an optimization slow-start (`docs/findings.md` RECT-MAE-UNITS-001). `rect_only`/`both` are removed from the arm list; no poisson-mode 20ep rect arm (out of scope). |
| H2 | Training weighting inert at gs1 (`weight_only` ≡ `neither`, bit-exact) | Must remain bit-exact (structural: `object_big=False` guard) — a divergence would indicate nondeterminism, not a real effect |
| H3 | Inference probe weighting > uniform (MAE 0.100 vs 0.140) | Does the ordering persist as the model improves, and does the margin shrink? |
| H4 | `amp_phase` output mildly worse (0.194 vs 0.140) + grid texture | Parameterization handicap, or just slower convergence? |
| H5 | cnn baseline worse than hybrid_resnet (corr 0.952 vs 0.974) | Architecture gap, or does cnn catch up with more steps? |
| H6 | all arms ran `torch_mae_pred_l2_match_target=off` (legacy lines128 paper contract, inherited from the integration test); the forward-looking baseline (`docs/model_baselines.md`) says `on` | New arm `l2match_on` (baseline knobs + `--torch-mae-pred-l2-match-target`): does the flag change the aligned-regime conclusions, esp. amplitude-scale calibration? |

## Invariants (do not deviate)

- Everything except `--epochs` is copied verbatim from the 5-epoch recipe: seed 3, batch 16, lr 2e-4 + ReduceLROnPlateau(0.5/patience 2/min_lr 1e-4/threshold 0.0), MAE PINN objective, N=128 gs1, Run1084 probe, `real_imag` output (except the `ampphase_out` arm), fno 12/32/4/2.
- **Arm list drops `rect_only` and `both`.** rect+MAE at 20ep has a known outcome — the objective is broken by construction (`RectangularMAELoss` double-square, `docs/findings.md` RECT-MAE-UNITS-001), so those two arms are neither trained nor evaluated, and no poisson-mode 20ep rect arm is added (out of scope). Trained arms (Task 3): `neither`, `weight_only`, `ampphase_out`, `cnn`, `l2match_on` (5). Inference-variant grid (Task 4): `neither`, `weight_only`, `ampphase_out`, `cnn` (4).
- **Contract lineage:** the recipe sits on the frozen legacy lines128 paper-benchmark contract (`torch_mae_pred_l2_match_target=off`, `probe_scale_mode=pad_extrapolate`, `seed=3` — the contract's canonical seed per the `...-legacy-best-e40-seed3` study-index entry), NOT the forward-looking baseline in `docs/model_baselines.md` (`l2_match on`; `pad_preserve` for newer lines_256 data). Note the sweep runbooks' promotion bar requires seed-rerank across `{3,11,17}`; this plan's single-seed-3 results stay below that bar by design (multi-seed is out of scope). The `l2match_on` arm (H6) is the single sanctioned crossing of that boundary; see the lineage note in `docs/findings.md` HYBRES-ALIGN-001.
- **Same frozen dataset** — hardlink from `.artifacts/varpro_ablation/ext_matrix_aligned/data/`; never regenerate.
- **New output root** `.artifacts/varpro_ablation/ext_matrix_aligned_20ep/` — the 5-epoch artifacts back committed findings and must not be touched.
- Note: ReduceLROnPlateau engages differently over 20 epochs than 5 — that is part of the recipe, not a confound to control away.
- Evaluation methodology identical to the 5-epoch grid: final `model.pt` weights, barycentric reassembly with coords-transpose bridge, 20px border crop, norm_Y_I=4.6975 bridge on `*_novarpro` variants only (REASSEMBLY-BRIDGE-001). `*_varpro` rows remain scale-unreliable until the VarPro units backlog is resolved; report them but draw no scale conclusions.
- Artifacts stay out of git. Only the two small script edits (Task 1) are committed. No "claude" in commit messages; no pushes.

## Amendment (2026-07-08) — re-baselined comparison anchor + execution hold

The evaluation pipeline moved under this plan after the 5-epoch baselines were recorded. Per `docs/plans/2026-07-07-mainref-followups.md` Task 8 (Step 1 executed 2026-07-08):

1. **Comparison anchor replaced.** All 20-epoch comparisons must use the re-baselined 5-epoch numbers at `.artifacts/varpro_ablation/ext_matrix_aligned_rebase/` (inference-only replay of the SAME 5-epoch checkpoints through the current pipeline), NOT the original `ext_matrix_aligned/` summaries. Pipeline commits between the two and their measured effects (full 24-row delta table in `.superpowers/sdd/ext/followups-task8-step1-report.md`): `d5f40106` — canvas grew 232→264 px but the +32 px is pure zero margin; the true dead-zero ring SHRANK ~20→16 px (previously-discarded boundary patches now included); the harness keeps the 20 px crop for comparability (slightly conservative, over-crops ~4 px of real content), which fully accounts for the small ≤3% `*_novarpro` amp_mae/amp_corr/phase_mae deltas and ~1–8% `canvas_amp_std` deltas. `22d77509` (probe-weighted merge probe layouts — no measurable effect on these gs1 arms). `d755b2ae` — VarPro solve crop-band fix is UNCONDITIONAL (all modes): `*_varpro` amp_mae drops 13–27% and s1/s2 move materially on every arm (e.g. ampphase_out/uniform_varpro s2 0.0155→−1.6253); `497d1f69` additionally applies the count-units fold on rectangular_scaled arms only. `229a0c5b` (COM derivation — verified numerically inert). *(Correction 2026-07-08: an earlier version of this amendment claimed a 3–5× `canvas_amp_std` drop from a "partial-coverage ramp" and relaxed H2's bit-exact bar to ≤1e-6 — both claims were artifacts of a `norm_Y_I` omission in the first re-baseline glue script, caught by a corrected rerun that cross-checked against the original grid-production script. They are retracted.)*
2. **H2's "bit-exact" bar STANDS.** The corrected re-baseline reproduces `weight_only` ≡ `neither` and `both` ≡ `rect_only` bit-identically in BOTH the old and new grids. A 20-epoch divergence would indicate a real effect (or nondeterminism regression), exactly as originally written.
3. **VarPro caveat retained, restated.** `*_varpro` rows remain scale-unreliable for absolute conclusions (these arms run `physics_forward_mode=amplitude`; the 2026-07-07 count-units fold is gated to rectangular_scaled — `497d1f69`). Their s1/s2 values did shift vs the original artifacts because the solve-basis fixes apply to the shared solve path; still report-only, no scale conclusions.
4. **Execution HELD (user adjudication 2026-07-08, Option R2).** Steps beyond the re-baseline (the ~4 h 20-epoch training) are deferred behind the N=128 recipe-collapse RCA (`docs/plans/2026-07-07-mainref-followups.md` Task 9). Context: the CNN count/poisson recipe flat-collapses at N=128 (`TORCH-N128-FLAT-AMP-001`), while this plan's cnn arm at N=128 under the MAE/amplitude recipe reconstructs (5 ep corr 0.952) — the collapse is recipe-conditional, which the RCA must resolve before 20-epoch conclusions are built on this substrate. Escalate to DROP if the RCA shows the collapse reaches these arms; revisit after the RCA otherwise.

## Tasks

### Task 1 — Parameterize the driver (small commit)

`scripts/studies/aligned_hybres_ablation_driver.sh`:
1. Replace the hard-coded `--epochs 5` with `--epochs "${ALIGNED_ABLATION_EPOCHS:-5}"` (default preserves the committed 5-epoch contract and the pytest that documents it).
2. Promote the commented cnn invocation (added in `299eabc4`) to a real `run_arm cnn --output-mode real_imag --training-patch-weighting central_mask --physics-forward-mode amplitude` with `--architecture cnn` (per-arm architecture override; smallest mechanism that works, e.g. an `ARCH_OVERRIDE` local in `run_arm`).
2b. Add a `run_arm l2match_on --output-mode real_imag --training-patch-weighting central_mask --physics-forward-mode amplitude --torch-mae-pred-l2-match-target` arm (H6 — baseline knobs plus the forward-looking loss-path flag).
2c. Remove (or comment out) the `rect_only` and `both` `run_arm` invocations — dropped per the Invariants (rect+MAE broken by construction, `docs/findings.md` RECT-MAE-UNITS-001). Final arm list: `neither`, `weight_only`, `ampphase_out`, `cnn`, `l2match_on`.
3. Smoke: `ALIGNED_ABLATION_EPOCHS=1 ALIGNED_ABLATION_ROOT=tmp/smoke_20ep bash scripts/studies/aligned_hybres_ablation_driver.sh` on one arm (comment others or add an arm filter env), confirm invocation.json shows epochs=1, then delete `tmp/smoke_20ep`.
4. Commit the driver edit only.

### Task 2 — Stage the new root

```bash
ROOT=.artifacts/varpro_ablation/ext_matrix_aligned_20ep
mkdir -p "$ROOT/data"
ln .artifacts/varpro_ablation/ext_matrix_aligned/data/train.npz "$ROOT/data/train.npz"
ln .artifacts/varpro_ablation/ext_matrix_aligned/data/test.npz  "$ROOT/data/test.npz"
```

### Task 3 — Train all 5 arms

```bash
ALIGNED_ABLATION_EPOCHS=20 ALIGNED_ABLATION_ROOT=.artifacts/varpro_ablation/ext_matrix_aligned_20ep \
  bash scripts/studies/aligned_hybres_ablation_driver.sh
```
- Sequential, one GPU job at a time; the driver already tracks exact PIDs and writes `<arm>.exit` resume markers.
- Budget ~30–40 min/arm ⇒ ~3.5–4.5 h total. Run under tmux (ptycho311 env).
- Gate per arm: exit code 0 AND fresh `runs/pinn_<arch>/metrics.json` + `visuals/`. Record native amp/phase MAE per arm as they land.

### Task 4 — Inference-variant grid (2×2 × 4 arms)

The 5-epoch grid was produced by session-scratch harnesses (`run_full_grid.py`, `aligned_inference_variants.py`, `xy_swap_experiment.py::build_bridge_npz`, plus the cnn adaptation under `scratchpad/cnn_arm/`). `/tmp` scratch is volatile — **first step of this task: port the harness into the repo** as `scripts/studies/aligned_ablation_variant_grid.py` (parameterized by root + arm list + run-subdir name, encoding the bridge/crop/norm methodology above), with a CPU-cheap unit test of the methodology-critical pieces (coords bridge orientation, border crop, norm_Y_I gating). If the scratch files still exist, port from them; otherwise reconstruct from `.superpowers/sdd/ext/task-plateau-oracle-report.md` + REASSEMBLY-BRIDGE-001.

Then run it against the 20ep root. Output contract per arm (identical to 5-epoch): `variants/<variant>/{metrics.json,canvas.npz}` + `variants_summary.json`, and a root-level `variants_summary.md` with the same methodology paragraph.

### Task 5 — Montage

Port the session-scratch renderer (`scratchpad/montage/render_montage.py`) alongside the grid script (or fold into it) so the montage is reproducible; render `montage/montage_amp.png` + `montage_phase.png` for the 20ep root (arms auto-detected; error scale from healthy arms only, collapsed arms clip).

### Task 6 — Analysis + docs

1. Build one comparison table: arm × variant × {5ep, 20ep} amp_mae/amp_corr/phase_mae (+ native MAE per arm), answering H2–H5 explicitly (H1 dropped — see Invariants).
2. H2 check is exact equality of `weight_only` vs `neither` metrics files, as before.
3. Append a "20-epoch rerun" subsection to `docs/plans/2026-07-01-varpro-ablation-phase1-findings.md` with the table and verdicts; if any 5-epoch conclusion flips (esp. H1 or H3), update HYBRES-ALIGN-001 in `docs/findings.md` rather than leaving a contradiction.
4. Commit docs (+ Task 4/5 scripts if not already committed).

## Out of scope

- Multi-seed replication (all numbers remain single-seed, seed 3).
- gs2 (unvalidated for hybrid_resnet), VarPro units-gap fix, border-ring proper fix (`InferenceConfig.window`) — separate backlog items.
- Changing the committed 5-epoch pytest (`tests/torch/test_grid_lines_hybrid_resnet_aligned_ablation.py`) — it pins the integration-aligned 5-epoch regime and stays as-is.

## Risks / notes

- **Best-vs-final checkpoint:** at 20 epochs, final weights may be past the val optimum (Lightning also saves a best-`mae_val` checkpoint). Methodology stays final-`model.pt` for comparability with the 5-epoch grid; if final diverges notably from best (check `mae_val` trajectory in mlruns), note it in the analysis rather than switching mid-study.
- **cnn amplitude-scale caveat carries over:** cnn's bridged MAE inflated ~3.2× vs native at 5 epochs (hybrid_resnet ~1.8×); prefer amp_corr for H5 verdicts.
- **LR schedule should be widened:** the current `ReduceLROnPlateau(0.5/patience 2/min_lr 1e-4)` permits at most one halving before hitting `min_lr` — it never fires in a 5-epoch run and fired exactly once (~epoch 15) in the POISSON-LADDER-001 RCA's 20-epoch control (`docs/findings.md`).

## Execution

Subagent-driven (superpowers:subagent-driven-development): one implementer per task with brief/report files under `.superpowers/sdd/ext/`, task review on committed code (Tasks 1, 4, 5), ledger updates in `.superpowers/sdd/progress.md`. Tasks 3–5 are long-running: tracked PIDs, per-arm resume markers, no duplicate writers to the output root.
