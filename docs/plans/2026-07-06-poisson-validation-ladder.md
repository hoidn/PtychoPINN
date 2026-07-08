# Poisson Validation Ladder Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development. Execute task-by-task with briefs/reports under `.superpowers/sdd/ext/` and ledger updates in `.superpowers/sdd/progress.md`.

**Goal:** Establish, rung by rung, whether generator-type models train healthily under the Poisson objective on the aligned grid-lines data — and only then give the `rectangular_scaled` ablation condition its first fair (poisson-native) test.

**Architecture:** One new dataset-conversion tool (exact inversion of the grid-lines amplitude normalization into count-convention twins), then three gated training rungs through the unmodified `grid_lines_torch_runner.py` (cnn → hybrid_resnet → hybrid_resnet+rect), then findings documentation. No protected-file edits anywhere in this plan.

**Why now (context):** The aligned ablation's `rect_only`/`both` collapse is broken-by-construction — under `torch_loss_mode='mae'`, `RectangularMAELoss` compares `I_pred²` against amplitude observations (`ptycho_torch/model.py:1716`, the documented double-square quirk; fno-stable's `MAELoss` at `:1691` is amplitude-domain, and NPZ `diffraction` is amplitude per DATA-001). The rect forward is poisson-native — but no generator architecture has a validated poisson run in the aligned regime (the only generator×poisson evidence is the superseded misaligned-regime study, `docs/findings.md` HYBRES-ALIGN-001), and the frozen aligned dataset is normalized-amplitude, which poisson cannot consume (`scripts/studies/varpro_probe_ablation_runner.py:301-307`; `ptycho_torch/dataloader.py:702-711`).

## REVISION R1 — 2026-07-06, post-Rung-1 root cause (normative; supersedes D1/D2 for the amplitude path)

The original premise — that the poisson losses cannot consume the normalized-amplitude dataset and therefore need count-convention twin datasets — is **wrong for the amplitude path**. The twin-fed Rung 1 collapse proved it:

- The amplitude-path pipeline already count-lifts internally: `ptycho_torch/workflows/components.py:100` derives `physics_scale = sqrt(nphotons / mean(sum(X²)))` (`helper.py:723-749` — the same formula as D1's S) and `model.py:2290-2291` multiplies BOTH pred and obs by it before `PoissonIntensityLayer` squares both sides. On the original normalized data physics_scale ≈ 328.7 and the squared observations are genuine counts. Poisson-on-normalized-data was count-correct all along.
- Feeding the pre-scaled `data_amp` twin degenerates the derivation (mean(sum(X²)) = nphotons by construction ⇒ physics_scale = 1.0); with `rms_scaling_constant` absent in the bridge path (defaults to 1.0, `components.py:519-528`), the bounded network head would have to emit count-scale amplitudes (~247/pixel) it cannot reach ⇒ `-k·log λ` dominance ⇒ saturate-to-constant collapse. Both recipes (lr 2e-4 and lr 1e-3/no-sched) hit the identical attractor: val loss frozen at 8.12105e7 from epoch 1, byte-identical metrics.json, constant amp≈7.97 canvas.
- This empirically confirms NORMALIZATION-001 (`ptycho_torch/dataloader.py:698-705`): NPZ `diffraction` must remain normalized amplitude; `nphotons` travels as metadata only.

Consequences, reflected in the tasks below:
1. Tasks 1–2 and the twin-fed Rung 1 runs are **retained as a documented negative control** (they prove the NORMALIZATION-001 violation mechanism empirically). The twins are not consumed by any further rung.
2. Rungs 1–2 are re-pointed at the ORIGINAL frozen aligned data (`ext_matrix_aligned/data/{train,test}.npz`). `--torch-loss-mode poisson` becomes the ONLY delta vs the MAE reference runs — same data, same recipe — making the loss-mode comparison clean and the D3 gates (vs 0.1402 / 0.0781) directly meaningful.
3. D2's rect-path row is unresolved, not validated: `rectangular_scaled` routes scale differently (`model.py:2259-2267` folds `sqrt(1/(scale²·physics_scale+1e-9))` into the forward; `RectangularPoissonLoss` squares neither side). New Task 5a (CPU, read-only) pins down that path's actual unit contract before Rung 3 commits to a dataset convention.
4. D4's smoke check is replaced: the runner logs no pred-vs-obs scale diagnostics (the twin-fed smoke "passed" on an init-vs-converged loss-ratio proxy and missed a 3–4 order-of-magnitude scale mismatch). The revised smoke computes the derived physics_scale directly from the training NPZ (numpy replica of `derive_intensity_scale_from_amplitudes`) and requires it to be O(10²) (expect ≈328.7); scale ≈1.0 is an immediate STOP.
5. Historical caveat for Task 6: the old-regime "generator×poisson collapse" evidence must be re-read — if those runs fed normalized data through the amplitude path they were already count-correct, so their failure had a different cause (recipe/lr or the rect path), which the revised rungs now isolate rather than assume.

## REVISION R2 — 2026-07-06, post-Task-5a (mechanism correction to R1; operational consequences of R1 unchanged)

Task 5a (`.superpowers/sdd/ext/task-ladder-t5a-report.md`, §1/§7 — empirically verified by toy trace of the real `_build_lightning_dataloaders` and controller-confirmed against the code) corrects R1's mechanism:

1. **The grid-lines runner bridge never populates the scale constants.** The runner builds a plain dict container (`grid_lines_torch_runner.py:1293-1299`) with no `physics_scaling_constant`/`scaling_constant`/`rms_scaling_constant`; `_attach_physics_scale` (`components.py:93`) is reachable only via `_ensure_container` (`components.py:262,322`), which the dict bypasses. `_select_scale(None)` → 1.0. So **physics_scale = rms_scale = scale = 1.0 in ALL ladder rungs** — R1's claim that "physics_scale ≈ 328.7 count-lifts internally" is false for this runner; there is no internal count lift.
2. **Corrected twin-collapse mechanism:** physics_scale was 1.0 for both original and twin data (no 328.7→1.0 degeneracy). The twin collapsed simply because count-scale observations (`(amp·S)²` ~O(10⁴)/px as intensity) are unreachable by the bounded forward at scale 1.0 ⇒ `−k·logλ` dominance ⇒ constant-output collapse. Corroborated by loss magnitudes: twin attractor 8.12e7 vs Rung-1R healthy ~1.3e4.
3. **Reframed claim for all rungs (Task 6 must use this wording):** the ladder's poisson rungs test a **normalized-units Poisson NLL** (rate=pred², obs=amp², both O(0.01–35)), NOT a genuine 1e9-count Poisson. Rungs remain valid as clean loss-mode ablations vs the MAE references (same data, same recipe, loss mode the only delta).
4. **D4 scale probe reinterpreted:** it remains a mandatory *input-convention guard* (S≈328.7 proves normalized-amplitude data; S≈1.0 detects count-twin leakage) but describes no scale that flows into training — nothing in the bridge consumes S.
5. **Rung 3 (count-level rect poisson) is DESCOPED** per Task 5a §8a: no dataset convention yields a fair count-level test through the unmodified bridge (count-intensity data forces trained/solved s1≈O(100) — the historical s1≈131 regime, now quantitatively explained as the Parseval-locked O(1) rect forward absorbing the count gap; the VarPro ~53× low s1 is the same unpinned-scale defect in the opposite direction). The minimal enabling change (populate the scale constants on the bridge container in `components.py`) alters all bridge training and needs its own plan. Task 5a §8b preserves an optional no-code apples-to-apples alternative (new normalized-intensity twin `amp²`, s1/s2 expected O(1)) — NOT scheduled under this plan; user may elect it separately. Note its information value is limited: the rect forward already collapses under MAE (HYBRES-ALIGN-001), so a rect+poisson run would not isolate the loss variable.

- FORBIDDEN edits: `ptycho/model.py`, `ptycho/diffsim.py`, `ptycho/tf_helper.py`, `ptycho/config/config.py`, `ptycho_torch/config_params.py`, `ptycho_torch/model.py`.
- Frozen inputs (read-only): `.artifacts/varpro_ablation/datasets/`, `.artifacts/varpro_ablation/ext_matrix_aligned/data/`.
- All new artifacts under `.artifacts/varpro_ablation/poisson_ladder/` — never committed. Commits: Task 1 code+test, Task 6 docs only. No "claude" in commit messages. No pushes.
- Long runs: `cmd ... & pid=$!; wait "$pid"`, per-run `.log`/`.exit` markers, one GPU job at a time, no duplicate writers to the output root.
- Training recipe = the aligned BASE recipe verbatim (`scripts/studies/aligned_hybres_ablation_driver.sh`: N=128 gs1, epochs 5, batch 16, lr 2e-4, ReduceLROnPlateau 0.5/2/1e-4/0.0, seed 3, adam 0.9/0.999 wd 0, Run1084 probe, real_imag output, central_mask weighting, fno 12/32/4/2) with ONLY `--torch-loss-mode poisson` and the dataset paths changed. One predefined fallback recipe: lr 1e-3, no scheduler (main's poisson-native). No other tuning.

## Design decisions (load-bearing — do not improvise around these)

> **R1 status:** D1 remains correct as math (the twin builder recovers S exactly). D2's *premise* — that the amplitude path needs count-convention data — is refuted by R1; its table is retained as the record of what was believed and as negative-control documentation. D3 is unchanged. D4 is amended per R1 item 4.

**D1 — Exact inversion, not arbitrary rescale.** *(superseded as a rung input per R1; retained as negative-control provenance)* The grid-lines sim draws Poisson counts at `nphotons=1e9` per pattern, then unconditionally normalizes amplitudes by `intensity_scale` (`ptycho/diffsim.py:63,121,141-142`); the scale is not saved, but is recoverable because it was defined to satisfy the photon budget: `S = sqrt(nphotons / mean_over_patterns(sum(amp²)))` — the same formula as `derive_intensity_scale_from_amplitudes` (`ptycho_torch/helper.py:723-749`). Multiplying the stored amplitudes by per-split `S` recovers the physical count-amplitudes actually drawn (noise statistics stay self-consistent), and the dataset's baked `nphotons=1e9` metadata becomes *true* for the twin. Do NOT rescale to an arbitrary target mean count (à la `make_synthetic_truth_datasets.py`'s 108): that would mislabel near-noiseless data as low-count.

**D2 — Two twins, because the two poisson losses disagree on conventions:** *(REFUTED for the amplitude row per R1 — the pipeline lifts normalized data to counts internally; the rect row is unresolved pending Task 5a)*

| forward | loss class | obs treatment | required `diffraction` convention |
|---|---|---|---|
| `amplitude` | `PoissonIntensityLayer` (model.py:1313) | squares rate AND obs | count-amplitude: `amp·S` (= √counts) |
| `rectangular_scaled` | `RectangularPoissonLoss` (model.py:1698) | squares NEITHER | count-intensity: `(amp·S)²` (= counts) |

Emit both: `poisson_ladder/data_amp/{train,test}.npz` and `poisson_ladder/data_intensity/{train,test}.npz`. Mispairing reproduces exactly the units-bug class that broke rect+MAE — the pairing above is normative for Tasks 3–5. Store float32 (peak counts ≈3e7 exceed uint16; `validate_args=False` accepts floats).

**D3 — Gates on the runner's native metrics** (`runs/pinn_<arch>/metrics.json`: mae/mse/psnr/ssim/ms_ssim/frc*): a rung PASSES if amplitude MAE ≤ 2× the same architecture's aligned-MAE reference (cnn 0.1402, hybrid_resnet 0.0781) AND the reconstruction shows no collapse signature (dot lattice / constant canvas) in `visuals/`. If the primary recipe fails, run the fallback recipe once. If both fail: the rung FAILS — stop the ladder there and report (per the 3-attempt rule; a rung failure is itself the finding).

**D4 (amended per R1) — Each rung: scale probe + 1-epoch calibration smoke before the 5-epoch run.** Poisson NLL magnitudes at 1e9 counts are enormous; `compute_loss` divides by `intensity_norm_factor = mean(observed)` (model.py:2280) but the gradient scale is untested. The runner logs NO pred-vs-obs scale diagnostics, so the smoke has two mandatory parts: (a) **scale probe** — compute the derived physics scale directly from the training NPZ with numpy (`sqrt(nphotons / mean_over_patterns(sum(diffraction²)))`, the exact formula of `helper.py:723-749`) and require it O(10²) (expect ≈328.7 on the aligned data); a value ≈1.0 means count-scale data leaked in — STOP immediately; (b) **1-epoch smoke run** — loss finite and decreasing within the epoch. Smoke artifacts go to `poisson_ladder/smoke_<rung>/`, deleted after the check.

---

### Task 1: Count-twin builder + tests (TDD, one commit)

**Files:**
- Create: `scripts/studies/make_aligned_count_twin.py`
- Test: `tests/torch/test_make_aligned_count_twin.py` (CPU-only, synthetic fixture — no GPU, no frozen-data dependency)

**Interfaces:** CLI `python scripts/studies/make_aligned_count_twin.py --src-dir <dir with train.npz,test.npz> --out-root <root>` → writes `<root>/data_amp/{train,test}.npz`, `<root>/data_intensity/{train,test}.npz`, `<root>/provenance.json` (per-split S, nphotons used, source paths+mtimes, formulas).

Behavior: per split, load NPZ; read `nphotons` from `_metadata` (fall back 1e9 with a warning); compute `S = sqrt(nphotons / mean_over_patterns(sum(diffraction²)))`; `data_amp.diffraction = diffraction·S` (float32); `data_intensity.diffraction = (diffraction·S)²` (float32); ALL other keys copied byte-identical (including `Y_I`, `Y_phi`, coords, `probeGuess`, `norm_Y_I`, `_metadata`).

- [ ] Step 1: write failing tests — construct a tiny synthetic NPZ where amplitudes were divided by a known `S_true` from known counts; assert: recovered S ≈ S_true (rtol 1e-6); `data_amp` mean total intensity per pattern ≈ nphotons; `data_intensity == data_amp²`; non-diffraction keys byte-identical; provenance records S per split.
- [ ] Step 2: run `pytest tests/torch/test_make_aligned_count_twin.py -v` → FAIL (module missing).
- [ ] Step 3: implement; re-run → PASS.
- [ ] Step 4: commit (script + test).

### Task 2: Build the twins (no commit)

- [ ] `python scripts/studies/make_aligned_count_twin.py --src-dir .artifacts/varpro_ablation/ext_matrix_aligned/data --out-root .artifacts/varpro_ablation/poisson_ladder`
- [ ] Sanity-print: S per split (expect O(10²–10³)), mean per-pattern total intensity ≈1e9, max intensity ≈3e7. Record in the task report.

### Task 3: Rung 1 — cnn + amplitude forward + poisson (the anchor)

The CNN/Autoencoder lineage is the one architecture with TF-era poisson pedigree; this rung validates data/loss/recipe with no new unknowns. Runner already supports `--architecture cnn` (commit 299eabc4) and `--torch-loss-mode poisson` (`grid_lines_torch_runner.py:2071`).

**Twin-fed attempt (R1 negative control — COMPLETE, verdict FAIL-by-design-artifact):**
- [x] Twin-fed smoke + full run + fallback executed (`rung1_cnn`, `rung1_cnn_lr1e3`); both collapsed to the identical constant-output attractor. Root cause = physics_scale degeneracy from the NORMALIZATION-001 violation, NOT a poisson-training defect. Evidence preserved in `.superpowers/sdd/ext/task-ladder-rung1-report.md` and the two run dirs (keep — they are the negative-control artifacts for Task 6).

**Revised rung (R1) — original data:**
- [ ] Scale probe + 1-epoch smoke per amended D4 (train NPZ = `ext_matrix_aligned/data/train.npz`; expect physics_scale ≈ 328.7).
- [ ] Full run: aligned BASE flags + `--architecture cnn --torch-loss-mode poisson --train-npz .artifacts/varpro_ablation/ext_matrix_aligned/data/train.npz --test-npz .artifacts/varpro_ablation/ext_matrix_aligned/data/test.npz --output-dir .artifacts/varpro_ablation/poisson_ladder/rung1_cnn_orig`.
- [ ] Gate per D3 vs cnn reference 0.1402 (metrics.json arrays are [amplitude, phase]; gate on `mae[0]`). On fail → fallback recipe once (`rung1_cnn_orig_lr1e3`). Record verdict.

### Task 4: Rung 2 — hybrid_resnet + amplitude forward + poisson (the missing validation)

- [ ] Same as the revised Task 3 with `--architecture hybrid_resnet`, original-data paths, output `rung2_hybres_orig`; gate vs 0.0781 (`mae[0]` ≤ 0.1562).
- [ ] Interpretation is part of the deliverable: PASS ⇒ the old misaligned-regime degeneracy is attributable to data/lr, not the poisson objective; FAIL (both recipes) ⇒ poisson×generator is a real defect — STOP, do not run Task 5, report as the primary finding.

### Task 5a: Rect-path unit-contract analysis (CPU, read-only — gates Rung 3)

R1 established that the amplitude path lifts normalized data internally; whether the `rectangular_scaled` path expects normalized amplitudes, count-amplitudes, or count-intensities is NOT settled by D2's table (which was premised on a wrong model of the amplitude path).

- [ ] Trace, in code, exactly what `RectangularPoissonLoss` compares at train time under this runner's bridge path: what `pred` is after the B5 folded `output_scale = sqrt(1/(scale²·physics_scale+1e-9))` (`model.py:2259-2267`), what `observed_images` holds, what units each side is in for (a) original normalized data and (b) each twin, and what role trainable s1/s2 play in bridging any residual gap (including whether the old s1≈131 drift is explained).
- [ ] Deliverable: a written unit-contract table + a concrete Rung 3 specification (dataset choice, expected s1/s2 magnitudes, expected physics_scale, smoke criteria) OR a finding that the rect path has a genuine units defect making a fair poisson test impossible without protected-file edits (⇒ Rung 3 is descoped and that is the finding).
- [ ] No commits; report only. This is analysis, not implementation.

### Task 5: Rung 3 — hybrid_resnet + rectangular_scaled + poisson (the fair rect test)

Only if Rung 2 passed AND Task 5a produced a concrete specification. Dataset convention per Task 5a (do NOT assume `data_intensity` — that was D2's unverified premise).

- [ ] Scale probe + 1-epoch smoke per amended D4 and Task 5a's criteria, additionally checking trained-s1/s2 drift stays in the range Task 5a predicts (the old-regime signature was s1≈131 on a collapsed canvas).
- [ ] Full run: aligned BASE + `--architecture hybrid_resnet --torch-loss-mode poisson --physics-forward-mode rectangular_scaled`, dataset per Task 5a, output `rung3_rect`.
- [ ] Gate: no collapse signature; compare against Rung 2's numbers (rect vs amplitude forward under the same objective — the ablation comparison that was impossible under MAE). No fixed MAE threshold here; the deliverable is the comparison, gated only on "not collapsed".

### Task 6: Documentation (one commit)

- [ ] `docs/findings.md`: (a) new entry (suggest `RECT-MAE-UNITS-001`) recording the rect+MAE double-square mechanism — objective `|I_pred² − A_meas|`, fixed point ⇒ spectral flattening ⇒ dot-lattice collapse, s1/s2 stayed ≈1 (checkpoint evidence), code cites model.py:1691/1716/2283, and the corrected convention analysis (D2 table as-believed + R1 refutation + Task 5a's contract); (b) new entry (suggest `POISSON-NORM-001`) for the R1 negative control: feeding count-scale diffraction violates NORMALIZATION-001 and provably collapses amplitude-path poisson training (evidence: `rung1_cnn`/`rung1_cnn_lr1e3` identical attractor). Mechanism per REVISION R2: in the grid-lines bridge (scale constants never populated; physics_scale=1.0) count-scale observations are unreachable by the bounded forward ⇒ `−k·logλ` dominance ⇒ constant-output collapse; on the native container path the same data would additionally degenerate `derive_intensity_scale_from_amplitudes` to 1.0. NORMALIZATION-001 is load-bearing, not advisory; (c) new entry (suggest `POISSON-LADDER-001`) with the revised rung verdicts, including the historical re-read caveat (R1 item 5: prior "generator×poisson collapse" evidence does not establish a poisson-objective defect).
- [ ] `docs/plans/2026-07-06-aligned-ablation-20epoch-rerun.md`: replace H1 (rect@20ep under MAE has a known outcome — the objective is broken by construction; point to RECT-MAE-UNITS-001) and drop `rect_only`/`both` from its arm list unless this ladder motivates a poisson-mode 20ep arm.
- [ ] `docs/development/TEST_SUITE_INDEX.md`: row for `tests/torch/test_make_aligned_count_twin.py`.
- [ ] Commit docs.

## Out of scope

- The principled forward→loss unit-contract refactor in `ptycho_torch/model.py` (protected; needs its own authorized plan; this ladder sidesteps it entirely).
- Making rect+MAE trainable (unnatural objective; superseded by the poisson-native test).
- Inference-variant grids / montage rows for ladder arms (native runner metrics + visuals suffice for validation verdicts; bridging is a separate concern per REASSEMBLY-BRIDGE-001).
- Multi-seed replication; 20-epoch versions of the rungs.

## Risks / notes

- Poisson NLL at ~1e9 counts is near-noiseless — the NLL is then ≈ a χ²-like intensity fit; that is the regime the aligned MAE recipe also trained in, so it is the right first validation point, but a PASS here does not certify low-count robustness.
- ~~`derive_intensity_scale_from_amplitudes` calibration on the twins is checked empirically by the D4 smokes~~ R1 resolved this: the calibration is correct on normalized data (S≈328.7) and degenerate on count data (1.0). The amended D4 scale probe checks it explicitly per rung.
- ~~The dataloader's "MUST be normalized" comment is advisory~~ R1 proved it load-bearing: count-scale inputs pass through without an assertion and collapse training. (A defensive assertion in the loader would be a separate, protected-file-adjacent change — out of scope here; noted for POISSON-NORM-001.)
- Empirical run cost (measured): a 5-epoch cnn run ≈ 3–4 min GPU; revised remaining ladder (2 rungs × probe+smoke+full, plus fallbacks and a possible Rung 3) well under 1 h. GPU is shared with the gs2 initiative via ledger coordination — poll before every launch.

---

## PHASE 2 — 2026-07-06, post-RCA: absolute-scale restoration (count-lift). AUTHORIZED.

**Phase 2 status (2026-07-06): CLOSED.** Code commits `4cf6d074`+`3305c352`+`8a985df7`, docs commit `357fecd8`. Outcome: units-contract fix landed opt-in (`--count-scale-mode auto`), default `off` after A/B falsification (POISSON-SCALE-001).

**Goal:** fix the missing absolute scale in the grid-lines dict-container path so the Poisson NLL operates on genuine photon counts (λ and k at nphotons=1e9 scale) instead of O(1) normalized units, with a regression gate proving the fix does not change training outcomes.

**Why (established this session, post-RCA):** the dict container built at `grid_lines_torch_runner.py:1293-1299` bypasses `_attach_physics_scale` (`components.py:262,322`), so `physics_scale=1.0` everywhere. Provably this did NOT cause the Rung 2 failure — the amplitude-path NLL satisfies an exact affine identity under a both-sides lift (NLL_c(θ)=c·NLL(θ)+const, c=S²), and all four scale-sensitivity conditions were verified benign for these runs (Adam `--optimizer adam`, single-term objective, fp32 [no precision flag → Lightning 32-true], grad-clip 0.0/off, weight_decay 0.0; measured per-tensor grad RMS min 1.2e-5 = 10³× above Adam ε). But the units contract is broken: the likelihood is ~S≈330× under-confident vs the data's actual 1e9-photon statistics, NLL magnitudes are uninterpretable, and any future composite objective (hybrid poisson+MAE, regularizers, wd>0) or optimizer/precision change turns the 1e5 scale error into a live first-order bug. The rect path is not even scale-equivariant (Task 5a s1≈131 pathology).

**Verified constants:** S_train=328.696, S_test=329.199 (`S=sqrt(1e9/mean(sum(diff²)))`, scale-probe convention from Phase 1). Expected logged loss after lift (CORRECTED by Task 7, measured live): ≈ current × ~1.2–1.4e4 ≈ O(1.5e8) near convergence — NOT ×S²≈1.1e5; S² scales the θ-dependent part/gradient (which is what guarantees mae-invariance), while the logged value includes Poisson normalization constants that do not scale by S². `intensity_norm_factor` (model.py:2279) is computed on UNSCALED obs and stays unchanged.

### Global constraints (Phase 2)
- Protected, do NOT edit: `ptycho/model.py`, `ptycho/diffsim.py`, `ptycho/tf_helper.py`, `ptycho/config/config.py`, `ptycho_torch/config_params.py`, `ptycho_torch/model.py`.
- Concurrent-session no-touch (gs2 initiative actively works here): `ptycho_torch/helper.py`, `ptycho_torch/data_container_bridge.py`, `ptycho_torch/patch_generator.py`. If the fix seems to require editing these, STOP and report BLOCKED.
- CRITICAL correctness invariant: the lift must multiply BOTH sides of the loss identically (physics_scale path, model.py:2290-2291). Setting `rms_scaling_constant` ≠ 1.0 changes the FORWARD (input/output scaling, model.py:2272-2274) and is NOT covered by the invariance argument — rms scale must remain exactly as today unless Task 7 proves the native convention requires otherwise AND the A/B gate stays green. A lift applied to one side only reproduces the POISSON-NORM-001 collapse.
- Out of scope for Phase 2: eps floor and hybrid poisson+MAE objective (protected model.py — next plan); TF backend; rect-path scale handling; changing the ProbeIllumination collation behavior (investigate/characterize only unless the fix is trivially localized outside protected and no-touch files).
- Frozen evidence: all existing `.artifacts/varpro_ablation/poisson_ladder/*` and `ext_matrix_aligned/*` run dirs. New runs go under `.artifacts/varpro_ablation/poisson_ladder/countscale/`.
- Commits: no "claude" anywhere in messages, no trailers, never `--no-verify`, no pushes. GPU: one job at a time, poll `nvidia-smi` free-twice before launch (gs2 shares the GPU).

### Task 7 — Scale-convention + collation investigation (CPU, read-only) ✅ gate for Task 8 — COMPLETE 2026-07-06
Report: `.superpowers/sdd/ext/task-p2t7-report.md`. Verdicts: physics_scale=S per split via `helper.derive_intensity_scale_from_amplitudes` (native convention exactly); seam = `run_torch_training` dict build (runner :1299/:1329) + new testable `components.derive_dict_physics_scale()` helper — existing `_get_tensor`/`_select_scale` wiring already threads a dict key to `compute_loss`, no no-touch/protected edits; rms DECOUPLED (forward reads only rms; physics enters only the loss-time both-sides multiply — verified bit-identical forward on real checkpoints); collation: S-lift alone sufficient (obs is B-independent, loss pins pred→obs ⇒ λ→counts for any B); effective-B RESOLVED = 16 (RCA-B "B=8" was a post-hoc-reconstruction artifact, live-vs-reconstruction ~4× gap — Task 9 must use LIVE logged losses only); collation fix would live in `components.py::__getitem__` :570-589 but needs full retune (82da7796 precedent) — stays investigate-only. Loss-magnitude expectation corrected (see Verified constants). Original deliverables spec:
No repo edits. Deliverables, all with file:line citations and toy-tensor verification where feasible:
1. The native `_attach_physics_scale` convention: exact formula, what `physics_scaling_constant` and `rms_scaling_constant` each mean downstream (trace into `compute_loss` model.py:2233-2304 and the forward call 2272-2276), per-split behavior, units. Decide what value the dict path must attach so obs_physics² = genuine counts (expected: physics_scale=S per the probe convention; rms untouched) — the dict-path convention MUST match the native path's semantics, not invent a new one.
2. The exact seam: where in the dict-container flow (`run_torch_training` → `_build_lightning_dataloaders` → batch `(td, pr, sc)` `_select_scale`) the constants can be attached WITHOUT touching no-touch files. Name the function(s) and the minimal diff shape.
3. ProbeIllumination collation (Track B caveat 4a, `task-rung2-rca-b-report.md` §4a): confirm the probe reaches the forward as (B,H,W) in this path, that `pad_and_diffract` mode-sums it (helper.py:603-624, read-only), the pred∝B scaling, and resolve or bound the effective-B=8-vs-16 question. Verdict: does the S-lift alone deliver λ=true counts at convergence, or off by a B-dependent factor? Propose (do not implement) the localization of a collation fix and whether it lands outside protected/no-touch files.
4. Output: an exact implementation spec for Task 8 (attachment point, formula, flag design, test cases) + expected post-lift loss magnitudes for the A/B gate.

### Task 8 — Implement scale attachment (TDD; `ptycho_torch/workflows/components.py` + `scripts/studies/grid_lines_torch_runner.py`) — COMPLETE (commits `4cf6d074`, `3305c352`, `8a985df7`)
Per Task 7's spec. Shape (Task 7 may refine details, not the gates):
- New runner flag `--count-scale-mode {auto,off}`, default `auto`: derive S from the training data via the probe convention (nphotons from config, default 1e9) and attach `physics_scaling_constant=S` (and whatever the native convention pairs with it, per Task 7) to the dict container / batch scale tensor. `off` preserves today's behavior (1.0) exactly. Record the derived S and the mode in the run's config.json/invocation provenance.
- TDD: write failing tests first in `tests/torch/test_dict_container_physics_scale.py` (tiny synthetic npz; CPU): (a) auto-mode container/batch exposes physics_scale=S per formula (rtol 1e-5, shape (b,1,1,1) after collation); (b) off-mode yields exactly 1.0; (c) both-sides-lift identity: compute_loss with physics=S equals the exact hand-computed `mean(−Independent(Poisson((pred·S)²),3).log_prob((obs·S)²))/mean(obs)` on a fixed toy batch. Do NOT assert L(auto)/L(off)==S² — that ratio is operating-point dependent (Task 7 §4b); if a ratio assertion is wanted, construct pred=c·obs with c large so the rate term dominates and assert ratio→S² within a few %. Run, confirm fail, implement minimal, confirm pass.
- Do not modify the loss classes or `ptycho_torch/model.py`. One commit (tests + implementation); report `pytest` evidence.

### Task 9 — A/B invariance rung (GPU, ~15 min total)
Exact Phase-1 recipes, original aligned data, `--count-scale-mode auto`, outputs under `.../poisson_ladder/countscale/`:
- cnn+poisson 5ep seed 3 → gate: |mae[0] − 0.1662| ≤ 0.03 (seed-level tolerance; C1 seed spread was 0.007).
- hybres+poisson 5ep seed 3 → gate: |mae[0] − 0.2839| ≤ 0.03.
- Lift-active proof (per run; CORRECTED by Task 7 §4b — the original "within ~2× of Phase-1 × S² ≈ 1.4e9" gate would false-fail a correct implementation): (1) PRIMARY: the run's recorded provenance `physics_scale` ≈ 328.7 (S_train) and `count_scale_mode=auto`; (2) SECONDARY sanity: live logged train loss ≥ ~100× the off-mode magnitude (expect ~1.2–1.4e4×, i.e. O(1.5e8) vs off ~1.27e4). A run passing the mae gate with loss still O(1e4) = probe failure, STOP. Use LIVE trainer-logged losses only — never post-hoc `compute_loss` reconstruction (known ~4× gap, Task 7 §3b).
- Both gates green ⇒ invariance confirmed, default `auto` stands. Either gate red ⇒ STOP: flip default to `off` in a fix commit, mark Phase 2 blocked, report (do not iterate on recipes).

**OUTCOME (2026-07-06): RED — invariance falsified in practice; default flipped to `off`.** cnn+poisson 5ep seed3 lifted: mae[0]=0.21154 vs gate 0.1662±0.03 (lift provably active: provenance S_train=328.6956, live loss 6.7876e7 ≈ 5358× Phase-1). hybres not launched (red-gate stop). Controls round (same recipe, same HEAD): `off` reproduces the frozen reference to 1.6e-5 (mae[0]=0.16622, loss 12667.08 — no code drift, gs1 metrics path intact) and a repeat `auto` run is byte-identical to the first (deterministic — no GPU nondeterminism). The discrepancy is therefore a genuine, deterministic effect of the lift. Mechanism (hypothesis, hedged): gradients scale exactly ×S², but Adam's ε=1e-8 (non-negligible only against the unlifted run's smallest per-tensor grad RMS ~1e-5) plus fp32 rounding inject O(0.1%) per-step update differences that ~2800 steps of trajectory chaos amplify into a different basin of the H1-underdetermined objective — outcome sensitivity to numerically-trivial perturbation is itself further evidence of objective flatness (POISSON-LADDER-001). The count-lift stays available OPT-IN (`--count-scale-mode auto`, units contract correct and tested); it is not outcome-preserving and MUST NOT be combined with cross-run comparisons against unlifted references. Reports: `.superpowers/sdd/ext/task-p2t9-report.md` (incl. Controls round). Runs: `.artifacts/.../countscale/{rung1_cnn,rung1_cnn_off,rung1_cnn_auto2}/`.

### Task 10 — Docs + close-out (ONE commit; framing updated for the RED Task 9 outcome) — COMPLETE (commit `357fecd8`)
- `docs/findings.md`: (a) new entry POISSON-SCALE-001 recording the missing-scale defect, the opt-in fix (`--count-scale-mode`, default `off` after the A/B), the affine-invariance argument + verified conditions, AND the decisive A/B falsification (deterministic basin shift under the lift; off-control byte-reproduces the reference; hedged ε/fp32-chaos mechanism hypothesis; practical rule: lifted and unlifted runs are not comparable and the lift is not outcome-preserving); (b) POISSON-LADDER-001: add the dose-regime caveat (likelihood was ~330× broader than the data's actual 1e9-photon statistics; MAE-beats-poisson is a high-dose-regime result that could flip at genuinely low dose) and a pointer to POISSON-SCALE-001; (c) extend the ProbeIllumination open item with Task 7's findings: mechanism confirmed (probe collated (B,H,W) → P-axis collision → pred∝B, toy-exact), effective-B resolved = 16 (the RCA-B "B=8" was a post-hoc-reconstruction artifact; live-vs-reconstruction ~4× gap), S-lift unaffected, collation fix localized to `components.py::__getitem__` :570-589 but requires retune (82da7796 precedent) — still open.
- `docs/development/TEST_SUITE_INDEX.md`: row for the new test file.
- This plan: mark Phase 2 tasks complete.
