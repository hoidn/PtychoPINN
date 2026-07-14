# Probe-Rank Physics Contract Fix — Design

**Date:** 2026-07-12
**Status:** Approved. Tasks 25--28 are `complete_final`. Task 28's canonical
rung1a under unit `dictionary_parity` returned PASS at amp/phase SSIM
`0.8913340876617375`/`0.9632217816205675`, with absolute deltas
`0.0054688232603687`/`0.0013551856818027` inside the locked `0.02`/`0.01`
gates. Its root is
`.artifacts/bridge_ladder/task28_gain16_seed3_unit_scaling_rerun`; rung
evidence/report SHA-256 values are
`a8741f93511cc90eb26777b1ff2cab0235ce812cff3661bad168632fce9ce711` and
`2f95c0e4651a2a48449797f4a34aed557ce70962c8e82dced0d40b4aa95a0cf6`.
Task 22 completed as diagnostic evidence; its CNN arms collapsed, so Task 30
now blocks Task 23 while Task 24 remains dependency-pending. The adversarial review
of root-cause
commit `2a9ee2ad9` completed 2026-07-12 with
both verdicts APPROVED and **every link of the root-cause chain independently
CONFIRMED** (emission ranks read at cited lines; broadcast executed directly —
amplitude gain exactly B, ratio 4.000000 at B=4; cross-sample probe coupling
demonstrated by perturbation; reshape isolation reproduced from scratch from
the sealed rung0 checkpoint: 0.936635 → 0.073175; rung1f evidence recomputed
exact).
**Authority:** User decision 2026-07-12 ("Fix physics contract now") on the
fix fork presented after root-cause confirmation; parent initiative
`docs/superpowers/plans/2026-07-09-ci-model-compatibility-ablation.md`
(Tasks 25–28 added by the same revision that introduces this document).
**Supersedes nothing.** This design adds a physics-contract phase; it does not
alter the initiative's evaluation contracts (bridge schema v3, evidence v2,
three-class adjudication).

---

## 1. Problem statement and evidence

Historical Task 21 evidence, before the Task 25 migration, found two probe
tensor layouts whose difference silently multiplied the predicted diffraction
field by the batch size:

- The then-current dictionary/inline dataset path emitted the probe **flat**: `probe =
  self.probe`, collated by the DataLoader to `(B, H, W)`
  (`ptycho_torch/workflows/components.py:995-997, :1050`, comment "the
  amplitude default must keep the pre-82da7796 raw convention").
- The mmap/`PtychoDataset` path emits the **documented** layout
  `(B, 1, 1, H, W)` (`ptycho_torch/dataloader.py:1554-1567`).
- `ProbeIllumination.forward` computes `x.unsqueeze(2) * probe`
  (`ptycho_torch/model.py` — `x_reshaped: (B, C, 1, H, W)`). A flat
  `(B, H, W)` probe right-align-broadcasts so the batch axis lands in the
  **mode (P) slot**, producing `(B, C, B, H, W)`.
- `pad_and_diffract` coherently sums the mode axis
  (`ptycho_torch/helper.py:670`, `torch.sum(input, dim=2)`), so B identical
  per-sample fields add: the predicted field — and therefore the predicted
  amplitude entering the MAE loss — is multiplied by **B** (16 on full
  batches, 2 on the final partial batch of the reference recipe).

Task 25 subsequently migrated the dictionary producer to the documented
layout and made the flat rank fail with `ProbeLayoutError`. The mechanism below
is historical evidence only; it is not a description of current dictionary
emission.

Consequences, all empirically established in Task 21 (ladder + step-parity
diagnostics, `.superpowers/sdd/task-21a-report.md`):

- The qualified Hybrid ResNet reference (amp SSIM 0.8984 / phase 0.9624)
  trains under an **accidental, batch-size-dependent physics gain**; the CI
  mmap path follows the documented contract (gain ×1) and reaches only
  0.4856/0.8383.
- Reshaping only the probe at fixed weights moves the training loss 12×
  (0.9361 → 0.0755); loss and gradients are otherwise bit-identical across
  flows on identical batches.
- One-variable GPU confirmation `rung1f_probe_layout` (flat layout injected
  into the otherwise most-aligned mmap configuration): amp SSIM
  0.4856 → **0.8959**, phase 0.8383 → **0.9587** — 99.7% recovery of the
  reference amplitude quality from this single variable.
- Everything else is exonerated with mechanism-level evidence: data bytes,
  batch composition and order, normalization constants, validation inputs,
  evaluation/stitching path (2×2 cross-eval matrix), initial weights
  (bit-identical three ways), RNG draw accounting.

Prior art inside the repo: this wart has bitten before. Commit `82da77960`
(2026-07-02) reshaped the inline-dataset probe to the documented layout to
unbreak `rectangular_scaled`; amplitude-mode quality collapsed (trained amp
MAE 0.0846 → 0.233 — the same signature as the ladder's) and commit
`8b3d7a011` **restored the flat convention for amplitude mode** without
root-causing why, codifying the accident as required behavior
(`components.py` "Task R1-fix (bisect-report.md #4)" comment). The gain has
been load-bearing and undocumented ever since.

Two physics-correctness notes that make the flat layout a genuine defect and
not a benign convention:

1. **Batch-size coupling.** The training objective's forward gain equals the
   batch size, so the objective differs across full (×16) and partial (×2)
   batches and would silently change under any batch-size change. A physics
   forward must not depend on a training-loader hyperparameter.
2. **Latent cross-sample mixing.** The gain is a pure scalar only because the
   probe is identical across a batch. If per-sample probes ever differ within
   a batch (multi-probe datasets, probe refinement), the coherent mode sum
   MIXES different samples' fields — silent physical nonsense. The layout
   must be banned at the contract level, not tolerated where it happens to be
   harmless. (The in-flight review of `2a9ee2ad9` independently verifies the
   pure-gain claim for the datasets used here.)

## 2. Decision

Adopted (user decision, 2026-07-12): **fix the physics contract now.**

1. Enforce the documented probe layout in the model: flat probe tensors are
   rejected fail-fast, in every physics mode.
2. Migrate the dictionary/inline-dataset emission to the documented layout
   (delete the pre-82da7796 amplitude exception).
3. Introduce an **explicit, documented amplitude physics gain** so the
   conditioning benefit the accidental gain provided is preserved by
   principle, not accident — value fixed by a calibration task, independent
   of batch size.
4. **Re-qualify the reference** (Hybrid + CNN) under corrected physics and
   re-pin every downstream floor; the old floors and bit-exact pins are
   thereby superseded, not silently broken.

Non-goals:

- No changes to TF-side core files (`ptycho/model.py`, `ptycho/diffsim.py`,
  `ptycho/tf_helper.py` remain untouched, per standing constraint).
- No bypass or weakening of the manuscript rectangular factorization
  (existing Task 22 constraint carries over).
- No change to evaluation/stitching/metric contracts (proven correct by the
  cross-eval matrix).
- Historical sealed evidence is never rewritten; it becomes superseded
  diagnostic history with explicit provenance notes.

## 3. Canonical contracts (IDL-style)

These contracts are added to the implementation architecture docs (shard
under `docs/specs/`, cross-referenced from code docstrings) in Task 25.

### 3.1 Probe tensor layout by stage

| Stage | Layout | Producer | Consumer |
|---|---|---|---|
| Container/dataset sample | `(C, P, H, W)` per sample; single shared probe stored `(1, 1, H, W)` equivalents expanded at emission | inline dataset (`components.py`), `PtychoDataset` (`dataloader.py`) | collate |
| Collated batch | `(B, C, P, H, W)`; the existing mmap emission `(B, 1, 1, H, W)` is the C=1, P=1 instance | collate | `ProbeIllumination.forward` |
| Physics forward | mode axis is dim 2 (P); coherent mode sum in `pad_and_diffract` is over true probe modes only | `ProbeIllumination` | `pad_and_diffract` |

### 3.2 `ProbeIllumination.forward(x, probe)` (ptycho_torch/model.py)

- **Precondition (new, enforced):** `probe.ndim == 5` and
  `probe.shape[-2:] == (N, N)` and `probe.shape[0] in (1, B)` and
  `probe.shape[1] in (1, C)`. Any tensor with `ndim < 5` — in particular the
  legacy flat `(B, H, W)` — raises `ProbeLayoutError` (typed, module-level)
  naming the offending shape, the documented contract, and this design doc.
  No implicit right-align broadcasting into the mode slot is reachable.
- **Behavior:** unchanged for contract-conforming inputs
  (`x.unsqueeze(2) * probe * mask`); bit-identical outputs for the mmap
  path's existing `(B, 1, 1, H, W)` emission.
- **Dependency note:** `pad_and_diffract` (helper.py:670) semantics are
  unchanged; with the precondition enforced its mode sum ranges over true
  modes only.

### 3.3 Explicit amplitude physics gain

- New field `amplitude_physics_gain: float` on the torch `ModelConfig`
  (`ptycho_torch/config_params.py`), default **1.0**, plumbed through
  `create_training_payload` overrides like every other torch-only knob and
  validated by the scaling contract (`ptycho_torch/scaling_contract.py`):
  finite, > 0; recorded in payload audit trail and Lightning hparams.
- Applied **once**, multiplicatively, to the predicted amplitude inside the
  amplitude-mode forward (site chosen in Task 25 so `rectangular_scaled` and
  CI paths are untouched; those modes ignore the field and the scaling
  contract validator rejects non-1.0 values for them, fail-closed).
- Rationale: the accidental ×B gain demonstrably conditions amplitude-mode
  training (0.486 vs 0.896 at B=16). The gain must survive as an explicit,
  batch-size-independent, provenance-carrying constant.
- The gain is a **training-objective** device; inference/reassembly of object
  patches does not apply it. Verified 2026-07-12: the accident itself is
  train/val-loop only — the inference path passes a rank-2 probe, which
  broadcasts at gain 1 — so historical checkpoints score identically under
  either physics (cross-eval matrix) and the explicit gain must likewise stay
  out of inference.
- Constant-gain fidelity (measured in the `2a9ee2ad9` review): a fixed G=16
  replicates the accidental objective on 99.82% of train steps (561/562; the
  final partial batch ran at gain 2, loss 0.8710 vs 0.0732, ≈+0.0014
  epoch-mean transient) and 45 of 46 val batches (the 9-sample val tail
  inflates reported val MAE ≈+0.007 constantly, ranking unaffected). rung1f,
  which carried the accident's partial-batch behavior, landed within 0.0025
  amp SSIM of the reference — a batch-size-decoupled constant can faithfully
  replace the accident, with the ≤0.3%-of-steps residual documented.

### 3.4 Emission migration (components.py inline dataset)

- The amplitude-mode branch emits the same `(C, P, H, W)` per-sample layout
  the rectangular branch already emits (the `82da77960` reshape, now
  unconditional); the `8b3d7a011` amplitude exception and its comment are
  removed and replaced by a pointer to this design and the new finding.
- The regression test added by `82da77960`
  (`test_inline_dataset_rectangular_scaled_batched.py`) is extended to pin
  the amplitude-mode layout as well.

## 4. Calibration methodology (Task 26)

The principled gain value was an empirical question with two candidate rules;
Task 26 answered it with a small predeclared experiment before any re-pinning:

- **Rule A — fixed explicit constant:** sweep
  `amplitude_physics_gain ∈ {1, 4, 16, 64}` on the exact reference recipe
  (N128/C1/Run1084 probe/seed 3/5 epochs, dictionary flow under corrected
  emission). Select by validation-loss trajectory + reconstruction
  amp/phase SSIM. The historical accident predicts 16 is near-optimal; the
  sweep tests whether quality is a plateau (any sufficiently large gain) or
  a peak (16 specifically).
- **Rule B — init-time self-calibration:** compute the gain once at training
  start so the initial predicted-amplitude scale matches the measured
  amplitude scale (the TF-side `intensity_scale` convention; cf.
  POISSON-NORM-001's lesson that unpopulated scale constants silently
  default). Deterministic, data-derived, batch-size-independent; sealed into
  hparams like a constant.
- Decision criterion: Rule B is preferred if it lands within the quality
  plateau identified by Rule A's sweep (self-calibrating beats a magic
  constant); otherwise the swept constant wins and its value is documented
  with the sweep evidence. Either way the chosen rule must reproduce
  reference-grade quality (amp SSIM ≥ 0.85 on the reference recipe) or the
  fix phase halts for re-design.
- Secondary instrumentation (cheap, same runs): decoder output
  statistics vs gain, to convert "gain helps conditioning" from hypothesis
  to measured mechanism; this also revisits the Task 22 CNN-saturation
  question, which the flat-probe gain plausibly explains.

**Calibration outcome (2026-07-12, informative):** Task 26 completed at
commit `72816630ea76318c65641e1214953a965a3c0404`. The fixed-constant sweep
produced amp/phase SSIM `0.4817288093`/`0.8077751461` at gain 1,
`0.7475006699`/`0.9273140460` at gain 4,
`0.8858652644`/`0.9618665959` at gain 16, and
`0.8305348576`/`0.9496021958` at gain 64. The quality plateau was exactly
`[16,16]`; Rule B derived gain `4.2441268779`, outside that plateau, so Rule A
won with selected gain 16 and no confirmation run. The halt criterion passed.
Gain 16 is the calibrated legacy normalized-amplitude training value for this
locked N128/Run1084/dictionary/seed-3 reference regime, not a universal physics
scale. `ModelConfig` continues to default to 1.0, and rectangular/CI scaling
continues to require exactly 1.0. Evidence:
`.artifacts/gain_calibration_v2_commit-72816630/sweep_summary.json`.

## 5. Reference re-qualification and floor re-pinning (Task 27)

**Implementation outcome (2026-07-12):** Both gain-16 run2 candidates passed
mandatory visual review, and the Hybrid/CNN floors, CNN immutable artifact,
integration CLI, and ladder baseline were atomically re-pinned. Final fresh
post-pin Hybrid and CNN controller requalification also passed, so Task 27 is
`complete_final`. The plan ledger and Task 27 report record the execution
commits, exact old->new values, final roots, and evidence identities;
historical findings below remain unchanged.

1. Re-run reference qualification for both architectures via the existing
   Task 20 harness under corrected physics + calibrated gain: Hybrid
   (5-epoch recipe) and CNN (20-epoch recreated-reference recipe).
2. Regenerate the frozen CNN floor artifact
   (`grid_lines_cnn_reference_floors.json`) and update its pinned SHA-256 in
   `grid_lines_reference_performance.toml`.
3. Update every pinned floor in one atomic commit, each old→new value listed
   in the commit message and the plan ledger:
   - `scripts/studies/specs/grid_lines_reference_performance.toml`
     (`fixture_amp_ssim_min`/`fixture_phase_ssim_min`, both arms)
   - `scripts/studies/specs/hybrid_resnet_ci_compatibility.toml` (same pins)
   - `tests/studies/test_grid_lines_reference_performance.py`,
     `tests/studies/test_torch_ablation_verdicts.py`,
     `tests/studies/test_grid_lines_bridge_ladder.py` (rung0 control value)
4. Visual review of the re-qualified reconstructions is mandatory before
   pins move (same standard as Task 20's qualification).
5. Superseded-history note: prior floors remain in git history and the
   ledger; `.artifacts/reference_qualification/` old runs are retained
   read-only as diagnostic history.
6. Known-flake caveat carries over: bit-exact pins are CPU-sensitive on some
   GitHub runners (5 known flaky parity tests); new pins follow the same
   tolerance policy the existing suite uses — this task must not widen or
   narrow that policy silently.

## 6. Bridge convergence validation (Task 28)

The decisive systems-level check that the root cause is fully closed: under
corrected physics + calibrated gain, the dictionary flow and the mmap flow
must **converge** on the reference recipe.

Progress: rung0 is checked at
`.artifacts/bridge_ladder/task28_gain16_seed3/rung0_dictionary/grid_lines_hybrid_resnet_reference/reference_evidence.json`
(SHA-256 `155ee5961e31f9cf82c012d6bb61591bd776551f728d66bb19e0f3abee6ad298`).
The exact twin is staged as dataset `n128_run1084_generic` under
`.artifacts/bridge_ladder/task28_gain16_seed3/datasets` with train/test counts
`8978`/`729` and output SHA-256 values
`628cac77ef85c3927e3d5407f509556f054267e71e567aed67500b8de5f6ae4e`/
`17b2aea9a9deeb3ead2ab78771f19b33a2612b2666196e20dd45fa1a51f2275b`.
Its source train/test SHA-256 values are
`c7615e11b2a500c891ed13be0747adba467b451a12e5a31dd18b7f338e89c916`/
`01dd9ff64d84e56b5950865640e79895d82813c0caa451f9552338c07a700699`;
tracked provenance SHA-256 is
`3f97e27de19a28eca85528893741e3558f035e338b8fda7c8a5f8636b8cbf569`.
Rung1a completed with amp/phase SSIM
`0.856505683935826`/`0.9498293416806348`, absolute deltas
`0.0293595804655428`/`0.01203725425813`, and quality verdict `fail` because
`ladder_absolute_amp_ssim_delta_exceeded`; there was no protocol failure and
the CLI returned `1`. Its effective probe matches the recipe, and its
train/test dataset hashes match the pins above. The sealed rung evidence is
`.artifacts/bridge_ladder/task28_gain16_seed3/convergence/rung1a_mmap_full_scanset/rung_evidence.json`
(SHA-256 `f7c3eba38f0a93529cc37315d1d3e42f43f1dd5b1e5b239300d68cdd71f9c132`).
The derived
`.artifacts/bridge_ladder/task28_gain16_seed3/convergence/ladder_report.json`
(SHA-256 `130096cf45fb9e193308f272c84b0179f1948d2c0abacf700634dfec762303c7`)
names `rung1a_mmap_full_scanset` as the first material degradation.
Its spec and evidence paths are repo-relative/logical identifiers, making the
derived report bytes independent of checkout and output-root location.

The diagnostic conclusion is now closed. The dictionary and generic twin
measurement arrays are byte-identical. Dictionary batches carry unit RMS and
physics constants; the failing rung1a used `DataConfig.normalize="Batch"`
(RMS approximately `1.33047`, physics approximately `1.9797e-4`). Historical
rung1c selected `normalize="None"`, restored unit constants, and passed against
rung0 at amp/phase SSIM `0.8913340876617375`/`0.9632217816205675`.
Sampler-only controls rung1d and rung1e passed, so the sampler is exonerated.
The synthetic simulator already owns amplitude conditioning through
`ptycho.diffsim.illuminate_and_diffract`; applying Batch normalization again in
the generic bridge was the mismatch.

Historical evidence is immutable and remains readable at these exact paths:

- old failing rung1a:
  `.artifacts/bridge_ladder/task28_gain16_seed3/convergence/rung1a_mmap_full_scanset/rung_evidence.json`
  (SHA-256 `f7c3eba38f0a93529cc37315d1d3e42f43f1dd5b1e5b239300d68cdd71f9c132`);
- rung1c normalization:
  `.artifacts/bridge_ladder/task28_gain16_seed3/convergence/rung1c_normalization_regime/rung_evidence.json`
  (SHA-256 `b9886b498880c35d4ef5e1a7c18b8c229e41704fd407879d431e2226e65940da`);
- rung1d sampler:
  `.artifacts/bridge_ladder/task28_gain16_seed3/convergence/rung1d_sampler_shuffle/rung_evidence.json`
  (SHA-256 `6df72e84ece6203f8c76326b635dd4835abb59cb9d00757cd3de6d75cd47fcad`);
- rung1e sampler plus unit normalization:
  `.artifacts/bridge_ladder/task28_gain16_seed3/convergence/rung1e_sampler_plus_unit_norm/rung_evidence.json`
  (SHA-256 `92f61a63870ea59993938e206aec52053f901907be7b23d6f9de4b76018cb897`).

Historical rung1f evidence remains proof of the old flat-rank mechanism only
and must not be rerun; its status is `historical_only`/non-runnable. Its immutable path is
`.artifacts/bridge_ladder/seed3_split/rung1f_probe_layout/rung_evidence.json`
(SHA-256 `230b35b9511483e6e409ab5a3e611e925e7ba09fd22bf10c8b3efbbdb2aae324`).
Its `mmap_probe_batch_shape="dictionary_flat"` lever recreates the prohibited
rank and now raises `ProbeLayoutError`; explicit gain 16 must not be combined
with the accidental rank gain.

The current executable ladder makes `dictionary_parity` the baseline
convention. That declaration is inert for rung0 dictionary loading and resolves
canonical rung1a to `DataConfig.normalize="None"` in both its prebuilt payload
and internal training overrides. Rungs 1c-1f are already archived from the
current TOML because rung1c would be a no-op under the new baseline; their paths
above preserve provenance. CI/count behavior is unchanged: the count rung
explicitly selects the loader/`Batch` convention.

Canonical rung1a passed at amp/phase SSIM
`0.8913340876617375`/`0.9632217816205675`; absolute deltas from rung0 were
`0.0054688232603687`/`0.0013551856818027` inside the locked `0.02`/`0.01`
gate. The fresh root is
`.artifacts/bridge_ladder/task28_gain16_seed3_unit_scaling_rerun`; rung
evidence/report SHA-256 values are
`a8741f93511cc90eb26777b1ff2cab0235ce812cff3661bad168632fce9ce711` and
`2f95c0e4651a2a48449797f4a34aed557ce70962c8e82dced0d40b4aa95a0cf6`.
Task 28's regression adjudication is only rung0 versus canonical rung1a. The
remaining rung1b-through-rung8 scaffold and historical injection/parser support
stay in place until Task 29 producer retirement. This conservative retention
decision is not remaining Task 28 work and does not authorize a tombstone.

## 7. Blast radius and compatibility

**Code (all torch-side; TF core untouched):**
`ptycho_torch/model.py` (ProbeIllumination precondition + gain application
site — core-adjacent; this plan revision explicitly authorizes the edit),
`ptycho_torch/workflows/components.py` (emission migration),
`ptycho_torch/config_params.py` + `config_factory.py` + `scaling_contract.py`
(gain field + validation), `ptycho_torch/dataloader.py` (no behavior change;
optional assert), ladder spec/harness (Task 28 trim).

**Contracts/docs:** new shard section for §3 contracts; `docs/findings.md`
gains PROBE-RANK-001 (mechanism, history `82da77960`/`8b3d7a011`, fix,
cross-refs POISSON-NORM-001, TORCH-REASSEMBLY-SIGN-001);
`docs/workflows/pytorch.md` notes the gain field.

**Pinned evidence:** all floors in §5.3; NEURIPS
`paper_evidence_manifest.json` — any manuscript number derived from
gain-affected training is flagged to the NEURIPS initiative owner in Task 24
publication language; this initiative does not edit manuscript claims
unilaterally.

**Checkpoint compatibility:** existing sealed checkpoints remain loadable and
evaluable (evaluation is checkpoint-determined and unchanged); they are
marked, via the finding, as trained under the accidental-gain objective.
Retraining is required for any new claim-grade run — that is Tasks 23–24 by
design.

**Concurrent initiatives (shared-tree hazard):** this branch
(`feature/ci-compatibility-ablation`) is isolated from the `fno-stable`
worktree's ~8 concurrent initiatives; the contract change reaches them only
at merge. The fail-fast error (not silent behavior change) is the designed
protection: any flow still emitting flat probes stops loudly with a pointer
to this design. Merge sequencing is coordinated at initiative close, not
mid-flight.

**Existing tests knowingly affected:** the `8b3d7a011` amplitude-exception
comment/tests, `test_inline_dataset_rectangular_scaled_batched.py`
(extended), pinned-floor tests (§5.3), any test constructing flat probes
against the model (inventory during Task 25 RED phase).

## 8. Test strategy (TDD, RED cases predeclared)

1. RED: flat `(B, H, W)` probe into `ProbeIllumination.forward` →
   `ProbeLayoutError` (today: silent pseudo-mode broadcast).
2. RED: amplitude-mode inline dataset emits documented per-sample layout;
   collated batch is `(B, C, P, H, W)` (today: flat).
3. RED: `amplitude_physics_gain` plumbed payload→hparams→forward; gain 16
   at fixed weights reproduces the step-parity loss (0.9361 → ≈0.0755
   band) — ties the explicit mechanism quantitatively to the measured
   accident.
4. RED: scaling contract rejects gain ≠ 1.0 for `rectangular_scaled`/CI
   modes.
5. GREEN + regression: mmap emission unchanged bit-for-bit; corrected
   dictionary flow and mmap flow produce identical batches through
   `ProbeIllumination` (extends the step-parity harness).
5a. RED: per-sample-distinct probes in one batch under the (banned) flat
   layout would couple samples (the review demonstrated sample 0's output
   changes when sample 3's probe is perturbed) — the `ProbeLayoutError` test
   matrix includes a multi-probe batch case so the mixing hazard is pinned,
   not just the gain.
6. Re-pinned floor tests (Task 27) and convergence gate test (Task 28).

## 9. Risks and open questions

- **R1:** Calibrated-gain quality may land below the historical floors
  (the accident may have been fortuitously good). Response: floors re-pin to
  corrected-physics reality; a drop is documented, not hidden — the old
  numbers were produced by unphysical batch-coupled gain and are not a valid
  target. User has accepted this by choosing this path.
- **R2 (resolved 2026-07-12):** Task 26 measured the gain response and selected
  fixed gain 16 for the locked legacy-amplitude reference regime under §4's
  predeclared criterion. This empirical calibration does not establish a
  universal physics scale.
- **R3 (resolved 2026-07-12):** the root-cause review landed with all links
  CONFIRMED, 0 Critical / 0 Important / 5 Minor (P-1..P-5, ledgered; P-5 —
  a multi-experiment probe-mixing guard — is subsumed by §3.2's fail-fast
  enforcement and §8's mixing test case).
- **R4:** Unknown in-repo flat-probe producers beyond the inline dataset.
  Mitigated by the fail-fast error plus a repo-wide RED-phase inventory
  (grep for model-bound probe constructions) in Task 25.
- **R5:** CNN reference (20-epoch recipe) may respond differently to
  corrected physics than Hybrid. Task 27 qualifies both independently; a
  CNN-specific failure becomes a scoped finding, not a blocker for the
  Hybrid track (matching Task 20's per-architecture structure).

## 10. Task mapping

| Plan task | Content | Gate |
|---|---|---|
| 25 | Contract enforcement + emission migration + explicit gain plumbing (§3, §8.1–5) | Design + root-cause reviews clean; TDD evidence |
| 26 | Gain calibration experiment (§4) | Complete: Rule A fixed gain 16 reached amp SSIM `0.8858652644` |
| 27 | Reference re-qualification + atomic floor re-pinning (§5) | Visual review + pins moved in one commit |
| 28 | Dictionary↔mmap convergence validation + ladder trim (§6) | Convergence gate passes |
| 22–24, 30 (revised) | Convergence diagnostic; CNN contract recovery; multi-seed execution; re-adjudication/publication | Task 22 complete; Task 30 blocks Task 23; Task 24 dependency-pending |
