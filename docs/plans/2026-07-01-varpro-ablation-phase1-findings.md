# VarPro Ablation Phase 1 — Findings (Tasks 1.1–1.6)

**Branch:** `varpro-ablation`, created via `git switch -c varpro-ablation 5bd07399` (`5bd07399` == `main` at the time of branching: "Refine DDP run name sync to match final latent_experimental state").

**Purpose:** This is the decision record Task 2.6 (on `fno-stable`) builds against when it ports `main`'s `RectangularScaledDiffraction` into `fno-stable`'s differently-structured loss stack, and the reference the hybrid-resnet extension plan (`docs/plans/2026-07-01-hybrid-resnet-varpro-probe-extension.md`) cites for the physical meaning of `s1,s2`. All line numbers below were re-verified directly against `ptycho_torch/model.py` as checked out on this branch (i.e., `main`), not copied from the reviewer's claim. Full provenance for every numbered finding lives in `.superpowers/sdd/plan-amendments-pending.md`; this doc cites the amendment number (`#N`) rather than restating the investigation narrative.

---

## Loss units and scale routing on main (5bd07399)

### 1. Loss-input units: does Poisson/MAE see intensity or amplitude, and do they square internally?

`RectangularScaledDiffraction.forward` (the module `ForwardModel` calls as `self.rect_scaler`) computes a genuine **intensity** quantity — the incoherent mode sum of squared FFT magnitudes:

```python
# ptycho_torch/model.py:729-733
psi_f = torch.fft.fftshift(torch.fft.fft2(exit_wave, norm='ortho'), dim=(-2,-1))

#Incoherent mode summation: intensity per mode, then sum over modes
I_pred = torch.sum(torch.abs(psi_f)**2, dim = 2) #(B,C,P,H,W) -> (B,C,H,W)
```
clamped non-negative at `model.py:755`: `I_pred = torch.clamp(I_pred, min=0.0)`, then returned.

`ForwardModel.forward` passes this straight through, unmodified, under the name `pred_scaled_intensity`:

```python
# ptycho_torch/model.py:694-702
pred_scaled_intensity = self.rect_scaler(
    x=extracted_patch_objs,
    I_raw=I_measured,
    probe=probe,
    scale=output_scale_factor,
    experiment_ids=experiment_ids,
    autograd=True)

return pred_scaled_intensity
```

`PtychoPINN.forward` (`model.py:1168-1179`) returns this same tensor unchanged as `x_out`, and `compute_loss` binds it to `pred`:

```python
# ptycho_torch/model.py:1540-1544
pred, real, imag = self(x, positions, probe,
                        input_scale_factor=rms_scale,
                        output_scale_factor=modified_output_scale,
                        experiment_ids=experiment_ids,
                        fine_tune=self._fine_tuning_mode)
```

So **the value that reaches the loss (`pred`) is already an intensity**, not an amplitude — it is literally `sum(|FFT(exit_wave)|**2)` over probe modes.

The two loss classes (`ptycho_torch/model.py:961-977`):

```python
class PoissonLoss(nn.Module):
    def __init__(self):
        super(PoissonLoss, self).__init__()
    def forward(self, pred, raw):
        poisson = PoissonIntensityLayer(pred)
        return poisson(raw)

class MAELoss(nn.Module):
    def __init__(self):
        super(MAELoss, self).__init__()
        self.mae = nn.L1Loss(reduction = 'none')

    def forward(self, pred, raw):
        #Note: Prediction has not been squared yet, must be squared here
        loss_mae = self.mae(pred**2, raw)

        return loss_mae
```

- **PoissonLoss**: consumes `pred` **directly, with no internal squaring** — `PoissonIntensityLayer.__init__` (`model.py:641-644`) binds `Lambda = intensities` (the Poisson rate parameter) straight to `pred`, and `forward` (`model.py:646-647`) evaluates `-self.poisson_dist.log_prob(x)` against `raw` with no transform. This is unit-consistent with `pred` already being an intensity: Poisson's rate parameter is expected to be an intensity/count-like quantity, and that is exactly what `RectangularScaledDiffraction` hands it.
- **MAELoss**: **squares `pred` internally** (`pred**2`, `model.py:975`) before the L1 comparison against `raw`, per its own comment ("Prediction has not been squared yet, must be squared here", `model.py:974`). This treats its `pred` argument as if it were an **amplitude** that still needs squaring to reach intensity units.

This is confirmed by the project's own unit test, `tests/torch/test_loss_units.py`, whose variable names encode the intended contract for each class:

```python
# tests/torch/test_loss_units.py:7-19
def test_poisson_intensity_layer_uses_intensities_directly():
    pred_intensity = torch.tensor([[[[4.0]]]])  # (B, C, H, W)
    obs_intensity = torch.tensor([[[[9.0]]]])
    layer = PoissonIntensityLayer(pred_intensity)
    loss = layer(obs_intensity)
    ...

# tests/torch/test_loss_units.py:22-30
def test_mae_loss_squares_predictions():
    pred_amp = torch.tensor([[[[2.0]]]])
    obs_intensity = torch.tensor([[[[3.5]]]])
    loss_fn = MAELoss()
    loss = loss_fn(pred_amp, obs_intensity)
    expected = torch.nn.functional.l1_loss(pred_amp ** 2, obs_intensity, reduction="none")
    ...
```

`PoissonIntensityLayer`'s test names its `pred` argument `pred_intensity` (no squaring). `MAELoss`'s test names its `pred` argument `pred_amp` (squared internally). **Net finding for Task 2.6**: on main, `compute_loss` always feeds `pred_scaled_intensity` (a true intensity) as the `pred` argument to whichever loss is configured. This is unit-consistent for the `Poisson` branch (the default; `ModelConfig.loss_function: Literal['MAE', 'Poisson'] = 'Poisson'`, `ptycho_torch/config_params.py:80`) but means the `MAE` branch's own internal squaring (`pred**2`) is applied to an already-intensity value rather than to an amplitude — i.e. `MAELoss` was authored assuming an amplitude-valued `pred`, and `compute_loss`'s Unsupervised/RectangularScaledDiffraction path does not supply one. **Any port of `RectangularScaledDiffraction` into `fno-stable`'s loss stack must decide explicitly how to handle the `MAE` branch's squaring assumption — it cannot be silently inherited as "square pred, then compare," because `pred` is already squared by construction.** For the `Poisson` branch (the actual default and the one exercised by `tests/torch/test_physics_scale_loss.py`), no such conflict exists: intensity in, intensity rate parameter, no extra squaring.

The `raw` argument in both loss classes is `x = batch[0]['images']` (`model.py:1524`), loaded via `_get_diffraction_stack` (`ptycho_torch/dataloader.py:107-150`), whose docstring states the returned array is `"Diffraction patterns (amplitude, float32)"` per the canonical `docs/data_contracts.md:15` convention ("amplitude, not intensity"). No sqrt/square transform is applied to `x` anywhere in `compute_loss` before it is passed as `raw`. This is noted for completeness but is out of scope to resolve here — it is a pre-existing tension between the loader's documented convention and the loss modules' own naming (`obs_intensity` in both loss unit tests), not something introduced by the varpro/physics-scale work this finding is about.

### 2. Scale routing: how `physics_scale` enters the forward pass (NOT multiplied at loss time)

`compute_loss` (`ptycho_torch/model.py:1522`) unpacks the batch:

```python
# ptycho_torch/model.py:1522-1530
def compute_loss(self, batch):
    """Loss computation with RMS normalization."""
    x = batch[0]['images']
    positions = batch[0]['coords_relative']
    probe = batch[1]
    rms_scale = batch[0]['rms_scaling_constant']
    physics_scale = batch[0]['physics_scaling_constant']
    experiment_ids = batch[0]['experiment_id']
    probe_scaling = batch[2]
```

- `physics_scale` comes from `batch[0]['physics_scaling_constant']` (a per-sample dict entry, batch position 0).
- `probe_scaling` comes from `batch[2]` (the third element of the batch tuple `(dict, probe, probe_scaling)`).

This tuple structure is confirmed by `tests/torch/test_physics_scale_loss.py::test_poisson_loss_uses_physics_scale`, which constructs `batch = ({... "physics_scaling_constant": ...}, probe, torch.ones(1))` and calls `model.compute_loss(batch)` directly, and asserts that varying only `physics_scaling_constant` changes the resulting loss (i.e., physics_scale demonstrably affects the loss through the forward path, not through a separate multiply).

Both are folded into a single `output_scale_factor` **before** the forward pass runs, not applied to `pred` afterward:

```python
# ptycho_torch/model.py:1538
modified_output_scale = torch.sqrt(1/(probe_scaling**2 * physics_scale + 1e-9))

# ptycho_torch/model.py:1540-1544
pred, real, imag = self(x, positions, probe,
                        input_scale_factor=rms_scale,
                        output_scale_factor=modified_output_scale,
                        experiment_ids=experiment_ids,
                        fine_tune=self._fine_tuning_mode)
```

`output_scale_factor=modified_output_scale` is then threaded through `PtychoPINN.forward` → `ForwardModel.forward` → `RectangularScaledDiffraction.forward`'s `scale` argument (`model.py:715`, used at `model.py:725-727`: `scale = scale.unsqueeze(dim=2)`; `exit_wave = scale * (s1 * (probe * x_a) + 1j * s2 * (probe * x_b))`). The physics scale is therefore baked into the predicted exit wave *before* the FFT and mode-sum that produce `pred_scaled_intensity` — by the time `pred` reaches `compute_loss`'s `self.Loss(pred, x)` call (`model.py:1549`), `physics_scale` has already been fully absorbed into `pred`.

**Explicitly: main does NOT multiply `pred * physics_scale` (or `x * physics_scale`) at loss time.** There is no `physics_scale` reference anywhere in `compute_loss` after line 1538 other than inside the `modified_output_scale` computation itself — grep confirms the only three occurrences of `modified_output_scale`/`physics_scaling_constant`/`physics_scale` in the loss path are lines 1528, 1538, and 1542.

**Contrast with `fno-stable`** (for context; verified via `git show e672d155:ptycho_torch/model.py` without switching branches, so this is `fno-stable`'s `origin/fno-stable` HEAD at the time of this finding, not independently re-derived from a live checkout):

```python
# fno-stable @ e672d155, ptycho_torch/model.py:1813-1816
if self.model_config.mode == 'Unsupervised':
    pred_physics = pred * physics_scale
    obs_physics = observed_images * physics_scale
```

`fno-stable` computes `pred` using a *plain* `output_scale_factor = rms_scale` (no `probe_scaling`/`physics_scale` folding at the forward call), and instead multiplies both `pred` and the observed images by `physics_scale` **after** the forward pass, at loss-computation time.

**Implication for Task 2.6**: porting `RectangularScaledDiffraction` into `fno-stable`'s loss stack must reproduce main's *routing* — folding `probe_scaling` and `physics_scale` into `output_scale_factor` via `modified_output_scale = torch.sqrt(1/(probe_scaling**2 * physics_scale + 1e-9))` and passing that into the forward call — not merely deleting fno-stable's `pred * physics_scale` / `obs_physics = observed_images * physics_scale` multiply and leaving the rest of fno-stable's structure untouched. Skipping the multiply without adding the pre-forward folding would silently drop physics scaling from the loss entirely.

---

## Self-review (Task 1.1)

- Every claim above is anchored to a quoted code block and file:line reference; anchors were read directly from the checked-out `varpro-ablation` tree (`git branch --show-current` → `varpro-ablation`; `git log -1` → `5bd07399`), not paraphrased from the task brief.
- The three reviewer-supplied anchors were independently re-confirmed by reading the file rather than trusting the brief: `compute_loss` def is at `model.py:1522` (exact), `modified_output_scale = torch.sqrt(1/(probe_scaling**2 * physics_scale + 1e-9))` is at `model.py:1538` (exact), and it is passed as `output_scale_factor=modified_output_scale` at `model.py:1542` (exact, one line after the reviewer's approximate "~L1542").
- The fno-stable contrast (`model.py:1815-1816`) was independently re-verified via `git show e672d155:ptycho_torch/model.py`, not merely copied from the brief; it is reported as `fno-stable`-branch context, not re-derived by switching this tree's branch (per task instructions, this branch stays on `main`).
- Went beyond the two required facts to flag one additional, directly relevant tension (the `MAELoss` squaring-assumption mismatch against an already-intensity `pred`) because Task 2.6 explicitly needs to decide how to handle the MAE branch when porting; this is reported as an observed fact (with the unit test as evidence) rather than as a verdict on whether it constitutes a bug.

## Path equivalence (Task 1.4b)

Main's real training ran through `ptycho_torch/train_lightning_only.py::main()` (main's CLAUDE.md documents only this entry; commit 93ca0fc0 fixes a live DDP deadlock in exactly this path one commit before HEAD). The `train.py cli_main -> components` "canonical" path was non-runnable at 5bd07399 (six pre-existing defects; five repaired in Tasks 1.3/1.3b) and, decisively, diverges SEMANTICALLY even when repaired: its container assembly never calls `hh.get_rms_scaling_factor` (rms_scaling_constant silently 1.0), never calls `hh.normalize_probe` (raw probe, no compensating scale), and computes `physics_scaling_constant` via a formula never cross-validated against the dataloader path's. PT ModelConfig/InferenceConfig construction matched exactly across both paths (including `training_patch_weighting`, `rect_s1s2_trainable`); actionable mismatches: `nphotons` (no auto-inference on the real path — set 1e9 explicitly) and inert `orchestrator` default.

**Decision (user-approved):** the ablation harness drives `train_lightning_only.py`; knobs flow as plain ModelConfig fields via its config construction; the 1-epoch smoke + s1/s2 freeze/moved verification re-homes to that entry (owed before the matrix). The five canonical-path repairs stay (18 tests fixed, merged-intent restorations); the canonical path's scaling-physics gap stands recorded here as a main-lineage defect, deliberately NOT reconstructed for the ablation. Full investigation: `.superpowers/sdd/task-1.4b-investigation.md` (scratch; substance captured here).

---

## Harness defects discovered en route to the matrix (Tasks 1.5–1.6)

These are pre-existing main-lineage defects surfaced while building/running the ablation harness. None were fixed beyond what is noted; each is recorded here so Task 2.6/2.9 inherit them as known facts rather than rediscovering them.

**Dead CLI inference flags.** The plan text specified driving per-variant inference through the `python -m ptycho_torch.inference` subprocess. Empirically, all 4 smoke variants came back bit-identical: the CLI's `_run_inference_and_reconstruct` parses and threads `patch_weighting`/`varpro_scaling` but then unconditionally calls uniform `hh.reassemble_patches_position_real`, silently discarding both knobs. The harness instead drives the in-process library path — `lightning_utils.load_checkpoint_with_configs` + `PtychoDataset` from the test NPZ + `reassembly.reconstruct_image_barycentric` — which does honor both knobs (`reassembly.py:1119` patch weighting, `:1255` varpro gate) and is main CLAUDE.md's documented inference pattern. The CLI flags remain dead on main; not fixed in Phase 1.

**Amplitude/count data-convention mismatch (root cause of the first matrix failure).** The Task 0.1 datasets were generated on `fno-stable` and follow its DATA-001 convention (normalized amplitude, per-pattern sum-of-squares = 1). Main's pipeline assumes photon-COUNT data throughout: `get_physics_scaling_factor` treats sums as intensity, `probe_scaling`/`modified_output_scale` calibration assumes count scale, and `.round()` + Poisson-loss integer support both only make sense on counts. Training every arm on normalized data produced uniformly degenerate runs (grad_norm→0 by epoch 2, loss pinned ~0.85, decoder rail_fraction ~0.66, canvas std ~1e-7) because the initial prediction was ~37x too small and the only available gradient direction was global amplitude, which rails the hardwired tanh bounds before structure forms (`plan-amendments-pending.md` #10). Corrective: all matrix datasets are regenerated in main-native count convention (`scripts/studies/make_synthetic_truth_datasets.py`), and every arm trains with `torch_loss_mode='poisson'` (main's native loss, integer-count-compatible) rather than the `mae` workaround used while chasing the Poisson-crashes-on-amplitude symptom. `PtychoDataset.memory_map_data`'s `.round()` — which had been zeroing spec-compliant amplitude data — is fixed per the DATA-001 references already in `dataloader.py`'s own docstrings.

---

## Demo-object selection: dead_leaves → lines (Task 1.6, amendments #12, #13, #13b, #13c)

**Why not fly001/fly64:** `fly001.npz`'s stored `objectGuess` is an all-ones placeholder (amp/phase std 0), so any error-vs-truth metric is meaningless against it, and the N=128 set simulated from that placeholder (`ptycho/nongrid_simulation`) gives near-identical diffraction across scan positions (cross-pattern deviation 0.002 vs 0.573 for real fly001) — no positional information, flat reconstructions regardless of method (#12). Both prior matrix attempts against fly-derived data are kept as evidence of these two failure modes, not as the demo.

**dead_leaves, first pass:** synthetic `dead_leaves` objects (real multi-scale ground truth, count-convention diffraction matching real fly001, mean count 108) looked like the right substitute, but the full ±π phase range hits the real_imag decoder's hardwired `ScaledTanh` bounds (real floor −0.8; a unit-amplitude object at |phase|→π maps to real≈−1, outside the box) — a genuine, still-standing representability ceiling (see next section). Chasing that ceiling produced two successive false leads that were later retracted:
- A "weak-phase (±0.8 rad) fixes it" claim (per-patch |amp| NCC 0.31→0.70) that did not reproduce on retraining (0.33/0.70 became 0.31/0.55 across five runs) — traced to a non-reproducible checkpoint (#13b).
- A "dead_leaves is uniformly ~0.31, box-limited or weakly-scattering" conclusion that was later found to be a **measurement artifact**: `diagnose_placement.py`/`diagnose_stitching.py` reused a fixed scratch directory, and `ptycho_torch`'s `build_test_dataset` keys its memory map by directory PATH, not content — so later runs forward-predicted on the FIRST dataset's stale diffraction while scoring against a freshly loaded, different truth object (tell: item count n=118, the dead_leaves bound-filter count, appearing where a fresh lines set should show n=59) (#13c).

**Pivot to lines (user-directed):** independent of the artifact, dead_leaves' weak amplitude contrast (amp_std ~0.15) plus the user's standing instruction ("switch to lines if you reach an impasse"; "grid lines confirmed working on fno-stable") motivated adding a synthetic **lines** demo object (`scripts/studies/make_lines_datasets.py`, `sim_object_image('lines')`): amplitude normalized to [0.3, 1.0], spatially varying phase ±0.5 rad (phase is required — a constant-phase, pure-amplitude object retriggers the VarPro `X3=0`, `s1↔s2` degeneracy). Lines became the primary demo object; dead_leaves (phase-compressed to stay inside the decoder box, `PHASE_MAX=0.5`) is retained as a harder cross-check object (see the comparison table below), and full ±π dead_leaves is retained as evidence of the box limit.

**Gate hardening (#13c):** `scripts/studies/recon_quality_gate.py` was fixed to allocate a fresh per-arm scratch directory (`shutil.rmtree` before each build) and to flatten gridsize-2 groups (place all `C=gridsize**2` patches, not just `[:,0]`) before computing the metric below. Any future harness/test pointing `build_test_dataset` at a new NPZ must use a fresh scratch dir — reusing one silently reintroduces the stale-map artifact.

---

## Reconstruction validation: 7-arm LINES matrix (Task 1.6)

**Metric:** gauge-quotiented canvas `|amp|` NCC — `recon_quality_gate.py`'s direct per-patch placement into the truth frame (object-pixel coordinates, one coordinate source, no `skimage.match_template`), with the reconstruction's global complex gauge removed by least-squares fit against truth before correlating. This is the metric that survived the artifact hunt in #13/#13b/#13c; canvas center-crop-against-`objectGuess` and `match_template`-based numbers are both known to be unreliable (see the flux-sweep section below for a second, independent instance of the same lesson). **Recon-quality gate threshold: > 0.6.**

| Arm | Description | Canvas \|amp\| NCC | Gate |
|---|---|---|---|
| `gs1_frozen` | gridsize 1, s1/s2 frozen at 1.0 | 0.972 | PASS |
| `gs1_trainable` | gridsize 1, s1/s2 trainable | 0.970 | PASS |
| `gs2_neither` | gridsize 2, no probe weighting / no train-time scaling | 0.915 | PASS |
| `gs2_probe_frozen` | gridsize 2, probe-weighted, s1/s2 frozen | 0.975 | PASS |
| `gs2_probe_trainable` | gridsize 2, probe-weighted, s1/s2 trainable | 0.975 | PASS |
| `gs2_neither_n128` | gridsize 2, N=128, neither knob | 0.740 | PASS |
| `gs2_probe_trainable_n128` | gridsize 2, N=128, both knobs | 0.914 | PASS |

All 7 arms pass. gridsize-1 arms sit near ceiling regardless of knob state (s1/s2 ≈ 1.0 there — RMS input-norm plus matched object amplitude leave nothing for the scale factorization to absorb, so frozen-vs-trainable is within training noise). The gridsize-2 signal is the intended ablation effect: **training-time probe weighting improves gridsize-2 reconstruction** — N=64 `gs2_neither` 0.915 → `gs2_probe_frozen`/`gs2_probe_trainable` 0.975; N=128 `gs2_neither_n128` 0.740 → `gs2_probe_trainable_n128` 0.914. Full provenance and the stale-map correction that produced these clean numbers: `plan-amendments-pending.md` #13c.

---

## dead_leaves cross-check (N=64 + N=128)

Same recon-quality-gate metric applied to the phase-compressed (`PHASE_MAX=0.5`, inside the real_imag decoder box) dead_leaves matrix, run against the same 7-arm grid (6 arms captured; `gs2_probe_trainable` N=64 not run for this object):

| Arm | Canvas \|amp\| NCC | Gate |
|---|---|---|
| `gs1_frozen` | 0.781 | PASS |
| `gs1_trainable` | 0.758 | PASS |
| `gs2_neither` | 0.646 | PASS |
| `gs2_probe_frozen` | 0.687 | PASS |
| `gs2_neither_n128` | 0.482 | **FAIL** (only sub-threshold arm) |
| `gs2_probe_trainable_n128` | 0.634 | PASS |

**Cross-check finding:** the training-time probe-weighting effect seen on lines reproduces on dead_leaves and is **stronger at N=128**:

| Object | N | neither | probe-weighted | delta |
|---|---|---|---|---|
| lines | 64 | 0.915 | 0.975 | +0.060 |
| lines | 128 | 0.740 | 0.914 | +0.174 |
| dead_leaves | 64 | 0.646 | 0.687 | +0.041 |
| dead_leaves | 128 | 0.482 | 0.634 | +0.152 |

dead_leaves is uniformly harder than lines at every matched arm (weak amplitude contrast relative to lines' strong contrast), and `gs2_neither_n128` on dead_leaves is the one arm across both objects that fails the gate — a difficulty floor, not a defect in the harness or the physics. `PHASE_MAX=0.5` keeps every dead_leaves object inside the real_imag decoder box (per the box constraint documented below); the residual difficulty is amplitude contrast, not phase representability.

---

## `s1, s2` physical semantics — corrected (amendment #14)

**This section retracts an earlier in-session framing.** `s1, s2` are the dynamic photon/amplitude-scale factorization described in the PtychoPINN-CI manuscript ("Contrast-invariant deep ptychography neural networks", Vong, Henke, Hoidn, Mehta, Shapiro, Hexemer, Schwarz), Eq 1/3/5 — **not** a decoder-box range extender. The real_imag decoder emits unit-less tanh textures `ã, b̃ ∈ [-1, 1]`; the physical object is

```
O = s1·ã + j·s2·b̃
```

Because rectangular coordinates are linear through the FT, the predicted intensity is quadratic in the two scalars (`model.py:705-758`, `RectangularScaledDiffraction`):

```
I = |s1·Ψ_a + s2·Ψ_b|² = s1²|Ψ_a|² + 2·s1·s2·Re[conj(Ψ_a)·Ψ_b] + s2²|Ψ_b|²
```

giving amplitude scale `c_A = √(s1² + s2²)` and phase contrast `c_φ = arctan(s2/s1)` (Eq 5). Scale enters via the Poisson NLL at training time (`PoissonLoss`, `model.py:1458`, applied as `Loss(pred, x)` at `:1551`) and via the inference least-squares solve (Eq 6); `forward_predict` (`model.py:1504`) applies only the RMS `input_scale_factor` and **no** output/physics scale, so at inference `s1, s2` carry the entire output scale, LS-solved against raw counts.

**Retracted:** the earlier framing that trainable `s1/s2` "extends the [decoder] box for strong phase" is wrong. `s1, s2` are two global per-dataset scalars; they cannot rescue individual box-railed pixels, and box representability is orthogonal to scaling. The real_imag decoder box documented above (hardwired `ScaledTanh` bounds, real floor −0.8) is a **separate, still-valid** representability constraint — full ±π phase at unit amplitude caps per-patch fidelity regardless of how `s1, s2` are set. Corollary: `rect_s1s2_trainable` (a training-time knob) is secondary — inference always re-solves `s1, s2` regardless of how they were left at the end of training; the headline comparison is inference varpro ON vs OFF (manuscript Fig 3c), not train-time frozen vs trainable. The gs1 arms show `s1, s2 ≈ 1.0` (trained value 0.999) simply because RMS input-norm plus matched object amplitude leave nothing to absorb.

Also corrected: the earlier "VarPro 5× cmae penalty" observed in an un-gauge-quotiented metric (`progress.md`, Task 1.6 dead_leaves run) was the √(mean-count) count-unit scale factor showing up in an amplitude-vs-normalized-truth comparison — not a gauge artifact from complex-scalar phase, and not a defect.

---

## Flux-sweep experiment (Task 1.6 addendum, amendment #14)

**Design:** one lines `gs1_frozen` checkpoint, trained once at mean-count 100 (Poisson NLL), evaluated at mean-count {1, 100, 10000} (four orders of flux; identical object/probe/scan positions, only count scale + quantization differ) × inference varpro {ON, OFF}. Scripts: `scripts/studies/make_flux_sweep.py` (generator, reuses `make_synthetic_truth_datasets` helpers + the frozen lines object) and `scripts/studies/flux_sweep_eval.py` (single-checkpoint multi-flux eval, fresh per-flux scratch per the #13c gate-hardening rule). Evidence-only — no new physics code; exercises the existing `RectangularScaledDiffraction` + varpro inference gate.

**Scale result (solid, reproducible):**

| mean-count | c_A = √(s1²+s2²) | c_A ratio vs mean=100 | √flux ratio (expected) | c_φ = arctan(s2/s1) |
|---|---|---|---|---|
| 1 | 0.913 | 0.098 | 0.100 | −65.0° |
| 100 | 9.279 | 1.000 | 1.000 | −65.8° |
| 10000 | 92.96 | 10.019 | 10.000 | −65.4° |

`c_A` tracks √flux exactly; `c_φ` is flux-invariant (intrinsic phase contrast, not a scale artifact). `|O|` with varpro OFF is flux-invariant (~1, the RMS-normalized texture); with varpro ON, `|O| ∝ √flux` (0.72 / 7.6 / 76 across the three fluxes).

**Fidelity result (validated via a hard anchor reproducing the recon-quality gate's 0.9722 for lines gs1 before trusting any number — the same "don't trust canvas center-crop" lesson from the demo-object section above applied here: a first attempt scored fidelity by center-cropping the `reconstruct_image_barycentric` canvas against the padded `objectGuess` and returned ~0.25–0.31, contradicting the gate; that was a framing artifact, discarded):**

| mean-count | \|amp\| NCC (varpro ON / OFF) | phase MAE (ON / OFF) |
|---|---|---|
| 1 | 0.835 / 0.818 | 0.171 / 0.081 |
| 100 | 0.952 / 0.975 | 0.167 / 0.045 |
| 10000 | 0.953 / 0.975 | 0.168 / 0.045 |

**Finding:** varpro-ON is modestly but reproducibly WORSE than OFF in gauge-quotiented fidelity at adequate flux (~0.02 |amp| NCC, ~4× phase MAE), even though both are good in absolute terms (>0.95 amp NCC). See the sign-flip section below for the mechanism.

**Caveat (reconciliation item):** on this pipeline, neither variant reaches `|O| ≈ truth` without a gauge — OFF stays at the RMS-normalized texture scale (~1), ON is in count-amplitude units — because the probe is normalized at inference, whereas the manuscript's convention leaves the probe un-normalized (so the probe itself carries flux and `s1, s2` stay ~1). Reconciling to the manuscript's `~1` output convention needs the un-normalized-probe inference path.

**Figure:** `.artifacts/varpro_ablation/composite/flux_sweep.png` (generated by `scripts/studies/plot_flux_sweep.py`).

---

## Sign-flip finding (this session)

The varpro solve returns `s2` opposite-signed to `s1` (`c_φ ≈ −65°` at every flux), and this anisotropic, sign-flipped scale is what drives the ~4× worse varpro-ON phase MAE in the flux-sweep fidelity table above: a single global complex gauge removes only isotropic scale + rotation, so an anisotropic real/imag rescale with a relative sign flip leaves a real, flux-independent fidelity penalty that gauge-quotienting cannot absorb.

**Verdict: not a bug.** The negative relative sign originates in the trained network's own real/imag decomposition, not in the VarPro solve: raw `corr(b̃, truth_imag) ≈ −0.85` measured on the decoder's textures *before* VarPro is applied — physics-only (Poisson NLL against diffraction) training never supervises phase sign, so the network is free to settle into either sign convention. The diffraction fit itself only weakly prefers the flipped sign (<0.5% residual difference), but VarPro's least-squares solve amplifies it (`|s2| ≈ 2.2·|s1|`). `enforce_physics_constraint`/`solve_lbfgs` canonicalize only the harmless joint sign of `(s1, s2)` together (an overall gauge); the *relative* sign between `s1` and `s2` is genuinely data-fitted, not a convention choice the code makes.

**Characterization:** the varpro phase-fidelity penalty is an inherent measurement-vs-truth identifiability gap in the current per-checkpoint global `(s1, s2)` gauge-fixing scheme — a real property of this training+inference pipeline, not a universal `|FFT|²` gauge ambiguity and not a code defect.

---

## Phase-2 implications

The merged `rectangular_scaled` + varpro feature's **primary value is cross-measurement photon-scale generalization** (train once, correct scale at inference for any flux — the manuscript's headline contribution), **not same-scale fidelity**, where it is slightly detrimental on this pipeline (flux-sweep fidelity table above). Task 2.6 acceptance framing and Task 2.9/B8 docs should state this explicitly and cite the PtychoPINN-CI manuscript. The real_imag decoder (Task 2.3) is a **prerequisite** for the scaling — FT-linearity is what makes intensity quadratic in `s1, s2` — so the two features are coupled, not independent options. Three items remain open reconciliation follow-ups rather than blockers (see below): the probe-normalization convention, the `(s1, s2)` sign identifiability, and the one sub-threshold dead_leaves arm.

---

## Open items

- **Probe-normalization reconciliation:** this pipeline normalizes the probe at inference, so varpro-ON output lands in count-amplitude units rather than the manuscript's `~1`; reconciling requires adopting the manuscript's un-normalized-probe inference convention (Sec 2.2).
- **`(s1, s2)` sign identifiability:** the relative sign between `s1` and `s2` is data-fitted from a physics-only-trained network with no phase-sign supervision; whether a future training or gauge-fixing change should constrain this sign is an open design question, not yet a defect to fix.
- **`gs2_neither_n128` dead_leaves sub-threshold:** the one gate failure (0.482) in the dead_leaves cross-check — a difficulty floor from dead_leaves' weak amplitude contrast at N=128 without probe weighting, not reproduced on lines at the same arm (0.740, PASS); resolved by probe weighting (`gs2_probe_trainable_n128` 0.634, PASS) but worth flagging if a future gate tightens its threshold.

---

## Representation/scaling extension (Task E4)

Composite comparison figures + a merged metrics table were generated by `scripts/studies/compose_varpro_comparison_grid.py` from Task E3's outputs (`.superpowers/sdd/ext/task-E3-report.md`); source data are the `ext_matrix`/`matrix_lines`/`ext_fluxsweep` harness artifacts under `.artifacts/varpro_ablation/`, not re-run. Full merged table: `.artifacts/varpro_ablation/composite/combined_metrics.md` (`.json` alongside it; both git-ignored).

**Probe-normalization caveat (amendment #14):** on this pipeline the probe is normalized at inference, so varpro-ON output (`repr_*`'s `probe_varpro` variant, `hybres_gs1_both`, every CNN-dyad `*_both` row, and Axis B's `_on` series) is in count-amplitude units, not the manuscript's `~1` normalized convention. Every amplitude panel in the three figures below is **gauge-quotiented** (`diagnose_placement.gauge`, the least-squares global complex scalar `α=<r,t>/<r,r>`) against the truth object before display or NCC — this divides out `c_A`/rotation but not the `s1≠s2` real/imag anisotropy (per the sign-flip finding above). The real-imag scatter column (Axis A) is the one exception: it plots the **raw**, un-gauged canvas per the Task E4 spec, so its absolute position/scale reflects each arm's own count-unit gauge, not a truth-relative one. **Gate-methodology validation (I2, `task-R3-report.md`):** the per-axis gate-path fidelity numbers cited below (`recon_quality_gate.py`'s direct per-patch placement + gauge-quotient, the same validated methodology used earlier in this document) were re-validated on-branch during R3 against the `cnn_ri` flux checkpoint — gate canvas |amp| NCC 0.8651 reproduces the eval log's own gauge-quotiented FIDELITY table almost exactly (0.8651 OFF / 0.8653 ON) — so this is not an untested or skipped anchor check.

**Axis A — representation (`repr_ampphase` vs `repr_realimag`, `probe_varpro`).** Figure: `.artifacts/varpro_ablation/composite/axis_a_representation.png`. Citing E3's numbers verbatim: `repr_realimag` phase MAE is roughly half of `repr_ampphase`'s (0.095 vs 0.174), and RI's decoder-texture channels are balanced (s1=0.028, s2=0.055, both substantial) while AP's channels are unequal but not collapsed — its weaker channel is attenuated to roughly 37% of the stronger (s1=0.029, s2=−0.011, |s2/s1| ≈ 0.37), not near zero — this is the manuscript's Fig 3 / Sec 3.2 mechanism (Vong, Henke, Hoidn, Mehta, Shapiro, Hexemer, Schwarz, "Contrast-invariant deep ptychography neural networks"). Visually, the rendered phase panels support this: AP's gauge-quotiented phase panel shows speckled, high-frequency texture that does not track the truth object's own low-contrast phase field, while RI's phase panel is close to uniformly dark, matching truth's mostly-flat phase background — consistent with RI's lower phase MAE. The raw-canvas real-imag scatter (unit-circle overlay) shows both arms as compact clouds well inside the unit circle (both far below 1 in these count-amplitude units, as expected under the probe-normalization caveat); AP's cloud sits closer to the real axis (thinner imaginary spread) than RI's, a weaker but directionally consistent echo of the same channel-balance story — the per-patch `s1/s2` numbers above remain the primary, harness-computed evidence for that finding, not this canvas-level scatter. Gate-path amplitude fidelity (`task-R3-report.md`) is high for both arms — `repr_ampphase` 0.8751, `repr_realimag` 0.8621 canvas |amp| NCC — confirming both CNN representations reconstruct the object well; phase MAE remains the more sensitive Axis-A discriminator between them.

**Corrected-units note (`task-ortho-fix-report.md`):** the quoted `s1`/`s2` values above were fitted before `ptycho_torch/reassembly.py`'s VarPro basis FFTs carried `norm='ortho'`; both channels are affected by the identical missing-normalization factor, so the fix rescales every `s1`/`s2` here by the same ×32 (`repr_realimag` s1≈0.90/s2≈1.76; `repr_ampphase` s1≈0.93/s2≈−0.35 in corrected units) — the ratio `|s2/s1|` and the ordering/degeneracy conclusion above are unchanged (a uniform rescale cancels in a ratio).

**Axis B — dynamic scaling across flux (`cnn_ri` and `hybres`).** Figure: `.artifacts/varpro_ablation/composite/axis_b_flux_scaling.png`; SCALE tables parsed directly from `.artifacts/varpro_ablation/ext_fluxsweep/{cnn_ri,hybres}_eval.txt` and cross-checked against E3's transcription. `c_A = √(s1²+s2²)` tracks the √(mean-count) reference line closely for **both** generators across all three flux decades (cnn_ri 0.159/1.0/10.015 vs the ideal 0.100/1.0/10.000, approximate only at the lowest, noisiest flux; hybres near-exact at 0.100/1.0/9.998) — extending the Phase-1 CNN-only √flux result to hybrid_resnet. `c_φ` is visually flat for hybres (~40–43° across all flux) and flat above the lowest flux point for cnn_ri (~60° at mean=100/10000, rising to 73° only at mean=1, the low-count-noise point) — flux-invariant phase contrast for both. `|O|_on` scales with flux while `|O|_off` stays flat for both generators, in count-amplitude units (the probe-normalization caveat above applies to every `_on` point here). As of the R3 regeneration, the eval logs also report a measurement-domain (Fourier) error metric (relative L2 between forward-simulated and measured diffraction amplitude): at mean=100, `cnn_ri` meas_err is 0.8920 ON / 2.9155 OFF and `hybres` is 0.8968 ON / 2.9730 OFF (`task-R3-report.md`) — varpro-ON substantially reduces measurement-domain error relative to OFF at adequate flux for both generators, though this metric is not yet plotted in the figure above.

**Axis C — hybrid_resnet neither-vs-both, merged with the Phase-1 CNN gs1 dyads.** Figure: `.artifacts/varpro_ablation/composite/axis_c_hybres_dyads.png`. Per E3: `hybres_gs1_both` (probe weighting + `rectangular_scaled` + inference varpro) improves amp/complex MAE over `hybres_gs1_neither` (amp_mae 0.644 vs 1.068, complex_mae 0.702 vs 1.581), but this is a metric gain on top of a weak underlying reconstruction — `hybres_gs1_both`'s raw canvas amplitude std is 0.0014, and the varpro solve applies a very large `s1=131.65` to compensate — inflating a near-degenerate canvas rather than reflecting a genuinely better image (gauged amplitude fidelity only ~0.055–0.08 of truth; `audit-numerics.md` CHECK 6 and the R3 gate NCC in `task-R3-report.md`). The merged figure makes this concrete rather than just numeric: both `hybres_gs1_*` rows render as near-black gauge-quotiented amplitude and salt-and-pepper-noise phase panels — visually uninformative, with no filament structure resembling truth — while every CNN-dyad row (`gs1_frozen`/`gs1_trainable`, both `_neither` and `_both`) renders a clean, correctly-placed reconstruction closely matching the truth amplitude panel. hybrid_resnet's Axis-B flux-sweep fidelity corroborates this at the canvas level (gauged amp-NCC ≈0.086, vs cnn_ri's ≈0.865) — the E4 rendering is the first place both the CNN comparator and the weak hybrid_resnet output are shown side by side on the same truth-relative color scale. Gate-path fidelity (`task-R3-report.md`) is uniformly low for both hybres arms — `hybres_gs1_neither` 0.08031, `hybres_gs1_both` 0.0783 canvas |amp| NCC, both well under the >0.6 recon-quality gate threshold used elsewhere in this document — confirming the merged figure's near-black/salt-and-pepper impression numerically. The Phase-1 `gs1_frozen`/`gs1_trainable` dyad rows have no comparable gate value here: their checkpoints are not loadable cross-branch on `fno-stable`, so only the raw MAE comparison above applies to them.

**Corrected-units note (`task-ortho-fix-report.md`):** `s1=131.65` was fitted before `ptycho_torch/reassembly.py`'s VarPro basis FFTs carried `norm='ortho'`; in corrected units `s1≈4212.8` (×32) — an even larger compensating scale on the same near-degenerate canvas, so the "inflating a near-degenerate canvas" reading is unchanged (if anything strengthened). The gate NCC and gauge-quotiented fidelity numbers cited here are exactly invariant to this fix (the gauge's complex least-squares scalar absorbs a uniform rescale). The `amp_mae`/`complex_mae` comparison against `hybres_gs1_neither` is *not* gauge-invariant (`align_global_phase` only phase-aligns, it does not rescale) and `hybres_gs1_neither` (`varpro_scaling=False`, `s1=s2=1` identity) never passes through the buggy FFT at all — so this specific MAE comparison spans the bug boundary and was not re-verified numerically post-fix; the qualitative "improves over `neither`" conclusion is not shown to flip, but the exact MAE margin should be re-measured before being cited as a precise number.

**Training-conditions caveat (pre-`8b3d7a01` dataloader fix).** All extension training runs behind Axes A–C above — both `repr_*` arms, `hybres_gs1_neither`, and both flux checkpoints (`cnn_ri`, `hybres`; amplitude mode) — predate the `8b3d7a01` fix for the `82da7796` regression (a per-sample scaling collapse plus an incorrect probe reshape in the amplitude-mode training dataloader; `.superpowers/sdd/ext/bisect-report.md`). Because every arm was trained under the same (regressed) dataloader conditions, the comparative conclusions drawn above remain internally consistent. Absolute reconstruction quality — `hybrid_resnet`'s weak recon in particular — may improve if these arms are retrained under the fixed dataloader; retraining was deliberately deferred rather than folded into this pass.

### Aligned-regime rerun (2026-07-02): hybrid_resnet recovers; rectangular_scaled is the failing knob

**Methodology.** Axis C's `hybres_gs1_*` arms above trained under a misaligned regime (`torch_loss_mode='poisson'`, lr 1e-3 with no scheduler, N=64 lines/fly001 data) rather than the config the trusted integration gate uses (`tests/torch/test_grid_lines_hybrid_resnet_integration.py::test_grid_lines_hybrid_resnet_metrics`: unsupervised PINN, MAE amplitude objective, lr 2e-4 + ReduceLROnPlateau, N=128 grid-lines/Run1084 data, 5 epochs, seed 3). To separate "hybrid_resnet cannot learn this task" from "this ablation ran hybrid_resnet under the wrong conditions," the ablation was rerun through the integration script itself (`scripts/studies/grid_lines_torch_runner.py`) with the integration test's flags copied verbatim and exactly one ablation knob varied per arm — the same N=128/gridsize=1/Run1084 dataset identity as the integration gate, not the Axis A–C extension datasets above.

| arm | knobs vs control | amp MAE | phase MAE | amp SSIM | phase SSIM |
|---|---|---|---|---|---|
| `neither` (control = integration cmd verbatim) | — | 0.0781 | 0.1304 | 0.8921 | 0.9615 |
| `weight_only` | `--training-patch-weighting probe` | BIT-IDENTICAL to `neither` (all four metrics, full precision) | | | |
| `rect_only` | `--physics-forward-mode rectangular_scaled` | 1.5609 | 1.3093 | 0.0035 | 0.0131 |
| `both` | probe weighting + `rectangular_scaled` | BIT-IDENTICAL to `rect_only` | | | |
| `ampphase_out` | `--output-mode amp_phase` | 0.0826 | 0.1487 | 0.8870 | 0.9495 |

The control (`neither`) reproduces the trusted integration run's amp MAE bit-for-bit (`0.0780567154288292`) — the rerun is aligned by construction, not by post-hoc tuning.

**Resolving the training-conditions caveat above.** The caveat paragraph immediately preceding this subsection records that every Axis A–C extension arm — including `hybres_gs1_neither`/`hybres_gs1_both` — trained under dataloader conditions that predate the `8b3d7a01` fix for the `82da7796` regression. This aligned rerun postdates that fix and shows **hybrid_resnet recovers** when run under the integration-aligned regime: the earlier near-black gauge-quotiented amplitude / salt-and-pepper phase / canvas `|amp|` NCC ~0.08 reading for `hybres_gs1_*` was a **training-regime artifact** of the misaligned ablation harness (wrong loss mode, wrong LR/scheduler, wrong N, wrong dataset), not an architecture failure of hybrid_resnet itself, and not solely attributable to the pre-fix dataloader. `rectangular_scaled` remains a genuine, reproducible training-time incompatibility with the MAE/PINN objective (~20x amp MAE, SSIM collapse toward 0) and is the real driver of any degenerate hybrid_resnet+`rectangular_scaled` result.

**gs1 weighting-inertness.** Training-side `training_patch_weighting` is structurally inert at gridsize 1: the reassembly-weighting dispatch in `ForwardModel.forward` is gated `if self.object_big:` (`ptycho_torch/model.py:1584`), and the grid-lines runner hard-codes `object_big=False` (`scripts/studies/grid_lines_torch_runner.py:1159`), so the `'probe'` branch is unreachable for this pipeline. Proven bit-for-bit twice (`weight_only` == `neither`; `both` == `rect_only`, all four metrics, full float precision). Weighting/VarPro remain live ONLY at inference-time reassembly (overlapping test patches).

**Pointers.** Driver script: `scripts/studies/aligned_hybres_ablation_driver.sh` (reruns all 5 arms sequentially against the exact integration config). Executable gate: `tests/torch/test_grid_lines_hybrid_resnet_aligned_ablation.py` (marker `grid_lines_hybrid_resnet_aligned_ablation`, GPU, ~10 min), encoding the `rectangular_scaled` collapse and the `training_patch_weighting` inertness as pass/fail assertions. Knowledge-base entry: `docs/findings.md` HYBRES-ALIGN-001. Artifacts: `.artifacts/varpro_ablation/ext_matrix_aligned/` (per-arm `runs/pinn_hybrid_resnet/{metrics,config,invocation}.json` + `visuals/`; per-arm `.log`/`.exit`; `data/` hardlinks). Inference-side companion: the inference-only 2x2x5 variant sweep (uniform/probe x varpro on/off x all 5 arms) is complete at `.artifacts/varpro_ablation/ext_matrix_aligned/variants_summary.md` — probe patch-weighting beats uniform on the healthy arms (amp MAE 0.0996 vs 0.1402, corr 0.9856 vs 0.9744), while the varpro rows remain scale-unreliable pending the units backlog (`docs/findings.md` REASSEMBLY-BRIDGE-001).

## 20-epoch rerun (2026-07-08)

Executed per `docs/plans/2026-07-06-aligned-ablation-20epoch-rerun.md` as amended (comparison anchor = the corrected re-baseline `ext_matrix_aligned_rebase/`, NOT the original 5-epoch summaries; evaluation pipeline = post-`d5f40106`/`22d77509`). Training: parameterized driver (`79787e7e`, smoke-validated epochs plumbing), 5 arms × 20 epochs, seed 3, all arms exit 0 with `"epochs": 20` in invocation records. Scoring: repo-ported harness `scripts/studies/aligned_ablation_variant_grid.py` (`18ff821e`), which passed a mandatory reproduction gate before use — re-scoring the 5-epoch checkpoints reproduced all 176 committed re-baseline metric fields and 16 canvas arrays within 1e-6 relative (max 5.4e-7). Montage: `scripts/studies/aligned_ablation_montage.py` (`080b1185`) → `ext_matrix_aligned_20ep/montage/montage_{amp,phase}.png`.

### Comparison table (bridged variants; 5ep = re-baselined anchor, 20ep = this rerun)

| arm | variant | amp_mae 5ep→20ep | amp_corr 5ep→20ep | phase_mae 5ep→20ep |
|---|---|---|---|---|
| neither | uniform_novarpro | 0.1374 → 0.1086 | 0.9743 → 0.9864 | 0.1290 → 0.0630 |
| neither | probe_novarpro | 0.1008 → 0.0701 | 0.9850 → 0.9935 | 0.1286 → 0.0583 |
| weight_only | (both novarpro) | ≡ neither (bit-exact) | ≡ | ≡ |
| ampphase_out | uniform_novarpro | 0.1924 → 0.1238 | 0.9488 → 0.9800 | 0.1466 → 0.0987 |
| ampphase_out | probe_novarpro | 0.2339 → 0.0904 | 0.9652 → 0.9878 | 0.1626 → 0.1094 |
| cnn | uniform_novarpro | 0.4516 → 0.5301 | 0.9501 → 0.9569 | 0.1782 → 0.1934 |
| cnn | probe_novarpro | 0.3371 → 0.3037 | 0.9618 → 0.9835 | 0.1761 → 0.1913 |

`*_varpro` rows: ~2.62 amp_mae on every arm at both epochs — report-only per the standing amplitude-mode scale-unreliability caveat (497d1f69 gates the count-units fold to rectangular_scaled); no scale conclusions drawn.

Native (unbridged) final metrics at 20ep: neither/weight_only amp 0.0384 / phase 0.0824; ampphase_out 0.0419 / 0.0745; cnn 0.0854 / 0.1875; l2match_on 0.2072 / 0.1223.

### Hypothesis verdicts

- **H2 (training weighting inert at gs1): HOLDS, bit-exact.** `weight_only` ≡ `neither` byte-identical in native metrics.json AND through the full bridged variant grid at 20 epochs. The structural `object_big=False` guard explanation stands.
- **H3 (inference probe weighting > uniform): HOLDS and strengthens.** Probe beats uniform on every arm at 20ep (neither 0.0701 vs 0.1086; ampphase 0.0904 vs 0.1238; cnn 0.3037 vs 0.5301). The single 5-epoch exception (ampphase: probe 0.2339 worse than uniform 0.1924) FLIPS to the expected ordering with training — it was a convergence artifact, not a real reversal. Absolute margin holds ~0.035–0.04 on hybres arms while baselines improve.
- **H4 (amp_phase handicap): mostly slower convergence.** Gap to `neither` shrinks 0.055 → 0.015 amp_mae (uniform_novarpro); native gap 0.0419 vs 0.0384. A small residual handicap remains at 20ep; no grid-texture-driven divergence.
- **H5 (cnn vs hybrid_resnet): architecture gap persists; cnn does not catch up.** Native amp MAE 2.2× (0.0854 vs 0.0384); bridged probe_novarpro 0.3037 vs 0.0701. amp_corr narrows but does not close (0.9835 vs 0.9935). cnn's uniform_novarpro amp_mae worsens (0.4516 → 0.5301) while its corr improves — the known cnn bridged-amplitude-scale caveat; prefer amp_corr for the H5 verdict, per the plan.
- **H6 (l2match_on): materially different amplitude calibration in this regime.** Native amp MAE 0.2072 vs neither's 0.0384 (5.4×); phase 0.1223 vs 0.0824. Caveat: `torch_mae_pred_l2_match_target=on` redefines the prediction's native scale, so this is a calibration observation, not directly a reconstruction-quality verdict; the arm was excluded from the bridged grid by design. The legacy `off` contract remains the better-calibrated choice for this aligned N=128 gs1 MAE regime as measured.

**No 5-epoch conclusion flips.** HYBRES-ALIGN-001 stands unchanged (H3's apparent ampphase reversal at 5ep resolving toward the expected ordering strengthens, not contradicts, the recorded conclusions). Artifacts: `.artifacts/varpro_ablation/ext_matrix_aligned_20ep/` (git-ignored), summary tables in each arm's `variants_summary.json` + root `variants_summary.md`.
