# Main to fno-stable Physics Reconciliation Backlog

**Date:** 2026-06-29
**Branch context:** `fno-stable` after merge commit `13978cf1` (`Merge main into fno-stable`)
**Main baseline inspected:** merge parent `5bd07399eddf12da07f244c8dcc9d764a0776a7c`
**Purpose:** Inventory numerical behavior differences between `main` and current `fno-stable` around CNN/FNO output representation, C-channel patch overlap, probe weighting, and intensity scaling; define backlog items needed to reconcile them without silently changing existing FNO/hybrid behavior.

**Status (2026-07-02):** B1, B2, B3, B5, B6, B7, B8 are COMPLETE; B4 is INTENTIONALLY
DEFERRED (both denominator rules gated and tested, no default change forced). All items
below are opt-in reconciliation modes, not new defaults — FNO/hybrid and current CNN
`amp_phase` behavior remain byte-unchanged. See each `### B*` section under "Backlog
Items" for evidence commit SHAs, and "Open Items Carried Forward (post-B8, 2026-07-02)"
for the two remaining open follow-ups plus one resolved-and-closed item. The Executive
Summary, Difference Inventory, and Implications sections below are preserved as the
original problem statement this backlog was written to close.

## Source Surfaces

- `ptycho_torch/model.py`
- `ptycho_torch/helper.py`
- `ptycho_torch/config_params.py`
- `ptycho_torch/config_factory.py`
- `ptycho_torch/reassembly.py`
- `tests/torch/test_generator_adapter.py`
- `tests/torch/test_model_output_modes.py`
- `tests/torch/test_fno_generators.py`
- `docs/findings.md`
- `docs/workflows/pytorch.md`

## Executive Summary

`main` and current `fno-stable` are not numerically equivalent for either `C=1` or `C>1`.

For `C>1`, the biggest difference is training-time object overlap: `main` used probe-weighted patch reassembly in the unsupervised forward model, while current `fno-stable` uses the plain central-mask/count-normalized reassembly helper. For `C=1`, inter-channel averaging degenerates, but numerical differences remain because `main` used rectangular real/imag CNN outputs plus rectangular intensity scaling, while current `fno-stable` routes CNN through the polar `amp_phase` adapter by default.

Probe-weighted and VarPro-style logic now exists in current `fno-stable` inference reassembly, but that does not restore `main`'s training-time forward-model numerics.

Stage boundary: both `main` and current `fno-stable` have an inference reassembly path that calls `model.forward_predict(...)`, stitches predicted patches into a canvas, and solves/applies real/imag scale factors after prediction. "Posthoc" in this backlog refers to that inference-stage operation: it can change the final assembled reconstruction, but it is outside the autograd/loss path. The behavioral gap is that `main` also used probe-weighted overlap and `RectangularScaledDiffraction` inside `ForwardModel.forward(...)`, while current `fno-stable` exposes `InferenceConfig.patch_weighting` and `InferenceConfig.varpro_scaling` only through inference reassembly.

## Difference Inventory

| ID | Surface | `main` behavior | current `fno-stable` behavior | Numerical impact | Reconciliation need |
| --- | --- | --- | --- | --- | --- |
| D1 | CNN output representation | Unsupervised CNN emits two tensors interpreted as real and imaginary components, combined by `real + 1j * imag`. | CNN falls through the shared adapter default `generator_output="amp_phase"` and combines two decoder outputs as `amp * exp(1j * phase)`. | High for `C=1` and `C>1`; changes object parameterization, gradients, loss surface, and output range. | Add opt-in CNN rectangular real/imag mode without changing existing CNN default. |
| D2 | CNN shared decoder | `ModelConfig.use_shared_decoder` exists; `Autoencoder` can use `Decoder_shared`, returning `2*C` channels split into real/imag. | Shared decoder classes/config are absent from the current `Autoencoder`; only separate amp/phase decoders exist. | Medium to high when reproducing `main` CNN runs; architecture differs even before physics. | Port shared decoder components and make them opt-in. |
| D3 | Output routing abstraction | No generalized `generator_output_mode`; main's unsupervised CNN path is hardcoded rectangular. | General adapter supports `amp_phase`, `amp_phase_logits`, and FNO/hybrid `real_imag` tensors `(B,H,W,C,2)`, but not CNN tuple real/imag. | High if attempting to reproduce main with CNN; current `real_imag` branch expects one tensor, not `(real, imag)`. | Extend adapter to accept tuple real/imag for CNN while preserving FNO/hybrid tensor path. |
| D4 | Training-time C-channel reassembly | `ForwardModel` calls `reassemble_patches_position_real_probe(..., probe=probe, use_probe_weights=True)`. | `ForwardModel` calls `reassemble_patches_position_real(...)` with central prototype mask/count normalization. | High for `C>1`; lower but nonzero for `C=1` due edge/support differences. | Restore probe-weighted training reassembly behind config. |
| D5 | Plain reassembly normalization | Main's plain helper clamps denominator to `min=1.0` and multiplies by boolean mask. | Current helper uses `non_zeros + 0.001` and does not hard-zero outside mask. | Medium; affects boundary/support pixels and parity tests. | Decide whether TF-style epsilon or main-style clamp is authoritative, then gate if both are needed. |
| D6 | Forward diffraction scaling | Main uses `RectangularScaledDiffraction` with per-dataset trainable `s1/s2` in autograd path and normal-equation solve utilities for non-autograd path. | Current forward model uses `pad_and_diffract(...)` then `IntensityScalerModule.inv_scale(...)`; VarPro exists in inference reassembly, not training forward. | High for `C=1` and `C>1`; changes predicted intensity/rate seen by loss. | Port rectangular scaling as opt-in training-forward mode, or explicitly reject main parity as a goal. |
| D7 | Probe mask defaults | Main `probe_mask=None` effectively means no mask unless provided. | Current config default is `probe_mask=False` with explicit mask resolution semantics and prior finding `PROBE-MASK-DEFAULT-001`. | Low to medium depending run; can change probe support and metric gates. | Keep current default unless reproducing main exactly; document compatibility setting. |
| D8 | Inference-time stitching | Main had an inference reassembly path that called `forward_predict(...)`, accumulated VarPro basis terms, stitched patches with optional probe weighting, solved `s1/s2`, and applied those constants to the final canvas. Separately, main also wired probe-weighted overlap and rectangular scaling into `ForwardModel.forward(...)`. | Current `reassembly.py` keeps the inference-stage stitching/scaling surface and adds explicit `InferenceConfig.varpro_scaling`, but the training forward model no longer uses main's probe-weighted reassembly or rectangular scaled diffraction. | Inference controls can improve the final stitched canvas but do not change training loss, gradients, or per-batch forward numerics. | Keep inference knobs separate from training-forward knobs; reconciliation requires tests for both stages. |
| D9 | Generator architectures | Main only has CNN generator shim. | Current branch has CNN plus FNO/hybrid/hybrid_resnet/etc. registry. | Reconciliation risks accidentally changing FNO/hybrid behavior while fixing CNN main parity. | Scope main-parity knobs to CNN or explicit `physics_forward_mode`. |
| D10 | Supervised mode | Main supervised path still combines autoencoder outputs as amp/phase. | Current supervised path uses the same generalized adapter and can accept alternate generator output modes. | Medium for supervised CNN parity if rectangular mode leaks into supervised. | Keep CNN rectangular mode unsupervised-only unless explicitly requested. |

## C=1 Implications

`C=1` removes the multi-channel averaging problem but does not remove numerical differences:

- Representation still differs: rectangular real/imag in `main` versus polar amp/phase in current CNN default.
- Forward scaling still differs: `RectangularScaledDiffraction` in `main` versus current `pad_and_diffract + inv_scale`.
- Reassembly support still differs when `object_big=True`: probe-weighted support in `main` versus central-mask support in current `fno-stable`.
- If `object_big=False`, the reassemble/extract step is skipped, but representation and scaling differences remain.

## C>1 Implications

For `C>1`, all `C=1` differences apply, plus:

- Main's training-time reassembly weights patch contributions by `sum_p |probe_p|^2`.
- Current training-time reassembly weights by a central binary prototype mask, not probe intensity.
- Current inference-time `patch_weighting="probe"` does not affect the training loss path.
- Generator shape support for `C>1` is not equivalent to validated grouped-neighbor physics correctness.

## Backlog Items

### B1: Add CNN Rectangular Output Mode

**Status:** COMPLETE (2026-07-02). Implemented as `ModelConfig.cnn_output_mode:
Literal['amp_phase', 'real_imag'] = 'amp_phase'` under subagent-driven-development Task
2.3, commit `b33c7975`. Reviewed clean (opus reviewer verified all named risks: default
byte-identical, supervised path unaffected, FNO/hybrid `real_imag` tensor path untouched,
CNN tuple `(real, imag)` path new). Evidence tests: `tests/torch/test_model_output_modes.py`,
`tests/torch/test_generator_adapter.py`. See `docs/findings.md#RECTANGULAR-SCALED-001` for
the decoder-box representability caveat this mode carries.
**Priority:** P0 if main parity is required; P1 otherwise
**Files likely touched:**

- `ptycho_torch/config_params.py`
- `ptycho_torch/model.py`
- `ptycho_torch/generators/cnn.py`
- `tests/torch/test_model_output_modes.py`
- `tests/torch/test_generator_adapter.py`

**Work:**

- Add an explicit CNN-scoped config, e.g. `cnn_output_mode: Literal["amp_phase", "real_imag"] = "amp_phase"`.
- Do not reuse `generator_output_mode` for CNN by default, because its current default is `real_imag` for FNO/hybrid and would silently alter CNN behavior.
- In `CnnGenerator.build_model(...)`, pass `generator_output=pt_model_config.cnn_output_mode`.
- Extend `_predict_complex_patches(...)` so `generator_output == "real_imag"` accepts both:
  - tuple `(real, imag)` with shape `(B,C,H,W)` from CNN,
  - tensor `(B,H,W,C,2)` from FNO/hybrid.

**Acceptance:**

- Existing CNN default still uses amp/phase and passes current tests.
- Opt-in CNN real/imag returns `torch.complex(real, imag)`.
- FNO/hybrid real/imag adapter behavior is unchanged.

### B2: Port Main's CNN Shared Decoder as Opt-In

**Status:** COMPLETE (2026-07-02). Implemented as `ModelConfig.use_shared_decoder: bool =
False` under Task 2.4, commit `bc23d657`. Reviewed clean (opus reviewer verified the
non-verbatim `Decoder_shared` is forced by fno-stable architectural divergence, not a
merge-over-rewrite violation; `FeatureRefinementBlock` ported byte-verbatim; Task 2.3
per-mode activation gating reproduced exactly via an `Identity` head + post-split
activations; default `use_shared_decoder=False` provably untouched, 17/17 pins green).
**Priority:** P1
**Depends on:** B1
**Files likely touched:**

- `ptycho_torch/config_params.py`
- `ptycho_torch/model.py`
- `tests/torch/test_model_output_modes.py`

**Work:**

- Port `FeatureRefinementBlock` and `Decoder_shared` from main parent `5bd07399`.
- Add `use_shared_decoder: bool = False` to `ModelConfig`.
- Update `Autoencoder` to choose either:
  - separate `Decoder_amp` + `Decoder_phase`, or
  - shared decoder returning `2*C_out` channels split into two tensors.
- Treat the shared decoder's two tensors according to `cnn_output_mode`.

**Acceptance:**

- `use_shared_decoder=False` is bitwise or close numerically to current architecture at initialization shape-contract level.
- `use_shared_decoder=True, cnn_output_mode="real_imag"` returns two `(B,C,N,N)` tensors interpreted as real/imag.
- Tests assert output shapes for `C=1` and `C=4`.

### B3: Restore Probe-Weighted Training Reassembly Behind Config

**Status:** COMPLETE (2026-07-02). Implemented as `ModelConfig.training_patch_weighting:
Literal['central_mask', 'probe', 'uniform'] = 'central_mask'` under Task 2.5, commit
`aec6329a`. Reviewed clean (default byte-identical at `model.py:1418-1421`; plug dispatch
inside `ForwardModel.forward(...)` verified; 3 coverage tests substantive incl.
offset/seam predicted values). The `'probe'` value dispatches to
`sum_p |probe_p|^2`-weighted reassembly before loss computation, satisfying the
"acceptance cannot be satisfied by only setting `InferenceConfig.patch_weighting`"
requirement below. Full cross-branch parity gate (fixture comparison) re-assigned to
Task 2.8 / B6 per the ledger's amendment #16 re-sequencing.
**Priority:** P0 for C>1 main parity
**Files likely touched:**

- `ptycho_torch/config_params.py`
- `ptycho_torch/helper.py`
- `ptycho_torch/model.py`
- `tests/torch/test_reassembly_multi_patch_parity.py`
- New focused tests under `tests/torch/test_training_forward_probe_weighted_reassembly.py`

**Work:**

- Reintroduce or rewrite `reassemble_patches_position_real_probe(...)` using structured tensor ops.
- Add a training-forward config such as `training_patch_weighting: Literal["central_mask", "probe"] = "central_mask"`.
- In `ForwardModel.forward(...)`, choose the reassembly helper based on this config.
- Keep inference `InferenceConfig.patch_weighting` separate from training-forward weighting.

**Acceptance:**

- With `training_patch_weighting="central_mask"`, current behavior remains unchanged.
- With `training_patch_weighting="probe"`, overlapping patches are weighted by `sum_p |probe_p|^2`.
- Acceptance cannot be satisfied by only setting `InferenceConfig.patch_weighting`; the probe-weighted path must execute inside `ForwardModel.forward(...)` before loss computation.
- A deterministic two-patch test shows probe weighting reduces edge corruption versus central-mask/uniform weighting.
- C=1 path has stable support behavior and no division-by-zero edge artifacts.

### B4: Decide and Gate Reassembly Denominator Semantics

**Status:** INTENTIONALLY DEFERRED (2026-07-02). Both denominator rules now coexist and
are individually pinned by tests: fno-stable's TF-parity style (`non_zeros + 0.001`, no
hard mask, `TORCH-REASSEMBLY-NORM-001`) remains the default `'central_mask'`/`'uniform'`
behavior, and `main`'s probe-weighted path (Task 2.5 / B3, `training_patch_weighting='probe'`)
is gated behind its own config value. No single default was chosen because no downstream
consumer currently requires one over the other; revisit if a consumer needs unification.
This satisfies B4's "if both are useful, expose an internal mode rather than changing
default silently" fallback explicitly, not as an unresolved gap.
**Priority:** P1
**Depends on:** B3
**Files likely touched:**

- `ptycho_torch/helper.py`
- `tests/torch/test_reassembly_multi_patch_parity.py`

**Work:**

- Decide which denominator rule is authoritative for the default:
  - current TF-parity style: `non_zeros + 0.001`, no hard output mask,
  - main style: `clamp(min=1.0)` plus hard mask.
- If both are useful, expose an internal mode rather than changing default silently.

**Acceptance:**

- Unit tests pin denominator behavior in uncovered, single-covered, and overlapped pixels.
- Existing parity tests explain which mode they exercise.

### B5: Port Rectangular Forward Scaling as an Opt-In Physics Mode

**Status:** COMPLETE (2026-07-02). Implemented as `ModelConfig.physics_forward_mode:
Literal['amplitude', 'rectangular_scaled'] = 'amplitude'` under Task 2.6, commit
`92ae03a8` (reviewed APPROVED with concerns resolved). `RectangularScaledDiffraction`
ported byte-verbatim from `main` plus a `requires_grad(rect_s1s2_trainable)` patch;
`RectangularPoissonLoss`/`RectangularMAELoss` replicate `main`'s intensity-domain loss
semantics including the MAE re-square quirk (deliberate, documented); probe-mask
semantics tested (default + masked + no-op guard); default `'amplitude'` path byte-stable
(Task 2.1 pin green). End-to-end trainability (not just forward-parity) confirmed
independently in Task 2.8 (commit `82da7796`: real training smoke, finite loss, nonzero
data-dependent gradients on all params incl. `s1`/`s2`). See
`docs/findings.md#RECTANGULAR-SCALED-001` for the corrected `s1`/`s2` physics and the
object_big padding residual this mode inherits from a pre-existing divergence.
**Priority:** P0 if exact main-forward parity is required
**Depends on:** B1
**Files likely touched:**

- `ptycho_torch/config_params.py`
- `ptycho_torch/model.py`
- `tests/torch/test_physics_scale_loss.py`
- New focused tests under `tests/torch/test_rectangular_scaled_forward.py`

**Work:**

- Port `RectangularScaledDiffraction` from main parent `5bd07399`, or factor equivalent logic from current VarPro code into a training-forward module.
- Add a config such as `physics_forward_mode: Literal["amplitude", "rectangular_scaled"] = "amplitude"`.
- Only allow `rectangular_scaled` when the complex object came from real/imag representation, or define a clear conversion if used with polar outputs.
- Do not treat `apply_varpro_canvas_scaling(...)` in inference reassembly as sufficient for this item; this backlog item is specifically about the autograd/loss path.
- Preserve current Poisson/MAE loss interfaces.

**Acceptance:**

- Default `physics_forward_mode="amplitude"` remains current behavior.
- `rectangular_scaled` path produces nonnegative predicted intensities with shape `(B,C,N,N)`.
- Deterministic synthetic test recovers known real/imag scale factors or matches main-parent output within tolerance for a frozen fixture.

### B6: Add Branch-Parity Fixture Tests

**Status:** COMPLETE (2026-07-02). Baseline fixtures (5 cases:
`c{1,4}_big{F,T}_{probe,uniform}.npz`) cherry-picked from the `varpro-ablation` branch's
Task 1.4 work (commits `a8cd0920` + fix `9681b8ce`) onto fno-stable, then wired to the
full mode matrix (`training_patch_weighting` + `rect_s1s2_trainable` + `physics_forward_mode`)
and gated in Task 2.8 (`tests/torch/test_cross_branch_rectangular_parity.py`, commit
`95bcde03`, APPROVED). Step 1 results: `C=1, object_big=False` bit-exact forward+losses
under fno-stable's **real** padding (`rtol=0, atol=0`); `C=1, object_big=True` bit-exact
only under a matched-padding monkeypatch (residual documented, see
`docs/findings.md#RECTANGULAR-SCALED-001`). Two-knob forcing verified load-bearing
(amplitude-mode max error 77.99 -> rectangular-mode 0.0 on the same fixture). Full bundle:
57 passed / 1 skipped / 11 xfail (=bigT padding residual, expected) / 1 xpass. Default
current FNO/hybrid tests continue to pass (Task 2.1 pin green throughout).
**Note (Task 2.9):** the amplitude-oracle `tests/torch/test_forward_parity_fixtures.py`
(cherry-picked from `varpro-ablation` Task 1.4, rebuilt configs default to
`physics_forward_mode='amplitude'` and ran against fixtures frozen in rectangular
mode) was deleted as superseded by the rectangular-forcing tests above; its unique
`rect_s1s2_trainable` `requires_grad` coverage was folded into
`test_rectangular_scaled_forward.py`. The "B6 tests pass" claim refers to the
rectangular-forcing suite, not the deleted amplitude-oracle module.
**Priority:** P0 before claiming reconciliation
**Depends on:** B1, B3, B5
**Files likely touched:**

- New fixture/test module under `tests/torch/`
- Optional small fixture artifact under `tests/fixtures/` if size is acceptable

**Work:**

- Build deterministic tiny fixtures for:
  - `C=1, object_big=True`,
  - `C=1, object_big=False`,
  - `C=4, object_big=True`.
- Compare current branch modes against frozen expected tensors from main-parent semantics.
- Avoid relying on trained checkpoints.

**Acceptance:**

- Tests isolate representation, reassembly, and scaling deltas independently.
- Reconciliation modes can match main-parent forward outputs within a documented tolerance.
- Default current modes continue to pass existing FNO/hybrid tests.

### B7: Update CLI/Factory Exposure

**Status:** COMPLETE (2026-07-02). Implemented under Task 2.7, commit `99a3acf0` +
fix `8992a8bf` (re-review APPROVED). 5 config-factory silent-drop-defeat tests + 4 runner
flags (mirrored in 3 places) wired to training via `_train_with_lightning`'s repurposed
`overrides` param -> `factory_overrides` (highest precedence) -> `create_training_payload`.
Unmocked end-to-end forwarding test added; no caller regressions; `ptycho/` untouched;
defaults byte-identical (218+37 tests reproduced). Forward-looking note carried in
`progress.md`: the independent `varpro-ablation` branch solves the same knob-forwarding
problem via a different mechanism (`_build_factory_overrides` + `pt_model_overrides`
kwarg) — a design collision to reconcile only if that branch's training-plumbing is ever
merged; it is not in the current cherry-pick list, so this is latent, not active.
**Priority:** P2
**Depends on:** B1, B3, B5
**Files likely touched:**

- `ptycho_torch/config_factory.py`
- `scripts/studies/grid_lines_torch_runner.py`
- CLI tests under `tests/torch/test_grid_lines_torch_runner.py`

**Work:**

- Decide which knobs should be user-facing versus internal test-only.
- Add factory override propagation for accepted knobs.
- Add CLI flags only if needed for study workflows.

**Acceptance:**

- Factory overrides propagate without touching canonical TensorFlow config fields unless explicitly required.
- Grid-lines runner can opt into main-compatible CNN mode for controlled comparisons.

### B8: Document Compatibility Modes

**Status:** COMPLETE (2026-07-02, this document's own update, Task 2.9). Mode matrix
(current defaults / main-compatible rectangular CNN / inference-only knobs) added to
`docs/workflows/pytorch.md`; CNN `(real, imag)` tuple output-mode contract added to
`ptycho_torch/generators/README.md`; consolidated physics finding added at
`docs/findings.md#RECTANGULAR-SCALED-001` (mode replacement, corrected `s1`/`s2`
semantics, MAE re-square parity quirk, decoder-box constraint, C=1-vs-main padding
residual); this backlog's B1-B8 statuses and the open items below recorded with evidence
commit SHAs. `InferenceConfig.patch_weighting`/`varpro_scaling` are explicitly documented
as inference-only (not training-forward controls, and currently not even threaded by the
`python -m ptycho_torch.inference` CLI subprocess path) per this item's original
acceptance requirement.
**Priority:** P2
**Depends on:** B1, B3, B5
**Files likely touched:**

- `docs/workflows/pytorch.md`
- `ptycho_torch/generators/README.md`
- `docs/findings.md` if a durable known behavior is established

**Work:**

- Document current-default mode versus main-compatible mode.
- Explicitly state that inference `patch_weighting` and `varpro_scaling` are not training-forward controls, even though main also had an inference-stage stitching/scaling path.
- Record when `C=1` is still numerically different from main and why.

**Acceptance:**

- Future readers can select one of:
  - current FNO/hybrid-compatible defaults,
  - main-compatible CNN rectangular forward path,
  - inference-only probe/VarPro stitching path.

### Open Items Carried Forward (post-B8, 2026-07-02)

B1-B8 are all complete or intentionally deferred (B4) as of this document's update. Two
items surfaced during B5/B6/B8 execution remain open, and one earlier-suspected blocker
is resolved and closed out here for the record.

**OPEN — bigT REAL-padding parity (deep):** The rectangular `object_big=True` fixture
parity test (Task 2.8) matches `main`-parent forward+loss output within the registered
cross-build contract (`rtol=1e-5, atol=1e-6`) only under a `get_padded_size` monkeypatch
that forces `main`'s padding buffer. fno-stable's real, currently-shipping
`ptycho_torch/helper.py::get_padded_size` returns `buffer=0` (commit `ba3f705d`), and that
buffer is shared by both the amplitude and rectangular forward paths — it predates this
reconciliation effort and reverting it breaks the Task 2.1 regression pin. Consequently
the `object_big=True` rectangular path is verified against `main` only under matched
(non-default) padding, not fno-stable's real regime; only `object_big=False` (`c1_bigF`)
is verified against its frozen fixture under real padding within the same cross-build
contract. **Action:**
reconcile `get_padded_size`'s buffer semantics with `main`'s (without breaking the Task
2.1 pin) and re-freeze/re-verify the bigT rectangular fixtures un-monkeypatched. See
`docs/findings.md#RECTANGULAR-SCALED-001`.

**OPEN — rectangular_scaled+MAE convergence note (not a defect):** `physics_forward_mode
='rectangular_scaled'` combined with `torch_loss_mode='mae'` exhibits a
near-vanishing-gradient early-training regime, driven by `RectangularMAELoss`'s
double-square (`model.py:1730`, a deliberate verbatim-from-main parity quirk, not a bug —
see `docs/findings.md#RECTANGULAR-SCALED-001`) combined with near-zero real/imag decoder
initialization. This is a convergence-*quality* investigation item, not a blocker: the
mode trains (finite loss, confirmed nonzero data-dependent gradient flow on all
parameters including `s1`/`s2` in the Task 2.8 independent training smoke). No action
required before using `rectangular_scaled`; revisit if a study needs faster MAE-mode
convergence.

**RESOLVED (closed) — end-to-end rectangular_scaled training crash:** An earlier
end-to-end crash when training with `physics_forward_mode='rectangular_scaled'` (inline
training-dataset collate shapes: `components.py`'s `_build_dataloaders` inline path
returned un-reshaped `scaling`/`probe` tensors, producing a spurious batch axis that
collided with `RectangularScaledDiffraction`'s `unsqueeze(dim=2)`) is fixed in commit
`82da7796` (Task 2.8 Step 2 fix, independently re-reviewed: real `bs=4` multi-epoch
training smoke with no crash, finite loss, nonzero gradients on all 61 params). Bounded
to `ptycho_torch/workflows/components.py`; also a correctness improvement for the
amplitude path per `ProbeIllumination`'s `(B,C,P,H,W)` tensor contract. No further action.

## Suggested Execution Order

Actual execution order (subagent-driven-development, Task 2.1-2.9) tracked fixtures
first, then B1/B2/B3/B5 in dependency order, deferred the full B6 parity gate to after
B5 landed (amendment #16), then B7/B8. All items below are COMPLETE or INTENTIONALLY
DEFERRED (B4) as of 2026-07-02; see each `### B*` section above for evidence.

1. ~~B6 fixture design for tiny deterministic cases~~ — baseline fixtures cherry-picked
   pre-B1/B3/B5, full parity gate completed after B5 (Task 2.8).
2. ~~B1 CNN rectangular output adapter.~~ — done (Task 2.3, `b33c7975`).
3. ~~B2 shared decoder port.~~ — done (Task 2.4, `bc23d657`).
4. ~~B3 probe-weighted training reassembly.~~ — done (Task 2.5, `aec6329a`).
5. ~~B4 denominator semantics gate.~~ — intentionally deferred (both modes gated+tested).
6. ~~B5 rectangular scaled forward model.~~ — done (Task 2.6, `92ae03a8`; end-to-end
   trainability confirmed in Task 2.8, `82da7796`).
7. ~~B7 factory/CLI exposure.~~ — done (Task 2.7, `99a3acf0` + `8992a8bf`).
8. ~~B8 docs and findings update.~~ — done (Task 2.9, this document).

## Non-Goals

- Do not make `generator_output_mode="real_imag"` apply to CNN by default.
- Do not change FNO/hybrid defaults while restoring main CNN compatibility.
- Do not treat inference-time probe weighting as equivalent to training-time probe-weighted overlap.
- Do not commit generated visual or numerical artifacts; store bulky evidence under `.artifacts/`.

## Open Questions

- Should main-compatible CNN rectangular mode be a paper-facing path or only a compatibility/audit path?
- Should probe-weighted training reassembly be available for FNO/hybrid, or restricted to CNN compatibility until validated?
- Should rectangular scaling replace or coexist with the newer physics-scale loss path? — **Answered by implementation:** it replaces the amplitude scaling stack outright when opted into (`physics_forward_mode='rectangular_scaled'`); it does not coexist within a single forward pass. See `docs/findings.md#RECTANGULAR-SCALED-001`.
- Which denominator rule is the desired default for future PyTorch parity: current TF-style epsilon or main's hard-mask clamp? — **Answered by B4:** no single default was chosen; both rules are gated and tested behind `training_patch_weighting`, revisit only if a consumer needs unification.

## Minimal Verification Bundle for Reconciliation

Before marking this backlog complete, collect:

- `python -m pytest -q tests/torch/test_generator_adapter.py tests/torch/test_model_output_modes.py`
- `python -m pytest -q tests/torch/test_training_forward_probe_weighted_reassembly.py`
- `python -m pytest -q tests/torch/test_rectangular_scaled_forward.py`
- Existing reassembly regressions:
  - `tests/torch/test_inference_reassembly_parity.py`
  - `tests/torch/test_inference_reassembly_aggregation.py`
  - `tests/torch/test_reassembly_multi_patch_parity.py`
  - `tests/torch/test_reassembly_sign_parity.py`
- At least one deterministic visual/numeric comparison artifact under `.artifacts/main_fno_stable_physics_reconciliation/`.
