# spec-ptycho-torch-probe-layout.md — Torch Probe Batch Layout and Amplitude Physics Gain (Normative)

Overview (Normative)
- Purpose: Define the documented probe tensor layout at every stage of the PyTorch training pipeline, the fail-fast precondition that enforces it in the physics forward, and the explicit amplitude physics gain that replaces the historical accidental batch-size gain.
- Provenance: 2026-07-12 probe-rank physics contract fix design (`docs/superpowers/specs/2026-07-12-probe-rank-physics-contract-fix-design.md` §3), adopted after the Task 21 root-cause chain was independently confirmed. Mechanism and history: `docs/findings.md` PROBE-RANK-001 (commits `82da77960` / `8b3d7a011`).
- Scope: `ptycho_torch/*` training paths only. TF-side core files (`ptycho/model.py`, `ptycho/diffsim.py`, `ptycho/tf_helper.py`) are untouched by this contract.

## 1. Probe tensor layout by stage (Normative)

| Stage | Layout | Producer | Consumer |
|---|---|---|---|
| Container/dataset sample | `(C, P, H, W)` per sample; a single shared probe is stored as `(H, W)`/`(P, H, W)` and expanded at emission | inline dataset (`ptycho_torch/workflows/components.py` `PtychoLightningDataset.__getitem__`), `PtychoDataset` (`ptycho_torch/dataloader.py` `__getitem__`) | DataLoader collate |
| Collated batch | `(B, C, P, H, W)`; the mmap emission `(B, 1, 1, H, W)` is the C=1, P=1 instance | collate | `ProbeIllumination.forward` (`ptycho_torch/model.py`) |
| Physics forward | mode axis is dim 2 (P); the coherent mode sum in `pad_and_diffract` (`ptycho_torch/helper.py`, `torch.sum(input, dim=2)`) ranges over TRUE probe modes only | `ProbeIllumination` | `pad_and_diffract` |

- Every physics mode (amplitude default, `rectangular_scaled`, CI) SHALL emit the per-sample `(C, P, H, W)` layout. The former amplitude-mode flat exception (the "pre-82da7796 raw convention" restored by `8b3d7a011`) is removed; flat `(B, H, W)` collated probes are banned.
- Rationale (Informative): a flat `(B, H, W)` probe right-align-broadcasts under `x.unsqueeze(2) * probe` so the batch axis lands in the mode slot, producing `(B, C, B, H, W)`; the coherent mode sum then multiplies the predicted field by B (a silent, batch-size-dependent physics gain — ×16 on full reference-recipe batches, ×2 on the final partial batch) and, for per-sample-distinct probes, MIXES different samples' fields.

## 2. `ProbeIllumination.forward(x, probe)` precondition (Normative)

- Enforced in `ptycho_torch/model.py` (`ProbeIllumination._require_documented_probe_layout`):
  - `probe.ndim == 5`
  - `probe.shape[-2:] == (N, N)`
  - `probe.shape[0] in (1, B)`
  - `probe.shape[1] in (1, C)`
- Any violation — in particular any sub-rank-5 probe — SHALL raise the typed, module-level `ProbeLayoutError` naming the offending shape, the documented contract, and this spec/finding. No implicit right-align broadcast into the mode slot is reachable.
- Behavior is unchanged (bit-identical) for contract-conforming inputs, including the mmap path's existing `(B, 1, 1, H, W)` emission.
- Dependency note: `pad_and_diffract`'s mode-sum semantics are unchanged; with this precondition its coherent sum ranges over true modes only.
- Tests: `tests/torch/test_probe_layout_contract.py` (error matrix incl. the multi-probe mixing case), `tests/torch/test_inline_dataset_rectangular_scaled_batched.py`, `tests/torch/test_inline_dataset_amplitude_scaling_regression.py`, `tests/torch/test_multimode_probe_and_from_np.py` (mmap emission pins).

## 3. Explicit amplitude physics gain (Normative)

- Field: `ModelConfig.amplitude_physics_gain: float` (`ptycho_torch/config_params.py`), default `1.0`.
- Plumbing: passes through `create_training_payload` overrides like every other torch-only knob; the effective value is always recorded in the payload audit trail (`overrides_applied`) and serialized into Lightning hparams via the persisted `model_config`.
- Validation (`ptycho_torch/scaling_contract.py::validate_amplitude_physics_gain`, invoked by `validate_scale_contract` in every mode): finite and > 0; whenever the rectangular/CI scaling path is active (`physics_forward_mode='rectangular_scaled'`) the value MUST be exactly 1.0 (fail-closed).
- Legacy normalized-amplitude regime: derive the gain once from the exact sealed training input and exact amplitude forward using the closed-form physical normalization in `docs/model_baselines.md`; share that value across architectures and legacy MAE/NLL profiles. Held-out quality, initialization RMS matching, and gain sweeps SHALL NOT select it. The Task 26 value 16 is retained only as the batch-16 broadcast-equivalent historical conditioner, not as a physical normalization or recommended baseline. Rectangular/CI scaling remains required to use exactly 1.0.
- Application: applied ONCE, multiplicatively, to the predicted amplitude inside the amplitude-mode branch of `ForwardModel.forward` (`ptycho_torch/model.py`), after `inv_scale` and before the optional trainable alpha/beta. `rectangular_scaled` and CI forwards are untouched. The value is read live from the shared `ModelConfig` instance so checkpoint-loaded modules honor it.
- Training-objective device only: inference/reassembly (`forward_predict` and downstream) SHALL NOT apply the gain. Historical checkpoints score identically under either physics because the accident was train/val-loop only.
- Rationale (Informative): the accidental ×B gain demonstrably conditioned amplitude-mode training (amp SSIM 0.486 vs 0.896 at B=16), and fixed G=16 replicates that accidental objective on 99.82% of reference-recipe train steps. Task 26's quality sweep therefore characterized legacy conditioning, not physical normalization. For the sealed Task 30 v3 lines input, the direct expression gives `G_phys=12.452229360013307`; that value is dataset-specific rather than a new universal default.
- Quantitative tie-back (sealed evidence): at the sealed rung0 reference weights on the rung1e first raster batch (CPU), documented rank at gain 1 → loss 0.936635; gain 16 → 0.0731752, reproducing the pre-fix flat-layout measurement 0.073175. Tests: `tests/torch/test_amplitude_physics_gain.py`, `tests/torch/test_amplitude_physics_gain_tie_back.py`; driver: `python -m scripts.studies.ablation.runtime_ladder_step_parity_cli`.

## 4. Compatibility notes (Informative)

- Checkpoints saved before this contract load normally: `ModelConfig(**hparams_dict)` fills the missing field with the 1.0 default; they are marked (via PROBE-RANK-001) as trained under the accidental-gain objective.
- Historical regression floors pinned under the accidental gain remain reproducible with explicit `--amplitude-physics-gain 16`; they are compatibility evidence, not the canonical starting value for a new run. New legacy-amplitude studies use the sealed-input derivation above.
- The ladder diagnostic lever `mmap_probe_batch_shape="dictionary_flat"` (`scripts/studies/ablation/runtime_ladder_injections.py`) remains retained for historical-evidence readability; any attempt to train through it now stops loudly with `ProbeLayoutError`. Plan Task 29 owns its removal during producer retirement and any migration-whitelist tombstone.
