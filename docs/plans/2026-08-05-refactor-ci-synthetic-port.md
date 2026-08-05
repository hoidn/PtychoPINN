# Port the CI Synthetic-Generation Surface to `refactor`

**Status:** DRAFT — pending user approval. Goal: make the validated
five-epoch CI recipe (documented on the `fno-stable` line in
`scripts/simulation/README.md`, "Validated five-epoch CI example") runnable
verbatim on this branch, so its example + flag table can be ported as-is.

**Source material:** the `fno-stable` branch in this same repository (both
branches share the object store; consult reference implementations with
`git show fno-stable:<path>` — do not cherry-pick, since this branch's
structures deliberately diverge). Key references:
`ptycho/simulation/flat_acquisition.py` (CI contract,
`_apply_count_intensity_contract`), `ptycho/workflows/synthetic_config.py`
(profile registry, `hybrid-resnet-lines-ci`),
`scripts/simulation/synthetic_pipeline.py` (contract CLI flags),
`ptycho_torch/reconstruction_evaluation.py` (`measurement_domain` keyword),
and the design that governs the gauge semantics,
`fno-stable:docs/plans/2026-08-04-ci-gauge-invariant-scaling.md`.

**What already exists on `refactor` (do not rebuild):** the Torch `ci`
profile (`config_factory.py`, locks `ci_intensity_v2` + `count_intensity` +
`rectangular_scaled` + Poisson), the `rect_s1s2_init` ModelConfig field with
the dose-closure solve, the strict `rect-s1s2-initialization-v1` startup
record, `validate_rect_s1s2_initialization_contract`, the public
`ptycho_synthetic` runner with stage manifests v2, and the mmap barycentric
reconstruction/evaluation workflow. The gap is purely the **synthetic
generation side of CI** plus its CLI/profile plumbing.

---

## 1. Gap inventory (verified against this branch's code)

| Missing piece | Where it lives on `fno-stable` | `refactor` state |
|---|---|---|
| Count-intensity emission (counts on disk; CI-scaled `probeGuess`; `count_amplitude_scale`; physical-probe digest) | `flat_acquisition.py` `_apply_count_intensity_contract` | `flat_acquisition.py` exists, legacy-amplitude only |
| `hybrid-resnet-lines-ci` profile | `synthetic_config.py` profile registry | Single hard-rejecting `_PROFILE_NAME = "synthetic-lines"` (`synthetic_config.py:75,1244`) |
| Contract CLI flags: `--scale-contract-version`, `--measurement-domain`, `--physics-forward-mode`, `--cnn-output-mode`, `--torch-loss-mode`, `--rect-s1s2-init` | `synthetic_pipeline.py` | Absent |
| Training-stage bridge: synthetic CI profile → Torch `ci` profile with `rect_s1s2_init=dose_closure` default | resolver `_resolve_model` + workflow wiring | Absent (Torch `ci` profile reachable only programmatically) |
| Evaluator count-domain acceptance | `reconstruction_evaluation.py` `measurement_domain` keyword | Absent |

Present already (verified): `--probe-transform`, `--train-raw-selection`,
`--training-groups`/`--validation-groups`, `--neighbor-count`/`--neighbor-pool-size`,
`--groups-per-center`, `--photons-per-pattern`, `--gradient-clip-*`,
`--plateau-*`, execution flags. The example uses no raster layout, so the
scan-geometry capability is **not** part of this port.

## 2. Port design

Follow this branch's established porting discipline: curated, coherent
commits adapted to `refactor`'s structures and stricter public contracts —
not patch transplants. Per-area decisions:

1. **`flat_acquisition` CI contract.** Add the count-intensity emission path
   mirroring `fno-stable` semantics: for `measurement_domain=count_intensity`,
   diffraction is stored as Poisson-realized counts and `probeGuess` is
   OVERWRITTEN with the CI-scaled physical probe (`probe_unscaled × S`,
   `S = derive_count_amplitude_scale(...)`), with `count_amplitude_scale`
   and the physical-probe digest recorded in the dataset manifest. Carry the
   contract docstring stating that for flat_acquisition data `probeGuess`
   **is** the CI-scaled physical probe — this convention was the subject of
   an entire RCA bug class (see the gauge design doc §1); it must be
   explicit, and the solved startup gauge is its runtime diagnostic.
2. **Profile registry.** Generalize the single-profile check into a
   two-entry registry: `synthetic-lines` (unchanged, byte-identical
   identity) and `hybrid-resnet-lines-ci`. The CI profile selects the
   count-intensity generation path, the coherent contract set, the
   `hybrid_resnet` model family fields used by the validated recipe, and
   `rect_s1s2_init=dose_closure` as its default (matching the Torch `ci`
   profile's field-locking style: contradicting a locked contract field
   fails closed; `dose_closure` itself remains overridable via
   `--rect-s1s2-init ones`).
3. **CLI flags.** Add the six contract flags to `synthetic_pipeline.py`
   with the same pairing rules as `fno-stable` (the units triple is
   inseparable; partial combinations rejected naming the offending field).
   `--rect-s1s2-init {ones,dose_closure}` threads to the model config and
   must compose with `validate_rect_s1s2_initialization_contract` (already
   on this branch — the validator is the port's ally, not new work).
4. **Training-stage bridge.** The CI profile's training stage resolves
   through the existing Torch `ci` profile path
   (`resolve_training_payload(..., profile="ci")` semantics) so contract
   locking, the dose-closure solve, and the strict startup record all come
   from code this branch already tests. The stage-manifest v2 training
   entry already requires `training_summary.json` with a mode-matched
   record — reuse rules apply unchanged.
5. **Evaluator.** Add the `measurement_domain` keyword to
   `reconstruction_evaluation` (count-domain runs must not require legacy
   amplitude diagnostics), mirroring `fno-stable:afc2f6674`'s behavior.
6. **Identity/digests.** The CI profile mints its own workflow/simulation
   digests. The sealed `synthetic-lines-v1` identity (recipe digest, stage
   manifests, any pinned hashes in tests) must remain byte-identical — the
   registry change is additive. Follow this branch's rule that
   `rect_s1s2_init` is part of workflow identity (both modes hash
   differently), already established by the gauge-fix port.

## 3. Validation plan

CPU first (TDD, adapt `fno-stable`'s batteries to this branch's test idioms):
1. `flat_acquisition` CI tests: counts realized at `photons-per-pattern`
   scale; `probeGuess` equals `probe_unscaled × S` exactly; manifest fields
   present; legacy path byte-unchanged.
2. Profile tests: registry resolves both names; unknown profiles still fail
   closed; contract-triple pairing rules; `hybrid-resnet-lines-ci` defaults
   (including `dose_closure`) and their override behavior.
3. CLI tests: flag plumbing, help discoverability, rejection messages.
4. Identity tests: sealed `synthetic-lines` digests byte-identical; CI
   profile digests stable and distinct per `rect_s1s2_init` mode.
5. Evaluator tests: `measurement_domain="count_intensity"` accepted without
   legacy amplitude diagnostics.
6. Full existing CPU suite green (this branch's focused batteries as run in
   root-merge sessions).

GPU acceptance (the point of the port):
7. Run the **verbatim** validated recipe — the exact command from
   `fno-stable:scripts/simulation/README.md`'s "Validated five-epoch CI
   example" (`ptycho_synthetic --profile hybrid-resnet-lines-ci ...
   --rect-s1s2-init dose_closure`, seed 3, 5 epochs). Acceptance: amplitude
   SSIM ≥ 0.78, phase ≥ 0.93; expect ≈0.815/0.939 with
   `solved_gauge ≈ 3.12` in the startup record (same data recipe and seed;
   allow the band, not the exact figures, across branch lineage).
8. Control: `--rect-s1s2-init ones` on the same recipe reproduces the slow
   baseline (≈0.70 amplitude SSIM) — demonstrating the port carries the
   mechanism, not just the numbers.

Docs (after 7 passes):
9. Port the example + non-obvious-flag table verbatim from `fno-stable`
   into this branch's `scripts/simulation/README.md`; reconcile
   `docs/CONFIGURATION.md`'s "Dose-closure initialization example" (keep the
   programmatic form, drop the cross-branch pointer, link the now-native CLI
   example); update the CI-profile sections and `docs/index.md` routing.

## 4. Non-goals

- No raster/`--scan-position-layout` port (the example doesn't use it).
- No changes to the sealed `synthetic-lines-v1` identity or its tests.
- No changes to protected physics modules; the dose-closure solve and
  validator are consumed as-is.
- No new quality-gate test sealing in this port (a CI five-epoch sealed
  gate is a separate decision, on either branch).
- No pushes beyond the local repository.

## 5. Risks

- **Structural drift:** this branch's `synthetic_config.py` is a rewrite;
  the profile registry must be built in its idiom, not transplanted —
  budget review time for resolver-shape differences.
- **Digest movement:** any accidental change to `synthetic-lines` identity
  fails the port; test 4 is the guard and should be written first.
- **Probe-convention ambiguity:** the CI `probeGuess`-is-physical convention
  is exactly the trap from the RCA; the contract docstring and a
  solved-gauge sanity note in docs are mandatory, not optional polish.
- **Acceptance variance:** cross-branch lineage may shift the five-epoch
  number within the band; the ≥0.78/≥0.93 floors (not the point values) are
  the gate.
