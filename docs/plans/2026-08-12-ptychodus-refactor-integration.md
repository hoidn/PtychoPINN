# Ptychodus ↔ PtychoPINN `refactor` Integration Fix

**Date:** 2026-08-12
**Branch:** `ptychodus-integration` (clone-local name for `refactor`) in `/home/ollie/Documents/PtychoPINN-refactor`; merges back to `refactor`.
**Consumer under test:** `~/Documents/ptychodus` (branch `misc`, editable install in `ptycho311`), adapter `src/ptychodus/model/ptychopinn/reconstructor.py`.

## Problem

The ptychodus `PtychoPINNTrainableReconstructor.train()` constructs
`TrainingConfig` with the historical **flat** keyword surface
(`train_data_file`, `test_data_file`, `nphotons`, `mae_weight`, `nll_weight`,
`realspace_mae_weight`, `realspace_weight`). On `refactor`, `TrainingConfig`
is a nested Pydantic `BaseSettings` (`data.*`, `tf_loss.*`, `sampling.*`) with
`extra="forbid"`, so construction raises `ValidationError` (7 ×
`extra_forbidden`) and the train/save/load/infer workflow dies at
`reconstructor.py:251`.

## Evidence (2026-08-12 recon)

- API smoke replay of the adapter's full `ptycho.*` surface against
  `refactor` @ `4d6a6b1a1`: **9/10 pass**; the only failure is flat
  `TrainingConfig` construction. Nested construction + `update_legacy_dict`
  passes. (`scratchpad/integration_smoke.py`)
- E2E `ptychodus/scripts/ptychopinn_tf_test.py` probe (SLAC NPZ
  `Run1084_recon3_postPC_shrunk_3.npz`, 2 epochs): all stages before training
  succeed (ModelCore init, reconstructor registration, pattern load, product
  open, `export_training_data` incl. key validation); fails only at the flat
  `TrainingConfig` construction. (`scratchpad/e2e_probe.log`)
- PtychoPINN-side interop suites on `refactor`
  (`tests/io/test_ptychodus_product_io.py`,
  `tests/io/test_ptychodus_interop_h5.py`): **4/4 pass**.
- `InferenceConfig`/`ModelConfig` adapter constructions are compatible
  unchanged (incl. deprecated `object_big` alias, strict ints — ptychodus
  stores `gridsize`/`n_filters_scale` as integer parameters).

## Decision

Fix on the **PtychoPINN side** (user preference; ptychodus stays untouched):
accept the historical flat root spellings on `TrainingConfig` as
**deprecated aliases**, lifted into their nested owners by a
`model_validator(mode="before")`.

Rationale:
- The codebase already encodes this exact idiom twice: `SamplingConfig`
  lifts `n_images`→`n_groups`, and `resolution._normalize_public_source`
  lifts flat model fields into `model.*` with *equal-duplicate accepted /
  unequal-duplicate rejected* semantics (documented in
  `docs/CONFIGURATION.md` §resolution).
- `extra="forbid"` stays intact for genuinely unknown keys; the alias set is
  enumerated and documented, and every use emits `DeprecationWarning`.
- Keeps ptychodus working unchanged against both `fno-stable` (flat) and
  `refactor` (nested) — the adapter is the only known external caller and is
  named as such by `specs/ptychodus_api_spec.md`.

### Alias table (flat root key → nested owner)

| Legacy flat key | Nested owner |
| :-- | :-- |
| `train_data_file`, `test_data_file`, `nphotons` | `data.*` |
| `mae_weight`, `nll_weight`, `realspace_mae_weight`, `realspace_weight` | `tf_loss.*` |
| `n_groups`, `n_images`, `n_subsample`, `subsample_seed`, `neighbor_count`, `enable_oversampling`, `neighbor_pool_size`, `sequential_sampling` | `sampling.*` |

Excluded by design: flat `optimizer`/`scheduler`/`gradient_clip_val`/
`gradient_clip_algorithm`/`torch_loss_mode` and friends — the `optimizer` and
`scheduler` names collide with the nested section names (type-ambiguous), no
external caller uses them, and Torch-side callers were already migrated.
YAGNI; revisit only on evidence.

### Semantics

1. Lifting applies only when the validator input is a plain mapping.
2. Alias present, section key absent or lacking that field → value lifted;
   alias key removed from root.
3. Alias and nested field both explicitly present: equal → accepted once;
   unequal → `ValueError` naming both spellings (mirrors
   `_normalize_public_source`).
4. Nested section supplied as a model instance + conflicting alias →
   rejected the same way; equal → alias dropped.
5. Every alias use emits `DeprecationWarning` naming the legacy spelling and
   its nested replacement. Pure nested construction warns nothing.
6. `n_images` lifts into `sampling.n_images` and then follows
   `SamplingConfig`'s existing alias conversion.
7. Unknown (non-alias) root keys still fail `extra_forbidden`.

## Task table (bookkeeping authority)

| # | Task | Route | Status |
| :- | :-- | :-- | :-- |
| 1 | Legacy flat-alias validator on `TrainingConfig` + tests (`tests/test_training_config_legacy_aliases.py`, TDD) | SDD implementer (opus — fable spawns 529-overloaded at dispatch time) + inline review | done — commit `154b5ce6a`, review approved no fix rounds |
| 2 | Docs: `specs/ptychodus_api_spec.md` §2.1/§5.2 + `docs/CONFIGURATION.md` training-authoring note | inline (docs-only per process memory) | done — commit `8d6d50ad9` |
| 3 | Verification: unit+interop suites, full E2E `ptychopinn_tf_test.py` on GPU, `ci/run_ci_tests.sh` regression check, commits, push `→ origin refactor` | inline | see Verification matrix |

### Task 1 spec (implementer brief source)

In `ptycho/config/config.py`, add a `model_validator(mode="before")` on
`TrainingConfig` (name: `_lift_legacy_flat_fields`) implementing the alias
table and semantics above. Module-level constant
`_LEGACY_TRAINING_ROOT_ALIASES: dict[str, tuple[str, str]]` maps flat key →
`(section, field)`.

Tests (new module `tests/test_training_config_legacy_aliases.py`), written
first, must cover:
- the exact ptychodus construction vector (14 flat kwargs incl.
  `model=ModelConfig(...)`) validates, and `model_dump()` equals the
  equivalent nested construction;
- parametrized lift of each of the 15 aliases;
- equal flat+nested duplicate accepted once; unequal raises with both
  spellings in the message (mapping section AND model-instance section
  variants);
- `DeprecationWarning` on alias use; no warning for pure nested;
- unknown root key still `extra_forbidden`;
- root `n_images` converts through to `sampling.n_groups`;
- `update_legacy_dict` projection of a flat-constructed config matches the
  nested-constructed one.

Constraints: do not weaken `extra="forbid"`; do not touch `ModelConfig`,
`InferenceConfig`, or `resolution.py`; keep `_TRAINING_INPUT_NAMES`
invariants in `resolution.py` valid (validator must not add model fields).
Gate: new module + `tests/test_public_config_pydantic_adoption.py` +
`tests/test_public_config_resolution.py` green.

## Verification matrix (Task 3)

| Check | Command (ptycho311, `PYTHONPATH=clone`) | Expectation | Result |
| :-- | :-- | :-- | :-- |
| New + adjacent unit suites | `pytest tests/test_training_config_legacy_aliases.py tests/test_public_config_pydantic_adoption.py tests/test_public_config_resolution.py` | green | new module 27/27; resolution 95/95; adoption module pre-existing red (20F identical at BASE `4d6a6b1a1` — stale vs BaseSettings migration `b45eacf64`, needs own task) |
| Interop IO | `pytest tests/io/test_ptychodus_product_io.py tests/io/test_ptychodus_interop_h5.py` | green | 4 passed |
| API smoke replay | `python integration_smoke.py` (flat-construction check flipped to expect success) | 10/10 | 10/10 |
| E2E | `ptychopinn_tf_test.py --nepochs 5 --test-samples 128` on GPU | exit 0, `wts.h5.zip` + `recon_object.npy` produced, non-empty object | exit 0; 5/5 epochs; `wts.h5.zip` (35 MB), `recon_product.h5`, `recon_object.npy` produced; object 64×64 complex64 non-degenerate (amp mean 0.70) |
| CI gate regression | `bash ci/run_ci_tests.sh` in clone | same contract as pre-fix (1572/9 skipped baseline @ 8b4dd1394) | exit 0; 1565 passed / 16 skipped / 0 failed / 0 errors — the 7 extra skips were `test_fixture_pytorch_integration.py` missing its generated local fixture in the clone; after copying `tests/fixtures/pytorch_integration/` from the main checkout those 7 pass (7/7), restoring exact 1572/9 parity |

## Risks / notes

- E2E trains on GPU (never CPU — process memory); GPU verified free at plan
  time (2.3/24 GiB, no compute jobs).
- The ptychodus adapter passes `train_data_file=Path()` placeholders (its
  historical "not used" convention); `DataConfig` path fields are
  `Path | None` with no suffix constraint, so this validates. Real paths are
  not required for `run_cdi_example`, which consumes the `RawData` objects.
- `.tmp/`-style leftovers in both repos are untouched.
- Push scope: clone → `origin refactor` (origin = local main checkout).
  Public GitHub push remains the user's call (origin-public rule).
