## Completed In This Pass

- Audited the recovered modes-32 lane against the approved capped CNS contract and confirmed no code repairs were needed.
- Accepted `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-spectral-modes32-compare/cns-spectral-modes32-40ep-20260428T014353Z` as the authoritative fresh `40`-epoch row after verifying tracker exit code `0` and required completion artifacts.
- Wrote `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-spectral-modes32-compare/compare_40ep_against_existing.json` and `.csv`, plus the rendered top-level `40`-epoch galleries.
- Published the durable summary, updated CNS/discoverability docs, and switched the implementation state from `RUNNING` to `COMPLETED`.

## Completed Plan Tasks

- Task 1: roadmap authorization, profile fairness, recovered artifact audit, and deterministic checks.
- Task 2: authoritative `40`-epoch root resolution from the tracked `014353Z` run.
- Task 3: anchored `40`-epoch compare emission against only `spectral_resnet_bottleneck_base`, `fno_base`, and `unet_strong`.
- Task 4: durable summary, CNS summary, docs index/studies index, progress ledger, and implementation-state updates.

## Remaining Required Plan Tasks

- None inside this approved backlog scope.
- Out-of-scope work still remains for the broader NeurIPS campaign: full-training benchmark rows and any later spectral-family follow-ups.

## Verification

- `pytest -q tests/studies/test_pdebench_image128_models.py tests/studies/test_pdebench_image128_runner.py` -> `65 passed`
- `python -m compileall -q scripts/studies/pdebench_image128 scripts/studies/run_pdebench_image128_suite.py` -> exit `0`
- required durable outputs present:
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_spectral_modes32_compare_summary.md`
  - `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-spectral-modes32-compare/compare_10ep_against_existing.json`
  - `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-spectral-modes32-compare/compare_40ep_against_existing.json`
- comparison standard: the anchored `10`/`40`-epoch sidecars enforce the fixed capped CNS contract and allow only the coupled `12 -> 32` change in `fno_modes` and `spectral_bottleneck_modes`

## Residual Risks

- This remains capped decision-support evidence only; it is not benchmark-complete CNS evidence.
- The modes-32 row improved the capped `10`-epoch slice, but at `40` epochs it regressed on aggregate error and `fRMSE_low` relative to the shared `12/12` spectral row while improving only `fRMSE_mid/high`.
- The modes-32 row is much larger (`42,388,614` params vs `8,186,726`), so there is no basis to promote it as the default spectral reference from this item alone.
