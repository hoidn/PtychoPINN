# Completed In This Pass

- Closed the capped CNS Hybrid-spectral to FFNO parameter-space item after the
  tracked `10`-epoch shell probes finished cleanly, the anchored
  `compare_10ep_against_existing.{json,csv}` sidecar was written from the frozen
  reference manifest, and the promotion gate determined that neither probe
  justified a `40`-epoch follow-up.
- Wrote the durable CNS-only summary, synced the umbrella CNS summary and docs
  indexes, recorded the initiative ledger entry, and emitted the final
  implementation-state bundle.

# Completed Plan Tasks

- Task 1: reconfirmed prerequisite authorities, reran the deterministic gate,
  and preserved the frozen `study_matrix.json` plus `reference_runs_{10,40}ep`
  manifests.
- Task 2: used the already-landed manual-only profile support for
  `spectral_resnet_bottleneck_base_down1` and
  `spectral_resnet_bottleneck_base_transpose`; no further code change was
  needed in this pass.
- Task 3: reused the existing inspect proof and frozen manifests, then validated
  them before compare collation.
- Task 4: confirmed the tmux-launched `10`-epoch pilot completed with `EXIT:0`
  and wrote the anchored `compare_10ep_against_existing` JSON/CSV plus sample
  galleries.
- Task 5: applied the declared promotion rule and closed the `40`-epoch set as
  empty because both fresh rows were clearly worse than
  `spectral_resnet_bottleneck_base` on `relative_l2`, `err_nRMSE`, and
  `fRMSE_high`.
- Task 6: wrote the durable summary, updated the umbrella CNS summary and docs
  discoverability, appended the initiative progress-ledger entry, and emitted
  this execution report and the final state bundle.

# Remaining Required Plan Tasks

- None. The item is complete because the approved `40`-epoch branch was
  conditional and the `10`-epoch promotion gate closed it.

# Verification

- `pytest -q tests/studies/test_pdebench_image128_models.py tests/studies/test_pdebench_image128_runner.py`
  passed: `86 passed in 57.86s`
  Log: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-hybrid-spectral-ffno-parameter-space/verification/final_pytest.log`
- `python -m compileall -q scripts/studies/pdebench_image128 scripts/studies/run_pdebench_image128_suite.py`
  passed with exit `0`
  Log: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-hybrid-spectral-ffno-parameter-space/verification/final_compileall.log`
- `pytest -v -m integration`
  passed: `5 passed, 4 skipped, 1651 deselected in 298.81s`
  Log: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-hybrid-spectral-ffno-parameter-space/verification/integration_pytest.log`
- Inspect/manifests validation, compare-generation logging, launcher-exit
  capture, and final output validation were archived under the same
  `verification/` directory.

# Residual Risks

- This lane remains capped `512 / 64 / 64`, decision-support-only evidence; it
  cannot support a benchmark or paper-grade promotion without a later
  full-training item.
- The negative shell result is specific to the current shared-spectral anchor
  and the two approved one-axis probes; it does not rule out every possible
  Hybrid-to-FFNO bridge under a different budget or contract.
- The stronger repo-local FFNO-family follow-up is still the prerequisite
  `ffno_bottleneck_localconv_base`; this item did not generate a new longer-run
  bridge row.
