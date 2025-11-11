### Turn Summary
Audited the Phase G dense hub after a clean pull; even with the allow_pickle fix (`5cd130d3`) on branch, `analysis/` still only has `blocker.log` and `{cli}` is limited to the stale Phase C/D/stdout logs, so no SSIM grid, verification, preview, metrics, or artifact-inventory artifacts exist yet.
Updated the Phase G checklist, hub plan, docs/fix_plan.md, and input.md to capture the clean-state reality check and restate the counted `run_phase_g_dense.py --clobber` + immediate `--post-verify-only` deliverables with pytest guardrails and artifact requirements.
Next: Ralph must rerun the mapped pytest selectors, execute the dense run + `--post-verify-only` sweep from `/home/ollie/Documents/PtychoPINN`, and publish MS-SSIM/MAE deltas plus preview/verifier/SSIM grid evidence into this hub and ledger docs.
Artifacts: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/

### Turn Summary
Fixed the NPZ pickle blocker (ValueError: Object arrays cannot be loaded when allow_pickle=False) by adding allow_pickle=True to np.load() calls in overlap.py:388 and training.py:409,416.
The original Phase D blocker is resolved; Phase D now progresses past data loading but fails with a different error: insufficient spacing acceptance rate for dense view (0.8% vs 10.0% required).
Next: Consult supervisor on whether to adjust Phase C scan spacing parameters or relax the overlap threshold to achieve minimum acceptance rate.
Artifacts: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/ (red/blocked_2025-11-11T173011Z.md, green/pytest_post_verify_only.log)
