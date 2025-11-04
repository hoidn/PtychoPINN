Summary: Restore Phase E training by adding a MemmapDatasetBridge fallback for legacy `diffraction` NPZs and rerunning CLI jobs to capture real bundles with SHA256 proofs.
Mode: TDD
Focus: STUDY-SYNTH-FLY64-DOSE-OVERLAP-001 — Phase G comparison & analysis (Phase E real bundle evidence)
Branch: feature/torchapi-newprompt
Mapped tests: pytest tests/torch/test_data_pipeline.py::test_memmap_bridge_accepts_diffraction_legacy -vv; pytest tests/study/test_dose_overlap_training.py::test_execute_training_job_persists_bundle -vv; pytest tests/study/test_dose_overlap_training.py -k training_cli -vv
Artifacts: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-06T030500Z/phase_e_memmap_diffraction_fallback/

Do Now:
- Implement: ptycho_torch/memmap_bridge.py::MemmapDatasetBridge.__init__ + tests/torch/test_data_pipeline.py::test_memmap_bridge_accepts_diffraction_legacy — add a DATA-001 compliant fallback that maps legacy `diffraction` keys to the expected channel-first tensor (preserving dtype and CONFIG-001 bridge), codify behavior with a new pytest that fails RED before the change.
- Validate: pytest tests/torch/test_data_pipeline.py::test_memmap_bridge_accepts_diffraction_legacy -vv | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-06T030500Z/phase_e_memmap_diffraction_fallback/red/pytest_memmap_diffraction_red.log (capture the KeyError prior to implementation).
- Validate: pytest tests/torch/test_data_pipeline.py::test_memmap_bridge_accepts_diffraction_legacy -vv | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-06T030500Z/phase_e_memmap_diffraction_fallback/green/pytest_memmap_diffraction_green.log (confirm fallback works after implementation).
- Validate: pytest tests/torch/test_data_pipeline.py -k memmap -vv | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-06T030500Z/phase_e_memmap_diffraction_fallback/green/pytest_memmap_suite_green.log (regression sweep for bridge behavior).
- Validate: pytest tests/study/test_dose_overlap_training.py::test_execute_training_job_persists_bundle -vv | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-06T030500Z/phase_e_memmap_diffraction_fallback/green/pytest_training_bundle_green.log (guard SHA256 + manifest invariants post-change).
- Validate: pytest tests/study/test_dose_overlap_training.py -k training_cli -vv | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-06T030500Z/phase_e_memmap_diffraction_fallback/green/pytest_training_cli_suite_green.log (ensure CLI harness remains GREEN).
- Collect: pytest tests/torch/test_data_pipeline.py --collect-only -k memmap -vv | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-06T030500Z/phase_e_memmap_diffraction_fallback/collect/pytest_memmap_collect.log (mapped selector guardrail).
- Prepare: if [ ! -d tmp/phase_c_f2_cli ]; then AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md python -m studies.fly64_dose_overlap.generation --base-npz datasets/fly64/fly64_shuffled.npz --output-root tmp/phase_c_f2_cli | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-06T030500Z/phase_e_memmap_diffraction_fallback/cli/phase_c_generation.log; fi
- Prepare: if [ ! -d tmp/phase_d_f2_cli ]; then AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md python -m studies.fly64_dose_overlap.overlap --phase-c-root tmp/phase_c_f2_cli --output-root tmp/phase_d_f2_cli | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-06T030500Z/phase_e_memmap_diffraction_fallback/cli/phase_d_overlap.log; fi
- Execute: AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md python -m studies.fly64_dose_overlap.training --phase-c-root tmp/phase_c_f2_cli --phase-d-root tmp/phase_d_f2_cli --artifact-root tmp/phase_e_training_gs2 --dose 1000 --view dense --gridsize 2 --accelerator cpu --deterministic --num-workers 0 | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-06T030500Z/phase_e_memmap_diffraction_fallback/cli/dose1000_dense_gs2.log
- Execute: AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md python -m studies.fly64_dose_overlap.training --phase-c-root tmp/phase_c_f2_cli --phase-d-root tmp/phase_d_f2_cli --artifact-root tmp/phase_e_training_gs2 --dose 1000 --view baseline --gridsize 1 --accelerator cpu --deterministic --num-workers 0 | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-06T030500Z/phase_e_memmap_diffraction_fallback/cli/dose1000_baseline_gs1.log
- Archive: mkdir -p plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-06T030500Z/phase_e_memmap_diffraction_fallback/data && cp tmp/phase_e_training_gs2/training_manifest.json plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-06T030500Z/phase_e_memmap_diffraction_fallback/data/training_manifest.json && cp tmp/phase_e_training_gs2/skip_summary.json plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-06T030500Z/phase_e_memmap_diffraction_fallback/data/skip_summary.json && find tmp/phase_e_training_gs2/dose_1000 -name 'wts.h5.zip' -exec cp {} plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-06T030500Z/phase_e_memmap_diffraction_fallback/data/ \;
- Summarize: sha256sum plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-06T030500Z/phase_e_memmap_diffraction_fallback/data/wts.h5.zip* | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-06T030500Z/phase_e_memmap_diffraction_fallback/analysis/bundle_checksums.txt
- Summarize: python - <<'PY' > plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-06T030500Z/phase_e_memmap_diffraction_fallback/analysis/manifest_pretty.json
import json, pathlib
manifest = json.loads(pathlib.Path("tmp/phase_e_training_gs2/training_manifest.json").read_text())
json.dump(manifest, open("/dev/stdout", "w"), indent=2, sort_keys=True)
PY
- Summarize: Update plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-06T030500Z/phase_e_memmap_diffraction_fallback/analysis/summary.md with bundle locations, SHA256 values, CLI status, and residual gaps before touching docs/registries.

Priorities & Rationale:
- Phase E6 test_strategy row needs real bundle evidence before G comparisons (plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/test_strategy.md:268).
- DATA-001 requires readers tolerate legacy `diffraction` keys; Memmap bridge currently violates this (specs/data_contracts.md:207).
- Phase G runs mandate spec §4.6 bundle persistence and integrity signals (specs/ptychodus_api_spec.md:239).
- Attempt #96 documented the diff3d KeyError blocker; this loop clears that to resume comparisons (docs/fix_plan.md:54).

How-To Map:
- Export AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md before CLI invocations.
- Use `mkdir -p` for artifact subdirectories `{plan,red,green,collect,cli,data,analysis,docs}` prior to logging.
- After CLI success, verify manifest `bundle_sha256` entries match `sha256sum` output; if mismatch, capture diff in summary and halt comparisons.
- Keep tmp datasets intact; rerun Phase C/D generation only when directories are missing.
- Use `python -m json.tool` if manifest_pretty rewrite fails (fallback command provided above).

Pitfalls To Avoid:
- Do not bypass MemmapDatasetBridge by stubbing train/test loading; we need real fallback coverage.
- Maintain dtype conversions (float32 for diffraction amplitude, complex64 for guesses); no silent upcasting.
- Keep CONFIG-001 bridge untouched—fallback must not reorder initialization.
- Avoid modifying Phase C generators in this loop; focus on bridge tolerance.
- Ensure CLI commands run with deterministic knobs to keep reproducibility logs valid.
- Do not delete tmp datasets or previous artifacts; copy outputs into the new hub instead.
- Skip modifying `ptycho/model.py`, `ptycho/diffsim.py`, or `ptycho/tf_helper.py` (protected modules).

If Blocked:
- If fallback still raises KeyError, capture full stack trace in `analysis/summary.md`, attach failing manifest/log, and mark attempt blocked in docs/fix_plan.md and galph_memory.md.
- If CLI still fails after fallback, record the precise error context (dataset path, exception) and stop before archiving partial bundles.

Findings Applied (Mandatory):
- POLICY-001 — Torch backend remains mandatory; CLI uses torch runner (docs/findings.md:8).
- CONFIG-001 — Preserve legacy dict initialization order (docs/findings.md:10).
- DATA-001 — Honor canonical NPZ schema while tolerating legacy keys (docs/findings.md:14).
- OVERSAMPLING-001 — Keep gridsize/K invariants when running CLI (docs/findings.md:17).

Pointers:
- specs/data_contracts.md:207 — Canonical diffraction key expectations.
- specs/ptychodus_api_spec.md:239 — Bundle persistence contract.
- plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/test_strategy.md:268 — Phase E6 evidence gate.
- docs/fix_plan.md:54 — Attempt #96 notes on diff3d schema blocker.
- plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-06T010500Z/phase_e_training_bundle_real_runs/analysis/summary.md:15 — Current blocker analysis and SHA256 status.

Next Up (optional): Once bundles exist, re-run Phase G dense comparisons with updated manifests.

Doc Sync Plan:
- After GREEN tests, run `pytest tests/torch/test_data_pipeline.py --collect-only -k memmap -vv` (log already captured) and update docs/TESTING_GUIDE.md Phase E section plus docs/development/TEST_SUITE_INDEX.md to register the new selector, placing updated docs under the same artifact hub.
