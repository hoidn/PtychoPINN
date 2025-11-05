Summary: Restore Phase C dataset generation by making canonicalization and patch tools metadata-aware so the dense Phase G pipeline can run.
Mode: TDD
Focus: STUDY-SYNTH-FLY64-DOSE-OVERLAP-001 — Phase G comparison & analysis
Branch: feature/torchapi-newprompt
Mapped tests:
  - pytest tests/tools/test_transpose_rename_convert_tool.py::test_canonicalize_preserves_metadata -vv
  - pytest tests/tools/test_generate_patches_tool.py::test_generate_patches_preserves_metadata -vv
Artifacts: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-07T130500Z/phase_c_metadata_pipeline/

Do Now (hard validity contract):
  - Implement (STUDY-SYNTH-FLY64-DOSE-OVERLAP-001): tests/tools/test_transpose_rename_convert_tool.py::test_canonicalize_preserves_metadata — create failing test covering metadata-bearing NPZ input (expect ValueError today) and log RED.
  - Implement (STUDY-SYNTH-FLY64-DOSE-OVERLAP-001): tests/tools/test_generate_patches_tool.py::test_generate_patches_preserves_metadata — add failing test ensuring metadata propagates through Y patch generation.
  - Implement (STUDY-SYNTH-FLY64-DOSE-OVERLAP-001): scripts/tools/transpose_rename_convert_tool.py::transpose_rename_convert — load via MetadataManager, record canonicalization transformation, and save with metadata.
  - Implement (STUDY-SYNTH-FLY64-DOSE-OVERLAP-001): scripts/tools/generate_patches_tool.py::generate_patches — load/save with MetadataManager while logging generate_patches transformation.
  - Validate (STUDY-SYNTH-FLY64-DOSE-OVERLAP-001): pytest tests/tools/test_transpose_rename_convert_tool.py::test_canonicalize_preserves_metadata -vv
  - Validate (STUDY-SYNTH-FLY64-DOSE-OVERLAP-001): pytest tests/tools/test_generate_patches_tool.py::test_generate_patches_preserves_metadata -vv
  - Validate (STUDY-SYNTH-FLY64-DOSE-OVERLAP-001): AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py --hub plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-07T130500Z/phase_c_metadata_pipeline --dose 1000 --view dense --splits train test

How-To Map:
  1. export AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md
  2. mkdir -p plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-07T130500Z/phase_c_metadata_pipeline/{red,green,collect,tests}
  3. Write regression tests in `tests/tools/test_transpose_rename_convert_tool.py` and `tests/tools/test_generate_patches_tool.py` using MetadataManager fixtures; run `pytest tests/tools/test_transpose_rename_convert_tool.py::test_canonicalize_preserves_metadata -vv` expecting ValueError, capture log to `red/pytest_transpose_metadata.log`; repeat for patches test saving to `red/pytest_patches_metadata.log`.
  4. Update `scripts/tools/transpose_rename_convert_tool.py` to ingest via `MetadataManager.load_with_metadata`, skip `_metadata` during array transforms, append canonicalization transformation record, and save via `MetadataManager.save_with_metadata`.
  5. Update `scripts/tools/generate_patches_tool.py` to load metadata, add generate_patches transformation entry, and save output with metadata.
  6. Rerun pytest selectors (GREEN) and write logs to `green/pytest_transpose_metadata.log` and `green/pytest_patches_metadata.log`; execute `pytest --collect-only tests/tools/test_transpose_rename_convert_tool.py -vv > collect/pytest_transpose_metadata_collect.log` (repeat for patches test) after GREEN.
  7. Execute dense orchestrator command (dose=1000 dense train/test) targeting the new hub; archive CLI log under `cli/phase_c_generation.log` and any blockers in `analysis/blocker.log`.
  8. Update `summary/summary.md` with RED/green evidence and pipeline status; amend `docs/TESTING_GUIDE.md` §2 + `docs/development/TEST_SUITE_INDEX.md` with new selectors and capture collect-only proof.
  9. Update `docs/fix_plan.md` Attempts and note any new finding about metadata propagation if warranted.

Pitfalls To Avoid:
  - Do not drop metadata; preserve and extend transformation history via MetadataManager.
  - Keep new tests lightweight (tiny arrays) to avoid long runtimes in CI.
  - Maintain TYPE-PATH-001 discipline when constructing file paths.
  - Do not touch stable physics modules (`ptycho/model.py`, `ptycho/diffsim.py`, `ptycho/tf_helper.py`).
  - Ensure pytest selectors collect ≥1 test before finalizing.
  - Stop pipeline on error and log blocker instead of retrying blindly.
  - Avoid writing artifacts outside the assigned hub or modifying previous timestamp directories.
  - Keep MetadataManager imports confined to tooling modules (no circular deps).

If Blocked:
  - Capture the failure signature in `analysis/blocker.log`, store offending CLI log under `cli/`, summarize in `summary/summary.md`, and ping supervisor via docs/fix_plan.md / galph_memory.md update.

Findings Applied (Mandatory):
  - DATA-001 — Canonical NPZ keys/dtypes must remain intact while handling metadata.
  - CONFIG-001 — Do not reorder legacy config initialization pathways.
  - TYPE-PATH-001 — Normalize any new Path usage.
  - OVERSAMPLING-001 — Leave dense overlap thresholds unchanged.
  - POLICY-001 — Ensure PyTorch dependency expectations stay satisfied (no optional gating).

Pointers:
  - scripts/tools/transpose_rename_convert_tool.py:1 (existing canonicalization logic)
  - scripts/tools/generate_patches_tool.py:1 (patch generation workflow)
  - scripts/simulation/simulate_and_save.py:112 (metadata saved during simulation)
  - docs/TESTING_GUIDE.md:210 (Phase G study selectors section)
  - docs/development/TEST_SUITE_INDEX.md:62 (tooling tests registry entry)

Next Up (optional):
  - Execute sparse view pipeline once dense run and metadata tooling are green.

Doc Sync Plan (Conditional):
  - After GREEN: `pytest --collect-only tests/tools/test_transpose_rename_convert_tool.py -vv > plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-07T130500Z/phase_c_metadata_pipeline/collect/pytest_transpose_metadata_collect.log`
  - After GREEN: `pytest --collect-only tests/tools/test_generate_patches_tool.py -vv > plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-07T130500Z/phase_c_metadata_pipeline/collect/pytest_patches_metadata_collect.log`
  - Update `docs/TESTING_GUIDE.md` §2 and `docs/development/TEST_SUITE_INDEX.md` to document the two new selectors after successful runs.

Mapped Tests Guardrail: Ensure both selectors above collect at least one test (`pytest --collect-only …` commands in Doc Sync Plan).

Hard Gate: Do not mark the attempt done unless Phase C completes without metadata-related ValueError and both pytest selectors pass; if later pipeline phases fail, log blocker and stop.
