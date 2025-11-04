# Phase E6 Dense/Baseline Evidence — Loop Plan (2025-11-06T17:05:00Z)

## Objective
Resume Phase E6 dose=1000 dense (gs2) and baseline (gs1) deterministic runs now that the PyTorch Path bug is fixed. Land full SHA parity coverage in tests, capture RED→GREEN proof, and archive the real-run artifacts with checksum verification.

## Context Refresh
- Previous loop fixed `ptycho_torch/workflows/components.py` Path normalization (TYPE-PATH-001) but deferred real-run evidence.
- Findings ledger lacks TYPE-PATH-001 entry; add alongside existing POLICY-001/CONFIG-001/DATA-001/OVERSAMPLING-001 references.
- `plans/.../bin/archive_phase_e_outputs.py` is the canonical tool for manifest copying + digest verification.

## Deliverables This Loop
1. Strengthened `test_training_cli_records_bundle_path` comparing stdout SHA lines vs manifest `bundle_sha256`.
2. RED→GREEN logs for the selector above (captured under this hub), plus collect-only proof.
3. Deterministic CLI executions for dense gs2 and baseline gs1 with logs, manifests, bundles, and SHA proof archived here.
4. Updated `analysis/summary.md` noting CLI evidence, archive outputs, and acknowledgement of TYPE-PATH-001 with doc sync instructions.
5. `docs/findings.md` updated with TYPE-PATH-001 (links to artifact hub + remediation notes).

## Step-by-Step
1. **Prep (idempotent):** Ensure Phase C/D assets exist (re-run generation/overlap commands only if `tmp/phase_c_f2_cli` or `tmp/phase_d_f2_cli` missing). Capture logs in `prep/`.
2. **TDD:**
   - Edit `tests/study/test_dose_overlap_training.py::test_training_cli_records_bundle_path` to assert stdout SHA matches manifest entry.
   - Run RED selector; tee to `red/pytest_training_cli_sha_red.log`.
   - Apply necessary adjustments in `studies/fly64_dose_overlap/training.py::main` (or helper) if CLI output needs normalization; rerun selector for GREEN.
   - Collect targeted selectors (`--collect-only`) and the focused `training_cli` subset once GREEN.
3. **Deterministic Runs:** Execute dense gs2 and baseline gs1 CLI runs with deterministic flags; tee logs to `cli/`.
4. **Archive + Proof:** Use `bin/archive_phase_e_outputs.py` to copy artifacts into `data/` and emit checksum proof (`analysis/bundle_checksums.txt`). Verify script output reports sha matches manifest values.
5. **Documentation:**
   - Append CLI + checksum results to `analysis/summary.md`.
   - Add TYPE-PATH-001 to `docs/findings.md` (include short description + mitigation referencing this hub).
   - Update `docs/fix_plan.md` Attempts History via standard template.

## References
- specs/ptychodus_api_spec.md §4.6 (bundle persistence + SHA contract)
- docs/TESTING_GUIDE.md §3.2 (Phase E selectors — set `AUTHORITATIVE_CMDS_DOC`)
- plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/test_strategy.md §268 (Phase E6 evidence definition)
- docs/findings.md (POLICY-001 / CONFIG-001 / DATA-001 / OVERSAMPLING-001)

## Exit Criteria
- Strengthened test GREEN with SHA parity assertion.
- Dense + baseline runs produce deterministic manifests and checksum proof in this hub.
- TYPE-PATH-001 recorded in findings with artifact link.
- `summary.md` details CLI evidence + next-step pointers (sparse view backlog).
