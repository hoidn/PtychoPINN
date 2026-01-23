Summary: Capture a maintainer-ready README that documents the photon_grid_study_20250826_152459 simulate→train→infer flow and ties every artifact back to the manifest so Maintainer <2> can rerun dose_experiments without touching production code.
Focus: DEBUG-SIM-LINES-DOSE-001 — Legacy dose_experiments ground-truth bundle
Branch: dose_experiments
Mapped tests: pytest tests/test_generic_loader.py::test_generic_loader -q
Artifacts: plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T001931Z/

Do Now (hard validity contract)
- Focus ID: DEBUG-SIM-LINES-DOSE-001.B1
- Implement: plans/active/DEBUG-SIM-LINES-DOSE-001/bin/generate_legacy_readme.py::main — new CLI that loads the Phase-A manifest JSON plus `plans/active/seed/reports/2026-01-22T024002Z/dose_baseline_summary.json`, then emits `README.md` in the artifacts path with (1) scenario/env overview, (2) canonical simulate/train/infer commands from `inbox/response_prepare_d0_response.md`, (3) environment warnings about TF/Keras 2.x per maintainer request, and (4) a provenance table mapping every dataset/baseline/inference artifact to size + SHA256 straight from the manifest so specs/data_contracts.md §RawData NPZ remains satisfied.
- Pytest: pytest tests/test_generic_loader.py::test_generic_loader -q
- Artifacts: plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T001931Z/

How-To Map
1. `mkdir -p plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T001931Z && rsync -a plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T001018Z/ground_truth_manifest.* plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T001931Z/` — keep the README, manifest, and CSV beside each other for Maintainer <2>.
2. Author `plans/active/DEBUG-SIM-LINES-DOSE-001/bin/generate_legacy_readme.py` using argparse + pathlib: load the manifest JSON, fold dataset/baseline/pinn entries into stage records (Simulation vs Training vs Inference), pull metric + parameter details from `dose_baseline_summary.json`, and render Markdown sections (Overview → Environment requirements referencing maintainer's TF/Keras 2.x constraint → Simulation command block referencing `notebooks/dose_dependence.ipynb`/`notebooks/dose.py` → Training/inference bash blocks from the maintainer response → Artifact provenance table driven by manifest data). Include inline spec cites (e.g., `specs/data_contracts.md §RawData NPZ`) in the README text so reviewers trace requirements.
3. `python plans/active/DEBUG-SIM-LINES-DOSE-001/bin/generate_legacy_readme.py --manifest plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T001018Z/ground_truth_manifest.json --baseline-summary plans/active/seed/reports/2026-01-22T024002Z/dose_baseline_summary.json --output-dir plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T001931Z | tee plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T001931Z/generate_readme.log` — verify the README includes the command sections, environment callouts, and the provenance table (size MB + SHA columns).
4. `pytest tests/test_generic_loader.py::test_generic_loader -q | tee plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T001931Z/pytest_loader.log` — proves NPZ references in the README still match a loader that queries specs/data_contracts.md §RawData NPZ (see docs/TESTING_GUIDE.md for selector context).

Pitfalls To Avoid
- No edits under shipped packages (`ptycho/*`, `scripts/*.py`, etc.); keep all new code inside `plans/active/DEBUG-SIM-LINES-DOSE-001/bin/`.
- Do not hand-type dataset sizes or hashes: always source from the manifest JSON to avoid drift.
- Keep README commands relative to repo root so Maintainer <2> can copy/paste without rewriting paths.
- Call out the TF/Keras 3.x incompatibility noted in the request so expectations are clear.
- Reference the exact spec sections (e.g., `specs/data_contracts.md §RawData NPZ`) instead of paraphrasing the schema.
- Preserve ASCII-only Markdown; avoid Unicode probes or em dashes in headings.
- Store every CLI/stdout log under the artifacts path; nothing should land in repo root or /tmp.
- Do not copy the 200 MB datasets into the new report directory yet; README + manifest only this loop.
- Ensure argparse rejects missing files early so any manifest path typo surfaces fast.
- Keep dataset provenance table sorted by photon dose (data_p1e3 → data_p1e9) for readability.

If Blocked
- If the manifest or baseline summary cannot be read, stop, collect the stack trace + `ls -l` of the offending path into `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T001931Z/blocker.md`, update `docs/fix_plan.md` Attempts History with the failure signature, and ping Galph before retrying.

Findings Applied (Mandatory)
- No relevant findings in the knowledge base.

Pointers
- inbox/request_dose_experiments_ground_truth_2026-01-22T014445Z.md:1 — maintainer scope + preferred drop path.
- inbox/response_prepare_d0_response.md:1 — authoritative simulate/train/infer commands + metrics to cite.
- specs/data_contracts.md:3 — NPZ key requirements for datasets listed in the README.
- plans/active/DEBUG-SIM-LINES-DOSE-001/implementation.md:18 — Phase B checklist + dependencies.
- docs/TESTING_GUIDE.md:1 — pytest selector reference for `tests/test_generic_loader.py`.

Next Up (optional)
- DEBUG-SIM-LINES-DOSE-001.B2 — polish provenance table + doc cross-links if Maintainer <2> needs additional columns once README lands.
