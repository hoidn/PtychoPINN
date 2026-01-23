Summary: Send Maintainer <2> the final ground-truth bundle response that cites the drop root, README/manifest assets, verification logs, tarball SHA, and the fresh loader pytest log.
Focus: DEBUG-SIM-LINES-DOSE-001.D1 — Legacy dose_experiments ground-truth maintainer response
Branch: dose_experiments
Mapped tests: pytest tests/test_generic_loader.py::test_generic_loader -q
Artifacts: plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T004049Z/

Do Now (hard validity contract)
- Focus ID: DEBUG-SIM-LINES-DOSE-001.D1
- Implement: inbox/response_dose_experiments_ground_truth.md::MaintainerReply — author a maintainer-facing note that answers the original inbox request, links to `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-22T014445Z/dose_experiments_ground_truth/`, summarizes the README/manifest artifacts, restates the 7 dataset + baseline tables, cites `bundle_verification.{json,md}` (total files, tarball size, SHA256), includes delivery instructions (`tar -xzf` & sha verify), and logs the rerun of `pytest tests/test_generic_loader.py::test_generic_loader -q`.
- Pytest: pytest tests/test_generic_loader.py::test_generic_loader -q
- Artifacts: plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T004049Z/

How-To Map
1. export AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md && mkdir -p plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T004049Z
2. pytest tests/test_generic_loader.py::test_generic_loader -q | tee plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T004049Z/pytest_loader.log
3. sha256sum plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-22T014445Z/dose_experiments_ground_truth.tar.gz | tee plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T004049Z/dose_experiments_ground_truth.tar.gz.sha256.check
4. Draft inbox/response_dose_experiments_ground_truth.md with sections for: (a) Delivery Summary referencing the drop root + docs (`docs/README.md`, manifest MD/JSON, baseline summary JSON), (b) Verification referencing plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T002823Z/bundle_verification.{json,md} (15/15 files, 278.18 MB, tarball SHA `7fe5e14e...`), (c) Tests citing the new pytest log + command, (d) How-To instructions for untarring + verifying SHA256 + re-running the helper CLIs under plans/active/DEBUG-SIM-LINES-DOSE-001/bin/, (e) Dataset/Baseline tables (reuse the entries already present in docs/README.md §6) and NPZ key requirements referencing specs/data_contracts.md §RawData NPZ, and (f) Next steps / confirmation ask.

Pitfalls To Avoid
- Do not mutate the delivered bundle or regenerate files; only reference the immutable drop built in 2026-01-23T002823Z artifacts.
- Keep the maintainer response concise but complete—avoid omitting tarball SHA, sizes, or script paths.
- Ensure the dataset/baseline tables preserve photon-dose ordering (1e3→1e9) and the exact SHA256 strings from the manifest/README, no retyping errors.
- Cite `specs/data_contracts.md` for NPZ key requirements instead of paraphrasing or inventing new schema text.
- Reference the exact README + manifest paths under docs/ inside the drop so Maintainer <2> can navigate without guessing.
- Capture the pytest output via tee; do not summarize from memory.
- Avoid touching shipped modules (`ptycho/*`, `scripts/`); all edits stay under inbox/ + plans/active/DEBUG-SIM-LINES-DOSE-001/.
- Do not delete or rename the tarball/sha files when recomputing sha256sum—write the check output to the new artifacts directory only.
- Maintain device-neutral language (no GPU assumptions) when describing how to re-run the helper CLIs.
- No environment changes (pip install, conda edits) per Environment Freeze; if TF loader breaks, stop and escalate.

If Blocked
- If the tarball or bundle directory is missing/corrupted, stop immediately, log the issue (path, error, sha mismatch) in plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T004049Z/blocker.md, update docs/fix_plan.md Attempts History plus galph_memory.md with the failure signature, and await maintainer guidance before rewriting the response.

Findings Applied (Mandatory)
- No relevant findings in the knowledge base.

Pointers
- docs/fix_plan.md:6 — Current focus statement and DEBUG-SIM-LINES-DOSE-001.D1 checklist.
- inbox/request_dose_experiments_ground_truth_2026-01-22T014445Z.md:1 — Maintainer requirements and drop expectations.
- plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-22T014445Z/dose_experiments_ground_truth/docs/README.md:1 — Canonical README sections (commands, tables, delivery details) to mirror in the response.
- plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T002823Z/bundle_verification.md:1 — Verification summary (15/15 files, tarball SHA/size) to cite verbatim.
- specs/data_contracts.md:5 — RawData NPZ key requirements for the dataset schema callout.
- docs/TESTING_GUIDE.md:5 — Pytest `tests/test_generic_loader.py` selector reference.

Next Up (optional)
- Close DEBUG-SIM-LINES-DOSE-001 in docs/fix_plan.md and galph_memory.md after Maintainer <2> acknowledges the bundle.
