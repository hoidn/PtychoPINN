# REFACTOR-MEMOIZE-CORE-001 — Phase C: document core cache helper + capture regression logs

**Summary:** Close out the memoization refactor by documenting the new `ptycho/cache.py` module and producing the missing pytest evidence for the synthetic helper selectors.

**Focus:** REFACTOR-MEMOIZE-CORE-001 — Move RawData memoization decorator into core module

**Branch:** paper

**Mapped tests:**
- `pytest --collect-only tests/scripts/test_synthetic_helpers.py -q`
- `pytest tests/scripts/test_synthetic_helpers.py::test_simulate_nongrid_seeded -v`
- `pytest tests/scripts/test_synthetic_helpers_cli_smoke.py -v`

**Artifacts:** `plans/active/REFACTOR-MEMOIZE-CORE-001/reports/2026-01-15T232107Z/`

---

## Do Now

- REFACTOR-MEMOIZE-CORE-001.C1-C2
  - Implement: `docs/index.md::Core Module Documentation` — insert a `ptycho/cache.py` entry (just below `ptycho/diffsim.py`) explaining that RawData caching lives there now, cite `docs/architecture.md §Data Pipeline`, and cross-link the shim for legacy scripts.
  - Implement: `scripts/simulation/README.md::Key Scripts/Cache Notes` — update the `synthetic_helpers.py` row (or add a short subsection) to state that nongrid simulation caching is provided by `ptycho.cache.memoize_raw_data`, describe the default cache root `.artifacts/synthetic_helpers/cache`, and include guidance for overriding with `--cache-dir` / `use_cache=False`.
  - Validate: `pytest --collect-only tests/scripts/test_synthetic_helpers.py -q`, `pytest tests/scripts/test_synthetic_helpers.py::test_simulate_nongrid_seeded -v`, `pytest tests/scripts/test_synthetic_helpers_cli_smoke.py -v` (archive logs under the artifacts path).
  - Artifacts: `plans/active/REFACTOR-MEMOIZE-CORE-001/reports/2026-01-15T232107Z/`

## How-To Map
1. Edit `docs/index.md` under **Core Module Documentation** and add a `ptycho/cache.py` subsection that (a) states the module hosts `memoize_raw_data`, (b) references `docs/architecture.md` (data loading pipeline) for normative behavior, and (c) mentions the compatibility shim (`scripts/simulation/cache_utils.py`).
2. Update `scripts/simulation/README.md` — within the "Key Scripts" table plus the paragraph that introduces `synthetic_helpers.py`, describe how the helpers use `ptycho.cache.memoize_raw_data`, clarify the cache directory (`.artifacts/synthetic_helpers/cache`), and explain how to bypass caching via `--no-cache`/`use_cache=False` or a custom `cache_dir`.
3. Collect selectors once to satisfy the guardrail:
   ```bash
   pytest --collect-only tests/scripts/test_synthetic_helpers.py -q \
     | tee plans/active/REFACTOR-MEMOIZE-CORE-001/reports/2026-01-15T232107Z/pytest_collect.log
   ```
4. Run the focused unit and CLI smoke tests, capturing stdout/stderr per TESTING_GUIDE:
   ```bash
   pytest tests/scripts/test_synthetic_helpers.py::test_simulate_nongrid_seeded -v \
     | tee plans/active/REFACTOR-MEMOIZE-CORE-001/reports/2026-01-15T232107Z/pytest_synthetic_helpers.log
   pytest tests/scripts/test_synthetic_helpers_cli_smoke.py -v \
     | tee plans/active/REFACTOR-MEMOIZE-CORE-001/reports/2026-01-15T232107Z/pytest_cli_smoke.log
   ```
5. If warnings about `DeprecationWarning` surface, document them in the logs (no filter changes) so reviewers can confirm the shim only fires once.

## Pitfalls To Avoid
1. Do not move or delete `scripts/simulation/cache_utils.py`; it must stay as the shim until downstream scripts migrate.
2. Keep `docs/index.md` edits scoped to the new entry—no reformatting of unrelated sections.
3. Avoid changing `ptycho/cache.py`; this loop is docs/tests only.
4. Use repo-root paths when running pytest so fixtures resolve (per PYTHON-ENV-001).
5. Capture raw pytest output with `tee`; no truncation or `-q` on the execution runs.
6. Treat `DeprecationWarning` as informational; do not silence it via `filterwarnings`.
7. Ensure `.artifacts/` paths stay gitignored—never add cache outputs to version control.
8. Resist adding new tests; reuse the mapped selectors unless a regression forces additional coverage.
9. If documentation references an authoritative spec, cite the exact section instead of paraphrasing the math/contract.
10. Verify the collect-only selector actually lists at least one node; failure requires stopping and reporting before running the full tests.

## If Blocked
- If the docs disagree on where caching lives, stop editing, capture the conflicting text snippet in `plans/active/REFACTOR-MEMOIZE-CORE-001/reports/2026-01-15T232107Z/blocker.md`, and update `docs/fix_plan.md` Attempts History with the citation.
- If either pytest command fails, keep the log file, leave the repo untouched, and add the failure summary + command to the same blocker file so we can triage before the next loop.

## Findings Applied
| Finding ID | Adherence |
|------------|-----------|
| ANTIPATTERN-001 | Documentation will call out that `ptycho/cache.py` is a side-effect-free core module per the stable-module policy. |
| MIGRATION-001 | Centralizes RawData helpers in `ptycho/` and documents the shim so no new script-level copies creep in. |

## Pointers
- `docs/architecture.md:1` — shared architecture + stable-module policy referenced in the new docs entry.
- `plans/active/REFACTOR-MEMOIZE-CORE-001/implementation.md:84` — checklist detailing Phase C requirements.
- `ptycho/cache.py:1` — canonical implementation of `memoize_raw_data` to summarize in docs.
- `scripts/simulation/synthetic_helpers.py:14` — shows current decorator usage and cache root to describe in README.
- `docs/TESTING_GUIDE.md:1` — command formatting + evidence requirements for pytest logs.

## Next Up
- After docs/tests land, mark C3 done by refreshing `plans/active/REFACTOR-MEMOIZE-CORE-001/summary.md` and `docs/fix_plan.md` with the final evidence pointer.
