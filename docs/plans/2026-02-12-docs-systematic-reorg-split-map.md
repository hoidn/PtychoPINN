# Documentation Systematic Reorg + Split-Map Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Produce a repeatable, evidence-backed procedure to review documentation, identify clutter, generate a split map (core vs supporting docs), and execute reorganization without breaking discoverability.

**Architecture:** Use a docs-governance pipeline with four stages: (1) inventory + scoring, (2) domain clustering + split-map proposal, (3) migration/redirection implementation, and (4) verification + adoption. Keep canonical specs unchanged, move high-noise content into supporting/reference files, and enforce navigation through `docs/index.md` plus machine-checked references.

**Tech Stack:** Markdown docs, Python CLI utilities, `pytest` docs checks, `rg`/shell tooling.

---

### Task 1: Establish Review Rubric and Scope Contract

**Files:**
- Create: `docs/governance/docs_reorg_rubric.md`
- Modify: `docs/index.md`
- Test: `tests/docs/test_docs_reorg_rubric.py`

**Step 1: Write the failing test**

```python
from pathlib import Path


def test_docs_reorg_rubric_has_required_sections():
    text = Path("docs/governance/docs_reorg_rubric.md").read_text(encoding="utf-8")
    required = [
        "## Inclusion Criteria",
        "## Exclusion Criteria",
        "## Scoring Dimensions",
        "## Split Decision Rules",
    ]
    for marker in required:
        assert marker in text
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/docs/test_docs_reorg_rubric.py::test_docs_reorg_rubric_has_required_sections -v`
Expected: FAIL because rubric doc does not exist.

**Step 3: Write minimal implementation**

Create `docs/governance/docs_reorg_rubric.md` with:
- Scope boundaries (what gets reorganized now vs later)
- 1-5 scoring for relevance, noise, duplication, staleness risk, discoverability risk
- Decision thresholds (keep/trim/split/archive)

Add a short entry in `docs/index.md` under governance docs pointing to the rubric.

**Step 4: Run test to verify it passes**

Run: `pytest tests/docs/test_docs_reorg_rubric.py::test_docs_reorg_rubric_has_required_sections -v`
Expected: PASS.

**Step 5: Commit**

```bash
git add docs/governance/docs_reorg_rubric.md docs/index.md tests/docs/test_docs_reorg_rubric.py
git commit -m "docs: add reorg rubric and index entry"
```

### Task 2: Build Documentation Inventory and Metrics Extractor

**Files:**
- Create: `scripts/docs/build_docs_inventory.py`
- Create: `docs/analysis/docs_inventory.csv`
- Test: `tests/docs/test_build_docs_inventory.py`

**Step 1: Write the failing test**

```python
from pathlib import Path
from scripts.docs.build_docs_inventory import collect_inventory


def test_collect_inventory_includes_md_files(tmp_path):
    (tmp_path / "a.md").write_text("hello", encoding="utf-8")
    rows = collect_inventory(root=tmp_path)
    assert any(r["path"].endswith("a.md") for r in rows)
    assert all("token_count" in r for r in rows)
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/docs/test_build_docs_inventory.py::test_collect_inventory_includes_md_files -v`
Expected: FAIL because module/function does not exist.

**Step 3: Write minimal implementation**

Implement `collect_inventory()` returning per-file fields:
- `path`, `domain`, `bytes`, `line_count`, `token_count_estimate`, `last_modified`

Add CLI mode:

```bash
python scripts/docs/build_docs_inventory.py --root docs --out docs/analysis/docs_inventory.csv
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/docs/test_build_docs_inventory.py::test_collect_inventory_includes_md_files -v`
Expected: PASS.

**Step 5: Commit**

```bash
git add scripts/docs/build_docs_inventory.py docs/analysis/docs_inventory.csv tests/docs/test_build_docs_inventory.py
git commit -m "docs: add inventory extractor for markdown corpus"
```

### Task 3: Add Relevance/Noise Scoring Pass

**Files:**
- Create: `scripts/docs/score_docs_for_reorg.py`
- Create: `docs/analysis/docs_reorg_scores.csv`
- Test: `tests/docs/test_score_docs_for_reorg.py`

**Step 1: Write the failing test**

```python
from scripts.docs.score_docs_for_reorg import classify_doc


def test_classify_doc_returns_decision_fields():
    row = classify_doc(path="docs/findings.md", token_count=10000, heading_count=20)
    assert "action" in row
    assert row["action"] in {"keep", "trim", "split", "archive"}
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/docs/test_score_docs_for_reorg.py::test_classify_doc_returns_decision_fields -v`
Expected: FAIL because script/function does not exist.

**Step 3: Write minimal implementation**

Implement deterministic scoring rules from rubric. Output columns:
- `path`, `relevance_score`, `noise_score`, `duplication_score`, `discoverability_risk`, `action`, `reason`

CLI:

```bash
python scripts/docs/score_docs_for_reorg.py \
  --inventory docs/analysis/docs_inventory.csv \
  --out docs/analysis/docs_reorg_scores.csv
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/docs/test_score_docs_for_reorg.py::test_classify_doc_returns_decision_fields -v`
Expected: PASS.

**Step 5: Commit**

```bash
git add scripts/docs/score_docs_for_reorg.py docs/analysis/docs_reorg_scores.csv tests/docs/test_score_docs_for_reorg.py
git commit -m "docs: add scoring pass for reorg decisions"
```

### Task 4: Generate Domain-Based Split Map (Core vs Supporting)

**Files:**
- Create: `scripts/docs/generate_split_map.py`
- Create: `docs/analysis/docs_split_map.md`
- Test: `tests/docs/test_generate_split_map.py`

**Step 1: Write the failing test**

```python
from pathlib import Path
from scripts.docs.generate_split_map import build_split_map


def test_split_map_contains_core_and_supporting_sections(tmp_path):
    output = build_split_map([])
    assert "## Core Documents" in output
    assert "## Supporting Documents" in output
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/docs/test_generate_split_map.py::test_split_map_contains_core_and_supporting_sections -v`
Expected: FAIL because split-map generator does not exist.

**Step 3: Write minimal implementation**

Generate a human-reviewable map with sections:
- Domain summaries (workflows, specs, debugging, studies, governance)
- Core docs (read-first)
- Supporting docs (read-when-relevant)
- Proposed moves/splits with source -> target mapping

CLI:

```bash
python scripts/docs/generate_split_map.py \
  --scores docs/analysis/docs_reorg_scores.csv \
  --out docs/analysis/docs_split_map.md
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/docs/test_generate_split_map.py::test_split_map_contains_core_and_supporting_sections -v`
Expected: PASS.

**Step 5: Commit**

```bash
git add scripts/docs/generate_split_map.py docs/analysis/docs_split_map.md tests/docs/test_generate_split_map.py
git commit -m "docs: add split-map generator for core/supporting docs"
```

### Task 5: Define Migration Ledger and Redirection Policy

**Files:**
- Create: `docs/governance/docs_reorg_migration_ledger.md`
- Create: `docs/governance/docs_redirect_policy.md`
- Modify: `docs/index.md`
- Test: `tests/docs/test_docs_redirect_policy.py`

**Step 1: Write the failing test**

```python
from pathlib import Path


def test_redirect_policy_defines_required_redirect_types():
    text = Path("docs/governance/docs_redirect_policy.md").read_text(encoding="utf-8")
    assert "Moved" in text
    assert "Split" in text
    assert "Deprecated" in text
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/docs/test_docs_redirect_policy.py::test_redirect_policy_defines_required_redirect_types -v`
Expected: FAIL because policy doc does not exist.

**Step 3: Write minimal implementation**

Create:
- `docs_reorg_migration_ledger.md` with rows: old path, new path(s), status, owner, date
- `docs_redirect_policy.md` with redirect templates and retention timelines

Update `docs/index.md` to include “Reorg in progress” and canonical pointers.

**Step 4: Run test to verify it passes**

Run: `pytest tests/docs/test_docs_redirect_policy.py::test_redirect_policy_defines_required_redirect_types -v`
Expected: PASS.

**Step 5: Commit**

```bash
git add docs/governance/docs_reorg_migration_ledger.md docs/governance/docs_redirect_policy.md docs/index.md tests/docs/test_docs_redirect_policy.py
git commit -m "docs: add migration ledger and redirect policy"
```

### Task 6: Pilot Reorganization on High-Noise Docs

**Files:**
- Modify: `docs/findings.md`
- Create: `docs/findings_core.md`
- Create: `docs/findings_reference.md`
- Modify: `docs/workflows/pytorch.md`
- Create: `docs/workflows/pytorch_reference.md`
- Modify: `docs/DEVELOPER_GUIDE.md`
- Create: `docs/DEVELOPER_GUIDE_reference.md`
- Test: `tests/docs/test_docs_pilot_reorg_navigation.py`

**Step 1: Write the failing test**

```python
from pathlib import Path


def test_core_docs_link_to_reference_companions():
    pairs = [
        ("docs/findings_core.md", "docs/findings_reference.md"),
        ("docs/workflows/pytorch.md", "docs/workflows/pytorch_reference.md"),
    ]
    for core, ref in pairs:
        text = Path(core).read_text(encoding="utf-8")
        assert ref in text
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/docs/test_docs_pilot_reorg_navigation.py::test_core_docs_link_to_reference_companions -v`
Expected: FAIL because companion docs/links do not exist.

**Step 3: Write minimal implementation**

Apply pilot split pattern:
- Keep must-know constraints + active policy in core docs
- Move historical deep detail into reference companions
- Add “Read when relevant” pointers and preserve old anchors where feasible

**Step 4: Run test to verify it passes**

Run: `pytest tests/docs/test_docs_pilot_reorg_navigation.py::test_core_docs_link_to_reference_companions -v`
Expected: PASS.

**Step 5: Commit**

```bash
git add docs/findings.md docs/findings_core.md docs/findings_reference.md docs/workflows/pytorch.md docs/workflows/pytorch_reference.md docs/DEVELOPER_GUIDE.md docs/DEVELOPER_GUIDE_reference.md tests/docs/test_docs_pilot_reorg_navigation.py
git commit -m "docs: pilot split of high-noise docs into core/reference"
```

### Task 7: Add Discoverability + Link Integrity Gate

**Files:**
- Create: `scripts/docs/verify_docs_navigation.py`
- Create: `tests/docs/test_verify_docs_navigation.py`
- Modify: `docs/TESTING_GUIDE.md`

**Step 1: Write the failing test**

```python
from scripts.docs.verify_docs_navigation import validate_index


def test_validate_index_reports_missing_targets(tmp_path):
    issues = validate_index(index_path=tmp_path / "index.md")
    assert isinstance(issues, list)
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/docs/test_verify_docs_navigation.py::test_validate_index_reports_missing_targets -v`
Expected: FAIL because validator does not exist.

**Step 3: Write minimal implementation**

Implement checks for:
- Broken markdown links
- Orphan core docs (not reachable from `docs/index.md`)
- Redirect ledger entries with missing targets

Document selector in `docs/TESTING_GUIDE.md`.

**Step 4: Run test to verify it passes**

Run: `pytest tests/docs/test_verify_docs_navigation.py::test_validate_index_reports_missing_targets -v`
Expected: PASS.

**Step 5: Commit**

```bash
git add scripts/docs/verify_docs_navigation.py tests/docs/test_verify_docs_navigation.py docs/TESTING_GUIDE.md
git commit -m "test(docs): add navigation and link integrity gate"
```

### Task 8: Full Procedure Dry-Run + Evidence Bundle

**Files:**
- Create: `docs/analysis/docs_reorg_runbook.md`
- Create: `.artifacts/docs_reorg/<timestamp>/` (git-ignored evidence)
- Modify: `docs/index.md`
- Test: `tests/docs/test_docs_reorg_runbook.py`

**Step 1: Write the failing test**

```python
from pathlib import Path


def test_reorg_runbook_contains_end_to_end_commands():
    text = Path("docs/analysis/docs_reorg_runbook.md").read_text(encoding="utf-8")
    assert "build_docs_inventory.py" in text
    assert "score_docs_for_reorg.py" in text
    assert "generate_split_map.py" in text
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/docs/test_docs_reorg_runbook.py::test_reorg_runbook_contains_end_to_end_commands -v`
Expected: FAIL because runbook does not exist.

**Step 3: Write minimal implementation**

Create runbook with exact pipeline commands:

```bash
python scripts/docs/build_docs_inventory.py --root docs --out docs/analysis/docs_inventory.csv
python scripts/docs/score_docs_for_reorg.py --inventory docs/analysis/docs_inventory.csv --out docs/analysis/docs_reorg_scores.csv
python scripts/docs/generate_split_map.py --scores docs/analysis/docs_reorg_scores.csv --out docs/analysis/docs_split_map.md
python scripts/docs/verify_docs_navigation.py --index docs/index.md --ledger docs/governance/docs_reorg_migration_ledger.md
pytest tests/docs -v
```

Store heavy artifacts under `.artifacts/docs_reorg/<timestamp>/` and link from runbook.

**Step 4: Run test to verify it passes**

Run: `pytest tests/docs/test_docs_reorg_runbook.py::test_reorg_runbook_contains_end_to_end_commands -v`
Expected: PASS.

**Step 5: Commit**

```bash
git add docs/analysis/docs_reorg_runbook.md docs/index.md tests/docs/test_docs_reorg_runbook.py
git commit -m "docs: add end-to-end reorg runbook and procedure"
```

## Global Verification (before declaring complete)

Run:

```bash
pytest tests/docs -v
python scripts/docs/verify_docs_navigation.py --index docs/index.md --ledger docs/governance/docs_reorg_migration_ledger.md
```

Expected:
- All docs tests PASS
- No broken links
- No orphaned core docs
- All migration ledger targets resolve

## Split-Map Output Contract

The generated `docs/analysis/docs_split_map.md` must include:
- A top-level "Core Documents" section (read-first)
- A top-level "Supporting Documents" section (read-when-relevant)
- A "Proposed Splits" table with explicit old->new path mapping
- A "Risk Flags" section for discoverability and fragmentation hazards

## Discoverability Guardrails (YAGNI + anti-fragmentation)

- Keep a single source of navigation truth: `docs/index.md`
- Use no more than two tiers for each domain during initial rollout (`core`, `reference`)
- Prefer splitting only files exceeding threshold (e.g., >3k-4k tokens + high noise score)
- Do not split specs in this initiative; only add better pointers from index
