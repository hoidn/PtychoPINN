# Paper Efficiency Table Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. Do not create worktrees.

**Goal:** Build a repo-local NeurIPS paper efficiency table that compiles parameter counts, provenance-backed training runtime, hardware metadata, and available inference-throughput fields from existing CDI/CNS evidence without rerunning training.

**Architecture:** Add a focused table-generation module that reads existing durable artifacts, normalizes row-level efficiency fields with source-path/source-field lineage, and emits JSON/CSV/TeX plus a durable summary. Keep manuscript editing out of this item; `hybrid_resnet_neurips_first_draft.tex` may consume the generated TeX in a later paper-refresh or appendix task, but this backlog item only creates the authoritative table assets and index updates.

**Tech Stack:** Python standard library, JSON/CSV/LaTeX table rendering, existing NeurIPS table artifacts, pytest, existing `scripts/studies/paper_results_refresh.py` refresh entrypoint.

---

## Source Of Truth

- Backlog item: `docs/backlog/active/2026-05-05-paper-efficiency-table.md`
- Paper evidence design: `docs/plans/NEURIPS-HYBRID-RESNET-2026/paper_evidence_package_design.md`
- Evidence index: `docs/plans/NEURIPS-HYBRID-RESNET-2026/paper_evidence_index.md`
- Evidence matrix: `docs/plans/NEURIPS-HYBRID-RESNET-2026/evidence_matrix.md`
- Model variant index: `docs/plans/NEURIPS-HYBRID-RESNET-2026/model_variant_index.json`
- CDI table source: `docs/plans/NEURIPS-HYBRID-RESNET-2026/tables/cdi_lines128_metrics_extended.json`
- CNS table source: `docs/plans/NEURIPS-HYBRID-RESNET-2026/tables/pdebench_cns_matched_condition_metrics.json`
- Current model-config appendix plan: `docs/plans/NEURIPS-HYBRID-RESNET-2026/2026-05-06-model-config-appendix-table-plan.md`

## Required Interpretation

- Parameter counts are compiled from existing evidence. Do not train or tune a model to obtain parameter counts.
- Training runtime is provenance context unless a common measurement protocol is proven in this item.
- Inference throughput must be labeled per row as `measured`, `missing`, or `not_comparable`.
- Runtime source fields such as `runtime_sec`, `command_wall_time_sec`, `train_wall_time_sec`, or `evaluation_wall_time_sec` must remain visible in machine-readable outputs.
- Do not present heterogeneous launch runtimes as normalized inference throughput.
- Do not combine CDI, CNS, BRDT, or WaveBench into one model-ranking table. Cross-lane output is allowed only when grouped by benchmark and claim boundary.
- Include the user-approved BRDT 40-epoch rows as secondary paper evidence. Preserve their `paper_approved_secondary_brdt` claim boundary and do not merge them into CDI/CNS rankings.
- Do not modify `/home/ollie/Documents/neurips`; that remains Phase 5 paper-facing bundle work.
- Do not modify `docs/plans/NEURIPS-HYBRID-RESNET-2026/hybrid_resnet_neurips_first_draft.tex` in this item. The generated `paper_efficiency_table.tex` is the manuscript-ready payload for later insertion.

## File Structure

- Create: `scripts/studies/paper_efficiency_table.py`
  - Owns row extraction, runtime/source-field normalization, throughput-status classification, grouped JSON/CSV/TeX rendering, and summary rendering.
- Modify: `scripts/studies/paper_results_refresh.py`
  - Adds `--write-efficiency-table` and calls `write_paper_efficiency_table`.
- Create: `tests/studies/test_paper_efficiency_table.py`
  - Unit tests for row normalization, runtime-field lineage, throughput classification, grouping, and TeX rendering.
- Modify: `tests/studies/test_paper_results_refresh.py`
  - Adds a focused CLI wiring test for `--write-efficiency-table`.
- Generate: `docs/plans/NEURIPS-HYBRID-RESNET-2026/paper_efficiency_table_summary.md`
- Generate: `docs/plans/NEURIPS-HYBRID-RESNET-2026/tables/paper_efficiency_table.json`
- Generate: `docs/plans/NEURIPS-HYBRID-RESNET-2026/tables/paper_efficiency_table.csv`
- Generate: `docs/plans/NEURIPS-HYBRID-RESNET-2026/tables/paper_efficiency_table.tex`
- Modify: `docs/plans/NEURIPS-HYBRID-RESNET-2026/evidence_matrix.md`
- Modify: `docs/plans/NEURIPS-HYBRID-RESNET-2026/paper_evidence_index.md`
- Modify: `docs/studies/index.md`

## Task 1: Add Efficiency Table Unit Tests

**Files:**
- Create: `tests/studies/test_paper_efficiency_table.py`

- [ ] **Step 1: Write runtime-lineage normalization test**

Create a small row fixture with multiple possible runtime fields. Assert the chosen field preserves value, source field name, source path, and status.

```python
from scripts.studies.paper_efficiency_table import normalize_efficiency_row


def test_normalize_efficiency_row_preserves_runtime_source_field():
    row = normalize_efficiency_row(
        benchmark="PDEBench CNS",
        row_id="spectral_resnet_bottleneck_base",
        model_label="SRU-Net*",
        source_path="docs/plans/NEURIPS-HYBRID-RESNET-2026/tables/pdebench_cns_matched_condition_metrics.json",
        payload={
            "parameter_count": 8186726,
            "runtime_sec": 1861.63,
            "hardware_runtime_note": "RTX 3090 provenance field",
        },
        claim_boundary="bounded_capped_decision_support_only",
    )

    assert row.training_runtime_seconds == 1861.63
    assert row.training_runtime_source_field == "runtime_sec"
    assert row.training_runtime_status == "provenance_context"
    assert row.source_path.endswith("pdebench_cns_matched_condition_metrics.json")
```

- [ ] **Step 2: Write throughput-status classification tests**

Assert that explicit measured throughput is preserved, absent throughput becomes `missing`, and heterogeneous runtime fields do not become throughput.

```python
from scripts.studies.paper_efficiency_table import classify_inference_throughput


def test_classify_inference_throughput_keeps_measured_value():
    result = classify_inference_throughput(
        {"samples_per_second": 385.3},
        source_path="docs/plans/NEURIPS-HYBRID-RESNET-2026/tables/brdt_decision_support_metrics.json",
    )

    assert result.status == "measured"
    assert result.samples_per_second == 385.3
    assert result.source_field == "samples_per_second"


def test_classify_inference_throughput_does_not_promote_training_runtime():
    result = classify_inference_throughput(
        {"runtime_sec": 1861.63},
        source_path="docs/plans/NEURIPS-HYBRID-RESNET-2026/tables/pdebench_cns_matched_condition_metrics.json",
    )

    assert result.status == "missing"
    assert result.samples_per_second is None
    assert result.source_field is None
```

- [ ] **Step 3: Write grouping and claim-boundary test**

Build three normalized rows for CDI, CNS, and BRDT. Render grouped rows and assert the output does not sort all rows into one cross-benchmark ranking.

```python
from scripts.studies.paper_efficiency_table import group_rows_by_benchmark


def test_group_rows_by_benchmark_preserves_claim_boundaries():
    rows = [
        {"benchmark": "CDI Lines128", "row_id": "pinn_hybrid_resnet", "claim_boundary": "paper_table"},
        {"benchmark": "PDEBench CNS", "row_id": "spectral_resnet_bottleneck_base", "claim_boundary": "bounded_capped_decision_support_only"},
        {"benchmark": "BRDT", "row_id": "ffno", "claim_boundary": "paper_approved_secondary_brdt"},
    ]

    grouped = group_rows_by_benchmark(rows)

    assert list(grouped) == ["CDI Lines128", "PDEBench CNS", "BRDT"]
    assert grouped["PDEBench CNS"][0]["claim_boundary"] == "bounded_capped_decision_support_only"
```

- [ ] **Step 4: Write TeX rendering test**

Assert the table has grouped benchmark labels, compact parameter/runtime columns, escaped underscores, and no misleading normalized-speed wording.

```python
from scripts.studies.paper_efficiency_table import render_efficiency_table_tex


def test_render_efficiency_table_tex_groups_rows_and_escapes_fields():
    tex = render_efficiency_table_tex(
        [
            {
                "benchmark": "PDEBench CNS",
                "row_id": "spectral_resnet_bottleneck_base",
                "model_label": "SRU-Net*",
                "parameter_count": 8186726,
                "training_runtime_seconds": 1861.63,
                "training_runtime_source_field": "runtime_sec",
                "inference_throughput_status": "missing",
                "inference_samples_per_second": None,
                "claim_boundary": "bounded_capped_decision_support_only",
            }
        ]
    )

    assert "PDEBench CNS" in tex
    assert "spectral\\_resnet\\_bottleneck\\_base" in tex
    assert "runtime\\_sec" in tex
    assert "normalized throughput" not in tex.lower()
```

- [ ] **Step 5: Run the new tests and confirm they fail before implementation**

Run:

```bash
pytest -q tests/studies/test_paper_efficiency_table.py
```

Expected: FAIL with missing module or missing function errors.

## Task 2: Implement Efficiency Extraction And Rendering

**Files:**
- Create: `scripts/studies/paper_efficiency_table.py`

- [ ] **Step 1: Add dataclasses**

Implement immutable row/result dataclasses:

```python
@dataclass(frozen=True)
class ThroughputEvidence:
    status: str
    samples_per_second: float | None
    latency_ms: float | None
    source_field: str | None
    source_path: str | None


@dataclass(frozen=True)
class EfficiencyRow:
    benchmark: str
    row_id: str
    model_label: str
    parameter_count: int | None
    parameter_count_source_field: str | None
    training_runtime_seconds: float | None
    training_runtime_source_field: str | None
    training_runtime_status: str
    hardware_label: str | None
    inference_throughput_status: str
    inference_samples_per_second: float | None
    inference_latency_ms: float | None
    throughput_source_field: str | None
    source_path: str
    claim_boundary: str
```

- [ ] **Step 2: Implement field helpers**

Add:

- `first_present(payload, field_names) -> tuple[str | None, object | None]`
- `format_parameter_count(value) -> str`
- `escape_latex(value) -> str`
- `classify_inference_throughput(payload, source_path) -> ThroughputEvidence`
- `normalize_efficiency_row(...) -> EfficiencyRow`

Runtime field precedence:

```python
RUNTIME_FIELDS = (
    "train_wall_time_sec",
    "training_wall_time_sec",
    "command_wall_time_sec",
    "runtime_sec",
    "evaluation_wall_time_sec",
    "eval_wall_time_sec",
)
```

Throughput fields:

```python
THROUGHPUT_FIELDS = (
    "samples_per_second",
    "throughput_samples_per_second",
    "inference_samples_per_second",
)

LATENCY_FIELDS = (
    "latency_ms",
    "inference_latency_ms",
    "forward_latency_ms",
)
```

- [ ] **Step 3: Implement CDI extraction**

Read:

- `docs/plans/NEURIPS-HYBRID-RESNET-2026/tables/cdi_lines128_metrics_extended.json`
- optional row manifests under `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-30-cdi-lines128-uno-table-extension/runs/complete_table_plus_uno_20260504T100347Z/`

Extract all rows present in the current CDI table. For each row, preserve:

- `row_id`
- display label
- parameter count
- runtime field and source field name, when present
- hardware/device note, when present
- inference throughput or latency only if explicitly present
- `claim_boundary` from the row, manifest, or table-level context

Do not reconstruct CDI parameter counts in this item. If the appendix/model-config table has a stricter unique-parameter convention, link to that in the summary but leave this efficiency table faithful to its source fields.

- [ ] **Step 4: Implement CNS extraction**

Read:

- `docs/plans/NEURIPS-HYBRID-RESNET-2026/tables/pdebench_cns_matched_condition_metrics.json`
- `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-04-cns-matched-condition-table-refresh/cns_paper_table_rows.json` when present

Extract the matched-condition headline roster only. Preserve CNS `runtime_sec` as training/runtime provenance, not inference throughput. Set throughput to `missing` unless an explicit inference throughput or latency field exists.

- [ ] **Step 5: Implement BRDT secondary-evidence extraction**

Read the user-approved BRDT 40-epoch secondary-evidence bundle:

- `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-05-brdt-supervised-born-40ep-paper-evidence/combined_metrics.json`
- `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-05-brdt-supervised-born-40ep-paper-evidence/split_manifest.json`

Include exactly the `hybrid_resnet` and `ffno` rows. Extract `runtime.parameter_count`, `runtime.wall_time_train_s`, `runtime.wall_time_eval_s`, `runtime.device_name`, and compute inference samples/s as `split_counts.test / runtime.wall_time_eval_s`. Use claim boundary `paper_approved_secondary_brdt`.

The older BRDT decision-support table may be listed as superseded context, but it must not replace the 40-epoch paper-approved rows.

- [ ] **Step 6: Implement writers**

Add:

```python
def collect_efficiency_rows(repo_root: Path) -> list[EfficiencyRow]:
    ...

def write_paper_efficiency_table(repo_root: Path = REPO_ROOT) -> dict[str, str]:
    ...
```

`write_paper_efficiency_table` must write:

- `docs/plans/NEURIPS-HYBRID-RESNET-2026/tables/paper_efficiency_table.json`
- `docs/plans/NEURIPS-HYBRID-RESNET-2026/tables/paper_efficiency_table.csv`
- `docs/plans/NEURIPS-HYBRID-RESNET-2026/tables/paper_efficiency_table.tex`
- `docs/plans/NEURIPS-HYBRID-RESNET-2026/paper_efficiency_table_summary.md`

The JSON payload must include:

- `schema_version`
- `generated_at_utc`
- `rows`
- `excluded_candidate_context`
- `notes`

- [ ] **Step 7: Run unit tests**

Run:

```bash
pytest -q tests/studies/test_paper_efficiency_table.py
```

Expected: PASS.

## Task 3: Wire The Refresh Entrypoint

**Files:**
- Modify: `scripts/studies/paper_results_refresh.py`
- Modify: `tests/studies/test_paper_results_refresh.py`

- [ ] **Step 1: Add a CLI wiring test**

In `tests/studies/test_paper_results_refresh.py`, add a test that monkeypatches `scripts.studies.paper_results_refresh.write_paper_efficiency_table`, calls `main(["--write-efficiency-table"])`, and asserts the returned/printed payload includes `paper_efficiency_table`.

- [ ] **Step 2: Run the focused CLI wiring test and confirm it fails**

Run:

```bash
pytest -q tests/studies/test_paper_results_refresh.py -k efficiency_table
```

Expected: FAIL because `--write-efficiency-table` is not implemented.

- [ ] **Step 3: Add the refresh flag**

Modify `scripts/studies/paper_results_refresh.py`:

- import `write_paper_efficiency_table`
- add `--write-efficiency-table`
- when set, call `write_paper_efficiency_table(REPO_ROOT)`
- include the returned paths under a `paper_efficiency_table` key in the refresh payload

Do not change existing `--write-model-config-table` behavior.

- [ ] **Step 4: Run refresh wiring tests**

Run:

```bash
pytest -q tests/studies/test_paper_results_refresh.py -k "efficiency_table or model_config"
```

Expected: PASS.

## Task 4: Generate The Efficiency Artifacts

**Files:**
- Generate: `docs/plans/NEURIPS-HYBRID-RESNET-2026/paper_efficiency_table_summary.md`
- Generate: `docs/plans/NEURIPS-HYBRID-RESNET-2026/tables/paper_efficiency_table.json`
- Generate: `docs/plans/NEURIPS-HYBRID-RESNET-2026/tables/paper_efficiency_table.csv`
- Generate: `docs/plans/NEURIPS-HYBRID-RESNET-2026/tables/paper_efficiency_table.tex`

- [ ] **Step 1: Run the backlog item input check**

Run:

```bash
python - <<'PY'
from pathlib import Path
required = [
    Path("docs/plans/NEURIPS-HYBRID-RESNET-2026/paper_evidence_package_design.md"),
    Path("docs/plans/NEURIPS-HYBRID-RESNET-2026/lines128_paper_benchmark_summary.md"),
    Path("docs/plans/NEURIPS-HYBRID-RESNET-2026/tables/cdi_lines128_metrics_extended.json"),
    Path("docs/plans/NEURIPS-HYBRID-RESNET-2026/tables/pdebench_cns_matched_condition_metrics.json"),
    Path("docs/plans/NEURIPS-HYBRID-RESNET-2026/paper_evidence_index.md"),
]
missing = [str(path) for path in required if not path.exists()]
if missing:
    raise SystemExit(f"missing paper efficiency table inputs: {missing}")
print("paper efficiency table inputs present")
PY
```

Expected: `paper efficiency table inputs present`.

- [ ] **Step 2: Generate the table assets**

Run:

```bash
python scripts/studies/paper_results_refresh.py --write-efficiency-table
```

Expected: command exits `0` and reports paths for JSON, CSV, TeX, and summary outputs.

- [ ] **Step 3: Validate generated JSON shape**

Run:

```bash
python - <<'PY'
import json
from pathlib import Path
path = Path("docs/plans/NEURIPS-HYBRID-RESNET-2026/tables/paper_efficiency_table.json")
data = json.loads(path.read_text())
assert data["schema_version"]
assert data["rows"]
for row in data["rows"]:
    for key in [
        "benchmark",
        "row_id",
        "model_label",
        "parameter_count",
        "training_runtime_source_field",
        "inference_throughput_status",
        "source_path",
        "claim_boundary",
    ]:
        assert key in row, (row.get("row_id"), key)
    assert row["inference_throughput_status"] in {"measured", "missing", "not_comparable"}
print(f"validated {len(data['rows'])} efficiency rows")
PY
```

Expected: prints the validated row count.

- [ ] **Step 4: Check for misleading throughput wording**

Run:

```bash
rg -n "normalized throughput|standardized throughput|runtime_sec.*throughput|command_wall_time_sec.*throughput" docs/plans/NEURIPS-HYBRID-RESNET-2026/paper_efficiency_table_summary.md docs/plans/NEURIPS-HYBRID-RESNET-2026/tables/paper_efficiency_table.*
```

Expected: no matches.

## Task 5: Update Discovery Documents

**Files:**
- Modify: `docs/plans/NEURIPS-HYBRID-RESNET-2026/evidence_matrix.md`
- Modify: `docs/plans/NEURIPS-HYBRID-RESNET-2026/paper_evidence_index.md`
- Modify: `docs/studies/index.md`

- [ ] **Step 1: Update the evidence matrix**

Add a row or short subsection pointing to:

- `docs/plans/NEURIPS-HYBRID-RESNET-2026/paper_efficiency_table_summary.md`
- `docs/plans/NEURIPS-HYBRID-RESNET-2026/tables/paper_efficiency_table.json`
- `docs/plans/NEURIPS-HYBRID-RESNET-2026/tables/paper_efficiency_table.tex`

State that the table compiles parameter/runtime/throughput evidence by benchmark and does not convert heterogeneous training runtimes into normalized speed claims.

- [ ] **Step 2: Update the paper evidence index**

Add an entry for backlog item `2026-05-05-paper-efficiency-table` with:

- evidence tier: `paper_packaging`
- claim boundary: `efficiency_context_grouped_by_benchmark`
- summary path
- JSON/CSV/TeX table paths
- note that BRDT 40-epoch rows are included as user-approved secondary paper evidence

- [ ] **Step 3: Update the studies index**

Add the efficiency table generator/output summary under the NeurIPS Hybrid ResNet study section so future manuscript or appendix work can find it from `docs/studies/index.md`.

- [ ] **Step 4: Run discovery scans**

Run:

```bash
rg -n "paper_efficiency_table|2026-05-05-paper-efficiency-table|efficiency_context_grouped_by_benchmark" docs/plans/NEURIPS-HYBRID-RESNET-2026/evidence_matrix.md docs/plans/NEURIPS-HYBRID-RESNET-2026/paper_evidence_index.md docs/studies/index.md
```

Expected: all three documents reference the new table or item.

## Task 6: Final Verification

**Files:**
- Validate all changed files and generated assets.

- [ ] **Step 1: Run focused tests**

Run:

```bash
pytest -q tests/studies/test_paper_efficiency_table.py tests/studies/test_paper_results_refresh.py -k "efficiency_table or model_config"
```

Expected: PASS.

- [ ] **Step 2: Run compileall check from the backlog item**

Run:

```bash
python -m compileall -q scripts/studies ptycho_torch
```

Expected: exits `0`.

- [ ] **Step 3: Run the backlog item input check again**

Run the exact input check from Task 4 Step 1.

Expected: `paper efficiency table inputs present`.

- [ ] **Step 4: Verify selector readiness**

From `/home/ollie/Documents/agent-orchestration`, refresh the active backlog manifest:

```bash
python -m workflows.library.scripts.build_neurips_backlog_manifest \
  --backlog-root /home/ollie/Documents/PtychoPINN/docs/backlog/active \
  --output /tmp/neurips-active-manifest.json
python - <<'PY'
import json
from pathlib import Path
data = json.loads(Path("/tmp/neurips-active-manifest.json").read_text())
invalid = [item for item in data.get("invalid_items", []) if item.get("item_id") == "2026-05-05-paper-efficiency-table"]
if invalid:
    raise SystemExit(invalid)
print("paper efficiency backlog item is not invalid in refreshed manifest")
PY
```

Expected: `paper efficiency backlog item is not invalid in refreshed manifest`.

- [ ] **Step 5: Inspect diff**

Run:

```bash
git diff -- scripts/studies/paper_efficiency_table.py scripts/studies/paper_results_refresh.py tests/studies/test_paper_efficiency_table.py tests/studies/test_paper_results_refresh.py docs/plans/NEURIPS-HYBRID-RESNET-2026/paper_efficiency_table_summary.md docs/plans/NEURIPS-HYBRID-RESNET-2026/tables/paper_efficiency_table.json docs/plans/NEURIPS-HYBRID-RESNET-2026/tables/paper_efficiency_table.csv docs/plans/NEURIPS-HYBRID-RESNET-2026/tables/paper_efficiency_table.tex docs/plans/NEURIPS-HYBRID-RESNET-2026/evidence_matrix.md docs/plans/NEURIPS-HYBRID-RESNET-2026/paper_evidence_index.md docs/studies/index.md
```

Expected: only planned files changed, with no bulky artifacts and no manuscript `.tex` edits.

## Completion Criteria

- `paper_efficiency_table.{json,csv,tex}` exists under `docs/plans/NEURIPS-HYBRID-RESNET-2026/tables/`.
- `paper_efficiency_table_summary.md` exists and explains claim boundaries, runtime caveats, and any excluded candidate context.
- Each row has parameter count lineage, runtime source field/path, hardware metadata where available, inference-throughput status, and claim boundary.
- BRDT 40-epoch rows are included as secondary paper evidence with their own claim boundary.
- Focused tests pass.
- `python -m compileall -q scripts/studies ptycho_torch` passes.
- Refreshed backlog manifest no longer marks `2026-05-05-paper-efficiency-table` invalid for a missing `plan_path`.

## Residual Risks

- Existing source artifacts may use different runtime field names or omit hardware fields. Preserve source paths and mark missing fields honestly rather than inventing values.
- CDI source parameter counts may not follow the stricter unique-parameter convention planned for the model-config appendix table. This item reports the source-recorded count and should cross-reference the appendix table once that plan is implemented.
- Lightweight inference probes are authorized only for otherwise-headline rows with missing throughput. If a probe would require training, model tuning, new datasets, or broad environment setup, record the row as `missing` or `not_comparable` and describe the blocker in the summary. The BRDT 40-epoch rows already carry evaluation wall time over the 256-sample test split, so compute samples/s from those fields instead of rerunning evaluation.
- Manuscript insertion is intentionally deferred. A later paper-refresh task should decide whether to include `tables/paper_efficiency_table.tex` in `hybrid_resnet_neurips_first_draft.tex`.
