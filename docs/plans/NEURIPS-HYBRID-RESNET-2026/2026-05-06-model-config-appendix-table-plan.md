# Model Configuration Appendix Table Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. Do not create worktrees.

**Goal:** Add a reproducible appendix/methods table that records the effective model configuration for each benchmark/model row and makes parameter-count conventions explicit.

**Architecture:** Add a small paper-table generation module that reads durable row artifacts and emits `json`, `csv`, and `tex` config-table assets. Keep the evidence policy contract-based: original row labels such as "decision-support" do not determine inclusion; complete row config, source paths, and claim boundaries do. Update the manuscript to reference the appendix table from Methods and use the generated table in an appendix/supplement section.

**Tech Stack:** Python standard library, JSON/CSV/LaTeX table rendering, existing NeurIPS artifact manifests, pytest, existing `scripts/studies/paper_results_refresh.py` paper refresh entrypoint.

---

## Source Of Truth

- Manuscript: `docs/plans/NEURIPS-HYBRID-RESNET-2026/hybrid_resnet_neurips_first_draft.tex`
- Central model index: `docs/plans/NEURIPS-HYBRID-RESNET-2026/model_variant_index.json`
- Evidence index: `docs/plans/NEURIPS-HYBRID-RESNET-2026/paper_evidence_index.md`
- CDI table source: `docs/plans/NEURIPS-HYBRID-RESNET-2026/tables/cdi_lines128_metrics_extended.json`
- CDI row configs and manifests:
  - `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-30-cdi-lines128-uno-table-extension/runs/complete_table_plus_uno_20260504T100347Z/model_manifest.json`
  - `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-30-cdi-lines128-uno-table-extension/runs/complete_table_plus_uno_20260504T100347Z/runs/<row_id>/config.json`
  - `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-30-cdi-lines128-uno-table-extension/runs/complete_table_plus_uno_20260504T100347Z/runs/<row_id>/model.pt`
- CDI supervised FFNO source:
  - `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-supervised-equivalent-rows/runs/supervised_ffno_extension_20260430T180217Z/model_manifest.json`
  - `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-supervised-equivalent-rows/runs/supervised_ffno_extension_20260430T180217Z/runs/supervised_ffno/config.json`
- CNS table source: `docs/plans/NEURIPS-HYBRID-RESNET-2026/tables/pdebench_cns_matched_condition_metrics.json`
- CNS table-generation artifact:
  - `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-04-cns-matched-condition-table-refresh/cns_paper_table_rows.json`
- BRDT table source: `docs/plans/NEURIPS-HYBRID-RESNET-2026/tables/brdt_decision_support_metrics.json`
- BRDT row metrics/config sources:
  - `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-brdt-four-row-preflight/metrics.json`
  - `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-brdt-four-row-preflight/preflight_manifest.json`
  - `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-05-brdt-supervised-born-40ep-paper-evidence/combined_metrics.json`
  - `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-05-brdt-supervised-born-40ep-paper-evidence/rows/<row_id>/model_profile.json`

> **BRDT FFNO caveat (2026-05-06):** the historical BRDT `ffno` profile at
> `36,674` parameters used the old FFNO-local-refiner proxy. Do not emit it as a
> pure FFNO configuration. The corrected no-refiner BRDT FFNO adapter should be
> represented from the corrected rerun root, with `cnn_blocks` absent and an
> expected parameter count of `27,394`.

> **CDI FFNO caveat (2026-05-06):** historical CDI `pinn_ffno` and
> `supervised_ffno` rows used `fno_cnn_blocks=2`. Emit them as
> `FFNO-local proxy` until corrected no-refiner (`fno_cnn_blocks=0`) reruns and
> the no-refiner table refresh replace their source roots.

## Parameter Count Convention

Use `unique_trainable_params` as the manuscript-facing count:

- Count trainable parameters in the effective prediction model used for that row.
- Do not count duplicate aliases in wrapper state dicts.
- Preserve the raw artifact-reported count as `raw_recorded_parameter_count` when it differs.
- For CDI PyTorch rows, detect duplicate `model.generator.*` and `model.autoencoder.*` state-dict aliases and count only one copy when they match by suffix and tensor shape/count.
- For TensorFlow/CNN rows, use the existing recorded count unless a deterministic unique-count extractor already exists.
- For CNS and BRDT rows, use artifact `parameter_count` directly because those row runners already count model parameters at construction/evaluation time.
- Classical/non-neural rows use `0` and `parameter_count_kind=non_neural`.

The appendix table caption must say counts are unique trainable parameters in the effective prediction model. Do not use raw wrapper/state-dict counts in manuscript-facing text.

## File Structure

- Create: `scripts/studies/paper_model_config_table.py`
  - Owns row extraction, CDI state-dict deduping, config normalization, and table rendering.
- Modify: `scripts/studies/paper_results_refresh.py`
  - Adds a `--write-model-config-table` flag and calls the new module.
- Create: `tests/studies/test_paper_model_config_table.py`
  - Unit tests for parameter-count deduping, row extraction, and table rendering.
- Generate: `docs/plans/NEURIPS-HYBRID-RESNET-2026/tables/model_config_by_benchmark.json`
- Generate: `docs/plans/NEURIPS-HYBRID-RESNET-2026/tables/model_config_by_benchmark.csv`
- Generate: `docs/plans/NEURIPS-HYBRID-RESNET-2026/tables/model_config_by_benchmark.tex`
- Modify: `docs/plans/NEURIPS-HYBRID-RESNET-2026/hybrid_resnet_neurips_first_draft.tex`
  - Adds Methods reference, appendix table, and a short parameter-count convention note.
- Optional modify after validation: `docs/plans/NEURIPS-HYBRID-RESNET-2026/tables/pdebench_cns_matched_condition_metrics.tex`
  - Add params only if table width stays readable. Otherwise leave params in appendix only.
- Do not modify core model code.

## Task 1: Add Model Config Table Unit Tests

**Files:**
- Create: `tests/studies/test_paper_model_config_table.py`

- [ ] **Step 1: Write tests for CDI duplicate state-dict aliases**

Add a test that constructs a toy state dict with identical `model.generator.*` and `model.autoencoder.*` tensors plus unrelated forward-model scalars.

```python
import torch

from scripts.studies.paper_model_config_table import count_unique_state_dict_params


def test_count_unique_state_dict_params_dedupes_generator_autoencoder_aliases():
    state = {
        "model.generator.block.weight": torch.zeros(2, 3),
        "model.generator.block.bias": torch.zeros(2),
        "model.autoencoder.block.weight": torch.zeros(2, 3),
        "model.autoencoder.block.bias": torch.zeros(2),
        "model.forward_model.alpha": torch.zeros(1),
        "model.forward_model.beta": torch.zeros(1),
    }

    result = count_unique_state_dict_params(state)

    assert result.unique_trainable_params == 8
    assert result.raw_recorded_parameter_count == 16
    assert result.duplicate_groups == ["model.generator/model.autoencoder"]
```

- [ ] **Step 2: Write tests for non-duplicate state dicts**

```python
def test_count_unique_state_dict_params_keeps_nonmatching_autoencoder():
    state = {
        "model.generator.block.weight": torch.zeros(2, 3),
        "model.autoencoder.block.weight": torch.zeros(4, 3),
    }

    result = count_unique_state_dict_params(state)

    assert result.unique_trainable_params == 18
    assert result.raw_recorded_parameter_count == 18
    assert result.duplicate_groups == []
```

- [ ] **Step 3: Write row-normalization tests**

Create small fixture dicts for one CDI row, one CNS row, and one BRDT row. Assert that normalized rows include:

- `benchmark`
- `display_model`
- `row_id`
- `internal_architecture`
- `training_objective`
- `input_output_contract`
- `width`
- `fourier_modes`
- `encoder_blocks`
- `bottleneck_blocks`
- `downsampling`
- `skip_or_fusion`
- `unique_trainable_params`
- `parameter_count_source`
- `config_source`

- [ ] **Step 4: Write rendering test**

Assert the TeX renderer includes:

- `Unique params`
- `CDI Lines128`
- `PDEBench CNS`
- `BRDT`
- escaped underscores for row/config paths
- no raw internal phrases such as `command_wall_time_sec` in the table body

- [ ] **Step 5: Run tests and confirm they fail before implementation**

Run:

```bash
pytest -q tests/studies/test_paper_model_config_table.py
```

Expected: import or missing-function failures.

## Task 2: Implement Config Extraction And Rendering Module

**Files:**
- Create: `scripts/studies/paper_model_config_table.py`

- [ ] **Step 1: Add dataclasses and constants**

Implement:

```python
@dataclass(frozen=True)
class ParameterCountResult:
    unique_trainable_params: int
    raw_recorded_parameter_count: int
    parameter_count_kind: str
    duplicate_groups: list[str]


@dataclass(frozen=True)
class ModelConfigRow:
    benchmark: str
    display_model: str
    row_id: str
    internal_architecture: str
    training_objective: str
    input_output_contract: str
    width: str
    fourier_modes: str
    encoder_blocks: str
    bottleneck_blocks: str
    downsampling: str
    skip_or_fusion: str
    unique_trainable_params: int
    raw_recorded_parameter_count: int | None
    parameter_count_kind: str
    parameter_count_source: str
    config_source: str
    notes: str
```

- [ ] **Step 2: Implement state-dict unique count helper**

Implement `count_unique_state_dict_params(state_dict: Mapping[str, Any]) -> ParameterCountResult`.

Rules:

- Include tensors under `model.generator.*`, `model.autoencoder.*`, and other model keys.
- If every `model.generator.<suffix>` tensor has a matching `model.autoencoder.<suffix>` tensor with the same shape and numel, count only the generator copy.
- Count non-generator/non-autoencoder tensors normally.
- Return raw total before deduping.
- Mark `parameter_count_kind="unique_effective_trainable_params"`.

- [ ] **Step 3: Implement CDI row extractor**

Implement `load_cdi_config_rows(repo_root: Path) -> list[ModelConfigRow]`.

Use:

- main CDI UNO extension model manifest for rows: `pinn`, `pinn_fno_vanilla`, `pinn_ffno`, `pinn_hybrid_resnet`, `pinn_neuralop_uno`, `baseline`, `supervised_neuralop_uno`
- supervised FFNO extension manifest for `supervised_ffno`
- per-row `config.json` for config knobs
- per-row `model.pt` when present for CDI PyTorch unique-count deduping

Map config fields:

- `fno_width` -> `width`
- `fno_modes` -> `fourier_modes`
- `fno_blocks` -> `encoder_blocks`
- `hybrid_resnet_blocks` or `spectral_bottleneck_blocks` -> `bottleneck_blocks`
- `hybrid_downsample_steps` -> `downsampling`
- `hybrid_skip_connections`, `hybrid_skip_style`, `hybrid_encoder_*`, architecture id -> `skip_or_fusion`

For rows where config or unique param count is unavailable, emit explicit `not_recorded` or `uses_recorded_count` notes; do not invent values.

- [ ] **Step 4: Implement CNS row extractor**

Implement `load_cns_config_rows(repo_root: Path) -> list[ModelConfigRow]`.

Use `docs/plans/NEURIPS-HYBRID-RESNET-2026/tables/pdebench_cns_matched_condition_metrics.json`.

Required rows:

- `author_ffno_cns_base`
- `spectral_resnet_bottleneck_base`
- `fno_base`
- `unet_strong`

Use the table's fields for history length, epochs, batch size, parameter count, and source roots. For unavailable architecture internals, use clear values such as `artifact row profile` only when the source run root includes a model profile, otherwise `not_recorded_in_table_source`.

- [ ] **Step 5: Implement BRDT row extractor**

Implement `load_brdt_config_rows(repo_root: Path) -> list[ModelConfigRow]`.

Use:

- 40-epoch `combined_metrics.json` for `hybrid_resnet` and `ffno`
- 40-epoch `rows/<row_id>/model_profile.json` for exact `arch_kwargs`
- 20-epoch preflight `metrics.json` only for model-based Born inverse if included

Map BRDT fields:

- `hidden_channels` -> `width`
- `fno_modes` -> `fourier_modes`
- `fno_blocks` -> `encoder_blocks`
- `resnet_blocks` -> `bottleneck_blocks`
- `downsample_steps` -> `downsampling`
- `born_init_image -> q_pred` -> `input_output_contract`

- [ ] **Step 6: Implement writers**

Implement:

- `build_model_config_rows(repo_root: Path) -> list[ModelConfigRow]`
- `write_model_config_table(repo_root: Path, output_dir: Path) -> dict[str, str]`
- `render_model_config_tex(rows: Sequence[ModelConfigRow]) -> str`
- `write_model_config_csv(rows, path)`
- `write_model_config_json(rows, path)`

TeX should use compact columns:

```latex
\begin{tabular}{lllrrrrll}
\toprule
Benchmark & Row & Training & Width & Modes & Enc. & Bott. & Down & Unique params \\
...
\bottomrule
\end{tabular}
```

Keep detailed strings in JSON/CSV; keep TeX readable.

- [ ] **Step 7: Run tests**

Run:

```bash
pytest -q tests/studies/test_paper_model_config_table.py
```

Expected: PASS.

## Task 3: Wire Into Paper Refresh

**Files:**
- Modify: `scripts/studies/paper_results_refresh.py`
- Modify: `tests/studies/test_paper_results_refresh.py`

- [ ] **Step 1: Add a narrow integration test**

In `tests/studies/test_paper_results_refresh.py`, add or extend a CLI/parser-level test that verifies `--write-model-config-table` is accepted and calls the new writer when monkeypatched.

- [ ] **Step 2: Add import and flag**

In `paper_results_refresh.py`, import:

```python
from scripts.studies.paper_model_config_table import write_model_config_table
```

Add argparse flag:

```python
parser.add_argument(
    "--write-model-config-table",
    action="store_true",
    help="Emit paper-local model configuration appendix table assets under tables/.",
)
```

- [ ] **Step 3: Call writer from main**

When the flag is present:

```python
paths = write_model_config_table(REPO_ROOT, TABLES_DIR)
print(json.dumps({"model_config_table": paths}, indent=2))
```

- [ ] **Step 4: Run focused tests**

Run:

```bash
pytest -q tests/studies/test_paper_model_config_table.py tests/studies/test_paper_results_refresh.py -k "model_config or cdi or cns or brdt"
```

Expected: PASS.

## Task 4: Generate The Appendix Table Assets

**Files:**
- Generate: `docs/plans/NEURIPS-HYBRID-RESNET-2026/tables/model_config_by_benchmark.json`
- Generate: `docs/plans/NEURIPS-HYBRID-RESNET-2026/tables/model_config_by_benchmark.csv`
- Generate: `docs/plans/NEURIPS-HYBRID-RESNET-2026/tables/model_config_by_benchmark.tex`

- [ ] **Step 1: Run the refresh command**

Run:

```bash
python scripts/studies/paper_results_refresh.py --write-model-config-table
```

Expected: writes the three table assets and exits 0.

- [ ] **Step 2: Inspect generated JSON**

Run:

```bash
python -m json.tool docs/plans/NEURIPS-HYBRID-RESNET-2026/tables/model_config_by_benchmark.json >/tmp/model_config_by_benchmark.pretty.json
```

Expected: exits 0.

- [ ] **Step 3: Verify required rows are present**

Run:

```bash
python - <<'PY'
import json
from pathlib import Path
rows = json.loads(Path("docs/plans/NEURIPS-HYBRID-RESNET-2026/tables/model_config_by_benchmark.json").read_text())["rows"]
required = {
    ("CDI Lines128", "pinn_hybrid_resnet"),
    ("CDI Lines128", "pinn_ffno"),
    ("CDI Lines128", "pinn_fno_vanilla"),
    ("PDEBench CNS", "author_ffno_cns_base"),
    ("PDEBench CNS", "spectral_resnet_bottleneck_base"),
    ("PDEBench CNS", "fno_base"),
    ("PDEBench CNS", "unet_strong"),
    ("BRDT", "hybrid_resnet"),
    ("BRDT", "ffno"),
}
seen = {(row["benchmark"], row["row_id"]) for row in rows}
missing = sorted(required - seen)
raise SystemExit(f"missing config rows: {missing}") if missing else None
PY
```

Expected: no output and exit 0.

- [ ] **Step 4: Verify CDI SRU-Net dedupe**

Run:

```bash
python - <<'PY'
import json
from pathlib import Path
rows = json.loads(Path("docs/plans/NEURIPS-HYBRID-RESNET-2026/tables/model_config_by_benchmark.json").read_text())["rows"]
row = next(r for r in rows if r["benchmark"] == "CDI Lines128" and r["row_id"] == "pinn_hybrid_resnet")
assert row["unique_trainable_params"] < row["raw_recorded_parameter_count"], row
assert "dedup" in row["parameter_count_kind"], row
PY
```

Expected: exit 0.

## Task 5: Incorporate The Table Into The Manuscript

**Files:**
- Modify: `docs/plans/NEURIPS-HYBRID-RESNET-2026/hybrid_resnet_neurips_first_draft.tex`

- [ ] **Step 1: Add Methods reference**

In the experimental protocol/methods area, add a concise sentence:

```latex
Appendix Table~\ref{tab:model_config_by_benchmark} reports the effective
configuration of each reported model row; parameter counts are unique trainable
parameters in the effective prediction model.
```

- [ ] **Step 2: Keep main result tables focused**

Do not add config columns to the CDI main result tables. Leave CDI metrics tables readable and use the appendix table for model config and parameter counts.

- [ ] **Step 3: CNS table decision**

If `tables/pdebench_cns_matched_condition_metrics.tex` remains readable with params, add a compact `Params` column. If it becomes too wide, leave CNS params in appendix only and add a caption sentence:

```latex
Model sizes and row configurations are reported in Appendix
Table~\ref{tab:model_config_by_benchmark}.
```

- [ ] **Step 4: BRDT table decision**

Keep BRDT params and evaluation time in the main BRDT table because the compactness claim is directly discussed there. Ensure the caption points to Appendix Table~\ref{tab:model_config_by_benchmark} for exact BRDT adapter settings.

- [ ] **Step 5: Add appendix table before bibliography**

Insert before `\begin{thebibliography}{9}`:

```latex
\clearpage
\appendix
\section{Model Configurations}

\begin{table*}[t]
\centering
\scriptsize
\setlength{\tabcolsep}{3pt}
\caption{\textbf{Model configurations by benchmark.} Parameter counts are
unique trainable parameters in the effective prediction model. Raw wrapper
state-dict counts are not used when they duplicate the same generator under
multiple aliases.}
\label{tab:model_config_by_benchmark}
\resizebox{\textwidth}{!}{\input{tables/model_config_by_benchmark.tex}}
\end{table*}
```

- [ ] **Step 6: Scan for misleading raw count language**

Run:

```bash
rg -n "18,006,600|18006600|raw wrapper|state-dict|parameter count|Params" docs/plans/NEURIPS-HYBRID-RESNET-2026/hybrid_resnet_neurips_first_draft.tex
```

Expected: no raw CDI SRU-Net wrapper count is presented as manuscript-facing params.

## Task 6: Validate Manuscript And Artifacts

**Files:**
- Validate generated table assets and manuscript.

- [ ] **Step 1: Run focused tests**

Run:

```bash
pytest -q tests/studies/test_paper_model_config_table.py tests/studies/test_paper_results_refresh.py -k "model_config or cdi or cns or brdt"
```

Expected: PASS.

- [ ] **Step 2: Compile the paper**

Run from the paper directory:

```bash
cd docs/plans/NEURIPS-HYBRID-RESNET-2026
pdflatex -interaction=nonstopmode hybrid_resnet_neurips_first_draft.tex
pdflatex -interaction=nonstopmode hybrid_resnet_neurips_first_draft.tex
```

Expected: both commands exit 0.

- [ ] **Step 3: Scan PDF text**

Run:

```bash
pdftotext docs/plans/NEURIPS-HYBRID-RESNET-2026/hybrid_resnet_neurips_first_draft.pdf /tmp/hybrid_resnet_neurips_first_draft.txt
rg -n "Model configurations|unique trainable|CDI Lines128|PDEBench CNS|BRDT" /tmp/hybrid_resnet_neurips_first_draft.txt
```

Expected: appendix table text appears.

- [ ] **Step 4: Check git diff**

Run:

```bash
git diff -- scripts/studies/paper_model_config_table.py scripts/studies/paper_results_refresh.py tests/studies/test_paper_model_config_table.py tests/studies/test_paper_results_refresh.py docs/plans/NEURIPS-HYBRID-RESNET-2026/hybrid_resnet_neurips_first_draft.tex docs/plans/NEURIPS-HYBRID-RESNET-2026/tables/model_config_by_benchmark.json docs/plans/NEURIPS-HYBRID-RESNET-2026/tables/model_config_by_benchmark.csv docs/plans/NEURIPS-HYBRID-RESNET-2026/tables/model_config_by_benchmark.tex
```

Expected: only planned files changed, with no bulky artifacts.

## Task 7: Optional Package Rebuild

Only do this if the user wants a fresh paper zip after reviewing the table in the PDF.

- [ ] **Step 1: Rebuild package with freshness checks**

Use the existing paper zip workflow already used for this manuscript. Verify that the `.tex`, generated table, and PDF in the zip match the working tree versions.

## Completion Criteria

- Generated appendix table exists in JSON/CSV/TeX form.
- Table includes CDI, CNS, and BRDT model rows currently reported in manuscript tables.
- CDI PyTorch SRU-Net parameter count is not the raw duplicated wrapper count.
- Manuscript references the appendix table from Methods and includes the appendix table before the bibliography.
- Focused tests pass.
- Paper compiles successfully.

## Residual Risks

- Existing CDI TensorFlow/CNN parameter counts may still be recorded counts rather than independently reconstructed unique counts. The generated row should say `uses_recorded_count` when that is true.
- CNS runtime fields are training/runtime provenance, not standardized inference throughput. Do not relabel them as inference throughput without a separate benchmark.
- BRDT remains secondary context; adding exact configs must not promote it beyond the current claim boundary.
