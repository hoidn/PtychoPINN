# NeurIPS Lines128 U-NO Table Extension Design

## Context And Authority

- Status: draft design
- Date: 2026-04-30
- Initiative: NeurIPS Hybrid ResNet 2026
- Parent evidence lane: `N=128` grid-lines CDI paper benchmark
- Authoritative completed base table:
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-paper-benchmark-execution/runs/complete_table_20260430T150757Z_repair_tmux`
- Relevant docs:
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/lines128_paper_benchmark_design.md`
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/lines128_paper_benchmark_summary.md`
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/lines128_supervised_equivalent_rows_summary.md`
  - `docs/workflows/pytorch.md`
  - `ptycho_torch/generators/README.md`

## Purpose

Add U-NO to the completed Lines128 CDI benchmark as an append-only extension.
The extension must answer whether an external NeuralOperator U-NO body is a
useful comparator under the same locked `lines128` CDI contract already used
for the paper table.

This is not a rerun of the completed table. Existing Hybrid ResNet, spectral,
FNO, FFNO, CNN supervised, and CNN PINN rows remain preserved evidence. The U-NO
work creates a new extended bundle that references those rows and runs only the
new U-NO rows.

## Naming Decision

Use `neuralop_uno` as the repo architecture ID.

Reason: this repo already has an internal class named `HybridUNOGenerator`, but
that is the legacy Hybrid U-NO-style FNO/CNN architecture, not the external
NeuralOperator U-NO implementation. The paper-facing label can be `U-NO`, but
the code and manifests should use `neuralop_uno` to avoid provenance ambiguity.

Planned rows:

| Row key | Paper label | Architecture ID | Training procedure |
| --- | --- | --- | --- |
| `pinn_neuralop_uno` | `U-NO + PINN` | `neuralop_uno` | `pinn` |
| `supervised_neuralop_uno` | `U-NO + supervised` | `neuralop_uno` | `supervised` |

## Environment And External Package Contract

Primary environment: `ptycho311`.

The U-NO implementation should use the installed NeuralOperator package:

- distribution package: `neuraloperator`
- import module: `neuralop`
- local verified version on 2026-04-30: `neuraloperator==2.0.0`
- local verified import:

```bash
source /home/ollie/miniconda3/etc/profile.d/conda.sh
conda activate ptycho311
python - <<'PY'
import neuralop
from neuralop.models import UNO
print(neuralop.__version__)
print(UNO)
PY
```

If the package is missing or incompatible in a fresh environment, install the
pinned package into `ptycho311`:

```bash
source /home/ollie/miniconda3/etc/profile.d/conda.sh
conda activate ptycho311
python -m pip install "neuraloperator==2.0.0"
python -m pip show neuraloperator
python - <<'PY'
from neuralop.models import UNO
print(UNO)
PY
```

The design does not require cloning an external U-NO repository. If
`neuralop.models.UNO` is unavailable or its constructor API is incompatible, the
implementation must stop with a row-level environment/API blocker instead of
substituting a different U-NO-like model.

Each U-NO row must record:

- Python executable and version
- `torch` version, CUDA version, and GPU
- `neuraloperator` package version from `pip show neuraloperator`
- `neuralop.__version__`
- `neuralop.models.UNO` constructor signature or a checked-in API summary
- git commit and dirty-state note

## U-NO Model Contract

The adapter should wrap `neuralop.models.UNO` behind the existing Torch
generator contract used by `PtychoPINN_Lightning`.

Local verified constructor signature:

```text
UNO(
    in_channels,
    out_channels,
    hidden_channels,
    lifting_channels=256,
    projection_channels=256,
    positional_embedding='grid',
    n_layers=4,
    uno_out_channels=None,
    uno_n_modes=None,
    uno_scalings=None,
    ...
)
```

The U-NO preflight must freeze the exact values before benchmark execution.
Initial recommended settings:

- `in_channels`: derived from the same real-valued input channel contract as
  existing Torch generators
- `out_channels`: `2`, for real/imag output before the existing complex-output
  adapter
- `hidden_channels`: `32`
- `lifting_channels`: `128` unless memory preflight requires `64`
- `projection_channels`: `128` unless memory preflight requires `64`
- `n_layers`: `4`
- `uno_n_modes`: align with the locked `fno_modes=12` contract where the API
  accepts per-layer modes
- `positional_embedding`: `grid`
- `generator_output_mode`: `real_imag`

The implementation may adjust these defaults only in the preflight/design
amendment, before seeing benchmark metrics. Do not tune U-NO after observing
the other rows.

## Fixed Lines128 Contract

Both U-NO rows must use the completed table's fixed contract:

- `N=128`
- `gridsize=1`
- synthetic grid-lines data
- `set_phi=True`
- custom Run1084 probe
- `probe_scale_mode=pad_extrapolate`
- `probe_smoothing_sigma=0.5`
- `nimgs_train=2`
- `nimgs_test=2`
- `nphotons=1e9`
- `seed=3`
- `torch_epochs=40`
- `torch_learning_rate=2e-4`
- `torch_scheduler=ReduceLROnPlateau`
- `torch_plateau_factor=0.5`
- `torch_plateau_patience=2`
- `torch_plateau_min_lr=1e-4`
- `torch_plateau_threshold=0.0`
- `torch_loss_mode=mae`
- `torch_mae_pred_l2_match_target=off`
- `torch_output_mode=real_imag`
- `probe_mask=off`
- fixed visual sample ids: `0`, `1`
- selected FNO comparator remains `fno_vanilla`

The extension must not change these fields to make U-NO easier to run.

## Append-Only Bundle Contract

The U-NO work must produce a new derived root, for example:

```text
.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/<uno-item>/runs/complete_table_plus_uno_<timestamp>
```

The derived root must include:

- the six completed base rows promoted from
  `complete_table_20260430T150757Z_repair_tmux`
- `pinn_neuralop_uno` as a fresh row
- `supervised_neuralop_uno` as a fresh row
- merged `metrics.json`, `model_manifest.json`, `paper_benchmark_manifest.json`
- `metrics_table.csv`, `metrics_table.tex`, and `metrics_table_best.tex`
- combined visual bundle with the same fixed sample ids and shared scales
- row-local invocation/config/history/metrics/reconstruction artifacts for
  the two U-NO rows
- explicit lineage fields showing that base rows were reused/promoted and U-NO
  rows were freshly run

The original complete-table root must remain immutable and must not be
rewritten. The new bundle's claim boundary should be
`complete_lines128_cdi_benchmark_plus_uno_extension`.

## Implementation Components

1. Environment/API preflight
   - Verify `neuraloperator==2.0.0` in `ptycho311`.
   - Import `neuralop.models.UNO`.
   - Record the constructor signature.
   - Run a tiny forward-pass shape smoke with the selected input/output
     channel contract.

2. Generator integration
   - Add a `ptycho_torch/generators/neuralop_uno.py` wrapper.
   - Register `neuralop_uno` in `ptycho_torch/generators/registry.py`.
   - Add `neuralop_uno` to the architecture literal in
     `ptycho_torch/config_params.py`.
   - Ensure both `mode=Unsupervised` and `mode=Supervised` use the same U-NO
     generator body when `architecture=neuralop_uno`.
   - Fail closed with a clear blocker if U-NO cannot emit the required
     `real_imag` output shape.

3. Runner and wrapper routing
   - Teach `grid_lines_torch_runner.py` and `grid_lines_compare_wrapper.py`
     the row keys `pinn_neuralop_uno` and `supervised_neuralop_uno`.
   - Preserve existing row routing unchanged.
   - Add an append-only mode or execution path that promotes the base six rows
     from the completed root and launches only the two U-NO rows.

4. Reporting
   - Extend the merged metrics/table/visual collation to include the two U-NO
     rows.
   - Preserve row display labels separately from architecture IDs.
   - Emit a durable summary under
     `docs/plans/NEURIPS-HYBRID-RESNET-2026/`.
   - Update `docs/studies/index.md` and `docs/index.md` so the U-NO extension
     is discoverable.

## Backlog Decomposition

Recommended backlog items:

1. `2026-04-30-cdi-lines128-uno-design-preflight`
   - verify environment and U-NO API
   - freeze constructor defaults
   - write install/API summary
   - produce `ready_for_uno_rows` or an explicit blocker

2. `2026-04-30-cdi-lines128-uno-generator-integration`
   - implement `neuralop_uno` generator registry support
   - prove supervised and PINN construction both use the U-NO generator body
   - add focused shape/config/provenance tests

3. `2026-04-30-cdi-lines128-uno-table-extension`
   - run only `pinn_neuralop_uno` and `supervised_neuralop_uno`
   - promote the completed six-row base table without rerunning it
   - emit the extended table/visual/provenance bundle
   - update summaries and indexes

If the implementation team wants fewer queue items, combine the first two. Do
not combine integration and benchmark execution unless the environment/API
preflight is already green.

## Required Tests And Checks

Focused tests:

- `neuralop_uno` registry resolution succeeds when `neuraloperator` is present.
- Missing `neuraloperator` produces a clear blocker, not an unrelated import
  crash.
- U-NO forward-pass smoke returns the expected real/imag shape.
- `architecture=neuralop_uno, mode=Unsupervised` uses the U-NO generator body.
- `architecture=neuralop_uno, mode=Supervised` uses the same U-NO generator
  body.
- Append-only collation does not rerun or rewrite base rows.
- Extended manifest records base-row lineage and fresh U-NO row provenance.

Required checks:

```bash
python - <<'PY'
from neuralop.models import UNO
print(UNO)
PY

pytest -q tests/torch/test_generator_registry.py tests/torch/test_loss_modes.py
pytest -q tests/test_grid_lines_compare_wrapper.py tests/torch/test_grid_lines_torch_runner.py
python -m compileall -q ptycho_torch scripts/studies
```

The table-extension backlog item should also run the relevant Lines128
benchmark wrapper smoke and archive its log under the item verification
directory.

## Failure And Blocker Policy

Acceptable blockers:

- `neuraloperator==2.0.0` cannot be installed or imported in `ptycho311`.
- `neuralop.models.UNO` cannot support the required input/output shape without
  changing the locked Lines128 contract.
- supervised and PINN paths cannot be made to share the same U-NO generator
  body without broader model refactoring.
- U-NO training fails due to memory at the frozen defaults after a documented
  attempt to use the preflight-approved smaller lifting/projection widths.

Unacceptable fallbacks:

- rerunning all existing rows merely to add U-NO
- silently replacing U-NO with the existing internal `hybrid` or
  `HybridUNOGenerator`
- changing seed, probe, dataset, split, loss, epochs, or visual sample policy
  after seeing U-NO metrics
- overwriting the authoritative six-row complete-table root
- reporting U-NO as paper-complete if row-local provenance or launcher proof is
  incomplete

## Expected Outcome

The final U-NO extension should support statements of the form:

> Under the locked Lines128 CDI contract, an external NeuralOperator U-NO row
> was appended to the completed six-row paper table in both PINN and supervised
> modes without rerunning the existing rows.

Any ranking claim must cite the extended bundle root and must distinguish the
original six-row paper table from the later U-NO extension.
