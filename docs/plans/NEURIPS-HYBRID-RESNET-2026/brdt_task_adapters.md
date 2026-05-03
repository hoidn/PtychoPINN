# BRDT Task Adapters Summary

## Identity

- Initiative: `NEURIPS-HYBRID-RESNET-2026`
- Backlog item: `2026-04-29-brdt-task-adapters`
- Plan: `docs/plans/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-brdt-task-adapters/execution_plan.md`
- Tier: `feasibility` (additive candidate work; not manuscript evidence)
- Artifact root:
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-brdt-task-adapters/`
- Machine-readable artifacts:
  - `fast_dev_run/adapter_contract.json`
  - `fast_dev_run/eval_summary.json`
  - `fast_dev_run/invocation.json` and `fast_dev_run/invocation.sh`

## Prerequisite Status

This item consumes the locked BRDT operator and dataset authorities
without redefining them:

- operator authority:
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-brdt-operator-validation/operator_validation.json`
  (verdict `pass_with_documented_limits`);
- dataset authority:
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-brdt-dataset-preflight/dataset_manifest.json`;
- dataset-contract helpers consumed verbatim from
  `scripts/studies/born_rytov_dt/dataset_contract.py`
  (`build_manifest`, `compute_train_normalization`, `normalize_q`,
  `unnormalize_q`, `reject_normalized_q_to_operator`).

The dataset preflight was completed earlier without adapter, training,
or evaluation surfaces; this item is the first place those exist.

## First Shared Input Contract

The first bounded four-row contract uses `input_mode = born_init_image`
for every row. Direct sinogram input is explicitly rejected at:

- `scripts/studies/born_rytov_dt/run_config.py` (`RowConfig.__post_init__`
  raises if `input_mode in REJECTED_INPUT_MODES`),
- `scripts/studies/born_rytov_dt/data.py` (`assert_input_mode_supported`
  raises for `sinogram` / `direct_sinogram`),
- the adapter-contract payload (`row_schema.rejected_input_modes`).

Mixing direct-sinogram rows into the first bounded BRDT table is
therefore a contract-level error rather than a soft convention.

The optional confidence/mask channel is exposed via the train/eval
``--in-channels`` argument (1 or 2). When ``--in-channels 2`` is
selected, the second channel is filled with zeros for the bounded
preflight; later items may populate it from the dataset.

## Row Metadata Schema

Each row carries the locked metadata fields (see `RowConfig`):

| Field | Purpose |
| --- | --- |
| `row_id` | unique identifier inside one table; visible row identity (e.g. `sru_net`) |
| `model` | internal architecture ID; one of `classical_born_backprop`, `unet`, `fno_vanilla`, `hybrid_resnet` |
| `training` | training procedure label (e.g. `supervised + Born consistency`); never relabeled `PINN` for supervised+physics rows |
| `input_mode` | first-contract value is always `born_init_image` |
| `dataset_id` | canonical dataset name (e.g. `brdt128_sparse_fullview_preflight`) |
| `operator_version` | operator validation report path / SHA so geometry choices are traceable |
| `row_status` | one of `ready`, `blocked`, `feasibility_only`, `completed`, `skipped` |
| `paper_label` | visible label (e.g. `SRU-Net`); distinct from internal architecture ID |

The visible row identity and the internal architecture ID stay
distinct: the Hybrid-family row may be presented as either
`Hybrid ResNet` (`row_id="hybrid_resnet"`, `paper_label="Hybrid ResNet"`)
or `SRU-Net` (`row_id="sru_net"`, `paper_label="SRU-Net"`), but the
internal `model` field is **always** `hybrid_resnet` because the
adapter body is identical across both presentations. ``sru_net`` is a
visible row identifier (in `SUPPORTED_ROW_IDS`), NOT a member of
`SUPPORTED_ARCHITECTURES`. The CLI ``--architecture sru_net`` selects
the visible SRU-Net row directly; the architecture choice is
authoritative for the row identity surfaced in the adapter contract.

`run_config.required_row_fields()` is the durable surface the four-row
preflight should consume.

## Adapter Contract

Adapters are ordinary real-channel `model(x) -> q_pred` modules under
`scripts/studies/born_rytov_dt/models.py`, NOT registered in the
PtychoPINN CDI generator registry:

- input: `(B, C_in, 128, 128)` real tensor (the
  `born_init_image` representation),
- output: `(B, 1, 128, 128)` real tensor in normalized or physical q
  units depending on the run config (`output_space`).

The bounded roster reuses existing model bodies through task-local
adapters only:

- `unet`: compact U-Net body local to this study;
- `fno_vanilla`: `neuralop.models.FNO` (gated; missing dependency
  surfaces as a row-level blocker via `AdapterBuildError`);
- `hybrid_resnet` / `sru_net`: small Hybrid ResNet body assembled from
  the existing `ptycho_torch.generators.hybrid_resnet` /
  `resnet_components` components.

## Loss Contract

`scripts/studies/born_rytov_dt/lightning_module.py::BRDTTrainingModule`
owns the supervised + Born consistency loss with image, physics,
relative-physics, TV, and positivity terms. Default weights live in
`run_config.LossWeights` and match the candidate-lane design.

The unnormalize-before-physics rule is enforced in exactly one place
inside `BRDTTrainingModule.to_physical_q`, which routes the conversion
through `dataset_contract.unnormalize_q` and runs the
`dataset_contract.reject_normalized_q_to_operator` guard with a
routing tag derived from the conversion path actually taken (so the
guard is not invoked with a literal):

```python
def to_physical_q(self, q_pred):
    routing = "normalized_q"           # default to unsafe
    if self.output_space == "normalized_q":
        q_phys = dc.unnormalize_q(q_pred, self.normalization)
        routing = "physical_q"         # set ONLY after a successful conversion
    elif self.output_space == "physical_q":
        q_phys = q_pred
        routing = "physical_q"
    dc.reject_normalized_q_to_operator(routing)
    return q_phys
```

Callers (`compute_loss`, `train.py`, `evaluate.py`) MUST go through
`to_physical_q` before invoking the operator; the guard then raises
`ValueError` if the conversion path failed to mark the tensor as
`physical_q`. The mean/std arithmetic itself is no longer duplicated
in this module — both the data-loading path and the training path now
share the single `dataset_contract.unnormalize_q` implementation.

The default training label is `supervised + Born consistency`. Rows
that pair supervised image loss with Born consistency are never
relabeled `PINN` or `PINN-only`; `RowConfig` rejects such labels.

## Classical Reference Path

`scripts/studies/born_rytov_dt/classical.py` owns both the
`born_init_image` derivation and the classical reference row.

- Backend selection (`detect_classical_backend`):
  - prefers `odtbrain.backpropagate_2d` when ODTbrain is importable,
    labeled `external_oracle`;
  - otherwise falls back to a local backprop adjoint built from the
    locked `BornRytovForward2D` autograd path, labeled
    `feasibility_only`.
- The local-adjoint backend shares spectral machinery with the forward
  operator and therefore cannot serve as an independent oracle. It is
  acceptable for adapter readiness only; benchmark-grade rows must
  re-run with the ODTbrain backend or document the gap.
- Sanity-run artifacts record the backend in
  `adapter_contract.classical_backend` so downstream consumers can
  enforce the appropriate claim boundary without re-deriving it.

If a planned dependency is missing (e.g. ODTbrain or `neuralop`), the
affected row is recorded with `row_status = "blocked"` plus a
`blocker_reason` tag (`odtbrain_unavailable`, `neuralop_unavailable`)
rather than silently disappearing.

## Train / Eval Entrypoints

```bash
# Sanity training, single batch (CPU example):
python -m scripts.studies.born_rytov_dt.train \
  --architecture unet \
  --manifest .artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-brdt-dataset-preflight/dataset_manifest.json \
  --output-root .artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-brdt-task-adapters/fast_dev_run \
  --fast-dev-run --device cpu --batch-size 2 --in-channels 1

# Eval-only dry-run (no model forward):
python -m scripts.studies.born_rytov_dt.evaluate \
  --architecture unet \
  --manifest .artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-brdt-dataset-preflight/dataset_manifest.json \
  --output-root .artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-brdt-task-adapters/eval_dry \
  --dry-run --device cpu
```

Both entrypoints accept `--architecture` from the bounded roster
(`classical_born_backprop`, `unet`, `fno_vanilla`, `hybrid_resnet`,
`sru_net`), the dataset manifest path, and an output root. Every
successful invocation — neural training, classical-only training,
neural evaluation, classical evaluation, and `--dry-run` — emits:

- `invocation.json` and `invocation.sh` provenance,
- `adapter_contract.json` (durable adapter contract for downstream
  consumers; schema version `brdt_adapter_contract_v1`),
- `eval_summary.json` (per-run sanity summary with `row_status` and
  loss/eval metrics; omitted for the `--dry-run` path which only
  validates the contract surface).

The `adapter_contract.json` payload always carries the full bounded
roster; the row matching the executed `--architecture` is annotated
with the per-run `sanity_summary` and the resolved `row_status`. Both
entrypoints treat `--architecture` as authoritative for the visible
row identity: passing `--architecture sru_net` surfaces the SRU-Net
row regardless of `--hybrid-label` (no silent fallback to the
`hybrid_resnet` row).

## Reproduction

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ptycho311
python -m pytest -q tests/studies/test_born_rytov_dt_adapters.py
python -m compileall -q scripts/studies/born_rytov_dt ptycho_torch
```

Both must succeed.

## Claim Boundary

This item is **adapter readiness only**. It does not authorize BRDT
manuscript evidence, does not register BRDT as a CDI generator, does
not run the bounded four-row BRDT preflight, and does not produce a
decision-support split. CDI `lines128` and PDEBench CNS remain the
required manuscript pillars; promoting BRDT into manuscript evidence
requires a separately checked-in roadmap or evidence-package amendment.

The sanity run on the smoke dataset is feasibility-only and is
recorded as such in `adapter_contract.extra` and `eval_summary.json`
(`row_status: "feasibility_only"`).

## Handoff

Downstream consumers should:

- read the operator authority from
  `operator_validation.json` (`operator` block) and the dataset
  authority from `dataset_manifest.json` rather than from this summary;
- consume the adapter contract from
  `adapter_contract.json` (`schema_version: brdt_adapter_contract_v1`);
- depend on the locked `dataset_contract` helpers
  (`unnormalize_q`, `reject_normalized_q_to_operator`, etc.) instead of
  re-deriving normalization or operator-input rules;
- treat the local-adjoint backend as feasibility-only and re-run the
  classical row with ODTbrain before any decision-support claim.
