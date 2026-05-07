# BRDT Sinogram-Input Rework Plan

Date: 2026-05-07

## Goal

Rework the BRDT experiment so the learned models consume the measured complex
sinogram directly, rather than a precomputed Born inverse image. The Born inverse
should remain a non-learned baseline and visualization reference, not the model
input.

Old contract:

```text
measured sinogram -> fixed Born inverse image -> neural model -> q_hat
```

New contract:

```text
measured complex sinogram -> neural model -> q_hat
q_hat -> Born forward model -> predicted sinogram -> physics-consistency loss
```

This changes the scientific claim. Existing BRDT 40-epoch results remain valid
lineage for the old Born-image-input contract. They may remain visible only when
the manuscript explicitly describes that old input contract. They must not be
used to support a sinogram-input BRDT claim, because their learned-model input
contract does not match the new one.

## Backlog Decision

Create new backlog items rather than requeue old completed items. The reason is
contract mismatch, not the old items' labels or whether earlier notes called
them candidate, secondary, or paper evidence.

Completed items are immutable evidence for the old contract:

- `2026-05-06-brdt-corrected-ffno-40ep-rerun`
- `2026-05-06-brdt-corrected-ffno-row-rerun`
- `2026-05-05-brdt-supervised-born-40ep-paper-evidence`
- older BRDT FFNO/preflight lineage items

Do not move these from `done` back to `active`. Instead:

- Add contract-supersession notes in paper-evidence indexes and summaries once
  the new run succeeds.
- Keep the old artifacts discoverable as Born-image-input lineage.
- Make the new sinogram-input run the manuscript evidence source after it
  passes the 40-epoch evidence gate.

New backlog items:

1. `2026-05-07-brdt-sinogram-input-adapter-contract`
   - Purpose: implement and test the new input contract without running the
     full 40-epoch experiment.
   - Priority: `3` unless the operator explicitly promotes it further. This
     keeps the existing priority-`1` and priority-`2` active SRU-Net mechanism
     items ahead of it, while placing it above the WaveBench queue.
   - Exit condition: fast tests and smoke runs prove that SRU-Net and FFNO
     accept complex sinograms as input and that the loss still compares
     predicted sinograms to measured sinograms through the Born forward model.

2. `2026-05-07-brdt-sinogram-input-40ep-paper-evidence`
   - Purpose: run the paper evidence experiment under the new contract.
   - Priority: `4`, depends on the adapter-contract item, and remains above the
     WaveBench queue.
   - Exit condition: completed 40-epoch SRU-Net and FFNO rows, refreshed
     metrics, refreshed Figure 3, refreshed tables, updated manuscript zip.

## Required Code Changes

### 1. Input-mode schema

Files:

- `scripts/studies/born_rytov_dt/run_config.py`
- `scripts/studies/born_rytov_dt/data.py`
- `tests/studies/test_born_rytov_dt_adapters.py`
- `tests/studies/test_born_rytov_dt_preflight.py`

Tasks:

- Replace the one-mode contract `born_init_image` with explicit support for
  `sinogram`.
- Remove or revise tests that currently assert `sinogram` is rejected.
- Keep `born_init_image` available for historical runner reproducibility. Do not
  delete old runner support solely to enforce the new contract. Instead, make
  the new paper runners require `sinogram` and make paper-refresh scripts select
  only artifact roots whose config snapshot records `input_mode=sinogram`.
- Add schema tests that verify:
  - `input_mode="sinogram"` is accepted.
  - new paper-evidence row configs use `sinogram`.
  - old Born-image-input configs are not selected by the new runner or by
    paper-refresh scripts after the new run succeeds.

### 2. Dataset and batch contract

Files:

- `scripts/studies/born_rytov_dt/data.py`
- related adapter/preflight tests

Current dataset already exposes:

```text
sinogram: (angles, detectors, 2)
q_true: target scattering potential
```

Tasks:

- Make the training/evaluation input tensor use the real/imaginary sinogram
  channels directly.
- Document the exact tensor shape in code comments and tests. Expected model
  input shape should be one of:
  - `(B, 2, n_angles, n_detectors)`, preferred for channels-first models, or
  - `(B, n_angles, n_detectors, 2)` before the model adapter converts it.
- Assert that the displayed sinogram magnitude in Figure 3 is derived from the
  same complex tensor used as the neural model input.
- Keep Born inverse generation available only for:
  - the non-learned baseline,
  - optional visualization,
  - historical artifact comparison.

### 3. Model adapters

Files:

- `scripts/studies/born_rytov_dt/models.py`
- tests in `tests/studies/test_born_rytov_dt_adapters.py`

Tasks:

- Add BRDT sinogram-input wrappers for SRU-Net and FFNO.
- The wrappers must map a complex sinogram tensor to `q_hat` on the object grid.
- Do not use the fixed Born inverse as preprocessing inside the model.
- Accept only the minimal task-specific lift/projection required to map between
  sinogram tensor shape and model feature width/output grid.
- Add forward-shape tests:
  - SRU-Net: `(B, 2, 64, 128) -> (B, 1, 128, 128)`
  - FFNO: `(B, 2, 64, 128) -> (B, 1, 128, 128)`
- Add a negative test that fails if a model adapter calls the Born inverse
  derivation helper in the input path.

Implementation guidance:

- Prefer a small task-local sinogram encoder/projection bridge over modifying
  global SRU-Net or FFNO modules.
- The bridge can lift sinogram channels, resample from sinogram coordinates to
  object-grid coordinates, and pass features into the existing architecture.
- This bridge is an input adapter, not a reconstruction algorithm. It must not
  compute a Born backprojection.

### 4. Training, evaluation, and preflight paths

Files:

- `scripts/studies/born_rytov_dt/train.py`
- `scripts/studies/born_rytov_dt/evaluate.py`
- `scripts/studies/born_rytov_dt/run_preflight.py`
- existing BRDT runners, or a new runner listed below

Tasks:

- Replace `_prepare_model_input` for the new path so it returns the complex
  sinogram tensor, not `derive_born_init_image(...)`.
- Keep the loss path unchanged in principle:
  - model predicts `q_hat`;
  - Born forward operator maps `q_hat` to predicted sinogram;
  - loss compares predicted sinogram to measured sinogram.
- Add tests that prove the measured sinogram is used in two places:
  - as neural model input,
  - as physics-consistency target.
- Make preflight output print the current input mode and tensor shape.
- Create a small smoke path that runs one or two train/eval batches under
  `input_mode="sinogram"`.

### 5. New paper runner

Recommended new file:

- `scripts/studies/born_rytov_dt/run_sinogram_input_40ep.py`

Tasks:

- Run SRU-Net and FFNO with the same:
  - dataset split: 2048/256/256,
  - 40 epochs,
  - supervised object loss plus Born-consistency loss,
  - scheduler and optimizer settings used by the current paper lane unless
    intentionally changed.
- Write artifacts under:

```text
.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-07-brdt-sinogram-input-40ep-paper-evidence/
```

- Emit:
  - per-row metrics CSV,
  - config snapshots,
  - source arrays for sample 255,
  - Figure 3 source arrays,
  - timing/throughput measurements over the 256-sample test split,
  - training loss per epoch.

Do not overwrite the previous `2026-05-06` BRDT evidence bundle.

## Required Plan, Spec, and Backlog Updates

### Consistency-pass findings

- `semantic_conflict`: the current BRDT adapter contract explicitly rejects
  `sinogram`, while the new scientific contract requires it as learned-model
  input.
- `stale_duplicate`: `brdt_task_adapters.md` and `brdt_preflight_summary.md`
  describe `born_init_image` as the locked input contract. Those statements
  remain true only for the historical preflight lane.
- `routing_mismatch`: `docs/backlog/index.md` and existing active/done paths
  still describe the `2026-05-06` BRDT reruns as the current path to BRDT
  paper evidence. New active backlog entries and the index must supersede that
  route for sinogram-input claims.
- `label_driven_policy`: earlier wording around candidate, secondary, or paper
  evidence must not decide admissibility. The deciding field is whether the
  artifact root satisfies the current input, split, loss, metric, visual, and
  provenance contract.
- `discoverability_gap`: new artifact roots will not be findable unless the
  backlog index, evidence matrix, manifest, model-variant index, and paper
  refresh scripts all point to the new source.

### New backlog files

Create:

- `docs/backlog/active/2026-05-07-brdt-sinogram-input-adapter-contract.md`
- `docs/backlog/active/2026-05-07-brdt-sinogram-input-40ep-paper-evidence.md`

Each backlog item should state:

- the model input is the measured complex sinogram;
- Born inverse is a non-learned baseline only;
- the old Born-image-input rows are superseded for sinogram-input manuscript
  BRDT claims;
- success requires tests, artifact lineage, and manuscript refresh.

Also update:

- `docs/backlog/index.md`
- `docs/backlog/roadmap_gate.json` only if the existing `candidate-` gate does
  not admit the new backlog item's roadmap phase.

Use roadmap phase `candidate-brdt-sinogram-input` or an equivalent
`candidate-` prefix so the existing gate admits the work without opening Phase 4
or Phase 5. The adapter-contract item should be listed as the prerequisite for
the 40-epoch paper-evidence item.

### New execution plans

Create:

- `docs/plans/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-07-brdt-sinogram-input-adapter-contract/execution_plan.md`
- `docs/plans/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-07-brdt-sinogram-input-40ep-paper-evidence/execution_plan.md`

### Existing evidence/index updates

After the new 40-epoch run succeeds, update:

- `docs/plans/NEURIPS-HYBRID-RESNET-2026/brdt_task_adapters.md`
- `docs/plans/NEURIPS-HYBRID-RESNET-2026/brdt_preflight_summary.md`
- `docs/plans/NEURIPS-HYBRID-RESNET-2026/paper_evidence_index.md`
- `docs/plans/NEURIPS-HYBRID-RESNET-2026/paper_evidence_manifest.json`
- `docs/plans/NEURIPS-HYBRID-RESNET-2026/evidence_matrix.md`
- `docs/plans/NEURIPS-HYBRID-RESNET-2026/ablation_index.json`
- `docs/plans/NEURIPS-HYBRID-RESNET-2026/model_variant_index.json`
- `docs/plans/NEURIPS-HYBRID-RESNET-2026/evidence_inventory.md`

Update these to mark the old BRDT rows as historical Born-image-input lineage
and the new `2026-05-07` run as the current BRDT manuscript source. In
`brdt_task_adapters.md` and `brdt_preflight_summary.md`, preserve the old
`born_init_image` statements as the first bounded preflight contract and add a
separate sinogram-input contract section rather than rewriting history.

### Historical summaries

Add brief contract-supersession notes where useful, without rewriting old
outcomes:

- `docs/plans/NEURIPS-HYBRID-RESNET-2026/brdt_corrected_ffno_40ep_rerun_summary.md`
- `docs/backlog/done/2026-05-06-brdt-corrected-ffno-40ep-rerun.md`
- `docs/backlog/done/2026-05-05-brdt-supervised-born-40ep-paper-evidence.md`

Suggested wording:

```text
Contract-supersession note: This completed item used the historical
Born-image-input contract. It remains valid lineage for that contract and may be
cited only as such. Sinogram-input BRDT manuscript claims are governed by
2026-05-07-brdt-sinogram-input-40ep-paper-evidence.
```

## Manuscript and Artifact Updates After Rerun

Files:

- `docs/plans/NEURIPS-HYBRID-RESNET-2026/hybrid_resnet_neurips_first_draft.tex`
- `docs/plans/NEURIPS-HYBRID-RESNET-2026/figures/brdt_sample_0255_context_recon_error.png`
- `docs/plans/NEURIPS-HYBRID-RESNET-2026/tables/model_config_by_benchmark.tex`
- `docs/plans/NEURIPS-HYBRID-RESNET-2026/tables/paper_efficiency_table.tex`
- `scripts/studies/paper_results_refresh.py`
- `scripts/studies/paper_model_config_table.py`
- `scripts/studies/paper_efficiency_table.py`

Tasks:

- Replace manuscript statements that say the ML input is a Born-derived image.
- State that BRDT neural models take the measured complex sinogram as input.
- State that the displayed sinogram panel is the magnitude of that complex input.
- State that the Born inverse is a non-learned reference, not a learned-model
  input.
- Regenerate the BRDT table from the new metrics.
- Regenerate the efficiency table from the new timing outputs.
- Regenerate Figure 3 from sample 255 with the current optimal layout:

```text
Top row:
Target q | Born inverse | FFNO | SRU-Net

Bottom row:
Input |s_obs| | |Born inverse - target| | |FFNO - target| | |SRU-Net - target|
```

Figure caption contract:

- The input panel is the measured sinogram magnitude.
- The neural models consume the complex real/imaginary sinogram, not the
  displayed magnitude alone.
- Error panels use a shared target-domain color scale.
- The Born inverse is a non-learned reference.

## Verification Plan

### Adapter-contract verification

Run narrow tests first:

```bash
pytest --collect-only -q tests/studies/test_born_rytov_dt_adapters.py tests/studies/test_born_rytov_dt_preflight.py
pytest -q tests/studies/test_born_rytov_dt_adapters.py -k "sinogram or input_mode or model"
pytest -q tests/studies/test_born_rytov_dt_preflight.py -k "sinogram or input_mode or brdt"
python -m compileall -q scripts/studies/born_rytov_dt ptycho_torch
```

Add or run a fast smoke check:

```bash
python -m scripts.studies.born_rytov_dt.run_sinogram_input_smoke
```

If the smoke command is new, include it in the adapter-contract backlog item.

### Full-run verification

Use `tmux` and the `ptycho311` environment for the long run:

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ptycho311
python -m scripts.studies.born_rytov_dt.run_sinogram_input_40ep
```

Expected completed rows:

- `sru_net`
- `ffno`
- `born_inverse` or equivalent non-learned baseline metrics

Gate the run on:

- both neural rows reach 40 epochs;
- metrics CSV contains image-space error, measurement error, PSNR, SSIM,
  parameter count, and throughput;
- sample 255 source arrays exist;
- training loss per epoch is recorded;
- config snapshot states `input_mode=sinogram`.

### Paper-refresh verification

After refreshing manuscript artifacts:

```bash
python scripts/studies/paper_results_refresh.py
python scripts/studies/paper_model_config_table.py
python scripts/studies/paper_efficiency_table.py
```

Build the manuscript and package:

```bash
latexmk -pdf docs/plans/NEURIPS-HYBRID-RESNET-2026/hybrid_resnet_neurips_first_draft.tex
```

Then verify:

- no missing figures or tables;
- no stale phrases such as `Born input`, `born_init_image` in visible BRDT
  manuscript prose;
- no stale paper-refresh source constants still point BRDT manuscript assets to
  the old `2026-05-06` artifact root after the new run succeeds;
- no manuscript claim points to the old `2026-05-06` BRDT artifact root;
- the zip contains the PDF and refreshed BRDT figure/table assets.

## Acceptance Criteria

- New BRDT paper evidence uses complex sinogram input for learned models.
- Born inverse appears only as a non-learned reference.
- Tests fail if the new learned-model input path silently derives a Born image.
- 40-epoch SRU-Net and FFNO rows complete under the new contract.
- Figure 3, BRDT table, efficiency table, model configuration table, evidence
  manifest, and manuscript prose all point to the new artifact root.
- Old completed BRDT backlog items remain discoverable but are clearly marked
  as Born-image-input lineage that cannot support sinogram-input claims.
