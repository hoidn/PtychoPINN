# Lines128 NeuralOperator U-NO Preflight Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:executing-plans` or `superpowers:subagent-driven-development` to implement this plan task-by-task. Keep this file as the execution authority for the selected backlog item.

**Goal:** Verify the external NeuralOperator `UNO` environment/API contract in `ptycho311`, freeze the exact `neuralop_uno` row settings for the locked `lines128` CDI contract, and emit a durable readiness-or-blocker summary without adding generator registry support or benchmark rows.

**Architecture:** Keep this item preflight-only. Add one bounded study helper that inspects the installed `neuraloperator` package, records runtime/package/signature provenance, runs only a tiny dummy forward/shape probe against the locked `lines128` CDI channel contract, then writes a machine-readable decision artifact plus a durable summary. Treat the existing six-row `lines128` bundle and the draft U-NO extension design as binding authority; do not reopen the completed table or touch the execution wrappers for actual rows.

**Tech Stack:** Python 3.11 via PATH `python`, `ptycho311`, PyTorch, optional CUDA, `neuraloperator==2.0.0`, Markdown/JSON artifact writing, pytest.

---

## Selected Backlog Objective

- Verify the local U-NO environment/API contract and freeze the exact
  `neuralop_uno` row settings needed for a later append-only `lines128` table
  extension.
- Emit one durable preflight outcome:
  - `ready_for_uno_generator_integration`
  - `blocked_neuraloperator_missing_or_incompatible`
  - `blocked_uno_shape_contract_mismatch`

## Scope

In scope:

- consume `docs/plans/NEURIPS-HYBRID-RESNET-2026/lines128_uno_table_extension_design.md`
- consume the authoritative completed base-table summary
  `docs/plans/NEURIPS-HYBRID-RESNET-2026/lines128_paper_benchmark_summary.md`
- verify in `ptycho311`:
  - distribution package `neuraloperator`
  - import module `neuralop`
  - `neuralop.models.UNO`
- record:
  - Python executable/version
  - Torch version
  - CUDA version
  - GPU visibility
  - `pip show neuraloperator` provenance
  - `neuralop.__version__`
  - local `inspect.signature(UNO)` or an equivalent frozen API summary
- freeze pre-metric U-NO settings for the later integration item:
  - `in_channels`
  - `out_channels`
  - `hidden_channels`
  - `lifting_channels`
  - `projection_channels`
  - `n_layers`
  - `uno_n_modes`
  - `positional_embedding`
  - `generator_output_mode`
- run only tiny API/shape probes with dummy tensors on the locked `N=128`,
  `gridsize=1`, `real_imag` CDI contract
- write a durable preflight summary and machine-readable decision artifact
- update the relevant NeurIPS evidence indexes if the preflight becomes a
  completed durable backlog output

Out of scope:

- adding `neuralop_uno` to the generator registry
- modifying model/config architecture enums for runtime use
- editing the compare wrapper or paper-table launcher to execute U-NO rows
- running any paper benchmark row
- changing the completed six-row `lines128` authority
- tuning U-NO defaults after seeing benchmark metrics
- creating or writing `/home/ollie/Documents/neurips/`

## Explicit Non-Goals

- Do not substitute the repo’s internal `HybridUNOGenerator` for external
  NeuralOperator `UNO`.
- Do not touch `ptycho_torch/generators/registry.py`,
  `ptycho/config/config.py`, `scripts/studies/grid_lines_torch_runner.py`,
  `scripts/studies/grid_lines_compare_wrapper.py`, or
  `scripts/studies/lines128_paper_benchmark.py` unless a narrow follow-on item
  explicitly reopens generator integration.
- Do not mark the item `BLOCKED` just because the current shell is not yet in
  `ptycho311`, because `neuraloperator` is missing before a narrow install
  attempt, or because an initial probe/import/test fails. Diagnose, apply one
  narrow fix, and rerun first.
- Do not launch tmux-backed training or benchmark execution in this item.

## Binding Constraints And Prerequisite Status

Strategic and roadmap constraints:

- `docs/steering.md` requires equal-footing and apples-to-apples discipline.
  This item therefore freezes U-NO settings before any metrics exist and cannot
  relax the `lines128` contract to make U-NO easier to integrate.
- `docs/plans/2026-04-20-neurips-hybrid-resnet-submission-roadmap.md` places
  U-NO only in the optional append-only follow-up lane under Phase `3.3f`. It
  is not required Phase 3 completion work and cannot rewrite the authoritative
  six-row bundle.
- `docs/plans/2026-04-20-neurips-hybrid-resnet-submission-design.md` keeps the
  CDI headline fixed at `128x128` and treats later extensions as bounded,
  provenance-heavy add-ons.
- The selected backlog item explicitly authorizes only environment/API
  verification, frozen settings, and tiny shape probes.

Prerequisite status that matters here:

- `state/NEURIPS-HYBRID-RESNET-2026/progress_ledger.json` records only the
  early initiative phases; it does not yet capture the later `2026-04-29` and
  `2026-04-30` CDI backlog completions this item depends on.
- For this item, treat the checked-in CDI authorities below as the binding
  prerequisites instead of waiting for a ledger rewrite:
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/lines128_paper_benchmark_summary.md`
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/lines128_supervised_equivalent_rows_summary.md`
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/lines128_uno_table_extension_design.md`

Locked `lines128` contract facts that implementation must preserve:

- fixed base-table root:
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-paper-benchmark-execution/runs/complete_table_20260430T150757Z_repair_tmux`
- selected FNO comparator: `fno_vanilla`
- fixed seed policy: `seed=3`
- fixed sample ids: `0`, `1`
- fixed CDI U-NO preflight channel/output contract:
  - input patch contract derived from the current CDI generator path:
    `in_channels=1`
  - real/imag output contract for the later wrapper:
    `out_channels=2`
  - locked output mode:
    `generator_output_mode=real_imag`
- fixed optimizer/training defaults to carry forward unchanged into the later
  U-NO row item:
  - `torch_epochs=40`
  - `torch_learning_rate=2e-4`
  - `torch_scheduler=ReduceLROnPlateau`
  - `torch_plateau_factor=0.5`
  - `torch_plateau_patience=2`
  - `torch_plateau_min_lr=1e-4`
  - `torch_plateau_threshold=0.0`
  - `torch_loss_mode=mae`

Failure-handling policy for this item:

- If `neuraloperator` is absent in `ptycho311`, perform one narrow remedial
  install of the pinned package (`neuraloperator==2.0.0`) into `ptycho311`,
  rerun the probes, and only then consider
  `blocked_neuraloperator_missing_or_incompatible`.
- If `UNO` imports but the signature or forward output differs from the draft
  design, freeze the actual API if it can still satisfy the locked CDI
  contract. Only emit `blocked_uno_shape_contract_mismatch` when the output
  shape or constructor requirements remain incompatible after one documented
  narrow interpretation/fix attempt.
- Reserve `BLOCKED` for an unrecoverable external package/API mismatch,
  irreconcilable output-contract conflict, unavailable required environment, or
  another failure that remains unresolved after the documented narrow fix
  attempt.

## Implementation Architecture

- **Preflight probe unit**
  - One bounded helper script owns package discovery, runtime/package
    provenance capture, constructor-signature capture, and the tiny forward
    probe.
- **Decision publication unit**
  - Machine-readable preflight outputs declare the frozen U-NO settings, raw
    observed API surface, probe outcome, and final item status.
- **Durable knowledge unit**
  - The design doc and summary become the authoritative handoff for the later
    generator-integration item; evidence indexes are updated only as audit/readiness
    surfaces, not as new model rows.

## Concrete File And Artifact Targets

Likely code targets:

- Create: `scripts/studies/lines128_uno_preflight.py`
- Modify only if reusable provenance helpers reduce duplication cleanly:
  `scripts/studies/paper_provenance.py`

Likely test targets:

- Create: `tests/studies/test_lines128_uno_preflight.py`
- Modify only if a shared helper changes:
  `tests/studies/test_paper_provenance.py`

Durable document targets:

- Modify: `docs/plans/NEURIPS-HYBRID-RESNET-2026/lines128_uno_table_extension_design.md`
- Create: `docs/plans/NEURIPS-HYBRID-RESNET-2026/lines128_uno_preflight_summary.md`
- Modify: `docs/plans/NEURIPS-HYBRID-RESNET-2026/paper_evidence_index.md`
- Modify: `docs/plans/NEURIPS-HYBRID-RESNET-2026/evidence_matrix.md`
- Modify: `docs/plans/NEURIPS-HYBRID-RESNET-2026/ablation_index.json`
- Modify `docs/index.md` if the new preflight summary becomes a durable
  discoverable authority distinct from the design doc

Item-local artifact root:

- `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-30-cdi-lines128-uno-design-preflight/`

Expected item-local artifacts:

- `preflight_inputs.json`
- `environment_probe.json`
- `pip_show_neuraloperator.txt`
- `uno_signature.json`
- `uno_shape_probe.json`
- `preflight_decision.json`
- archived verification logs under `verification/`

Do not create model-row artifact roots or promoted table bundles in this item.

## Execution Checklist

### Tranche 1: Freeze Authorities, Contract Inputs, And Decision Schema

- [ ] Create
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-30-cdi-lines128-uno-design-preflight/preflight_inputs.json`
  with the exact source authorities this item consumes:
  - `selected-item-context.md`
  - `lines128_uno_table_extension_design.md`
  - `lines128_paper_benchmark_summary.md`
  - `lines128_supervised_equivalent_rows_summary.md`
- [ ] Copy the locked `lines128` contract into the preflight input payload:
  `N=128`, `gridsize=1`, `set_phi=True`, custom Run1084 probe,
  `probe_scale_mode=pad_extrapolate`, `probe_smoothing_sigma=0.5`,
  `nimgs_train=2`, `nimgs_test=2`, `nphotons=1e9`, `seed=3`,
  `torch_output_mode=real_imag`, and the fixed scheduler/loss fields.
- [ ] Freeze the later wrapper-facing U-NO channel policy before probing:
  - `in_channels=1`
  - `out_channels=2`
  - `generator_output_mode=real_imag`
- [ ] Freeze the preflight decision schema with required top-level fields:
  - `status`
  - `package_status`
  - `runtime_provenance`
  - `package_provenance`
  - `uno_signature`
  - `locked_contract`
  - `frozen_uno_settings`
  - `shape_probe`
  - `next_item_recommendation`
  - `blocker_reason`
- [ ] Freeze the only allowed status values:
  - `ready_for_uno_generator_integration`
  - `blocked_neuraloperator_missing_or_incompatible`
  - `blocked_uno_shape_contract_mismatch`

Verification before moving on:

- [ ] Run the first required deterministic check verbatim:

```bash
python - <<'PY'
from pathlib import Path
required = [
    Path("docs/plans/NEURIPS-HYBRID-RESNET-2026/lines128_uno_table_extension_design.md"),
    Path("docs/plans/NEURIPS-HYBRID-RESNET-2026/lines128_paper_benchmark_summary.md"),
]
missing = [str(p) for p in required if not p.exists()]
if missing:
    raise SystemExit(f"missing U-NO preflight inputs: {missing}")
print("U-NO preflight inputs present")
PY
```

- [ ] Archive the command output under the item-local `verification/` directory.

### Tranche 2: Red Tests For Preflight Status Mapping And Shape Acceptance

- [ ] Add `tests/studies/test_lines128_uno_preflight.py`.
- [ ] Write failing tests for the bounded preflight contract:
  - missing `neuralop` or missing `UNO` returns
    `blocked_neuraloperator_missing_or_incompatible`
  - successful import records `neuralop.__version__`, constructor signature,
    and package provenance
  - a raw `UNO` output that is already `B x 2 x 128 x 128` or can be
    losslessly interpreted as one complex channel in `real_imag` form is
    accepted
  - an output rank/channel layout that cannot be mapped into the locked CDI
    real/imag contract returns `blocked_uno_shape_contract_mismatch`
  - the frozen defaults are recorded before any metric-bearing execution:
    `hidden_channels=32`, `lifting_channels=128`, `projection_channels=128`,
    `n_layers=4`, `positional_embedding=grid`, `generator_output_mode=real_imag`
  - `uno_n_modes` is frozen in the exact canonical form the live API accepts:
    prefer an explicit per-layer four-entry `12` sequence when supported;
    otherwise record the accepted scalar/int form and explain it in the summary
- [ ] If `paper_provenance.py` gets a reusable helper, add focused regression
  coverage in `tests/studies/test_paper_provenance.py`.
- [ ] Run the new pytest selector and verify it fails before implementation.

Verification before moving on:

- [ ] Archive the red-phase pytest log under:
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-30-cdi-lines128-uno-design-preflight/verification/pytest_red.log`

### Tranche 3: Implement The Bounded U-NO Preflight Helper

- [ ] Create `scripts/studies/lines128_uno_preflight.py` as the single
  preflight entrypoint for this backlog item.
- [ ] The helper must:
  - run under PATH `python`
  - assume `ptycho311` is activated by the caller
  - gather runtime provenance using the existing paper-provenance conventions
  - capture `pip show neuraloperator` text
  - import `neuralop` and `neuralop.models.UNO`
  - record `neuralop.__version__`
  - record `inspect.signature(UNO)` or a stable structured equivalent
  - instantiate `UNO` only with pre-metric settings consistent with the draft
    design and the locked CDI contract
  - run a tiny dummy forward probe on shape `(B=2, C=1, H=128, W=128)`
  - determine whether the observed output can be mapped cleanly into the later
    `real_imag` wrapper contract
  - write the JSON artifacts listed in this plan
- [ ] Keep the helper narrow:
  - no registry writes
  - no compare-wrapper invocation
  - no dataset generation
  - no benchmark or training dispatch
- [ ] If `neuraloperator` is missing, perform the single permitted narrow fix:
  install `neuraloperator==2.0.0` into `ptycho311`, then rerun the helper.
- [ ] If the package import succeeds but `UNO` has an incompatible constructor,
  capture the actual signature and stop with
  `blocked_neuraloperator_missing_or_incompatible`.
- [ ] If the constructor works but the forward output cannot satisfy the locked
  CDI shape contract after one narrow interpretation, stop with
  `blocked_uno_shape_contract_mismatch`.

Verification before moving on:

- [ ] Rerun the tranche-2 pytest selector and verify it passes.
- [ ] Archive the green-phase pytest log under:
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-30-cdi-lines128-uno-design-preflight/verification/pytest_green.log`
- [ ] If a new script was added, run:

```bash
python -m compileall -q scripts/studies
```

- [ ] Archive the compile log or the successful command transcript under the
  same `verification/` directory.

### Tranche 4: Run The Live `ptycho311` Probe And Freeze The Exact U-NO Settings

- [ ] Activate `ptycho311` and run the second required deterministic check
  there. Use PATH `python`, not a repo-local interpreter wrapper:

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ptycho311
python - <<'PY'
import neuralop
from neuralop.models import UNO
print(f"neuralop {neuralop.__version__}: {UNO}")
PY
```

- [ ] Archive the output under:
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-30-cdi-lines128-uno-design-preflight/verification/neuralop_import_check.log`
- [ ] Run the new preflight helper in the same activated environment and write:
  - `environment_probe.json`
  - `pip_show_neuraloperator.txt`
  - `uno_signature.json`
  - `uno_shape_probe.json`
  - `preflight_decision.json`
- [ ] Freeze the exact U-NO settings from the observed API, not from later
  metric feedback. Unless the live API forces a documented change, keep:
  - `in_channels=1`
  - `out_channels=2`
  - `hidden_channels=32`
  - `lifting_channels=128`
  - `projection_channels=128`
  - `n_layers=4`
  - `positional_embedding="grid"`
  - `generator_output_mode="real_imag"`
  - `uno_n_modes` aligned to the locked `fno_modes=12` contract in the exact
    form accepted by the live API
- [ ] Record the accepted raw-output interpretation explicitly:
  - raw `UNO` output shape observed
  - whether a thin adapter to the repo’s `real_imag` contract is lossless
  - whether the later integration item must add only a layout adapter versus a
    deeper semantic conversion
- [ ] If the probe passes, set status
  `ready_for_uno_generator_integration` and name the next backlog item as the
  generator-integration tranche.
- [ ] If the probe fails after the documented narrow fix attempt, set the
  appropriate blocker status and include the exact blocker reason.

Verification before moving on:

- [ ] Confirm every JSON artifact is parseable and that the final decision file
  names the same frozen settings as the summary/design updates.
- [ ] Confirm no model-row directories, benchmark tables, or promoted bundles
  were created by this item.

### Tranche 5: Publish Durable Summary, Update The Design, And Refresh Indexes

- [ ] Write
  `docs/plans/NEURIPS-HYBRID-RESNET-2026/lines128_uno_preflight_summary.md`.
- [ ] The summary must include:
  - final status
  - authoritative input paths
  - exact package/runtime provenance
  - observed `UNO` signature
  - frozen U-NO settings
  - accepted raw-output shape and wrapper-mapping decision
  - explicit reminder that no registry support or benchmark rows ran
  - the handoff target for the later generator-integration item
- [ ] Update
  `docs/plans/NEURIPS-HYBRID-RESNET-2026/lines128_uno_table_extension_design.md`
  to replace provisional wording with the verified/frozen API facts and to
  reference the new preflight summary as the current environment-contract
  authority.
- [ ] Update `docs/plans/NEURIPS-HYBRID-RESNET-2026/paper_evidence_index.md`
  with a completed backlog row for this preflight outcome.
- [ ] Update `docs/plans/NEURIPS-HYBRID-RESNET-2026/evidence_matrix.md` and
  `docs/plans/NEURIPS-HYBRID-RESNET-2026/ablation_index.json` because this is
  a readiness/audit-style completed output, not a new evaluated model row.
- [ ] Do not add a `model_variant_index.json` row unless implementation
  accidentally evaluates real benchmark rows, which this plan forbids.
- [ ] Update `docs/index.md` if the new preflight summary should be directly
  discoverable separately from the extension design.

Verification before completion:

- [ ] Rerun the first deterministic check after document updates.
- [ ] Rerun the green pytest selector if any summary/design schema tests depend
  on document content.
- [ ] Confirm the summary, updated design doc, and index entries all point to
  the same artifact root and final status.

## Verification Commands

Required deterministic checks for this item:

```bash
python - <<'PY'
from pathlib import Path
required = [
    Path("docs/plans/NEURIPS-HYBRID-RESNET-2026/lines128_uno_table_extension_design.md"),
    Path("docs/plans/NEURIPS-HYBRID-RESNET-2026/lines128_paper_benchmark_summary.md"),
]
missing = [str(p) for p in required if not p.exists()]
if missing:
    raise SystemExit(f"missing U-NO preflight inputs: {missing}")
print("U-NO preflight inputs present")
PY
```

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ptycho311
python - <<'PY'
import neuralop
from neuralop.models import UNO
print(f"neuralop {neuralop.__version__}: {UNO}")
PY
```

Focused expected pytest selector:

```bash
pytest -v tests/studies/test_lines128_uno_preflight.py
```

Add this if `paper_provenance.py` changes:

```bash
pytest -v tests/studies/test_lines128_uno_preflight.py tests/studies/test_paper_provenance.py
```

Optional syntax gate if a new helper script is created:

```bash
python -m compileall -q scripts/studies
```

## Completion Criteria

- The live `ptycho311` environment has been probed and the result is captured in
  durable artifacts.
- The exact `UNO` constructor/API facts are frozen in the design doc and
  preflight summary.
- The final status is one of the three allowed outcomes and is justified by the
  recorded evidence.
- No generator registry support, row execution, or benchmark-table mutation was
  performed.
- The relevant evidence indexes are updated consistently for a readiness/audit
  output.
