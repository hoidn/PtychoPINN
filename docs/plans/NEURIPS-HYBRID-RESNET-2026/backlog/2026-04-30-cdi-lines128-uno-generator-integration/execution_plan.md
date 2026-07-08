# Lines128 NeuralOperator U-NO Generator Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:executing-plans` or `superpowers:subagent-driven-development` to implement this plan task-by-task. Keep this file as the execution authority for the selected backlog item.

**Goal:** Integrate `neuralop_uno` as a real PyTorch generator architecture for the locked `lines128` CDI lane so both PINN and supervised training procedures use the external NeuralOperator U-NO body through the existing Lightning/generator-registry path, without launching benchmark rows or mutating the completed paper bundle.

**Architecture:** Add one bounded generator adapter module around `neuralop.models.UNO`, then wire that adapter through the existing registry, config, checkpoint-rebuild, and direct runner surfaces that already support other CDI Torch generators. Keep the contract intentionally narrow: this item is for the preflight-frozen `N=128`, `gridsize=1`, `real_imag`, external-U-NO path only. Fail closed on unsupported channel/output cases instead of guessing a broader CDI contract.

**Tech Stack:** PATH `python`, PyTorch/Lightning, external `neuraloperator==2.0.0` via `neuralop`, `ptycho_torch` generator registry, grid-lines Torch runner, pytest, `compileall`, Markdown docs, repo-local `.artifacts/` verification logs.

---

## Selected Backlog Objective

- Implement backlog item `2026-04-30-cdi-lines128-uno-generator-integration`.
- Add `neuralop_uno` as a real registered Torch generator architecture.
- Prove that:
  - `architecture=neuralop_uno, mode=Unsupervised` uses the U-NO generator body.
  - `architecture=neuralop_uno, mode=Supervised` uses the same U-NO generator body through `Ptycho_Supervised`, not the legacy supervised autoencoder path.
- Preserve the existing CDI `real_imag` generator output contract and fail closed when the external U-NO output cannot satisfy it.

## Scope And Explicit Non-Goals

In scope:

- create `ptycho_torch/generators/neuralop_uno.py` as the repo adapter around `neuralop.models.UNO`
- register `neuralop_uno` in both config surfaces and the generator registry
- extend checkpoint rebuild / model reconstruction so saved `neuralop_uno` checkpoints can be reloaded without ad hoc code paths
- make the direct grid-lines Torch runner accept and label `neuralop_uno` for both `pinn` and `supervised` procedures
- add focused regression coverage for:
  - registry resolution
  - adapter forward/output-shape behavior
  - supervised and PINN Lightning construction
  - checkpoint rebuild coverage
  - missing/incompatible `neuraloperator` dependency behavior
- update durable architecture/workflow docs where the supported-architecture surface changes

Explicit non-goals:

- do not launch full `lines128` benchmark rows
- do not add append-only paper-bundle execution or promotion logic
- do not modify `scripts/studies/grid_lines_compare_wrapper.py` unless a narrowly scoped integration bug makes that unavoidable; model-ID/table-extension work belongs to a later append-only row item
- do not tune U-NO hyperparameters beyond the frozen preflight settings
- do not broaden the CDI contract to `gridsize>1`, multi-channel grouped outputs, or non-`real_imag` output modes by inference
- do not replace external NeuralOperator `UNO` with the repo’s internal `HybridUNOGenerator`
- do not modify `ptycho/model.py`, `ptycho/diffsim.py`, or `ptycho/tf_helper.py`
- do not write `/home/ollie/Documents/neurips/`

## Binding Constraints And Prerequisite Status

Strategic and roadmap constraints:

- `docs/steering.md` requires equal-footing discipline and forbids silently relaxing protocol. This item must therefore preserve the locked `lines128` contract and fail closed rather than silently reshaping the architecture contract to make U-NO easier.
- `docs/plans/2026-04-20-neurips-hybrid-resnet-submission-roadmap.md` places U-NO in optional Phase `3.3f` append-only follow-up work. This item is implementation-enabling plumbing only; it cannot rewrite the authoritative six-row bundle or become a hidden benchmark-launch scope expansion.
- `docs/plans/2026-04-20-neurips-hybrid-resnet-submission-design.md` keeps the CDI headline at `128x128` and treats later comparator additions as bounded, provenance-heavy extensions.

Prerequisite status that matters:

- `state/NEURIPS-HYBRID-RESNET-2026/progress_ledger.json` confirms the initiative’s early phases are complete and shows no blocked tranche that prevents current Phase 3 CDI-preparation work.
- The backlog-item prerequisite `2026-04-30-cdi-lines128-uno-design-preflight` is satisfied by `docs/plans/NEURIPS-HYBRID-RESNET-2026/lines128_uno_preflight_summary.md`, even though that microstate is not yet reflected in the ledger.
- Treat the following as binding authority for this item:
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/lines128_uno_preflight_summary.md`
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/lines128_uno_table_extension_design.md`
  - `docs/backlog/in_progress/2026-04-30-cdi-lines128-uno-generator-integration.md`

Frozen U-NO settings from preflight that implementation must use:

- `in_channels=1`
- `out_channels=2`
- `hidden_channels=32`
- `lifting_channels=128`
- `projection_channels=128`
- `n_layers=4`
- `uno_out_channels=[32, 64, 64, 32]`
- `uno_n_modes=[[12, 12], [12, 12], [12, 12], [12, 12]]`
- `uno_scalings=[[1.0, 1.0], [0.5, 0.5], [1, 1], [2, 2]]`
- `positional_embedding="grid"`
- `channel_mlp_skip="linear"`
- `generator_output_mode="real_imag"`

Claim boundaries that must remain explicit:

- This item only guarantees the external U-NO path for the locked `lines128` CDI contract.
- The raw verified NeuralOperator output was `B x 2 x 128 x 128`; the repo adapter must convert that into the existing CDI `real_imag` generator contract of `(B, H, W, C, 2)` with `C=1`.
- If `data_config.C != 1`, if the external U-NO shape is not exactly compatible, or if a non-`real_imag` output mode is requested, stop with a clear runtime error. Do not invent multi-channel grouping behavior in this item.

Failure policy:

- Do not mark the item `BLOCKED` for ordinary import, test, path, or environment failures. Diagnose, patch narrowly, and rerun first.
- Reserve `BLOCKED` for:
  - external `neuraloperator` API loss or incompatibility that remains unresolved after one narrow fix attempt
  - an unrecoverable CDI output-contract mismatch
  - a broader roadmap or user-authority conflict that this item cannot narrow away

## Implementation Architecture

- **Adapter unit**
  - `ptycho_torch/generators/neuralop_uno.py` owns the external `UNO` import, frozen constructor defaults, shape validation, `real_imag` output adaptation, and fail-closed dependency checks.
- **Runtime plumbing unit**
  - Existing registry/config/checkpoint/runner surfaces gain just enough awareness of `neuralop_uno` for direct PINN and supervised construction, checkpoint reload, and runner labeling.
- **Proof and docs unit**
  - Focused regression tests prove architecture identity and failure behavior. Architecture/workflow docs are updated to describe the new supported surface and its intentional limits.

## Concrete File And Artifact Targets

Core code targets:

- Create: `ptycho_torch/generators/neuralop_uno.py`
- Modify: `ptycho_torch/generators/registry.py`
- Modify: `ptycho/config/config.py`
- Modify: `ptycho_torch/config_params.py`
- Modify: `ptycho_torch/model.py`
- Modify only if direct-runner labeling/support needs it: `scripts/studies/grid_lines_torch_runner.py`

Likely test targets:

- Create: `tests/torch/test_neuralop_uno_generator.py`
- Modify: `tests/torch/test_generator_registry.py`
- Modify: `tests/torch/test_lightning_checkpoint.py`
- Modify: `tests/torch/test_grid_lines_torch_runner.py`
- Modify only if a shared supervised-construction assertion belongs there cleanly: `tests/torch/test_workflows_components.py`

Durable documentation targets:

- Modify: `ptycho_torch/generators/README.md`
- Modify: `docs/workflows/pytorch.md`
- Modify: `docs/CONFIGURATION.md`
- Modify: `docs/architecture_torch.md`
- Update `docs/findings.md` only if implementation uncovers a durable trap not already captured by the preflight summary or the architecture docs

Item-local verification artifact root:

- `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-30-cdi-lines128-uno-generator-integration/verification/`

Evidence-index policy for this item:

- This item does not produce a benchmark row, table bundle, or paper-facing artifact by itself.
- Do not update `model_variant_index.json`, `ablation_index.json`, or `paper_evidence_index.md` unless implementation also writes a new durable summary or design amendment that later planning/review needs to discover.
- If a new durable summary or discoverable architecture note is added, update `docs/index.md` accordingly; otherwise no index change is required.

## Execution Checklist

### Task 1: Add Red Tests For The New Architecture Boundary

**Files:**
- Create: `tests/torch/test_neuralop_uno_generator.py`
- Modify: `tests/torch/test_generator_registry.py`
- Modify: `tests/torch/test_lightning_checkpoint.py`
- Modify: `tests/torch/test_grid_lines_torch_runner.py`
- Optional modify: `tests/torch/test_workflows_components.py`

- [ ] Add a registry test that `TrainingConfig(model=ModelConfig(architecture="neuralop_uno"))` resolves to a generator whose `name` is `neuralop_uno`.
- [ ] Add focused adapter tests that build the U-NO core with the frozen preflight settings and verify the adapted output is `(B, H, W, 1, 2)` for a dummy `N=128` input.
- [ ] Add a fail-closed test for unsupported `data_config.C != 1`.
- [ ] Add a fail-closed test for unsupported generator output modes other than `real_imag`.
- [ ] Add a missing-dependency test that proves the builder raises a clear actionable error when `neuralop.models.UNO` is unavailable or incompatible.
- [ ] Add supervised and PINN construction tests proving both modes use the U-NO generator body, and that supervised mode still lands in `Ptycho_Supervised` with `generator_output="real_imag"`.
- [ ] Extend checkpoint coverage so `neuralop_uno` appears in the architecture-to-generator-class expectations and checkpoint reload rebuilds the correct generator module.
- [ ] Add or update a direct runner test proving `TorchRunnerConfig(architecture="neuralop_uno")` is accepted for both `training_procedure="pinn"` and `training_procedure="supervised"`.

**Verification before moving on:**

- [ ] The new/updated tests fail for the right reasons before implementation.
- [ ] The red failures show missing registry wiring, missing checkpoint rebuild support, or missing adapter behavior, not unrelated infrastructure noise.

### Task 2: Implement The External U-NO Adapter With A Narrow CDI Contract

**Files:**
- Create: `ptycho_torch/generators/neuralop_uno.py`

- [ ] Implement a generator-registry wrapper class named `NeuralopUnoGenerator` with `name = "neuralop_uno"`.
- [ ] Implement the underlying module so it imports external `neuralop.models.UNO` lazily and raises a clear error if `neuraloperator==2.0.0` is missing or the API surface is incompatible.
- [ ] Hard-code the preflight-frozen constructor values instead of inventing new tuning knobs in this item.
- [ ] Enforce the locked CDI assumptions explicitly:
  - only `C=1`
  - only `real_imag`
  - only shape-compatible channel-first raw UNO output
- [ ] Adapt raw `UNO` output from `B x 2 x H x W` into repo contract `(B, H, W, 1, 2)` and validate the shape before returning.
- [ ] Keep the adapter small and local. If extra helper functions are needed, keep them inside `neuralop_uno.py` unless another existing generator can reuse them immediately.

**Verification before moving on:**

- [ ] Focused adapter tests are green, including the failure-path tests.
- [ ] The adapter error messages mention the actual unsupported condition (`missing neuraloperator`, `unsupported C`, `unsupported output mode`, or output-shape mismatch).

### Task 3: Wire Registry, Config, Checkpoint Rebuild, And Direct Runner Support

**Files:**
- Modify: `ptycho_torch/generators/registry.py`
- Modify: `ptycho/config/config.py`
- Modify: `ptycho_torch/config_params.py`
- Modify: `ptycho_torch/model.py`
- Modify if needed for direct-runner labeling and acceptance docs: `scripts/studies/grid_lines_torch_runner.py`

- [ ] Register `neuralop_uno` in the generator registry beside the other Torch generator architectures.
- [ ] Add `neuralop_uno` to both config-surface architecture literals so the supported architecture list is explicit in repo configs and editor/type tooling.
- [ ] Extend `_build_generator_module_from_config()` and any related checkpoint reconstruction path so saved `neuralop_uno` modules reload correctly.
- [ ] Preserve the existing supervised/PINN routing shape:
  - `Unsupervised` still builds `PtychoPINN`
  - `Supervised` still builds `Ptycho_Supervised`
  - both receive the same U-NO generator module
- [ ] Update direct-runner labels/help text if needed so `neuralop_uno` is represented cleanly in invocation metadata and per-row paper labels.
- [ ] Keep compare-wrapper model-ID mapping out of scope unless direct runner support is impossible without one narrowly scoped change; if that edge case appears, document it in the execution report before widening scope.

**Verification before moving on:**

- [ ] Registry, construction, checkpoint, and runner-acceptance tests are green.
- [ ] Existing FFNO/FNO/Hybrid/Hybrid-ResNet behavior remains untouched by the new wiring.

### Task 4: Update Durable Docs For The New Supported Surface

**Files:**
- Modify: `ptycho_torch/generators/README.md`
- Modify: `docs/workflows/pytorch.md`
- Modify: `docs/CONFIGURATION.md`
- Modify: `docs/architecture_torch.md`
- Conditional: `docs/index.md`

- [ ] Add `neuralop_uno` to the supported architecture lists where those docs describe the live CDI Torch generator surface.
- [ ] Document the intentional limits of this item:
  - external dependency on `neuraloperator`
  - current support is the locked `lines128` CDI path with `C=1`
  - `real_imag` is the only supported generator output mode for this architecture in the current repo state
- [ ] Keep compare-wrapper/table-extension docs unchanged unless the implementation truly changes those surfaces.
- [ ] Update `docs/index.md` only if implementation adds a new durable doc or materially changes how users discover the U-NO architecture support.

**Verification before moving on:**

- [ ] The docs agree with the code on architecture name, dependency, and current scope boundary.
- [ ] No doc claims that full U-NO benchmark rows or compare-wrapper model IDs are already integrated if they are not.

### Task 5: Run Focused Selectors And Mandatory Deterministic Gates

**Files / artifacts:**
- Archive logs under `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-30-cdi-lines128-uno-generator-integration/verification/`

- [ ] Run focused selectors for the new adapter/wiring/checkpoint/runner coverage while iterating.
- [ ] After the focused selectors are green, run the backlog item’s required deterministic checks exactly as written below.
- [ ] If any check fails, diagnose, patch narrowly, and rerun. Do not mark the item `BLOCKED` for ordinary verification failures.
- [ ] Do not launch any `lines128` benchmark row, append-only bundle, or compare-wrapper promotion work from this item even if all checks pass. That is follow-on work.

Suggested focused selectors before the mandatory gates:

```bash
pytest -q tests/torch/test_neuralop_uno_generator.py tests/torch/test_generator_registry.py tests/torch/test_lightning_checkpoint.py tests/torch/test_grid_lines_torch_runner.py
```

Required deterministic checks:

```bash
pytest -q tests/torch/test_generator_registry.py tests/torch/test_loss_modes.py
python -m compileall -q ptycho_torch scripts/studies
python - <<'PY'
from pathlib import Path
required = [
    Path("ptycho_torch/generators/neuralop_uno.py"),
    Path("docs/plans/NEURIPS-HYBRID-RESNET-2026/lines128_uno_table_extension_design.md"),
]
missing = [str(p) for p in required if not p.exists()]
if missing:
    raise SystemExit(f"missing U-NO integration outputs: {missing}")
print("U-NO integration outputs present")
PY
```

**Verification before completion:**

- [ ] The focused selectors are green.
- [ ] All three required deterministic checks are green and archived.
- [ ] No benchmark row was launched.
- [ ] The implementation report explicitly states whether any durable doc/index surface changed and why.
