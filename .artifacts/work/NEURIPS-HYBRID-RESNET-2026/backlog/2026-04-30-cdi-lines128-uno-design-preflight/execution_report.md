# Execution Report

## Completed In This Pass

- created the bounded U-NO preflight helper at
  `scripts/studies/lines128_uno_preflight.py`
- added focused regression coverage in
  `tests/studies/test_lines128_uno_preflight.py`
- emitted the item-local preflight artifacts:
  `preflight_inputs.json`, `environment_probe.json`,
  `pip_show_neuraloperator.txt`, `uno_signature.json`,
  `uno_shape_probe.json`, and `preflight_decision.json`
- ran the live `ptycho311` probe and froze the verified external UNO contract:
  `neuraloperator==2.0.0`, direct `B x 2 x 128 x 128` raw output acceptance,
  `uno_out_channels=[32,64,64,32]`,
  `uno_n_modes=[[12,12],[12,12],[12,12],[12,12]]`,
  `uno_scalings=[[1.0,1.0],[0.5,0.5],[1,1],[2,2]]`,
  `channel_mlp_skip="linear"`
- published the durable summary and updated the U-NO design and NeurIPS
  evidence indexes to point at the new preflight authority

## Completed Plan Tasks

- Tranche 1: froze authorities and decision schema in
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-30-cdi-lines128-uno-design-preflight/preflight_inputs.json`
- Tranche 2: added red tests for package-missing, signature/provenance capture,
  accepted direct/lossless output layouts, scalar fallback behavior, and
  unmappable output blocking
- Tranche 3: implemented the bounded preflight helper and artifact writers
- Tranche 4: ran the live `ptycho311` import/probe and captured the accepted
  runtime package/API facts
- Tranche 5: wrote
  `docs/plans/NEURIPS-HYBRID-RESNET-2026/lines128_uno_preflight_summary.md`,
  updated the extension design, refreshed `paper_evidence_index.md`,
  `evidence_matrix.md`, `ablation_index.json`, and `docs/index.md`

## Remaining Required Plan Tasks

- none for this preflight pass
- later backlog work remains out of scope here:
  `2026-04-30-cdi-lines128-uno-generator-integration`

## Verification

- required deterministic input check:
  `python - <<'PY' ...` -> archived at
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-30-cdi-lines128-uno-design-preflight/verification/required_inputs_check.log`
- post-doc deterministic input check:
  `python - <<'PY' ...` -> archived at
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-30-cdi-lines128-uno-design-preflight/verification/required_inputs_check_post_docs.log`
- focused pytest selector:
  `pytest -v tests/studies/test_lines128_uno_preflight.py`
  -> archived green log:
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-30-cdi-lines128-uno-design-preflight/verification/pytest_green.log`
- compile gate:
  `python -m compileall -q scripts/studies`
  -> archived at
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-30-cdi-lines128-uno-design-preflight/verification/compileall.log`
- required import check in `ptycho311`:
  `python - <<'PY' import neuralop; from neuralop.models import UNO; ...`
  -> archived at
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-30-cdi-lines128-uno-design-preflight/verification/neuralop_import_check.log`
- live helper run:
  `python scripts/studies/lines128_uno_preflight.py --output-root ...`
  -> archived at
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-30-cdi-lines128-uno-design-preflight/verification/live_preflight.log`
- artifact parse/no-row-run check:
  verified `environment_probe.json`, `uno_signature.json`,
  `uno_shape_probe.json`, `preflight_decision.json`, and
  `preflight_inputs.json` parse as JSON and `runs_entries == []`

## Residual Risks

- the live preflight proves the environment/API surface only; generator
  registration, Lightning wiring, launcher routing, and actual U-NO rows remain
  unimplemented
- the accepted constructor surface differs from the draft signature defaults:
  later work must preserve the explicit `uno_out_channels`,
  nested per-layer `uno_n_modes`, `uno_scalings`, and
  `channel_mlp_skip="linear"` choices unless a new reviewed preflight revises
  them
- the live preflight log still contains non-fatal TensorFlow/XLA duplicate
  registration warnings from the environment startup path
