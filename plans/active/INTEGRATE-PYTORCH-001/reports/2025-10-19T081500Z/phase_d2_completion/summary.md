# Phase C1 Design Summary (Attempt #22)

- Authored `inference_design.md` detailing PyTorch inference/stitching flow, reuse of Lightning prediction loop, and reassembly helpers to mirror TensorFlow behaviour.
- Highlighted test harness expectations for Phase C2 (new `TestReassembleCdiImageTorchRed`), including selector and artifact path for red run.
- Documented risks around coordinate scaling, complex dtype handling, and device transfers ahead of implementation.
- Next engineer loop: execute C2 per design, capture red log at `pytest_stitch_red.log`, and update plan + ledger accordingly.

## Phase C2 Red Run Status (Attempt #23)

- `TestReassembleCdiImageTorchRed` added (tests/torch/test_workflows_components.py:1076-1410) with four contract tests covering NotImplemented guard, flip/transpose permutations, orchestration delegation, and return tuple signature.
- Targeted selector (`pytest tests/torch/test_workflows_components.py::TestReassembleCdiImageTorchRed -vv`) captured at `pytest_stitch_red.log` — 7/8 tests hit the expected `NotImplementedError`; the delegation test failed earlier on `_ensure_container` because `RawDataTorch.generate_grouped_data` rejects the `dataset_path` kwarg.
- Documented the TypeError as a pre-existing adapter bug to resolve in Phase C3 before implementing the stitching path.
