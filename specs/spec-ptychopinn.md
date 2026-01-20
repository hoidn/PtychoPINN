# PtychoPINN Spec — Index (Normative)

This index lists the normative specification shards for PtychoPINN (TensorFlow‑based physics‑informed ptychography). The shards together form the contract that implementations SHALL satisfy.

- spec-ptycho-core.md — Core physics, math, geometry, and data contracts (inputs/outputs, formats).
- spec-ptycho-runtime.md — TensorFlow runtime guardrails (device/dtype, XLA, determinism).
- spec-ptycho-workflow.md — End‑to‑end pipeline: ingestion → grouping → normalization → model → loss → stitching → evaluation.
- spec-ptycho-interfaces.md — Public API surface and data/file interfaces with precedence rules.
- spec-ptycho-conformance.md — Acceptance tests (PTY‑AT‑XXX) and conformance profiles.
- spec-ptycho-tracing.md — Debug/tracing obligations and first‑divergence workflow.

References (informative)
- ptycho/diffsim.py, ptycho/tf_helper.py, ptycho/model.py, ptycho/raw_data.py, ptycho/loader.py, ptycho/probe.py, ptycho/fourier.py, ptycho/image/*, ptycho/train_pinn.py.

