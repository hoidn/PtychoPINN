# CDI FFNO Generator Lines Best-Config Backlog Plan

## Goal

Use FFNO as a generator in the CDI/ptycho grid-lines reconstruction path and
compare it against Hybrid ResNet on the best available lines data/training
configuration recorded in the study index.

## Motivation

The PDEBench CNS FFNO work tests FFNO as a forward-modeling baseline. This item
tests a different question: whether FFNO is useful as a reconstruction generator
inside the CDI/ptycho training stack where Hybrid ResNet is the current model of
interest.

## Scope

- Start from the study-indexed lines runbooks and best documented
  data/training configuration, especially the `lines_256` entries in
  `docs/studies/index.md`.
- Add or expose an FFNO generator option for the CDI/ptycho Torch training path.
- Compare against the best compatible Hybrid ResNet configuration under the
  same dataset, split, training budget, loss/metric contract, and
  inference/stitching path.
- Produce quantitative metrics and the standard amplitude/phase comparison
  figures.

## Boundaries

- Do not use PDEBench CNS results as evidence for CDI generator quality.
- Do not change the lines data identity or probe-scaling contract while adding
  FFNO.
- If FFNO cannot satisfy the existing generator output contract, write a blocker
  rather than changing the CDI workflow contract silently.

## Success Criteria

- A named FFNO generator profile exists for the CDI/ptycho Torch path.
- The run summary compares FFNO and Hybrid ResNet on the same best lines
  configuration.
- The result is discoverable from `docs/studies/index.md` or a linked summary.
