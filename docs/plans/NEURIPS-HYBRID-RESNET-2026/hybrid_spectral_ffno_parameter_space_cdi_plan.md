# CDI Hybrid-Spectral To FFNO Parameter-Space Backlog Plan

## Goal

Explore intermediate architecture points between Hybrid-spectral and FFNO on
CDI/ptycho using the best study-indexed lines configuration.

## Motivation

The former mixed CNS/CDI item combined a Phase 2 PDEBench study with Phase 3
CDI follow-up work. That made roadmap selection brittle because a single queue
file was both current-phase eligible and future-phase blocked.

This plan preserves the CDI intent as a separate Phase 3 follow-up that can run
after the CDI FFNO generator baseline exists and Phase 3 is explicitly open.

## Scope

Run this as a staged architecture study, not a Cartesian sweep.

Required axes:

- encoder/downsampling:
  - Hybrid-style downsampled shell
  - reduced or removed downsampling where feasible
- decoder:
  - canonical Hybrid decoder shell
  - FFNO-like direct projection or lighter decoder where feasible
- bottleneck:
  - Hybrid-spectral bottleneck
  - FFNO-style bottleneck
  - intermediate local-plus-factorized variants

Required domain:

- CDI/ptycho using the best study-indexed lines configuration.

## Boundaries

- Do not run PDEBench CNS rows from this item.
- Do not use CNS metrics as a proxy for CDI evidence.
- Do not start this before the CDI FFNO generator best-config item produces an
  accepted baseline or Phase 3 is explicitly opened by the roadmap.
- Keep each row attributable: change one architecture-family axis at a time
  within a staged matrix.

## Success Criteria

- The final summary distinguishes encoder/downsampling, decoder, and bottleneck
  effects on CDI/ptycho.
- Each reported row records its study-indexed lines configuration, model
  contract, and changed architecture axis.
- The study records which intermediate points are worth promoting into future
  CDI defaults or paper-facing CDI comparisons.
