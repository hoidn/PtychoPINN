# PtychoPINN Documentation Hub

This index provides a comprehensive overview of all available documentation with detailed descriptions to help you quickly find relevant information.

## Critical Gotchas ⚠️

These are the most common pitfalls that cause subtle, hard-to-debug failures. **Read these first when debugging.**

| Gotcha | Symptom | Fix | Reference |
|--------|---------|-----|-----------|
| **MODULE-SINGLETON-001** | Model expects shape (B,N,N,1) but data has (B,N,N,4) after changing gridsize | Use `create_model_with_gridsize()` factory instead of `model.autoencoder` singleton | [Troubleshooting](debugging/TROUBLESHOOTING.md#model-architecture-mismatch-after-changing-gridsize) |
| **CONFIG-001** | Shape mismatch - gridsize not synced | Call `update_legacy_dict(params.cfg, config)` before data loading | [Quick Reference](debugging/QUICK_REFERENCE_PARAMS.md) |
| **CONFIG-001 Exception** | CONFIG-001 doesn't fix model architecture | Module-level singletons are created at import time; use factory functions | [Quick Reference](debugging/QUICK_REFERENCE_PARAMS.md#️-critical-exception-module-level-singletons-module-singleton-001) |
| **ANTIPATTERN-001** | Hidden crashes from import-time side effects | Push work into functions with explicit arguments | [Developer Guide](DEVELOPER_GUIDE.md#21-anti-pattern-side-effects-on-import) |

---

## Quick Start

### [README](../README.md) - Project Overview
**Description:** High-level project introduction with installation instructions and basic usage examples.  
**Keywords:** installation, overview, setup, quickstart  
**Use this when:** First time setting up PtychoPINN or need basic installation instructions.

### [Quick Reference: Parameters](debugging/QUICK_REFERENCE_PARAMS.md) CRITICAL
**Description:** Essential cheatsheet for params.cfg initialization - covers the critical `update_legacy_dict()` call required before data operations and debugging shape mismatch errors.  
**Keywords:** params.cfg, initialization, gridsize, shape-mismatch, troubleshooting  
**Use this when:** Getting shape mismatch errors, debugging configuration issues, or need to understand params.cfg initialization pattern.

### [Workflow Guide](WORKFLOW_GUIDE.md)
**Description:** Comprehensive guide covering complete PtychoPINN workflows from training through evaluation and model comparison, including common patterns and troubleshooting.  
**Keywords:** workflow, training, evaluation, model-comparison, troubleshooting  
**Use this when:** Starting a new project, planning experiments, or need to understand the complete train→evaluate→compare workflow.

### [Model Comparison Guide](MODEL_COMPARISON_GUIDE.md)
**Description:** How to run 2-way/3-way comparisons, pick entry points (direct compare vs wrappers vs studies), reuse existing runs, plug in PtyChi/Tike reconstructions, apply subsampling consistently, and handle fixed-canvas alignment.  
**Keywords:** comparison, studies, ptychi, subsampling, artifacts, alignment  
**Use this when:** You need to compare models, reuse existing outputs, or decide which comparison script to run.

### [PyTorch Workflow Guide](workflows/pytorch.md)
**Description:** End-to-end instructions for configuring, training, and running inference with the PyTorch implementation, highlighting differences from the TensorFlow pipelines and reusing the shared data contracts.  
**Highlights:** Includes a synthetic grid‑lines Hybrid ResNet reference command (N=128, set‑phi) under “Common Workflows.”  
**Keywords:** pytorch, lightning, mlflow, configuration, training  
**Use this when:** Working on the `ptycho_torch/` stack or porting TensorFlow workflows to PyTorch.

### [Model Baselines](model_baselines.md)
**Description:** Canonical recommended training baselines for model families such as `hybrid_resnet`, including the distinction between project-recommended baselines, raw configuration defaults, and study-specific overrides.
**Keywords:** baselines, hybrid_resnet, recommended-params, scheduler, learning-rate, studies
**Use this when:** You need the current recommended starting parameters for a real study or wrapper and do not want to infer "best practice" from defaults, prompts, or tests.

### [NeurIPS Hybrid ResNet Submission Design](plans/2026-04-20-neurips-hybrid-resnet-submission-design.md)
**Description:** Approved design brief for a NeurIPS 2026 Hybrid ResNet submission campaign, defining the CDI pillar, the amended native `128x128` PDEBench image-suite pillar, fresh `128x128` CDI anchor regeneration strategy, `256x256` higher-mode scaling hypothesis, and the planned `/home/ollie/Documents/neurips/index.md` evidence-map policy. That local manuscript index is intentionally not required to exist until the roadmap evidence-bundle phase creates it.
**Keywords:** neurips, hybrid_resnet, roadmap, submission, cdi, pde, evidence
**Use this when:** Planning or executing the Hybrid ResNet NeurIPS submission work, deciding which CDI/PDE evidence is in scope, or assembling paper-facing artifacts.

### [NeurIPS Hybrid ResNet Submission Roadmap](plans/2026-04-20-neurips-hybrid-resnet-submission-roadmap.md)
**Description:** Phase-by-phase roadmap for the NeurIPS 2026 Hybrid ResNet campaign, prioritizing evidence inventory with explicit lost-run handling, required PDEBench `128x128` image-suite execution, CNS paper-contract packaging, fresh `128x128` CDI anchor regeneration/verification, complete `lines128` benchmark packaging, optional `256x256` scaling evidence, and later evidence-bundle assembly.
**Keywords:** neurips, hybrid_resnet, roadmap, phases, pde, cdi, cns, lines128, paper-evidence, artifact-index
**Use this when:** You need the execution order, gates, expected artifacts, fallback decisions, or paper-evidence package routing for the submission campaign.

### [NeurIPS Steered Backlog Drain Workflow Design](plans/2026-04-22-neurips-steered-backlog-drain-workflow-design.md)
**Description:** Design for a steering-aware backlog drain workflow that selects the next NeurIPS backlog item from strategic intent, roadmap state, and repo progress; moves it into `docs/backlog/in_progress/`; optionally syncs the roadmap; always drafts a fresh plan; then runs implementation/review before repeating.
**Keywords:** neurips, workflow, backlog, steering, roadmap, selection, planning, implementation
**Use this when:** Designing or implementing the adaptive NeurIPS backlog workflow, deciding what should stay repo-local versus reusable, or clarifying the queue, selector, roadmap-sync, and fresh-plan contracts.

### [NeurIPS Steered Backlog Drain Workflow Implementation Plan](plans/2026-04-22-neurips-steered-backlog-drain-workflow-implementation-plan.md)
**Description:** Concrete implementation plan for the NeurIPS steering-aware backlog drain workflow, covering the steering document contract, backlog `in_progress` queue state, deterministic manifest and queue helpers, adaptive selector, narrow roadmap sync, fresh-plan rewrite behavior, top-level drain workflow, and required tests/smoke checks.
**Keywords:** neurips, workflow, backlog, steering, roadmap, implementation-plan, selection, smoke-test
**Use this when:** Implementing the adaptive backlog workflow, checking which files and prompts belong to each phase, or verifying the required helper tests, runtime smoke, and orchestrator dry-run coverage.

### [NeurIPS Backlog Implementation-State Contract Plan](plans/2026-04-23-neurips-backlog-implementation-state-contract-plan.md)
**Description:** Follow-on implementation plan for fixing the backlog drain's execution-state model so long-running selected items can end in `WAITING` instead of being incorrectly marked `BLOCKED`, with explicit `COMPLETED | RUNNING | BLOCKED` semantics, separate progress versus final report surfaces, and workflow routing changes across the implementation phase, selected-item workflow, and top-level drain.
**Keywords:** neurips, workflow, backlog, implementation, waiting, blocked, running, execution-state
**Use this when:** Fixing the selected-item implementation-state bug, updating the drain to support non-terminal experiment progress, or reviewing how to separate semantic item state from workflow failure.

### [NeurIPS Backlog Roadmap Gap Fix Plan](plans/2026-04-28-neurips-backlog-roadmap-gap-fix-plan.md)
**Description:** Implementation plan for hardening the NeurIPS backlog drain so deterministic roadmap gating happens before provider selection, missing authorized Phase 2 work can be drafted as a controlled backlog-gap item, and active items move to `in_progress` only after roadmap sync accepts them.
**Keywords:** neurips, workflow, backlog, roadmap-gate, gap-draft, selector, in-progress
**Use this when:** Fixing or reviewing backlog drain phase-routing behavior, especially when active items are out of phase with the current roadmap gate.

### [NeurIPS Steering Document](steering.md)
**Description:** Human-authored strategic intent for the NeurIPS Hybrid ResNet backlog drain, capturing ordered priorities, comparison standards, fairness constraints, known blockers, and non-goals.
**Keywords:** neurips, steering, priorities, fairness, blockers, workflow
**Use this when:** Selecting the next backlog item, reviewing whether a plan stayed within current paper priorities, or deciding whether roadmap drift is real.

### [NeurIPS Backlog Dependency Index](backlog/index.md)
**Description:** Human-readable dependency map for the current NeurIPS backlog queue, recording which items are parallel, which roadmap phase they belong to, and when a real backlog-to-backlog prerequisite should be added to frontmatter.
**Keywords:** neurips, backlog, dependencies, roadmap, queue, selection
**Use this when:** Deciding whether a backlog item is blocked by another item, checking which CNS studies are parallel versus serial, or reviewing selector behavior against the current queue graph.

### [NeurIPS Steered Backlog Drain Runbook](workflows/neurips_steered_backlog_drain.md)
**Description:** Operator runbook for the steering-aware NeurIPS backlog drain workflow, including queue lifecycle, authoritative state roots, and launch/resume commands from the `ptycho311` environment.
**Keywords:** neurips, workflow, runbook, backlog, resume, ptycho311
**Use this when:** Launching or resuming the backlog drain, checking the queue-state lifecycle, or confirming which documents and state files the workflow consumes.

### [NeurIPS Hybrid ResNet Phase 0 Evidence Inventory](plans/NEURIPS-HYBRID-RESNET-2026/evidence_inventory.md)
**Description:** Durable Phase 0 inventory for the NeurIPS Hybrid ResNet campaign, separating paper-grade, decision-support, and not-usable CDI artifacts; recording N=256 as secondary scaling context; and listing neutral PDE candidates for Phase 1.
**Keywords:** neurips, hybrid_resnet, evidence-inventory, cdi, pde, n256, provenance
**Use this when:** Checking what CDI/PDE evidence was found before launching Phase 1 benchmark selection or Phase 3 CDI regeneration.

### [NeurIPS Hybrid ResNet CDI Anchor Regeneration Plan](plans/NEURIPS-HYBRID-RESNET-2026/cdi_anchor_regeneration_plan.md)
**Description:** Phase 0 regeneration note for the missing paper-grade `128x128` grid-lines Hybrid ResNet anchor, including baseline settings, wrapper/runner command source, provenance capture, metric contract, qualitative output plan, and runtime guardrails.
**Keywords:** neurips, hybrid_resnet, cdi, regeneration, 128x128, grid-lines, provenance
**Use this when:** Preparing the Roadmap Phase 3 fresh CDI anchor run after confirming no complete paper-grade historical anchor was recovered.

### [NeurIPS Hybrid ResNet Paper Evidence Package Design](plans/NEURIPS-HYBRID-RESNET-2026/paper_evidence_package_design.md)
**Description:** Package-level design for the paper evidence needed before result claims are drafted, tying together CDI `lines128`, PDEBench CNS, provenance, numeric tables, visual comparison bundles, claim boundaries, and backlog decomposition.
**Keywords:** neurips, hybrid_resnet, paper, evidence, cdi, cns, tables, figures, provenance
**Use this when:** Deciding what evidence must exist before drafting result claims, splitting paper-evidence work into backlog items, or checking paper-grade versus decision-support boundaries across CDI and CNS.

### [NeurIPS Hybrid ResNet Paper Evidence Index](plans/NEURIPS-HYBRID-RESNET-2026/paper_evidence_index.md)
**Description:** Durable outcome map for completed NeurIPS backlog items, linking each completed CNS/CDI evidence item to its summary authority, artifact root, evidence tier, protocol/cap, and downstream paper use.
**Keywords:** neurips, hybrid_resnet, paper, evidence-index, completed-backlog, outcomes, artifacts, claim-boundary
**Use this when:** Locating completed backlog results, checking whether an artifact is paper-facing or decision-support only, finding table/figure bundle roots, or auditing which summary owns a claim.

### [NeurIPS Born/Rytov Diffraction Tomography Candidate Lane Design](plans/NEURIPS-HYBRID-RESNET-2026/born_rytov_dt_candidate_lane_design.md)
**Description:** Active candidate additional inverse-scattering lane for the NeurIPS SRU-Net manuscript, defining the Born/Rytov physical model, dataset contract, operator validation gates, dependencies, environment requirements, and a limited four-row preflight. This is concurrent candidate work on equal footing with WaveBench inverse source; it does not replace CDI `lines128` or PDEBench CNS and cannot support paper-table claims without a later evidence-package amendment.
**Keywords:** neurips, brdt, born, rytov, diffraction-tomography, inverse-scattering, candidate, sru-net, physics
**Use this when:** Planning or reviewing the BRDT candidate preflight, checking physical target and normalization conventions, or deciding whether a cheap-forward inverse-scattering lane is ready for a later roadmap/evidence amendment.

### [NeurIPS Inverse-Wave Benchmark Rationale](plans/NEURIPS-HYBRID-RESNET-2026/inverse_wave_benchmark_rationale.md)
**Description:** Decision rationale comparing BRDT, WaveBench inverse source, OpenFWI, OpenSWI, Fourier ptychography, holography, and diffraction tomography as possible additional inverse-wave lanes. It records why BRDT and WaveBench are both active candidate preflights and why neither candidate can replace CDI `lines128` or PDEBench CNS.
**Keywords:** neurips, inverse-wave, brdt, wavebench, openfwi, openswi, benchmark-rationale, candidate
**Use this when:** Deciding how optional inverse-wave candidate work should be routed, reviewing why BRDT and WaveBench are on equal footing, or preventing optional benchmark work from being confused with the required CDI/CNS evidence package.

### [NeurIPS WaveBench Inverse Source Benchmark Design](plans/NEURIPS-HYBRID-RESNET-2026/wavebench_inverse_source_benchmark_design.md)
**Description:** Active candidate design for WaveBench inverse source reconstruction as a possible inverse-wave evidence lane, including supervised and physics-informed variants, dataset/forward-model validation requirements, baseline compatibility, and shared-encoder cautions. This is concurrent candidate work on equal footing with BRDT.
**Keywords:** neurips, wavebench, inverse-source, wave-equation, fno, unet, candidate
**Use this when:** Planning or reviewing the WaveBench candidate preflight, checking what must be verified before physics-informed rows are credible, or distinguishing WaveBench inverse source from OpenFWI and BRDT.

### [NeurIPS Hybrid ResNet PDEBench CNS Paper Contract Decision](plans/NEURIPS-HYBRID-RESNET-2026/pdebench_cns_paper_contract_decision.md)
**Description:** Durable decision that fixes the authoritative CNS paper lane to the bounded capped `history_len=2` contract, records the exact normalization and training-recipe bindings, names the exact headline row roster and authored-FFNO cutoff/status, rejects immediate full-training on compute/deadline grounds, and keeps `history_len=3` as adjacent capped context only.
**Keywords:** neurips, hybrid_resnet, pdebench, cns, paper-contract, capped, history_len, ffno, claim-boundary
**Use this when:** Locking the CNS headline row contract, deciding whether a later item may reuse existing capped CNS rows, checking the exact normalization or training recipe wording that downstream items must repeat, or checking why the current paper lane is not a full-training benchmark claim.

### [NeurIPS Hybrid ResNet PDEBench CNS Paper Row Lock Summary](plans/NEURIPS-HYBRID-RESNET-2026/pdebench_cns_paper_row_lock_summary.md)
**Description:** Durable summary for the locked bounded capped CNS paper-row bundle, naming the accepted headline and continuity rows, pointing to the machine-readable row manifest and audit, recording the preserved provenance gaps on the reused roots, and fixing the downstream claim boundary to `capped_decision_support`.
**Keywords:** neurips, hybrid_resnet, pdebench, cns, row-lock, capped, manifest, ffno, provenance
**Use this when:** Building the downstream CNS table/figure bundle, checking which run roots are the locked bounded rows, or confirming why the reused rows are acceptable for capped assembly but not yet paper-grade provenance-complete evidence.

### [PDEBench CNS Paper Table/Figure Bundle Summary](plans/NEURIPS-HYBRID-RESNET-2026/pdebench_cns_paper_table_figure_bundle_summary.md)
**Description:** Durable summary for the assembled bounded CNS paper table/figure bundle, including the `1024` same-cap audit outcome, the final `512 / 64 / 64` fallback lock decision, the fixed sample/shared-scale policy, the emitted JSON/CSV/TeX plus figure artifacts, and the preserved provenance boundary that still blocks `paper_grade`.
**Keywords:** neurips, hybrid_resnet, pdebench, cns, table, figure, bundle, fallback, scales
**Use this when:** Locating the assembled CNS paper bundle artifacts, checking whether the bundle used the `1024` or `512` lane, or confirming the exact fixed-sample and claim-boundary policy.

### [NeurIPS Lines128 Paper-Quality CDI Benchmark Design](plans/NEURIPS-HYBRID-RESNET-2026/lines128_paper_benchmark_design.md)
**Description:** Detailed design for a paper-quality `N=128` grid-lines CDI benchmark, including the fixed `lines128` contract, required Hybrid/FNO/FFNO rows, metric schema, visual comparison artifacts, provenance requirements, and benchmark execution handoff.
**Keywords:** neurips, cdi, lines128, grid-lines, hybrid_resnet, fno, ffno, benchmark, paper-grade
**Use this when:** Planning or reviewing the CDI `lines128` benchmark rows, preflight contract reconstruction, FFNO generator integration, or paper-grade CDI table and figure generation.

### [NeurIPS Lines128 Paper Benchmark Preflight](plans/NEURIPS-HYBRID-RESNET-2026/lines128_paper_benchmark_preflight.md)
**Description:** Recovery and launch-safety preflight for the fixed-contract `N=128` CDI `ffno` versus `hybrid_resnet` row pair, capturing the recovered contract sources, stable-root audit, active-writer check, exact command, and the no-duplicate-run resume decision.
**Keywords:** neurips, cdi, lines128, preflight, ffno, hybrid_resnet, grid-lines, contract, resume
**Use this when:** Checking whether the active Lines128 FFNO-versus-Hybrid compare should resume or relaunch, or auditing which historical and live artifacts fixed the contract.

### [NeurIPS Lines128 Paper Benchmark Harness Preflight](plans/NEURIPS-HYBRID-RESNET-2026/lines128_paper_benchmark_harness_preflight.md)
**Description:** Contract freeze and decision note for the later `lines128` paper benchmark harness, naming the minimum row roster, fixing `fno_vanilla` as the FNO comparator, recording seed policy, and separating readiness-only harness work from the later full benchmark launch.
**Keywords:** neurips, cdi, lines128, harness, preflight, hybrid_resnet, fno_vanilla, ffno, spectral
**Use this when:** Checking which rows are in the minimum harness subset, what the selected FNO comparator is, whether spectral/FFNO are supported for the harness, or whether the current item authorizes a full paper benchmark launch.

### [NeurIPS Lines128 Minimum Paper Table Execution Authority](plans/NEURIPS-HYBRID-RESNET-2026/lines128_minimum_paper_table_execution_authority.md)
**Description:** Checked-in launch authority for the minimum draftable `lines128` CDI subset, preserving the frozen contract/comparator/seed from the harness preflight while authorizing exactly four rows: supervised CDI `cnn`, PINN CDI `cnn`, `pinn_hybrid_resnet`, and `pinn_fno_vanilla`.
**Keywords:** neurips, cdi, lines128, execution-authority, minimum-subset, cnn, hybrid_resnet, fno_vanilla
**Use this when:** Verifying whether the minimum draftable CDI subset may launch, auditing the exact four-row roster and training procedures, or checking which later rows remain out of scope for the complete table.

### [NeurIPS Lines128 Minimum Paper Table Summary](plans/NEURIPS-HYBRID-RESNET-2026/lines128_minimum_paper_table_summary.md)
**Description:** Durable completion summary for the minimum draftable `lines128` CDI subset, recording the chosen fresh-rerun plus same-root bundle-regeneration path, the authoritative `paper_complete` four-row bundle, archived verification evidence, exact row roster, and the preserved boundary between this minimum subset and the later complete `lines128` table.
**Keywords:** neurips, cdi, lines128, minimum-subset, paper-complete, bundle-regeneration, hybrid_resnet, fno_vanilla
**Use this when:** Locating the finished minimum CDI paper-table artifacts, checking which repaired fresh-rerun root was promoted to the authoritative four-row bundle, or confirming what remains out of scope for the later complete `lines128` benchmark.

### [NeurIPS Lines128 Paper Benchmark Harness Summary](plans/NEURIPS-HYBRID-RESNET-2026/lines128_paper_benchmark_harness_summary.md)
**Description:** Implementation summary for the readiness-only `lines128` paper benchmark harness pass, covering the thin harness entry point, spectral-row wrapper routing, paper-schema validation artifacts, and the explicit boundary that the full benchmark remains unlaunched.
**Keywords:** neurips, cdi, lines128, harness, summary, benchmark_incomplete, spectral, ffno
**Use this when:** Auditing what the harness pass completed, locating the readiness-only validation artifacts, or confirming why the merged validation result is intentionally incomplete.

### [NeurIPS Lines128 Complete Paper Benchmark Summary](plans/NEURIPS-HYBRID-RESNET-2026/lines128_paper_benchmark_summary.md)
**Description:** Durable completion summary for the authoritative six-row `lines128` CDI paper bundle, recording the initial spectral-rerun root, the corrected promoted-row launcher-proof recovery path, the final tmux-backed `paper_complete` authoritative root, archived verification logs, and the preserved distinction between prerequisite FFNO evidence, minimum-subset evidence, and the final complete-table bundle.
**Keywords:** neurips, cdi, lines128, paper-complete, ffno, spectral, benchmark, summary
**Use this when:** Locating the finished six-row CDI paper bundle, checking which repaired tmux-backed root is authoritative, or verifying how the spectral rerun and promoted FFNO/minimum-subset rows were assembled into the final claim-bearing package.

### [NeurIPS Hybrid ResNet CDI FFNO Generator Lines Best-Config Summary](plans/NEURIPS-HYBRID-RESNET-2026/cdi_ffno_generator_lines_best_config_summary.md)
**Description:** Execution summary for the fixed-contract `N=128` CDI `ffno` versus `hybrid_resnet` row pair, pinning the recovered `lines128` contract, stable artifact root, exact compare command, and the claim boundary that this pair is prerequisite evidence for the later paper benchmark rather than the full four-row package.
**Keywords:** neurips, cdi, ffno, hybrid_resnet, lines128, grid-lines, compare, summary
**Use this when:** Checking the exact FFNO-versus-Hybrid CDI compare contract, locating the fresh artifact root, or distinguishing this prerequisite row pair from the later paper-quality benchmark harness.

### [NeurIPS Hybrid ResNet PDE Benchmark Selection](plans/NEURIPS-HYBRID-RESNET-2026/pde_benchmark_selection.md)
**Description:** Roadmap Phase 1 benchmark scorecard and primary/fallback provenance for the required PDE pillar, originally selecting PDEBench SWE as primary and OpenFWI FlatVel-A as fallback, then amended to a compact native `128x128` PDEBench image suite covering SWE, Darcy Flow, and 2D Compressible Navier-Stokes.
**Keywords:** neurips, hybrid_resnet, pde, benchmark-selection, pdebench, openfwi, phase-1, darcy, cns, 2d-cfd
**Use this when:** Checking why PDEBench, OpenFWI, and PDEArena Maxwell-3D were selected or rejected, or why the current Phase 2 target is the PDEBench `128x128` image suite.

### [NeurIPS Hybrid ResNet PDEBench 128x128 Image-Suite Plan](plans/NEURIPS-HYBRID-RESNET-2026/pdebench_128x128_image_suite_plan.md)
**Description:** Roadmap Phase 2 plan for a compact native `128x128` PDEBench image suite covering SWE (`swe`), Darcy Flow (`darcy`), and 2D Compressible Navier-Stokes (`2d_cfd_cns`), including data/schema preflight, shared adapter reuse from the existing SWE study, smoke-readiness boundaries, capped-pilot limitations, full available training-split benchmark requirements, ablations, and suite-summary gates.
**Keywords:** neurips, hybrid_resnet, pdebench, 128x128, swe, darcy, cns, 2d-cfd, image-suite, phase-2
**Use this when:** Drafting or reviewing the next Phase 2 implementation tranche, deciding how to generalize the existing SWE study components, or checking why smoke runs cannot be used as model-performance evidence.

### [NeurIPS Hybrid ResNet PDEBench 128x128 Image-Suite Preflight](plans/NEURIPS-HYBRID-RESNET-2026/pdebench_128x128_image_suite_preflight.md)
**Description:** Stage A execution summary for the PDEBench `128x128` image-suite plan, recording SWE and Darcy schema/readiness plus the active CNS missing-file/staging state until the official `128x128` 2D_CFD file is fully downloaded, checksum-verified, and inspected.
**Keywords:** neurips, hybrid_resnet, pdebench, 128x128, preflight, swe, darcy, cns, 2d-cfd, data-schema
**Use this when:** Checking staged SWE/Darcy data contracts, auditing CNS data staging, or confirming whether suite-level benchmark claims are still blocked by the CNS schema gate.

### [NeurIPS Hybrid ResNet PDEBench 2D Compressible Navier-Stokes Design](plans/NEURIPS-HYBRID-RESNET-2026/pdebench_2d_cfd_cns_design.md)
**Description:** Draft design for replacing the 2D diffusion-reaction suite member with an official `128x128` PDEBench 2D Compressible Navier-Stokes target, including direct-file feasibility, storage gates, selected CNS file candidates, four-field history-window contract, normalization, fRMSE high-frequency shock-capture diagnostics, and benchmark claim boundaries.
**Keywords:** neurips, hybrid_resnet, pdebench, 2d_cfd, cns, compressible-navier-stokes, history-window, frmse, shock-capture, 128x128
**Use this when:** Planning or reviewing the 2D CNS adapter, deciding which official `128x128` PDEBench 2D_CFD file to stage, or checking why the full 551 GB `2d_cfd` family is out of local-root scope.

### [NeurIPS Hybrid ResNet PDEBench 2D Compressible Navier-Stokes Physics Regularization Design](plans/NEURIPS-HYBRID-RESNET-2026/pdebench_2d_cfd_cns_physics_regularization_design.md)
**Description:** Approved design for a reusable PDEBench image-suite physics-loss framework with a first `2d_cfd_cns` backend, defining the generic interface, default-off behavior, fail-closed unsupported-task contract, and the v1 term bundle of positivity, continuity residual, and global mass.
**Keywords:** neurips, hybrid_resnet, pdebench, 2d_cfd, cns, physics-regularization, continuity, mass, positivity
**Use this when:** Planning or reviewing CNS physics-loss work, deciding what terms belong in the first physics-regularized training path, or checking the reusable-vs-task-local boundary before implementation.

### [NeurIPS Hybrid ResNet PDEBench 2D Compressible Navier-Stokes Summary](plans/NEURIPS-HYBRID-RESNET-2026/pdebench_2d_cfd_cns_summary.md)
**Description:** Roadmap Phase 2 implementation summary for the PDEBench `2d_cfd_cns` adapter, including the selected official `128x128` CNS file, field/history contract, verification evidence, the canonical `hybrid_resnet_cns` skip-add-plus-pixelshuffle profile decision, readiness-smoke follow-up, the implemented default-off CNS physics-regularization framework, and the later pointers to the separate CNS paper-contract and row-lock authorities.
**Keywords:** neurips, hybrid_resnet, pdebench, 2d_cfd, cns, implementation-summary, data-gate, readiness, physics-regularization, paper-contract
**Use this when:** Checking what CNS support is implemented, whether the official HDF5 has been checksum-verified, how the default-off physics-loss framework is wired, or where the later headline paper-contract and row-lock decisions are recorded.

### [Hybrid Upsampler Artifact Study Results](plans/2026-04-21-hybrid-upsampler-artifact-study-results.md)
**Description:** Results summary for the post-skip-add PDEBench `2d_cfd_cns` upsampler rerun, comparing transpose, bilinear-conv, and pixelshuffle decoders under the same canonical CNS shell and recording the promotion of `pixelshuffle` into the default `hybrid_resnet_cns` profile.
**Keywords:** neurips, hybrid_resnet, pdebench, 2d_cfd, cns, upsampler, pixelshuffle, bilinear, transpose, summary
**Use this when:** Reviewing the capped same-shell CNS upsampler compare, locating the rendered galleries, or checking why the canonical CNS Hybrid row now defaults to `pixelshuffle`.

### [NeurIPS Hybrid ResNet PDEBench Spectral ResNet Bottleneck Variant Design](plans/NEURIPS-HYBRID-RESNET-2026/pdebench_spectral_resnet_bottleneck_design.md)
**Description:** Draft design for a bottleneck-only PDEBench image-suite variant that keeps the current ResNet `3x3` bottleneck body, adds a shared factorized spectral residual branch, keeps the current encoder/downsample/upsample shell, and exposes the result as a separate `spectral_resnet_bottleneck_net` model family rather than under `hybrid_resnet`.
**Keywords:** neurips, hybrid_resnet, pdebench, spectral-resnet, bottleneck, factorized-fourier, cns, darcy, architecture-design
**Use this when:** Planning a shared-spectral bottleneck experiment for the PDEBench image suite, deciding what factorized spectral pieces to import into the current supervised adapter, or checking why the variant is intentionally named outside the `hybrid_resnet_*` profile family.

### [NeurIPS Hybrid ResNet PDEBench FFNO-Close Bottleneck Variant Design](plans/NEURIPS-HYBRID-RESNET-2026/pdebench_ffno_bottleneck_design.md)
**Description:** Draft design for a bottleneck-only PDEBench image-suite variant that introduces an FFNO-close spectral-plus-feedforward bottleneck family, keeps the current canonical CNS skip-add shell fixed across all rows, and defines a fair three-way bottleneck comparison against the local Hybrid bottleneck and the existing spectral-ResNet bottleneck.
**Keywords:** neurips, hybrid_resnet, pdebench, ffno, bottleneck, cns, fairness, skip-add, architecture-design
**Use this when:** Planning an FFNO-close bottleneck experiment for the PDEBench CNS suite, deciding how to keep the shell fixed while swapping only the bottleneck, or checking what counts as a fair comparison against `hybrid_resnet_cns` and `spectral_resnet_bottleneck_base`.

### [NeurIPS Hybrid ResNet PDEBench FFNO-Close Bottleneck Compare Summary](plans/NEURIPS-HYBRID-RESNET-2026/pdebench_ffno_bottleneck_summary.md)
**Description:** Implementation summary for the FFNO-close bottleneck tranche, covering the new `ffno_bottleneck_net` family, the shared canonical CNS skip-add shell contract, targeted verification, the capped three-row CNS comparison, and gallery artifact paths.
**Keywords:** neurips, hybrid_resnet, pdebench, ffno, bottleneck, cns, summary, capped-compare, spectral-resnet
**Use this when:** Checking what was implemented in the FFNO-close bottleneck tranche, finding the capped same-shell CNS comparison artifacts, or seeing how the local, spectral, and FFNO-close bottlenecks compared on the first controlled run.

### [NeurIPS Hybrid ResNet PDEBench Author FFNO Equal-Footing Plan](plans/NEURIPS-HYBRID-RESNET-2026/pdebench_author_ffno_equal_footing_plan.md)
**Description:** Backlog plan for running the authors' actual FFNO model on the same capped PDEBench `2d_cfd_cns` slice and epoch budgets already used for the local spectral, FNO, and U-Net rows, with the goal of a fair equal-footing comparison.
**Keywords:** neurips, pdebench, ffno, author-model, cns, equal-footing, backlog, compare
**Use this when:** Queueing or implementing the official-author FFNO baseline on the local CNS contract, or checking what must stay fixed for a fair comparison against `spectral_resnet_bottleneck_base`, `fno_base`, and `unet_strong`.

### [NeurIPS Hybrid ResNet PDEBench Author FFNO Equal-Footing Summary](plans/NEURIPS-HYBRID-RESNET-2026/pdebench_author_ffno_equal_footing_summary.md)
**Description:** Implementation summary for the official-author FFNO CNS equal-footing backlog item, covering the pinned `fourierflow` source, local host-environment provenance, adapter/profile integration, smoke-gate fit check, frozen reused local reference rows, fresh authored FFNO capped runs, and merged cross-run compare artifacts kept separate from the local FFNO-close bottleneck proxy experiment.
**Keywords:** neurips, pdebench, ffno, author-model, cns, equal-footing, summary, fourierflow
**Use this when:** Checking how the authored FFNO row was hosted locally, finding the fresh author-run and merged compare artifacts, or distinguishing the real author baseline from the local `ffno_bottleneck_base` proxy path.

### [NeurIPS Hybrid ResNet PDEBench CNS Markov History-1 Compare Design](plans/NEURIPS-HYBRID-RESNET-2026/pdebench_cns_markov_history1_compare_design.md)
**Description:** Design for a controlled PDEBench `2d_cfd_cns` ablation that switches the supervised contract from `history_len=2` to Markov-style `history_len=1` and reruns the local spectral, Hybrid, FNO, and U-Net rows on the same capped slice and epoch budgets.
**Keywords:** neurips, pdebench, cns, markov, history_len, history-1, spectral_resnet, fno, unet, compare
**Use this when:** Planning or executing the lower-context Markov-style CNS compare, or checking what must stay fixed for a fair `history_len=1` versus `history_len=2` result.

### [NeurIPS Hybrid ResNet PDEBench CNS Markov History-1 Compare Summary](plans/NEURIPS-HYBRID-RESNET-2026/pdebench_cns_markov_history1_compare_summary.md)
**Description:** Implementation summary for the controlled PDEBench `2d_cfd_cns` Markov-style history-length ablation, covering the frozen `history_len=2` reference manifest, the missing `40`-epoch `hybrid_resnet_cns` backfill, the fresh capped `history_len=1` pilot runs at `10` and `40` epochs, and the cross-history compare sidecars.
**Keywords:** neurips, pdebench, cns, markov, history_len, history-1, spectral_resnet, hybrid_resnet_cns, fno, unet, summary
**Use this when:** Checking whether reducing CNS temporal context helped the spectral row, whether the capped four-row ranking changed, or locating the audited reference manifest and cross-history compare artifacts.

### [NeurIPS Hybrid ResNet PDEBench CNS History Length 3+ Compare Summary](plans/NEURIPS-HYBRID-RESNET-2026/pdebench_cns_history_len3plus_compare_summary.md)
**Description:** Implementation summary for the controlled longer-context PDEBench `2d_cfd_cns` ablation, covering the frozen `history_len=2` reference manifest, the mandatory capped `history_len=3` inspect plus `10`/`40`-epoch pilot runs, the anchored cross-history compare sidecars, and the closed `history_len=4` gate after mixed `10`/`40`-epoch spectral signals.
**Keywords:** neurips, pdebench, cns, history_len, history-3, history-4, spectral_resnet, hybrid_resnet_cns, fno, unet, summary
**Use this when:** Checking whether increasing CNS temporal context to `history_len=3` helped on the capped four-row contract, whether the ranking changed at `10` or `40` epochs, or why the optional `history_len=4` branch stayed closed.

### [NeurIPS Hybrid ResNet PDEBench Hybrid-Spectral CNS Architecture Ablation Design](plans/NEURIPS-HYBRID-RESNET-2026/pdebench_hybrid_spectral_cns_architecture_ablation_design.md)
**Description:** Design for a CNS-only Hybrid-spectral architecture ablation that fixes the canonical skip-add plus pixelshuffle shell and studies spectral-family internals such as weight sharing and bottleneck depth without mixing in CDI/ptycho, Markov-history, or physics-regularization changes.
**Keywords:** neurips, pdebench, cns, hybrid-spectral, architecture, ablation, weight-sharing, depth, pixelshuffle, skip-add
**Use this when:** Planning or reviewing a focused CNS Hybrid-spectral architecture ablation, checking which shell choices are fixed, or separating CNS spectral-family questions from CDI/ptycho architecture work.

### [NeurIPS Hybrid ResNet PDEBench CNS Spectral Modes-32 Compare Plan](plans/NEURIPS-HYBRID-RESNET-2026/pdebench_spectral_modes32_compare_plan.md)
**Description:** Backlog plan for a focused capped CNS ablation that raises both the encoder spectral modes and the spectral bottleneck modes from `12` to `32` for the spectral variant and checks whether that improves metrics over the current `12/12` row.
**Keywords:** neurips, pdebench, cns, spectral_resnet, modes, fno_modes, spectral_bottleneck_modes, compare
**Use this when:** Queueing or implementing the higher-mode spectral CNS ablation, or checking which spectral-mode knobs must move together for a fair result.

### [NeurIPS Hybrid ResNet PDEBench CNS Spectral Modes-32 Compare Summary](plans/NEURIPS-HYBRID-RESNET-2026/pdebench_spectral_modes32_compare_summary.md)
**Description:** Implementation summary for the capped modes-32 CNS backlog item, covering the reused fresh `10`-epoch row, the authoritative fresh `40`-epoch row, the anchored `10`/`40`-epoch sidecars, and the result that modes-32 helped on the `10`-epoch slice but at `40` epochs improved only `fRMSE_mid/high` while losing aggregate and low-band error to the shared `12/12` spectral row.
**Keywords:** neurips, pdebench, cns, spectral_resnet, modes32, fno_modes, spectral_bottleneck_modes, summary
**Use this when:** Checking whether the higher-mode spectral row should replace the shared `12/12` capped CNS reference, locating the fresh modes-32 artifacts, or reviewing the exact fixed-contract compare boundary.

### [NeurIPS Hybrid ResNet PDEBench CNS Spectral Modes-24 Convergence Summary](plans/NEURIPS-HYBRID-RESNET-2026/pdebench_spectral_modes24_convergence_summary.md)
**Description:** Implementation summary for the capped modes-24 CNS convergence backlog item, covering the fixed-contract inspect proof, the shared batch-size confirmation at `16`, the fresh paired `80`-epoch shared spectral `12/12` versus `24/24` run, the emitted convergence audit, and the inconclusive result that both rows were still materially improving at stop time while the shared base row kept the better final eval metrics.
**Keywords:** neurips, pdebench, cns, spectral_resnet, modes24, convergence, fno_modes, spectral_bottleneck_modes, summary
**Use this when:** Checking whether the shared spectral `24/24` row looks worthwhile once obvious under-convergence is reduced, locating the convergence-audit payload and resolved batch-size record, or reviewing why this capped `80`-epoch follow-up still does not justify a default-profile promotion.

### [NeurIPS Hybrid ResNet PDEBench CNS Shared-Blocks10 1024-Cap Longer-Convergence Summary](plans/NEURIPS-HYBRID-RESNET-2026/pdebench_cns_shared_blocks10_1024cap_longer_convergence_summary.md)
**Description:** Implementation summary for the capped shared-blocks10 longer-convergence backlog item, covering the frozen `40`-epoch `1024cap` reference manifest, exact `80`-epoch inspect proof, fresh shared-blocks10 rerun, emitted convergence audit, shell-validated `40ep -> 80ep` delta payload, and the bounded mixed-budget result that the earlier `40`-epoch `1024cap` row materially understated shared-blocks10 without settling the same-budget architecture ranking.
**Keywords:** neurips, pdebench, cns, spectral_resnet, shared_blocks10, convergence, 1024cap, 80ep, summary
**Use this when:** Checking whether the old `1024cap` shared-blocks10 row was under-converged, locating the convergence-audit and epoch-budget-delta payloads, or reviewing why the refreshed result still stays capped decision-support evidence rather than a default-profile promotion.

### [NeurIPS Hybrid ResNet PDEBench CNS Hybrid-Spectral Architecture Ablation Summary](plans/NEURIPS-HYBRID-RESNET-2026/pdebench_cns_hybrid_spectral_arch_ablation_summary.md)
**Description:** Implementation summary for the capped CNS hybrid-spectral architecture backlog item, covering the fresh `10`/`40`-epoch sharing pilots, the fresh `40`-epoch shared-depth pilot, the larger-cap finalist confirmation sidecars, and the result that the `10`-block shared bottleneck won the depth pilot but the shared base row recovered the aggregate lead on the `1024 / 128 / 128` confirmation slice.
**Keywords:** neurips, pdebench, cns, hybrid-spectral, architecture, weight-sharing, depth, spectral_resnet, summary
**Use this when:** Checking whether disabling weight sharing still helps after `40` epochs, whether deeper shared spectral bottlenecks survive a larger-cap confirmation pass, or locating the final capped comparison artifacts for this CNS-only architecture lane.

### [NeurIPS Hybrid ResNet PDEBench CNS Hybrid-Spectral 2048-Cap Scaling Summary](plans/NEURIPS-HYBRID-RESNET-2026/pdebench_cns_hybrid_spectral_scaling_2048cap_summary.md)
**Description:** Implementation summary for the capped CNS hybrid-spectral scaling follow-up, covering the frozen `512cap` and `1024cap` finalist manifests, the fresh `2048 / 256 / 256` `40`-epoch rerun, the helper-generated `512 -> 1024 -> 2048` scaling payload, the documented post-run inspect-gate repair, and the bounded result that the shared base row remains the stronger aggregate local reference while the deeper shared row keeps only narrower higher-frequency advantages.
**Keywords:** neurips, pdebench, cns, hybrid-spectral, scaling, 2048cap, spectral_resnet, summary
**Use this when:** Checking whether either hybrid-spectral finalist scales faster as the capped CNS training slice grows to `2048`, locating the authoritative scaling JSON/CSV payload, reviewing the inspect-gate sequencing deviation, or reviewing why the `2048cap` follow-up does not justify a default-profile promotion.

### [NeurIPS Hybrid ResNet PDEBench CNS Hybrid-Spectral Architectural Ablation Plan](plans/NEURIPS-HYBRID-RESNET-2026/pdebench_cns_hybrid_spectral_arch_ablation_plan.md)
**Description:** Implementation plan for a bounded PDEBench `2d_cfd_cns` hybrid-spectral architectural ablation that fixes the canonical CNS shell (`skip-add` + `pixelshuffle`) and varies only spectral bottleneck sharing and depth, with explicit external-audit invariants and larger-cap confirmation.
**Keywords:** neurips, pdebench, cns, hybrid-spectral, spectral_resnet, ablation, skip-add, pixelshuffle, depth, weight-sharing
**Use this when:** Planning or executing the CNS-only hybrid-spectral architecture study, or checking which shell fields must stay fixed so the compare is externally auditable.

### [NeurIPS Hybrid ResNet PDEBench GNOT Paper-Default CNS Compare Plan](plans/NEURIPS-HYBRID-RESNET-2026/pdebench_gnot_paper_default_cns_compare_plan.md)
**Description:** Backlog plan for rerunning the already integrated official GNOT baseline on the local PDEBench `2d_cfd_cns` contract using the patched paper-style GNOT recipe, then comparing it directly against the current spectral anchor under the same capped CNS slice.
**Keywords:** neurips, pdebench, cns, gnot, paper-default, baseline, compare, dgl
**Use this when:** Queueing or implementing the paper-default GNOT rerun on the local CNS contract, or checking which environment, recipe, and comparison surface must stay fixed.

### [NeurIPS Hybrid ResNet PDEBench GNOT CNS Compare Summary](plans/NEURIPS-HYBRID-RESNET-2026/pdebench_gnot_cns_compare_summary.md)
**Description:** Implementation summary for the official GNOT CNS baseline lane, covering the `ptycho311_2` CUDA+DGL host requirement, the first equal-footing fairness probe, a fresh paper-default smoke gate, the same-contract paper-default `40`-epoch follow-up, and the anchored compare sidecar against the pinned spectral `40`-epoch row.
**Keywords:** neurips, pdebench, cns, gnot, paper-default, spectral_resnet, dgl, summary
**Use this when:** Checking the current local GNOT interpretation on the capped CNS contract, locating the smoke/run roots and compare sidecar, or distinguishing the fairness probe from the paper-default follow-up.

### [NeurIPS Hybrid ResNet PDEBench FFNO Convolutional Features CNS Plan](plans/NEURIPS-HYBRID-RESNET-2026/pdebench_ffno_convolutional_features_cns_plan.md)
**Description:** Backlog plan for testing whether adding local convolutional features to FFNO improves capped PDEBench `2d_cfd_cns` performance against authored FFNO, FFNO-close, and Hybrid-spectral anchors.
**Keywords:** neurips, pdebench, cns, ffno, convolution, local-features, ablation
**Use this when:** Queueing or planning the FFNO-family CNS extension that adds convolutional feature paths without changing the local CNS contract.

### [NeurIPS Hybrid ResNet PDEBench FFNO Convolutional Features CNS Summary](plans/NEURIPS-HYBRID-RESNET-2026/pdebench_ffno_convolutional_features_cns_summary.md)
**Description:** Implementation summary for the capped FFNO local-convolution CNS backlog item, covering the Task 1 inspection audit, the required `40`-epoch FFNO-close backfill, the authoritative `10`/`40`-epoch local-conv rows, the anchored compare sidecars, and the result that the local branch materially improved the repo-local FFNO proxy and beat the capped shared-spectral local row while still trailing the official authored FFNO `40`-epoch row.
**Keywords:** neurips, pdebench, cns, ffno, convolution, local-features, summary
**Use this when:** Checking whether the local-conv FFNO follow-up is worth carrying forward, locating the authoritative capped compare artifacts, or reviewing the exact fixed-contract boundary for this FFNO-family extension.

### [NeurIPS Hybrid ResNet PDEBench CNS Hybrid-Spectral To FFNO Parameter-Space Summary](plans/NEURIPS-HYBRID-RESNET-2026/pdebench_cns_hybrid_spectral_ffno_parameter_space_summary.md)
**Description:** Implementation summary for the capped CNS shell-bridge follow-up between the current Hybrid-spectral and repo-local FFNO-family lanes, covering the frozen study matrix, the fresh `spectral_resnet_bottleneck_base_down1` and `spectral_resnet_bottleneck_base_transpose` probes, the anchored `10`-epoch compare sidecar, the closed `40`-epoch promotion gate, and the carry-forward result that the spectral anchor remains the aggregate local shell reference while FFNO local-conv stays the stronger repo-local FFNO-family alternative.
**Keywords:** neurips, pdebench, cns, hybrid-spectral, ffno, parameter-space, shell, decoder, downsampling, summary
**Use this when:** Checking whether the Hybrid-spectral shell should move toward a lighter downsampling path or a transpose decoder on the capped CNS contract, locating the anchored `10`-epoch compare artifacts, or reviewing why the item closed without a `40`-epoch follow-up.

### [NeurIPS Hybrid ResNet CDI FFNO Generator Lines Plan](plans/NEURIPS-HYBRID-RESNET-2026/cdi_ffno_generator_lines_best_config_plan.md)
**Description:** Backlog plan for using FFNO as a CDI/ptycho Torch generator on the best study-indexed lines configuration and comparing it against Hybrid ResNet under the same data, training, stitching, and metric contract.
**Keywords:** neurips, cdi, ptycho, ffno, generator, lines_256, hybrid_resnet
**Use this when:** Planning the CDI/ptycho FFNO generator experiment or checking why CNS FFNO evidence does not answer CDI generator quality.

### [NeurIPS Lines128 Paper-Quality CDI Benchmark Design](plans/NEURIPS-HYBRID-RESNET-2026/lines128_paper_benchmark_design.md)
**Description:** Draft design for a paper-quality `N=128` grid-lines CDI benchmark comparing Hybrid ResNet, Hybrid-spectral, FNO/FNO-vanilla, and FFNO under one frozen best-configuration contract, with shared runner/wrapper architecture, metrics-table requirements, visual reconstruction outputs, provenance gates, and backlog decomposition.
**Keywords:** neurips, cdi, ptycho, grid-lines, lines128, hybrid_resnet, hybrid-spectral, fno, ffno, benchmark, paper-quality
**Use this when:** Planning the NeurIPS CDI benchmark table/figure tranche, deciding why FFNO support must land before the full benchmark, or checking the fixed `N=128` benchmark contract and required paper-grade artifacts.

### [NeurIPS Hybrid ResNet Hybrid-Spectral To FFNO Parameter-Space Plan](plans/NEURIPS-HYBRID-RESNET-2026/hybrid_spectral_ffno_parameter_space_cns_cdi_plan.md)
**Description:** Backlog plan for a staged architecture study across intermediate points between Hybrid-spectral and FFNO, including encoder/downsampling, decoder, and bottleneck axes on both CNS and CDI/ptycho.
**Keywords:** neurips, hybrid-spectral, ffno, architecture, parameter-space, cns, cdi, ptycho
**Use this when:** Planning a broader cross-domain architecture sweep after the narrower CNS and CDI FFNO follow-ups have clarified the immediate design space.

### [NeurIPS Hybrid ResNet PDEBench Spectral Weight-Sharing CNS Compare Summary](plans/NEURIPS-HYBRID-RESNET-2026/pdebench_spectral_weight_sharing_summary.md)
**Description:** Implementation summary for the shared-vs-non-shared spectral bottleneck tranche, covering the manual `spectral_resnet_bottleneck_noshare` profile, the fixed canonical CNS skip-add shell fairness boundary, targeted verification, the capped two-row CNS comparison, and the rendered prediction/error galleries.
**Keywords:** neurips, hybrid_resnet, pdebench, spectral-resnet, weight-sharing, cns, summary, capped-compare
**Use this when:** Checking whether disabling spectral weight sharing helps on the capped CNS slice, locating the shared-vs-non-shared artifacts, or reviewing the exact fairness boundary before planning a longer follow-up run.

### [NeurIPS Hybrid ResNet PDEBench Darcy Static Operator Benchmark Plan](plans/NEURIPS-HYBRID-RESNET-2026/tranches/phase-2-pdebench-darcy-static-operator-benchmark/execution_plan.md)
**Description:** Roadmap Phase 2 implementation plan for the PDEBench Darcy Flow beta `1.0` static operator benchmark, including `nu -> tensor` data/split/normalization contracts, full available training-split requirements, strong FNO and non-toy U-Net baseline gates, and literature-calibrated target values from PDEBench and later HAMLET/OFormer context.
**Keywords:** neurips, hybrid_resnet, pdebench, darcy, static-operator, strong-baseline, fno, unet, sota, nrmse
**Use this when:** Implementing or reviewing the Darcy member of the PDEBench `128x128` image suite, deciding what counts as a strong local baseline, or checking expected U-Net/FNO/SOTA metric ranges before interpreting Hybrid ResNet.

### [NeurIPS Hybrid ResNet PDEBench Darcy Static Operator Summary](plans/NEURIPS-HYBRID-RESNET-2026/pdebench_darcy_static_operator_summary.md)
**Description:** Roadmap Phase 2 implementation summary for the Darcy Flow beta `1.0` static-operator adapter, tests, readiness run, full-run budget, artifact paths, and remaining benchmark-incomplete state.
**Keywords:** neurips, hybrid_resnet, pdebench, darcy, static-operator, readiness, full-budget, benchmark-incomplete
**Use this when:** Checking what Darcy support is implemented, where readiness artifacts live, or why no full strong-baseline benchmark claim exists yet.

### [NeurIPS Hybrid ResNet PDEBench SWE Smoke Gate](plans/NEURIPS-HYBRID-RESNET-2026/pdebench_swe_smoke_gate.md)
**Description:** Roadmap Phase 2 smoke/data-contract gate for the PDEBench 2D Shallow Water Equations (`2D_rdb_NA_NA.h5`) task, now retained as readiness/provenance evidence for the SWE member of the amended native `128x128` PDEBench image suite. Smoke metrics are not benchmark-performance evidence.
**Keywords:** neurips, hybrid_resnet, pdebench, swe, smoke-gate, phase-2, err_nRMSE
**Use this when:** Checking the SWE data-contract details, staged file identity, or smoke-readiness evidence before building the broader PDEBench image-suite adapter.

### [NeurIPS Hybrid ResNet PDE Execution Summary](plans/NEURIPS-HYBRID-RESNET-2026/pde_execution_summary.md)
**Description:** Historical Roadmap Phase 2 longer-execution summary for PDEBench SWE one-step prediction, including dataset/license identity, splits, normalization, local results, and the post-review caveat that the selected run is underconfigured, unseeded decision-support evidence only. Superseded for the next PDE tranche by the PDEBench `128x128` image-suite plan.
**Keywords:** neurips, hybrid_resnet, pdebench, swe, phase-2, pde-execution, pivot, openfwi
**Use this when:** Auditing the historical SWE longer-run artifacts or understanding why new suite runs must use the corrected fixed-seed grid-lines recipe.

### [NeurIPS Hybrid ResNet OpenFWI FlatVel-A Fallback Smoke Gate](plans/NEURIPS-HYBRID-RESNET-2026/openfwi_flatvel_a_fallback_smoke_gate.md)
**Description:** Roadmap Phase 2 fallback smoke/data-access gate for OpenFWI FlatVel-A, now deferred as an optional fallback or adjacent inverse-wave extension while the native `128x128` PDEBench image suite is viable. Smoke metrics are sanity/provenance artifacts only, not benchmark-performance evidence.
**Keywords:** neurips, hybrid_resnet, openfwi, flatvel-a, fallback, smoke-gate, phase-2, pde
**Use this when:** Auditing OpenFWI readiness evidence or explicitly deciding to replace or extend the PDEBench image suite with an inverse-wave benchmark.

### [NeurIPS Hybrid ResNet Plan/Implementation Workflow](../workflows/examples/neurips_hybrid_resnet_plan_impl_review.yaml)
**Description:** Call-based orchestration workflow that loops over next-scope selection from the approved NeurIPS Hybrid ResNet design, roadmap, and progress ledger, then runs plan drafting/review and implementation/review for each selected scope.
**Keywords:** workflow, neurips, hybrid_resnet, roadmap, tranche-selection, plan-review, implementation-review
**Use this when:** You want to drain the approved design and roadmap through adaptive plan scopes while preserving roadmap gates, selected-scope context, and progress history across agent-orchestration iterations.

### [PtychoViT Workflow Guide](workflows/ptychovit.md)
**Description:** Source-pinned interop contract for running the `pinn_ptychovit` model arm from grid-lines studies, including paired HDF5 requirements, checkpoint semantics, and troubleshooting notes.
**Highlights:** Includes `Known Local Dataset Paths (Snapshot: 2026-03-03)` with `scan807`/`cameraman256` path preflight and canonical checkpoint guidance.
**Keywords:** ptychovit, interop, hdf5, checkpoints, grid-lines, studies
**Use this when:** Integrating or debugging the PtychoViT backend path in `scripts/studies/grid_lines_compare_wrapper.py`.

### [Studies Index](studies/index.md)
**Description:** Catalog of active study runbooks, including canonical NERSC scan807+cameraman command lines with concrete `--scan807-*`, `--cameraman-*`, and `--ptychovit-checkpoint` flags, plus links to study-specific loop contracts such as `lines_256`.
**Keywords:** studies, runbooks, nersc, scan807, cameraman, ptychovit, lines_256
**Use this when:** You need copy/paste-ready orchestration commands, concrete local dataset/checkpoint flag patterns, or the exact study doc for an active loop.

### [Lines 256 Dataset Note](studies/lines_256_dataset.md)
**Description:** Repo-local note for the `lines_256` architecture-experiment dataset alias, including the regenerated working NPZ pair, its `pad_preserve` probe-scaling contract, provenance, direct-run usage guidance, and the warning that `outputs/` is only the current compatibility location rather than the preferred long-term storage home.
**Keywords:** lines_256, custom_npz_pair_n256, n256, grid-lines, dataset, npz, outputs, persistent-data
**Use this when:** You need the exact `N=256` lines dataset for a single-dataset experiment, or need to understand both the current compatibility path and the persistent-data policy for this study input.

### [Lines 256 Architecture Loop](studies/lines_256_arch_improvement_loop.md)
**Description:** Exact `autoresearch`-style loop contract for `lines_256` architecture experiments, including fresh baseline generation through the fixed-parameter thin wrapper, inheritance of the project Hybrid ResNet baseline schedule, the untracked TSV ledger path, the session-local champion rule, the comparison-PNG gallery contract, keep/discard reset behavior, and the dedicated-run-checkout rule for this DSL-level rollback/checkpoint workflow.
**Keywords:** lines_256, architecture-loop, amp_ssim, baseline, wrapper, results.tsv, compare_amp_phase, probe, git-reset, session, checkout
**Use this when:** You need a deterministic autonomous experiment loop for the `lines_256` prompt-library agent or want the authoritative rule for baseline generation, inherited-vs-overridden baseline settings, result recording, keep/discard decisions, and how this workflow should be isolated from normal branch work.

### [Lines 256 Controller Loop (V2)](studies/lines_256_controller_loop.md)
**Description:** Parallel controller-driven `lines_256` study path that keeps baseline, scoring, keep/discard, timeout/crash, and resume behavior in a session-local Python controller while the legacy YAML loop remains available during validation.
**Keywords:** lines_256, controller, v2, session-local, run_config, source, timeout, resume
**Use this when:** You want the new controller-based path, need the v2 state/output roots, or need to understand how the parallel rollout differs from the legacy loop.

### [PyTorch Model Loading & Inference Guide](../ptycho_torch/README.md)
**Description:** Practical guide to loading PyTorch models for inference, including the recommended CLI path and manual reconstruction from state_dict-only `model.pt` files.  
**Keywords:** pytorch, inference, model-loading, state_dict, lightning, config-factory  
**Use this when:** You need to run inference or benchmarks with PyTorch models and want a reliable, repeatable loading path.

### [Knowledge Base Ledger](findings.md)
**Description:** Centralized record of critical discoveries, conventions, and recurring bugs, maintained as the agent's long-term memory.  
**Keywords:** knowledge-base, lessons-learned, debugging, conventions  
**Use this when:** Investigating an issue, planning a change, or verifying whether a problem has prior art.

### [Compare Models Spec](../specs/compare_models_spec.md)
**Description:** Interface/behavior contract for `compare_models.py` (CLI + future API), including inputs, outputs, sampling, registration, stitching, metrics, and recon NPZ expectations.  
**Keywords:** spec, comparison, interface, contract  
**Use this when:** You need authoritative details on `compare_models.py` behavior or are refactoring the compare pipeline.

### [PtychoViT Interop Contract Spec](../specs/ptychovit_interop_contract.md)
**Description:** Normative interop contract for `pinn_ptychovit` bridge inputs/outputs, including HDF5 object/position frame semantics, normalization requirements, and reconstruction assembly parity rules.
**Keywords:** spec, ptychovit, interop, hdf5, coordinates, normalization, stitching
**Use this when:** Implementing or debugging the PtychoViT bridge path and validating contract compliance.

## Project Management

### [PROJECT_STATUS](../PROJECT_STATUS.md)
**Description:** Current development status tracker and active initiatives overview.  
**Keywords:** project-status, tracking, initiatives  
**Use this when:** Need to see current project status and active development work.

### [CLAUDE Instructions](../CLAUDE.md)
**Description:** AI agent guidance defining critical data management rules, configuration requirements, and development conventions.  
**Keywords:** ai-agent, conventions, guidelines, critical-rules  
**Use this when:** Working as an AI agent or need to understand project conventions and critical requirements.

### [Orchestration Start Here](workflows/orchestration_start_here.md)
**Description:** Canonical onboarding for backlog-loop orchestration that explicitly separates orchestration policy, DSL workflow authoring, and runtime step execution; includes workflow/step/prompt/plan boundaries, plan-to-workflow contract model, and the repo-local rule that live-checkout git-safety guidance is only required for workflows with DSL-level rollback/checkpoint behavior.
**Keywords:** orchestration, workflow, step, prompt, plan, ownership-boundaries, producer-consumer, git
**Use this when:** Authoring DSL workflows, editing backlog-loop prompts/plans, or deciding whether a workflow such as `lines_256` needs a dedicated run checkout.

### [Implementation Plan Template](plans/templates/implementation_plan.md)
**Description:** Repo-specific phased plan template with initiative header, exit criteria, per‑phase checklists, Do Now block, artifacts index, and plan‑update protocol reminder.  
**Keywords:** template, plan, phased, checklist, exit-criteria  
**Use this when:** Creating or reshaping an initiative plan to match project conventions.

### [Design Template](plans/templates/design_template.md)
**Description:** General design/ADR template for initiatives that need scope, rationale, contracts, dependency discovery, provenance, pivot criteria, and planning handoff before implementation.
**Keywords:** template, design, adr, dependency-discovery, provenance, pivot-criteria
**Use this when:** Drafting a design source of truth before an implementation plan, especially when choices affect architecture, scientific claims, external dependencies, data contracts, or reviewer-facing evidence.

### [Workflow Contract Plan Template](plans/templates/workflow_contract_plan.md)
**Description:** Minimal template for plans intended for workflow execution, with a strict workflow-level contract surface (`< 3` artifacts), producer/consumer mapping, and required plan-author/workflow-author coordination checklist.  
**Keywords:** template, workflow-contract, producer-consumer, artifact-lineage, orchestration  
**Use this when:** Writing plans that must expose only a small, explicit set of workflow-level inputs/outputs.

### [Debug Task Plan Template (Optional)](plans/templates/debug_task_plan.md)
**Description:** Optional task-first template for debugging/evidence initiatives with hypothesis ledger, deterministic diagnostics, and decision-gate sections.  
**Keywords:** template, debug, hypothesis, investigation, decision-gate  
**Use this when:** Running scoped debugging investigations that should not replace the canonical implementation-plan schema.

### [Workflow Backlog Item Template](backlog/templates/backlog_item_workflow.md)
**Description:** Template for backlog queue items consumed by the agent-orchestration backlog loop, including required frontmatter (`priority`, `plan_path`, `check_commands`) and directory-placement queue rules (`active/paused/done`).  
**Keywords:** template, backlog, workflow, queue, orchestrator  
**Use this when:** Creating or updating `docs/backlog/active/` items for automated plan-slice execution.

### [Agent Git Setup (Runbook)](../prompts/git_setup_agent.md)
**Description:** Step-by-step, idempotent setup for Git in supervisor/loop/orchestrator environments. Covers global config defaults, submodule hygiene for .claude/claude, .gitignore hardening for logs/outputs/data, safe pull wrappers, and recovery playbooks.  
**Keywords:** git, automation, supervisor, loop, submodules, rebase, setup  
**Use this when:** Bringing the orchestration stack to a new repo or stabilizing pull/rebase behavior.

### [Git Hygiene Guidelines](../prompts/git_hygiene.md)
**Description:** Reusable practices to keep automation-friendly repositories clean and conflict-free: submodule policy (ignore=dirty for tooling), ignore lists, safe pull patterns, CI guidance, and verification checklist.  
**Keywords:** git-hygiene, submodules, ignore, CI, rerere, autosquash  
**Use this when:** Maintaining multiple repos that run the same supervisor/loop and you want consistent, low-friction Git behavior.

### [Agent-Orchestration Backlog Loop Runbook](workflows/agent_orchestration_backlog_loop.md)
**Description:** Maintainer runbook for launching and monitoring the backlog-driven plan-slice workflow via `agent-orchestration`, including tmux kickoff, backlog frontmatter contract, review/fix loop behavior, and resume commands.  
**Keywords:** orchestrator, workflow, backlog, tmux, review-loop, codex, claude  
**Use this when:** Kicking off long-running backlog execution in PtychoPINN using `workflows/agent_orchestration/backlog_plan_slice_impl_review_loop.yaml`.

### [Prompt Drafting Guide](PROMPT_DRAFTING_GUIDE.md)
**Description:** Practical rules for writing orchestration prompts with clean abstraction boundaries, contract-driven inputs/outputs, KISS defaults, and flexible scope handling across single or multiple plan/backlog artifacts.  
**Keywords:** prompts, orchestration, contracts, consumed-artifacts, output-contract, prompt-engineering  
**Use this when:** Creating or revising workflow prompt files in `prompts/workflows/backlog_plan_loop/` (or similar orchestration prompt sets).

## Architecture & Development

### Core Documentation

#### [Developer Guide](DEVELOPER_GUIDE.md) ⚠️ ESSENTIAL
**Description:** Comprehensive architectural guide covering the "two-system" architecture (legacy grid-based vs. modern coordinate-based), critical anti-patterns, data pipeline contracts, TDD methodology, and configuration management best practices. Includes PYTHON-ENV-001 (Interpreter & Subprocess Policy) as the single source of truth for interpreter selection.  
**Keywords:** architecture, TDD, anti-patterns, configuration, data-pipeline  
**Use this when:** Starting any development work, debugging shape mismatches, understanding the codebase architecture, or implementing new features using TDD methodology.

#### [Invocation Logging Guide](development/INVOCATION_LOGGING_GUIDE.md)
**Description:** Repo-wide standard for logging exact script/orchestration command invocations via `invocation.json` and `invocation.sh`, including placement rules and testing expectations.  
**Keywords:** invocation, provenance, reproducibility, cli, orchestration, logging  
**Use this when:** Adding or modifying CLI entrypoints and wrappers that write run artifacts.

#### [Architecture Overview](architecture.md)
**Description:** Shared core architecture (data contracts, grouping, configuration, orchestration concepts) across both backends. Includes Scripts Overview and stable-modules policy, plus config lifecycle snippet (`update_legacy_dict(params.cfg, config)`). Backend-specific sequences and diagrams are in the TF/PyTorch pages below.  
**Keywords:** components, data-flow, scripts, shared-architecture, stable-modules, configuration  
**Use this when:** Getting oriented with shared architecture and how scripts map to orchestrators.

#### [Architecture — TensorFlow](architecture_tf.md)
**Description:** TensorFlow-specific architecture: component diagram, training and inference sequences, component reference, stable modules, and function/container mapping (TF ↔ PyTorch).  
**Keywords:** tensorflow, components, training, inference, containers, mapping  
**Use this when:** Implementing or debugging the TensorFlow backend and its workflows.

#### [Architecture — PyTorch](architecture_torch.md)
**Description:** PyTorch-specific architecture: component diagram, Lightning-based training and inference sequences, component reference, and function/container mapping (PyTorch ↔ TF). Includes FNO/hybrid generators via `ptycho_torch/generators/fno.py`.
**Keywords:** pytorch, lightning, components, training, inference, containers, mapping, fno, hybrid
**Use this when:** Implementing or debugging the PyTorch backend and its workflows, or using FNO/hybrid architectures.

#### [Testing Guide](TESTING_GUIDE.md)
**Description:** Comprehensive testing strategy covering unit tests, integration tests, TDD methodology, regression testing practices, and specific guidance for testing CLI parameters and backward compatibility.  
**Keywords:** testing, TDD, integration, regression, CLI-testing  
**Use this when:** Writing new tests, running the test suite, implementing TDD cycles, or ensuring backward compatibility.

#### <doc-ref type="test-index">docs/development/TEST_SUITE_INDEX.md</doc-ref>
**Description:** Machine-generated catalog of every `tests/` module with purpose statements, key test names, and direct execution commands.  
**Keywords:** test-coverage, discovery, regression, navigation  
**Use this when:** Locating the right test to extend, auditing coverage during code reviews, or coordinating TDD cycles across the suite.

#### [Troubleshooting Guide](debugging/TROUBLESHOOTING.md)
**Description:** Practical debugging guide for common issues including shape mismatch errors, configuration precedence problems, oversampling setup, and quick debugging commands with solutions.  
**Keywords:** debugging, shape-mismatch, configuration, oversampling, quick-fixes  
**Use this when:** Encountering shape mismatch errors (especially gridsize-related), debugging configuration issues, or need quick diagnostic commands.

#### [Debugging Methodology](debugging/debugging.md)
**Description:** Standard four-step process (verify contracts → sync configuration → isolate component → capture failing test) required for all new investigations.  
**Keywords:** methodology, workflow, testing, triage  
**Use this when:** Beginning any new bug hunt or postmortem to ensure consistent, auditable steps.

#### [Undocumented Conventions](debugging/undocumented_conventions.md)
**Description:** Living list of subtle behaviors (e.g., two-system assumptions, legacy sync order) that commonly cause regressions when overlooked.  
**Keywords:** conventions, gotchas, params.cfg, legacy-system  
**Use this when:** Reviewing legacy code, onboarding new teammates, or double-checking implicit assumptions.

### Configuration & Data

#### [Configuration Guide](CONFIGURATION.md)
**Description:** Canonical reference for the modern dataclass-based configuration system with comprehensive parameter documentation for ModelConfig, TrainingConfig, and InferenceConfig classes.  
**Keywords:** configuration, parameters, dataclass, YAML, command-line  
**Use this when:** Setting up training or inference runs, understanding parameter precedence, or creating reproducible experiment configurations.

#### [Data Contracts](../specs/data_contracts.md) CRITICAL
**Description:** Official format specifications for NPZ datasets including required keys, data types, shapes, and normalization requirements.
**Keywords:** NPZ-format, data-contracts, normalization, diffraction, amplitude
**Use this when:** Creating or validating datasets, troubleshooting data format errors, or understanding amplitude vs intensity requirements.

#### [Data Management Guide](DATA_MANAGEMENT_GUIDE.md)
**Description:** Best practices for managing NPZ and HDF5 data files, including the distinction between durable dataset/checkpoint storage under `datasets/` and cleanup-prone run artifacts under `outputs/`, plus git hygiene rules and Ptychodus export workflow guidance.
**Keywords:** data-management, git-hygiene, file-types, NPZ, HDF5, ptychodus-export, datasets, outputs, persistent-data
**Use this when:** Understanding data file organization, deciding where long-lived study inputs should live, exporting reconstructions to Ptychodus format, or ensuring data files are not committed to git.

#### [Data Normalization Guide](DATA_NORMALIZATION_GUIDE.md)
**Description:** Explains the three distinct types of normalization (physics, statistical, display) and their proper application throughout the data pipeline to avoid common scaling errors.
**Keywords:** normalization, intensity_scale, physics, statistical, pipeline
**Use this when:** Debugging normalization issues, implementing new data loading features, or resolving scaling-related bugs.

#### [Data Generation Guide](DATA_GENERATION_GUIDE.md) CRITICAL
**Description:** Comprehensive guide to the two data generation pipelines: grid-based (`mk_simdata`) for notebook-compatible workflows and nongrid (`generate_simulated_data`) for production scripts. Covers parameter mappings, entry points, and container construction.
**Keywords:** simulation, synthetic-data, grid, nongrid, mk_simdata, generate_simulated_data, params.cfg
**Use this when:** Implementing dose studies, generating synthetic datasets, choosing between grid and nongrid simulation, or debugging data generation issues.

#### [GridSize & n_groups Guide](GRIDSIZE_N_GROUPS_GUIDE.md) CRITICAL
**Description:** Explains the unified n_groups parameter behavior across different gridsize values, eliminating confusion between individual images vs groups.  
**Keywords:** n_groups, gridsize, sampling, groups, patterns  
**Use this when:** Understanding group formation with different gridsize values or troubleshooting unexpected dataset sizes.

### Specialized Topics

#### [Sampling User Guide](SAMPLING_USER_GUIDE.md)
**Description:** Comprehensive guide to flexible sampling control using n_subsample and n_groups parameters for memory management and training efficiency.  
**Keywords:** sampling, n_subsample, n_groups, memory, reproducibility  
**Use this when:** Controlling memory usage during training or implementing diverse sampling strategies.

#### [Pty-Chi Migration Guide](PTYCHI_MIGRATION_GUIDE.md)
**Description:** Complete migration guide for replacing Tike with pty-chi in three-way comparisons.
**Keywords:** pty-chi, tike-replacement, performance-optimization, reconstruction  
**Use this when:** Want to speed up iterative ptychographic reconstructions or migrate from Tike to pty-chi.

#### [Memory Optimization Guide](memory.md)
**Description:** Technical context document for Phase 5 of the Independent Sampling Control initiative detailing current implementation state and documentation gaps.  
**Keywords:** sampling-control, phase5-context, memory-management, parameter-interpretation  
**Use this when:** Working on Phase 5 of the sampling control initiative or need context on current implementation state.

## Workflows & Scripts

### Core Workflows

#### [Training](../scripts/training/README.md)
**Description:** Comprehensive guide for training PtychoPINN models with NPZ data, including configuration, sampling modes (random/sequential), logging, and next steps for evaluation.  
**Keywords:** training, configuration, sampling, logging, model-artifacts  
**Use this when:** Training a PtychoPINN model from scratch with proper configuration and understanding all training options.

#### [Inference](../scripts/inference/README.md)
**Description:** Guide for running inference with trained PtychoPINN models on test data, generating reconstruction visualizations and probe analysis.  
**Keywords:** inference, reconstruction, visualization, model-loading, testing  
**Use this when:** Have a trained model and want to generate reconstructions from new test data.

#### Evaluation (via Studies) — [Studies Guide](../scripts/studies/README.md)
**Description:** Use the Studies tooling to run evaluations and aggregate metrics across runs; for single reconstructions, see Inference. Includes `grid_lines_torch_runner.py` for FNO/hybrid architecture training.
**Keywords:** evaluation, studies, metrics, analysis, aggregation, fno, hybrid, torch-runner
**Use this when:** Running evaluations across datasets, comparing models, or training FNO/hybrid architectures via the Torch runner.

#### [Simulation](../scripts/simulation/README.md)
**Description:** Two-stage modular simulation architecture for generating ptychographic datasets: Stage 1 creates object/probe inputs, Stage 2 simulates diffraction patterns.  
**Keywords:** simulation, diffraction, modular, synthetic-data, workflow  
**Use this when:** Need to generate synthetic ptychographic datasets from custom objects or existing reconstructions.

### Advanced Workflows

#### [Model Comparison](../scripts/studies/README.md)
**Description:** Systematic generalization study framework for training models across dataset sizes, supporting synthetic/experimental data with statistical robustness.  
**Keywords:** generalization, multi-trial, statistical-analysis, comparison  
**Use this when:** Conducting rigorous performance studies across training set sizes with statistical robustness.

#### [Reconstruction with Pty-Chi](../scripts/reconstruction/README.md)
**Description:** Traditional iterative reconstruction methods (Tike, Pty-chi) for baseline comparison against neural network approaches with configurable algorithms.  
**Keywords:** iterative-reconstruction, tike, pty-chi, baseline-comparison  
**Use this when:** Need traditional iterative reconstructions as baselines for comparison with PtychoPINN.

#### [Data Preprocessing Tools](../scripts/tools/README.md)
**Description:** Essential data preprocessing pipeline tools for format conversion, dataset preparation, splitting, and visualization with recovery patterns.  
**Keywords:** preprocessing, format-conversion, dataset-preparation, visualization  
**Use this when:** Have raw experimental data that needs format standardization or want to prepare datasets for training.

### Command Reference

#### [Commands Reference](COMMANDS_REFERENCE.md)
**Description:** Quick reference guide for essential PtychoPINN workflows including data preparation golden paths, training, inference, evaluation, and troubleshooting commands.  
**Keywords:** quick-reference, command-line, workflows, best-practices  
**Use this when:** Need rapid lookup of common command patterns and want to follow established golden paths.

## Studies & Analysis

### Study Guides

#### [Generalization Study Guide](studies/GENERALIZATION_STUDY_GUIDE.md)
**Description:** Complete guide for running multi-model generalization studies comparing PtychoPINN and baseline performance across different training set sizes.  
**Keywords:** generalization-study, model-comparison, training-sizes, automated-workflow  
**Use this when:** Systematically comparing model performance across different amounts of training data.

#### [Studies Guide](../scripts/studies/README.md)
**Description:** Study scripts and workflows for generalization experiments, comparisons, and metrics aggregation.  
**Keywords:** studies, generalization, comparisons, workflows  
**Use this when:** Running study workflows or comparing multiple models.

#### [Studies Index](studies/index.md)
**Description:** Registry of reproducible study runbooks with exact launcher scripts, full CLI commands, and output directories.  
**Keywords:** studies-index, runbook, cli, reproducibility, outputs  
**Use this when:** Finding the exact commands and artifacts for a specific study run.

## Datasets & Experiments

#### [FLY64 Dataset Guide](FLY64_DATASET_GUIDE.md)
**Description:** Comprehensive guide for working with the FLY64 experimental ptychography dataset, including preprocessing requirements and specialized subset datasets.  
**Keywords:** fly64, experimental-data, preprocessing, dataset-variants  
**Use this when:** Working with real experimental ptychography data (fly64) or need to understand dataset preprocessing requirements.

#### [FLY001 N=128 Dataset Guide](FLY001_128_DATASET_GUIDE.md)
**Description:** Reproducible preparation guide for the fly001 `N=128` dataset, including canonicalization, deterministic top/bottom split generation, and manifest provenance fields used by external grid-lines studies.  
**Keywords:** fly001, n128, external-dataset, canonicalization, split, provenance  
**Use this when:** Preparing or validating the disjoint train/test fly001 `N=128` external-study datasets.

#### [FLY64 Generalization Analysis](FLY64_GENERALIZATION_STUDY_ANALYSIS.md)
**Description:** Detailed analysis of a complete generalization study on fly64 dataset revealing unexpected results where baseline models significantly outperform PtychoPINN.  
**Keywords:** fly64-analysis, unexpected-results, baseline-superiority, methodology-validation  
**Use this when:** Need to understand why baseline models might outperform PINNs on experimental data.

#### [Tike Reassembly Artifact Fix](TIKE_REASSEMBLY_ARTIFACT_FIX.md)
**Description:** Implementation guide for fixing visual reassembly artifacts in Tike reconstructions processed through PtychoPINN's comparison pipeline.  
**Keywords:** tike-integration, artifact-fixing, boundary-handling, coordinate-systems  
**Use this when:** Encounter visual grid artifacts in Tike reconstruction comparisons.

## Core Module Documentation

### Neural Network & Physics (`ptycho/`)

#### `ptycho/generators/` - Generator Registry
**Description:** Modular generator registry enabling architecture selection via `config.model.architecture`. Supports CNN (default), FNO, and hybrid architectures with a consistent interface for adding new generators.
**Key Dependencies:** `ModelConfig.architecture` field, `ptycho.model` for CNN implementation
**Critical For:** Architecture selection, adding new generator types, extending model capabilities
**Key Modules:**
- `registry.py`: Central registry with `resolve_generator()` function
- `cnn.py`: CNN-based U-Net generator (default)
- `README.md`: Guide for adding new generators

#### `ptycho/model.py` - Neural Network Architecture ⚠️ SINGLETON WARNING
**Description:** U-Net-based physics-informed neural network combining deep learning with differentiable ptychographic forward modeling. Features custom Keras layers for physics constraints.
**Key Dependencies:** Global `params.cfg` state at import time, `tf_helper` for tensor operations
**Critical For:** Training workflows, inference, physics-informed reconstruction
**⚠️ CRITICAL:** `model.autoencoder` and `model.diffraction_to_obj` are **module-level singletons** created at import time. They capture `params.cfg['gridsize']` when imported, NOT when used. If you change gridsize after importing this module, use `create_model_with_gridsize(gridsize, N)` to get correctly-sized models. See [MODULE-SINGLETON-001](findings.md) and [Troubleshooting](debugging/TROUBLESHOOTING.md#model-architecture-mismatch-after-changing-gridsize).

#### `ptycho/diffsim.py` - Physics Simulation Layer
**Description:** Core forward physics implementation simulating the complete ptychographic measurement process: object illumination → coherent diffraction → Poisson photon noise.  
**Key Dependencies:** `params.cfg` for physics parameters (nphotons, N), `tf_helper` for differentiable operations  
**Critical For:** Training data generation, physics loss constraints, synthetic dataset creation

### Specifications

#### [PtychoPINN Spec — Index](specs/spec-ptychopinn.md)
**Description:** Index of normative spec shards for the TensorFlow‑based physics‑informed ptychography pipeline (core physics, runtime, workflow, interfaces, conformance, tracing).  
**Keywords:** spec, ptychography, TensorFlow, physics‑informed, index  
**Use this when:** You need the top‑level map of all PtychoPINN specifications.

#### [PtychoPINN Core Physics & Data Contracts](specs/spec-ptycho-core.md)
**Description:** Normative definition of the forward model (object·probe→FFT→|F|²/N²→sqrt), Poisson observation, intensity scaling symmetry, coordinates/patch extraction, probe/masking/smoothing, valid inputs, losses, and outputs.  
**Keywords:** physics, FFT, Poisson, scaling, offsets, probe, contracts  
**Use this when:** Implementing or auditing the physical/mathematical operations and strict data shapes.

#### [PtychoPINN Runtime & Execution](specs/spec-ptycho-runtime.md)
**Description:** TensorFlow runtime guardrails: dtype/device policy, XLA translation/compile modes, vectorization/streaming, graph hygiene, environment flags, and error conditions.  
**Keywords:** runtime, TensorFlow, XLA, vectorization, determinism  
**Use this when:** Tuning performance, enabling XLA, or validating execution safety constraints.

#### [PtychoPINN Workflow](specs/spec-ptycho-workflow.md)
**Description:** End‑to‑end pipeline: NPZ ingest → grouping → normalization → model → loss/optimization → inference → stitching → evaluation; staging knobs and guards.  
**Keywords:** workflow, grouping, normalization, training, inference, stitching, evaluation  
**Use this when:** Building or verifying the full training/inference pipeline.

#### [PtychoPINN Interfaces](specs/spec-ptycho-interfaces.md)
**Description:** Public API surface and data/file interfaces (RawData, loader, models, training/eval), model I/O contracts, precedence rules for params/env, and error conditions.  
**Keywords:** API, data‑contracts, shapes, params, precedence  
**Use this when:** Integrating modules, writing loaders, or consuming model interfaces.

#### [PtychoPINN Conformance Tests](specs/spec-ptycho-conformance.md)
**Description:** Acceptance tests PTY‑AT‑XXX: forward amplitude equivalence, Poisson semantics, grouping/coords shapes, translation round‑trip, intensity scaling symmetry, positive intensity for NLL, loader contracts, inference determinism, stitch border math.  
**Keywords:** conformance, acceptance‑tests, parity, validation  
**Use this when:** Certifying a build or diagnosing regressions against the spec.

#### [PtychoPINN Tracing & Debug](specs/spec-ptycho-tracing.md)
**Description:** Tracing obligations for physics/intermediate tensors, coordinate/translation traces, scaling invariants, and first‑divergence workflow with artifact guidance.  
**Keywords:** tracing, debug, parity, diagnostics  
**Use this when:** Investigating numerical/physics divergences or instrumentation gaps.

#### [Ptychodus Integration API Spec](../specs/ptychodus_api_spec.md)
**Description:** Normative API contract for integrating PtychoPINN with Ptychodus, covering configuration bridging, backend selection, data ingestion, lifecycle, and persistence expectations.  
**Keywords:** spec, API, integration, config-bridge, backend  
**Use this when:** Implementing or validating a backend used by Ptychodus, or wiring configs through `update_legacy_dict()` per the contract.

#### [Ptychodus Data Contracts](../specs/data_contracts.md)
**Description:** Normative HDF5 product format (metadata, positions, probe, object, loss history) used by Ptychodus product readers/writers; includes shapes, dtypes, and units.  
**Keywords:** data-contracts, HDF5, product, metadata, probe, object  
**Use this when:** Writing/reading product files or converting datasets to the Ptychodus product format.

#### [Config Bridge (TensorFlow ↔ PyTorch)](specs/spec-ptycho-config-bridge.md)
**Description:** Normative mapping between TensorFlow dataclass configs and PyTorch config singletons, including field transformations (grid_size→gridsize, epochs→nepochs, mode→model_type), defaults/overrides, update_legacy_dict flow, and validation rules.  
**Keywords:** config, bridge, translation, params.cfg, dataclasses, pytorch  
**Use this when:** Translating configuration between backends, ensuring CONFIG‑001 compliance, or verifying field mappings.

#### [Overlap Metrics Spec](specs/overlap_metrics.md)
**Description:** Overlap-driven sampling and reporting for Phase D. Defines three 2D disc-overlap metrics (group-based, image-based, and group↔group COM-based), explicit controls via `s_img` and `n_groups`, and removes spacing/packing acceptance gates.  
**Keywords:** overlap, metrics, s_img, n_groups, probe-diameter, gridsize  
**Use this when:** Implementing or validating Phase D overlap behavior and reporting measured overlaps instead of geometry gating.

### Configuration & Workflows

#### `ptycho/config/` - Configuration System
**Description:** Modern dataclass-based configuration with one-way translation to legacy `params.cfg`. Provides type-safe configuration management.  
**Key Dependencies:** `KEY_MAPPINGS` for legacy compatibility, YAML loading utilities  
**Critical For:** All workflows requiring parameter management, CLI argument parsing

#### `ptycho/workflows/` - High-Level Workflow Functions
**Description:** Orchestration layer bridging CLI scripts and core modules. Chains together complete pipelines including data loading, configuration management, and training.
**Key Dependencies:** Core modules integration, configuration system, data pipeline
**Critical For:** End-to-end workflows, `run_cdi_example()`, `grid_lines_workflow`, training orchestration
**Key Modules:**
- `grid_lines_workflow.py`: End-to-end grid-based pipeline (probe prep → simulation → train → infer → stitch → metrics)

### Data Pipeline

#### `ptycho/loader.py` - Data Loading Utilities
**Description:** NumPy-to-TensorFlow conversion layer transforming grouped data into GPU-ready tensors. Handles dtype conversion and tensor reshaping for multi-channel architectures.  
**Key Dependencies:** `raw_data.py` for grouped data, TensorFlow tensor operations  
**Critical For:** Model training, inference data preparation, tensor pipeline

#### `ptycho/raw_data.py` - Raw Data Handling
**Description:** First-stage data pipeline transforming NPZ files into structured containers with efficient coordinate grouping. Implements "sample-then-group" strategy for 10-100x performance improvement.  
**Key Dependencies:** `params.cfg` for grouping parameters, scikit-learn for nearest neighbors  
**Critical For:** Data preprocessing, coordinate grouping, NPZ file ingestion

#### `ptycho/tf_helper.py` - TensorFlow Utilities
**Description:** Essential tensor transformation operations including three-format conversion system (Grid/Channel/Flat), patch extraction/reassembly, and batched operations.  
**Key Dependencies:** Global `params.cfg` for tensor dimensions (N, gridsize, offset)  
**Critical For:** Data format conversion, patch operations, model training/inference

### Evaluation & Baselines

#### `ptycho/evaluation.py` - Evaluation Metrics
**Description:** Central quality assessment orchestrating multiple metrics (SSIM, MS-SSIM, FRC, MAE). Handles complex data preprocessing, phase alignment, and standardized interfaces.  
**Key Dependencies:** `ptycho.FRC`, `ptycho.image` for registration, scikit-image metrics  
**Critical For:** Training validation, model comparison, research analysis

#### `ptycho/baselines.py` - Baseline Models
**Description:** Supervised learning baseline using dual-output U-Net architecture for pure data-driven reconstruction without physics constraints.  
**Key Dependencies:** Standard TensorFlow/Keras layers, `tf_helper` for compatibility  
**Critical For:** Model comparison studies, benchmarking, performance evaluation

#### `ptycho/misc.py` - Utility Functions
**Description:** Caching decorators (`@memoize_disk_and_memory`) for expensive computations, output path generation, and image processing helpers.  
**Key Dependencies:** `params.cfg` for configuration state, filesystem operations  
**Critical For:** Performance optimization, caching, utility operations across modules

## Finding Information

### By Task
- **Authoring or executing backlog-loop DSL workflows**: [Orchestration Start Here](workflows/orchestration_start_here.md) → [Agent-Orchestration Backlog Loop Runbook](workflows/agent_orchestration_backlog_loop.md) → [Prompt Drafting Guide](PROMPT_DRAFTING_GUIDE.md)
- **Starting a new feature**: [Developer Guide](DEVELOPER_GUIDE.md) → [Workflow Contract Plan Template](plans/templates/workflow_contract_plan.md)
- **Running experiments**: [Workflow Guide](WORKFLOW_GUIDE.md) → [Commands Reference](COMMANDS_REFERENCE.md)
- **Debugging issues**: [Troubleshooting](debugging/TROUBLESHOOTING.md) → [Quick Reference Params](debugging/QUICK_REFERENCE_PARAMS.md)
- **Understanding data**: [Data Contracts](../specs/data_contracts.md) → [Data Normalization](DATA_NORMALIZATION_GUIDE.md)
- **Fixing shape mismatches**: [Quick Reference Params](debugging/QUICK_REFERENCE_PARAMS.md) → [Troubleshooting](debugging/TROUBLESHOOTING.md)
- **Training models**: [Training README](../scripts/training/README.md) → [Configuration Guide](CONFIGURATION.md)
- **Evaluating models**: [Model Comparison Guide](MODEL_COMPARISON_GUIDE.md) → [Studies Guide](../scripts/studies/README.md)

### By User Type
- **New Users**: [README](../README.md) → [Workflow Guide](WORKFLOW_GUIDE.md) → [Training](../scripts/training/README.md)
- **Developers**: [Developer Guide](DEVELOPER_GUIDE.md) → [Testing Guide](TESTING_GUIDE.md) → [Architecture](architecture.md)
- **Researchers**: [Generalization Studies](studies/GENERALIZATION_STUDY_GUIDE.md) → [Model Comparison](../scripts/studies/README.md)
- **AI Agents (DSL authoring + backlog-loop runtime)**: [CLAUDE.md](../CLAUDE.md) → [Orchestration Start Here](workflows/orchestration_start_here.md) → [Agent-Orchestration Backlog Loop Runbook](workflows/agent_orchestration_backlog_loop.md)

## Documentation Standards

When adding new documentation:
1. Update this index with detailed descriptions (1-2 lines minimum)
2. Include keywords/tags for searchability
3. Add "Use this when..." guidance
4. Use the `<doc-ref>` XML tagging system for cross-references
5. Ensure bidirectional linking
6. Add to [PROJECT_STATUS.md](../PROJECT_STATUS.md) if it's an initiative document

## Bug Reports & Fixes

### [TF Bundle Loader Keras 3 Graph Disconnect](bugs/2026-02-11-tf-loader-keras3-graph-disconnect.md) - OPEN
**Description:** Some valid TensorFlow `wts.h5.zip` bundles fail to load via Keras 3 deserialization with `ValueError: inputs not connected to outputs`.
**Status:** Open (reported 2026-02-11)
**Tracking Backlog:** [Loader Fallback Hardening](backlog/2026-02-11-tf-loader-keras3-fallback.md)

### [MATH-POLAR-001: CombineComplexLayer Bug](bugs/MATH_POLAR_001.md) - FIXED
**Description:** `CombineComplexLayer` incorrectly combined amplitude and phase as Z=A+iφ instead of Z=A*exp(iφ). This broke phase averaging in patch stitching.
**Status:** Fixed 2026-01-26
**Fix:** Apply Euler's formula; add `use_polar=False` flag for loading legacy models.

### [XLA Inference Bug](bugs/XLA_INFERENCE_BUG.md) - FIXED
**Description:** PINN models with XLA compilation failed during inference with dynamic batch sizes due to `tf.repeat`/`tf.tile` operations in the Translation layer.
**Status:** Fixed 2026-01-26
**Fix:** Force Translation layer to use XLA-safe `tf.gather` path; set `object_big=False` in grid_lines_workflow.

---

## 🔗 External Resources

- **Paper**: [Nature Scientific Reports Publication](https://www.nature.com/articles/s41598-023-48351-7)
- **GitHub Issues**: Report bugs and request features
- **License**: [LICENSE](../LICENSE)

---

*Last updated: February 2026*
*added detailed descriptions for improved navigation*
