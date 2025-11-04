Summary: Catalog Phase G comparison prerequisites and flag missing artifacts so execution planning has ground truth.
Mode: Docs
Focus: STUDY-SYNTH-FLY64-DOSE-OVERLAP-001.G0 — Evidence inventory & harness prep
Branch: feature/torchapi-newprompt
Mapped tests: none — evidence-only
Artifacts: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-05T162500Z/phase_g_inventory/

Do Now:
- Implement: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-05T162500Z/phase_g_inventory/analysis/inventory.md — enumerate Phase C datasets, Phase E PINN/baseline checkpoints, and Phase F manifests for every dose/view/split, explicitly marking where `ptychi_reconstruction.npz` or other comparison inputs are missing and citing the source reports.
- Capture: find plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001 -maxdepth 6 -type f -name "*manifest*.json" | sort | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-05T162500Z/phase_g_inventory/analysis/phase_f_manifest_listing.txt; find plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001 -maxdepth 8 -name "ptychi_reconstruction.npz" | sort | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-05T162500Z/phase_g_inventory/analysis/ptychi_npz_inventory.txt.

Priorities & Rationale:
- plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-05T140500Z/phase_g_comparison_plan/plan/plan.md:16 — G0.1 requires an authoritative inventory before execution work can proceed.
- plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/test_strategy.md:259 — Phase G coverage notes inventory + real-run selectors missing.
- docs/fix_plan.md:31 — Status callout flags G0.1/G2 as outstanding; this loop must unblock G2 by validating prerequisites.
- specs/data_contracts.md:1 — Inventory must confirm candidate NPZ inputs adhere to the canonical DATA-001 contract or note gaps.

How-To Map:
- export AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md
- find plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001 -maxdepth 6 -type f -name "*manifest*.json" | sort | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-05T162500Z/phase_g_inventory/analysis/phase_f_manifest_listing.txt
- find plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001 -maxdepth 8 -name "ptychi_reconstruction.npz" | sort | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-05T162500Z/phase_g_inventory/analysis/ptychi_npz_inventory.txt
- ls datasets/fly64 | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-05T162500Z/phase_g_inventory/analysis/datasets_fly64_listing.txt

Pitfalls To Avoid:
- Do not modify production code or tests; this loop is documentation-only.
- Keep inventory outputs scoped to relevant files to prevent giant logs.
- Preserve existing artifacts; annotate rather than overwrite manifests.
- Cite absolute evidence paths (reports/2025-11-04T*, datasets/*) instead of informal notes.
- If an expected artifact is missing, log it in the inventory with TODO + pointer rather than fabricating data.
- Avoid assumptions about future CLI behavior; record only what is verifiable now.

If Blocked:
- If required directories are absent, capture the failing command output to plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-05T162500Z/phase_g_inventory/analysis/blockers.log, summarize the gap in inventory.md, and note the block in docs/fix_plan.md Attempts History.

Findings Applied (Mandatory):
- POLICY-001 — Assume torch>=2.2 available; inventory should reflect PyTorch-backed artifacts without suggesting torch-optional fallbacks.
- CONFIG-001 — Track which assets already include CONFIG-001 bridge evidence to avoid re-running legacy modules unsafely.
- DATA-001 — Validate dataset NPZ candidates reference DATA-001 compliant sources or document contract violations.
- OVERSAMPLING-001 — Record sparse acceptance metadata sources so later comparisons can justify low-acceptance sparse runs.

Pointers:
- plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-05T140500Z/phase_g_comparison_plan/plan/plan.md:9
- plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/test_strategy.md:248
- docs/fix_plan.md:64
- specs/data_contracts.md:1

Next Up (optional):
- Implement comparison execution (G2) once inventory confirms available NPZ + checkpoint assets.
