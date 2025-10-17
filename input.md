Summary: Promote PyTorch to a required dependency before touching torch guard code.
Mode: none
Focus: INTEGRATE-PYTORCH-001 / Phase F3.1 Dependency Management
Branch: feature/torchapi
Mapped tests: pytest --collect-only tests/torch/ -q
Artifacts: plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T193400Z/{dependency_update.md,pytest_collect.log}
Do Now:
- INTEGRATE-PYTORCH-001 Phase F3 — F3.1 @ plans/active/INTEGRATE-PYTORCH-001/phase_f_torch_mandatory.md (tests: pytest --collect-only tests/torch/ -q): add torch>=2.2 to packaging (setup.py) and document the environment/CI implications in dependency_update.md, then verify torch importability and attach pytest collection output.
If Blocked: Capture the failure (pip install log, pytest stack trace) in dependency_update.md, note the blocker in docs/fix_plan.md Attempts History, and stop before modifying any guard code.
Priorities & Rationale:
- plans/active/INTEGRATE-PYTORCH-001/phase_f_torch_mandatory.md:45 — F3.1 is the gating step before guard removal.
- plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T192500Z/migration_plan.md — Dependency gate (Section Phase F3.1) details tasks and validation.
- setup.py:1-60 — Current install_requires lacks PyTorch; needs update to enforce torch availability.
- docs/fix_plan.md:138-143 — Latest attempts close Phase F2 and set expectation that Phase F3 begins with dependency promotion.
How-To Map:
- Packaging: edit setup.py install_requires to include 'torch>=2.2', keeping alphabetical order if practical; record the diff in dependency_update.md.
- Environment check: `pip install -e .` followed by `python -c "import torch; print(torch.__version__)"`; log version string and install duration in dependency_update.md.
- Test probe: `pytest --collect-only tests/torch/ -q | tee plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T193400Z/pytest_collect.log` to ensure collection succeeds with torch present.
- Artifact write-up: Summarize changes, commands, and observations under plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T193400Z/dependency_update.md (include timestamps, command outputs, and next checks for CI).
Pitfalls To Avoid:
- Do not remove TORCH_AVAILABLE guards yet—F3.2 handles code changes.
- Avoid pinning torch to GPU-specific wheels; use a generic >=2.2 constraint per governance notes.
- Keep install_requires deterministic; no duplicate entries or trailing commas.
- Document any environment issues (CUDA availability, pip resolution) instead of patching ad hoc workarounds.
- Do not modify documentation/specs in this loop; defer to Phase F4.
- Skip global pip cache cleanup unless necessary; record any cleanup commands if used.
- Ensure pytest collection runs after reinstall so we catch import regressions immediately.
Pointers:
- plans/active/INTEGRATE-PYTORCH-001/phase_f_torch_mandatory.md:39
- plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T192500Z/migration_plan.md
- setup.py:15-55
- docs/fix_plan.md:138-143
Next Up: Phase F3.2 guard removal once torch dependency is enforced and validated.
