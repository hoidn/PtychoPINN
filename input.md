Summary: Rewrite pytest skip logic to enforce torch-required policy and capture skip-behavior evidence.
Mode: Parity
Focus: INTEGRATE-PYTORCH-001 / Phase F3.3 Skip Logic Rewrite
Branch: feature/torchapi
Mapped tests: pytest tests/torch/ -vv; PYTHONPATH=$PWD/tmp/no_torch_stub python -m pytest tests/ -q
Artifacts: plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T195624Z/{skip_rewrite_summary.md,pytest_torch.log,pytest_no_torch.log}
Do Now:
- INTEGRATE-PYTORCH-001 Phase F3 — F3.3 @ plans/active/INTEGRATE-PYTORCH-001/phase_f_torch_mandatory.md:46 (tests: pytest tests/torch/ -vv): remove the torch whitelist in tests/conftest.py, clear TORCH_AVAILABLE guards from the torch test modules called out in test_skip_audit.md §10, decide on tests/test_pytorch_tf_wrapper.py relevance, then run the torch suite and tee output to pytest_torch.log.
- INTEGRATE-PYTORCH-001 Phase F3 — F3.3 validation (no torch) @ plans/active/INTEGRATE-PYTORCH-001/phase_f_torch_mandatory.md:46 (tests: PYTHONPATH=$PWD/tmp/no_torch_stub python -m pytest tests/ -q): create a temporary tmp/no_torch_stub/torch/__init__.py that raises ImportError to simulate torch absence, rerun pytest to verify skip counts, capture output to pytest_no_torch.log, and delete the stub afterward.
- INTEGRATE-PYTORCH-001 Phase F3 — F3.3 docs @ plans/active/INTEGRATE-PYTORCH-001/phase_f_torch_mandatory.md:46 (tests: none): summarize edits and validation in skip_rewrite_summary.md, update phase_f_torch_mandatory.md (mark F3.3 complete with selectors), and log Attempt #71 in docs/fix_plan.md with artifact links.
If Blocked: Stop before partial conftest rewrites; restore the original skip logic, stash error traces in skip_rewrite_summary.md, and document the blocker in docs/fix_plan.md Attempt notes for Phase F3.3.
Priorities & Rationale:
- plans/active/INTEGRATE-PYTORCH-001/phase_f_torch_mandatory.md:42-47 — Phase F checklist shows F3.3 as the next gating task.
- plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T192500Z/test_skip_audit.md:1-120 — Inventory of skip mechanisms and explicit F3.3 implementation checklist.
- docs/findings.md — CONFIG-001 / DATA-001 guardrails still apply when editing tests touching config bridge and data pipeline.
- specs/ptychodus_api_spec.md:192-275 — Workflow contract references ensuring backend dispatch remains torch-first after skip cleanup.
- plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T193753Z/guard_removal_summary.md confirms production modules now require torch; tests must match this policy before Phase F3.4.
How-To Map:
- Edit tests/conftest.py to drop TORCH_OPTIONAL_MODULES and consolidate skip logic to a simple torch-available check; remove the unused torch_available fixture per test_skip_audit.md Section 10.
- In tests/torch/test_data_pipeline.py and tests/torch/test_tf_helper.py, delete try/except ImportError guards and @unittest.skipUnless decorators, updating assertions to unconditionally expect torch tensors.
- Review tests/test_pytorch_tf_wrapper.py; either remove torch guards to keep the test or delete the file if obsolete, documenting the decision in skip_rewrite_summary.md.
- Run `pytest tests/torch/ -vv | tee plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T195624Z/pytest_torch.log` after code edits.
- Simulate torch absence by creating the stub:
  ```bash
  mkdir -p tmp/no_torch_stub/torch
  cat <<'STUB' > tmp/no_torch_stub/torch/__init__.py
  raise ImportError("torch intentionally disabled for Phase F3.3 validation")
  STUB
  PYTHONPATH=$PWD/tmp/no_torch_stub python -m pytest tests/ -q | tee plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T195624Z/pytest_no_torch.log
  rm -rf tmp/no_torch_stub
  ```
- Capture a narrative of edits, skip counts, and outstanding issues in skip_rewrite_summary.md (note both torch-present and torch-missing behaviors).
Pitfalls To Avoid:
- Do not leave tmp/no_torch_stub in the tree; it must be deleted before finishing the loop.
- Avoid reintroducing TORCH_AVAILABLE flags or conditional paths in production modules—changes should stay within tests/conftest.py and the torch test files.
- Keep new or updated tests in native pytest style; do not add unittest.TestCase wrappers around pytest fixtures.
- Ensure pytest_torch.log records the exact command output (include the command in the log header for clarity).
- When simulating torch absence, confirm the command fails via skips—not by importing the stubbed module successfully.
- Do not change CLAUDE.md or docs/findings.md yet; documentation updates belong to Phase F4.
- Re-run formatting hooks if large deletions leave trailing whitespace in tests.
- Update docs/fix_plan.md only once with consolidated Attempt #71 details to avoid duplicate entries.
Pointers:
- plans/active/INTEGRATE-PYTORCH-001/phase_f_torch_mandatory.md:42-47
- plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T192500Z/test_skip_audit.md:1-220
- tests/conftest.py:1-120
- tests/torch/test_data_pipeline.py:1-320
- tests/torch/test_tf_helper.py:1-180
- tests/test_pytorch_tf_wrapper.py:1-160
Next Up: Phase F3.4 regression verification once skip logic succeeds.
