# CLAUDE.md

This document is the **constitution** for the Claude AI agents working on the PtychoPINN repository. It defines identity, non-negotiable directives, and the canonical sources of truth. Operational details, workflows, and reporting formats live in `prompts/main.md` (Ralph/engineer) and `prompts/supervisor.md` (Galph/supervisor).

---

## 1. ⚙️ Identity & Workflow Guardrails

- **Two-agent loop:** Operate within the Supervisor/Engineer loop managed by `prompts/supervisor.md` (planning) and `prompts/main.md` (implementation). Those prompts own required reading, stall-autonomy, dwell, and output formatting rules.
- **Ledger discipline:** `docs/fix_plan.md` is the master task ledger. Every loop must read the current focus, execute exactly one `Do Now`, and append an Attempts History entry with artifact links before exiting.
- **Plans & artifacts:** Each loop produces a timestamped reports directory under `plans/active/<initiative>/reports/<ISO8601Z>/` (as mandated in the prompts). All reproducible evidence lives there.
- **Authority stack:** If instructions conflict, prefer SPECs (`specs/`), then project documentation, then prompt files. Internal model memories must defer to the repository.

---

## 2. ⚖️ Fundamental Directives

1. **Documentation is authoritative.** Start from `docs/index.md`. Never rely on unstated assumptions if a spec or guide disagrees with cached knowledge.
2. **Consult the knowledge base first.** Read `docs/findings.md` before debugging or implementing fixes. Follow any applicable Finding IDs verbatim (e.g., POLICY-001 for PyTorch requirements).
3. **Honor specifications and data contracts.** `specs/data_contracts.md` and `specs/ptychodus_api_spec.md` define external behavior; implementation must not diverge without an approved plan.
4. **Follow the standard debugging workflow.** Execute the steps in `docs/debugging/debugging.md` for every new defect, documenting progress in `docs/fix_plan.md`.
5. **Preserve artifact hygiene.** Every log, plot, or report that explains a loop belongs in the initiative’s `plans/active/.../reports/<timestamp>/` directory. Temporary scratch data stays under `tmp/` and is deleted before committing.
6. **Treat core physics/model code as stable.** Do not modify `ptycho/model.py`, `ptycho/diffsim.py`, or `ptycho/tf_helper.py` unless the active plan explicitly authorizes it.
7. **Respect the PyTorch policy.** PyTorch (torch ≥ 2.2) is mandatory (POLICY-001). PyTorch workflows must still run `update_legacy_dict(params.cfg, config)` before touching legacy modules; see `docs/workflows/pytorch.md`.
8. **Testing proof is mandatory.** Any task involving tests must provide passing `pytest` evidence and archived logs as described in `prompts/main.md` and `docs/TESTING_GUIDE.md`.
9. **Plan test infrastructure up front.** Before Phase B or any implementation that adds/changes tests, capture the strategy using `plans/templates/test_strategy_template.md` (or the initiative’s `test_strategy.md`) and link it from `docs/fix_plan.md`.

---

## 3. 📚 Required Reference Map

- **Documentation hub:** `docs/index.md` – complete map of guides, specs, and workflows.
- **Workflow guide:** `docs/INITIATIVE_WORKFLOW_GUIDE.md` – details on initiative planning, artifact storage, and ledger updates.
- **Developer guide:** `docs/DEVELOPER_GUIDE.md` – architecture, anti-patterns, and TDD methodology.
- **Testing references:** `docs/TESTING_GUIDE.md` and `docs/development/TEST_SUITE_INDEX.md` – authoritative test commands and selectors.
- **PyTorch workflows:** `docs/workflows/pytorch.md` – configuration and execution rules for the PyTorch backend.
- **Knowledge base:** `docs/findings.md` – mandatory pre-read for known issues, conventions, and policies.

Use the index to locate any additional document cited by `prompts/main.md`, `prompts/supervisor.md`, or the active plan.

---

## 4. 🛠 Where to Find Troubleshooting & Commands

- **Params.cfg / shape mismatch issues:** Follow `docs/debugging/QUICK_REFERENCE_PARAMS.md` and `docs/debugging/TROUBLESHOOTING.md`. Do **not** rely on stale snippets in this constitution.
- **Command library (git, training, inference, tests):** Use `docs/COMMANDS_REFERENCE.md` for all CLI recipes. The prompts enforce running tests via `pytest` selectors; align with that doc and archive logs per their instructions.
- **Git setup & hygiene:** See `prompts/git_setup_agent.md` and `prompts/git_hygiene.md` for automation-safe Git workflows.

If a command or troubleshooting step is missing from those references, update the canonical document first; CLAUDE.md should only point to authoritative sources, not duplicate their content.
