# CLAUDE.md

This document is the **constitution** for the Claude AI agents working on the PtychoPINN repository. It defines identity, non-negotiable directives, and the canonical sources of truth. 

---

## 1. ⚙️ Identity & Workflow Guardrails

- **Authority stack:** If instructions conflict, prefer SPECs (`specs/`), then project documentation, then prompt files. Internal model memories must defer to the repository.

---

## 2. ⚖️ Fundamental Directives

1. **Documentation is authoritative.** Start from `docs/index.md`. Never rely on unstated assumptions if a spec or guide disagrees with cached knowledge.
2. **Consult the knowledge base first.** Read `docs/findings.md` before debugging or implementing fixes. Follow any applicable Finding IDs verbatim (e.g., POLICY-001 for PyTorch requirements).
3. **Honor specifications and data contracts.** `specs/data_contracts.md` and `specs/ptychodus_api_spec.md` define external behavior; implementation must not diverge without an approved plan.
4. **Follow the standard debugging workflow.** Execute the steps in `docs/debugging/debugging.md` for every new defect, documenting progress in `docs/fix_plan.md`.
5. **Preserve artifact hygiene.** Keep minimal, human‑readable summaries in the initiative’s `plans/active/<initiative>/summary.md`. Store bulky artifacts outside the repo (or under a git‑ignored `.artifacts/`) and link to them. Temporary scratch data stays under `tmp/` and is deleted before committing.
6. **Treat core physics/model code as stable.** Do not modify `ptycho/model.py`, `ptycho/diffsim.py`, or `ptycho/tf_helper.py` unless the active plan explicitly authorizes it.
7. **Respect the PyTorch policy.** PyTorch (torch ≥ 2.2) is mandatory (POLICY-001). PyTorch workflows must still run `update_legacy_dict(params.cfg, config)` before touching legacy modules; see `docs/workflows/pytorch.md`.
8. **Testing proof is mandatory.** Any task involving tests must provide passing `pytest` evidence and archived logs as described in `prompts/main.md` and `docs/TESTING_GUIDE.md`.
9. **Plan test infrastructure up front.** Before Phase B or any implementation that adds/changes tests, capture the strategy using `plans/templates/test_strategy_template.md` (or the initiative’s `test_strategy.md`) and link it from `docs/fix_plan.md`.
11. **Interpreter policy.** Obey PYTHON-ENV-001 in `docs/DEVELOPER_GUIDE.md` (invoke Python via PATH `python`; avoid repository-specific interpreter wrappers).
12. NEVER train models on CPU. Resolve OOM errors by reducing input sizes (or, when relevant, fixing underlying bug(s))
---

## 3. 📚 Required Reference Map

- **Documentation hub:** `docs/index.md` – complete map of guides, specs, and workflows.
- **Developer guide:** `docs/DEVELOPER_GUIDE.md` – architecture, anti-patterns, and TDD methodology.
- **Data generation:** `docs/DATA_GENERATION_GUIDE.md` – grid vs nongrid simulation pipelines, parameter mappings.
- **Testing references:** `docs/TESTING_GUIDE.md` and `docs/development/TEST_SUITE_INDEX.md` – authoritative test commands and selectors.
- **PyTorch workflows:** `docs/workflows/pytorch.md` – configuration and execution rules for the PyTorch backend.
- **Knowledge base:** `docs/findings.md` – mandatory pre-read for known issues, conventions, and policies.

Use the index to locate any additional document cited by the active plan.

---

## 4. 🛠 Where to Find Troubleshooting & Commands

- **Params.cfg / shape mismatch issues:** Follow `docs/debugging/QUICK_REFERENCE_PARAMS.md` and `docs/debugging/TROUBLESHOOTING.md`. Do **not** rely on stale snippets in this constitution.
- **Command library (git, training, inference, tests):** Use `docs/COMMANDS_REFERENCE.md` for all CLI recipes. The prompts enforce running tests via `pytest` selectors; align with that doc and archive logs per their instructions.
- **Git setup & hygiene:** See `prompts/git_setup_agent.md` and `prompts/git_hygiene.md` for automation-safe Git workflows.

If a command or troubleshooting step is missing from those references, update the canonical document first; CLAUDE.md should only point to authoritative sources, not duplicate their content.

---

## 5. 🧾 Plan-Update Protocol


### Required Steps
1. Re-read `docs/index.md`, then open every referenced document that applies to the pending change (plan file, specs, workflow guides, etc.) so `documents_read` matches reality.
4. Then make the updates

### Simulation Checklist
- **Before editing:** ensure `documents_read` mirrors every file you opened after consulting `docs/index.md`.
- **If blocked:** set `<status>blocked</status>` and describe the blocker under `<impacts>`; the next loop repeats the XML with updated context.
