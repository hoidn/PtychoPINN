# CLAUDE.md


---

## 0. Rules / suggestions 
- use the tmux skill when launching long-running commands

## 1. ⚙️ Identity & Workflow Guardrails

- **Plans & artifacts:** Keep evidence lean. Store plans under `docs/plans/` (default single-file plan: `docs/plans/YYYY-MM-DD-<initiative>.md`). If an initiative needs a folder, use `docs/plans/<initiative>/summary.md` for loop summaries. Store bulky artifacts outside the repo (or under a git‑ignored `.artifacts/` folder) and link to them from the plan/ledger.
- **Authority stack:** If instructions conflict, prefer SPECs (`specs/`), then project documentation, then prompt files. Internal model memories must defer to the repository.

---

## 2. ⚖️ Fundamental Directives

1. **Documentation is authoritative.** Start from `docs/index.md`. Never rely on unstated assumptions if a spec or guide disagrees with cached knowledge.
2. **Consult the knowledge base first.** Read `docs/findings.md` before debugging or implementing fixes. Follow any applicable Finding IDs verbatim (e.g., POLICY-001 for PyTorch requirements).
3. **Honor specifications and data contracts.** `specs/data_contracts.md` and `specs/ptychodus_api_spec.md` define external behavior; implementation must not diverge without an approved plan.
5. **Preserve artifact hygiene.** Keep minimal, human-readable summaries in `docs/plans/` (single-file plan by default, or `docs/plans/<initiative>/summary.md` when a folder is used). Store bulky artifacts outside the repo (or under a git-ignored `.artifacts/`) and link to them. Temporary scratch data stays under `tmp/` and is deleted before committing.
6. **Treat core physics/model code as stable.** Do not modify `ptycho/model.py`, `ptycho/diffsim.py`, or `ptycho/tf_helper.py` unless the active plan explicitly authorizes it.
7. **Respect the PyTorch policy.** PyTorch (torch ≥ 2.2) is mandatory (POLICY-001). PyTorch workflows must still run `update_legacy_dict(params.cfg, config)` before touching legacy modules; see `docs/workflows/pytorch.md`.
8. **Testing proof is mandatory.** Any task involving tests must provide passing `pytest` evidence and archived logs as described in `prompts/main.md` and `docs/TESTING_GUIDE.md`.
11. **Interpreter policy.** Obey PYTHON-ENV-001 in `docs/DEVELOPER_GUIDE.md` (invoke Python via PATH `python`; avoid repository-specific interpreter wrappers).
12. **Worktree submodule policy.** In new worktrees (especially `git bisect` runs), initialize submodules before tests (`git submodule update --init --recursive`) and verify required submodule files exist (e.g., `ptycho/FRC/*`), otherwise test results are invalid.

---

## 3. 📚 Required Reference Map

- **Documentation hub:** `docs/index.md` – complete map of guides, specs, and workflows.
- **Workflow guide:** `docs/INITIATIVE_WORKFLOW_GUIDE.md` – details on initiative planning, artifact storage, and ledger updates.
- **Developer guide:** `docs/DEVELOPER_GUIDE.md` – architecture, anti-patterns, and TDD methodology.
- **Data generation:** `docs/DATA_GENERATION_GUIDE.md` – grid vs nongrid simulation pipelines, parameter mappings.
- **Testing references:** `docs/TESTING_GUIDE.md` and `docs/development/TEST_SUITE_INDEX.md` – authoritative test commands and selectors.
- **PyTorch workflows:** `docs/workflows/pytorch.md` – configuration and execution rules for the PyTorch backend.
- **Knowledge base:** `docs/findings.md` – mandatory pre-read for known issues, conventions, and policies.

Use the index to locate any additional document cited by `prompts/main.md`, `prompts/supervisor.md`, or the active plan.

---

## 4. 🛠 Where to Find Troubleshooting & Commands

- **Params.cfg / shape mismatch issues:** Follow `docs/debugging/QUICK_REFERENCE_PARAMS.md` and `docs/debugging/TROUBLESHOOTING.md`. Do **not** rely on stale snippets in this constitution.
- **Command library (git, training, inference, tests):** Use `docs/COMMANDS_REFERENCE.md` for all CLI recipes. The prompts enforce running tests via `pytest` selectors; align with that doc and archive logs per their instructions.
- **Git setup & hygiene:** See `prompts/git_setup_agent.md` and `prompts/git_hygiene.md` for automation-safe Git workflows.
- **Known bugs:** See `docs/bugs/` directory for documented bugs and workarounds (e.g., `XLA_INFERENCE_BUG.md` for PINN inference issues).

- Remove “evidence‑only” git exceptions. Always perform normal pull/rebase hygiene. Do not commit bulky artifacts; store them externally or under `.artifacts/` and link from the plan/ledger.

If a command or troubleshooting step is missing from those references, update the canonical document first; CLAUDE.md should only point to authoritative sources, not duplicate their content.

---

## 5. 🧾 Plan-Update Protocol


### Required Steps
1. Re-read `docs/index.md`, then open every referenced document that applies to the pending change (plan file, specs, workflow guides, etc.) so `documents_read` matches reality.
4. Then make the updates

### Simulation Checklist
- **Before editing:** ensure `documents_read` mirrors every file you opened after consulting `docs/index.md`.
- **After editing:** confirm the XML, plan diff, and ledger entry all cite the same focus ID and plan path.
- **If blocked:** set `<status>blocked</status>` and describe the blocker under `<impacts>`; the next loop repeats the XML with updated context.
