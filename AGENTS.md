# AGENTS.md


---

## 0. Rules / suggestions
- use the tmux skill when launching long-running commands. When using tmux, activate the ptycho311 conda env or point pythonpath to its executable

### Long-Run Sync Guardrail
- For long-running commands, track and wait on the exact launched PID (`cmd ... & pid=$!; wait "$pid"`).
- Do not use broad `pgrep -f` polling loops as the primary completion check.
  Command-string polling can match the wrapper shell or watcher itself and
  hang after the real child process exits.
- If tailing logs while a command runs, tail in a separate process, wait on the
  command PID, then stop the tail process and propagate the command exit code.
- Do not launch a duplicate run if another process is already writing to the same `--output-root`.
- Consider a run complete only when:
  1. the tracked PID exits with code `0`, and
  2. required output artifacts for that step exist and are freshly written.

## 1. ⚙️ Identity & Workflow Guardrails

- **Plans & artifacts:** Keep evidence lean. Store plans under `docs/plans/` (default single-file plan: `docs/plans/YYYY-MM-DD-<initiative>.md`). If an initiative needs a folder, use `docs/plans/<initiative>/summary.md` for loop summaries. Store bulky artifacts outside the repo (or under a git‑ignored `.artifacts/` folder) and link to them from the plan/ledger.
- **Authority stack:** If instructions conflict, prefer SPECs (specs/ for external interop contracts; docs/specs/ for internal spec-ptycho-* shards), then project documentation, then prompt files. Internal model memories must defer to the repository.

---

## 2. ⚖️ Fundamental Directives

1. **Documentation is authoritative.** Start from `docs/index.md`. Never rely on unstated assumptions if a spec or guide disagrees with cached knowledge.
2. **Honor specifications and data contracts.** `specs/data_contracts.md` and `specs/ptychodus_api_spec.md` define external behavior; implementation must not diverge without an approved plan.
3. **Treat core physics/model code as stable.** Do not modify `ptycho/model.py`, `ptycho/diffsim.py`, or `ptycho/tf_helper.py` unless the active plan explicitly authorizes it.
4. **Respect the PyTorch policy.** PyTorch (torch ≥ 2.2) is mandatory (POLICY-001). PyTorch workflows must still run `update_legacy_dict(params.cfg, config)` before touching legacy modules; see `docs/workflows/pytorch.md`.
5. **Testing proof is mandatory.** Any task involving tests must provide fresh passing `pytest` evidence as described in `docs/TESTING_GUIDE.md`.
6. **Interpreter policy.** Obey PYTHON-ENV-001 in `docs/DEVELOPER_GUIDE.md` (invoke Python via PATH `python`; avoid repository-specific interpreter wrappers).
7. Do not create worktrees, especially not when executing plans or implementing features.
8. **Worktree submodule policy.** In existing worktrees (especially `git bisect` runs), initialize submodules before tests (`git submodule update --init --recursive`) and verify required submodule files exist (e.g., `ptycho/FRC/*`), otherwise test results are invalid.

---

## 3. 📚 Reference Map

- **Documentation hub:** `docs/index.md` – complete map of guides, specs, and workflows.
- **Workflow runbooks:** `docs/workflows/agent_orchestration_backlog_loop.md` and `docs/backlog/index.md` – backlog execution, queue state, and plan routing.
- **Developer guide:** `docs/DEVELOPER_GUIDE.md` – architecture, anti-patterns, and TDD methodology.
- **Data generation:** `docs/DATA_GENERATION_GUIDE.md` – grid vs nongrid simulation pipelines, parameter mappings.
- **Testing references:** `docs/TESTING_GUIDE.md` and `docs/development/TEST_SUITE_INDEX.md` – authoritative test commands and selectors.
- **PyTorch workflows:** `docs/workflows/pytorch.md` – configuration and execution rules for the PyTorch backend.
- **Knowledge base:** `docs/findings.md` – known issues, conventions, and policies to consult when relevant.

Use the index to locate any additional document cited by the active plan.

---

## 4. 🛠 Where to Find Troubleshooting & Commands

- **Params.cfg / shape mismatch issues:** Follow `docs/debugging/QUICK_REFERENCE_PARAMS.md` and `docs/debugging/TROUBLESHOOTING.md`.
- **Command library (training, inference, tests):** Consult `docs/COMMANDS_REFERENCE.md` for relevant CLI recipes.
- **Known bugs:** See `docs/bugs/` directory for documented bugs and workarounds (e.g., `XLA_INFERENCE_BUG.md` for PINN inference issues).
