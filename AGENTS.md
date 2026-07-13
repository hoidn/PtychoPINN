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
- **Repository authority stack:** Prefer normative specifications (`specs/` for external interop contracts and `docs/specs/` for internal contracts), then the explicitly approved active design or plan within its stated scope, then project guides, then prompt files. Current contract text outranks summaries, status labels, and historical execution records. Internal model memories defer to the repository.

### Scope, Evidence, And Procedural Authority
- Authorized files and paths grant mutation permission only. They do not create deliverables, test ownership, whole-file verification obligations, documentation work, or future-task requirements.
- Completion gates come only from the current user request, applicable governing specifications and safety policies, and explicitly approved acceptance criteria in the active plan. Todos, reports, reviews, summaries, ledgers, prior commands, and historical runs may record or decompose those gates, but MUST NOT broaden or invent them.
- A controller-added procedure is a bounded task-local tactic, not a governing rule. It expires when its named risk is resolved or the task ends. Repetition, successful use, recording, citation, copying, or literal restatement does not promote it into policy; persistent requirements require explicit adoption by a higher-authority source.
- Checks create evidence; evidence does not create requirements. An observation may prove that an existing requirement, affected contract, or sourced safety invariant is violated, regardless of whether the check was required. Otherwise, a supplemental check is not independently a completion gate.

---

## 2. ⚖️ Fundamental Directives

1. **Current documentation contracts are authoritative.** Start from `docs/index.md` to locate applicable specifications, designs, plans, and guides; the index is a routing surface, not an independent semantic authority. Never rely on cached knowledge when current governing text is available.
2. **Honor specifications and data contracts.** `specs/data_contracts.md` and `specs/ptychodus_api_spec.md` define external behavior; implementation must not diverge without an approved plan.
3. **Treat core physics/model code as stable.** Do not modify `ptycho/model.py`, `ptycho/diffsim.py`, or `ptycho/tf_helper.py` unless the active plan explicitly authorizes it.
4. **Respect the PyTorch policy.** PyTorch (torch ≥ 2.2) is mandatory (POLICY-001). PyTorch workflows must still run `update_legacy_dict(params.cfg, config)` before touching legacy modules; see `docs/workflows/pytorch.md`.
5. **Behavioral claims require claim-matched evidence.** Run every exact command or selector required by current authority. Where none is named, use the smallest fresh evidence set sufficient to support—and capable of falsifying—the complete current acceptance claim and its affected governing invariants. A touched or authorized test file does not make every test in that file required. Classify supplemental failures against current requirements before treating them as blockers, and archive logs only where the active plan or `docs/TESTING_GUIDE.md` requires durable evidence.
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
- **Command library (training, inference, tests):** Consult `docs/COMMANDS_REFERENCE.md` for applicable CLI invocation mechanics. A recipe does not expand the current acceptance contract.
- **Known bugs:** See `docs/bugs/` directory for documented bugs and workarounds (e.g., `XLA_INFERENCE_BUG.md` for PINN inference issues).
