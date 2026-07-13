# Test Strategy Supplement (Optional)

Ordinary RED/GREEN steps, selectors, expected results, and regression checks
belong directly in the canonical [implementation plan
template](implementation_plan.md). Use this supplement only when the work
changes test architecture, CI tiers, expensive scientific validation, or an
environment/dependency capability contract.

Prefer adding the relevant section to the single plan at
`docs/plans/YYYY-MM-DD-<initiative>.md`. Create a companion strategy only for
genuinely multi-document work.

## Strategy delta

- **Implementation plan:** <repository-relative path>
- **Why a separate strategy is needed:** test architecture | CI tiers |
  expensive scientific validation | environment capability
- **Current behavior:** <existing suite/tier/environment contract>
- **Proposed change:** <new boundary and why it is needed>
- **Authority:** <testing guide, CI policy, scientific contract, or accepted
  design>
- **Out of scope:** <tests, platforms, and claims unaffected>

## Test boundary and environment matrix

| Layer / capability | Behavior or claim | Real dependency/path | Mock or fixture boundary | Environment / CI tier | Implementation-plan evidence |
|---|---|---|---|---|---|
| <unit/integration/scientific> | <observable claim> | <what runs for real> | <what is substituted and why> | <CPU/GPU/dependency/tier> | <task and verification-ladder entry that own the command and expected result> |

## Evidence rules

- Prefer real code paths. Mock only unavailable or nondeterministic external
  boundaries, not the internal behavior under test, and keep mocks faithful to
  the real interface.
- Pair mocks with the smallest feasible real integration or contract check when
  the mocked boundary is part of the acceptance claim.
- A skip is not a pass. Use an explicit capability check and reason; record
  which environment executes the skipped behavior for real. An unavailable
  required capability is a blocker, not an acceptable skip.
- Include negative evidence that can falsify the contract: for example invalid
  input rejection, missing dependency handling, stale artifact rejection, or a
  scientific control expected not to meet the target.
- For expensive scientific validation, define dataset identity, seeds,
  baseline/control, metric, tolerance, budget, provenance, and early stop/pivot
  criteria only to the extent required by the claim.
- Use project- or request-owned coverage thresholds when they exist; do not
  invent a generic percentage gate.

## Tier and failure policy

- **Fast/default tier:** <what runs, where, and maximum intended cost>
- **Integration/accelerated tier:** <trigger, environment, and ownership>
- **Scheduled/expensive tier:** <trigger, budget, and claim boundary>
- **Failure/skip routing:** <which result blocks, retries, quarantines, or needs
  a documented environment rerun>

## Acceptance

- [ ] Every changed test boundary has a real-vs-mock rationale.
- [ ] Required behavior executes in at least one named environment; skips are
  explicit and do not support success claims.
- [ ] Negative cases can distinguish enforcement from a vacuous pass.
- [ ] Each matrix row points to the implementation-plan task and verification
  entry that own its runnable command and expected result; neither is duplicated
  here.
- [ ] Expensive or scientific evidence is proportional to the claim and has a
  clear stop condition.
