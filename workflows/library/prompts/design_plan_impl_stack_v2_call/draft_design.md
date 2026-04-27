Read the `Consumed Artifacts` section first and treat it as the authoritative input list.
Read the consumed `brief` artifact before acting.

Draft a principled design / ADR from the brief.

If `docs/templates/design_template.md` is present, read it and use it for document structure. Omit irrelevant optional sections rather than padding the design. If the work is small, keep the document short while still making the scope, key decisions, verification, and handoff clear.

Focus on:
- the problem and scope
- the core contracts and invariants
- the implementation shape, including module/component boundaries when the work affects nontrivial tooling, validators, generators, services, workflow changes, stable APIs, data contracts, durable artifacts, or long-lived project structure
- whether internal refactoring or debt paydown is required before feature work
- explicit non-goals and sequencing constraints

For the output contract's `design_path`, read the path recorded in that file and write the design document to that current-checkout-relative path. Leave the `design_path` file containing only the path.
