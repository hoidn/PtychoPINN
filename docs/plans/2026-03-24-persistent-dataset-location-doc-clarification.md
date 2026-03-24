# Persistent Dataset Location Documentation Clarification

## Goal

Clarify, in discoverable project docs, that datasets which must persist long term do not belong under `outputs/`, because `outputs/` is cleanup-prone and intended for ephemeral run artifacts.

## Scope

- Update the main data-management guidance to distinguish durable datasets from ephemeral outputs.
- Update the data-generation guidance so the current `lines_256` pair is not documented as if `outputs/` were the correct long-term home.
- Update the `lines_256` study note to flag its current `outputs/` location as a compatibility location rather than the desired persistent convention.
- Update `docs/index.md` and `docs/studies/index.md` so the rule is discoverable from the documentation hub.

## Non-Goals

- Moving the actual `lines_256` dataset files.
- Changing workflow YAML or runbook paths.
- Redesigning the general storage layout beyond clarifying the policy.
