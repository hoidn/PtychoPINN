# Workflow Idea Queue

This queue is for workflow-consumable experiment ideas, not general implementation
backlog work.

For `lines_256` v2, the controller checks `active/` before free-form proposal
selection. If one or more queue items are present, it selects the first
Markdown file in lexicographic order and treats it as the highest-priority idea
to try next.

## Layout

- `active/`: eligible queue items
- `accepted/`: items that were tried and ended in `KEEP`
- `discarded/`: items that were tried and ended in `DISCARD`
- `blocked/`: items that ended in `BLOCKED`
- `crashed/`: items that ended in `CRASH`
- `timed_out/`: items that ended in `TIMEOUT`
- `templates/`: authoring templates

## Queue Rules

- Write one idea per Markdown file.
- Use simple lexicographic filename ordering; earlier names run first.
- The workflow does **not** remove an item from `active/` when it starts
  working on it.
- The controller moves the file only after a terminal result is recorded.
- Queue items are guidance, not rigid schemas. Keep them readable and
  hypothesis-focused.

## Suggested Naming

Use a sortable prefix such as:

`YYYY-MM-DD-<short-hypothesis>.md`

Example:

`2026-03-31-hybrid-resnet-32x32-cnn-detour.md`
