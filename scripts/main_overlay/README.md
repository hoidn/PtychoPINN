# Publishing the local branch to main

`main` is downstream of `fno-stable-local`. Publish a filtered source snapshot;
different commit hashes do not make downstream copies independent API changes.

## Rules

- [Exclusions](main_overlay_exclude.txt) remove private architectures, their
  study dependencies, and internal documentation. Code and doc pruning are
  both part of this inventory.
- [Content patch](main_overlay_patches/public-surface.patch) removes private
  entries from shared files and retains public-facing documentation.
- [Gitlink exclusions](main_overlay_gitlink_exclude.txt) omit private tooling
  and local agent guidance.
- [Grafts](main_overlay_graft.txt) preserve target-owned CI, public-only assets,
  and this publishing machinery. They are read from the target, not the source.
- [Builder](build_main_overlay.py) checks patch anchors, excluded-family
  references, dangling imports, imports, and exact output-tree inventory.

The current transform is anchored to source `d78ccb5c6`. The baseline comparison
used local `0f0337b38` and downstream `4c134c79b`; the source already contained
main's direct training/reconstruction APIs. Refresh patches when anchors drift.
Never restore excluded runtime paths merely to make a stale patch apply.

Main retains six files under `docs/`: configuration, commands, normalization,
FLY64 data, PyTorch workflows, and the custom-architecture guide. It also retains
selected READMEs and external contracts. The builder's prose grep exemptions
are not permission to publish additional internal documentation.

Documentation pruning originated in main commits `c797f8e0e` and `f50ab2bbe`.
The `refactor` branch used a different retained inventory (`c1514584b` and
`564a466b4`); this transform targets main, not refactor.

## Port and verify

In an isolated clone, initialize submodules and verify `ptycho/FRC/` exists.
Use the target's tool and rules with its tree as the graft source:

```bash
python scripts/main_overlay/build_main_overlay.py SOURCE_SHA --graft-from main
```

Keep scratch storage outside any Git checkout. If `TMPDIR` is inside a
checkout, pass an external `--scratch` directory; Git may otherwise skip
patch paths relative to the enclosing repository. The output gates catch this.

Inspect the resulting source/target diff, then run `bash ci/run_ci_tests.sh -n 8
--dist loadfile` and relevant serial CUDA train/reload/inference tests. Check
that strict older-bundle loading still works. Commit the verified tree on top
of main without merging private source ancestry. Leave the local development
branch unchanged; pushing is a separate action.
