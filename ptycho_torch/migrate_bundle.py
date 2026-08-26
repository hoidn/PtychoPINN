"""Offline era-detecting bundle migration entry point.

Migrate a pre-JSON or metadata-free ``wts.h5.zip`` bundle into the current
``torch-artifact-v5`` era:

    python -m ptycho_torch.migrate_bundle SOURCE_DIR OUT_DIR

``SOURCE_DIR`` and ``OUT_DIR`` each hold a ``wts.h5.zip`` archive.  The legacy
dill-decoding implementation lives in ``scripts/migrate_legacy_bundle.py`` so
that ``ptycho_torch`` itself stays dill-free (Ratchet B); this module is the
canonical offline CLI.
"""

import argparse


def migrate_bundle(source_dir, out_dir):
    """Migrate ``<source_dir>/wts.h5.zip`` and write ``<out_dir>/wts.h5.zip``.

    Delegates to the dill-hosting legacy implementation in
    ``scripts/migrate_legacy_bundle.py``.
    """
    from scripts.migrate_legacy_bundle import migrate_bundle as _migrate

    return _migrate(source_dir, out_dir)


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="python -m ptycho_torch.migrate_bundle",
        description=(
            "Migrate a pre-JSON or metadata-free wts.h5.zip to the versioned "
            "JSON manifest + sealed-identity era."
        ),
    )
    parser.add_argument(
        "source_dir",
        help="Directory containing the source wts.h5.zip archive.",
    )
    parser.add_argument(
        "out_dir",
        help="Directory that will receive the migrated wts.h5.zip archive.",
    )
    args = parser.parse_args(argv)

    out_zip = migrate_bundle(args.source_dir, args.out_dir)
    print(f"Migrated bundle written to {out_zip}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
