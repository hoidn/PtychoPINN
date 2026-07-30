# Pre-Pydantic Torch configuration fixtures

These fixtures freeze the Torch configuration wire payloads produced before
the Pydantic/string-enum migration. They were captured from source revision
`99efda11155119161d371d5d0e5ec7c33a720594`.

From the repository root, reproduce all three payloads on standard output with:

```bash
python tests/fixtures/config/generate_pre_migration_fixtures.py --stdout
```

Each fixture is compact canonical JSON followed by one newline. The SHA-256
digests cover that exact UTF-8 byte stream, including the final newline.

| Fixture | SHA-256 |
| --- | --- |
| `pydantic_pre_migration_portable_v1.json` | `3851004d18c298bf7f3cc3e01883f8a4fc50c9443f6c64750df2c840b33af9a4` |
| `pydantic_pre_migration_portable_v2.json` | `fc034f1e5c2ef882ac76988bf4a2a9cf5d0d126aedb4c0df099d65ddc673547e` |
| `pydantic_pre_migration_tensor_mask.json` | `8a4ca0741092187c93f43965115a28ae0d81cc54bebde7ce0942a1ca2fc86c14` |
