# Pre-Pydantic internal Torch configuration fixtures

These fixtures freeze the internal Torch configuration wire payloads produced
before the simulation Pydantic boundary port. They were captured from source
revision `f762bd27bccca3f9dfe9ecfad500af9589cb7777`.

From the repository root, reproduce all three payloads on standard output with:

```bash
python tests/fixtures/config/generate_pre_migration_fixtures.py --stdout
```

Each fixture is compact canonical JSON followed by one newline. The SHA-256
digests cover that exact UTF-8 byte stream, including the final newline.

| Fixture | SHA-256 |
| --- | --- |
| `pydantic_pre_migration_torch_artifact_v1.json` | `b0aeda235ad66177cabc88739346e4cd606689723f7981dc0ecaa7d376835910` |
| `pydantic_pre_migration_torch_artifact_v2.json` | `b665b094c6195e7b39e6b9ec46c3eb69933cc82f6597a8e2018666fe3b801839` |
| `pydantic_pre_migration_torch_tensor_mask.json` | `8bc8aa3430f0468ab8485faa1dd15b6c400a2d2750902228baeb8de3262b1303` |
