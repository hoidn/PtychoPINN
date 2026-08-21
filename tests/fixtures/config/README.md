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

The historical payloads record `rect_s1s2_init="data"`, a retired mode that
current code intentionally rejects. To preserve these exact pre-migration
bytes while keeping the generator executable, it first constructs and validates
a current model with `rect_s1s2_init="ones"`, encodes the serialized payload,
and changes only that encoded field back to the historical `"data"` spelling.
Loading these fixtures therefore requires historical code or retraining. Tests
of the tensor tag use a copied payload changed to `"ones"`; the frozen fixture
itself remains byte-for-byte unchanged.

| Fixture | SHA-256 |
| --- | --- |
| `pydantic_pre_migration_portable_v1.json` | `3851004d18c298bf7f3cc3e01883f8a4fc50c9443f6c64750df2c840b33af9a4` |
| `pydantic_pre_migration_portable_v2.json` | `4234ef6494782f9a9dff926c64704ad9b2484cd8bb963b1029c81813bc95f8ed` |
| `pydantic_pre_migration_tensor_mask.json` | `94dad3273d553ec8a6c97984de46235b33a4ec1f19cf0a5372130193fdd4daa7` |
