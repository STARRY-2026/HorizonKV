# Internal HERMES backend

`qwenvl_hermes.py`, `abstract_hermes.py`, and `reindex_3d.py` were copied from
the local HorizonKV `hermes/inference/` implementation on 2026-10-23.
Only the three imports in `qwenvl_hermes.py` were changed to package-relative
imports so the algorithm loads from `kvbench.hermes_backend` without importing
the external `inference` package. No model, scoring or cache logic was edited.
The bounded runner records SHA-256 of each copied runtime file in its protocol.
`legacy_runner_source_20261023.txt` is the byte-identical pre-chunk-audit
adapter source kept solely to re-audit earlier HERMES results; it is not run.

Original SHA-256 values:

| Source | SHA-256 |
|---|---|
| `qwenvl_hermes.py` | `29327fca48ecd395cc2aaff9945671c3d767bacbc1b782229d03b9953b202c51` |
| `abstract_hermes.py` | `cbf62ad9f6dada32db598d7c37c27f995bbb37375e12a282a88cf4426f495dc0` |
| `reindex_3d.py` | `0f0ce790a91581f0ad529bd7e477b02862c98b4025b5933a1e8fcfb2326be73f` |
