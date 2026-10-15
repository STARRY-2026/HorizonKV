# Unified KV evaluation

Status: **complete**. Full benchmarks remain paused.

| Method | Run status | Audit | Correct / questions | Compression events | Peak GPU GiB |
|---|---|---|---:|---:|---:|
| adakv | complete | passed | 1 / 3 | 9 | 15.853 |
| lukv | complete | passed | 2 / 3 | 9 | 15.853 |
| pyramidkv | complete | passed | 2 / 3 | 9 | 15.853 |
| sparsemm | complete | passed | 1 / 3 | 9 | 15.853 |
| uniform | complete | passed | 2 / 3 | 9 | 15.853 |

Bounded integration validation; not official full benchmark reproduction.
Small sample accuracy does not establish a method ranking.
The full method is a large-budget control and must show zero compression events.

Native results, logs, source snapshots and audit reports are kept in each method directory.
