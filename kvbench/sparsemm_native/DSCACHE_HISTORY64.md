# 旧紧凑历史 K500（SimpleStream）

入口是 `python -m sparsemm_native.run_simplestream_dscache_history64 --arm pae_history_k500`。该配置在 112 个 KV 头中固定选择 56 个单帧近期头；这些头保留 436 个近期位置和最多 64 个历史位置，其余 56 个四帧头保留 500 个近期位置。总视觉容量上限为 56,000 个槽位。历史候选逐帧加入，并由问题无关的伪查询分数驱逐。它是旧紧凑历史配置，与后续的 anchor/ordered/transfer 变体分开。

头分组来自仓库内的 `assets/recent1or4/per_head_allocations.csv`（SHA-256 `acd6ab409a6b83dfa42535a4298e714565c1d7ecba401b9c2033d5da7bbb6fdf`）。旧对照还使用 `assets/dp/per_head_allocation_dual_k500.csv`（SHA-256 `957a8d89b79cae86d1390478a451e658932de5e4a990b9fa4813c353c815b317`）。物理逐头 ragged-cache 辅助模块已复制到 `sparsemm_native/hermes_kv_head_ragged.py`，与原 HERMES `head_analysis/hermes_kv_head_ragged.py` 字节一致（SHA-256 `61c89974ac7e803664e17bd56772e3dbd65b095def444feec5d193476184cf3c`）。

SimpleStream 使用 [EvolvingLMMs-Lab/SimpleStream](https://github.com/EvolvingLMMs-Lab/SimpleStream) 的固定提交 `be227ee8ce90a258b286a34161688c2a59eeeadb`。入口会检查检出的 HEAD；上游源码不复制进 KVBench。模型权重、OVO 官方标注和视频、同题近期基线 `queries.jsonl` 仍为外置输入；完整 `results/` 不属于源码发布范围。使用本方法时需提供 SimpleStream Qwen2.5 运行环境，版本约束见 `run_simplestream_k500_input_only.py` 的 `REQUIRED_VERSIONS`。

从 `HorizonKV/kvbench` 执行：

```bash
git clone https://github.com/EvolvingLMMs-Lab/SimpleStream.git ../SimpleStream
git -C ../SimpleStream checkout be227ee8ce90a258b286a34161688c2a59eeeadb

export KV_BENCH_SIMPLESTREAM_ROOT="$(realpath ../SimpleStream)"
export KV_BENCH_HERMES_RESOURCE_ROOT=/path/to/hermes-resources
export KV_BENCH_OVO_CHUNKS_ROOT=/path/to/ovobench-chunks
export KV_BENCH_MODEL_PATH=/path/to/Qwen2.5-VL-7B-Instruct
# 如标注或源视频位于其他目录，可另设 KV_BENCH_OVO_ANNOTATION、KV_BENCH_OVO_SOURCE_VIDEOS。

python -m sparsemm_native.run_simplestream_dscache_history64 \
  --arm pae_history_k500 \
  --manifest /path/to/recent-baseline/query_manifest.json \
  --baseline-queries /path/to/recent-baseline/queries.jsonl \
  --output-root results/pae_history_k500 \
  --prepare-only

# 预检成功后，用相同参数去掉 --prepare-only 启动推理。
```

`--manifest` 应来自同一环境、同一批题的固定 896 近期基线；该清单中的源视频绝对路径须与当前外置 OVO 资源相符。基线的 `queries.jsonl` 必须已有清单中每道题的完成行，且包含采样帧、chunk 和 prompt 哈希。程序在推理时逐题检查这些输入与近期基线一致，续跑也会重新核对代码和协议哈希。请使用新的输出目录，不要复用旧实验的 `results/`。

CPU 测试：

```bash
python -m pytest -q tests/test_simplestream_dscache_history64.py tests/test_dscache_position_agnostic.py
```

`--prepare-only` 只验证资源、配额与协议生成，不表示 GPU 推理或完整 OVO 实验通过。
