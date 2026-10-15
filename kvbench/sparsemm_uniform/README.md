# Uniform：均匀 KV 头预算 + 注意力驱逐

Uniform 描述的是**预算如何分配**：所有 decoder 层、所有物理 KV 头都使用相同容量 K。头内保留哪些 token 则由注意力分数决定。它可以搭配不同打分器；本次接入使用与当前 LU-KV 对照一致的直接注意力打分。

在当前 Qwen2.5-VL-7B 的 28 层、每层 4 个 KV 头上，K1000 表示 112 个 KV 头各自最多保留 1000 个视觉 token，总视觉容量 112000。W32 时，每头保留最近 32 个视觉 token，其余最多 968 个历史 token 按 FP32 因果 Q-to-K 注意力、GQA mean、avgpool(kernel=5) 后的分数 top-k 选择。14 个初始文本 token 单独保护，不占视觉预算。

预算示例：

| 策略 | 各层各头容量 | 头内选择 |
|---|---|---|
| Uniform | 全部固定 K | 当前视频 chunk 的直接注意力 |
| PyramidKV | 按层变化 | 直接注意力 |
| AdaKV | 根据在线注意力在头间分配 | 直接注意力 |
| SparseMM | 根据视觉头校准分数固定分配 | 直接注意力 |
| 当前 LU-KV | 根据离线长期效用 profile 和当前压缩比例分配 | 直接注意力 |

实现最小扩展 `sparsemm_baselines/core.py` 的选择器与原生 METHOD 注册，复用既有 HF Qwen2.5-VL 解码器、ragged cache、绝对 MRoPE 位置、视频采样、问答事务和跨视频重置。对应 pinned SparseMM 的 `SnapKVCluster(pyram_mode=False)`；其上游方法名为 `snapkv`，本地框架入口为 `uniform`。新增流式适配支持 Q<W、前缀保护和逐 temporal patch 压缩。严格 short-cache gate 为视觉 token 数 < K 时完整保留，等于 K 时也执行注意力 top-k，保留原 Torch top-k 的并列分数处理。

本地旧 HERMES Uniform 的头预算也是均匀的，但其驱逐使用 HERMES 伪问题和 recency 打分。本次没有复用旧准确率，不需要校准资产，也不执行 HERMES probes。预算均匀与沿时间/位置均匀抽取 token 是不同策略。

```bash
cd /home/yujun/program
env PYTHONPATH=/home/yujun/program:/home/yujun/program/sparsemm_baselines/upstream/lmms-eval:/home/yujun/program/sparsemm_protokv/runtime_deps \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MALLOC_ARENA_MAX=2 \
  /home/yujun/.conda/envs/hermes-qwen/bin/python -m sparsemm_uniform.run_streamingbench \
  --anno-path /home/yujun/program/sparsemm_lukv/results/extended_validation/boundary3v8q_20261014/annotations.json \
  --max-videos 3 --budget 1000 --budget-scope visual --dtype bfloat16 \
  --sample-fps 0.5 --encode-chunk-size 16 --small-validation \
  --output-dir /home/yujun/program/sparsemm_uniform/results/small_validation/boundary3v8q_20261014/uniform_k1000_visual
```

原框架仍通过 `python -m lmms_eval --model qwen2_5_vl_stream --tasks streamingbench_local` 执行推理。独立审计入口：`python -m sparsemm_uniform.audit OUTPUT --min-compression-events-per-video 1`。输出目录不可覆盖或续跑；只允许显式最多 3 视频、8 题、问题时刻不超过 120 秒的小验证。全量实验与 ProtoKV 保持暂停。

来源：[SparseMM 固定提交中的 SnapKV](https://github.com/CR400AF-A/SparseMM/blob/33f2f0ab96d1d99516bcd2092caccab60516ff2c/sparsemm/sparsemm_utils.py#L174)、[LU-KV 论文中的预算分配与头内打分](https://arxiv.org/html/2602.08585v2)。
