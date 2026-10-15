# SparseMM 原生流式适配

接入固定提交 `33f2f0ab96d1d99516bcd2092caccab60516ff2c` 的 SparseMM / lmms-eval，通过原生 `qwen2_5_vl_stream` 模型与 `streamingbench_local` 任务运行，方法名为 `METHOD=sparsemm`。

实现复用共享 Qwen2.5-VL 原生 decoder、MRoPE、ragged KV cache 和 StreamingBench 视频/问答流程。`policy.py` 复现上游 `SparseMM` 类的容量分配：全局归一化 query-head 视觉分数，再按 GQA 组求和；`ratio=0.1`，保留最近 32 个视觉 token，历史容量按 float64 round-to-even 取整。K1000 的 112 个 KV heads 总容量为 **111995**，保留原舍入差额，没有补齐到 112000。

在线逐视觉 temporal patch 更新缓存，使用 FP32 causal Q-to-K 注意力、GQA mean 和 avgpool(kernel=5) 排序历史 token。短缓存 gate 比较当前物理视觉历史与 K-W；文本前缀单独保护，保留绝对 MRoPE 位置。容量不足的头不向其他头转移配额。问答临时缓存恢复、每视频重置和无 ProtoKV bank 均有运行检查。

默认分数来自本地 **Qwen2.5-VL-7B、1000 张 SynthDOG 图片**的完整校准：

`hermes/results/myobservations/sparsemm_official_calibration_20261010/calibration/head_scores.csv`

CSV SHA-256 为 `9cc0f47f24a0af4306ce7367806516d119dc4f57df77d064a3eac59c05b69dab`。`assets.py` 在加载权重前核验完整校准 manifest、规范化 protocol、raw scores、CSV 均值及模型/源码来源。校准使用 slow processor；当前视频评估沿用其他方法的 fast processor。这是论文方法向 Qwen2.5 流式视频任务的适配，小规模结果不能视为论文完整 benchmark 复现。

同条件小规模验证：

```bash
cd /home/yujun/program
env PYTHONPATH=/home/yujun/program:/home/yujun/program/sparsemm_baselines/upstream/lmms-eval:/home/yujun/program/sparsemm_protokv/runtime_deps \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MALLOC_ARENA_MAX=2 \
  /home/yujun/.conda/envs/hermes-qwen/bin/python -m sparsemm_native.run_streamingbench \
  --anno-path /home/yujun/program/sparsemm_lukv/results/extended_validation/boundary3v8q_20261014/annotations.json \
  --max-videos 3 --budget 1000 --budget-scope visual --dtype bfloat16 \
  --sample-fps 0.5 --encode-chunk-size 16 --small-validation \
  --output-dir /home/yujun/program/sparsemm_native/results/small_validation/boundary3v8q_20261014/sparsemm_k1000_visual
```

输出目录不可覆盖或续跑。默认只允许最多 3 视频、8 问题、问题时刻不超过 120 秒的显式小验证；全量实验保持暂停。独立审计入口为 `python -m sparsemm_native.audit OUTPUT --min-compression-events-per-video 1`。

上游方法与代码：[CR400AF-A/SparseMM](https://github.com/CR400AF-A/SparseMM)。
