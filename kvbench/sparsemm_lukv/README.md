# LU-KV：注意力校准与 SparseMM 原生接入

当前默认方法使用**以纯视觉注意力排序重新校准的 LU 预算曲线，在线驱逐也使用同一注意力排序**。离线长期效用保留 LU 的投影 Value 定义，并按作者实现进行层内归一化。在线不计算 Value 效用，不调用 HERMES 的 local/global/mixed probe，不混合 recency 分数，不用历史问答评分。

使用固定版本 SparseMM 原版 lmms-eval，外部模型 `qwen2_5_vl_stream`、任务 `streamingbench_local`、`METHOD=lukv`，适配 Transformers 4.57.1 原生 Qwen2.5-VL-7B 与本地 StreamingBench。

## 校准和驱逐的评分

排序读取当前视觉 patch 最后 `min(32, query_len)` 个实际 post-MRoPE Q 和视频 K：FP32 `QKᵀ/sqrt(d)` → causal softmax → query位置均值 → 共享物理KV头的GQA均值 → 宽度5 avgpool。校准与在线共用 SparseMM `_head_scores`，并在同一设备采用 `torch.sort(descending=True)`，避免不同排序函数的 ties 行为差异。

校准时保留完整视频 KV、不驱逐。用真实校准问答的答案预测位置计算长期效用：每个query头先得到 `||V @ Wo_q||₂`，除以该层所有query头和持久上下文位置的范数总和；再与对应答案位置的原生注意力相乘，GQA取均值，并在最多16个答案预测位置上取最大值。注意力归一化包括初始文本、视频、当前问题及已生成答案，效用仅记录视频位置。问答后恢复完整视频缓存。该观察不增加模型前向。

每个观测按注意力排序形成效用随容量的曲线，PAVA构造凹包，再用边际效用贪心分配多档全局容量。先在视频内平均保留率，再在视频间等权平均；没有旧版0.5均匀混合和750–1500容量限幅。独立CPU参考核对排名、Oracle、原生模型与缓存恢复。

## 在线预算随上下文变化

默认资产：`assets/attention_native10v_20261014/profile.json`。这是多压缩率曲线，不是固定K1000的单张容量表。

每个视觉patch先冻结所有层的预算，使用未压缩累计视觉token数T；平均视觉预算K、最近窗口W=32时：

```text
T <= K：每个头保留全部可用视觉位置
T > K：历史保留率 rho = (K-W)/(T-W)
       查表/插值得到各头历史保留率 p[layer,head](rho)
       原始容量 = W + p[layer,head](rho) * (T-W)
```

容量在 `[W,T]` 内投影并精确取整，全局目标为 `28*4*min(K,T)`。各头容量可随T变化；实际存储量另计。某个头的目标容量上调时，先前已驱逐的token无法恢复，因此实际容量可能小于目标，不会使用当前问题或跨头重新分配去补齐。

各头根据注意力保留历史高分token，并保护初始文本与最近32个视觉位置。最近窗口计入视觉容量，初始文本额外计算。历史索引保持分数排序，原始MRoPE位置随索引保存。问答临时KV在回答后恢复；视频间清空缓存。

## 校准入口与范围

```bash
cd /home/yujun/program
/home/yujun/.conda/envs/hermes-qwen/bin/python -m sparsemm_lukv.run_calibration \
  --run-calibration \
  --output-dir /home/yujun/program/sparsemm_lukv/results/calibration/MY_NEW_CALIBRATION \
  --assets-dir /home/yujun/program/sparsemm_lukv/assets/MY_NEW_PROFILE
```

默认从现有100视频校准池确定性选择10视频、11原始问题，覆盖10类任务，时间15–102秒；按视频ID和basename排除全250视频评测集。只采原题原时间，使用当前原生分辨率、0.5FPS、2帧temporal patch、16帧读取批次。`--dry-run`只做数据预检。校准最多10视频、每视频2题、120秒前缀；已有证据或资产拒绝覆盖。

结果在 `results/calibration/native_attention10v_20261014/`，保存每个观测的注意力分数、排名、投影效用、模型/数据/源码指纹与恢复记录。默认`profile.json`仅用投影长期效用构建，另存的attention效用是诊断数据。`budget_k1000_at_T*.csv`仅展示某个参考T的容量，不是在线固定预算。

## 小规模原生评测

```bash
cd /home/yujun/program
/home/yujun/.conda/envs/hermes-qwen/bin/python -m sparsemm_lukv.run_streamingbench \
  --small-validation \
  --anno-path /home/yujun/program/sparsemm_protokv/results/small_validation/annotations.json \
  --max-videos 2 --budget 1000 --budget-scope visual \
  --output-dir /home/yujun/program/sparsemm_lukv/results/MY_NEW_RUN
```

通用入口为 `python -m sparsemm_protokv.run_streamingbench --method lukv ...`。用 `--lukv-profile-path /absolute/path/profile.json`选择新的校准曲线。校准源、观测、协议和模型指纹不匹配时，在加载GPU模型前拒绝执行。历史固定CSV仅可用 `--lukv-budget-path`显式选取；默认不回退旧资产。两种资产参数互斥。

独立标准库审计：

```bash
/home/yujun/.conda/envs/hermes-qwen/bin/python -m sparsemm_lukv.audit OUTPUT_DIR
```

审计独立重算每patch预算插值、投影和取整，核验所有层的目标/实际容量、索引、MRoPE位置、零probe、问答恢复以及校准/评测数据隔离。注意力分数的数学一致性由独立CPU参考测试验证；审计不重新跑模型。

全量评测、ProtoKV和其他已暂停实验继续暂停。原生评测入口仍仅允许最多3视频、8题、120秒前缀。视频读取用batch-scoped Decord，复制像素后立即关闭解码器。`selector_seconds`已包含在`language_video_seconds`中，不可相加；详细状态检查的小样本计时不代表正式吞吐。

## 与论文的关系及历史结果

[LU论文§4.3和附录E](https://arxiv.org/html/2602.08585v2#S4.SS3)、[作者校准实现](https://github.com/baidu-baige/LU-KV/blob/main/evaluation/curve_data/step1_qwen.py)支持离线效用校准与在线查表。当前仍是视频适配：使用物理GQA头、保护全部初始文本、最近32个视觉位置、最多16个MCQA答案预测位置；为了支持固定全局K的长视频流，使用最近窗口下限，未加入论文的每头至少保留1%上下文约束。小校准集与小验证不足以建立论文或完整评测成绩。

旧HERMES在线评分结果在`results/small_validation/`；旧预算迁移+attention驱逐结果在`results/visual_attention_validation/`。新校准版验证结果单独保存，不能混报旧成绩。各版本源码/说明快照在`results/source_snapshots/`及对应运行目录的`source_snapshot/`。
