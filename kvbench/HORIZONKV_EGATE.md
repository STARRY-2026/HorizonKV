# HorizonKV-EGate：容量曲线配额与证据门控双遍回答

**HorizonKV-EGate (K500)** 是本仓库对固定 K500、近期／历史双缓存和在线 E 选项门控这一实验配置的名称。其运行标识仍是 `pae_online_e_gate_k500`；必须同时传入 `assets/onefour_k500_offline/per_head_allocation.csv`，才会使用本文所述的容量曲线预算。它是 SimpleStream 上的 Qwen2.5-VL OVO 选择题实验入口，不是 `python -m kvbench run --method ...` 注册表中的方法。

## 配额从哪里来

冻结的 `assets/onefour_k500_offline/bank_curves.npz` 保存 112 个 KV 头的离线近期效用曲线 `R_h` 和历史效用曲线 `H_h`。对每个头独立求

```text
r_h = argmax_{32 <= r <= 500} [R_h(r) + H_h(500-r)]
s_h = 500-r_h
```

`r_h` 是近期配额，`s_h` 是历史配额。生成工具是 `tools/plot_onefour_k500_offline_allocation.py`，输出的冻结表是 `assets/onefour_k500_offline/per_head_allocation.csv`。每头总配额 500，112 头合计 56,000；冻结结果的近期配额合计 27,364、历史配额合计 28,636。56 个头使用单帧近期分支，另外 56 个使用四帧分支；分组来自 `assets/recent1or4/per_head_allocations.csv`。曲线目标是离线、完整缓存上的投影答案效用代理，**不是 OVO 答题准确率**。原始校准观察和完整实验结果未包含在仓库内。

## 每道题如何运行

1. 按固定 0.5 fps 采样，图像等比缩放并填充到 896×896；每张图得到 1024 个视觉 token。最后四张采样图构成近期范围，更早的图像构成历史范围，两者不重叠。
2. 历史图像逐帧进入容量有界的缓存。模型使用与当前问题无关的伪查询注意力分数，逐头筛选历史候选；近期分支在最后一帧或最后四帧的候选中按伪查询分数筛选，并保护末尾 32 个视觉位置。因而 `r_h`、`s_h` 是容量配额，**不是简单地截取最后 `r_h` 个 KV**，也不表示实际占用必定填满。历史未用满时，剩余容量可由近期使用；总视觉容量仍受每头 500 的约束。
3. **近期回答遍**只带近期 KV，在原选择题后添加“所示近期画面证据不足”的选项。原题没有 E 时新增 E；原题已有 E 时新增 F。该遍的答案选项包含这个新增选项。
4. **历史回答遍**使用相同的近期 KV，加上筛选后的历史 KV；它使用原题的选项，**不添加证据不足选项**。代码检查两遍的近期 KV 选择及回答位置一致。历史 Key 的位置处理和逐头 ragged-cache 组合见 `sparsemm_native/dscache_pae_history.py` 与 `sparsemm_native/dscache_position_agnostic.py`。
5. 如果近期遍选择新增的“证据不足”选项，输出历史遍的原题答案；否则输出近期遍的原题答案。门控只在回答阶段使用，历史和近期条目的选择不读取题目内容。逐题审计记录两遍答案、选项概率、门控来源、实际视觉槽位、输入帧和位置校验。

本方法与 `pae_history_k500` 不同：后者只做一次近期＋历史回答，没有 E 门控。`pae_online_margin_k500` 做双遍但以选项概率差决定答案来源，不添加 E。`pae_online_e_recent_only_k500` 比较两个**近期专用**回答遍，不把历史 KV 加入第二遍，主要作为对照。若想让两遍都添加 E，目前的 `pae_online_e_gate_k500` **没有**这样实现。

## 运行

先按 [SimpleStream 专项说明](sparsemm_native/DSCACHE_HISTORY64.md)取得固定提交 `be227ee8ce90a258b286a34161688c2a59eeeadb`、Qwen2.5-VL-7B-Instruct、OVO 标注和视频，以及**同题、同固定 896 输入**的已完成近期基线 `query_manifest.json`、`queries.jsonl`。专用 Python 环境要求 `transformers==4.52.4`、`accelerate==1.8.1`、`qwen-vl-utils==0.0.14`，还需要兼容的 CUDA PyTorch 与 FlashAttention。通用 `requirements/runtime.txt` 的版本不适用于这条 SimpleStream 入口。

从 `HorizonKV/kvbench` 目录执行，先用 `--prepare-only` 核验资源、配额及协议；成功后用完全相同的参数和输出目录去掉 `--prepare-only` 启动 GPU 推理：

```bash
export KV_BENCH_SIMPLESTREAM_ROOT=/path/to/SimpleStream
export KV_BENCH_HERMES_RESOURCE_ROOT=/path/to/hermes-resources
export KV_BENCH_OVO_CHUNKS_ROOT=/path/to/ovobench-chunks
export KV_BENCH_MODEL_PATH=/path/to/Qwen2.5-VL-7B-Instruct

python -m sparsemm_native.run_simplestream_dscache_history64 \
  --arm pae_online_e_gate_k500 \
  --allocation-csv assets/onefour_k500_offline/per_head_allocation.csv \
  --manifest /path/to/recent-baseline/query_manifest.json \
  --baseline-queries /path/to/recent-baseline/queries.jsonl \
  --output-root results/horizonkv_egate_k500 \
  --prepare-only
```

`protocol.json` 固定代码、预算表、基线和资源哈希；`queries.jsonl` 保存逐题答案，`ragged_query_audit.jsonl` 保存双遍门控和实际 KV 占用，`summary.json` 保存准确率汇总。代码或预算改变后须使用新的输出目录。可用 `--vision-cache-dir /path/to/cache` 启用可选视觉 embedding 缓存。旧方法和每个门控变体应使用各自独立的输出目录。

冻结曲线到预算表的重建命令：

```bash
python tools/plot_onefour_k500_offline_allocation.py --output-dir /tmp/horizonkv-egate-allocation
cmp /tmp/horizonkv-egate-allocation/per_head_allocation.csv assets/onefour_k500_offline/per_head_allocation.csv
```

这条重建命令还需要 `matplotlib`；已核验其输出与冻结 CSV 逐字节一致。相关 CPU 测试和 `--prepare-only` 已通过，**完整 GPU 推理及全量 OVO 效果尚未在发布工作区验证**。
