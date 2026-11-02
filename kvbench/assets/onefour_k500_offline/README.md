# 固定 K500 的逐头近期／历史配额

`bank_curves.npz` 保存 112 个 KV 头的近期 `recent` 与历史 `history` 离线效用曲线。它来自本地 HERMES 两缓存 K500、Recent 最少 32 个位置的校准结果；SHA-256 为 `9c848a4812a8e8d6c7ddaa75cc3c87b1c5cd6ede8fa28d5072e6ef1daee76636`。这些曲线是数值校准资产，不是 OVO 答题准确率或模型输出。

`per_head_allocation.csv` 是用 `tools/plot_onefour_k500_offline_allocation.py` 对每个头分别选择 `argmax R_h(r) + H_h(500-r)`、`32 <= r <= 500` 得到的冻结表。每头总容量为 500，112 头总容量为 56,000；近期合计 27,364，历史合计 28,636。`summary.json` 记录目标函数、分组和资产哈希。56 个单帧头的分组来自 `../recent1or4/per_head_allocations.csv`。

从 `kvbench/` 目录重新生成并逐字节比较冻结结果：

```bash
python tools/plot_onefour_k500_offline_allocation.py --output-dir /tmp/onefour-k500-rebuilt
cmp /tmp/onefour-k500-rebuilt/per_head_allocation.csv assets/onefour_k500_offline/per_head_allocation.csv
cmp /tmp/onefour-k500-rebuilt/summary.json assets/onefour_k500_offline/summary.json
```

需要 `numpy` 和 `matplotlib`。推理时使用 `python -m sparsemm_native.run_simplestream_dscache_history64 --arm pae_history_k500 --allocation-csv assets/onefour_k500_offline/per_head_allocation.csv`，并提供专项说明中的 manifest、近期基线和外置模型／视频。上游校准观察、完整实验日志和视频不随本资产上传。
