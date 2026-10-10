# 主要实验结果选集

本目录附上项目已有的主要实验结果：14 项方法/预算/数据划分记录，以及对应的逐题预测、评测报告和配置来源。本次整理重新核算了所有准确率，**没有重新运行全量实验**。

模型均为 Qwen2.5-VL-7B-Instruct，采样率为 0.5 FPS。K 表示平均每个物理 KV head 的视觉缓存预算。以下准确率为逐题 `qa_acc` 的平均值，不是任务宏平均。

| 方法 | K | 数据划分 | 分辨率 | 视频 / 问题 | 正确数 | 准确率 |
|---|---:|---|---|---:|---:|---:|
| [Uniform ragged HERMES](runs/uniform_k100_first150/results.csv) | 100 | first150 | native | 150 / 754 | 422 | 55.97% |
| [DP + head recency alpha/k](runs/dp_k100_first150/results.csv) | 100 | first150 | native | 150 / 754 | 419 | 55.57% |
| [Uniform ragged HERMES](runs/uniform_k250_first150/results.csv) | 250 | first150 | native | 150 / 754 | 440 | 58.36% |
| [DP + head recency alpha/k](runs/dp_k250_first150/results.csv) | 250 | first150 | native | 150 / 754 | 454 | 60.21% |
| [Uniform ragged HERMES](runs/uniform_k500_first150/results.csv) | 500 | first150 | native | 150 / 754 | 477 | 63.26% |
| [DP + head recency alpha/k](runs/dp_k500_first150/results.csv) | 500 | first150 | native | 150 / 754 | 489 | 64.85% |
| [Uniform ragged HERMES](runs/uniform_k1000_first250/results.csv) | 1000 | first250 | native | 250 / 1255 | 875 | 69.72% |
| [DP + head recency alpha/k](runs/dp_k1000_first250/results.csv) | 1000 | first250 | native | 250 / 1255 | 886 | 70.60% |
| [Native v8 / MemHeadKV](runs/native_v8_k1000_first150/results.csv) | 1000 | first150 | native | 150 / 754 | 516 | 68.44% |
| [Uniform ragged HERMES](runs/uniform_ragged_k1000_first150/results.csv) | 1000 | first150 | native | 150 / 754 | 506 | 67.11% |
| [Timestamp v8 (g0)](runs/timestamp_v8_g0_fixed448_k1000_half/results.csv) | 1000 | half250 | fixed448 | 250 / 1255 | 882 | 70.28% |
| [Historical uniform ragged (relative, window=0)](runs/implicit_uniform_ragged_relative_fixed448_k1000_half/results.csv) | 1000 | half250 | fixed448 | 250 / 1255 | 899 | 71.63% |
| [HorizonKV v0.6 (historical)](runs/historical_horizonkv_v06_k6000_half/results.csv) | 6000 | half250 | native | 250 / 1255 | 902 | 71.87% |
| [Original dense HERMES](runs/original_dense_hermes_k6000_half/results.csv) | 6000 | half250 | native | 250 / 1255 | 928 | 73.94% |

## 文件与核对方式

- [main_results.csv](main_results.csv)：统一主结果表；[task_results.csv](task_results.csv)：逐任务统计。
- [index.json](index.json)：完整索引，包括预算、划分、分辨率、问题顺序哈希和排除原因。
- 每项 `runs/<id>/` 包含预测 CSV、评测报告和 `metadata.json`，后者记录原路径、启动参数或脚本证据、配置差异及 SHA256。
- 单文件结果与原项目逐字节一致。合并结果明确标为 derived；原分片与原评测报告保存在 `source_runs/`，合并只连接既有行，保留顺序和重复问题。
- [source_manifest.json](source_manifest.json) 记录所有复制的原结果与报告路径及哈希。已逐项复算 `qa_acc == 100 × (pred_choice == correct_choice)` 并核对报告样本数和两位小数准确率。
- 所有对照组均核对了完整问题、选项、答案与顺序。CSV 不包含时间戳，因此按已核对的原始视频内问题顺序恢复问题索引，不能仅按问题文本去重。

## 比较边界

1. DP 的四组 uniform 对照使用相同 native 划分、K 和 SparseMM/ragged 设置。K100 的负结果也保留；K1000 uniform 由 first50、51–150、151–250 三个不重叠的完成片合并。
2. native v8 使用 first150，DP K1000 使用 first250，不能直接将两行均值相减。v8 自带同 first150 的 uniform 对照；历史整数舍入比名义总预算多 4 个槽。
3. fixed448 历史 uniform 基线是 ragged relative/window=0，不是 dense HERMES，也不是当前 portable uniform 的 SparseMM/window=32。精确 legacy 复现参数见对应 metadata；timestamp 与该基线可作历史结果对照，但不隔离单一 selector 变量。
4. HorizonKV 附的是 **历史 v0.6 half 完整结果**，不是最新 v0.10。其 pilot profile 的 10 个校准视频中有 7 个与 half 重叠；不能当作严格独立测试。dense HERMES K6000 同 half 结果来自较早代码版本，本次没有控制历史代码版本重跑。
5. DP 的校准来源核对见 [dp_calibration_audit.json](dp_calibration_audit.json)：48 个有效校准视频、68 个观测与所选 DP 评测视频没有交集。这不证明评测集未参与方案选择；重新执行校准审计需要原始、未打包的大型 trace。
6. 历史运行的完整环境/代码提交并未逐次冻结，当前安装说明与 GPU smoke 证明当前代码的运行链路，不保证逐项精确复现历史数值。历史文档的 BF16 标签不能代替实际运行精度证据。

## 没有纳入的结果

- current-ourtop 的 half 输出只有 71 视频 / 358 问分片，缺少完整结果与评测报告。
- K800 没找到已完成且匹配的 DP 主结果；K250 小 pilot 实际是 uniform 容量加 alpha/k，不当作 DP 主结果。
- HorizonKV v0.10 只有面向 7 个特定视频的 35 问 pilot，不与 half 完整结果混列。
- 大型 trace、视频帧、原始日志和大量消融不随本选集打包。

运行新实验见 [EXPERIMENTS.md](../EXPERIMENTS.md)；本次 GPU 冒烟验证见 [RUNTIME_VALIDATION.md](../RUNTIME_VALIDATION.md)。
