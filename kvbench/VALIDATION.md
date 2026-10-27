# 整合验证

统一项目入口：`python -m kvbench`，Python API：`from kvbench import RunConfig, run, audit, summarize`。

## Qwen3-VL-8B 双数据集小规模验证（2026-10-23）

本机权重：`/home/yujun/program/hermes/models/Qwen3-VL-8B-Instruct`。实验性流式桥接位于 `sparsemm_baselines/qwen3_adapter.py`，脚本为 `tools/qwen3vl_matrix_smoke.py`、`tools/qwen3vl_policy_smoke.py`、`tools/audit_qwen3vl_smoke.py`。每种方法只跑 OVO 转换 MCQA 标注的 1117 号视频一题和 StreamingBench 18 号视频一题，0.5 FPS、整块处理、块级最大 1,200,000 像素、视觉 K=500；`full` 用大容量作无驱逐对照。OVO 单帧为 `[1,58,78]` / 1131 个视觉 token，StreamingBench 四帧为 `[2,24,44]` / 528 个。两种网格分别与原生 Qwen3 处理器对同一帧块的结果一致。

AdaKV、PyramidKV、Uniform、`full` 在两份数据上共 8/8 个实际推理通过；前三种各出现一次 KV 压缩，`full` 无驱逐。LU-KV、SparseMM、`dp`、`dp_dual` 的在线选择器也完成 8/8 个推理，且各出现一次压缩，但容量表和分数是 **36 层 × 8 KV 头的合成测试夹具**，不是 Qwen3 的离线校准，不能作为论文方法效果或准确率结果。16 个运行的缓存形状、视觉网格、问答后缓存恢复和压缩记录通过统一审计。详见 `results/qwen3vl8b_methods_matrix_20261023b/`、`results/qwen3vl8b_policy_synthetic_matrix_20261023b/` 和 `results/qwen3vl8b_methods_audit_20261023b.json`。OVO 的转换题不等于原始九类题协议。

复跑时在项目根目录使用当前 `hermes-qwen` 环境并设置 `PYTHONPATH=$(pwd)`，分别执行 `python tools/qwen3vl_matrix_smoke.py` 与 `python tools/qwen3vl_policy_smoke.py`，传入 `--model-path`、`--ovo-annotations configs/ovobench_smoke.json`、`--streaming-annotations /home/yujun/program/hermes/data/streamingbench/streamingbench_realtime_half_userpaths.json`、全新的 `--output-dir`，以及 `--video-max-pixels 1200000`。两个脚本各自最多运行 4 个方法 × 2 个数据集；审计使用 `python tools/audit_qwen3vl_smoke.py --roots <两个输出目录> --output <新审计 JSON>`。旧结果路径不允许覆盖。

当前桥接保留 Qwen3 的 Q/K 归一化、交错 M-RoPE 和 DeepStack，但没有实现原生视频时间戳文本标记；它适合验证共同视觉输入和选择器端到端链路，不能据此声称与原生 Qwen3 全上下文推理逐值等价。原有 LU/SparseMM/DP 校准资产的模型形状与 Qwen3 不符。HERMES 的 Qwen3 适配后来已接入统一 `kvbench run`，并完成两数据集各一题的 GPU 验证，见下文。独立的 `recent1or4_history_cap4000` 仅服务 OVO 九类题，且现有配额是 Qwen2.5 的 28×4 形状，也未在 Qwen3 上运行。ProtoKV 按本轮要求排除。全量实验继续暂停。

### Qwen3 HERMES 适配（2026-10-24）

统一 `kvbench run --method hermes` 现在根据 `config.json` 选择原 Qwen2.5 后端或 Qwen3 适配。Qwen3-VL-8B 的 36 层按浅层 10%、深层 30%、其余中层分组，取整后分别为 3、11、22 层。预测查询、手工 QK 注意力、时近性融合、跨层分数融合与每层共享视觉位置驱逐按 HERMES 路径实现；Qwen3 使用自身 Q/K norm、M-RoPE 与 DeepStack。与原 Qwen2.5 实现相比，Qwen3 驱逐后保留原键位置，不做压缩重排。

Qwen3 SparseMM 窗口化视觉变体已完成 1000/1000 张 SynthDOG 校准，`results/qwen3vl8b_sparsemm_window4_1000_oldpixel_20261024a/` 包含 `head_scores.csv` 和 `allocation_k500.json`，协议状态为 `complete_variant`。它是窗口注意力＋4 层全局注意力的**实验性模型变体**，不是原生预训练 Qwen3 或官方 SparseMM 的效果证据。

2026-10-25 新增 LU-KV 的完整 HERMES 伪查询排序校准。`results/qwen3vl8b_lukv_hermes_pseudo_10v_20261025b/` 在原生 Qwen3 视觉拓扑上，对与旧 LU 相同的 10 个评测互斥视频完成 11 次观测；每次分别采集 local、global、mixed 伪查询对逐 KV 头的注意力，应用 HERMES 浅/中/深层归一化与时间衰减、跨层反向修正，再用原 LU 的投影 Value 效用优化逐头容量。参考 T=576、K=500 的总量是 144,000，较纯注意力版有 257/288 个头改变配额。该适配把 HERMES 原本逐层共享位置的打分扩展到逐 KV 头，并保留 LU 的 recent32 窗口与容量分配；它不是原始 HERMES 的逐层共享 KV 策略。运行时选用新 `profile.json` 会在整块视觉前向后做相同的三组伪查询与逐头驱逐；旧 profile 继续走纯注意力在线选择。原生拓扑的 OVO/StreamingBench 各一题测试位于 `results/qwen3vl8b_lukv_hermes_pseudo_2dataset_smoke_20261025b/`，窗口化拓扑的兼容性测试位于 `results/qwen3vl8b_lukv_hermes_pseudo_window4_2dataset_smoke_20261025a/`；两组测试均压缩并恢复缓存，对应 `..._audit_20261025b.json` 与 `...window4_2dataset_audit_20261025a.json` 均通过。两个测试视频的视觉 KV 实占均为 144,000。小样本仅验证链路，窗口化在线测试没有重新校准预算，不能用正确数排序方法；全量实验仍暂停。先前将旧分数直接套归一化公式的 `...hermes_formula_native10v...` 目录标记为 `superseded_exploratory`，不得当作本次完整伪查询校准。

HERMES 的初次 Qwen3 原生全局视觉注意力运行在模型加载后被 `SIGKILL` 终止，见 `results/qwen3_hermes_k500_streaming1_20261024a/`；窗口化重试通过视觉前向后，因追踪配置访问错误中止，见 `...window4_k500_streaming1_20261024a/`。修正后，`results/qwen3_hermes_window4_k500_streaming1_20261024b/` 在 StreamingBench 视频 3 的第一题完成端到端推理和审计：原视频 6 帧、720×1280，经处理为网格 `[3,44,80]`、2640 个视觉 token；压缩后浅／中层各 500 个视觉位置、深层各 499 个，加上 14 个前缀 token。问答后 KV 恢复成功，本题预测 D、正确 D，峰值 GPU 分配 17.12 GiB。随后 `results/qwen3_hermes_window4_k500_ovo1_20261024a/` 在 OVO-Bench 转换 MCQA 视频 1117 的第一题完成相同链路：单帧 1080×1440、1530 个视觉 token、1 次压缩、缓存恢复与审计通过；预测 D、正确 A，峰值 GPU 分配 16.80 GiB。**每数据集仅一题，只证明小规模链路可用，不能推断准确率或与其他方法的公平效果比较。**

### Qwen3 九方法双数据集扩展验证（2026-10-24）

应进一步多题验证的要求，在 OVO-Bench 官方原有选择题的统一格式子集中选 1117、34、1325 三个视频各一题，在 StreamingBench 中选视频 11 的两题（16/37 秒）和视频 18 的一题（8 秒）。OVO 的选项和正确索引来自官方标注，统一入口更改了提示词格式和字母提取方式。九种方法都用 Qwen3-VL-8B 窗口注意力＋4 层全局注意力视觉变体、0.5 FPS、每块最多 120 万像素和 K=500；完整 KV 对照使用无驱逐大预算。八种选择器的统一运行位于 `results/qwen3vl8b_extended8_window4_k500_20261024a/`，HERMES 的 OVO 和 StreamingBench 运行分别位于 `results/qwen3_hermes_extended_window4_k500_ovo3_20261024a/`、`results/qwen3_hermes_extended_window4_k500_streaming3_20261024a/`。联合报告为 `results/qwen3vl8b_extended9_window4_k500_20261024a.json`。

18 个方法×数据集组合共 54 次问答均完成，18 组缓存与压缩审计通过；每种方法在每个数据集均为 3 题。跨九种方法逐块核对了视频 ID、视觉网格和视觉 token 数，全部相同。OVO 三个视频的视觉 token 分别为 1131、1131、588；StreamingBench 的三个视频块为 576、546、528。Full 对照零驱逐，其他八种方法均触发压缩。同视频多题的缓存恢复通过。

本轮 LU-KV 使用 10 视频 profile，SparseMM 使用完成的 1000 图窗口化分数，DP/DP-dual 使用 10 视频注意力预算；但 LU 和 DP 的离线资产是在**原生 Qwen3 视觉拓扑**下取得，在线评测改为窗口化变体，属于跨拓扑资产迁移。HERMES 是逐层共享视觉位置预算，其他方法多为逐 KV 头容量；OVO 使用转换 MCQA，Qwen3 流式桥接仍省略原生时间戳文本标记。故这一轮只可用于检查多题运行和输入对齐，不应根据小样本正确数给方法排序或报告正式准确率。全量实验继续暂停。

### Qwen3 离线校准试点（2026-10-24）

已经开始用 Qwen3-VL-8B 的真实前向替换上述合成资产，产出均放在新目录，未覆盖旧校准或历史结果：

| 方法 | 真实校准证据 | 在线验证 | 当前限制 |
|---|---|---|---|
| LU-KV | 与 StreamingBench 评测标注按视频 ID/文件名排重的 3 个视频、3 次回答观测；纯注意力排序、投影 Value 效用；36×8 profile，K500 参考配额 144,000 槽 | OVO/StreamingBench 各一题通过 | 仅 3 视频试点；视频桥接尚无原生时间戳文本 |
| SparseMM | 3 张真实 SynthDOG OCR 图像；沿用 top-1 注意力落框记分规则生成 36×32 query-head CSV，再按原分配式汇聚到 36×8 | OVO/StreamingBench 各一题通过 | 原 Qwen2.5 正式校准为 1000 图、最多 1024 解码 token；此试点仅 3 图、最多 64 token，且框映射按 Qwen3 网格适配，不能代表稳定分数或官方协议 |
| DP、DP-dual | 用两个 1104-token LU 观测的实测投影效用，按注意力历史排序求精确 K500 全局容量及近期/历史拆分；288 头共 144,000 槽 | 两方法在两个数据集共四题通过 | 这是**注意力排序的 Qwen3 适配试点**，原 DP 离线候选采用 HERMES 排序，故不应称为原 DP 的逐项复现 |

校准结果分别位于 `results/qwen3vl8b_lukv_calibration_3v_20261024b/`、`results/qwen3vl8b_sparsemm_ocr_calibration_3img_20261024b/` 和 `results/qwen3vl8b_dp_attention_calibration_2v_20261024a/`。SparseMM 的首轮 `...3img_20261024a/` 资产已用于在线验证；优化末位 logits 后重采集的 `...3img_20261024b/` 的 `head_scores.csv` 与原始逐图分数 JSON 均与首轮逐字节同哈希。三组实测资产驱动的 8 个在线运行位于对应的 `qwen3vl8b_lukv_real3v_profile_smoke_20261024a/`、`qwen3vl8b_sparsemm_real3img_scores_smoke_20261024a/`、`qwen3vl8b_dp_real2v_attention_smoke_20261024a/`；共同审计 `results/qwen3vl8b_real_pilot_methods_audit_20261024a.json` 为 8/8 通过，逐项核对原生处理器网格、压缩和问答缓存恢复。新增 DP 曲线/全局预算的两个小型 CPU 精确性测试通过。校准与在线试点均不是正式准确率比较；扩大校准规模及原生 Qwen3 时间戳处理仍是后续工作。

第一次 3 视频 LU 采集在第二段视频的 Decord EOF 重试上限处中断，目录 `results/qwen3vl8b_lukv_calibration_3v_20261024a/` 标记为失败。随后采用按批释放解码器并提高 EOF 重试上限，在独立的 `...3v_20261024b/` 目录完整重跑 3/3 个观测；失败目录没有被当成完成资产使用。

新增 Qwen3 脚本均通过 `py_compile`；统一 CLI 的 24 项 CPU 测试通过。全量 `unittest discover` 当前运行 150 项，出现 1 个缺失 `lmms_eval` 路径的错误和 3 个暂停实验冻结源码 SHA 不匹配的失败；本轮未改写相关冻结源码，故不能报告整个旧测试套件通过。

CPU 测试可在项目目录使用当前环境重跑：

```bash
/home/yujun/.conda/envs/hermes-qwen/bin/python -m unittest discover -s tests -v
```

- 整合阶段 19 项统一接口 CPU 测试通过，覆盖单方法/批量选择、参数传递、暂停边界、禁止覆盖、迁移审计、失败报告，以及完整 KV 对照的零驱逐要求。上传前加入嵌套 HorizonKV 资源路径验证，20 项 CPU 测试通过。原日志留在本机 `results/interface_cpu_tests.log`。
- 7 种入口（五种常用方法、ProtoKV、完整 KV 对照）均通过 dry-run 命令与协议生成；该检查没有进行 ProtoKV 或完整 KV 推理。
- 5 种非 ProtoKV 方法从新目录实际完成同一组 OVO 的 3 视频/3 题推理和独立审计。共 15 个预测（包括答案文本）与整合前完全一致；每种方法有 9 次真实压缩、3 次问答缓存恢复，首个回答使用 28 层原生注意力。无运行或数值检查失败。
- 67 项复制的源码及校准文件 SHA256 与迁移清单一致；固定上游提交 `33f2f0ab96d1d99516bcd2092caccab60516ff2c` 未修改。旧项目与历史结果保留。

| 方法 | 正确题数 | 与整合前一致 | 审计 |
|---|---:|---|---|
| AdaKV | 1/3 | 是 | 通过 |
| PyramidKV | 2/3 | 是 | 通过 |
| LU-KV | 2/3 | 是 | 通过 |
| SparseMM | 1/3 | 是 | 通过 |
| Uniform | 2/3 | 是 | 通过 |

可随 GitHub 仓库查看的精简证据在 [validation_examples](validation_examples/README.md)；完整证据保存在本机未上传的 `results/ovobench_regression_20261015/`：

- [统一报告](validation_examples/ovobench_3v3q_20261015/report.md) 与 [汇总 JSON](validation_examples/ovobench_3v3q_20261015/summary.json)。
- [迁移回归核对](validation_examples/ovobench_3v3q_20261015/migration_regression.json)：逐题预测相同、共同源码指纹相同、所有本地资产副本指纹相同。
- 每种方法的逐题 `results.csv` 与 `audit_report.json` 也在精简证据目录中。
- 每方法目录下的 `source_snapshot`、`run_manifest.json`、`protocol.json`、逐题 CSV、缓存轨迹和审计报告。
- `interface_source_snapshot/` 保存真实推理时的统一入口源码。随后修复了 `full` 对照汇总判定的一处报告边界，并由 CPU 测试覆盖；五种方法的计算路径和算法源码没有修改。回归 JSON 同时记录测试时与最终接口指纹。

本次只验证整合后的运行行为，不是官方全量 OVO 评测，也不据这 3 题比较方法效果。没有启动全量实验或 ProtoKV 推理。

项目更名为 `kvbench` 后，19 项 CPU 测试、五种方法的 dry-run 与五份历史结果的只读审计再次通过；15 个预测仍与原始对照一致。LU/SparseMM 审计改为比较规范化后的资产路径，保持原 SHA256 校验。更名验证写入独立的 `results/rename_validation_20261015/`，原实验协议、缓存轨迹、源码快照和审计记录未改写。

上传前，在 HorizonKV 仓库的 `kvbench/` 嵌套布局下再次运行 20 项 CPU 测试、`python -m kvbench methods`，并对五种默认方法执行 dry-run；均通过。dry-run 只检查命令和协议，不加载模型、不给出推理或审计成功结论。使用了本机已有模型路径；新机器仍须另行下载模型和视频。

## LLaVA-OneVision 后端小规模验证（2026-10-15）

`--model-path` 指向本机 `llava-onevision-qwen2-7b-ov-hf` 时，统一入口自动选择 `llava_onevision_stream`。LLaVA 视频处理为 384×384，实际每帧生成 196 个视觉 token。CPU 接口测试共 24 项通过。

- OVO 1117 单视频/单题，visual K=64：AdaKV、PyramidKV、Uniform、ProtoKV、`full` 分别完成真实推理并通过各自审计。前四种发生 1 次压缩，`full` 为零驱逐；每种都有 1 次问答后缓存恢复。
- LU-KV 使用与评测集互斥的 1 视频/1 题 LLaVA 小校准，生成模型指纹绑定的 projected-utility profile。相同 OVO 单题 K=64 推理完成，发生 1 次压缩，审计通过。此 profile 只用于接口验证，不代表可靠的正式离线预算。
- OVO 1325 的 26 秒视频前缀输入 13 帧；Uniform K=64 完成 13 次视频更新和压缩，审计通过。该题预测错误，因此这些小样本只证明运行链路，不用于比较准确率。
- SparseMM 使用显式生成、与 LLaVA 配置指纹绑定的**合成常数分数**完成同一 OVO 单题，触发 1 次压缩并通过审计；汇总标记 `synthetic_smoke_only`。这只验证第七个方法入口的在线链路，没有 OCR 校准，也不是论文方法成绩。默认 Qwen 分数在创建运行目录前被拒绝。正式 LLaVA 测评仍需适配并完成真实 SynthDOG OCR 校准。
- 七种方法在单个 `python -m kvbench run --methods ... --dry-run` 命令中完成协议与资产路由检查，记录位于 `/tmp/kvbench_llava_all7_dryrun_20261015/`；dry-run 不执行推理，也不应显示审计通过。

本机完整运行记录位于 `/tmp/kvbench_llava_uniform_e2e_20261015/`、`/tmp/kvbench_llava_methods_e2e_final_20261015/`、`/tmp/kvbench_llava_lukv_e2e_20261015/`、`/tmp/kvbench_llava_sparsemm_smoke_e2e_20261015/` 和 `/tmp/kvbench_llava_multiframe_e2e_20261015/`。临时 LU profile 与证据分别位于 `/tmp/kvbench_llava_lukv_assets_20261015/`、`/tmp/kvbench_llava_lukv_calibration_20261015/`；SparseMM 合成测试分数位于 `/tmp/kvbench_llava_sparsemm_smoke_asset_20261015/`，均未作为正式资产随仓库分发。全量实验仍暂停。

## HERMES 与 DP 显式入口（2026-10-23）

统一注册表新增 `dp`、`dp_dual` 与 `hermes`，均不进入默认五方法批量运行。`dp` 固定 K=500、使用随目录提供的 DP `capacity_n` 表和原 SparseMM 在线选择器，协议明确标记 `recent_r/history_q` 未在线使用；`dp_dual` 固定 K=500，在线执行 `recent_r/history_q` 双缓存并独立审计逐头 token 索引、近期窗口和容量；`hermes` 调用原始 Qwen2.5-VL HERMES 推理代码，使用其原生 `kv_size` 和 float16，单独标记预算语义。三者仍受 3 视频、8 题、120 秒限制。DP 方法可显式设置 151200 像素上限以与旧 K=500 协议一致。

### HERMES 块级输入对照

`kvbench run` 在包含 HERMES 的 Qwen 方法集合中自动将其他方法切到 `hermes_chunk`；旧单方法默认仍是 `temporal_patch`。九个方法的 K=500/FP16 无 GPU 预检已通过。实际视频的 CPU 预处理核对显示，3 帧整块与 2+1 帧逐 patch 编码得到完全相同的 Qwen 像素 patch。

单视频/单题真实验证：默认 OVO 1117 的 1 帧，Uniform 与 HERMES 均获得相同网格 `[1,78,102]` 和 1,989 个视觉 token；裁剪到 6 秒的 OVO 1325 机制样本含 3 帧，两者均只进行一次块级更新，网格 `[2,52,92]`、2,392 个 token 相同。两个实验的双方审计及集合级 `visual_input_comparison.json` 均通过。DP-dual 也在相同 3 帧样本上完成块级推理及独立审计。完整本机证据在 `/tmp/kvbench_hermes_chunk_k500_e2e_20261023c/`、`/tmp/kvbench_hermes_chunk_three_frame_e2e_20261023a/` 和 `/tmp/kvbench_hermes_chunk_dp_dual_three_frame_20261023a/`；6 秒题目只用于机制验证，不计入 OVO 准确率。现有审计核对尺寸、网格和 token 数，尚未逐值核对视觉特征或压缩后位置编号；K=500 的实际占用语义仍应分别报告。全量实验保持暂停。

接口与固定预算相关 CPU 测试 26 项、双缓存策略 CPU 测试 26 项通过；`dp`、`dp` 加 151200 像素上限、`dp_dual`、`hermes` 各完成 1 视频/1 题 dry-run 协议生成。`dp_dual` 的 dry-run 分配表经独立审计入口的配额规则检查，总计 56,000 个视觉槽位；HERMES 源码可在当前环境导入。dry-run 不加载模型、不产生预测或审计通过结论。验证时唯一 A100 正被另一项已启动的 OVO 实验占用，因此未启动这三个新入口的 GPU 端到端推理。全量实验仍保持原状态。

## 内置 HERMES 与近期 1/4 帧方法（2026-10-23）

HERMES Qwen2.5-VL 压缩实现已复制到 `kvbench/hermes_backend/`。`qwenvl_hermes.py` 与原文件的 diff 只有三处包内 import；`abstract_hermes.py` 与 `reindex_3d.py` 字节一致。统一适配器不再向 `sys.path` 插入外部 `hermes` 源码路径，运行协议会记录三份内置源码 SHA-256。OVO `recent1or4_history_cap4000` 在独立的 `run-ovo` / `audit-ovo` 入口下注册，使用仓库内冻结的校准 CSV，保持既有九类题 runner 的源码不变。该入口只接受显式指定的 1–8 题、最多 3 段源视频；已有 394/1468 道题的暂停全量目录没有被重启或改写。

本机真实单题链路：`/tmp/kvbench_hermes_internal_e2e_20261023/` 使用内置 HERMES 完成 OVO 1117 的 MCQA 适配题，预测 A、答对 1/1，触发 1 次压缩和 1 次问答缓存恢复，审计通过；`/tmp/kvbench_recent1or4_e2e_retry_20261023/` 使用九类题原始提示词完成 OVO 1117，1 帧，答错 0/1，预算、源码快照和结果审计通过。两者的问题提示词和评分接口不同，不能据此比较准确率。初次 OVO 新入口测试缺少 `lmms-eval` 子进程路径，在模型加载后中止且未写入答案；已修复环境传递并在新目录重跑成功。24 项统一接口及新 OVO CPU 测试通过。两个单题测试都不代表全量实验成绩。
