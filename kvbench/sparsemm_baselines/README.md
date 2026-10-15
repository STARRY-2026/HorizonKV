# 用 SparseMM 框架复现 AdaKV / PyramidKV

本目录以 [SparseMM](https://github.com/CR400AF-A/SparseMM) 固定提交 `33f2f0ab96d1d99516bcd2092caccab60516ff2c` 为来源，为 AdaKV 和 PyramidKV 提供统一入口。完整官方源码保存在 `upstream/`，没有修改其文件。

ProtoKV 的原生框架接入与当前 Qwen2.5-VL 适配已放在 [sparsemm_protokv](../sparsemm_protokv/README.md)：使用此处原版 lmms-eval 的插件入口切换三方法，已通过真实小规模验证；全量仍暂停。

**实现、原始算法一致性和本地数据适配已完成；两种方法的真实 7B 扩展验证均通过运行检查。完整评测和 ProtoKV 仍保持暂停，没有启动或排队全量任务。** 本次完成 6 组、58 次问答，主测试覆盖 9 个视频、23 道题和 10 类任务；这是选择的稳定性子集。详见 [扩展验证报告](results/extended_validation/combined_report.md)。

## 两个运行入口

| 入口 | 执行内容 | 模型/环境 |
| --- | --- | --- |
| `run_lmms_eval.py` | 调用官方 SparseMM 的 `lmms-eval`、原始 monkeypatch 和压缩器，按 `METHOD` / `BUDGET` 切换方法 | 官方 Qwen2-VL 或 LLaVA；独立旧版环境 |
| `run_streamingbench.py` | 将同一 SparseMM 基线算法及变长缓存方式接到本地视频数据，逐视频块压缩、临时问答后恢复缓存 | 本地 Qwen2.5-VL-7B；现有 `hermes-qwen` 环境 |

视频入口独立实现 Qwen2.5-VL 的 decoder、GQA、M-RoPE、物理变长 KV 和答案解码；不导入 HERMES 的推理器、attention、压缩策略，也不使用此前 `adakv/` 或 `kvcache_factory/` 的算法模块。它只沿用本地数据格式、选择题提示和评测脚本。SparseMM 原版的 Qwen2-VL / transformers 4.46.2 接口不能直接应用于 Qwen2.5-VL / transformers 4.57.1，因此这里显式提供模型移植层。

## 与 SparseMM 保持一致的基线设置

| 配置 | AdaKV | PyramidKV |
| --- | --- | --- |
| 观察窗口 | 32 | 32 |
| 池化 | avgpool，核 5 | avgpool，核 5 |
| GQA 评分聚合 | query 位置均值后，KV 组内均值 | 同左 |
| 容量分配 | 同层跨 KV head 自适应，normalization=True，floor=0.2 | 随层深递减，beta=20 |
| 整数规则 | 原始 floor + round | 原始整除层间步长 |
| 排序 | 原始 Torch sort/topk 行为 | 原始 Torch topk 行为 |

`core.py` 从未修改的 `upstream/sparsemm/sparsemm_utils.py` 加载原始 `AdaKVCluster` 和 `SnapKVCluster(pyram_mode=True)`，并将实际选出的 K/V 与视频扩展逐项比较。AdaKV 的 normalization 是先用每头高分历史位置的注意力质量占比调整评分，再跨头分配容量。

默认 `budget=1000`、`budget_scope=prompt`：初始系统文本和视觉 KV 共用预算，最近窗口也计入预算。AdaKV 的 1000 是同层每头的基准平均值，个别 head 可更长；原始整数取整会产生少量偏差。PyramidKV 在 Qwen 的 28 层下，长输入时每头容量从首层 1920 递减到末层 84，平均 **1002**，总计 **112224** 个 KV 位置。没有为强制达到精确 1000 而改写分配公式。

若要沿用其他本地视频方法的“1000 视觉 KV、系统文本另计”口径，请显式选择 `--budget-scope visual`。此模式完整保留每头初始文本，仅对视觉历史池化和分配容量；两种预算口径使用不同输出目录，结果不能混合。

## 视频适配的明确差异

- 数据读取按最多 16 帧分组；模型按一个 temporal patch（2 帧）前向和压缩。原 SparseMM 只在完整 prompt prefill 时压缩，视频入口将这一机制扩展到每个视觉块。
- 原图像评测的问题已包含在完整 prompt 中。视频压缩只使用当前视频查询，真实问题在回答时临时追加，因此不会在早期视频压缩中使用未来问题。
- 保留 SparseMM 的“当前块使用完整当前 KV 计算 attention，再保存压缩缓存”行为。历史位置按原始分数顺序保存，并将最近窗口追加在末尾。
- Pyramid 层预算基于累计已见 token 重新计算；每个视频重置状态。这样避免原版首个样本的 `pyram_init` 和长度裁剪影响后续视频。
- 视频 M-RoPE 的时间轴取实际已处理的采样帧时间，空间轴使用固定初始原点；剪枝、问答或分辨率变化不会重置视频时间。问题所在块只有奇数帧时，由原生处理器补齐 temporal patch，下一块仍按已消费原始帧数推进时间。
- 默认 Qwen 使用权重原生的 BF16，注意力重要性评分使用 FP32，并关闭评分区域的外部 autocast，避免此前 FP16 QK 溢出。`--score-precision input` 可选择输入精度。非有限 Q/K 或评分会明确报错；历史注意力质量恰为零时，normalization 比率定义为 0，并记录该事件。
- 缓存采用 SparseMM 的每头连续展平布局，通过 FlashAttention varlen 计算。视频多 token 追加使用 PyTorch 拼接，以适配现有 torch ABI；官方单 token `tiny_api_cuda` 仍由原生评测入口使用。
- 问答成功或异常都会恢复缓存张量引用、位置和视频游标。预算约束压缩后持久缓存；当前块前向及答案生成期间存在临时额外 KV。

因此视频结果应标为 **AdaKV / PyramidKV via SparseMM, streaming adaptation**。原始图像论文数字应通过原生入口单独评测。

## 本地数据与验证

- 标注：`/home/yujun/program/hermes/data/streamingbench/streamingbench_realtime_half_userpaths.json`，250 视频、1255 题；两个方法的数据路径、选项和时间顺序审计均通过。
- 模型：`/home/yujun/program/hermes/models/Qwen2.5-VL-7B-Instruct`。
- 采样：0.5 FPS；为保持本地数据协议一致，源 FPS 四舍五入后用整数步长抽帧；问答截止位置为 `ceil(end_time * sample_fps)`。
- 原生视觉处理器，不强制缩放到固定画布。
- 运行时：`/home/yujun/.conda/envs/hermes-qwen/bin/python`，torch 2.9.1+cu130、transformers 4.57.1、flash-attn 2.8.3。

以下命令只检查配置与数据，不导入模型或使用 GPU：

```bash
python3 /home/yujun/program/sparsemm_baselines/run_streamingbench.py --method adakv --dry-run
python3 /home/yujun/program/sparsemm_baselines/run_streamingbench.py --method pyramidkv --dry-run
```

测试使用固定源码和微型真实 Qwen，无需下载模型：

```bash
PYTHONPATH=/home/yujun/program \
/home/yujun/.conda/envs/hermes-qwen/bin/python -m unittest \
  sparsemm_baselines.test_core sparsemm_baselines.test_adapter \
  sparsemm_baselines.test_runner -v
```

49 项 CPU 测试全部通过，覆盖原始压缩器 K/V 一致性、归一化与层预算、不等长历史、GQA、FP16 溢出、原生 SDPA 输出、M-RoPE、前缀保护、问答异常恢复、跨视频重置、视频时间轴、数据映射、完整视频续跑、暂停保护及真实冒烟审计。另有 6 组小张量 CUDA FlashAttention 检查通过，与手工注意力和独立 CPU FP64 参考比较，未加载模型权重。验证证据保存在 `results/validation/`；每个方法的 dry-run 清单保存在各自结果目录。

新增 11 项 CPU 测试通过，验证扩展轨迹审计、最终隐藏状态与 logits 有限值检查，以及问答期间缓存原位改动的检测。日志在 `results/extended_validation/helper_tests.log`。

## 扩展真实模型验证

2026-10-14 使用相同源码、模型和数据完成以下测试：

| 预算 / 范围 | 每种方法的测试规模 | AdaKV 答对 | PyramidKV 答对 |
| --- | --- | --- | --- |
| 1000 / prompt | 9 视频、23 题，覆盖 10 类任务，最长前缀 502 秒 | 20/23 | 21/23 |
| 64 / prompt | 1 视频、3 题 | 3/3 | 3/3 |
| 256 / visual | 1 视频、3 题 | 2/3 | 2/3 |

共 58 次问答运行、624 次视频压缩，六组独立轨迹审计均通过。主测试与 visual 测试的 52 次问答均验证了缓存张量引用和内容、位置、长度及视频游标完全恢复；1120 次 decoder 最终 norm 输出和 518 次问答 logits 检查均为有限值。64-token 两组没有执行这些额外检查。边界测试使用原始截止时间 2、80、80 秒，覆盖奇数帧补齐和重复时间点回答。

分数只描述所选子集，不能代表全量准确率；额外检查会克隆缓存，显存峰值仅用于诊断。完整配置、逐视频结果与检查范围见 [扩展报告](results/extended_validation/combined_report.md) 和 `combined_report.json`。重新核验全部六组结果：

```bash
PYTHONPATH=/home/yujun/program \
/home/yujun/.conda/envs/hermes-qwen/bin/python -m sparsemm_baselines.summarize_extended --check-only
```

真实运行可使用 `run_instrumented.py --validation-report OUTPUT/runtime_checks.json`，其余参数与 `run_streamingbench.py` 相同。此入口保留暂停检查和 GPU 锁；仅明确授权的测试才传入 `SPARSEMM_RESUME=1`。每组结束后运行 `python -m sparsemm_baselines.audit_extended OUTPUT --min-compression-events-per-video 2`。

## 恢复实验时的命令

当前全量暂停标记保持有效。两组单题冒烟已于 2026-10-14（北京时间）执行并通过：AdaKV 峰值张量显存 15.78 GiB，PyramidKV 15.79 GiB；实际压缩、答案生成和问答后的缓存恢复均正常。结果分别在 `results/adakv_smoke_k1000_prompt/` 和 `results/pyramidkv_smoke_k1000_prompt/`。下面保留独立运行入口，完整评测尚未执行。

```bash
# 先分别跑一个真实问题
SPARSEMM_RESUME=1 SPARSEMM_METHOD=adakv \
bash /home/yujun/program/sparsemm_baselines/run_smoke.sh

SPARSEMM_RESUME=1 SPARSEMM_METHOD=pyramidkv \
bash /home/yujun/program/sparsemm_baselines/run_smoke.sh

# 冒烟结果及缓存审计通过后，再分别评测前 250 视频
SPARSEMM_RESUME=1 SPARSEMM_METHOD=adakv \
bash /home/yujun/program/sparsemm_baselines/run_first250.sh

SPARSEMM_RESUME=1 SPARSEMM_METHOD=pyramidkv \
bash /home/yujun/program/sparsemm_baselines/run_first250.sh
```

也可在恢复后用 `SPARSEMM_RESUME=1 bash /home/yujun/program/sparsemm_baselines/run_comparison.sh` 顺序执行两种方法。它先完成两个真实冒烟与 `validate_smoke.py` 的缓存审计，再开始两组完整评测；结果写入 `results/comparison/`。任何一步失败即停止后续步骤。查看暂停状态可执行 `bash /home/yujun/program/sparsemm_baselines/status.sh`。

入口先检查暂停状态，之后以非阻塞方式取得 `/tmp/hermes-gpu0.lock` 并检查至少 28000 MiB 空闲显存。GPU 忙时直接退出，不排队。结果分别输出到 `results/adakv_first250_k1000_prompt/` 和 `results/pyramidkv_first250_k1000_prompt/`。选择视觉预算时设置 `SPARSEMM_SCOPE=visual`。

每个目录包括独立 CSV、准确率评测、逐层逐头实际缓存长度、运行配置和源码/模型配置/运行库哈希。只有完整视频会写入续跑 CSV；加载模型前会核对已有题目、答案、预测解析、准确率、配置和源码版本。

## 原生 SparseMM 图像评测

```bash
# 仅生成并校验官方框架命令
python3 /home/yujun/program/sparsemm_baselines/run_lmms_eval.py \
  --method adakv --budget 256 --model qwen2_vl \
  --model-args pretrained=/path/to/Qwen2-VL-7B-Instruct \
  --tasks ocrbench --dry-run
```

将 `--method` 改为 `pyramidkv` 即使用同一官方入口切换方法。正式执行需要指定匹配旧版依赖的 `--python`，准备本地 Qwen2-VL/LLaVA 权重与评测数据，并明确传入 `--resume-experiments`。本机目前只有 Qwen2.5-VL 和 LLaVA-OneVision 权重，没有原生 Qwen2-VL 权重或专用旧版环境，所以原生入口目前只完成 dry-run。

依赖记录分别在 `requirements-native.txt` 和 `requirements-streaming.txt`。原生安装需注意上游两个包对 NumPy 的锁定不一致，具体处理写在原生依赖文件注释中；不要将旧版 transformers 安装到现有 Qwen2.5-VL 环境。
