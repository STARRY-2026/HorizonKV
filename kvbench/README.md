# KVBench

统一管理 Qwen2.5-VL 与 LLaVA-OneVision 的 AdaKV、PyramidKV、LU-KV、SparseMM、Uniform 和 ProtoKV，并提供显式的 DP 容量对照、内置 HERMES Qwen 后端及独立 OVO 九类题入口。方法源码、固定版本的 SparseMM 上游框架、LU profile、SparseMM 分数资产与当前评测依赖副本均在本目录。内部保留原 Python 包名，以兼容已验证的实现和源码快照。Qwen3-VL-8B 的窗口化实验变体已有除 ProtoKV 外九种方法的双数据集扩展验证：每种方法在 OVO-Bench 官方选择题的统一格式子集和 StreamingBench 各测 3 题，视觉网格与 token 数逐块一致，18 组运行和审计通过。LU 和注意力排序 DP 的资产仍来自原生视觉拓扑；SparseMM 的窗口化变体完成 1000 图校准。这些运行不能作为正式方法效果比较，详见 [VALIDATION.md](VALIDATION.md)。

StreamingBench 默认使用 **Qwen2.5-VL-7B-Instruct / BF16 / 原分辨率 / 0.5 fps / visual K=1000**，默认方法组为五种非 ProtoKV 方法；其小规模运行限制最多 3 个视频、8 道题、120 秒视频前缀。独立的 `run-ovo` 入口要求显式选择最多 8 道题、3 段源视频。全量实验继续暂停。

GitHub 版本位于 `HorizonKV/kvbench/`。克隆时使用 `git clone --recurse-submodules`，随后进入 `kvbench/` 运行。HERMES 压缩实现已放入 `kvbench/hermes_backend/`；顶层 HorizonKV 仓库仍提供评分脚本及模型和数据资源，KVBench 会自动建立指向它的本地 `hermes` 资源链接。GPU 环境先按顶层 [README](../README.md) 配置；模型权重、视频和部分冻结校准证据需要另行提供，详见下文资源依赖说明。

## 使用

旧紧凑历史 K500 的 SimpleStream 实验使用独立入口 `run_simplestream_dscache_history64.py --arm pae_history_k500`；运行环境、固定上游提交、外置基线输入和预检命令见 [专项说明](sparsemm_native/DSCACHE_HISTORY64.md)。

在项目目录执行，下列 `PY` 使用已经验证的 Python 环境：

```bash
cd /path/to/HorizonKV/kvbench
PY=python

# 查看统一方法注册表，不加载模型。
$PY -m kvbench methods

# 单方法小规模运行，结束后自动审计。
$PY -m kvbench run --method adakv --output-dir results/adakv_example

# DP 离线逐头容量 + SparseMM 在线注意力驱逐；使用固定 K=500 资产。
$PY -m kvbench run --method dp --output-dir results/dp_capacity_example

# DP 双缓存：recent_r 滑动窗口 + history_q 注意力排序；固定 K=500。
$PY -m kvbench run --method dp_dual --output-dir results/dp_dual_example

# 原始 HERMES 预测式压缩；原实现加载 float16，预算语义单独标记。
$PY -m kvbench run --method hermes --dtype float16 --output-dir results/hermes_example

# Qwen3-VL-8B 的 HERMES 层分组适配（浅层 10%、深层 30%、其余中层）
$PY -m kvbench run --method hermes --model-path /home/yujun/program/hermes/models/Qwen3-VL-8B-Instruct \
  --dtype bfloat16 --budget 500 --max-videos 1 --max-questions 1 --dry-run \
  --output-dir results/qwen3_hermes_dryrun

# HERMES 块级输入协议的 K=500 小规模对照：两种方法都按评测器的整块
# （默认最多 16 帧）编码和更新 KV，运行后核对每块的尺寸、网格和 token 数。
$PY -m kvbench run --methods uniform,hermes --budget 500 --dtype float16 \
  --output-dir results/hermes_chunk_k500_smoke

# OVO 九类题的近期 1/4 帧 + 封顶历史配额；必须显式指定小规模题目 ID。
$PY -m kvbench run-ovo --method recent1or4_history_cap4000 \
  --ids 1117 --output-dir results/recent1or4_smoke

# 使用已下载的 LLaVA-OneVision 7B；后端根据模型 config.json 自动选择。
$PY -m kvbench run --method uniform \
  --model-path /home/yujun/program/hermes/models/llava-onevision-qwen2-7b-ov-hf \
  --max-videos 1 --max-questions 1 \
  --output-dir results/llava_uniform_smoke

# 批量运行五种方法，串行使用 GPU，结束后逐项审计并汇总。
$PY -m kvbench run \
  --methods adakv,pyramidkv,lukv,sparsemm,uniform \
  --allow-calibration-transfer \
  --output-dir results/five_methods_example

# 仅生成各方法协议与命令，不进行推理。
$PY -m kvbench run --methods adakv,pyramidkv,lukv,sparsemm,uniform \
  --allow-calibration-transfer --dry-run --output-dir results/dryrun_example

# 对已经完成的结果重新审计、汇总；不重新推理。
$PY -m kvbench audit --output-dir results/five_methods_example --allow-calibration-transfer
$PY -m kvbench summarize --output-dir results/five_methods_example
```

`--output-dir` 是一次比较实验的根目录，必须使用新路径。即使只运行一个方法，也会在该目录下创建相应方法子目录。省略 `--methods` 默认选择上述五种方法；在默认 OVO 子集上运行 LU-KV 时需要 `--allow-calibration-transfer`，明确启用对冻结 profile 的跨数据集审计。它不会进行新的校准。

Qwen2.5-VL 的全部十种方法支持 `--fixed-frame-size 896`：同一采样帧先等比 bicubic 抗锯齿缩放，再居中填充 RGB 127 灰边到 896×896，编码器不再改变空间尺寸。原始画面的比例保留，视频长度不会改变画布。每个视觉时间单元为 1024 token；视频两个真实帧通常融合为一个单元，16 帧块为 8192 token，奇数块末尾按原处理器补帧。逐图编码时每张图为 1024 token。这个输入量与回答前的 K500 缓存预算分别记录，不能互相替代。

固定配置位于 `configs/ovobench_fixed896_all_methods.json`，覆盖 AdaKV、PyramidKV、LU-KV、SparseMM、Uniform、DP、DP 双缓存、HERMES、ProtoKV 和 Full。包含 HERMES 的同题比较沿用已有规则，统一使用 float16 和相同的整块视频编码粒度；Full 保持 1,000,000 的无驱逐控制预算。此预设使用现有 OVO 三题接口验证范围。先生成所有方法协议：

```bash
$PY tools/run_ovobench_fixed896.py --dry-run --output-dir results/ovo_fixed896_check
```

固定输入实验记录每块实际网格、token 数和准备后 RGB 哈希，多方法运行会逐项核对。`--fixed-frame-size` 与 `--video-max-pixels` 互斥；不传固定尺寸时保留各方法原输入配置。

包含 HERMES 的 Qwen 比较会自动选用 `video_chunk_mode=hermes_chunk`：其他方法也把最多 `--encode-chunk-size` 帧作为一次 Qwen 视频前向，并在块末更新/压缩 KV。单独运行旧方法仍默认 `temporal_patch`；可显式使用 `--video-chunk-mode hermes_chunk` 重跑同协议。旧结果与块级结果必须分别报告。每种方法的 `protocol.json` 记录实际粒度；多方法块级运行写出 `visual_input_comparison.json`，逐块核对源帧尺寸、处理后尺寸、视频网格和视觉 token 数。该审计尚不比较视觉特征张量值及语言模型位置编码；HERMES 压缩后的重排和预算口径也仍是其方法的一部分。LLaVA 使用逐帧视觉入口，不参与这条 HERMES 块级 Qwen 对照。

`--model-path` 支持 Qwen2.5-VL 与 LLaVA-OneVision，自动切换对应的注意力补丁、位置编码和视频特征入口。AdaKV、PyramidKV、Uniform、ProtoKV 和 `full` 可直接选择 LLaVA。LU-KV 需要用同一 LLaVA 模型生成冻结 profile 并传入 `--lukv-profile-path`；一个互斥视频生成的小 profile 已通过单题链路验证，只用于接口测试。SparseMM 的 LLaVA 在线路径也已跑通，但目前只能使用明确标记的合成分数资产做接口测试；`tools/make_sparsemm_smoke_scores.py` 可生成这种模型绑定资产，结果汇总会标记 `synthetic_smoke_only`。官方 SynthDOG OCR 校准与证据审计仍绑定 Qwen2.5-VL，现有默认分数不能跨模型使用；LLaVA 论文测评前必须补做真实 OCR 校准。LLaVA 的小规模验证详见 [VALIDATION.md](VALIDATION.md)，完整数据集结果尚未产生。

从任意工作目录也可调用该目录下的绝对入口：

```bash
python /path/to/HorizonKV/kvbench/run.py methods
```

本项目按源码目录运行；可选的控制台入口通过本机 editable 安装启用：`$PY -m pip install --no-deps --no-build-isolation -e .`，随后使用 `kvbench methods`。固定上游和依赖目录随源码目录使用；当前交付没有制作独立 wheel。

## 统一参数与方法

`run` 的常用参数：`--anno-path`、`--model-path`、`--budget`、`--max-videos`、`--max-questions`（每视频）、`--sample-fps`、`--encode-chunk-size`、`--dtype`、`--config-json`。运行始终使用 `visual` 预算，原控制器仍负责 GPU 锁、原生 lmms-eval、数值检查、问答缓存恢复和源码快照。`--min-compression-events-per-video 1` 可要求每个视频实际触发压缩；默认 0 允许短缓存合法跳过驱逐。

| 方法名 | 预算与保留策略 | 默认批量运行 |
|---|---|---|
| `adakv` | 注意力驱动的自适应头预算 | 是 |
| `pyramidkv` | 按层分配金字塔预算，注意力驱逐 | 是 |
| `lukv` | 冻结长期效用 profile 随输入长度分配头预算；纯注意力排序和驱逐 | 是 |
| `sparsemm` | 固定视觉头校准分数分配预算；在线注意力驱逐 | 是 |
| `uniform` | 各层各 KV 头相同预算；最近窗口加注意力 Top-K | 是 |
| `protokv` | 原型记忆流式适配 | 否，仅显式选择 |
| `dp` | 我们的 K=500 DP `capacity_n` 逐头分配，在线仍用 SparseMM 注意力驱逐；不使用 DP 的近期/历史拆分 | 否，仅显式选择 |
| `dp_dual` | 我们的 K=500 DP 近期/历史双缓存；历史 token 由注意力排序 | 否，仅显式选择 |
| `hermes` | Qwen2.5 原始 HERMES 后端；Qwen3 为预测查询压缩适配。`kv_size` 是每层各 KV 头共享的视觉位置目标，初始文本额外保留 | 否，仅显式选择 |
| `full` | AdaKV 路径使用预算 1,000,000；审计必须证明零驱逐 | 否，仅显式选择 |

`recent1or4_history_cap4000` 是独立的 **OVO-Bench 九类题**方法，使用 `run-ovo` / `audit-ovo`，不进入 StreamingBench 的 `run --methods`。它按 1 FPS 输入，每头近期独立重建最近 1 或 4 帧，历史最多保留离线分配的 4,000 个 token；112 个头的历史配额总计 108,069。视频内各题使用该视频最长作答前缀确定的统一 SimpleStream 缩放尺寸，并复用同源视频前缀。统一入口要求显式给出最多 8 道题、最多 3 段源视频、最多 120 秒视频前缀；官方全量运行仍暂停。`assets/recent1or4/` 保存其冻结校准 CSV；原有 OVO 运行源码和已落盘结果保持不变。

`dp` 对应此前 K=500 对照中的**旧 DP 容量注入版**；`dp_dual` 则在线执行 `recent_r/history_q` 两个缓存配额，并复用 OVO 双缓存审计的逐头索引与容量核验。两份 CSV 位于 `assets/dp/`，分别可用 `--dp-budget-path` 和 `--dp-dual-budget-path` 替换。`dp`、`dp_dual` 与 `sparsemm` 单独运行时可设置相同的 `--video-max-pixels 151200`，对齐有像素上限的 OVO 输入。HERMES 的 `kv_size` 是每层共享的视觉位置目标，DP 是逐 KV 头的视觉容量；初始文本在两者中都额外保留。加上 HERMES 固定使用 `float16`，汇总结果是同题系统级对照，不代表逐头策略或实际 KV 占用严格相等。这些显式入口均保持最多 3 视频、8 题、120 秒的限制。

Qwen2.5 HERMES 的压缩、位置重排及缓存源码已内置于 `kvbench/hermes_backend/`；只运行 `hermes` 且显式提供模型和标注路径时，不再需要外部 `hermes` 源码链接。Qwen3 适配在 `sparsemm_baselines/qwen3_hermes.py`，使用现有 Qwen3 流式输入桥接和 HERMES 的预测查询、层分组、注意力与时近性打分、跨层融合及逐层共享位置筛选。Qwen3 默认采用本项目的窗口注意力＋4 层全局注意力视觉变体，以保持与当前 Qwen3 校准相同的视觉拓扑；这是实验性模型变体，会记录在协议中。36 层按取整规则划为浅层 0–2、中层 3–24、深层 25–35。这里的 10% 和 30% 是层数比例，不是 KV 预算比例；`--budget 500` 表示每层保留最多 500 个视觉位置（深层按原算法减 1），并在每个 KV 头复用这些位置。Qwen3 适配保留被选中键的原 M-RoPE 位置，不执行原 Qwen2.5 代码的压缩位置重排，因此应单独标注为适配实现。模型权重和视频本身仍需提供。

LU-KV 保留原效用目标中的 Value/输出投影；默认 profile 来自纯注意力校准。Qwen3 另有可选的完整 HERMES 伪查询排序 profile：`results/qwen3vl8b_lukv_hermes_pseudo_10v_20261025b/profile.json`，其在线选择也使用伪查询打分，详见 [VALIDATION.md](VALIDATION.md)。可用 `--lukv-profile-path` 指定同格式 profile，`--sparsemm-score-path` 指定具有完整校准证据的 SparseMM CSV。`full` 只表示当前小规模输入上的无驱逐对照，不能推断任意长视频都不会压缩。

ProtoKV 在 StreamingBench 上可显式使用 `--video-max-pixels 200704` 限制 Qwen2.5-VL 每帧的预处理像素数。该参数也支持单独运行 Qwen2.5-VL 的 `sparsemm`、`dp` 或 `dp_dual`，但不支持多方法批量或 HERMES；默认不启用。它保留各方法的 FPS 与 K，但改变视觉输入，因此结果需单独标记，不能直接与无像素上限方法比较准确率。控制器在 `protocol.json` 中记录像素上限，`timing.json` 记录模型加载、视频处理及原型更新时间。ProtoKV 例子：

```bash
$PY -m kvbench run --method protokv \
  --anno-path /home/yujun/program/hermes/data/streamingbench/streamingbench_realtime_half_userpaths.json \
  --video-max-pixels 200704 --max-videos 1 --max-questions 1 \
  --output-dir results/protokv_pixelcap_example
```

2026-10-16 的首视频首题对照（6 帧，同一模型、问题和预算）：原分辨率 3588 个视觉 token、原型更新 6.45 秒、视频处理 13.39 秒；像素上限 200704 时为 756 个 token、原型更新 1.99 秒、视频处理 9.78 秒。两次都答对且通过状态恢复审计。视频处理加速约 1.37 倍，单题不足以估计整个数据集的速度或准确率。结果见 `results/protokv_native_timed_1q_20261016/` 与 `results/protokv_pixelcap_timed_1q_20261016/`。全量实验仍暂停。

## Python API

```python
from pathlib import Path
from kvbench import RunConfig, run, audit, summarize

result = run(RunConfig(
    methods=("adakv", "pyramidkv", "lukv", "sparsemm", "uniform"),
    anno_path=Path("configs/ovobench_smoke.json"),
    output_dir=Path("results/python_example"),
    budget=1000,
    max_videos=3,
    max_questions=1,
    allow_calibration_transfer=True,
))

# 同一个方法注册表及结果格式。
audit("results/python_example", allow_calibration_transfer=True)
summary = summarize("results/python_example")
```

## 数据与输出

`configs/ovobench_smoke.json` 是上一次端到端测试的同一组 3 个官方选择题（FPD 1117、EPM 34、OCR 1325）。选项和正确索引来自官方标注；每题独立初始化缓存，使用问题时刻之前的视频前缀。标注仅映射为统一 MCQA 字段与提示词，并引用现有原视频；本次整合没有新增全量 OVO 九类题适配。

```json
[
  {
    "video_id": "unique_video_or_query_id",
    "video_path": "/absolute/path/to/video.mp4",
    "conversations": [
      {
        "question": "Question text",
        "choices": ["Choice A", "Choice B"],
        "answer": "Choice A",
        "end_time": 10.0,
        "task": "MCQA"
      }
    ]
  }
]
```

同一视频内的问题按时间递增并共享视频缓存。OVO 独立问题应各自使用唯一 ID 和单独文档。当前加载器要求选项文本唯一，答案与选项精确一致；不能将 OVO 原始官方 JSON 直接传入。前向任务和官方分组计分尚未接入，当前得分用于小规模链路验证。

```text
results/<experiment>/
  suite_manifest.json        # 方法选择、接口来源、运行状态
  summary.json               # 统一机器可读汇总
  report.md                  # 统一结果表
  adakv_k1000_visual/        # 其他方法使用相同目录结构
    kvbench_controller.log
    kvbench_audit.log
    protocol.json
    run_manifest.json
    source_snapshot/
    1_0.csv                  # 逐题预测
    cache_trace.jsonl
    runtime_checks.json
    audit_report.json
    summary.json
    lmms/
```

## 文件布局与资源依赖

```text
kvbench/
  kvbench/                  # 统一 CLI、Python API、方法注册表、内置 HERMES 后端
  run.py                    # 可从任意目录调用的入口
  configs/                  # 默认 OVO 小规模测试标注
  assets/sparsemm/          # 分数、原始观察统计和冻结清单
  assets/recent1or4/        # OVO 近期 1/4 帧方法冻结分头校准表
  sparsemm_baselines/       # AdaKV/PyramidKV 和固定 upstream（含 .git）
  sparsemm_protokv/         # 共享缓存、Qwen2.5-VL 插件、lmms任务和 runtime_deps
  sparsemm_lukv/            # LU-KV 与 attention profile
  sparsemm_native/          # SparseMM
  sparsemm_uniform/         # Uniform
  protokv/                  # ProtoKV 原型库核心与参考状态
  requirements/
  tests/
  MIGRATION_MANIFEST.json   # 原文件路径、复制文件及 SHA256
  hermes -> ../hermes       # 模型、数据集、评分及历史证据资源入口
  results/
```

模型、视频、评分脚本和历史校准证据继续只读使用现有资源。模型在 `/home/yujun/program/hermes/models/Qwen2.5-VL-7B-Instruct`，OVO 在 `/home/yujun/program/hermes/data/ovobench/official_20261014`。LU profile 的冻结证据仍引用原 `/home/yujun/program/sparsemm_lukv/results/calibration/native_attention10v_20261014`；SparseMM 的协议、校准数据和历史源码证据仍引用 `hermes/results/myobservations/`。这些路径及其内容 SHA256 会被现有审计验证，整合保留证据原字节。

资源默认值集中在 `kvbench/paths.py`，支持 `KV_BENCH_RESOURCE_ROOT`、`KV_BENCH_MODEL_PATH`、`KV_BENCH_ANNO_PATH`、`KV_BENCH_LUKV_PROFILE_PATH`、`KV_BENCH_SPARSEMM_SCORE_PATH`。修改资源根时需显式调整 `hermes` 资源链接，接口会检查其目标一致。迁移到其他机器时还需迁移冻结证据及其路径绑定；仅复制此目录不足以运行外置模型和数据。

本机验证使用 `/home/yujun/.conda/envs/hermes-qwen/bin/python`（Python 3.12/Linux）。核心版本见 `requirements/runtime.txt`，评测补充依赖见 `sparsemm_protokv/requirements-eval.txt`。本机的 `runtime_deps` 是环境副本，GitHub 不上传；克隆后需要安装对应依赖。不要按上游旧 Qwen2-VL 的依赖文件覆盖当前环境。

旧项目目录与历史结果保持原位。新运行只写入本项目指定的新结果目录，保留全量暂停标记；ProtoKV 只有显式指定时才会启动。

项目已改名为 `kvbench`。旧名 `sparsemm_suite` 在本机保留为兼容软链接，供冻结实验记录中的原始绝对路径使用；当前运行命令和 GitHub 仓库根目录使用 `kvbench`。历史协议、缓存轨迹、源码快照与审计报告保留原字节。

## 整合验证

整合阶段 19 项接口 CPU 测试通过；加入嵌套仓库路径检查后，上传前 20 项通过。五种非 ProtoKV 方法在同一组 OVO 3 视频/3 题上完成真实推理与审计，15 个逐题预测与整合前完全一致。详细证据、测试命令和随仓库上传的精简结果见 [VALIDATION.md](VALIDATION.md)。

GitHub 上传范围、上游子模块和外部资源说明见 [GITHUB_UPLOAD.md](GITHUB_UPLOAD.md)。
