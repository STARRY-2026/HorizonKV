# KVBench

统一管理 Qwen2.5-VL 的 AdaKV、PyramidKV、LU-KV、SparseMM、Uniform 和 ProtoKV，提供同一套运行、审计、汇总接口。方法源码、固定版本的 SparseMM 上游框架、LU profile、SparseMM 分数资产与当前评测依赖副本均在本目录。内部保留原 Python 包名，以兼容已验证的实现和源码快照。

OVO-Bench 全量实验使用的近期四帧 DP 双缓存实现、冻结预算表和运行说明见 [sparsemm_native/RECENT4_DP.md](sparsemm_native/RECENT4_DP.md)。

当前默认使用 **Qwen2.5-VL-7B-Instruct / BF16 / 原分辨率 / 0.5 fps / visual K=1000**。默认方法组为五种非 ProtoKV 方法。每次运行限制最多 3 个视频、8 道题、120 秒视频前缀；全量实验继续暂停。

GitHub 版本位于 `HorizonKV/kvbench/`。克隆时使用 `git clone --recurse-submodules`，随后进入 `kvbench/` 运行。顶层 HorizonKV 仓库提供 HERMES 评分脚本，KVBench 会自动建立指向它的本地 `hermes` 资源链接。GPU 环境先按顶层 [README](../README.md) 配置；模型权重、视频和部分冻结校准证据需要另行提供，详见下文资源依赖说明。

## 使用

在项目目录执行，下列 `PY` 使用已经验证的 Python 环境：

```bash
cd /path/to/HorizonKV/kvbench
PY=python

# 查看统一方法注册表，不加载模型。
$PY -m kvbench methods

# 单方法小规模运行，结束后自动审计。
$PY -m kvbench run --method adakv --output-dir results/adakv_example

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
| `full` | AdaKV 路径使用预算 1,000,000；审计必须证明零驱逐 | 否，仅显式选择 |

LU-KV 保留原效用目标中的 Value/输出投影，不使用 HERMES 打分；profile 默认来自已完成的纯注意力校准。可用 `--lukv-profile-path` 指定同格式 profile，`--sparsemm-score-path` 指定具有完整校准证据的 SparseMM CSV。`full` 只表示当前小规模输入上的无驱逐对照，不能推断任意长视频都不会压缩。

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

`configs/ovobench_smoke.json` 是上一次端到端测试的同一组 3 个官方问题（FPD 1117、EPM 34、OCR 1325）。每题独立初始化缓存，严格使用问题时刻之前的视频前缀。标注已经转换为统一 MCQA 格式，并引用现有原视频；本次整合没有新增全量 OVO 任务适配。

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
  kvbench/                  # 统一 CLI、Python API、方法注册表、资源路径
  run.py                    # 可从任意目录调用的入口
  configs/                  # 默认 OVO 小规模测试标注
  assets/sparsemm/          # 分数、原始观察统计和冻结清单
  sparsemm_baselines/       # AdaKV/PyramidKV 和固定 upstream（含 .git）
  sparsemm_protokv/         # 共享缓存、Qwen2.5-VL 插件、lmms任务和 runtime_deps
  sparsemm_lukv/            # LU-KV 与 attention profile
  sparsemm_native/          # SparseMM
  sparsemm_uniform/         # Uniform
  protokv/                  # ProtoKV 原型库核心与参考状态
  requirements/
  tests/
  MIGRATION_MANIFEST.json   # 原文件路径、复制文件及 SHA256
  hermes -> ../hermes       # 外部资源入口
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
