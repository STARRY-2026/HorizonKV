# OVO-Bench：近期四帧 DP 双缓存

本目录保存 2026-10-21 完成 OVO-Bench 官方 3035 题实验的 **`dp_recent4`** 方法。核心入口是 [`recent4.py`](recent4.py)，评测入口是 [`run_ovobench_recent4.py`](run_ovobench_recent4.py)，只运行此方法的双进程控制器是 [`supervise_ovobench_recent4_only.py`](supervise_ovobench_recent4_only.py)。这些 Python 文件及其共同依赖从该次运行时的源码逐字节复制；`run_ovobench_recent4.py` 会在每次运行的协议中记录源码和资源 SHA-256。

## 方法

Qwen2.5-VL 的一个视觉时间单元通常包含连续采样的两帧。每个 KV 头的近期区保留最后 **两个时间单元**，通常对应四张真实帧；奇数帧终点会由模型补齐最后一个单元。每次新单元进入前，根据实际视觉 token 数更新近期区容量。因此近期区的 token 数随图像尺寸变化，并非固定四个 token，也不是固定每头 500 token。

总视觉 KV 目标是 28 层 × 4 个 KV 头 × 每头平均 500 = **56,000 个槽位**。除近期区以外的剩余槽位，按照冻结的 DP 历史配额 `history_q` 比例在全部头之间分配，使用最大余数法保证整数总量。近期区按时间滑动；离开近期区的 token 进入历史候选集，历史区满时按 SparseMM 的在线注意力分数驱逐。最后 32 个受保护 token 已包含在近期区内，不额外占用预算。文本前缀单独保留。短缓存阶段沿用该次实验的 `legacy_global` gate。

预算来源：[`history_weights_fixed_capacity.csv`](../assets/dp_recent4/history_weights_fixed_capacity.csv)，SHA-256 `957a8d89b79cae86d1390478a451e658932de5e4a990b9fa4813c353c815b317`。它提供 `history_q` 权重，旧 CSV 中的逐头容量上下限**不用于**这一版动态分配。评测协议仍会校验旧容量表；其原始副本是 [`legacy_capacity_reference.csv`](../assets/dp_recent4/legacy_capacity_reference.csv)，SHA-256 `fa164606d409a30acda7a6c881a9af24a2e956d7d494ff0417fc03aa90acff84`。SparseMM 视觉头分数和校准证明沿用 [`assets/sparsemm/`](../assets/sparsemm/)，分数 CSV SHA-256 `9cc0f47f24a0af4306ce7367806516d119dc4f57df77d064a3eac59c05b69dab`。

## 已完成实验

模型为 Qwen2.5-VL-7B-Instruct，BF16；OVO-Bench 官方 12 类、3035 题、644 个源视频；源视频按 0.5 fps 取样，图像最大像素 151200；视频按 16 帧读取，生成最多 16 token；采用官方问题模板和离线评分。相同视频的多题复用已处理的视频前缀，每次答题的临时 KV 在回答后恢复。两个 GPU 进程运行，已完成全部 3035 题。

- 逐题正确：**1439/3035 = 47.41%**。
- 官方三组等权总分：**52.43%**。
- 历史和近期的实际容量是逐头、随输入变化的；500 是全部视觉 KV 头的平均目标，不代表每头固定保留 500。

完整逐题预测、视频、模型权重和运行日志不上传到源码仓库。该实验的 `protocol.json` 可用源码 SHA-256、模型配置 SHA-256 和两张预算表 SHA-256 核对来源。

## 运行依赖

评测脚本沿用原实验的路径布局，要求 `kvbench/hermes` 指向包含 Qwen2.5-VL-7B、OVO-Bench 标注和视频、SparseMM 完整校准证据的 HERMES 工作目录。标注、模型与校准 manifest 内含冻结的路径和哈希；换机器时需要重新配置这些资源，单独克隆本仓库不足以直接启动 GPU 全量推理。两张预算 CSV 已随本提交保存；原脚本从 HERMES 的历史实验目录读取它们，以保持已验证协议的路径和 SHA-256 一致。

在完成资源配置并安装 [`requirements/runtime.txt`](../requirements/runtime.txt) 的环境后，从 `kvbench/` 目录运行单题核验：

```bash
PYTHONPATH="$PWD:$PWD/sparsemm_baselines/upstream/lmms-eval" \
python -m sparsemm_native.run_ovobench_recent4 \
  --arm dp_recent4 --output-root /path/to/new-output \
  --ids 31 --budget 500 --fps 0.5 --chunk-size 16 \
  --max-new-tokens 16 --video-max-pixels 151200
```

全量单进程可将上例的 `--ids 31` 去掉，并增加 `--max-new 100000`。已完成实验的双进程控制器入口为 `python -m sparsemm_native.supervise_ovobench_recent4_only --output-root /path/to/new-output`；它只启动近期四帧 DP 推理，但其历史对照报告预检还需要原实验的 SparseMM、旧 DP 结果目录。这些结果不随源码上传。运行前须确认 GPU 空闲及资源路径。
