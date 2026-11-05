# SparseMM 原生 ProtoKV / Qwen2.5-VL 插件

本目录使用固定 SparseMM 源码中的原版 `lmms-eval` 扩展接口：`LMMS_EVAL_PLUGINS=sparsemm_protokv` 注册 `qwen2_5_vl_stream`，`METHOD` 切换 `protokv`、`adakv`、`pyramidkv`。参考源码仍在 `../sparsemm_baselines/upstream/`，没有修改。

插件针对当前 Transformers 4.57.1 和本地 Qwen2.5-VL-7B。它替换原生 `Qwen2_5_VLAttention.forward`，decoder、MLP、norm 仍由 Hugging Face 模型执行。普通 HF Cache 自动回落原实现。原版 Qwen2-VL / 4.46.2 入口继续保留。

当前已通过 32 项 CPU 检查和三方法真实小规模验证：每种方法 2 视频 / 4 题，ProtoKV 4/4，AdaKV 与 PyramidKV 各 3/4；12 次问答的数值与精确状态恢复检查通过。完整证据见 [验证报告](results/small_validation/report.md)。全量实验继续暂停。

## 数据与在线流程

`streamingbench_local` 将一个视频及其所有问题作为一个评测文档；视频内共享记忆，跨视频重置。原生评测器返回答案数组，任务按题数加权计分，并与逐题 CSV 独立核对。任务直接读取本地 JSON，不下载数据。

视频以 0.5 FPS、原生分辨率读取。语言模型按 Qwen temporal patch（通常两个采样帧）前向，随后按视觉 token 顺序更新记忆。M-RoPE 使用原始帧时钟和固定空间原点；奇数帧补齐不改变实际消费的帧数。

ProtoKV 复用 `protokv/full_state.py` 的全层全头原型库，直接保存 RoPE 前的 key。问题到来时临时合成伪 KV，按每头锚点旋转，并加入 `log(mass)` 与无效槽掩码；问题后缀前向和首答案 token 即使用这些状态。问答成功或异常均恢复视频缓存与游标，原型库保持只读。

## 小规模运行

Python：`/home/yujun/.conda/envs/hermes-qwen/bin/python`。评测依赖隔离在 `runtime_deps/`，版本见 `requirements-eval.txt`。安装时使用该 Python 的 `pip install --no-deps --target /home/yujun/program/sparsemm_protokv/runtime_deps -r /home/yujun/program/sparsemm_protokv/requirements-eval.txt`。

以下控制器实际启动原版 `python -m lmms_eval`，记录完整命令、依赖、源码与数据指纹，并持有共享 GPU 锁：

```bash
PYTHONPATH=/home/yujun/program \
/home/yujun/.conda/envs/hermes-qwen/bin/python -m sparsemm_protokv.run_streamingbench \
  --method protokv --budget 1000 --budget-scope visual \
  --anno-path /home/yujun/program/sparsemm_protokv/results/small_validation/annotations.json \
  --output-dir /home/yujun/program/sparsemm_protokv/results/my_protokv_check \
  --small-validation
```

改 `--method` 即切换对照方法；也可使用 `METHOD` / `BUDGET` 环境变量。`--dry-run` 只检查协议与生成命令。已有 CSV 目录拒绝覆盖。

全量实验继续暂停。控制器与原生模型入口的 `--small-validation` 限制为最多 3 视频、8 题、120 秒前缀；GPU 忙时直接退出，不后台排队。直接使用原生 CLI 时还需设置插件、依赖路径、`STREAMINGBENCH_ANNOTATIONS`，传入 `--include_path /home/yujun/program/sparsemm_protokv/tasks` 和模型参数 `small_validation=true`。

用户另行授权的五视频计时使用独立入口：

```bash
cd /home/yujun/program
/home/yujun/.conda/envs/hermes-qwen/bin/python -m sparsemm_protokv.run_streamingbench \
  --method protokv --five-video-timing --budget 1000 --budget-scope visual \
  --output-dir /home/yujun/program/sparsemm_protokv/results/five_video_timing/NEW_RUN
```

该入口固定原标注前五个完整视频（ID 3、5、6、11、12）和全部 25 题，source/subset SHA 双重校验，不接受截题或扩展视频。评测从视频起点处理到最后一个提问时间，预计 1019 个原始采样帧、515 次时间补丁前向；不处理最后一题之后的片尾。所有暂停标记保留。`timing.json` 分别保存模型加载、每视频墙钟时间、视觉预处理与编码、视频语言模型前向、ProtoKV 原型更新及问答时间；CUDA 在阶段边界同步。问答时间及该文件的 TTFT 包含持久状态克隆检查，分项总和之外还包括读取、初始提示和日志等开销。

## 预算与验证范围

默认 BF16、`budget_scope=visual`。ProtoKV 的 1000 视觉位置分为近端 256 与 93×8 伪位置，初始文本另计；prompt 预算则先扣除实际文本长度。AdaKV/PyramidKV 保留 SparseMM 原始整数分配规则及其容量取整偏差。

原型库中心、PQ、空间统计和近端原始 key 均单独统计字节。相同位置额度不等于相同显存。小验证会克隆持久状态检查逐位恢复，峰值显存与时间属于诊断数据。

本插件沿用本地论文实现的 warm-up=64、PQ smoothing=1 等显式选择；结果应标为 `ProtoKV paper-based reproduction via SparseMM, streaming adaptation`。

CPU 检查覆盖原生模型一致性、首答案读取原型、GQA/偏置/掩码、异常恢复、算法参考状态、跨视频时间轴和暂停保护。真实结果与独立审计保存在 `results/small_validation/`：

```bash
PYTHONPATH=/home/yujun/program \
/home/yujun/.conda/envs/hermes-qwen/bin/python -m sparsemm_protokv.audit OUTPUT_DIR
```

关键代码：`monkeypatch.py` / `qwen2_5.py` 注册与 attention；`cache.py` 在线状态和查询事务；`streaming.py` 原生前向调度；`models/` / `tasks/` 为原版 lmms-eval 插件。
