# HERMES 上传 GitHub 审计与操作方案

更新日期：2026-10-10。原项目为 `/home/yujun/program/hermes`；发布源码为本整理目录下的 `snapshot/`。本文中的文档链接和清单路径以发布源码根目录为准。

已将整理、验证的源码以独立新历史发布至 [STARRY-2026/HorizonKV](https://github.com/STARRY-2026/HorizonKV)。原项目、原 Git 历史和 Gitee 远程均未修改。本文保留导出范围、验证依据和重新发布操作说明。

## 使用独立新历史的原因

原项目基础提交为 `6020490`，远程是 `https://gitee.com/charlottenaixu/hermes.git`。工作区包含未提交和未跟踪的新方法源码，旧历史又混有大量实验结果和日志，因此不能只导出已有提交或直接推送原仓库。

历史 `eval/eval_open_ended.py:43` 曾出现疑似真实 API 密钥格式，涉及引入提交 `a452f4f` 和替换提交 `452d540`；当前源码已是空占位值。本次未输出或验证该值。修改当前文件或增加 `.gitignore` 无法清除旧提交中的内容，所以发布包不携带旧 `.git`。若该密钥仍由你管理，应撤销或轮换。

## 发布内容与运行入口

保留 `inference/`、`video_qa/`、`head_analysis/`、`eval/`、`scripts/`、`experiments/`、`tests/`、`docs/`，以及标注、运行配置、依赖说明、MIT License 和上游来源。`head_analysis/` 是当前推理入口的实际依赖，不能省略。

`results/` 中用于运行的评分、预算和策略表也保留原相对路径，包括 DP 各预算的 score/recency 表、v8、current-ourtop、HorizonKV 和其他保留方法的配置。K800 的 runtime 表由已有冻结分配确定性导出，来源和核对结果随表保存。另在 [`experiment_results/`](experiment_results/README.md) 附上 14 项完整主要结果：DP K100/250/500/1000 及匹配 uniform、native v8、timestamp v8、历史 HorizonKV v0.6 和相应对照。包含原始逐题预测、评测报告、配置来源、哈希及分片合并说明；保留负结果。未完成的 current-ourtop 和特定小 pilot 不混作完整主结果。模型权重、原始视频、原始日志、大型 trace、缓存、其余大量实验产物和旧 Git 历史不上传。

最终文件数量、体积和逐文件 SHA-256 以 [EXPORT_SUMMARY.json](EXPORT_SUMMARY.json) 和 [SNAPSHOT_MANIFEST.csv](SNAPSHOT_MANIFEST.csv) 为准。清单区分原项目源码、发布新增文件和发布修复；修复后的入口不应被误报为与原项目逐字节一致。小型运行输入由 `.gitignore` 明确放行。

发布修复增加可移植模型/输出路径、子进程失败传播、完整结果检查和输出目录保护，**没有修改 KV 策略算法**。统一入口为 `tools/run_experiment.py`：

- `hermes`：原始 dense HERMES；`uniform`：DP 的匹配 ragged 对照。
- `dp`：冻结的精确 head 容量及 recency alpha/k，支持 K100/250/500/800/1000。
- `v8` / `memheadkv`：native 分配；`timestamp-v8` 和 `current-ourtop`：分别命名的 fixed448 timestamp 变体。
- `horizonkv`：独立 Recent + Memory 运行路径，冻结预算 K6000。

安装步骤见 [RUNTIME_SETUP.md](RUNTIME_SETUP.md)，运行与全量实验命令见 [EXPERIMENTS.md](EXPERIMENTS.md)，实测记录见 [RUNTIME_VALIDATION.md](RUNTIME_VALIDATION.md)。模型和 StreamingBench 视频须单独准备，通过 `--model-path` 和 `--video-root` 指定，不依赖原服务器路径。完整标注为 498 视频／2495 问，half 为 250／1255；比较时须统一数据划分、分辨率和预算。

## 已完成的验证及边界

- 最终 CPU 策略、协议、环境工具和失败处理测试：**85 passed**。
- 已在现有 Qwen2.5-VL-7B 环境、A100 40 GB 上运行真实视频 GPU 冒烟，使用原始问题的 **84 秒前缀、1 道问答**，完成视频编码、3 次实际 KV 压缩、答案生成和评测。
- 原始 HERMES K1000 的直接入口运行成功；统一入口的 uniform K1000、DP K1000/K800/K100、native v8 K1000、timestamp-v8 K1000、current-ourtop K1000 和 HorizonKV K6000 均返回 **0**，各得到预期的 1 条预测。DP K250/K500 已做冻结资产与 CPU 检查，未在本次单独运行 GPU 冒烟。
- 外部原始运行日志、预测和 manifest 保存在 `/home/yujun/program/hermes_github_review/runtime_validation_20261010`，不随源码打包；发布包提供验证摘要。视频路径重定位保留问题、ID 和顺序。

这些结果验证了指定环境中的实际执行，**没有验证全量基准准确率，也没有完成全新环境安装**。安装说明和依赖锁定以现有 Qwen 环境为依据；LLaVA、其他模型尺寸、独立 DSCache、其他数据集和全部历史分析程序尚未获得同等认证。旧 Qwen/LLaVA requirements 和机器固定路径脚本仅保留为历史资料。

统一入口使用冻结输入，不会在评测问题上重新校准。重新生成历史 profile 可能需要未打包的原始观察和大型 trace。ProtoKV/InfiniPot-V 仅在显式启用时需要外部项目；统一入口关闭这两个适配器。

## 重新导出本地包

已验证的 `snapshot/` 已存在。重新导出时使用新的输出目录，并保留本整理目录的 `release_overlay/`；它包含运行修复、新协议、工具和文档，不能只重新复制原项目而丢掉这些更改。

```bash
cd /home/yujun/program/hermes_github_review
python3 export_snapshot.py \
  --source /home/yujun/program/hermes \
  --output /home/yujun/program/hermes_github_review/snapshot-new
```

导出工具拒绝覆盖已有输出，也拒绝写进原项目。它不会执行 `git clean`、`git reset`、历史重写或网络推送。最终使用的目录和压缩包应携带与其内容匹配的导出清单。

## 推送审查后的源码到 GitHub

先在 GitHub 创建空仓库，不额外初始化 README、License 或 `.gitignore`。只在新导出目录执行以下命令，并替换仓库 URL：

```bash
cd /home/yujun/program/hermes_github_review/snapshot
git init -b main
git add .
git diff --cached --stat
git status --short
git commit -m "Publish runnable HERMES source and frozen research protocols"
git remote add origin https://github.com/YOUR_ACCOUNT/YOUR_REPO.git
git push -u origin main
```

这里的 `git add .` 仅用于已过滤、没有旧历史的发布目录。保留原作者 MIT License 和引用信息，使用常规 GitHub SSH/HTTPS 登录；不要把 token 写入源码或远程 URL。若要沿用旧 Git 历史，须另行处理历史凭据和实验产物。
