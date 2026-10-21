# GitHub 上传范围

目标仓库是 `https://github.com/STARRY-2026/HorizonKV`，本项目作为仓库根目录下的 `kvbench/` 独立文件夹上传。本机源码在 `/home/yujun/program/kvbench`。旁边的旧 `sparsemm_*`、`protokv` 和 `hermes` 工程无需再次复制；HorizonKV 根目录已有 HERMES 源码。

本机旧项目名 `sparsemm_suite` 是指向 `kvbench` 的历史路径兼容链接，位于仓库根目录之外，不属于上传范围。

## 纳入源码仓库

| 路径 | 内容 |
|---|---|
| `kvbench/`、`run.py`、`pyproject.toml` | 统一 CLI、Python API、方法注册与项目元数据 |
| `sparsemm_baselines/` | AdaKV/PyramidKV；其中 `upstream` 作为 Git 子模块 |
| `sparsemm_protokv/` | Qwen2.5-VL 原生插件、缓存、任务和控制器，排除 `runtime_deps` 和结果 |
| `sparsemm_lukv/`、`sparsemm_native/`、`sparsemm_uniform/`、`protokv/` | 方法实现及包内校准 profile |
| `assets/` | SparseMM 分数、raw scores、冻结 manifest/protocol，审计需要保留 |
| `configs/` | 当前小规模 MCQA 标注；其中的视频路径是本机路径 |
| `requirements/`、`tests/`、`tools/` | 依赖记录、CPU 测试、迁移回归核对工具 |
| `README.md`、`VALIDATION.md`、`MIGRATION_MANIFEST.json` | 使用说明、验证记录、迁移来源 |
| `validation_examples/` | 五种方法在 3 视频/3 题上的精简真实推理结果与审计摘要 |
| `sparsemm_native/RECENT4_DP.md`、`sparsemm_native/recent4.py` 及其 OVO-Bench 入口和依赖 | 已完成的近期四帧 DP 双缓存实现与运行说明 |
| `assets/dp_recent4/` | 当次实验使用的两张冻结预算 CSV 原始副本 |
| `.gitignore`、本文件和 `GITHUB_UPLOAD_FILES.txt` | 忽略规则和逐文件上传清单 |

`GITHUB_UPLOAD_FILES.txt` 列出相对 `kvbench/` 的上传文件。子模块只记录路径和固定提交，不将整个上游仓库重新存进主仓库。子模块声明放在 HorizonKV 仓库**根目录**的 `.gitmodules`。

## 已忽略的本机文件

- 根目录和方法包内的 `results/`，包括预测、缓存轨迹、源码快照和运行日志。
- `hermes` 资源软链接；链接目标中的模型、视频和数据集。
- `sparsemm_protokv/runtime_deps/`；其他机器通过依赖文件重新安装。
- `__pycache__/`、`*.pyc`、`*.log`、环境和构建目录。

完整验证产物在本机 `results/ovobench_regression_20261015/`。随仓库上传的是 `validation_examples/` 中的汇总报告、逐题预测和审计报告；这些只是 3 视频/3 题的接口验证，不是完整基准结果。

## 上游子模块

已经添加 `.gitmodules`：

- 仓库：`https://github.com/CR400AF-A/SparseMM.git`
- 路径：`kvbench/sparsemm_baselines/upstream`
- 当前固定提交：`33f2f0ab96d1d99516bcd2092caccab60516ff2c`

目前上游工作树已经位于该提交。Git 会记录 mode `160000` 的 gitlink；根目录 `.gitmodules` 记录正确的嵌套路径。不要删除本机上游的 `.git`：现有控制器用它验证固定提交和未修改状态，也不需要将 `.git` 元数据上传。子模块说明参考 [Git 官方文档](https://git-scm.com/book/en/v2/Git-Tools-Submodules)。

HorizonKV 仓库已在 GitHub。上传时在其根目录执行：

```bash
git add .gitmodules kvbench
git ls-files --stage kvbench/sparsemm_baselines/upstream
git diff --cached --stat
```

`git ls-files --stage` 对上游应显示 `160000 33f2f0ab96d1d99516bcd2092caccab60516ff2c`。仓库主目录只保存子模块引用，其他人使用下面的命令拉取固定上游版本。

之后应使用 Git 检出子模块：

```bash
git clone --recurse-submodules https://github.com/STARRY-2026/HorizonKV.git
# 对已有的普通 clone：
git submodule update --init --recursive
```

## 当前运行依赖

当前上传内容是源码框架；推理还需要本机资源：

- Qwen2.5-VL 模型与原视频。
- HorizonKV 根目录的 `eval/eval_multiple_choice.py` 评分脚本。
- LU-KV 的冻结校准 manifest、观察数据、标注和校准视频。
- SparseMM 的历史校准协议、SynthDOG 标注、数据清单和校准源码证据。

标注和冻结证据包含 `/home/yujun/program/...` 绝对路径。`KV_BENCH_*` 环境变量可以指定模型、标注和主要资产，但不会自动迁移冻结 JSON 内部所有路径。其他机器需配置这些资源及证据绑定后才能推理；仅 clone 源码不能直接复现本机实验。详情见 `README.md` 的资源依赖说明。

第三方来源通过固定上游子模块保留，其原有许可证位于上游各组件目录。当前项目未为所有来源代码新增统一许可证。
