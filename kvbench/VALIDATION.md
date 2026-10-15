# 整合验证

统一项目入口：`python -m kvbench`，Python API：`from kvbench import RunConfig, run, audit, summarize`。

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
