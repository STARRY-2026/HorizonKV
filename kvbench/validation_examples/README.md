# KVBench 小规模验证结果

`ovobench_3v3q_20261015/` 保存 2026-10-15 的五种方法、同一组 3 个视频和 3 道 MCQA 题的精简结果。`report.md` 与 `summary.json` 给出汇总；每个方法的 `results.csv` 和 `audit_report.json` 分别记录逐题预测与审计结果；`migration_regression.json` 记录与整合前运行的对照。文件保持原始字节，内部绝对路径指向当时的本机运行位置，仅供溯源。

这组结果用于验证统一接口、真实压缩和审计链路。样本量很小，不能据此比较方法优劣，也不能当作完整 OVO 基准结果。完整运行日志、缓存轨迹和源码快照留在本机未上传的 `results/` 中。
