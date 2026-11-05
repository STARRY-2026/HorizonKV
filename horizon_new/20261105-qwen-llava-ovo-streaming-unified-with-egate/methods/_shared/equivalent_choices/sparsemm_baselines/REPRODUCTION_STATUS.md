# 复现状态

更新时间：2026-10-14（北京时间）。

新增原生框架插件：见 [sparsemm_protokv 接入状态](../sparsemm_protokv/REPRODUCTION_STATUS.md)。ProtoKV / AdaKV / PyramidKV 已在同一原生 lmms-eval 入口、BF16 和 1000 视觉预算下完成各 2 视频 / 4 题小验证，分别答对 4/4、3/4、3/4；全量仍暂停。下文保留此前独立视频移植入口的验证记录。

- SparseMM 原始源码固定在 `33f2f0ab96d1d99516bcd2092caccab60516ff2c`。
- AdaKV / PyramidKV 共用 SparseMM 配置、评分和整数分配规则；与原始压缩器选出的 K/V 一致性测试通过。
- Qwen2.5-VL 独立模型接口、真实变长缓存、视频时间轴、StreamingBench 数据与评测入口已完成。
- 原生 SparseMM lmms-eval 命令入口已准备；正式执行所需的旧版环境和原模型权重尚未准备。
- 本地 250 视频 / 1255 题的两种方法 dry-run 均通过。
- 49 项基础 CPU 测试、11 项新增审计工具 CPU 测试及 6 组小张量 CUDA 注意力检查通过；这些测试不加载真实模型权重。
- AdaKV / PyramidKV 各完成一个真实 Qwen2.5-VL-7B 视频问题，均答对；3 次视频前向产生实际压缩，缓存长度与问答恢复审计通过。
- 扩展真实 7B 验证完成 6 组、58 次问答、624 次视频压缩，全部通过运行检查。主测试每种方法使用相同 9 视频 / 23 题、覆盖 10 类任务，最长前缀 502 秒；AdaKV 答对 20/23，PyramidKV 答对 21/23。
- 64-token prompt 边界测试两种方法均为 3/3；256-token visual 测试均为 2/3。包含奇数帧补齐、连续问答和重复时间点回答。
- 主测试与 visual 测试共通过 52 次问答缓存逐位恢复、1120 次 decoder 最终 norm 输出和 518 次问答 logits 有限值检查。64-token 两组只有运行器检查与轨迹审计，没有额外执行这些检查。
- 扩展测试峰值张量显存约 16.05 GiB；额外检查克隆缓存，不能据此给出推理效率结论。完整评测与 ProtoKV 仍暂停，没有启动或后台排队。

验证统计及源码指纹见 `results/validation/summary.json` 和 `SOURCE_MANIFEST.json`。扩展结果见 [完整报告](results/extended_validation/combined_report.md)；每组保留配置、CSV、压缩轨迹和独立审计。当前没有完整数据集成绩或推理效率结论。
