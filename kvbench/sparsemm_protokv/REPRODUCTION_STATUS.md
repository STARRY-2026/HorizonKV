# 接入状态

2026-10-14（北京时间）：ProtoKV 已作为 SparseMM 原生 lmms-eval 外部模型插件接入，适配当前 Qwen2.5-VL-7B / Transformers 4.57.1 和本地 StreamingBench。

- 原始 SparseMM 固定提交 `33f2f0ab96d1d99516bcd2092caccab60516ff2c`，参考源码未修改。
- `METHOD=protokv/adakv/pyramidkv` 使用同一原生模型注册、视频处理、绝对时间轴、问题模板与评测任务。
- ProtoKV 全层全头银行复用现有 `FullProtoBank`；首答案读取近端和原型状态，查询包含 mass bias 与无效槽屏蔽。
- 32 项 CPU 检查通过。三方法各完成 2 视频 / 4 题 / 24 次压缩：ProtoKV 4/4，AdaKV 3/4，PyramidKV 3/4。
- 12 次问答均完成原生 28 层前向、数值检查和持久状态逐位恢复；ProtoKV 两视频均有 112/112 组 PQ 就绪。
- 三组原生 lmms 分数、逐题 CSV、独立缓存审计和源码配置一致性检查全部通过。结果属于小规模集成验证。
- 全量实验与旧 HERMES ProtoKV 实验继续暂停。控制器和原生插件都限制显式小验证，结果目录不覆盖、不自动续跑、不后台排队。

入口和依赖见 [README](README.md)，结果见 [验证报告](results/small_validation/report.md)，指纹见 `SOURCE_MANIFEST.json`。
