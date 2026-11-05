# 统一评测架构与边界

当前目录分为公共控制层与冻结后端。六种 baseline 保持原有实现；新接入的 HorizonKV-EGate 采用独立方法目录与后端，复用统一视觉输入配置，同时保留自己的分支、位置策略与两遍回答协议。

```text
run.py / summarize.py
        │
        ├── core/registry.py   模型、方法、输入描述和能力范围
        └── core/dispatch.py   参数校验、后端命令、完成后的来源记录
                  │
                  ├── qwen/           原有 Qwen 适配
                  ├── qwen_full2495/  含等价选项计分的冻结 Qwen 适配
                  ├── llava/          原有 LLaVA 适配
                  └── egate/          薄入口 → methods/horizonkv_egate/
                              │
                              └── 本地 methods/ 与 backbones/ / 外部模型、数据、Python 环境
```

## 已完成的控制层整理

- 统一声明输入配置和当前支持范围；尚不支持的等价选项组合会明确报错。
- `--plan-only` 只打印规范化参数、配置指纹和实际后端命令，不读取视频、不加载模型、不启动 GPU 进程。
- `--dry-run` 仍调用原有后端进行数据与配置预检；预检成功不代表 GPU 推理已经验证。
- 正式任务成功返回且存在后端 manifest 后，原子写入 `launcher_plan.json`，记录明确后端、完整命令和控制层代码指纹。
- 后续续跑核对已保存的控制层配置；后端仍负责原有数据指纹、结果和分片一致性检查。
- 汇总优先读取明确的后端来源，不再凭“是否恰好有 2495 题”推测计分规则。
- 旧 Qwen StreamingBench 结果、尚未结束的任务没有来源记录时，用 `summarize.py OUTPUT --backend qwen` 或 `--backend qwen_full2495` 显式选择原运行协议。

来源记录在后端成功结束后才写入，因为旧后端会拒绝预先放入额外文件的结果目录。中途退出的任务依靠原后端 manifest/CSV 续跑；此次未替换续跑机制。不同进程不得同时向同一输出目录写结果。

## 后续值得继续的架构工作

1. 将分散的绝对路径、临时 Python 环境和冻结依赖集中到可配置的路径表，补齐逐方法依赖检查。当前目录仍不是可单独拷走的完整发布包。
2. 将 heldout/full、重复选项和校准集重叠规则抽成数据协议。先对齐逐题正确标记与汇总，再消除 qwen_full2495 的重复代码。
3. 提取共用的任务启动、退出状态、受控停止和仅重试失败分片的调度器。保留现有分片与评分协议后，再研究视频长度感知的负载均衡。
4. 逐步定义 BackboneAdapter 与 CachePolicy 接口。近期高分辨率帧、时间位置编码、GQA 头数、变长 KV 和各方法压缩时机均需契约测试，不能通过统一函数名假定等价。
5. 引入统一监控：每题耗时、系统内存峰值、加速器内存峰值和进度。继续避免逐 token 的大 cache JSON。

当前重构的收益是配置可靠性与可维护性，不是模型推理提速。后端仍含旧分支和重复调度逻辑。EGate 现已接入独立后端，四个组合的 GPU smoke 记录见 EGATE_VALIDATION.md；小样本验证不等于完整集准确率或长视频稳定性结论。

## 方法源码本地化更新

六种 baseline 已整理到 methods/ 下，普通统一入口改用本地源码、预算资产和 LLaVA 项目源码；算法共享部分保存在 methods/_shared/。SOURCE_MANIFEST.json 记录来源与当前指纹。qwen_full2495 的计分协议副本目前仍保留。模型、数据和 Python/PPU 环境仍为外部依赖。历史独立校准/审计工具可能引用原校准输入；不属于根入口评测依赖。

## HorizonKV-EGate 更新

方法代码及冻结预算位于 methods/horizonkv_egate/，统一参数名为 horizonkv-egate，实验标识仍为 pae_online_e_gate_k500。方法内部保留四帧批次、独立 one 分支和紧凑位置策略；所有视觉输入使用普通帧尺寸，答题直接使用流式维护的 KV。逐题 CSV 支持续跑，worker_status.json 仅覆盖最后一次小型诊断，不保存逐 token KV trace。分片结果与答案来源均由方法专用汇总校验。
