# HorizonKV-EGate

统一入口参数：`--method horizonkv-egate`。原实验标识保持 `pae_online_e_gate_k500`。实现参考 `20261104-egate-dscache-input-work-v2/egate/` 的 KV-only recent-transfer 版本，移除了运行时对旧 SimpleStream 工程的引用；本目录包含方法所需策略、协议和预算表，公共 ragged cache 来自本项目 `methods/_shared/`。

四帧为一个不重叠记忆批次。`full` 近期分支包含当前批次已到达的帧；`one` 分支用相同文本前缀单独编码最新一帧，不读取 full 或历史缓存。112 个 KV 头的冻结 CSV 指定各自使用 one/four 分支及近期/历史配额，56 个头属于每种分支，所有头 recent+history=500。近期使用最近32视觉 query 的 SnapKV 风格筛选并保护最新32 token；历史筛选不强制保护近期 token。

下一批到达时，只把上一批按头选出的近期 raw-K/V 与已有该头历史 raw-K/V 合并，再筛到历史配额。旧帧特征不保存、不重新过语言模型，驱逐的 KV 不能通过旧视觉特征重建。每个头独立保存历史 K/V，不用各头位置的并集构建物理缓存。位置并集仅用于共享坐标映射。

回答前一次性选好近期 token，为已有历史保留预算，两遍共享该近期集合和起始位置。第一遍用近期 KV，加证据不足选项（4选项题用E、5选项题用F）；第二遍用近期+历史 KV 和原始选项，两遍均实际执行。第一遍选证据不足时采用第二遍，否则采用第一遍。保留原版“第一个 logit 在可用答案字母中约束选择”的解码规则，不改成其他 baseline 的多 token 自回归回答。为 OVO 的题型扩展到1..8个原始选项，附加下一个字母；4/5选项的提示和门控规则不变。

## 输入与适配边界

- Qwen2.5-VL：1 fps，所有帧目标面积448²。普通单元最多2帧，原生 temporal merge 的 token 以较新帧标识；one 分支在流式输入时单独编码最新单帧，也使用448²。四帧记忆边界必要时切分输入单元。紧凑 M-RoPE 是方法内部位置策略，不声称与 DSCache 的时间组织逐数值相同。
- LLaVA-OV：1 fps，384×384，每帧池化为196 token。one 分支在流式输入时维护，不在答题时重编码。
- 使用动态帧/行/列标识，移除旧版固定1024 token/帧假设。较早 raw-K/V 根据活跃帧重新施加 RoPE，Value 保持不变。
- CSV 原样复制，SHA256为 `d7f644848b3521a18fdb73441e9a42052933141513b4753e273796539909a0d3`。未按当前输入重新校准；LLaVA 还是 Qwen 预算的跨 backbone 转用。当前首先验证实现可运行，不据 smoke 宣称准确率或预算最优。
- 每头500限制的是回答时筛选后的视觉 KV；文本前缀不计入500，问句后缀另行前向。构建近期分支时允许临时缓存当前批次；近期完整分支、独立最新分支和原始Key备份仍会有额外显存开销，不能把进程总显存等同于112×500个KV。
- CPU 不累积整段视频特征。只保留当前批次 KV、受限历史 KV；解码和预处理仍使用临时 CPU 数据。

## 文件

- `adapter.py`：普通视觉输入、流式近期分支与已有 KV 答题。
- `policy.py`：按头近期筛选、四帧批次转移、两遍共享选择和 ragged 回答缓存。
- `decoder.py`、`attention.py`：两种 backbone 的语言模型前向和 FlashAttention GQA。
- `positions.py`：不同网格大小的永久 token 标识和活跃位置映射。
- `protocol.py`、`assets/per_head_allocation.csv`：冻结配额、提示、门控规则。
- `run.py`、`worker.py`、`data.py`、`summarize.py`：分片、逐题 CSV 续跑及严格汇总。

通过项目根 `run.py` 调用；不要直接调用复制旧目录中的 `llava/egate/`。每题只写小型结果行，`worker_status.json` 覆盖保存最新运行诊断，未写/读取逐 token 的 cache.json。中途续跑会重放当前未完成视频的先前视觉输入以恢复相同状态，已完成题不再回答。
