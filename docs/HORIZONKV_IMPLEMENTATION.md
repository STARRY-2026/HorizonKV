# HorizonKV 实际技术实现与迭代记录

> 文档版本：v0.6  
> 方法状态：固定双区首版已实现并通过单问题 GPU smoke，正式实验待运行  
> 最后更新：2026-08-26  
> 方法动机与论文叙事见：[`../yujun.md`](../yujun.md)

## 1. 文档用途

本文档是 HorizonKV 的工程事实源，用于记录：

- 当前仓库已经具备的能力；
- 方法从公式到代码的数据契约；
- 每一版实现的范围、决策、实验和已知问题；
- 后续讨论形成的修改，不覆盖历史结论，而是通过版本记录追踪。

`yujun.md` 负责回答“为什么做、论文如何叙述”；本文档负责回答“具体怎么做、做到哪一步、如何证明实现正确”。如果两份文档对实现状态的描述冲突，以代码、冻结产物和可复现实验为准。

## 2. 状态约定

| 标记 | 含义 |
|---|---|
| `[DONE]` | 代码已存在，并有至少一次可追溯运行或测试 |
| `[PARTIAL]` | 已有可复用实现，但不等价于完整 HorizonKV |
| `[TODO]` | 尚未实现 |
| `[VERIFY]` | 已实现或已有数据，但仍缺严格验证 |
| `[PROPOSED]` | 已形成可执行候选方案，但尚未经过讨论确认 |
| `[DECISION]` | 已确认的设计决策 |
| `[QUESTION]` | 需要后续讨论的问题 |

版本号分为两类：

- **文档版本**：本文档本身的修改，如 v0.1、v0.2；
- **方法版本**：可运行的 HorizonKV 实现，如 M0、M1、M2。

只修改文档不代表方法版本已经推进。

## 3. 实现目标与边界

### 3.1 目标

在 Qwen2.5-VL-7B、未知问题流视频场景下，对 28 层 × 4 个物理 KV Head 建立冻结的二维时间画像：

```text
P_g = (H_g, N_g)

H_g: 远历史信息的持续复用需求
N_g: 新到信息在后续 chunk 中的延续需求
```

在固定全局视觉 KV 总预算下，画像同时决定：

1. 每个物理 KV Head 的总容量 `B_g`；
2. Head 内历史区容量 `B_hist_g`；
3. Head 内新信息区容量 `B_new_g`；
4. 后续版本中的 aging、晋升和压缩行为。

### 3.2 第一阶段不做的事情

- 不根据测试问题动态修改缓存分配；
- 不把旧 `memory score` 直接重命名为 `H_g`；
- 不假设低 `H_g` 必然对应高 `N_g`；
- 不在 M0 同时更换画像、token selector 和底层 kernel；
- 不把现有 RoleRecent 的 75.94 结果归因于完整 HorizonKV；
- 不在严格计时完成前宣称 TTFT 加速。

### 3.3 v0.6 当前实现方案：固定双区 HorizonKV

本节是 v0.6 代码实现的事实规范。它替代早期版本中“固定最新一帧”、
“运行时动态调整 Recent/Memory 容量”和“NEW/HIST 动态借用预算”的设想；
后文旧方案只保留为设计演进记录。首版运行时只使用冻结的 pseudo-query
画像，不读取当前 StreamingBench 真实问题。

每个物理 KV Head 维护两个固定容量的视觉缓存区：

```text
Text Prefix
    永久保留，不计入视觉预算

RecentCache
    固定容量 2691 visual tokens
    token 级 FIFO

MemoryCache
    固定 per-head 容量
    接收从 RecentCache 溢出的旧 token
    按 Head 类型执行个性化驱逐
```

数据生命周期为：

```text
新视觉 token -> RecentCache 尾部
RecentCache 超过 2691 -> 最旧 token 迁入 MemoryCache 候选
MemoryCache 超过该 Head 的固定容量 -> 个性化驱逐
```

`RecentCache` 表示“最新 2691 个视觉 token”，不表示固定一帧或固定两秒。
Qwen 仍按原始自适应分辨率处理视频；当每帧约 500 token 时，Recent 可以
同时覆盖约 5 帧，当一帧为 2691 token 时则约覆盖 1 帧。2691 来自当前
Qwen 自适应处理器在 StreamingBench half 1255 个问题上的单帧 token 上限
统计，不来自 fixed896 强制画布。若更换 processor 后单帧上限变化，必须显式
修改策略配置并重新记录实验，不能在运行时改变 Memory 容量。

两个区域的容量不互借。Recent 未填满时，空余逻辑容量不转给 Memory；
Memory 的容量只由离线 Head 类型决定。物理 ragged cache 不预分配空槽，
因此 Recent 的未占用容量不会产生对应的 K/V 显存占用。

#### 3.3.1 离线 pseudo-only 五桶画像

每个 Head 使用固定 pseudo probes 得到：

```text
P_g = [p0, p1, p2, p3, p4+]
MemoryMass = p1 + p2 + p3 + p4
Continuity = MemoryMass * normalized_entropy(normalize([p1,p2,p3,p4]))
TrendSlope = linear_slope(normalize([p1,p2,p3,p4]), x=[1,2,3,4])
FarBias = p4 / MemoryMass
```

冻结分类规则为：

```python
if FarBias >= 0.32 or TrendSlope >= 0:
    HeadType = "rising_far"
elif Continuity >= 0.55:
    HeadType = "sustained_gentle"
elif Continuity < 0.40 or TrendSlope <= -0.115 or MemoryMass < 0.43:
    HeadType = "decay_aggressive"
else:
    HeadType = "normal_sustained"
```

当前 10-video fixed896 pseudo pilot 共得到：

| Head 类型 | 数量 |
|---|---:|
| `decay_aggressive` | 21 |
| `normal_sustained` | 68 |
| `sustained_gentle` | 13 |
| `rising_far` | 10 |

fixed896 只用于让小规模离线 Attention 画像具有可比的 token 网格；运行时
不启用 fixed canvas，仍使用 Qwen 自适应视觉分辨率。当前策略表是 pilot
产物，不是正式 calibration/test 隔离后的最终画像。

#### 3.3.2 固定 Head 容量

平均视觉预算为每 Head 6000，112 个 Head 的整数总预算严格为 672000：

| Head 类型 | Recent | Memory | Total |
|---|---:|---:|---:|
| `decay_aggressive` | 2691 | 2629/2630 | 5320/5321 |
| `normal_sustained` | 2691 | 3287 | 5978 |
| `sustained_gentle` | 2691 | 3780 | 6471 |
| `rising_far` | 2691 | 4273 | 6964 |

实际逐 Head 数值以冻结策略表为准：

```text
results/myobservations/horizonkv_pilot_10v_fixed896/
  pseudo_trend_memory_budget_6000_qwen_adaptive_recent2691.csv
```

CSV 中 `RecentBudget`、`MemoryBudget`、`TotalBudget` 都是固定容量。文本前缀
不计入这些视觉预算。任何一次压缩结束后均须满足：

```text
actual_recent <= RecentBudget
actual_memory <= MemoryBudget
actual_visual <= TotalBudget
sum(TotalBudget over 112 heads) == 672000
```

#### 3.3.3 Memory 时间状态与 Attention EMA

每个视觉 token 维护独立于 M-RoPE position 的 `source_chunk_id`：

```text
chunk_age = current_chunk_id - source_chunk_id
warm: chunk_age <= 2
cold: chunk_age >= 3
```

当前 chunk 中因 Recent FIFO 溢出而进入 Memory 的 token，其
`chunk_age=0`，属于 warm。`source_chunk_id` 和 Attention EMA 必须在 append、
prune、truncate 时与 K/V/position 同步移动，不能从压缩后 position 反推来源。

在线只使用与 pilot 相同的三个固定 pseudo probes：

```text
neutral: Summarize the important visual information observed in the video so far.
history: What information introduced much earlier in the video is still important
         for understanding the video now?
latest:  What new visual information appeared or changed in the most recent segment?
```

对三个 probe 分别取最后 8 个 query token 做 pooling，并完成 GQA query-head 到物理 KV Head
聚合，对每个 probe 的 token 分数归一化后等权平均，并更新：

```text
attention_ema = 0.9 * old_ema + 0.1 * current_attention
```

新 token 的 EMA 使用本轮 Attention 初始化。

#### 3.3.4 四类 Memory 驱逐

统一保留分数为：

```text
base_score = recency_weight * recency_norm
           + attention_weight * attention_ema_norm
retention_score = base_score / (1 + chunk_age) ^ age_exponent
```

| Head 类型 | recency/attention | cold 保底 | age exponent | 特殊保护 |
|---|---:|---:|---:|---|
| `decay_aggressive` | 70/30 | 25% | 1.00 | 无 |
| `normal_sustained` | 60/40 | 35% | 0.50 | 无 |
| `sustained_gentle` | 50/50 | 40% | 0.25 | 无 |
| `rising_far` | 50/50 | 50% | 0.25 | `age>=4` 时 exponent 减半 |

驱逐分两步：

1. 从 cold token 中保留 `cold_anchor * MemoryBudget`，在已有 source chunk
   之间均匀分配覆盖配额，每个 chunk 内优先选择 retention score 高的 token；
2. 剩余容量从所有尚未入选的 Memory 候选中按 retention score 全局 top-k。

cold 候选不足时，其空余配额回填给 warm；所有索引最终必须有序、唯一。

#### 3.3.5 首版压缩边界与兼容性

首版复用现有 ragged prefill 流程：先追加完整 video chunk，再运行三个 pseudo
probe，随后压缩。因此压缩后的缓存严格满足预算，但追加 chunk 时仍可能出现
与原 HERMES 相同的临时显存峰值。该峰值必须记录，不能把 post-compression
长度表述为全过程峰值；逐帧或 micro-chunk 严格压缩留作后续版本。

HorizonKV 使用独立、默认关闭的策略入口。未提供策略表时，原 HERMES、
RoleKV、per-head budget 和 ragged selector 行为保持不变。首版禁止 HorizonKV
与 RoleKV、旧 memory/current split、defer-last、multi-scale、layered-resolution、
transient-highres 和 feature-buffer 组合，避免多种机制共同改变预算语义。

## 4. 当前代码基线

### 4.1 已有能力

| 状态 | 能力 | 现有位置 | HorizonKV 中的用途 |
|---|---|---|---|
| `[DONE]` | per-KV-head 总预算 | `head_analysis/hermes_kv_head_budget.py` | 承载 `B_g` |
| `[DONE]` | memory/current 二分预算 | `head_analysis/hermes_kv_head_budget.py` | M0 承载 `B_hist_g/B_new_g` |
| `[DONE]` | 物理 ragged KV 与 ragged prefill | `head_analysis/hermes_kv_head_ragged.py`、`head_analysis/hermes_kv_head_ragged_prefill.py` | 不同 Head 保存不同 token 集合和长度 |
| `[DONE]` | 最新 chunk 前后长度快照 | `hermes_kv_head_budget.py`、`hermes_kv_head_ragged_prefill.py` | 区分 previous/current region |
| `[PARTIAL]` | memory/current/mixed 角色化近期保留 | `install_kv_head_role_recent_retention()` | `N_g` 与时间分区的早期代理 |
| `[PARTIAL]` | defer-last-frame | `video_qa/run_infer.py`、`video_qa/base.py` | 新信息保护的早期工程探索 |
| `[DONE]` | Qwen 流式视频编码与问答入口 | `inference/qwenvl_hermes.py` | 在线生命周期入口 |
| `[DONE]` | StreamingBench 运行与结果目录组织 | `video_qa/run_infer.py` | 主实验入口 |

### 4.2 尚缺的关键闭环

| 状态 | 缺口 | 完成标准 |
|---|---|---|
| `[TODO]` | 按视频隔离的 calibration/test manifest | 同一视频不能同时进入画像学习和正式测试 |
| `[TODO]` | 直接计算 `H_g` | 输出 112 个 Head 的原始值、归一化值和置信度 |
| `[TODO]` | 跨多个未来间隔计算 `N_g` | 能追踪 chunk `t` 在 `t+1...t+D` 的后续读出 |
| `[TODO]` | 画像稳定化与收缩 | bootstrap 后输出 `reliability` 和 shrunk profile |
| `[TODO]` | `H+N` 联合总预算分配 | 精确满足总预算、上下界和整数约束 |
| `[TODO]` | 显式 NewBuffer/HistCache aging | token 年龄不再只由“上一次/当前 chunk”隐式表达 |
| `[TODO]` | 多预算及反向/随机消融 | 至少在 2000、4000、6000 三个平均预算验证 |
| `[TODO]` | 严格效率测量 | ingestion、compression、prefill、decode 分段计时 |

### 4.3 当前结果锚点

以下结果只作为实现回归基线，不代表完整 HorizonKV 的结果：

| 配置 | StreamingBench half Avg | 说明 |
|---|---:|---|
| uniform half | 73.94 | 统一缓存基线 |
| `ours_top100` | 74.34 | 单一历史代理 |
| `v8_blend3_scores` | 74.90 | 当前较均衡的 Head-aware 重分配 |
| RoleRecent 扩展配置 | 75.94 | 混合了 recent/defer/cap 等机制 |
| 完整 `H+N` HorizonKV | 待验证 | 当前没有对应实现和结果 |

## 5. 总体实现架构

```text
Calibration videos
    |
    v
Attention probe + chunk provenance
    |
    v
raw_horizon_observations.parquet/csv
    |
    +--> H aggregation --------+
    |                           |
    +--> N trajectory ---------+--> stabilization --> horizon_profile.csv
                                                        |
                                                        v
                                               budget allocator
                                                        |
                                                        v
                                              horizon_budget.csv
                                                        |
                                                        v
Streaming inference --> NewBuffer --> aging --> HistCache --> delayed QA
                            |                         |
                            +---- fixed B_new/B_hist-+
```

实现分成三个独立层次：

1. **画像层**：只负责从 calibration 数据生成冻结 `H/N`；
2. **分配层**：只负责把画像映射成严格守恒的整数预算；
3. **运行时层**：只消费冻结预算，不读取测试答案或未来问题。

三个层次必须能够分别做单元测试和替换，不能把画像计算嵌入在线推理路径。

## 6. 冻结产物的数据契约

### 6.1 Calibration split manifest

建议文件：`results/horizonkv/splits/<split_name>.json`

```json
{
  "schema_version": 1,
  "dataset": "streamingbench",
  "seed": 0,
  "split_unit": "video",
  "calibration_video_ids": [],
  "test_video_ids": [],
  "sha256": "..."
}
```

硬约束：

- 两个 video id 集合交集为空；
- 一个视频下的全部问题属于同一 split；
- profile 和实验配置中记录 manifest 路径及哈希；
- 正式 test 运行不能重新生成 profile。

### 6.2 原始观测

建议文件：`raw_horizon_observations.csv`。首版使用 CSV 便于检查；数据量过大时再改 Parquet。

必需字段：

| 字段 | 类型 | 含义 |
|---|---|---|
| `video_id` | str | 视频标识 |
| `probe_id` | str | 固定 probe 或 calibration question 标识 |
| `layer` | int | `[0, 27]` |
| `kv_head` | int | `[0, 3]` |
| `source_chunk` | int | 被读 token 来源 chunk |
| `read_chunk` | int | 发起读出的 chunk |
| `delta` | int | `read_chunk - source_chunk` |
| `region` | enum | `history/new/sink/text/boundary` |
| `attention_mass` | float | 区域 Attention 总量 |
| `token_count` | int | 区域 token 数 |
| `visual_attention_mass` | float | 当次视觉 Attention 总量 |
| `valid` | bool | 是否参与聚合 |

`H_g` 与 `N_g` 必须能够从该原始表重新计算，不能只保存最终 112 个分数。

### 6.3 Head profile

建议文件：`horizon_profile.csv`，严格一行对应一个物理 KV Head，共 112 行。

```text
schema_version,model,dataset,split_sha256,layer,kv_head,
h_raw,n_raw,h_norm,n_norm,reliability,h_shrunk,n_shrunk,
n_videos,n_observations,profile_version
```

读取时必须验证：

- `(layer, kv_head)` 唯一且完整；
- 行数与模型配置一致；
- 数值有限，归一化值位于约定范围；
- `model`、层数和 KV Head 数与运行模型一致；
- test 配置引用的 split 哈希与 profile 元数据一致。

### 6.4 Budget table

建议文件：`horizon_budget.csv`。

```text
schema_version,profile_sha256,layer,kv_head,h,n,
importance,total_budget,hist_ratio,hist_budget,new_budget,
budget_version
```

运行前必须验证以下不变量：

```text
total_budget == hist_budget + new_budget
B_min <= total_budget <= B_max
sum(total_budget) == num_layers * num_kv_heads * B_avg
0 <= hist_ratio <= 1
```

整数舍入后的预算必须精确守恒，不允许只接受“近似等于总预算”。

## 7. 离线画像实现

### 7.1 建议代码入口

| 文件 | 职责 |
|---|---|
| `head_analysis/profile_horizonkv.py` | 流式重放 calibration 视频，写原始观测 |
| `head_analysis/build_horizonkv_profile.py` | 计算 H/N、bootstrap、收缩并冻结 profile |
| `head_analysis/build_horizonkv_budget.py` | 从 profile 生成严格守恒预算表 |

不直接修改旧画像产物；新实现使用新的 schema/version，便于与 `ours_top100`、v8 和 RoleRecent 公平对照。

### 7.2 统一时间复用曲线 `[PROPOSED]`

不直接从两个互不相关的 Attention 统计量分别定义 H 和 N。先为每个物理 KV Head 建立一条按 token 年龄分组的时间复用曲线，再从曲线的短期段和长期段导出 N/H。

对 calibration 视频中时刻 `t` 到达的真实问题，记录问题语义 token 对不同来源时间视觉 token 的 Attention。对物理 KV Head `g` 和年龄区间 `delta`：

```text
mass_g,t(delta) = Head g 对该年龄区间视觉 token 的 Attention 总量
prior_t(delta)  = 该年龄区间 token 数 / 全部视觉 token 数
reuse_g,t(delta) = log(1 + mass_g,t(delta) / (prior_t(delta) + eps))
```

该定义同时保留绝对 Attention 强度和相对 token 数量的富集程度。若只使用区域 Attention share，旧区域会因 token 更多而天然占优；若只使用 per-token ratio，极少量 Attention 也可能因分母很小产生异常高分。

问题 Attention 的采集约束：

1. 使用 full cache 或足够大的参考缓存重放 calibration 视频，避免现有压缩策略先删除 token 后反向污染画像；
2. 每个视觉 token 必须记录 `source_frame/source_time`，不能只记录 previous/current；
3. 使用问题和选项的语义 token，排除 system prompt、固定模板和 answer prefix；
4. eager Attention forward 不写回在线 KV Cache；
5. Qwen2.5-VL-7B 中属于同一物理 KV Head 的 7 个 query heads 做组内聚合；
6. 排除文本、特殊 token、attention sink 和 chunk 拼接边界；
7. token 年龄优先使用秒或采样帧表示，避免不等长 chunk 造成偏差。

按视频等权聚合，避免长视频或问题数量多的视频主导画像：

```text
R_g(delta) = median_over_videos(
    mean_over_questions_in_video(reuse_g,q(delta))
)
```

当前 `obs_prev_current_chunk_attention_eager.py` 只区分 previous/latest，无法直接产生多年龄曲线；`profile_effective_memory_readout.py` 已接近真实问题读出，但仍需增加 token 来源时间和 age-bin 聚合。

### 7.3 从复用曲线导出 `N_g/H_g` `[PROPOSED]`

设新信息保护范围为 `D_new`：

```text
N_raw_g = sum(w_delta * R_g(delta), delta=1..D_new)
H_raw_g = equal_bin_mean(R_g(delta), delta>D_new)
```

- `N_g` 表示新到 token 在至少一个后续时刻仍被读取的需求；
- `H_g` 表示 token 进入远历史后的持续读出需求；
- `delta=0` 不进入 `N_g`，因为它只能反映即时显著性，不能反映延续需求；
- 长期部分按 age bin 等权，不能让样本更多的近历史区间主导 H。

候选时间分桶：

```text
N: 1, 2, 3-4 个流式步
H: 5-8, 9-16, 16+ 个流式步
```

这组边界尚未确认。应先收集完整复用曲线，再在独立 validation 视频上从 `D_new ∈ {1, 2, 4}` 中选择。最终 `D_new` 还应与 M1 运行时真正支持的新信息生命周期一致。

真实 calibration 问题读出作为主候选信号；固定 local/global pseudo-query 作为完全 query-agnostic 的消融和稳定性对照。使用 calibration 问题不等于在线读取未来问题，但论文必须明确这是 distribution-level offline calibration。

### 7.4 稳定化

首版流程：

1. 对每层内部的 `H/N` 分别做 median/MAD robust normalization；
2. 以视频为采样单位做 bootstrap；
3. 从置信区间宽度得到 `reliability ∈ [0,1]`；
4. 向同层均值收缩：`P_shrunk = r * P + (1-r) * layer_mean`；
5. 同时保留 raw、normalized 和 shrunk 值，避免无法追溯。

`reliability` 的具体映射公式在第一次 calibration 数据分布可视化后确定，不能预先假设一个没有数据依据的阈值。

### 7.5 因果校验 `[PROPOSED]`

Attention 画像只是低成本估计器，不能单独证明增加对应预算有用。冻结画像前，在 validation 视频上进行等预算干预：

```text
H causal effect =
    NLL(删除 Head g 的远历史 token) - NLL(完整参考缓存)

N causal effect =
    NLL(让 Head g 的新信息提前过期) - NLL(正常生命周期)
```

主统计量使用 teacher-forced answer NLL，QA accuracy 作为最终任务指标。首轮无需逐一消融 112 个 Head，可以比较 `top / bottom / random` 三组：

```text
高 H 组的历史删除损失 > 低 H 组
高 N 组的提前过期损失 > 低 N 组
learned 排序的干预响应 > random
learned 排序的干预响应 > inverted
```

如果这些关系不成立，不能通过调预算掩盖问题，应先修改 Attention proxy、时间分桶或聚合方式。

### 7.6 数值例子 `[PROPOSED]`

假设一个 calibration 视频被划分为多个时间段，问题在时刻 `t=10` 到达。根据视觉 token 来源时间，将当前缓存划分为五个年龄区间：

| 年龄区间 | `delta=1` | `delta=2` | `delta=3-4` | `delta=5-8` | `delta=9+` |
|---|---:|---:|---:|---:|---:|
| token 数量占比 `prior` | 0.10 | 0.10 | 0.20 | 0.40 | 0.20 |

同一个问题中，两个物理 KV Head 的 Attention mass 如下。Attention mass 是相对全部可见 key 的质量，因此每行不要求和为 1，剩余 Attention 可能落在文本或其他 token 上。

| Head | `delta=1` | `delta=2` | `delta=3-4` | `delta=5-8` | `delta=9+` |
|---|---:|---:|---:|---:|---:|
| A：偏新信息延续 | 0.06 | 0.04 | 0.06 | 0.10 | 0.04 |
| B：偏远历史复用 | 0.01 | 0.01 | 0.03 | 0.20 | 0.12 |

代入：

```text
reuse(delta) = log(1 + attention_mass(delta) / prior(delta))
```

得到时间复用曲线：

| Head | `R(1)` | `R(2)` | `R(3-4)` | `R(5-8)` | `R(9+)` |
|---|---:|---:|---:|---:|---:|
| A | 0.470 | 0.336 | 0.262 | 0.223 | 0.182 |
| B | 0.095 | 0.095 | 0.140 | 0.405 | 0.470 |

若候选定义使用：

```text
N_raw = 0.5 * R(1) + 0.3 * R(2) + 0.2 * R(3-4)
H_raw = 0.5 * R(5-8) + 0.5 * R(9+)
```

则：

| Head | `N_raw` | `H_raw` | 解释 |
|---|---:|---:|---|
| A | 0.388 | 0.203 | 新到信息在前几个未来时刻仍有较强读出 |
| B | 0.104 | 0.438 | 对近期信息响应弱，但明显读取远历史 |

真实 profile 会先对一个视频内的多个问题求均值，再跨视频取中位数，以上只展示单个问题的计算。

假设层内归一化和可靠性收缩后，两个 Head 的总预算分别为 6100 和 5900，且暂取 `kappa=1`：

```text
hist_ratio = H / (H + N)
```

| Head | 总预算 | 历史预算（约） | 新信息预算（约） |
|---|---:|---:|---:|
| A | 6100 | 2100 | 4000 |
| B | 5900 | 4800 | 1100 |

这个例子说明：H/N 不只决定谁获得更大的总缓存，还决定相近总容量的 Head 如何在历史和新信息之间分工。

最后在 validation 视频上做等量干预。假设每次删除相同数量 token 后得到：

| 干预 | Head A 的 answer NLL 增量 | Head B 的 answer NLL 增量 |
|---|---:|---:|
| 删除远历史 | +0.08 | +0.43 |
| 让新信息提前过期 | +0.37 | +0.05 |

该结果与画像方向一致，才说明 A 的高 N、B 的高 H 具有实际缓存意义。如果干预结果相反，即使 Attention 曲线看起来清晰，也不能冻结这版 H/N 定义。

### 7.7 StreamingBench 真实视频例子

真实样本来自 `data/streamingbench/streamingbench_realtime.json` 的 `video_id=1`：

```text
video: /data/streamingbench/videos/sample_416_real.mp4
内容: 驾驶舱飞行画面
问题到达时间: 157 秒
问题: What colors are the pilot's handheld controls right now?
答案: Black.
任务: Attribute Recognition
采样率: 0.5 FPS
```

到 157 秒时共采样 79 帧。受前一个 6 秒问题边界影响，实际流式编码区间为：

```text
sample frame  0-2   -> 约 0-4 秒
sample frame  3-18  -> 约 6-36 秒
sample frame 19-34  -> 约 38-68 秒
sample frame 35-50  -> 约 70-100 秒
sample frame 51-66  -> 约 102-132 秒
sample frame 67-78  -> 约 134-156 秒
question arrives   -> 157 秒
```

现有 `raw_prev_current_attention_kv.csv` 在最后一个编码区间 `frame_start=67, frame_end=79` 保存了真实观测。当时：

```text
visual_start = 14
pre_len = 6014
post_len = 22160
previous visual tokens = 6014 - 14 = 6000
latest chunk tokens = 22160 - 6014 = 16146
latest token fraction = 16146 / (6000 + 16146) = 0.729071
```

以 Layer 0、物理 KV Head 0 为例，关联的 query heads 是 0-6。已有 local/global pseudo-query 观测为：

| Probe | previous mass | latest mass | previous share | latest share |
|---|---:|---:|---:|---:|
| local | 5.198730 | 0.440910 | 0.921820 | 0.078180 |
| global | 3.843750 | 0.330190 | 0.920893 | 0.079107 |

表中 mass 是 7 个 query heads 的和，因此可以大于 1；share 和相对密度不受求和或求均值影响。现有 observer 的计算为：

```text
local_latest_per_token_ratio
  = (0.440910 / 16146) / (5.198730 / 6000)
  = 0.031517

global_latest_per_token_ratio
  = (0.330190 / 16146) / (3.843750 / 6000)
  = 0.031922

s_obs = (0.078180 + 0.079107) / 2
      = 0.078644

b_obs = (log(0.031517) + log(0.031922)) / 2
      = -3.450845

r_obs = exp(b_obs)
      = 0.031719
```

解释：最新区占 72.91% 的视觉 token，但只获得约 7.86% 的 local/global Attention；其平均单 token Attention 密度只有 previous 区的 3.17%。所以 Layer 0、KV Head 0 在这次观测中明显偏 previous。

同一时刻 Layer 0、KV Head 2 的 `s_obs=0.250877`、`r_obs=0.124404`，对最新区的相对响应约为 Head 0 的 3.9 倍。这是同层不同物理 KV Head 时间偏好不同的真实例子。

但这些数值还不是正式 H/N：

- 读出来自固定 pseudo-query，不是 157 秒到达的真实问题；
- previous 区已经被 HERMES 压缩到 6000 token，观测受到旧策略影响；
- CSV 只区分 previous/latest，没有保存 token 的来源时间；
- 因此不能从这份 CSV 拆出 `delta=1,2,3...`，也不能给出该视频真实的 H/N。

正式离线采集时，应在 full-cache 重放中使用这个真实问题，将来源区间映射为：

```text
frame 67-78: delta=0，刚到达，不能进入 N
frame 51-66: delta=1，进入 N 候选
frame 35-50: delta=2，进入 N 候选
frame 19-34: delta=3，取决于选定的 D_new
frame  3-18: delta=4，进入 N 或 H 取决于 D_new
frame  0-2 : 最远历史，进入 H
```

然后用真实问题 Attention 分别计算每个年龄区间的 `reuse_g(delta)`，跨其他 calibration 视频聚合后才能得到冻结 H/N。

### 7.8 Attention 分数计算细节 `[PROPOSED]`

对每层，Qwen eager attention 的基本计算是：

```text
score[l,h,j,i] = dot(Q[l,h,j], K[l,g(h),i]) / sqrt(head_dim)
                 + causal_mask + padding_mask

A[l,h,j,:] = softmax(score[l,h,j,:])
```

其中：

- `l` 是层；
- `h` 是 query head；
- `g(h)=floor(h/7)` 是它共享的物理 KV Head；
- `j` 是问题语义 token；
- `i` 是缓存 key/token；
- Q/K 在点积前已经应用 Qwen2.5-VL 的 M-RoPE；
- 推理阶段没有 attention dropout。

Qwen2.5-VL-7B 的单层 Attention tensor 形状为：

```text
[batch=1, query_heads=28, question_tokens=Q, cached_tokens=K]
```

对物理 KV Head `g`，先对它对应的 7 个 query heads 和选中的问题语义 token 求平均：

```text
A_bar[l,g,i]
  = mean over h in {7g,...,7g+6}
    mean over semantic question token j
    A[l,h,j,i]
```

此时 `sum_i A_bar[l,g,i] = 1`。正式 profile 必须使用平均值而不是 7 个 Head 的和，否则绝对 mass 会随 GQA group size 改变。

对年龄区间 `delta`：

```text
mass[l,g,delta]
  = sum A_bar[l,g,i]
    for visual token i whose source age belongs to delta

prior[delta]
  = number of visual tokens in delta / number of all visual tokens

reuse[l,g,delta]
  = log(1 + mass[l,g,delta] / (prior[delta] + eps))
```

这里的 `mass` 相对全部 key 计算，而 `prior` 只在视觉 token 内计算。这样一个主要读取文本、几乎不读取视觉缓存的 Head 不会仅凭某个视觉区间内部比例较高而获得虚假的高分。

问题 token 的主口径建议包括真实 MCQA 中的问题和选项，排除 system prompt、固定回答模板和 answer prefix。另做 question-only pooling 消融，检查选项文本是否显著改变 Head 排序。

不同读出源不能在 raw mass 层直接混合。真实问题、local pseudo-query 和 global pseudo-query 应分别生成曲线，完成层内归一化后再比较或融合。

### 7.9 10-video pilot 的时间预算向量结论 `[PROPOSED]`

当前 10-video pilot 使用：

```text
结果目录：
results/myobservations/horizonkv_pilot_10v/
results/myobservations/horizonkv_pilot_10v_fixed672/
results/myobservations/horizonkv_pilot_10v_fixed896/

模型：Qwen2.5-VL-7B
数据：StreamingBench 10 个视频，50 个问题
采样率：0.5 FPS
chunk：16 frames，约 32 秒
query_pool：last_n=4
fixed_frame_canvas_size：448 / 672 / 896
```

时间桶按来源 chunk 与当前 chunk 的距离定义：

```text
delta = current_chunk_idx - token_chunk_idx

delta=0  : 当前 / 最新 chunk
delta=1  : 上一个 chunk
delta=2  : 更早近历史
delta=3  : 中远历史
delta=4+ : 远历史
```

截至 v0.5，后续方案主口径按用户要求以 pseudo-query 为准；真实问题结果只作为对照，不混入 pseudo 主方案。

fixed896 下两类查询的全体 KV Head 平均时间向量为：

| 查询来源 | `[delta0, delta1, delta2, delta3, delta4+]` |
|---|---|
| pseudo-query | `[0.479, 0.223, 0.098, 0.077, 0.124]` |
| real question | `[0.447, 0.227, 0.107, 0.080, 0.139]` |

结论：两类查询都显示整体偏近期；real question 的远历史占比略高，但当前不作为预算主口径。

基于 fixed896 pseudo-query，首版可执行的 Head 分类和缓存比例建议如下：

| 类型 | Head 数 | 建议预算比例 `[delta0, delta1, delta2, delta3, delta4+]` | 解释 |
|---|---:|---|---|
| Recent-dominant | 16 | `[0.50, 0.22, 0.10, 0.09, 0.09]` | 强保当前与上一 chunk |
| Recent-leaning | 16 | `[0.45, 0.23, 0.12, 0.10, 0.11]` | 偏近期，但保留少量远历史 |
| Mixed-default | 60 | `[0.40, 0.22, 0.13, 0.11, 0.15]` | 默认稳健分配 |
| Balanced | 7 | `[0.36, 0.22, 0.13, 0.12, 0.17]` | 更均衡覆盖多个时间段 |
| History/bimodal | 13 | `[0.35, 0.20, 0.14, 0.13, 0.20]` | 当前 + 远历史双峰保护 |

其中 `[0.35, 0.20, 0.14, 0.13, 0.20]` 是当前最重要的一类比例。它不表示“历史型 Head 不需要近期”，而表示该类 Head 同时需要保留当前 chunk 与远历史 token。当前实验中更合理的论文表述应是：

```text
Head-specific temporal cache allocation learns a per-head temporal preference vector.
The observed heads are mostly recent-biased, but a stable subset shows current+far-history bimodality.
Therefore cache should be allocated per head and per time bin, instead of using one global temporal policy.
```

fixed896 pseudo-query 中远历史占比最高的特殊 Head：

| Head | `[delta0, delta1, delta2, delta3, delta4+]` | 说明 |
|---|---|---|
| L16 KV1 | `[0.375, 0.187, 0.110, 0.112, 0.215]` | 当前最强 pseudo 远历史 Head |
| L22 KV0 | `[0.371, 0.197, 0.107, 0.121, 0.204]` | 远历史和中远历史都较高 |
| L4 KV1 | `[0.385, 0.196, 0.120, 0.094, 0.204]` | 跨 448/672/896 稳定 |
| L14 KV0 | `[0.380, 0.198, 0.120, 0.099, 0.203]` | 跨 448/672/896 稳定 |
| L24 KV1 | `[0.381, 0.216, 0.108, 0.095, 0.199]` | 跨 448/672/896 稳定 |
| L5 KV1 | `[0.405, 0.184, 0.120, 0.095, 0.196]` | 稳定远历史候选 |
| L2 KV2 | `[0.417, 0.185, 0.118, 0.086, 0.194]` | 稳定远历史候选 |
| L5 KV2 | `[0.412, 0.178, 0.122, 0.095, 0.193]` | 稳定远历史候选 |
| L9 KV3 | `[0.422, 0.186, 0.116, 0.090, 0.185]` | 稳定远历史候选 |

fixed896 real-question 对照中，远历史最明显的特殊 Head：

| Head | `[delta0, delta1, delta2, delta3, delta4+]` | 说明 |
|---|---|---|
| L16 KV1 | `[0.359, 0.164, 0.105, 0.109, 0.263]` | real/pseudo 均突出，优先特殊保护 |
| L24 KV1 | `[0.366, 0.185, 0.106, 0.100, 0.244]` | real/pseudo 均突出，优先特殊保护 |
| L4 KV1 | `[0.371, 0.198, 0.119, 0.085, 0.226]` | 两类查询一致偏远历史 |
| L17 KV2 | `[0.376, 0.192, 0.101, 0.108, 0.224]` | real 更强，pseudo 也在 top far 附近 |
| L20 KV1 | `[0.368, 0.206, 0.106, 0.110, 0.211]` | real 历史型，pseudo 不作为主保护依据 |

固定分辨率下的主证据不是强二元 H/N 分化，而是稳定的相对时间偏好差异：

- recent-heavy Head 的分化很明显；
- 纯历史型 Head 很少；
- 有实际意义的是 current+far-history bimodal Head；
- 因此第一版 runtime 不应把 Head 简单切成 H/N 两组，而应加载每个 Head 的 5-bin 时间预算。

运行时首版建议：

```text
1. delta0 当前 chunk 尽量全保，作为安全下界；
2. 可压缩部分按每个 Head 的 5-bin 预算分配；
3. delta0/delta1 桶内优先 recency；
4. delta2/delta3 桶内使用 recency + attention/reuse score；
5. delta4+ 桶内使用 attention/reuse topK + 时间均匀覆盖；
6. History/bimodal Head 的 delta4+ 下界设为 18%-20%；
7. Recent-dominant Head 的 delta4+ 控制在 8%-10%。
```

建议冻结为首个预算产物：

```text
head_temporal_policy.csv

layer,kv_head,query_source,class,
delta0_budget_ratio,delta1_budget_ratio,delta2_budget_ratio,
delta3_budget_ratio,delta4plus_budget_ratio,
evidence_resolution,n_videos,n_questions
```

首个 runtime 实验应优先用 pseudo-only 的 `query_source=pseudo` 预算表；real-question 预算表保留为独立对照实验，不能和 pseudo 结果混合后再宣称 query-agnostic。

## 8. 预算分配实现

### 8.1 Head 总容量

```text
importance_g = alpha * H_shrunk_g + beta * N_shrunk_g
weight_g = softmax(importance_g / tau)
raw_budget_g = B_min + remaining_budget * weight_g
```

随后执行带 `B_min/B_max` 的 water-filling，并用 largest-remainder 方法整数化。建议实现为纯 NumPy 函数，输入输出不依赖模型，便于穷举测试。

函数契约建议为：

```python
allocate_head_budgets(
    h: np.ndarray,              # [L, H_kv]
    n: np.ndarray,              # [L, H_kv]
    average_budget: int,
    alpha: float,
    beta: float,
    temperature: float,
    min_budget: int,
    max_budget: int,
) -> np.ndarray                 # int32 [L, H_kv]
```

### 8.2 Head 内历史/新信息预算

```text
hist_ratio_g = clip(
    (H_g + eps) / (H_g + kappa * N_g + 2*eps),
    rho_min,
    rho_max,
)
hist_budget_g = round(total_budget_g * hist_ratio_g)
new_budget_g = total_budget_g - hist_budget_g
```

M0 可复用现有 `build_kv_head_split_budget_table()` 和 `_select_split_budget_tokens()`，但需要新增“直接加载冻结 `hist_ratio`/budget table”的路径，避免运行时再次对 score 做 min-max 归一化。

### 8.3 公平比较约束

- uniform、H-only、N-only、H+N、random、inverted 使用完全相同的全局总预算；
- 所有方法使用同一个 token selector 和压缩触发条件；
- 只有被消融的画像维度或预算映射可以变化；
- 每次运行保存实际 112 个预算，而不只保存 CLI 参数。

## 9. 在线运行时实现

### 9.1 方法版本 M0：接入现有 region split

目标：以最小代码改动验证冻结 `H+N` 画像和预算分配是否有效。

```text
新 chunk 到达
    -> 现有 encode_video_chunk
    -> 保存 pre/post head lengths
    -> previous positions 视为 history
    -> latest chunk positions 视为 new
    -> 按冻结 B_hist/B_new 分区 top-k
    -> 现有 ragged physical compact
```

M0 的限制：`new` 仅表示“最新 chunk”，没有显式多步 aging，因此只能验证联合预算的第一轮信号，不能声称完整实现了 `N_g` 生命周期。

### 9.2 方法版本 M1：显式 aging 生命周期

每个物理 KV Head 至少维护：

```text
key/value storage
absolute_position
source_chunk_id
age_in_chunks
region: TEXT | NEW | HIST
importance state（若 selector 需要）
```

状态迁移：

```text
APPEND:
    incoming visual token -> NEW(age=0)

ADVANCE:
    every completed chunk increments age

PROMOTE:
    NEW token reaching new_horizon becomes historical candidate

COMPACT:
    keep <= B_new from NEW
    keep <= B_hist from HIST + promoted candidates
    overflow fill only within the frozen total B_g

QUERY:
    read frozen cache; no retroactive re-compression using question content
```

关键实现不变量：

- 文本前缀不计入视觉预算，且不能被视觉压缩误删；
- 每个保留 token 的 K、V、position 和 provenance 同步移动；
- M-RoPE 所需位置元数据必须与 ragged token 顺序一致；
- Head 内保留索引有序、唯一且在合法范围；
- query 后恢复视频缓存时，按每个 Head 的真实 prefill 长度截断；
- strict 模式下任何时刻都不能超过 `B_g`，high-watermark 模式必须单独报告峰值。

### 9.3 方法版本 M2：系统优化

在 M1 准确性闭环后再实施：

- 按相近 Head 长度分桶；
- 使用 packed K/V + offsets/lengths，减少 Python list 调度；
- 合并 gather、position update 和 compact；
- 在 ingestion 阶段异步或 chunk-level 压缩；
- 限制极端 `B_max`，避免最长 Head 主导 kernel；
- 用 CUDA Event 和显式同步测量各阶段。

M2 的验收不是“代码更复杂”，而是在相同准确率和预算下实际降低延迟或提高吞吐。

## 10. CLI 与配置设计

建议新增单一方法入口，避免继续组合大量互相冲突的旧 flag：

```text
--horizonkv
--horizonkv_profile <horizon_profile.csv>
--horizonkv_budget <horizon_budget.csv>
--horizonkv_runtime_version {m0,m1,m2}
--horizonkv_budget_mode {strict,high_watermark}
--horizonkv_high_watermark_ratio 1.1
--horizonkv_new_horizon_chunks 1
--horizonkv_trace_dir <path>
```

预算生成脚本单独接受：

```text
--average_budget
--alpha
--beta
--temperature
--min_budget
--max_budget
--kappa
--rho_min
--rho_max
```

`video_qa/run_infer.py` 只负责校验和转发冻结产物；它不负责从测试数据重新计算画像。

冲突规则：HorizonKV 首次接入时应禁止与下列旧策略同时启用，以保证实验可解释：

- `kv_head_role_recent_*`；
- `kv_head_budget_memory_current_split` 的旧 score-to-ratio 路径；
- `defer_last_frame_compress`；
- 会覆盖 per-head budget 的 RoleKV 配额模式。

后续若决定组合，必须建立独立方法版本并补齐组合消融。

## 11. 可观测性与运行产物

每次运行至少保存：

```text
run_config.json
split_manifest.json or split reference + sha256
horizon_profile.csv or profile reference + sha256
horizon_budget.csv
head_cache_stats.csv
timing.json
eval_results.txt
```

`head_cache_stats.csv` 建议字段：

```text
video_id,chunk_id,layer,kv_head,
target_total,target_hist,target_new,
actual_hist,actual_new,overflow,pre_len,post_len,
compression_ms
```

必须报告：

- 名义总预算与真实平均/最大 Head 长度；
- 峰值视觉 KV 数量，尤其是 high-watermark 模式；
- ingestion、compression、query prefill、decode 的独立时间；
- TTFT mean/median/P95；
- QA 总分和各子任务分数。

## 12. 验证与测试计划

### 12.1 单元测试

| 测试 | 验收条件 |
|---|---|
| profile schema | 缺 Head、重复 Head、NaN、模型不匹配均报错 |
| split leakage | calibration/test video id 有交集时报错 |
| allocator conservation | 多组随机输入下整数预算总和精确守恒 |
| allocator bounds | 每个 Head 始终满足 min/max |
| H-only/N-only | 关闭一个维度后结果不受该维度输入变化影响 |
| region selection | 分区不足时正确 overflow fill，无重复索引 |
| provenance compact | K/V/position/chunk_id 经 gather 后完全对齐 |
| query rollback | 回答结束后恢复各 Head 原缓存长度和内容 |

### 12.2 集成测试

1. 单视频、两个 chunk、极小预算，人工检查 NEW -> HIST；
2. Qwen 1 个视频 smoke，比较 HorizonKV 关闭时与原路径一致；
3. 5 题 smoke-full 只检查运行正确性，不记录为正式 full 结果；
4. StreamingBench half 共 1255 题，用于稳定对照；
5. StreamingBench full 共 2495 题，完成后才记录正式 full 结论。

### 12.3 主实验矩阵

在每个 `B_avg ∈ {2000, 4000, 6000}` 上运行：

| 方法 | H | N | 画像顺序 | 目的 |
|---|---:|---:|---|---|
| Uniform | 0 | 0 | uniform | 基线 |
| H-only | 1 | 0 | learned | 验证历史维度 |
| N-only | 0 | 1 | learned | 验证延续维度 |
| H+N | 1 | 1 | learned | 完整方法 |
| Random | 1 | 1 | shuffled | 排除任意非均匀预算收益 |
| Inverted | 1 | 1 | reversed | 验证画像方向性 |

主表之外还需要：layer-wise only、旧 v8、RoleRecent 75.94 配置和 oracle 上界。

### 12.4 full-cache / full-prefix 正确性状态 `[VERIFY]`

需要区分两个概念：

1. **画像 pilot 的 full-prefix no-compression**：用于采集 Attention 分布，不生成答案，因此不能判断 QA 是否正确；
2. **问答实验的 full/full-cache baseline**：需要查看 `results.csv` 和 `eval_results.txt`。

当前仓库中可追溯的相关问答结果：

| 目录 | 样本数 | Average qa_acc | 结论 |
|---|---:|---:|---|
| `results/qwen2.5_vl_7b/streamingbench/fps0.5-kv6000-hermes` | 2495 | 76.95 | 完整 StreamingBench 结果，不是全对 |
| `results/qwen2.5_vl_7b/streamingbench/fps0.5-kv6000-streamingvlm` | 2495 | 76.95 | 完整 StreamingBench 结果，不是全对 |
| `results/qwen2.5_vl_7b/streamingbench/fps0.5-kv6000-hermes-smoke-full` | 5 | 80.00 | smoke-full，只跑 5 题，4/5 正确 |

`fps0.5-kv6000-hermes-smoke-full` 的 5 题中，错误样例为：

```text
video_id: 1
question: What colors are the pilot's handheld controls right now?
gold: A) Black.
pred: D) Black and orange.
qa_acc: 0
```

因此目前不能说“保留全部 cache 后回答都正确”。更准确的状态是：

```text
full-prefix no-compression 画像采集没有 QA 正确性指标；
已有 smoke-full QA 不是全对；
已有完整 2495 题 baseline 也不是全对。
```

后续若要回答 full-cache 上界问题，需要单独运行明确配置的 full-cache QA baseline，并保存：

```text
save_dir
run_config.json
results.csv
eval_results.txt
实际是否触发 compression
实际峰值 token/cache 长度
```

## 13. 分阶段实施清单

### M0：画像与预算闭环

- `[TODO]` 生成按视频隔离的 split manifest；
- `[TODO]` 实现原始观测采集；
- `[TODO]` 实现 `H/N` 聚合和冻结 profile；
- `[TODO]` 实现严格守恒 allocator；
- `[TODO]` 为现有 ragged split 增加直接加载 budget table；
- `[TODO]` 完成单元测试和单视频 smoke；
- `[TODO]` 在 budget=6000 完成 uniform/H/N/H+N/random/inverted 初筛。

### M1：真实时间生命周期

- `[TODO]` 增加 `source_chunk_id/age/region` 元数据；
- `[TODO]` 实现 NEW 多 chunk 保留与晋升；
- `[TODO]` 实现 strict/high-watermark 两种触发；
- `[TODO]` 验证 M-RoPE、ragged prefill 和 query rollback；
- `[TODO]` 完成三预算 half 主实验；
- `[TODO]` 完成 frozen-profile test split。

### M2：效率闭环

- `[TODO]` 统一 CUDA 计时；
- `[TODO]` 完成长短 Head 分桶；
- `[TODO]` 评估 packed layout；
- `[TODO]` 报告峰值显存、吞吐、TTFT 和 decode tokens/s；
- `[TODO]` 判断是否形成 accuracy-memory-latency Pareto 改善。

## 14. 当前设计决策与待讨论问题

### 已确认

- `[DECISION]` calibration/test 必须按视频划分；
- `[DECISION]` 在线阶段只加载冻结画像和预算；
- `[DECISION]` M0 复用现有 token selector，先隔离画像变量；
- `[DECISION]` 总预算必须在整数化后精确守恒；
- `[DECISION]` 75.94 只作为组合策略参考点；
- `[DECISION]` 当前 TTFT 数据表明尚未实现加速。

### 待讨论

- `[QUESTION]` 是否确认以真实 calibration 问题读出为主、固定 pseudo-query 为 query-agnostic 消融？
- `[QUESTION]` 是否确认先收集完整时间复用曲线，再在 validation 上选择 `D_new ∈ {1,2,4}`？
- `[QUESTION]` 时间分桶最终使用秒、采样帧，还是流式步？当前建议使用秒或采样帧。
- `[QUESTION]` 画像稳定化的 reliability 应由 CI 宽度、rank stability 还是二者联合给出？
- `[QUESTION]` 总预算先 layer-wise 再层内分配，还是在 112 个 Head 上全局 water-filling？
- `[QUESTION]` M1 中 `new_horizon` 应由 `N_g` 决定，还是只让 `N_g` 决定 `B_new`？
- `[QUESTION]` high-watermark 的临时超额是否计入论文主预算和峰值显存？建议必须计入峰值。
- `[QUESTION]` M0 的成功门槛是超过 74.90，还是只要求 H+N 显著优于 random/inverted？

## 15. 迭代记录

### 文档 v0.6（2026-08-26）

- 将运行时结构冻结为固定容量 `RecentCache + MemoryCache`；
- `RecentCache` 统一为 2691 visual-token FIFO，不再定义为固定最新一帧；
- 明确低 token 帧可以在 Recent 中同时保留多帧，Recent/Memory 容量不互借；
- 固定四类 Head 的 Memory 容量，使 112 个 Head 平均预算严格为 6000；
- 在线驱逐只使用与 pilot 一致的三个固定 pseudo probes；
- 定义 `source_chunk_id`、Attention EMA、warm/cold、cold anchor 和四类 age penalty；
- 明确首版复用 ragged prefill，保证压缩后预算但仍需报告追加 chunk 的临时峰值；
- 规定 HorizonKV 为默认关闭的独立入口，首版禁止与旧组合策略叠加。

实现变化：

- 新增 `head_analysis/horizonkv_policy.py`，负责冻结策略校验、固定 pseudo
  probes、Attention EMA、Recent FIFO 和四类 Memory 驱逐；
- `HeadRaggedLayerCache` 增加默认关闭的 `flat_source_chunk` 与
  `flat_attention_ema` sidecar，并在 append/prune/truncate 中同步更新；
- ragged prefill 在 HorizonKV 启用时记录 chunk provenance，并通过独立 selector
  分支压缩；未启用时继续执行原 selector；
- `video_qa/base.py` 和 `video_qa/run_infer.py` 增加
  `--horizonkv_policy`、`--horizonkv_recent_capacity` 和
  `--horizonkv_ema_gamma`；
- 新增 `tests/test_horizonkv_policy.py` 的 7 个 CPU 测试。

验证记录：

```text
CPU tests: 7/7 passed
Smoke annotation:
  results/myobservations/horizonkv_runtime_smoke_1q/anno_1q.json
Smoke output:
  results/myobservations/horizonkv_runtime_smoke_1q_last8/1_0.csv
Input: StreamingBench video_id=136, end_time=19s, sample_fps=0.5
Compression:
  visual budget range = [5320, 6964]
  Recent occupied = 2691 for all heads in this sample
  Memory occupied range = [2629, 4273]
QA: 1/1 correct
Default-off regression:
  results/myobservations/horizonkv_regression_ragged_hermes_1q/1_0.csv
  original ragged HERMES completed, uniform visual budget=6000, QA 1/1 correct
```

该 smoke 只证明代码路径可运行、预算生效和问答回滚成功，不构成方法准确率结论。

### 文档 v0.5（2026-08-26）

- 写入 10-video pilot 的 5-bin 时间预算向量结论；
- 同时记录 pseudo-query 与 real-question 两类查询的 fixed896 对照；
- 明确后续主方案按用户要求采用 pseudo-only，real-question 只作为独立对照；
- 记录 History/bimodal 类预算比例 `[0.35, 0.20, 0.14, 0.13, 0.20]`；
- 特别标出 L16 KV1、L22 KV0、L4 KV1、L14 KV0、L24 KV1 等特殊 Head；
- 补充 full-cache / full-prefix 正确性状态：现有可追溯结果不能证明 full cache 全部回答正确，smoke-full 为 5 题 80.00%。

### 文档 v0.4（2026-08-26）

- 增加 StreamingBench `sample_416_real.mp4` 的真实视频和问题例子；
- 使用已有 Layer 0 KV Head 0/2 数据复算 previous/latest Attention 指标；
- 明确已有 pseudo-query、压缩缓存、二分 region 数据不能冒充真实 H/N；
- 补充 Qwen GQA Attention tensor、M-RoPE 后点积、query-token pooling 和物理 KV Head 聚合细节；
- 明确正式 absolute mass 必须在 7 个共享 query heads 上求平均；
- 给出该真实问题在 full-cache age-bin 采集中的时间映射。

### 文档 v0.3（2026-08-26）

- 增加单问题、双 Head 的完整数值例子；
- 展示 Attention mass、token prior、时间复用曲线和 H/N 的逐步计算；
- 展示 H/N 如何同时影响 Head 总容量与历史/新信息预算拆分；
- 增加与画像方向对应的 answer NLL 因果干预示例。

### 文档 v0.2（2026-08-26）

- 记录 H/N 离线确定方案，当前状态为 `[PROPOSED]`；
- 将两个独立 Attention 指标改为统一时间复用曲线 `R_g(delta)`；
- 建议从短期曲线导出 N、从远期曲线导出 H，并排除 `delta=0`；
- 明确 full-cache 采集、token 来源时间、物理 KV Head 聚合和按视频等权要求；
- 增加基于 teacher-forced answer NLL 的历史删除/新信息提前过期因果校验；
- 保留真实 calibration 问题与固定 pseudo-query 的信号来源决策。

### 文档 v0.1（2026-08-26）

- 建立独立的实际技术实现文档；
- 盘点现有 per-head budget、memory/current split、ragged prefill、RoleRecent 和 defer-last 能力；
- 定义 profile、budget、runtime 三层架构；
- 定义 M0/M1/M2 路线、数据契约、测试和实验矩阵；
- 明确完整 `H+N` HorizonKV 当前仍为 `[TODO]`。

### 后续版本记录模板

```text
### 文档 vX.Y（YYYY-MM-DD）

讨论结论：
- ...

实现变化：
- ...

新增证据：
- 实验目录：...
- 配置：...
- 结果：...

已知问题：
- ...

下一步：
- ...
```
