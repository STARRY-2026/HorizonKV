# Memory/Current Budget Split 方法说明

## 概述

HERMES 的原始 KV-head 预算机制给每个 `[layer, kv_head]` 分配一个总预算（整数），选择 token 时按 `score = attn×(1-α) + recency×α` 统一打分后取 top-k。这种方法不区分 token 的来源（旧记忆 vs 新 chunk），无法针对不同 head 的记忆偏好差异化分配预算。

**Memory/Current Budget Split** 将每个 KV head 的总预算拆分为「记忆预算」和「当前预算」两部分，分别从旧 chunk 的 token 和新 chunk 的 token 中独立选取。记忆偏好越强的 head，记忆预算占比越大。

---

## 核心概念

### 区域划分

每次 `encode_video_chunk` 追加新帧时，系统会在追加前后各拍一张快照（snapshot），记录每个 KV head 的 token 数量：

| 区域 | 定义 | 含义 |
|------|------|------|
| **Memory Region** | `[text_keep, pre_len)` | 最新 chunk **之前**就存在的旧 visual token |
| **Current Region** | `[pre_len, head_len)` | 最新 chunk **刚追加**的 visual token |

- 第一个 chunk：memory region 为空，全部预算给 current
- 后续 chunk：上一轮压缩后保留的旧 token 构成 memory region，新追加的构成 current region

### 记忆偏好

每个 `[layer, kv_head]` 的记忆偏好由 `ours_top100.csv` 中的 **memory-core internal attention score** 决定：

- **高分 head**：对长期记忆 token 关注度高 → 记忆偏好强 → memory_ratio 大
- **低分 head**：对当前 token 关注度高 → 记忆偏好弱 → memory_ratio 小

### 选择逻辑

对每个 `[layer, kv_head]`：

```
1. 计算 combined = attn_norm×(1−α) + recency×α          ← 和原始 HERMES 完全一样
2. 强制保留最近 W 个 token（window_size）                ← sparsemm 机制不变
3. 剩余预算拆分:
   memory_budget  = round((total − W) × memory_ratio)
   current_budget = (total − W) − memory_budget
4. 从 Memory Region  按 combined 取 top memory_budget  个  ← 新增
5. 从 Current Region 按 combined 取 top current_budget 个  ← 新增
6. 合并 = text_prefix + mem_selected + cur_selected + recent_window
```

不足补齐：某区域可用 token 少于其预算时，从另一区域 overflow fill。

---

## 参数说明

### `--kv_head_budget_memory_current_split`

| 属性 | 值 |
|------|-----|
| 类型 | bool |
| 默认值 | `False` |
| 说明 | 总开关。设为 `True` 启用记忆/当前预算拆分 |

### `--kv_head_budget_memory_scores`

| 属性 | 值 |
|------|-----|
| 类型 | str (path) |
| 默认值 | `None`（fallback 到 `--kv_head_budget_scores`） |
| 说明 | per-KV-head 记忆偏好分数 CSV 文件路径。格式：`layer,kv_head,score,...`（112 行 = 28 层 × 4 KV head）。推荐使用 `results/observations/memory_core_budget_scores/ours_top100.csv` |

分数列优先识别顺序：`internal_top100_mean_attention` > `score` > `spiky_readout_score` > 其他数值列。

### `--kv_head_budget_memory_ratio`

| 属性 | 值 |
|------|-----|
| 类型 | float |
| 默认值 | `0.7` |
| 范围 | `[0.0, 1.0]`（但内部会被 clip 到 `[0.1, 0.9]`） |
| 说明 | **全局基准记忆占比**。所有 112 个 KV head 的 memory_ratio 以此为中心浮动。`0.7` 表示默认 70% 预算给记忆 token，30% 给当前 token |

### `--kv_head_budget_split_dynamic_scale`

| 属性 | 值 |
|------|-----|
| 类型 | float |
| 默认值 | `0.3` |
| 说明 | **per-head 动态调节幅度**。每个 head 根据自身记忆偏好分数在基准线上下浮动的最大偏移量 |

#### 两个参数的关系

```python
# per-head memory_ratio 的计算公式
norm = (head_score - min_score) / (max_score - min_score)     # 全局归一化到 [0, 1]
memory_ratio = base_memory_ratio + dynamic_scale × (2×norm - 1)
memory_ratio = clip(memory_ratio, 0.1, 0.9)
```

以默认值 `base=0.7, scale=0.3` 为例：

| Head 类型 | score | norm | memory_ratio | 含义 |
|-----------|-------|------|:---:|------|
| 最低分 head | 0.0002 | 0.0 | **0.40** | 40% 记忆，60% 当前 |
| 中等分 head | 0.0023 | 0.5 | **0.70** | 70% 记忆，30% 当前 |
| 最高分 head | 0.0044 | 1.0 | **0.90**（clip） | 90% 记忆，10% 当前 |

实际数据中（`ours_top100.csv`），浅层的 head 通常在 0.4 左右（偏当前），深层的 head 通常在 0.8-0.9（偏记忆），这与直觉一致——浅层处理局部细节，深层处理全局语义。

---

## 使用示例

### Dense Union 路径

```bash
python video_qa/run_infer.py \
  --model qwen2.5_vl_7b \
  --dataset streamingbench \
  --num_chunks 1 \
  --sample_fps 0.5 \
  --kv_size 6000 \
  --compress_mode hermes \
  --kv_head_budget_scores results/myobservations/budget_reallocation_scores/reallocated_ours_top100.csv \
  --kv_head_budget_scheme sparsemm \
  --kv_head_budget_sparsemm_ratio 0.2 \
  --kv_head_budget_sparsemm_window_size 32 \
  --kv_head_budget_union_cap_ratio 1.0 \
  --kv_head_budget_memory_current_split \
  --kv_head_budget_memory_scores results/observations/memory_core_budget_scores/ours_top100.csv \
  --kv_head_budget_memory_ratio 0.7 \
  --kv_head_budget_split_dynamic_scale 0.3
```

### Ragged Prefill 路径

```bash
python video_qa/run_infer.py \
  --model qwen2.5_vl_7b \
  --dataset streamingbench \
  --num_chunks 1 \
  --sample_fps 0.5 \
  --kv_size 6000 \
  --compress_mode hermes \
  --kv_head_budget_scores results/myobservations/budget_reallocation_scores/reallocated_ours_top100.csv \
  --kv_head_budget_scheme sparsemm \
  --kv_head_budget_sparsemm_ratio 0.2 \
  --kv_head_budget_sparsemm_window_size 32 \
  --kv_head_ragged_prefill \
  --kv_head_budget_memory_current_split \
  --kv_head_budget_memory_scores results/observations/memory_core_budget_scores/ours_top100.csv \
  --kv_head_budget_memory_ratio 0.7 \
  --kv_head_budget_split_dynamic_scale 0.3
```

---

## 兼容性

### 向后兼容

新 flag **默认关闭**。不加 `--kv_head_budget_memory_current_split` 时，执行路径与修改前完全一致，不会进入任何 split 分支。

### 与 RoleKV 的关系

当 `--rolekv_ragged` 同时启用时，RoleKV 的 quota 机制优先。split 模式在非 RoleKV 的 ragged 和 dense 路径下独立工作。

### 后台运行进程

修改磁盘上的 Python 文件不会影响正在运行的进程。Python 在启动时将代码加载到内存，后台 `nohup` 进程不受文件修改影响。

---

## 代码架构

### 核心函数（`head_analysis/hermes_kv_head_budget.py`）

| 函数 | 作用 |
|------|------|
| `load_kv_head_memory_scores(csv_path)` | 从 CSV 加载 per-KV-head 记忆分数 `[28, 4]` |
| `compute_memory_current_split_ratios(kv_scores, base, scale)` | 将分数映射为 memory_ratio `[28, 4]` |
| `build_kv_head_split_budget_table(total, ratios)` | 将总预算拆分为 `[28, 4, 2]`（mem, cur） |
| `_select_split_budget_tokens(...)` | 从两个区域独立 topk 选择 |
| `apply_kv_head_budget(...)` | 总入口，安装 split config 和 pre/post lens 追踪 |

### 数据流

```
ours_top100.csv  ──→ load_kv_head_memory_scores()  ──→ [28,4] scores
        │
        └──→ compute_memory_current_split_ratios()  ──→ [28,4] memory_ratios
                │
budget_scores.csv ──→ build_kv_head_budget_table()  ──→ [28,4] total_budget
                │                       │
                └──→ build_kv_head_split_budget_table() ──→ [28,4,2]
                        │
                        ├── dense 路径:  kv_head_budget_prune()
                        └── ragged 路径: _select_ragged_keep_indices()
                                │
                                └──→ _select_split_budget_tokens()
                                      ├─ memory region:  top-k → mem_selected
                                      └─ current region: top-k → cur_selected
```

### 修改的文件

| 文件 | 改动 |
|------|------|
| `head_analysis/hermes_kv_head_budget.py` | 新增 4 个函数 + `apply_kv_head_budget` 新参数 + dense 路径 split 分支 |
| `head_analysis/hermes_kv_head_ragged_prefill.py` | pre/post lens 通用保存 + `_ragged_split_budget_table` + 选择函数 split 分支 |
| `video_qa/base.py` | 4 个新 CLI flag + wiring 传参 |
| `video_qa/run_infer.py` | 4 个新 CLI flag + save_dir suffix + subprocess cmd 构造 |

---

## 参数调优建议

| 场景 | `memory_ratio` | `dynamic_scale` | 效果 |
|------|:---:|:---:|------|
| 保守（接近原始行为） | 0.5 | 0.0 | 所有 head 均分记忆/当前预算 |
| 默认推荐 | 0.7 | 0.3 | 偏记忆，head 间有适度差异 |
| 激进记忆偏好 | 0.8 | 0.4 | 整体更偏记忆，差异更大 |
| 激进当前偏好 | 0.3 | 0.2 | 整体更偏最新 chunk |

- 如果视频较长、需要跨 chunk 记忆 → 增大 `memory_ratio`
- 如果视频变化快、当前帧更重要 → 减小 `memory_ratio`
- 如果发现不同任务最优的 head 差异大 → 增大 `dynamic_scale`
- 如果 head 差异引入噪声 → 减小 `dynamic_scale`
