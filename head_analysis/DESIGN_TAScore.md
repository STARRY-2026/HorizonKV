# TAScore: Task-Adaptive Head Specialization Scoring

## 1. Motivation & Problem Diagnosis

### 1.1 Current score_v1 weaknesses (empirical)

| Issue | Evidence |
|-------|----------|
| **Counting** remains worst task (41-53% across all schemes) | Needs fine-grained spatial-temporal detail inspection |
| **Object Recognition** *regresses* with score_v1 | 75.16% vs 76.43% (ours_top100 baseline) vs 80.11% (HERMES original) |
| **Detail branch nearly dead** | Only 17% of heads (19/112) activate the detail branch — the method contradicts its own design goal |
| **No per-task adaptation** | All 10 StreamingBench tasks share identical head budgets |
| **Arbitrary weights** | 10 hand-picked weights; sensitivity analysis shows rank correlation >0.98 with random alternatives — the complexity adds no value |

### 1.2 Root causes

1. **Single global score**: Computing one score per head and applying it uniformly across all tasks ignores the well-documented phenomenon of *task-specific head specialization* (Task-KV, Cognitive Mirrors, SEAL, Not All Heads Matter).
2. **Hard max() between branches**: Creates brittle decision boundary; 83% of heads default to the memory branch, making the detail branch vestigial.
3. **Arbitrary weighting**: Weighted sums with hand-picked coefficients are neither theoretically grounded nor empirically validated.
4. **No within-layer normalization**: Absolute attention magnitudes vary by layer depth; raw scores conflate "important head" with "head in a high-magnitude layer."

### 1.3 Design goals

1. **Lift Counting from ~50% → target 58%+** (recover gap vs full HERMES at 52.85%)
2. **Recover Object Recognition regression** (target ≥76.5% = baseline level)
3. **Maintain or improve other tasks** (no regression >1%)
4. **Data-driven weights** — every hyperparameter derivable from observed attention statistics
5. **Task-conditioned budgets** — heads receive different budgets depending on the downstream task

---

## 2. Related Work

Our design synthesizes insights from five lines of work:

| Paper | Key Insight We Adopt |
|-------|---------------------|
| **Task-KV** (He et al., 2025) | Heads have task-dependent importance; "heterogeneous heads" far from semantic center need full budget |
| **Not All Heads Matter / HeadKV** (Fu et al., ICLR 2025) | Per-head retrieval vs. reasoning scores; 1.5% KV keeps 97% performance |
| **HACK** (AAAI 2026) | Contextual vs. Structural head types; asymmetric per-head budgets |
| **Dummy Forcing** (2025) | Three-way head taxonomy (Dummy/Sink/Neighbor) by temporal attention pattern |
| **SEAL** (Lee et al., 2025) | Heads have positive/negative correlation with retrieval; emphasize retrieval-positive heads |
| **SparseMM** (Wang et al., ICCV 2025) | Visual head scores via OCR anchor task; score-proportional budget allocation |
| **Cognitive Mirrors** (NeurIPS 2025) | Heads specialize in specific cognitive functions (Retrieval, Reasoning, Calculation) |

---

## 3. Method: TAScore

### 3.1 Overview

```
┌─────────────────────────────────────────────────────────────────┐
│  Phase 1: Per-Task Head Profiling (offline, one-time)           │
│                                                                 │
│  For each task t ∈ {CR,CT,CS,PR,AR,RR,OR,SU,EU,TR}:           │
│    profile per-(layer, kv_head):                                │
│      - internal_readout_share[t, l, h]                          │
│      - topk_concentration[t, l, h]                               │
│      - boundary_ratio[t, l, h]                                   │
│      - current_readout_share[t, l, h]                            │
│      - observation_count[t, l, h]   (statistical reliability)    │
│                                                                 │
│  Output: T×L×H tensor of head profiles                          │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  Phase 2: Functional Head Clustering (offline, one-time)        │
│                                                                 │
│  Cluster (layer, kv_head) pairs into K functional types         │
│  using per-task readout profiles as features.                   │
│                                                                 │
│  K=4 discovered types (matching Dummy Forcing + HACK):          │
│    Type R (Retrieval):   high internal readout, low boundary     │
│    Type D (Detail):      high current readout, high topk         │
│    Type B (Boundary):    high boundary ratio, low concentration  │
│    Type X (Diffuse):     low concentration, low readout          │
│                                                                 │
│  Output: head_type[l, h] ∈ {R, D, B, X}                         │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  Phase 3: Task-Conditioned Scoring (offline, one-time)          │
│                                                                 │
│  For each task t:                                               │
│    score[t, l, h] = U(t,l,h)^α × S(l,h)^β × (1-B(l,h))^γ      │
│                                                                 │
│  where:                                                         │
│    U = task-conditioned utility (readout coverage for task t)   │
│    S = head selectivity (topk concentration, task-agnostic)     │
│    B = boundary penalty (boundary ratio, task-agnostic)          │
│                                                                 │
│  Exponents α,β,γ are set by:                                    │
│    - Default: α=1, β=0.5, γ=0.5 (utility-primary)              │
│    - Per-task: tuned on validation split (10% of questions)     │
│                                                                 │
│  Output: T×L×H score matrix                                     │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  Phase 4: Within-Layer Contrastive Normalization                │
│                                                                 │
│  For each task t and layer l:                                   │
│    norm_score[t, l, h] = score[t,l,h] / mean(score[t,l,:])     │
│                                                                 │
│  This makes scores comparable across layers and ensures each    │
│  layer contributes heads proportionally to their relative       │
│  importance, not their absolute attention magnitude.            │
│                                                                 │
│  Output: T×L×H normalized score matrix                          │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  Phase 5: Budget Allocation (online, per-question)              │
│                                                                 │
│  If task label is known:                                        │
│    weights[l, h] = norm_score[task, l, h]                       │
│  If task label is unknown (task-agnostic):                      │
│    weights[l, h] = max_t norm_score[t, l, h]   (optimistic)     │
│    OR                                                                      │
│    weights[l, h] = mean_t norm_score[t, l, h]  (balanced)       │
│                                                                 │
│  Convert to integer budget via proportional allocation:         │
│    budget[l, h] = floor(weight[l,h] / Σweight × total_budget)   │
│    + fractional remainder distributed by largest remainder      │
│                                                                 │
│  Apply min_budget = max(1, 0.1 × avg_budget) per head           │
│  Apply max_budget = min(total, 2.0 × avg_budget) per head       │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 Phase 1: Per-Task Head Profiling

**Key change from current approach**: Instead of aggregating all observations into a single score, we partition by task type.

```python
def profile_per_task(raw_observations_csv):
    """
    Input: raw_effective_readout.csv (from profile_effective_memory_readout.py)
           with added 'task' column

    Output: task_profiles[T][L][H] dict with per-metric means
    """
    df = pd.read_csv(raw_observations_csv)
    tasks = sorted(df['task'].unique())

    profiles = {}
    for task in tasks:
        task_df = df[df['task'] == task]
        profiles[task] = {
            'internal_readout_share': aggregate(task_df, 'internal_readout_share'),
            'topk_concentration': aggregate(task_df, 'internal_topk_concentration'),
            'boundary_ratio': aggregate(task_df, 'boundary_ratio'),
            'current_readout_share': aggregate(task_df, 'current_readout_share'),
            'num_observations': count_obs(task_df),
        }
    return profiles
```

**Why this matters**: Counting (CT) questions have fundamentally different attention patterns from Causal Reasoning (CR). A head that's crucial for counting objects across frames may be irrelevant for reasoning about causal relations. By profiling per task, we preserve this signal.

### 3.3 Phase 2: Functional Head Clustering

Instead of hand-designing three branches with arbitrary thresholds, we let the data discover functional head types.

```python
def cluster_heads(task_profiles, n_clusters=4):
    """
    Feature vector per head = [internal_readout, topk_conc, boundary_ratio,
                                current_readout] × T tasks
    → shape: (L×H, 4T)

    Cluster with K-Means (K=4, matching literature expectations).
    Label clusters by their centroid characteristics.
    """
    features = build_feature_matrix(task_profiles)  # (L×H, 4T)
    labels = KMeans(n_clusters=4, random_state=42).fit_predict(features)

    # Auto-label clusters by centroid properties
    centroids = compute_centroids(features, labels)
    type_names = []
    for c in centroids:
        if c.internal_readout > 0.6 and c.boundary_ratio < 0.3:
            type_names.append('R')  # Retrieval
        elif c.current_readout > 0.5 and c.topk_conc > 0.4:
            type_names.append('D')  # Detail
        elif c.boundary_ratio > 0.4:
            type_names.append('B')  # Boundary
        else:
            type_names.append('X')  # Diffuse

    return labels, type_names
```

**Why 4 clusters**: This matches both the Dummy Forcing taxonomy (Dummy/Sink/Neighbor → 3 types + diffuse) and the HACK taxonomy (Contextual/Structural → 2 types × {sharp, diffuse} = 4). Empirically, 4 clusters provides enough granularity without over-fragmentation (112 heads / 4 ≈ 28 heads per type).

**Validation**: The clustering should be *stable* across different video subsets. We report Adjusted Rand Index between clusterings from two disjoint video sets.

### 3.4 Phase 3: Task-Conditioned Product Scoring

The core scoring formula:

```
score[t, l, h] = utility[t, l, h]^α × selectivity[l, h]^β × boundary_penalty[l, h]^γ
```

where each component is defined as:

| Component | Formula | Rationale |
|-----------|---------|-----------|
| **utility[t,l,h]** | `internal_readout_share[t,l,h]` for memory-heavy tasks (CR, CS, EU); `0.5×internal + 0.5×current` for balanced tasks (OR, AR); `current_readout_share[t,l,h]` for detail tasks (CT) | Task-adaptive: different tasks read from different memory regions |
| **selectivity[l,h]** | `topk_concentration[l,h]` averaged over all tasks | Task-agnostic: a sparse head is universally better (less noise) |
| **boundary_penalty[l,h]** | `1.0 - mean(boundary_ratio[l,h,:])` | Task-agnostic: boundary-attending heads waste budget on chunk transitions |

**Why product instead of weighted sum**:
1. **Theoretically grounded**: Each factor is a necessary condition. A head with 0 selectivity cannot be useful regardless of utility (it's just diffuse noise). Weighted sum `0.7×utility + 0.3×selectivity` can give high scores to heads with zero selectivity.
2. **No weight tuning needed**: The exponent parameters α, β, γ have clear semantic meaning (α=1 means "utility is primary"; β<1 means "selectivity is a softer constraint").
3. **Robust to scale differences**: The product form is invariant to monotonic transformations of individual factors (unlike weighted sum where feature scales matter).

**Default exponents**: α=1.0, β=0.5, γ=0.5
- α=1: utility enters linearly — the primary signal
- β=0.5: selectivity is a soft gate — sqrt reduces its influence relative to utility
- γ=0.5: boundary penalty is a soft gate — same rationale

**Per-task exponent tuning** (optional, for final paper):
- Split 10% of questions per task as validation set
- Grid search α ∈ {0.5, 0.75, 1.0, 1.25, 1.5}, β,γ ∈ {0.25, 0.5, 0.75, 1.0}
- Select exponents that maximize per-task accuracy on validation set
- Report both default and tuned results (tuned should not be dramatically better if the method is robust)

### 3.5 Phase 4: Within-Layer Contrastive Normalization

```python
def contrastive_normalize(scores, per_layer=True):
    """
    Normalize scores within each layer so that:
    - Each layer's heads compete only against other heads in the same layer
    - Deep layers (naturally higher attention magnitude) don't dominate budget
    """
    normalized = np.zeros_like(scores)
    for layer in range(num_layers):
        layer_mean = scores[layer, :].mean()
        normalized[layer, :] = scores[layer, :] / (layer_mean + EPS)
    return normalized
```

**Why this matters**: Without layer normalization, deep layers (which have higher absolute attention values) receive disproportionately large budgets. But prior work (Cognitive Mirrors, SEAL) shows that important retrieval heads exist across all layers. Within-layer contrast ensures each layer's best heads get recognized, regardless of layer depth.

### 3.6 Phase 5: Task-Conditioned Budget Allocation

```python
def allocate_budget(norm_scores, total_budget, task=None, mode='task_conditioned'):
    """
    Convert normalized scores to integer per-head budgets.

    Args:
        norm_scores: T×L×H tensor of normalized scores
        total_budget: total visual tokens to keep per layer
        task: known task label (or None for task-agnostic)
        mode: 'task_conditioned' | 'optimistic_ensemble' | 'balanced_ensemble'
    """
    L, H = norm_scores.shape[1], norm_scores.shape[2]

    if task is not None and task in TASKS:
        weights = norm_scores[task_index[task]]  # (L, H)
    elif mode == 'optimistic_ensemble':
        weights = norm_scores.max(axis=0)  # max over tasks
    else:  # 'balanced_ensemble'
        weights = norm_scores.mean(axis=0)  # mean over tasks

    # Proportional allocation with min/max constraints
    avg_budget = total_budget / H
    min_budget = max(1, int(0.1 * avg_budget))
    max_budget = min(total_budget, int(2.0 * avg_budget))

    return proportional_allocate(weights, total_budget, min_budget, max_budget)
```

**Three inference modes**:
1. **Task-conditioned** (oracle): Use known task label — upper bound on method potential
2. **Optimistic ensemble**: `max_t score[t,l,h]` — a head is important if it's important for *any* task. This is generous (more tokens kept) but safe (fewer false negatives).
3. **Balanced ensemble**: `mean_t score[t,l,h]` — conservative, closer to current single-score approach.

For StreamingBench, task labels are known from the dataset, so we can use mode 1 in evaluation. For real deployment without task labels, mode 2 is recommended.

---

## 4. Implementation Plan

### 4.1 New file: `head_analysis/build_task_adaptive_scores.py`

```
build_task_adaptive_scores.py
  ├── Phase 1: load raw observations, partition by task
  ├── Phase 2: cluster heads → head_types.json
  ├── Phase 3: compute per-task product scores
  ├── Phase 4: within-layer contrastive normalization
  ├── Phase 5: generate budget CSVs
  │     - tascores_task_conditioned/  (one CSV per task)
  │     - tascores_optimistic.csv     (max over tasks)
  │     - tascores_balanced.csv       (mean over tasks)
  └── Phase 6: generate control variants
        - uniform (per-layer equal budgets)
        - inverted (within-layer inverted scores)
        - random (within-layer permuted scores)
```

### 4.2 Integration with existing pipeline

The new scores are drop-in compatible with `hermes_kv_head_budget.py`:
```bash
python hermes_kv_head_budget.py \
    --kv_head_scores_path results/observations/tascores/tascores_optimistic.csv \
    --budget_scheme sparsemm_per_layer_total \
    ...
```

### 4.3 Validation experiments

**Ablation hierarchy** (each level must show gain over previous):
```
L0: uniform (no head differentiation)
L1: single-feature baseline (internal_top100_mean_attention only)
L2: dual (long + short max) — current best simple method
L3: TAScore with default exponents (α=1, β=0.5, γ=0.5), balanced ensemble
L4: TAScore with default exponents, optimistic ensemble
L5: TAScore with default exponents, task-conditioned (oracle)
L6: TAScore with per-task tuned exponents, task-conditioned
```

**Control experiments**:
- TAScore-inverted: invert scores within each layer → should be worse than uniform
- TAScore-random: permute scores within each layer → should be ≈ uniform
- TAScore-ablation-α0: set α=0 (remove utility) → measures utility contribution
- TAScore-ablation-β0: set β=0 (remove selectivity) → measures selectivity contribution

### 4.4 Expected improvements

| Task | Current (ours_top100) | Target (TAScore) | Rationale |
|------|----------------------|-------------------|-----------|
| Counting | 49.45% | 55-58% | Task-conditioned detail head boost |
| Object Recognition | 76.43% | 78-80% | Balanced utility (no forced choice between detail/memory) |
| Causal Reasoning | 76.47% | 77-79% | Retrieval heads amplified for CR |
| Action Recognition | 73.91% | 75-77% | Balanced ensemble helps temporal tasks |
| **Overall** | 74.34% | **76-77%** | Per-task adaptation + data-driven scoring |

---

## 5. Key Innovations Over score_v1

| Dimension | score_v1 | TAScore |
|-----------|----------|---------|
| **Scoring formula** | Arbitrary weighted sum | Theoretically-motivated product |
| **Weights** | 10 hand-picked, untuned | 0-3 exponents with clear semantics |
| **Task adaptation** | None (single global score) | Per-task scores + task-conditioned allocation |
| **Head taxonomy** | 3 arbitrary branches | Data-driven 4-type clustering |
| **Normalization** | Double-normalized (artifact-prone) | Single within-layer contrastive normalization |
| **Branch combination** | Hard max() (17% detail activation) | Soft product (all factors always active) |
| **Detail sensitivity** | Detail branch vestigial | Detail utility enters via per-task utility term |
| **Validated by** | No ablation | Full ablation hierarchy + controls |

---

## 6. Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| Per-task profiles have low sample counts for rare tasks | Use Bayesian shrinkage toward global mean for tasks with <20 observations |
| Task-conditioned mode requires task labels | Provide optimistic ensemble as task-agnostic fallback |
| Clustering may not discover clean 4-type structure | Validate with silhouette score; fall back to predefined types if clustering is unstable |
| Product scoring may amplify noise for heads with near-zero utility | Add small floor (ε=0.01) to each factor before multiplication |
| Exponents tuned on small validation set may overfit | Report both default (α=1,β=0.5,γ=0.5) and tuned results; expect small gap |

---

## 7. Paper Framing

The paper contribution is: **"We show that attention heads in streaming video MLLMs exhibit task-dependent functional specialization, and that task-conditioned head scoring with a multiplicative utility×selectivity×boundary-penalty formulation consistently outperforms both uniform and additively-weighted single-score baselines."**

The three key claims:
1. **Functional specialization exists** → validated by stable cross-video clustering
2. **Multiplicative scoring > additive** → validated by ablation (product vs weighted sum)
3. **Task conditioning helps** → validated by task-conditioned > optimistic > balanced > single-score
