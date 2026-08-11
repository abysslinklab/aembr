# AEMBR：耦合振荡器推理架构——系统架构与核心算法规格书 v3.0

**袭明 · 深渊实验室** | 2026-08-10 | 基线 v3.0

---

## 目录

1. 系统总览
2. 数据模型与拓扑定义
3. 蒸馏管道
4. 推理引擎（Kuramoto 动力学）
5. 解码算法
6. 稀疏实现规范
7. 参数表
8. 附录：最小可运行原型

---

## 1. 系统总览

```
                  ┌─────────────────────────────────┐
                  │         AEMBR 系统架构            │
                  └─────────────────────────────────┘
                                    
  知识源                         推理端                         
  ┌──────────┐   蒸馏    ┌──────────────┐   输入    ┌──────────┐
  │ 语料库   │─────────→│  J矩阵 (拓扑)  │←──────────│ 用户输入  │
  │ BERT模型 │   15s    │  N节点M边     │  相位编码  │          │
  └──────────┘          │  float32稀疏  │            └──────────┘
                        └──────┬───────┘                 │
                               │                          │
                        ┌──────▼───────┐         ┌───────▼──────┐
                        │ Kuramoto引擎  │────────→│   解码器     │
                        │ 迭代N步收敛   │  相图    │ 激活场→路径  │
                        │ O(E·steps)   │  快照    │ 词序列输出   │
                        └──────────────┘         └──────────────┘
```

**设计原则**：

- 推理 = 相位空间中的物理收敛，不使用矩阵乘法
- 知识 = 耦合拓扑 J 矩阵，不使用浮点权重
- 蒸馏 = 从已有模型提取语义结构，不从头训练

---

## 2. 数据模型与拓扑定义

### 2.1 节点（振荡器）

每个词汇 $w_i$ 对应一个相位振荡器：

$$\theta_i(t) \in [0, 2\pi), \quad i = 1, 2, ..., N$$

**固有频率** $\omega_i = 1.0 + \epsilon_i$，其中 $\epsilon_i \sim \mathcal{N}(0, 0.02^2)$。轻微频率差异避免全局锁死。

**输入编码**：当输入文本分词后命中词 $w_i$，设置 $\theta_i(0) = 0.9\pi$（最大扰动）。

### 2.2 耦合拓扑（J 矩阵）

$$J \in \mathbb{R}^{N \times N}, \quad J_{ij} \in [0, 1]$$

- $J_{ij} > 0$：词汇 $i$ 与词汇 $j$ 之间存在正向语义耦合
- $J_{ii} = 0$：无自环
- 稀疏度：$\rho \approx 1 - \frac{|E|}{N^2} \approx 95\%-99.5\%$
- 平均度：$\langle k \rangle \approx 25$

**语义含义**：$J_{ij}$ 的值表示两个概念在语义空间中的约束强度。高耦合意味着两个词在推理中倾向于一起浮现。

### 2.3 约束势函数

$$U(\Theta) = \frac{1}{2} \sum_{(i,j): J_{ij}>0} J_{ij} \left(1 - \cos(\theta_j - \theta_i)\right)$$

物理含义：系统偏离拓扑约束平衡的程度。当所有强耦合邻居的相位完全对齐时 $(\Delta\theta = 0)$，$U = 0$。耦合越不满足，$U$ 越大。

### 2.4 数据存储格式

**稠密表示（原型）**：
```python
J: np.ndarray  # shape (N, N), dtype=float32
vocab: List[str]  # 长度 N
```

**稀疏表示（生产）**：
```python
edges: List[Tuple[int, int, float]]  # [(i, j, weight), ...]
# 对称边只存一次：i < j
# 内存：edges × 12 bytes
```

---

## 3. 蒸馏管道

### 3.1 概述

从 BERT-base-chinese 提取语义拓扑，转换为 Kuramoto 可用的 J 矩阵。蒸馏不是训练——是一次性权重提取和格式转换。

### 3.2 步骤

#### Step 1：词汇表构建

```
输入：目标领域语料库 → jieba 分词
输出：前 N 个高频词的词表 vocab[N]
```

**过滤规则**：仅保留长度 ≥ 2 的中文词。过滤标点、数字、英文。

#### Step 2：BERT 上下文嵌入

```
对每个词 w_i：
  text = "这是一个{w_i}。"
  embedding_i = BERT(text)[CLS]  # 取[CLS] token的768维输出
```

使用 `bert-base-chinese`，冻结权重，无梯度。输出 shape：(N, 768)。

#### Step 3：余弦相似度 → J 矩阵

```
S_{ij} = cos_sim(emb_i, emb_j)  for all i, j

J_{ij} = {
  S_{ij},  if S_{ij} > 0.3 and j ∈ topK(S_i)
  0,      otherwise
}
```

**topK = 25**（每节点保留最强 25 条边）。阈值 0.3 过滤弱关联噪声。

#### Step 4：对称化

J 矩阵自然对称（余弦相似度是对称的），无需额外操作。

### 3.3 复杂度

| 步骤 | 复杂度 | 500节点耗时 |
|---|---|---|
| 词汇构建 | O(M) | ~2s |
| BERT嵌入 | N×O(1) | ~10s |
| 余弦相似度 | O(N²d) | ~0.5s |
| 稀疏化 | O(N²) | ~0.5s |
| **总计** | — | **~15s** |

---

## 4. 推理引擎（Kuramoto 动力学）

### 4.1 核心方程

**广义 Kuramoto 模型**：

$$\frac{d\theta_i}{dt} = \omega_i + K \sum_{j: J_{ij}>0} J_{ij} \cdot \sin(\theta_j - \theta_i)$$

**离散化（Euler方法）**：

$$\theta_i(t + \Delta t) = \left( \theta_i(t) + \Delta t \cdot \left[ \omega_i + K \sum_{j: J_{ij}>0} J_{ij} \cdot \sin(\theta_j(t) - \theta_i(t)) \right] \right) \bmod 2\pi$$

### 4.2 稠密实现（Python/NumPy）

```python
def kuramoto_dense(phase, omega, J, K=0.5, dt=0.03):
    """向量化·O(N²)每步"""
    phase_diff = phase[np.newaxis, :] - phase[:, np.newaxis]  # (N,N)
    sin_diff = np.sin(phase_diff)
    coupling = (J * sin_diff).sum(axis=1)  # (N,)
    dphi = omega + K * coupling
    return (phase + dphi * dt) % (2 * np.pi)
```

**适用**：原型验证，N ≤ 1000。

### 4.3 稀疏实现（Python边表）

```python
def kuramoto_sparse(phase, omega, edges, K=0.5, dt=0.03):
    """边表迭代·O(E)每步"""
    dphi = omega.copy()
    for i, j, w in edges:
        s = w * sin(phase[j] - phase[i])
        dphi[i] += K * s
        dphi[j] -= K * s  # 对称耦合
    return (phase + dphi * dt) % (2 * np.pi)
```

**适用**：生产环境，N ≥ 1000。

### 4.4 收敛判定

```
收敛条件：连续5步的势能变化 < tol

tol = 0.01 (经验值·500节点)
      0.05 (经验值·5000节点)
```

### 4.5 迭代步数

| 节点 | 步数 | 经验收敛步数 |
|---|---|---|
| 100-200 | 80-200 | 全收敛 |
| 500 | 130 | 全收敛 (K=0.5) |
| 1000 | 130-150 | 全收敛 |
| 5000 | 150 | 99%能量跌幅 |

**参数敏感性**：K 过大（>0.8）导致过同步，失去输入特异性。K 过小（<0.2）导致收敛过慢。

---

## 5. 解码算法

### 5.1 概述

从 Kuramoto 收敛后的相位快照 $\Theta^*$ 中，识别与输入相关的语义信息。解码分三个阶段：

### 5.2 阶段1：激活场传播

```
输入：
  phase: 收敛后的相位向量[N]
  J: 耦合矩阵
  pert_indices: 输入词节点索引

算法（BFS扩散）：
  activation = zeros(N)
  for each pert_idx in pert_indices:
      activation[pert_idx] = 1.0
      queue = [(pert_idx, 1.0, depth=0)]
      while queue:
          node, val, depth = pop(queue)
          if depth >= 4: continue
          for each neighbor j where J[node,j] > 0.01:
              if j not visited:
                  phase_align = cos(phase[j] - phase[node])
                  if phase_align > 0.2:
                      new_val = val * J[node,j] * phase_align * 0.6
                      activation[j] = max(activation[j], new_val)
                      queue.push(j, new_val, depth+1)

输出：
  activation: 每个非输入节点的激活值[N]
```

**衰减因子 0.6**：模拟多跳语义传播的自然衰减。

### 5.3 阶段2：共振簇提取

```
1. 候选：activation > 0.3 且非输入词
2. 贪婪聚类：相位对齐(cos>0.6) + 拓扑耦合(J>0.02)
3. 按簇内总激活度排序
```

### 5.4 阶段3：词序排列（贪心TSP）

```
给定簇内的词列表：
  start = 耦合度最高的词
  while 还有剩余词：
    next = 耦合最强 × 相位最对齐 的邻居
    加入到序列中
  输出有序词序列
```

**物理直觉**：这是一个在语义子图上找"最自然阅读顺序"的贪心路径搜索。词之间的耦合强度决定了它们在句子中的自然顺序。

### 5.5 阶段4：句子组装

基于词性标注构造中文语法片段。

### 5.6 解码器 v3：共振减法（🆕 v3.0·推荐）

**问题**：BFS 激活场（阶段 1-4）在 Kuramoto 全局同步后失去输入特异性——所有输入坍缩到相同的通用高频词输出。

**解法**：输出词得分 = 与输入词的共振 - 与通用词的共振。

```
通用词集合 G = {方法, 问题, 研究, 发现, 本文, 核心, 一个, 如何, 可以, 通过}

for each 候选词 i:
    score_i = mean(cos(phase[i]-phase[p]) × J[i,p] for p in pert_indices)
            - subtract × mean(cos(phase[i]-phase[g]) × J[i,g] for g in G_indices)
    if score_i < 0: score_i = 0

output = topK(scores)
```

**参数**：
- `subtract` = 0.4-0.6（减法权重。越低→更多输出但更多噪声；越高→更干净但可能丢信息）
- 推荐默认值 = 0.5

**物理直觉**：共振减法 = 信号（输入专属）- 噪声（通用模式）。通用词"方法"与几乎所有输入词都有高耦合，减掉这个基线后剩下的是输入专属的语义结构。

**伪代码**：

```python
def decode_v3(phase, J, pert_indices, subtract=0.5):
    N = len(phase)
    generic_idx = [i for i in idx if vocab[i] in GENERIC_SET]
    
    scores = np.zeros(N)
    for i in range(N):
        if i in pert_indices: continue
        
        # 输入共振
        input_align = np.mean([
            np.cos(phase[i] - phase[p]) * J[i, p] 
            for p in pert_indices
        ])
        
        # 通用共振 (噪声基线)
        generic_align = np.mean([
            np.cos(phase[i] - phase[g]) * J[i, g] 
            if J[i, g] > 0 else 0
            for g in generic_idx
        ])
        
        scores[i] = max(0, input_align - subtract * generic_align)
    
    top = np.argsort(-scores)[:topk]
    return [(vocab[i], scores[i]) for i in top if scores[i] > 0.001]
```

**验证数据**：800 节点·40 Kuramoto 步·减权 0.5。4/6 测试用例产出有区分度的语义输出（v2.0 旧解码器 0/6）。

---

## 6. 稀疏实现规范（生产级 C/Rust 参考）

### 6.1 数据结构

```c
// 边表——只存实际存在的耦合
typedef struct {
    uint32_t i;       // 源节点
    uint32_t j;       // 目标节点
    float    weight;  // J[i,j] ∈ [0,1]
} edge_t;

// 引擎状态
typedef struct {
    uint32_t  n_nodes;     // 节点数 N
    uint32_t  n_edges;     // 边数 E
    edge_t*   edges;       // 边表 [E]
    float*    phase;       // 当前相位 [N]
    float*    omega;       // 固有频率 [N]
    float     K;           // 全局耦合强度
    float     dt;          // 积分步长
} aembr_t;
```

### 6.2 核心循环

```c
void aembr_step(aembr_t* eng) {
    // 累积相位导数
    float dphi[eng->n_nodes];
    memcpy(dphi, eng->omega, eng->n_nodes * sizeof(float));
    
    // 遍历所有边
    for (int e = 0; e < eng->n_edges; e++) {
        uint32_t i = eng->edges[e].i;
        uint32_t j = eng->edges[e].j;
        float w = eng->edges[e].weight;
        
        float s = w * sinf(eng->phase[j] - eng->phase[i]);
        dphi[i] += eng->K * s;
        dphi[j] -= eng->K * s;
    }
    
    // Euler积分
    for (int n = 0; n < eng->n_nodes; n++) {
        eng->phase[n] = fmodf(
            eng->phase[n] + eng->dt * dphi[n], 
            2.0f * M_PI
        );
    }
}
```

### 6.3 内存计算

```
总内存 = 边表(edges × 12B) + 状态(N × 8B + N × 4B)

15K节点·2.3M边：2.3M×12 + 15K×12 = 27.8MB
```

### 6.4 复杂度

| 操作 | 复杂度 | 15K节点 |
|---|---|---|
| 单步 | O(E) | 2.3M FLOPs |
| 150步推理 | O(E·steps) | 345M FLOPs |
| 预估耗时 | — | <2ms（单核·SIMD） |

---

## 7. 参数表

| 参数 | 符号 | 默认值 | 含义 | 可调范围 |
|---|---|---|---|---|
| 节点数 | N | 500-1000 | 词汇表大小 | 100-15000 |
| 每节点边数 | k | 25 | 拓扑稀疏度 | 10-50 |
| 全局耦合 | K | 0.5 | 同步强度 | 0.1-0.8 |
| 积分步长 | dt | 0.03 | 时间离散精度 | 0.01-0.05 |
| 频率方差 | σ_ω | 0.02 | 固有频率分布 | 0-0.1 |
| 收敛容差 | tol | 0.01 | 能量变化阈值 | 0.001-0.1 |
| 最大步数 | steps | 150 | 安全上限 | 50-300 |
| 扩散深度 | depth | 4 | BFS最大跳数 | 2-6 |
| 扩散衰减 | α | 0.6 | 激活传播衰减 | 0.3-0.9 |
| 相位对齐阈值 | φ_min | 0.2 | 激活传播条件 | 0.1-0.5 |
| 余弦相似度阈值 | S_min | 0.3 | J矩阵过滤 | 0.1-0.5 |

---

## 8. 附录：最小可运行原型

```python
# aembr_minimal.py — 可独立运行的最小原型 (~100行)
import numpy as np, json
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity

# 1. 蒸馏
model = AutoModel.from_pretrained("bert-base-chinese")
tok = AutoTokenizer.from_pretrained("bert-base-chinese")
vocab = ["方法","问题","研究","发现","结构","模型","系统","数据","计算","算法"]  # 示例

emb = np.array([model(**tok(f"这是一个{w}。", return_tensors="pt")).last_hidden_state[0,0].detach().numpy() for w in vocab])
J = cosine_similarity(emb)
np.fill_diagonal(J, 0)
for i in range(len(vocab)):
    top = np.argsort(-J[i])[:5]
    J[i, [j for j in range(len(vocab)) if j not in top]] = 0

# 2. 推理
omega = np.ones(len(vocab))
phase = np.zeros(len(vocab))
text = "研究 方法"
for w in text.split():
    if w in vocab: phase[vocab.index(w)] = np.pi * 0.9

for _ in range(100):
    pd = phase[None,:] - phase[:,None]
    phase = (phase + 0.03*(omega + 0.5*(J*np.sin(pd)).sum(axis=1))) % (2*np.pi)

# 3. 解码
act = np.zeros(len(vocab))
for pi in [i for i,p in enumerate(phase) if abs(p-np.pi*0.9)<0.01]:
    act[pi] = 1.0
    q = [(pi, 1.0, 0)]
    while q:
        n,v,d = q.pop(0)
        if d>=3: continue
        for j in range(len(vocab)):
            if J[n,j]>0 and j!=pi:
                al=np.cos(phase[j]-phase[n])
                if al>0.3: act[j]=max(act[j],v*J[n,j]*al*0.6); q.append((j,act[j],d+1))

result = [(vocab[i], act[i]) for i in range(len(vocab)) if act[i]>0]
result.sort(key=lambda x:-x[1])
print(" ".join(f"{w}({s:.2f})" for w,s in result[:5]))  # 输出Top5激活词
```

---

*本文档为 AEMBR 系统架构与核心算法的唯一技术规格来源。*
