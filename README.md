# AEMBR — 耦合振荡器推理引擎

**蒸馏替代训练 · 相位同步替代矩阵乘法**

[![公共领域](https://img.shields.io/badge/license-NonCommercial%20%7C%20Public%20Research-brightgreen)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10+-blue)](aembr_demo.py)
[![Topics](https://img.shields.io/badge/topics-Kuramoto%20%7C%20Edge%20AI%20%7C%20Transformer%20Alternative-orange)](#)

> A Coupled Oscillator Inference Engine — replacing Transformer attention with phase synchronization.
>
> BERT distill → J-matrix topology → Kuramoto convergence → semantic reasoning.
>
> **No training. No GPU. No backprop. Phone CPU only.**

---

## Why This Exists

| | Transformer | AEMBR |
|---|---|---|
| **Inference** | Matrix multiply O(N²) | Phase sync O(E) |
| **Memory** (semantic layer) | ~10 GB (embedding) | ~30 MB (topology) |
| **FLOPs** | 18 Trillion | 340 Million |
| **Training** | GPU cluster · $2-5M | Distill · 15 sec · CPU |
| **Deployment** | Datacenter | Phone · MCU · Any CPU |

> AEMBR captures semantic similarity at 1/300 the memory of a Transformer's embedding layer.
> Full-scale inference at 15K nodes requires ~30 MB — practical for any CPU.
> AEMBR does NOT replicate a Transformer's full 140 GB of weights (syntax, fact memory, multi-lingual). It focuses on the semantic coupling layer.

Knowledge already exists in language structure. You don't need backpropagation to extract it — just distill the topology and let physics converge.

---

## Quick Start — 30 Seconds to Results

```bash
pip install numpy jieba torch transformers scikit-learn
python aembr_demo.py
```

Output:
```
分布式系统共识协议 → 协同(0.35) 集群(0.33) 算法(0.31) 规则(0.29)
密码学数据安全     → 加密(0.42) 安全(0.38) 隐私(0.35) 保护(0.31)
神经网络深度学习   → 训练(0.39) 学习(0.36) 网络(0.33) 模型(0.31)
区块链去中心化     → 节点(0.41) 共识(0.37) 去中心(0.34) 验证(0.30)
量子计算信息       → 量子(0.44) 比特(0.39) 粒子(0.35) 测量(0.32)
人工智能机器人     → 智能(0.42) 学习(0.38) 控制(0.34) 感知(0.31)
```

---

## How It Works

### Core Equation — Generalized Kuramoto Model

$$\frac{d\theta_i}{dt} = \omega_i + K \sum_{j: J_{ij}>0} J_{ij} \cdot \sin(\theta_j - \theta_i)$$

- $\theta_i$ : phase of word $i$
- $J_{ij}$ : semantic coupling strength between words $i$ and $j$
- Convergence in ~50-150 steps → stable phase map → decode to semantic output

### Pipeline

```
语料库 → jieba分词 → 词汇表(V)
    ↓
BERT嵌入(768维) → 余弦相似度 → J矩阵(稀疏·top-25)
    ↓
输入文本 → 相位扰动 → Kuramoto收敛(50-150步) → 共振减法解码
    ↓
语义输出(激活词)
```

### Architecture Diagram

```
┌──────────────┐    Distill     ┌──────────────┐    Input      ┌──────────────┐
│  Knowledge   │───────────────→│  J-Matrix    │←──────────────│  User Query   │
│  (BERT)      │    15 sec      │  (Topology)   │  Phase Encode │              │
└──────────────┘                └──────┬───────┘               └──────────────┘
                                      │
                               ┌──────▼───────┐        ┌──────▼───────┐
                               │  Kuramoto     │───────→│   Decoder     │
                               │  Convergence  │ Phase  │   Resonance   │
                               │  O(E·steps)   │  Map   │   Subtraction │
                               └──────────────┘        └──────────────┘
```

---

## Scaling

| Nodes | Edges | Memory (sparse) | Python | C (est.) | Hardware |
|---|---|---|---|---|---|
| 200 | 5,000 | 60 KB | 0.5s | 1ms | MCU |
| 500 | 12,500 | 150 KB | 2.8s | 6ms | MCU |
| 1,000 | 25,000 | 300 KB | 6.3s | 13ms | Phone |
| 5,000 | 118K | 624 KB | 14.4s | 29ms | Phone |
| 15,000 | 2.3M | **30 MB** | — | **<10ms** | Phone |

---

## Repository Structure

| File | Description |
|---|---|
| `aembr_demo.py` | **30-second demo**: distill → inference → output |
| `AEMBR_降临的物理学.md` | Paper: Arrival physics & Kuramoto semantic resonance |
| `AEMBR_架构与算法规格书_v1.0.md` | Architecture spec: math, algorithms, C reference |
| `AEMBR_验证与展开报告_v3.0.md` | Full validation: convergence, quality, scaling |

---

## Related Work

| Work | Approach | Difference from AEMBR |
|---|---|---|
| **AKOrN** (ICLR 2025 Oral, Miyato et al.) | Kuramoto neurons embedded in standard NN | Still requires training · GPU |
| **Kuramoto Attention** (Nunley, 2026) | Kuramoto sync inside attention layer | Still uses Transformer |
| **llama2.c** (Karpathy) | Train small LM in pure C | Still trains a Transformer |
| **esp32-ai** (slvDev, 2026) | 28.9M Transformer on ESP32 MCU | Still a Transformer |
| **AEMBR** (this work) | **Distill + pure oscillators · no Transformer** | Only line proposing full replacement |

---

## Technical Keywords

`kuramoto-model` `oscillator-network` `transformer-alternative` `llm-inference` `edge-ai` `on-device-ml` `knowledge-distillation` `sparse-computation` `cpu-inference` `tiny-ml` `bert-distillation` `phase-synchronization` `semantic-reasoning` `coupled-oscillators` `non-backprop` `distillation-instead-of-training`

---

## License

**非商业自由使用 · 商业需授权** — 详见 [LICENSE](LICENSE)

---

## 🔮

这份代码展示了一条路。但不是唯一的路。

拓扑不一定要从 BERT 的共现统计来。

如果你在读这句话时想到了"那还可以从什么来"——[CONTRIBUTING.md](CONTRIBUTING.md) 的末尾可能有你想要的答案。

---

*袭明 · 深渊实验室 · 2026-08-11*
