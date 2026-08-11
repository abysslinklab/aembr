#!/usr/bin/env python3
"""
AEMBR v0.2 — 输入感知快照解码
=========================================================================
修复: v0.1的全局收敛洗掉了输入特异性
方案: 部分收敛 + 输入扰动残差加权 + 局部簇快照
=========================================================================
"""

import numpy as np
from collections import Counter

# ─── 词汇与拓扑 (同v0.1) ──────────────────────────────────────────

VOCAB = ["猫","狗","老鼠","跑","追","咬","吃","在","草地","家里",
         "快","慢","大","小","看见","害怕","开心","疼","喵","汪"]

TOPOLOGY = {
    ("猫","老鼠"): 0.9,  ("猫","追"): 0.8,    ("猫","咬"): 0.5,
    ("猫","喵"): 0.7,    ("猫","看见"): 0.4,   ("猫","快"): 0.3,
    ("狗","老鼠"): 0.8,  ("狗","追"): 0.9,    ("狗","咬"): 0.7,
    ("狗","汪"): 0.7,    ("狗","看见"): 0.4,   ("狗","快"): 0.5,
    ("老鼠","跑"): 0.9,  ("老鼠","害怕"): 0.6, ("老鼠","小"): 0.8,
    ("老鼠","吃"): 0.3,  ("老鼠","疼"): 0.5,
    ("追","跑"): 0.9,    ("追","快"): 0.7,    ("咬","疼"): 0.8,
    ("咬","吃"): 0.6,    ("吃","开心"): 0.5,  ("害怕","跑"): 0.8,
    ("看见","老鼠"): 0.6,("看见","猫"): 0.5,  ("看见","狗"): 0.5,
    ("在","草地"): 0.6,  ("在","家里"): 0.6,  ("草地","跑"): 0.5,
    ("家里","开心"): 0.4,("大","狗"): 0.6,    ("小","老鼠"): 0.7,
    ("快","跑"): 0.8,    ("快","追"): 0.7,    ("慢","老鼠"): 0.4,
}


def build_network(vocab, topology, N=20):
    words = vocab[:N]
    W = len(words)
    idx = {w: i for i, w in enumerate(words)}
    J = np.zeros((W, W))
    for (a, b), s in topology.items():
        if a in idx and b in idx:
            i, j = idx[a], idx[b]
            J[i, j] = s
            J[j, i] = s * 0.7
    np.random.seed(42)
    omega = 1.0 + 0.05 * np.random.randn(W)
    return words, idx, J, omega


def encode(sentence, idx, J, W):
    phase = np.zeros(W)
    input_words = sentence.replace("。","").replace("，","").split()
    pert_indices = []
    for w in input_words:
        if w in idx:
            i = idx[w]
            phase[i] = np.pi * 0.9
            pert_indices.append(i)
    return phase, pert_indices, input_words


def kuramoto_step(phase, omega, J, K=0.4, dt=0.03):
    W = len(phase)
    dphi = np.zeros(W)
    for i in range(W):
        coupling = sum(J[i, j] * np.sin(phase[j] - phase[i]) for j in range(W) if J[i, j] > 0)
        dphi[i] = omega[i] + K * coupling
    return (phase + dphi * dt) % (2 * np.pi)


def run_partial(phase, omega, J, steps=80):
    """部分收敛——不完全同步, 保留输入特异性"""
    for _ in range(steps):
        phase = kuramoto_step(phase, omega, J)
    return phase


def decode_local(phase, words, idx, J, pert_indices, topk=8):
    """
    🆕 输入感知解码:
    - 不只取全局最稳定节点
    - 从输入扰动节点出发, 沿耦合边扩散, 找共振最强的局部簇
    """
    W = len(phase)
    
    # 每个节点的"激活度" = 与扰动节点的相位累积共振
    activation = np.zeros(W)
    
    for pi in pert_indices:
        # 初始扰动 = 1.0
        activation[pi] = 1.0
        
        # BFS扩散: 沿强耦合边传播激活
        visited = {pi}
        queue = [(pi, 1.0, 0)]
        
        while queue:
            node, act, depth = queue.pop(0)
            if depth >= 3:
                continue
            
            for j in range(W):
                if J[node, j] > 0.3 and j not in visited:
                    visited.add(j)
                    # 激活衰减 = 耦合强度 × 相位对齐度
                    phase_align = np.cos(phase[j] - phase[node])
                    if phase_align > 0.3:  # 只有共振显著时才传播
                        new_act = act * J[node, j] * phase_align * 0.7
                        activation[j] = max(activation[j], new_act)
                        queue.append((j, new_act, depth + 1))
    
    # 排除原始输入词 (它们天然激活最高)
    pert_set = set(pert_indices)
    
    # 按激活度排序, 但排除输入词自身
    non_input_scores = [(i, activation[i]) for i in range(W) if i not in pert_set]
    non_input_scores.sort(key=lambda x: -x[1])
    
    result = []
    for i, score in non_input_scores[:topk]:
        result.append({
            "词": words[i],
            "激活度": round(float(score), 3),
            "相位偏移": round(float(phase[i]), 2)
        })
    
    return result


def write_snapshot(sentence, result):
    """把快照写成自然语言"""
    words = [r["词"] for r in result if r["激活度"] > 0.3]
    actions = [w for w in words if w in ["追","跑","咬","吃","看见","害怕","开心","疼","叫"]]
    entities = [w for w in words if w in ["猫","狗","老鼠","草地","家里"]]
    
    if actions and entities:
        return f"「{sentence}」→ {entities[0]}{actions[0]}了(激活度{result[0]['激活度']})"
    elif words:
        return f"「{sentence}」→ 关联: {' '.join(words[:4])}"
    return f"「{sentence}」→ (无显著共振)"


def main():
    print("=" * 60)
    print("AEMBR v0.2 — 输入感知局部簇快照")
    print("=" * 60)
    
    words, idx, J, omega = build_network(VOCAB, TOPOLOGY)
    W = len(words)
    print(f"\n📡 {W}节点 {np.count_nonzero(J)}边 | K=0.4")
    
    tests = [
        "猫 追 老鼠",
        "狗 追 猫",
        "老鼠 看见 猫 害怕 跑",
        "狗 咬 猫 疼",
        "猫 在 草地 看见 老鼠",
        "狗 在 家里 吃",
    ]
    
    for sent in tests:
        print(f"\n{'─' * 50}")
        
        phase0, pert_idx, input_w = encode(sent, idx, J, W)
        phase1 = run_partial(phase0, omega, J, steps=80)
        snap = decode_local(phase1, words, idx, J, pert_idx)
        
        print(f"📝 输入: {sent}")
        
        # 差异展示
        for r in snap[:4]:
            bar = "█" * min(int(r["激活度"] * 10), 10)
            print(f"  {bar} {r['词']} ({r['激活度']})")
        
        print(f"  💬 {write_snapshot(sent, snap)}")


if __name__ == "__main__":
    main()
