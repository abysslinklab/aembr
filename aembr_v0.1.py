#!/usr/bin/env python3
"""
AEMBR (Abyss Emergent Meaning Builder & Resonator) — Prototype v0.1
=========================================================================
拓扑翻译层: 语言 → 拓扑场 → 振荡收敛 → 快照 → 语言

核心: 用耦合振荡器网络做最简单的"翻译-振荡-解码"验证。
不依赖外部模型, 不需要GPU。用小学算术理解"猫追老鼠"。

验证目标:
1. 语言输入能否被编码为相位扰动
2. 相位扰动能否在拓扑约束下收敛到有意义的稳定状态
3. 稳定状态能否被解码为可理解的语言输出
=========================================================================
"""

import numpy as np
from collections import Counter
import json

# ─── 第0层: 一个小世界 ─────────────────────────────────────────────

VOCAB = ["猫","狗","老鼠","跑","追","咬","吃","在","草地","家里",
         "快","慢","大","小","看见","害怕","开心","疼","喵","汪"]

# 手动拓扑关系 (耦合强度, 1=最强耦合)
# 这是"世界知识"的拓扑场——哪些词天然在一起
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

# ─── 第1层: 拓扑翻译层 (语言→拓扑场) ──────────────────────────────

def build_oscillator_network(vocab, topology, N=20):
    """从词汇和拓扑关系构建耦合振荡器网络"""
    words = vocab[:N]
    W = len(words)
    idx = {w: i for i, w in enumerate(words)}
    
    # 耦合矩阵 J (有向, 非对称)
    J = np.zeros((W, W))
    for (a, b), strength in topology.items():
        if a in idx and b in idx:
            i, j = idx[a], idx[b]
            J[i, j] = strength
            J[j, i] = strength * 0.7  # 反向稍弱
    
    # 固有频率 (随机但有小扰动)
    np.random.seed(42)
    omega = 1.0 + 0.1 * np.random.randn(W)
    
    return words, idx, J, omega


def encode_sentence(sentence, idx, J, omega):
    """
    翻译层: 将人类语言编码为相位扰动
    输入句子中的词 → 激活对应节点 → 注入初始相位扰动
    """
    W = len(idx)
    phase = np.zeros(W)  # 初始全部在0相位 (基态)
    
    input_words = sentence.replace("。","").replace("，","").split()
    
    perturbation = {}
    for word in input_words:
        if word in idx:
            i = idx[word]
            phase[i] = np.pi * 0.8  # 激活词相位跳到接近π (大扰动)
            perturbation[word] = i
    
    return phase, perturbation


# ─── 第2层: 振荡计算域 ─────────────────────────────────────────────

def kuramoto_step(phase, omega, J, K=0.3, dt=0.05):
    """
    耦合振荡器一步演化 (Kuramoto模型)
    dφ_i/dt = ω_i + K * Σ_j J_ij * sin(φ_j - φ_i)
    """
    W = len(phase)
    dphi = np.zeros(W)
    
    for i in range(W):
        coupling = 0.0
        for j in range(W):
            if J[i, j] > 0:
                coupling += J[i, j] * np.sin(phase[j] - phase[i])
        dphi[i] = omega[i] + K * coupling
    
    return (phase + dphi * dt) % (2 * np.pi)


def compute_potential(phase, J):
    """计算约束势能 U(Φ) —— 越低越稳定"""
    W = len(phase)
    U = 0.0
    for i in range(W):
        for j in range(W):
            if J[i, j] > 0:
                U += J[i, j] * (1 - np.cos(phase[j] - phase[i]))
    return U


def run_convergence(phase, omega, J, max_steps=500, tol=1e-4):
    """
    振荡计算域: 运行直到收敛
    返回: 最终相位, 势能演化曲线, 收敛步数
    """
    history = []
    prev_phase = phase.copy()
    
    for step in range(max_steps):
        phase = kuramoto_step(phase, omega, J)
        U = compute_potential(phase, J)
        history.append(U)
        
        # 检查收敛
        diff = np.max(np.abs(phase - prev_phase)) % (2 * np.pi)
        if step > 50 and diff < tol:
            break
        
        prev_phase = phase.copy()
    
    return phase, history, step + 1


# ─── 第3层: 快照解码层 (拓扑场→语言) ──────────────────────────────

def decode_phase_snapshot(phase, words, idx, J, topk=6):
    """
    快照解码: 从稳定相图中提取"认知状态"
    
    策略:
    1. 找到相位最稳定的节点簇 (低势能区)
    2. 找到与输入词相位差最小的词 (共振最强)
    3. 组合输出
    """
    W = len(phase)
    scores = np.zeros(W)
    
    # 计算每个节点的"稳定性得分"
    # 稳定 = 与强耦合邻居相位对齐
    for i in range(W):
        stability = 0.0
        total_j = 0.0
        for j in range(W):
            if J[i, j] > 0:
                # 相位差越小 = 耦合越满足 = 越稳定
                alignment = np.cos(phase[j] - phase[i])
                stability += J[i, j] * alignment
                total_j += J[i, j]
        if total_j > 0:
            scores[i] = stability / total_j
    
    # 排序取top
    ranked = np.argsort(-scores)
    
    result = []
    for i in ranked[:topk]:
        result.append({
            "词": words[i],
            "稳定性": round(float(scores[i]), 3),
            "相位": round(float(phase[i]), 3)
        })
    
    return result


# ─── 第4层: 主循环 ─────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("AEMBR v0.1 — 拓扑翻译·振荡·快照原型")
    print("=" * 60)
    
    # 建网
    words, idx, J, omega = build_oscillator_network(VOCAB, TOPOLOGY)
    W = len(words)
    print(f"\n📡 振荡器网络: {W} 节点, {np.count_nonzero(J)} 条边")
    
    # 测试句子
    test_sentences = [
        "猫 追 老鼠",
        "狗 追 猫",
        "老鼠 看见 猫 害怕 跑",
        "狗 咬 猫 疼",
        "猫 在 草地 看见 老鼠",
        "狗 在 家里 吃",
    ]
    
    for sentence in test_sentences:
        print(f"\n{'─' * 50}")
        print(f"📝 输入: {sentence}")
        
        # 翻译
        phase0, pert = encode_sentence(sentence, idx, J, omega)
        activated = list(pert.keys())
        print(f"🔑 激活节点: {activated}")
        
        # 振荡收敛
        phase_final, U_history, steps = run_convergence(phase0, omega, J)
        print(f"⏱  收敛: {steps}步 | U: {U_history[0]:.1f} → {U_history[-1]:.1f}")
        
        # 快照解码
        snapshot = decode_phase_snapshot(phase_final, words, idx, J)
        
        # 输出
        words_out = [s["词"] for s in snapshot]
        stabilities = [f"{s['稳定性']}" for s in snapshot]
        print(f"💎 快照: {' → '.join(words_out)}")
        print(f"📊 稳定性: {' | '.join(stabilities)}")
        
        # 拓扑一致性检查
        check_topology_consistency(phase_final, words, idx, J, activated)


def check_topology_consistency(phase, words, idx, J, activated_words):
    """检查: 激活词与输出快照之间的相位对齐是否合理"""
    activated_indices = [idx[w] for w in activated_words if w in idx]
    if len(activated_indices) < 2:
        return
    
    print(f"🔍 拓扑检查:")
    for i in activated_indices:
        # 找最同相的邻居
        neighbors = []
        for j in range(len(words)):
            if J[i, j] > 0:
                phase_diff = abs(phase[j] - phase[i])
                phase_diff = min(phase_diff, 2*np.pi - phase_diff)
                neighbors.append((words[j], phase_diff, J[i, j]))
        neighbors.sort(key=lambda x: x[1])
        
        # 显示top3最共振的词
        top3 = [f"{w}(Δφ={d:.2f},J={s:.1f})" for w, d, s in neighbors[:3]]
        print(f"  {words[i]} → {', '.join(top3)}")


if __name__ == "__main__":
    main()
