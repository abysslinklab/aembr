#!/usr/bin/env python3
"""
AEMBR v0.3 — 1000节点规模化原型
===================================
从真实中文语料自动提取拓扑, 验证Kuramoto推理在1000节点上的:
  - 收敛速度/步数
  - 收敛质量 (势函数下降曲线)
  - 输入感知解码质量
  - 稀疏度对推理速度的影响
===================================
"""
import numpy as np
import re, time, json, pathlib, random
from collections import Counter, defaultdict

# ═══════════════════════════════════════════════════════════
# 阶段1: 从真实语料提取词汇+构建拓扑
# ═══════════════════════════════════════════════════════════

def load_corpus(max_files=200, max_chars=500000):
    """从文澜阁中文概述加载语料"""
    base = pathlib.Path("H:/文澜阁")
    texts = []
    files_done = 0
    
    for cat_dir in sorted(base.iterdir()):
        if not cat_dir.is_dir() or not cat_dir.name.startswith("0"):
            continue
        papers_dir = cat_dir / "papers"
        if not papers_dir.exists():
            continue
        
        for md in sorted(papers_dir.glob("*_chinesereadme.md")):
            try:
                text = md.read_text(encoding='utf-8')[:3000]
                # Remove frontmatter
                parts = text.split('---', 2)
                body = parts[2] if len(parts) >= 3 else text
                # Remove markdown formatting
                body = re.sub(r'[#*`\-\[\]()>|]', ' ', body)
                body = re.sub(r'\s+', ' ', body).strip()
                if len(body) > 200:
                    texts.append(body)
                    files_done += 1
                    if files_done >= max_files:
                        break
            except:
                pass
        if files_done >= max_files:
            break
    
    all_text = ' '.join(texts)
    if len(all_text) > max_chars:
        all_text = all_text[:max_chars]
    print(f"📂 加载 {files_done} 篇文档, {len(all_text)} 字符")
    return all_text


def segment_chinese(text):
    """简化中文分词: 2-4字NGram + 单字过滤"""
    # Remove non-Chinese
    text = re.sub(r'[^\u4e00-\u9fff]', ' ', text)
    words = []
    
    # 提取2-4字词组 (模拟分词)
    chars = text.replace(' ', '')
    for n in [2, 3, 4]:
        for i in range(len(chars) - n + 1):
            words.append(chars[i:i+n])
    
    # 同时保留常见单字
    single_chars = re.findall(r'[\u4e00-\u9fff]', text)
    
    return words, single_chars


def build_vocabulary(all_text, vocab_size=1000):
    """从语料中提取高频词构建词汇表"""
    words, chars = segment_chinese(all_text)
    
    # 词频统计
    word_freq = Counter(words)
    char_freq = Counter(chars)
    
    # 取前vocab_size个词
    top_words = [w for w, _ in word_freq.most_common(vocab_size)]
    
    print(f"📊 词汇: {len(top_words)} 词 (词库大小 {len(word_freq)})")
    print(f"   示例: {top_words[:20]}")
    return top_words


def build_cooccurrence(text, vocab, window=5):
    """构建共现矩阵"""
    # 重新分词
    words, _ = segment_chinese(text)
    
    word_to_idx = {w: i for i, w in enumerate(vocab)}
    W = len(vocab)
    cooc = np.zeros((W, W), dtype=np.float32)
    
    # 滑动窗口统计共现
    for i in range(len(words) - window):
        center = words[i]
        if center not in word_to_idx:
            continue
        ci = word_to_idx[center]
        for j in range(1, window + 1):
            if i + j >= len(words):
                break
            ctx = words[i + j]
            if ctx in word_to_idx:
                cj = word_to_idx[ctx]
                cooc[ci, cj] += 1.0 / j  # 距离加权
    
    return cooc


def cooc_to_topology(cooc, vocab, min_count=3, max_edges_per_node=30):
    """共现矩阵→PMI加权拓扑图"""
    W = len(vocab)
    total = cooc.sum()
    
    # PMI = log(P(x,y) / (P(x)*P(y)))
    p_xy = cooc / (total + 1e-8)
    p_x = cooc.sum(axis=1, keepdims=True) / (total + 1e-8)
    p_y = cooc.sum(axis=0, keepdims=True) / (total + 1e-8)
    
    pmi = np.log((p_xy + 1e-8) / (p_x * p_y + 1e-8))
    pmi = np.maximum(pmi, 0)  # 只要正关联
    
    # 对每个节点, 只保留最强的max_edges_per_node条边
    J = np.zeros((W, W), dtype=np.float32)
    edge_count = 0
    
    for i in range(W):
        row = pmi[i].copy()
        row[i] = 0  # 去掉自环
        if row.sum() == 0:
            continue
        
        # 取最强k个邻居
        top_k = min(max_edges_per_node, (row > 0).sum())
        if top_k == 0:
            continue
        
        thresholds = np.sort(row[row > 0])[-top_k:]
        if len(thresholds) > 0:
            thresh = thresholds[0]
            mask = row >= thresh
            J[i, mask] = row[mask]
            J[mask, i] += row[mask] * 0.3  # 弱反向连接
            edge_count += mask.sum()
    
    # 归一化到 [0, 1]
    if J.max() > 0:
        J = J / J.max()
    
    # 稀疏度
    sparsity = 1.0 - (edge_count / (W * W))
    
    print(f"🔗 拓扑: {int(edge_count)} 条有向边, 稀疏度 {sparsity:.3%}")
    print(f"   平均度: {edge_count/W:.1f}")
    
    return J


# ═══════════════════════════════════════════════════════════
# 阶段2: Kuramoto推理引擎 (与v0.2兼容的优化版)
# ═══════════════════════════════════════════════════════════

def kuramoto_step_vectorized(phase, omega, J, K=0.4, dt=0.03):
    """向量化Kuramoto步——比逐节点循环快100倍"""
    W = len(phase)
    # sin(phase[j] - phase[i]) for all i,j pairs where J>0
    phase_diff = phase[np.newaxis, :] - phase[:, np.newaxis]
    sin_diff = np.sin(phase_diff)
    coupling = (J * sin_diff).sum(axis=1)
    dphi = omega + K * coupling
    return (phase + dphi * dt) % (2 * np.pi)


def run_convergence(phase, omega, J, K=0.4, max_steps=200, tol=0.001):
    """
    运行Kuramoto收敛
    返回: (final_phase, steps_taken, energy_curve, converged)
    """
    W = len(phase)
    energy_curve = []
    
    for step in range(max_steps):
        phase = kuramoto_step_vectorized(phase, omega, J, K=K)
        
        # 计算约束势函数
        phase_diff = phase[np.newaxis, :] - phase[:, np.newaxis]
        energy = (J * (1 - np.cos(phase_diff))).sum() / 2
        energy_curve.append(float(energy))
        
        # 收敛判定: 能量变化 < tol
        if step > 10 and len(energy_curve) >= 5:
            recent = energy_curve[-5:]
            if max(recent) - min(recent) < tol:
                return phase, step + 1, energy_curve, True
    
    return phase, max_steps, energy_curve, False


def encode_input(text, vocab, idx_map, J_shape):
    """将输入文本编码为相位扰动"""
    phase = np.zeros(J_shape)
    pert_indices = []
    input_parts = []
    
    # 简化分词
    chars = re.sub(r'[^\u4e00-\u9fff]', '', text)
    for n in [4, 3, 2]:
        for i in range(len(chars) - n + 1):
            w = chars[i:i+n]
            if w in idx_map:
                phase[idx_map[w]] = np.pi * 0.9
                pert_indices.append(idx_map[w])
                input_parts.append(w)
    
    # 单字作为后备
    if not pert_indices:
        for c in chars:
            if c in idx_map and idx_map[c] not in pert_indices:
                phase[idx_map[c]] = np.pi * 0.9
                pert_indices.append(idx_map[c])
                input_parts.append(c)
    
    return phase, pert_indices, input_parts


def decode_local_v2(phase, vocab, J, pert_indices, topk=10):
    """局部簇快照解码 - v0.2算法放大"""
    W = len(phase)
    activation = np.zeros(W)
    
    for pi in pert_indices:
        activation[pi] = 1.0
        visited = {pi}
        queue = [(pi, 1.0, 0)]
        
        while queue:
            node, act, depth = queue.pop(0)
            if depth >= 4:
                continue
            
            for j in range(W):
                if J[node, j] > 0.01 and j not in visited:
                    visited.add(j)
                    phase_align = np.cos(phase[j] - phase[node])
                    if phase_align > 0.2:
                        new_act = act * J[node, j] * phase_align * 0.6
                        activation[j] = max(activation[j], new_act)
                        queue.append((j, new_act, depth + 1))
    
    pert_set = set(pert_indices)
    non_input = [(i, activation[i]) for i in range(W) if i not in pert_set]
    non_input.sort(key=lambda x: -x[1])
    
    result = []
    for i, score in non_input[:topk]:
        if score > 0.01:
            result.append({"词": vocab[i], "激活度": round(float(score), 3)})
    
    return result


# ═══════════════════════════════════════════════════════════
# 阶段3: 基准测试
# ═══════════════════════════════════════════════════════════

def benchmark():
    print("=" * 60)
    print("AEMBR v0.3 — 1000节点规模化原型")
    print("=" * 60)
    
    # ── 数据准备 ──
    t0 = time.time()
    
    print("\n⏳ 加载语料...")
    text = load_corpus(max_files=200, max_chars=600000)
    
    print("⏳ 构建词汇表...")
    vocab = build_vocabulary(text, vocab_size=1000)
    
    print("⏳ 构建共现矩阵...")
    cooc = build_cooccurrence(text, vocab, window=5)
    
    print("⏳ 构建PMI拓扑...")
    J = cooc_to_topology(cooc, vocab, min_count=3, max_edges_per_node=25)
    
    W = len(vocab)
    idx_map = {w: i for i, w in enumerate(vocab)}
    
    # 固有频率
    np.random.seed(42)
    omega = 1.0 + 0.02 * np.random.randn(W)
    
    prep_time = time.time() - t0
    print(f"\n✅ 准备完成 ({prep_time:.1f}s)")
    
    # ── 收敛基准 ──
    print("\n" + "=" * 60)
    print("基准1: 收敛速度测试")
    print("=" * 60)
    
    for K in [0.2, 0.4, 0.6]:
        for N_test in [200, 500, 1000]:
            sub_W = min(N_test, W)
            sub_J = J[:sub_W, :sub_W]
            sub_omega = omega[:sub_W]
            
            phase0 = np.random.uniform(0, 2*np.pi, sub_W)
            
            t1 = time.time()
            phase_final, steps, energy, conv = run_convergence(
                phase0, sub_omega, sub_J, K=K, max_steps=300
            )
            elapsed = time.time() - t1
            
            energy_init = energy[0] if energy else 0
            energy_final = energy[-1] if energy else 0
            reduction = (energy_init - energy_final) / max(energy_init, 1e-8) * 100
            
            icon = "✅" if conv else "⚠️"
            print(f"  {icon} N={sub_W:4d} K={K} | {steps:3d}步, {elapsed:.2f}s | "
                  f"能量: {energy_init:.1f}→{energy_final:.2f} ({reduction:.0f}%)")
    
    # ── 推理质量测试 ──
    print("\n" + "=" * 60)
    print("基准2: 输入感知解码质量 (1000节点)")
    print("=" * 60)
    
    test_inputs = []
    
    # 从词汇表中随机组词作为输入
    random.seed(123)
    for _ in range(8):
        n_words = random.randint(3, 6)
        sample = random.sample(vocab, n_words)
        test_inputs.append(' '.join(sample))
    
    # 添加一些语义连贯的输入 (如果词汇表里有)
    semantic_tests = []
    for w in ["学习", "知识", "智能", "系统", "研究", "科学", "方法", "技术", "数据", "模型"]:
        if w in idx_map:
            semantic_tests.append(w)
    if len(semantic_tests) >= 3:
        test_inputs.append(' '.join(semantic_tests[:5]))
    
    for sent in test_inputs[:6]:
        phase0, pert_idx, input_parts = encode_input(sent, vocab, idx_map, W)
        
        if not pert_idx:
            print(f"  ⏭️  '{sent[:30]}...' → 无匹配词汇")
            continue
        
        t1 = time.time()
        phase1, steps, energy, conv = run_convergence(
            phase0, omega, J, K=0.4, max_steps=200
        )
        elapsed = time.time() - t1
        
        snap = decode_local_v2(phase1, vocab, J, pert_idx, topk=6)
        
        out_words = ' '.join([s['词'] for s in snap[:4]])
        print(f"  📝 '{sent[:40]}' → {steps}步, {elapsed:.2f}s")
        print(f"     💬 {out_words if out_words else '(无显著共振)'}")
    
    # ── 可扩展性报告 ──
    print("\n" + "=" * 60)
    print("基准3: 可扩展性报告")
    print("=" * 60)
    
    total_edges = int(np.count_nonzero(J))
    density = total_edges / (W * W) * 100
    mem_kb = J.nbytes / 1024 + phase.nbytes / 1024
    
    print(f"  节点: {W}")
    print(f"  边: {total_edges} (密度 {density:.2f}%)")
    print(f"  平均度: {total_edges/W:.1f}")
    print(f"  内存: {mem_kb:.0f} KB")
    print(f"  计算复杂度: O({W}×{total_edges/W:.1f}) = ~{int(W * total_edges/W):,} 次浮点运算/步")
    print(f"  单步推理耗时 (Python/NumPy): ~{elapsed/steps*1000:.1f} ms/步" if steps > 0 else "")
    
    # ── 外推到15K ──
    print(f"\n  外推到 15,000 节点 (密度相同):")
    ext_W = 15000
    ext_edges = int(ext_W * total_edges / W)
    ext_mem = ext_edges * 4 / 1024  # float32边权重
    print(f"    边: ~{ext_edges:,}")
    print(f"    内存: ~{ext_mem:.0f} KB = {ext_mem/1024:.1f} MB")
    print(f"    单步推理: ~{ext_W * total_edges / W / 1e6:.1f}M FLOPs/步")
    print(f"    预估C实现单步: ~{ext_W * total_edges / W / 1e9 * 10:.1f} ms/步")


if __name__ == "__main__":
    benchmark()
