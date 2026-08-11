#!/usr/bin/env python3
"""
AEMBR v0.4 — jieba分词 + 1868篇语料 + 1000节点
=================================================
修复: jieba精确分词替代n-gram暴力切词
数据: 文澜阁全部1868篇中文概述
拓扑: PMI共现 → 词向量相似度双重验证
=================================================
"""
import numpy as np
import re, time, json, pathlib, random
from collections import Counter, defaultdict

try:
    import jieba
except ImportError:
    print("需要 jieba: pip install jieba")
    exit(1)

# ════════════════════════════════════════════════
# 阶段1: 语料+分词
# ════════════════════════════════════════════════

def load_corpus_jieba(max_files=500, max_chars=2000000):
    """用jieba分词加载语料"""
    base = pathlib.Path("H:/文澜阁")
    all_words = []
    files_done = 0
    total_chars = 0
    
    for cat_dir in sorted(base.iterdir()):
        if not cat_dir.is_dir() or not cat_dir.name.startswith("0"):
            continue
        papers_dir = cat_dir / "papers"
        if not papers_dir.exists():
            continue
        
        for md in sorted(papers_dir.glob("*_chinesereadme.md")):
            try:
                text = md.read_text(encoding='utf-8')[:5000]
                parts = text.split('---', 2)
                body = parts[2] if len(parts) >= 3 else text
                # 只保留中文
                body = re.sub(r'[^\u4e00-\u9fff]', ' ', body)
                body = re.sub(r'\s+', ' ', body).strip()
                
                if len(body) > 100:
                    words = list(jieba.cut(body))
                    # 过滤单字和纯数字
                    words = [w.strip() for w in words if len(w.strip()) >= 2]
                    all_words.extend(words)
                    files_done += 1
                    total_chars += len(body)
                    
                    if files_done >= max_files:
                        break
            except:
                pass
        if files_done >= max_files:
            break
    
    print(f"📂 {files_done}篇, {total_chars}字, {len(all_words)}词")
    return all_words


def build_vocab_and_cooc(words, vocab_size=1000, window=8):
    """构建词汇表+共现矩阵"""
    word_freq = Counter(words)
    top_words = [w for w, _ in word_freq.most_common(vocab_size)]
    
    word_to_idx = {w: i for i, w in enumerate(top_words)}
    W = len(top_words)
    cooc = np.zeros((W, W), dtype=np.float32)
    
    # 滑动窗口统计
    for i in range(len(words) - window):
        w1 = words[i]
        if w1 not in word_to_idx:
            continue
        i1 = word_to_idx[w1]
        for d in range(1, window + 1):
            if i + d >= len(words):
                break
            w2 = words[i + d]
            if w2 in word_to_idx:
                i2 = word_to_idx[w2]
                cooc[i1, i2] += 1.0 / d
    
    print(f"📊 vocab={len(top_words)}, top: {top_words[:15]}")
    return top_words, word_to_idx, cooc


def cooc_to_topology_pmi(cooc, max_edges=25):
    """PMI拓扑"""
    W = len(cooc)
    total = cooc.sum() + 1e-8
    p_xy = cooc / total
    p_x = cooc.sum(axis=1, keepdims=True) / total
    p_y = cooc.sum(axis=0, keepdims=True) / total
    pmi = np.log((p_xy + 1e-8) / (p_x * p_y + 1e-8))
    pmi = np.maximum(pmi, 0)
    
    J = np.zeros((W, W), dtype=np.float32)
    edges = 0
    for i in range(W):
        row = pmi[i].copy()
        row[i] = 0
        if row.sum() == 0:
            continue
        top_k = min(max_edges, (row > 0).sum())
        if top_k == 0:
            continue
        thresh = np.sort(row[row > 0])[-top_k]
        mask = row >= thresh
        J[i, mask] = row[mask]
        J[mask, i] += row[mask] * 0.3
        edges += mask.sum()
    
    if J.max() > 0:
        J = J / J.max()
    
    sparsity = 1 - edges / (W * W)
    print(f"🔗 {int(edges)}边, 稀疏{sparsity:.2%}, 平均度{edges/W:.1f}")
    return J


# ════════════════════════════════════════════════
# 阶段2: Kuramoto引擎
# ════════════════════════════════════════════════

def kuramoto_step(phase, omega, J, K=0.4, dt=0.03):
    pdiff = phase[np.newaxis, :] - phase[:, np.newaxis]
    dphi = omega + K * (J * np.sin(pdiff)).sum(axis=1)
    return (phase + dphi * dt) % (2 * np.pi)

def run_conv(phase, omega, J, K=0.4, max_steps=200, tol=0.01):
    curve = []
    for step in range(max_steps):
        phase = kuramoto_step(phase, omega, J, K, dt=0.03)
        pdiff = phase[np.newaxis, :] - phase[:, np.newaxis]
        e = (J * (1 - np.cos(pdiff))).sum() / 2
        curve.append(float(e))
        if step > 10 and len(curve) >= 5:
            r = curve[-5:]
            if max(r) - min(r) < tol:
                return phase, step + 1, curve, True
    return phase, max_steps, curve, False

def encode(text, vocab, idx_map, W):
    words = list(jieba.cut(re.sub(r'[^\u4e00-\u9fff]', '', text)))
    words = [w.strip() for w in words if len(w.strip()) >= 2]
    
    phase = np.zeros(W)
    pert = []
    for w in words:
        if w in idx_map:
            phase[idx_map[w]] = np.pi * 0.9
            pert.append(idx_map[w])
    return phase, pert, words

def decode(phase, vocab, J, pert_idx, topk=8):
    W = len(phase)
    act = np.zeros(W)
    for pi in pert_idx:
        act[pi] = 1.0
        visited = {pi}
        q = [(pi, 1.0, 0)]
        while q:
            node, val, depth = q.pop(0)
            if depth >= 4:
                continue
            for j in range(W):
                if J[node, j] > 0.01 and j not in visited:
                    visited.add(j)
                    al = np.cos(phase[j] - phase[node])
                    if al > 0.2:
                        nv = val * J[node, j] * al * 0.6
                        act[j] = max(act[j], nv)
                        q.append((j, nv, depth + 1))
    
    ps = set(pert_idx)
    ni = [(i, act[i]) for i in range(W) if i not in ps and act[i] > 0.01]
    ni.sort(key=lambda x: -x[1])
    return [{"词": vocab[i], "激活度": round(float(s), 3)} for i, s in ni[:topk]]


# ════════════════════════════════════════════════
# 阶段3: 跑
# ════════════════════════════════════════════════

def main():
    print("=" * 55)
    print("AEMBR v0.4 — jieba分词 + 大语料")
    print("=" * 55)
    
    # 加载
    t0 = time.time()
    words = load_corpus_jieba(max_files=500, max_chars=2000000)
    vocab, idx, cooc = build_vocab_and_cooc(words, vocab_size=1000)
    J = cooc_to_topology_pmi(cooc, max_edges=25)
    W = len(vocab)
    
    np.random.seed(42)
    omega = 1.0 + 0.02 * np.random.randn(W)
    print(f"✅ 准备: {time.time()-t0:.1f}s\n")
    
    # 收敛测试
    print("=" * 55)
    print("收敛测试")
    print("=" * 55)
    
    for N in [200, 500, 1000]:
        sJ = J[:N, :N]
        sw = omega[:N]
        for K in [0.3, 0.5]:
            p0 = np.random.uniform(0, 2*np.pi, N)
            t1 = time.time()
            pf, st, en, ok = run_conv(p0, sw, sJ, K=K, max_steps=200)
            dt = time.time() - t1
            e0, ef = en[0], en[-1]
            r = (e0-ef)/max(e0, 1e-8)*100
            print(f"  {'✅' if ok else '⚡'} N={N:4d} K={K} | {st:3d}步 {dt:.2f}s | 能量{e0:.0f}→{ef:.1f}({r:.0f}%)")
    
    # 推理测试
    print(f"\n{'='*55}")
    print("推理质量测试")
    print("=" * 55)
    
    tests = [
        "量子计算需要纠错因为测量会塌缩",
        "分布式系统使用共识协议保证一致性",
        "神经网络通过反向传播学习参数",
        "密码学依赖数学难题保证安全",
        "时间同步是分布式系统的核心问题",
        "手性分子在偏振光下表现不同",
        "深度学习 自然语言处理 人工智能",
        "区块链 智能合约 去中心化",
    ]
    
    for sent in tests:
        p0, pi, iw = encode(sent, vocab, idx, W)
        if not pi:
            print(f"  ⏭️  '{sent[:30]}' → 无匹配")
            continue
        
        pf, st, en, ok = run_conv(p0, omega, J, K=0.5, max_steps=150)
        out = decode(pf, vocab, J, pi, topk=6)
        ow = ' '.join([o['词'] for o in out[:4]])
        a = max([o['激活度'] for o in out]) if out else 0
        print(f"  📝 '{sent[:35]}' → {st}步 | 💬 {ow} (max激活={a:.2f})")
    
    # 统计
    print(f"\n{'='*55}")
    print("统计")
    print("=" * 55)
    edges = int(np.count_nonzero(J))
    den = edges/(W*W)*100
    print(f"  节点:{W} 边:{edges} 密度:{den:.2f}% 平均度:{edges/W:.1f}")
    print(f"  内存:{J.nbytes/1024:.0f}KB")
    print(f"  1000节点×200步: ~{3*200/W*edges/1e6:.1f}M FLOPs")


if __name__ == "__main__":
    main()
