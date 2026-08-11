#!/usr/bin/env python3
"""
AEMBR v0.5 — 文本生成解码器
=====================================
从Kuramoto收敛后的相图快照 → 沿激活路径展开 → 完整句子
核心: 相位共振 = 语义相干, 沿最强共振链行走
=====================================
"""
import numpy as np
import re, time, pathlib, random
from collections import Counter, defaultdict
import jieba

# ════════════════════════════════════════════════
# 复用v0.4的拓扑引擎
# ════════════════════════════════════════════════

def build_engine():
    """复用v0.4的完整数据准备管道"""
    base = pathlib.Path("H:/文澜阁")
    all_words = []
    files_done = 0
    
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
                body = re.sub(r'[^\u4e00-\u9fff]', ' ', body)
                body = re.sub(r'\s+', ' ', body).strip()
                if len(body) > 100:
                    words = list(jieba.cut(body))
                    words = [w.strip() for w in words if len(w.strip()) >= 2]
                    all_words.extend(words)
                    files_done += 1
                    if files_done >= 500:
                        break
            except:
                pass
        if files_done >= 500:
            break
    
    # Vocabulary
    word_freq = Counter(all_words)
    vocab = [w for w, _ in word_freq.most_common(1000)]
    idx = {w: i for i, w in enumerate(vocab)}
    W = len(vocab)
    
    # Co-occurrence
    window = 8
    cooc = np.zeros((W, W), dtype=np.float32)
    for i in range(len(all_words) - window):
        w1 = all_words[i]
        if w1 not in idx: continue
        i1 = idx[w1]
        for d in range(1, window+1):
            if i+d >= len(all_words): break
            w2 = all_words[i+d]
            if w2 in idx:
                cooc[i1, idx[w2]] += 1.0 / d
    
    # PMI → J matrix
    total = cooc.sum() + 1e-8
    p_xy = cooc / total
    p_x = cooc.sum(axis=1, keepdims=True) / total
    p_y = cooc.sum(axis=0, keepdims=True) / total
    pmi = np.log((p_xy + 1e-8) / (p_x * p_y + 1e-8))
    pmi = np.maximum(pmi, 0)
    
    J = np.zeros((W, W), dtype=np.float32)
    for i in range(W):
        row = pmi[i].copy()
        row[i] = 0
        if row.sum() == 0: continue
        k = min(25, (row > 0).sum())
        if k == 0: continue
        thresh = np.sort(row[row > 0])[-k]
        mask = row >= thresh
        J[i, mask] = row[mask]
        J[mask, i] += row[mask] * 0.3
    
    if J.max() > 0:
        J = J / J.max()
    
    # 标记词性（简化：动词/名词/形容词的启发式规则）
    pos_tags = {}
    verb_suffix = {'化', '测', '算', '建', '证', '计', '估', '测', '定', '析'}
    noun_suffix = {'法', '器', '性', '度', '量', '值', '率', '数', '体', '系', '机', '器', '图', '程'}
    adj_suffix = {'的', '性', '化', '型'}
    
    for w in vocab:
        if any(w.endswith(s) for s in verb_suffix):
            pos_tags[w] = 'v'
        elif any(w.endswith(s) for s in noun_suffix):
            pos_tags[w] = 'n'
        elif len(w) == 1:
            pos_tags[w] = 'x'
        else:
            pos_tags[w] = 'n'  # default
    
    # 手动标记一些高频词
    manual = {
        '是': 'v', '有': 'v', '在': 'p', '和': 'c', '的': 'd',
        '可以': 'v', '能够': 'v', '需要': 'v', '通过': 'p',
        '提出': 'v', '发现': 'v', '证明': 'v', '实现': 'v',
        '导致': 'v', '产生': 'v', '具有': 'v', '存在': 'v',
        '方法': 'n', '系统': 'n', '模型': 'n', '数据': 'n',
        '问题': 'n', '结构': 'n', '结果': 'n', '理论': 'n',
        '显著': 'a', '重要': 'a', '复杂': 'a', '有效': 'a',
    }
    pos_tags.update(manual)
    
    omega = 1.0 + 0.02 * np.random.randn(W)
    
    print(f"✅ 引擎: {W}节点 {int(np.count_nonzero(J))}边")
    return vocab, idx, J, omega, pos_tags


def kuramoto_step(phase, omega, J, K=0.5, dt=0.03):
    pdiff = phase[np.newaxis, :] - phase[:, np.newaxis]
    dphi = omega + K * (J * np.sin(pdiff)).sum(axis=1)
    return (phase + dphi * dt) % (2 * np.pi)


def run_conv(phase, omega, J, K=0.5, steps=150):
    for _ in range(steps):
        phase = kuramoto_step(phase, omega, J, K)
    return phase


def encode(text, vocab, idx, W):
    words = list(jieba.cut(re.sub(r'[^\u4e00-\u9fff]', '', text)))
    words = [w.strip() for w in words if len(w.strip()) >= 2]
    phase = np.zeros(W)
    pert = []
    for w in words:
        if w in idx:
            phase[idx[w]] = np.pi * 0.9
            pert.append(idx[w])
    return phase, pert, words


# ════════════════════════════════════════════════
# 核心创新: 文本生成解码器
# ════════════════════════════════════════════════

def compute_activation_field(phase, J, pert_indices, depth=5):
    """
    多轮BFS扩散激活场
    从每个输入扰动节点出发，沿强耦合边传播激活
    多轮扩散→多重语义上下文叠加
    """
    W = len(phase)
    field = np.zeros(W)
    
    for pi in pert_indices:
        # 第1轮: 直接邻域
        local = np.zeros(W)
        local[pi] = 1.0
        visited = {pi}
        q = [(pi, 1.0, 0)]
        
        while q:
            node, val, d = q.pop(0)
            if d >= depth: continue
            for j in range(W):
                if J[node, j] > 0.05 and j not in visited:
                    visited.add(j)
                    al = np.cos(phase[j] - phase[node])
                    if al > 0.1:
                        nv = val * J[node, j] * max(al, 0.3) * 0.6
                        local[j] = max(local[j], nv)
                        q.append((j, nv, d+1))
        
        field += local
    
    # 归一化
    if field.max() > 0:
        field = field / field.max()
    return field


def find_resonance_clusters(phase, J, field, pert_indices, min_cluster_size=3):
    """
    在激活场中找到相位共振簇
    共振簇 = 相位对齐(cos>0.7) + 高激活度 + 强耦合边的连通分量
    """
    W = len(phase)
    pert_set = set(pert_indices)
    
    # 候选节点: 高激活度 + 非输入词
    candidates = [(i, field[i]) for i in range(W) 
                  if i not in pert_set and field[i] > 0.3]
    candidates.sort(key=lambda x: -x[1])
    
    if len(candidates) < min_cluster_size:
        return []
    
    # 贪婪聚类: 取前K个高激活节点，按相位一致性分组
    top_k = candidates[:15]
    clusters = []
    used = set()
    
    for i, score in top_k:
        if i in used: continue
        cluster = [i]
        used.add(i)
        
        for j, s2 in top_k:
            if j in used: continue
            # 相位对齐度
            align = np.cos(phase[j] - phase[i])
            # 拓扑耦合度
            coupling = J[i, j] + J[j, i]
            
            if align > 0.6 and coupling > 0.02:
                cluster.append(j)
                used.add(j)
        
        if len(cluster) >= min_cluster_size:
            clusters.append(cluster)
    
    # 按簇内总激活度排序
    clusters.sort(key=lambda c: sum(field[i] for i in c), reverse=True)
    return clusters


def order_by_mutual_coupling(phrase_words, J, idx, phase, vocab):
    """
    按耦合+相位对齐为词序列排序
    类似于TSP: 找一个遍历所有词的最短路径，边权 = 耦合强度 × 相位对齐
    """
    if len(phrase_words) <= 2:
        return phrase_words
    
    indices = [idx[w] for w in phrase_words if w in idx]
    if len(indices) <= 1:
        return phrase_words
    
    # 贪心路径: 从最强节点出发，每次跳向最对齐的邻居
    remaining = set(indices)
    start = max(remaining, key=lambda i: sum(J[i, j] for j in remaining if j != i))
    remaining.remove(start)
    
    ordered = [start]
    while remaining:
        last = ordered[-1]
        best = max(remaining, key=lambda j: J[last, j] * np.cos(phase[j] - phase[last]))
        ordered.append(best)
        remaining.remove(best)
    
    return [vocab[i] for i in ordered]


def generate_text(phase, J, field, pert_indices, vocab, idx, pos_tags, max_sentences=2):
    """
    主生成函数:
    1. 找共振簇 → 每个簇是一个候选短语
    2. 短语内排序 → 词序自然化
    3. 短语组合 → 多句连贯输出
    """
    clusters = find_resonance_clusters(phase, J, field, pert_indices)
    
    if not clusters:
        return "(无显著共振簇)"
    
    sentences = []
    used_words = set(pert_indices)
    
    for cluster in clusters[:max_sentences]:
        # 簇内词 → 去重 → 按耦合排序
        words = [vocab[i] for i in cluster if i not in used_words]
        if len(words) < 2:
            continue
        
        words = words[:8]  # 每句最多8词
        ordered = order_by_mutual_coupling(words, J, idx, phase, vocab)
        
        # 加连接词和标点
        sentence = assemble_sentence(ordered, pos_tags)
        if sentence:
            sentences.append(sentence)
            for i in cluster:
                used_words.add(i)
    
    if not sentences:
        return "(无连贯短语)"
    
    return '；'.join(sentences) + '。'


def assemble_sentence(words, pos_tags):
    """
    把词序列组装成自然中文句子
    策略: 名词堆叠→对象描述, 动词居中→动作描述
    """
    if not words:
        return ""
    
    # 分类
    nouns = [w for w in words if pos_tags.get(w, 'n') == 'n']
    verbs = [w for w in words if pos_tags.get(w, 'v') == 'v']
    adjs = [w for w in words if pos_tags.get(w, 'a') == 'a']
    
    # 策略1: 有动词 → "名词 + 动词 + 名词" 或 "形容词 + 名词 + 动词"
    if verbs and len(nouns) >= 2:
        return f"{nouns[0]}{verbs[0]}{nouns[1]}"
    
    # 策略2: 有形容词+名词 → "形容词+的+名词"
    if adjs and nouns:
        return f"{adjs[0]}的{nouns[0]}"
    
    # 策略3: 名词堆叠 → 并列
    if len(nouns) >= 2:
        return f"{'、'.join(nouns[:4])}等相关概念"
    
    # 降级: 直接拼
    return ''.join(words[:4])


# ════════════════════════════════════════════════
# 跑
# ════════════════════════════════════════════════

def main():
    print("=" * 55)
    print("AEMBR v0.5 — 文本生成解码器")
    print("=" * 55)
    
    # 建引擎
    t0 = time.time()
    vocab, idx, J, omega, pos_tags = build_engine()
    W = len(vocab)
    print(f"准备: {time.time()-t0:.1f}s\n")
    
    tests = [
        "量子计算需要纠错",
        "分布式系统使用共识协议",
        "神经网络通过反向传播学习",
        "密码学依赖数学难题",
        "时间同步是核心问题",
        "手性分子偏振光",
        "深度学习人工智能",
        "区块链智能合约",
    ]
    
    for sent in tests:
        print(f"{'─'*50}")
        # 编码
        phase0, pert, iw = encode(sent, vocab, idx, W)
        if len(pert) < 2:
            print(f"  📝 '{sent}' → 匹配不足")
            continue
        
        # 收敛
        phase1 = run_conv(phase0, omega, J, K=0.5, steps=130)
        
        # 激活场
        field = compute_activation_field(phase1, J, pert, depth=5)
        
        # 生成
        output = generate_text(phase1, J, field, pert, vocab, idx, pos_tags)
        
        # 展示激活场TOP
        ni = [(i, field[i]) for i in range(W) if i not in set(pert) and field[i] > 0.2]
        ni.sort(key=lambda x: -x[1])
        top_activated = [(vocab[i], f"{field[i]:.2f}") for i, _ in ni[:5]]
        
        print(f"  📥 '{sent}'")
        print(f"  📤 {output}")
        if top_activated:
            print(f"  🧠 激活场: {' | '.join(f'{w}({s})' for w,s in top_activated)}")
        print()


if __name__ == "__main__":
    main()
