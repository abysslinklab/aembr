#!/usr/bin/env python3
"""
AEMBR v0.6c — BERT嵌入蒸馏拓扑 (极简版)
=============================================
词汇: jieba语料500词
边: BERT上下文嵌入·余弦相似度·top25稀疏化
≈30秒跑完
"""
import numpy as np
import torch, json, time, pathlib, re, jieba
from collections import Counter
from transformers import AutoTokenizer, AutoModel

# ═══════════════════════════
# 1. 语料词汇
# ═══════════════════════════
def get_vocab(n=500):
    base = pathlib.Path("H:/文澜阁")
    words = []; done = 0
    for d in sorted(base.iterdir()):
        if not d.is_dir() or not d.name.startswith("0"): continue
        for m in sorted((d/"papers").glob("*_chinesereadme.md")):
            try:
                t = m.read_text(encoding='utf-8')[:5000]
                parts = t.split('---',2)
                b = re.sub(r'[^\u4e00-\u9fff]',' ',parts[2] if len(parts)>=3 else t).strip()
                if len(b)>100:
                    words.extend(w.strip() for w in jieba.cut(b) if len(w.strip())>=2)
                    done+=1; 
                    if done>=200: break
            except: pass
        if done>=200: break
    vocab = [w for w,_ in Counter(words).most_common(n)]
    print(f"📊 {len(vocab)}词: {' '.join(vocab[:8])}")
    return vocab

# ═══════════════════════════
# 2. BERT嵌入
# ═══════════════════════════
def get_embeddings(vocab):
    print("⏳ 加载 bert-base-chinese...")
    tok = AutoTokenizer.from_pretrained("bert-base-chinese")
    model = AutoModel.from_pretrained("bert-base-chinese")
    model.eval()
    
    embs = np.zeros((len(vocab), 768), dtype=np.float32)
    
    for i, w in enumerate(vocab):
        # 用简单模板构造上下文: "这是一个[词]。"
        text = f"这是一个{w}。"
        inp = tok(text, return_tensors="pt", truncation=True, max_length=10)
        with torch.no_grad():
            out = model(**inp)
        # 取[CLS] token的嵌入作为词表征
        embs[i] = out.last_hidden_state[0, 0].cpu().numpy()
        
        if (i+1)%100==0:
            print(f"  {i+1}/{len(vocab)}")
    
    print(f"✅ 嵌入: {embs.shape}")
    return embs

# ═══════════════════════════
# 3. 余弦 → J矩阵
# ═══════════════════════════
def cosine_to_J(embs, topk=25):
    from sklearn.metrics.pairwise import cosine_similarity
    sim = cosine_similarity(embs)
    np.fill_diagonal(sim, 0)
    
    W = len(embs)
    J = np.zeros((W,W), dtype=np.float32)
    edges = 0
    for i in range(W):
        idx = np.argpartition(-sim[i], topk)[:topk]
        for j in idx:
            if sim[i,j] > 0.3:
                J[i,j] = sim[i,j]
                edges += 1
    
    print(f"🔗 {edges}边, 稀疏{1-edges/(W*W):.2%}, 度{edges/W:.1f}")
    return J

# ═══════════════════════════
# Main
# ═══════════════════════════
t0 = time.time()
vocab = get_vocab(500)
emb = get_embeddings(vocab)
J = cosine_to_J(emb)

# 最强边
ix = np.argsort(J.ravel())[::-1]
print(f"\n最强20边:")
n=0
for k in ix:
    i,j = k//500, k%500
    if i<j and J[i,j]>0:
        print(f"  {vocab[i]} ↔ {vocab[j]}: {J[i,j]:.3f}")
        n+=1
        if n>=20: break

out = {"vocab":vocab,"J":J.tolist(),"n":len(vocab),"edges":int(np.count_nonzero(J)),"source":"BERT_cosine"}
pathlib.Path("H:/otherproject/深渊实验室/prototype/aembr_bert_topology.json").write_text(json.dumps(out,ensure_ascii=False))
print(f"\n✅ {time.time()-t0:.0f}s → aembr_bert_topology.json")
