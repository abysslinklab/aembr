#!/usr/bin/env python3
"""
AEMBR v0.7 — BERT蒸馏·5000节点
=================================
扩到5000词, 测试大规模收敛行为
"""
import numpy as np, torch, json, time, pathlib, re, jieba
from collections import Counter
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity

# ═══════════════════ 1. 语料词汇 ═══════════════════
print("⏳ 语料分词...")
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
                done+=1
                if done>=600: break
        except: pass
    if done>=600: break

vocab = [w for w,_ in Counter(words).most_common(5000)]
print(f"📊 {len(vocab)}词 → 语料{len(words)}词")

# ═══════════════════ 2. BERT嵌入 ═══════════════════
print("⏳ BERT嵌入...")
tok = AutoTokenizer.from_pretrained("bert-base-chinese")
model = AutoModel.from_pretrained("bert-base-chinese")
model.eval()
N = len(vocab)
emb = np.zeros((N, 768), dtype=np.float32)

t0 = time.time()
for i, w in enumerate(vocab):
    text = f"这是一个{w}。"
    inp = tok(text, return_tensors="pt", truncation=True, max_length=10)
    with torch.no_grad():
        out = model(**inp)
    emb[i] = out.last_hidden_state[0, 0].cpu().numpy()
    if (i+1)%1000==0:
        rate = (i+1)/(time.time()-t0)
        eta = (N-i-1)/rate
        print(f"  {i+1}/{N} | {rate:.0f}词/s | ETA {eta:.0f}s")

print(f"✅ 嵌入: {time.time()-t0:.0f}s")

# ═══════════════════ 3. J矩阵 ═══════════════════
print("⏳ 余弦→J矩阵...")
sim = cosine_similarity(emb)
np.fill_diagonal(sim, 0)

J = np.zeros((N, N), dtype=np.float32)
topk = min(25, N-1)
edges = 0
for i in range(N):
    indices = np.argpartition(-sim[i], topk)[:topk]
    for j in indices:
        if sim[i,j] > 0.3:
            J[i,j] = sim[i,j]
            edges += 1

print(f"🔗 {edges}边, 稀疏{1-edges/(N*N):.2%}, 度{edges/N:.1f}")

# ═══════════════════ 4. Kuramoto测试 ═══════════════════
idx = {w:i for i,w in enumerate(vocab)}
omega = 1.0 + 0.02*np.random.randn(N)

def ks(phase, omega, J, K=0.5, dt=0.03):
    pd = phase[None,:] - phase[:,None]
    return (phase + dt*(omega + K*(J*np.sin(pd)).sum(axis=1))) % (2*np.pi)

def run(phase, omega, J, K=0.5, steps=150):
    for _ in range(steps):
        phase = ks(phase, omega, J, K)
    return phase

def decode(phase, J, pert, topk=8):
    act = np.zeros(N)
    for pi in pert:
        act[pi]=1.0; visited={pi}; q=[(pi,1.0,0)]
        while q:
            n,v,d=q.pop(0)
            if d>=4: continue
            for j in range(N):
                if J[n,j]>0.01 and j not in visited:
                    visited.add(j)
                    al=np.cos(phase[j]-phase[n])
                    if al>0.2:
                        act[j]=max(act[j],v*J[n,j]*max(al,0.3)*0.6)
                        q.append((j,act[j],d+1))
    ps=set(pert)
    ni=[(i,act[i]) for i in range(N) if i not in ps and act[i]>0.01]
    ni.sort(key=lambda x:-x[1])
    return [(vocab[i],round(float(s),3)) for i,s in ni[:topk]]

# 收敛测试
print(f"\n{'='*55}")
print("收敛测试")
for K_val in [0.3, 0.5, 0.7]:
    p0 = np.random.uniform(0, 2*np.pi, N)
    t1 = time.time()
    pf = run(p0, omega, J, K=K_val, steps=150)
    dt = time.time() - t1
    pdiff = pf[None,:] - pf[:,None]
    energy = (J * (1 - np.cos(pdiff))).sum() / 2
    print(f"  K={K_val} | {dt:.1f}s | 势能={energy:.1f}")

# 推理测试
print(f"\n{'='*55}")
print("推理测试")
tests = ['量子计算需要纠错','分布式系统共识协议','神经网络深度学习','密码学安全','人工智能机器学习','区块链去中心化']
for sent in tests:
    ws = [w.strip() for w in jieba.cut(re.sub(r'[^\u4e00-\u9fff]','',sent)) if len(w.strip())>=2]
    phase = np.zeros(N)
    pert = [idx[w] for w in ws if w in idx]
    if len(pert)<2: continue
    for i in pert: phase[i]=np.pi*0.9
    t1=time.time()
    pf=run(phase,omega,J,K=0.5,steps=150)
    dt=time.time()-t1
    out=decode(pf,J,pert)
    print(f"  {sent} → {dt:.1f}s | {' '.join(f'{w}' for w,_ in out[:5])}")

# 保存
out = {"vocab":vocab,"J":J.tolist(),"n":N,"edges":int(edges),"source":"BERT_cosine_5K"}
pathlib.Path("H:/otherproject/深渊实验室/prototype/aembr_5k_topology.json").write_text(json.dumps(out,ensure_ascii=False))
print(f"\n✅ 保存: aembr_5k_topology.json ({N}节点)")
