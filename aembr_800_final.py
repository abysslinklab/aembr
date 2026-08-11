#!/usr/bin/env python3
"""AEMBR 800节点优化拓扑 + 完整推理验证"""
import json, numpy as np, jieba, re, time
from collections import Counter
from pathlib import Path
from sklearn.cluster import AgglomerativeClustering
import torch
from transformers import AutoTokenizer, AutoModel

t_total = time.time()

# ═══ 1. 800词表 ═══
print('═══ 1. 构建800词表 ═══')
base = Path('H:/文澜阁')
words, done = [], 0
for d in sorted(base.iterdir()):
    if not d.is_dir() or not d.name.startswith('0'): continue
    for m in sorted((d/'papers').glob('*_chinesereadme.md')):
        try:
            t=m.read_text(encoding='utf-8')[:5000]; parts=t.split('---',2)
            b=re.sub(r'[^\u4e00-\u9fff]',' ',parts[2] if len(parts)>=3 else t).strip()
            words.extend(w.strip() for w in jieba.cut(b) if len(w.strip())>=2)
            done+=1
            if done>=600: break
        except: pass
    if done>=600: break

# 取前800高频 + 强制加入测试所需词
vocab_base = [w for w,_ in Counter(words).most_common(800)]
extra = ['密码学','纠错','去中心化','中心化','链去','共识算法',
         '深度学习','强化学习','自然语言','计算机','机器人',
         '加密','解密','哈希','签名','验证','授权','量子比特','叠加态']
for w in extra:
    if w not in vocab_base and w in Counter(words):
        vocab_base.append(w)
vocab = vocab_base[:800]
N = len(vocab)
idx = {w:i for i,w in enumerate(vocab)}
print(f'{N}词, 示例: {vocab[:10]}')

# 覆盖检查
tests = ['分布式系统共识协议','密码学数学安全','神经网络深度学习',
         '区块链去中心化','量子计算纠错','人工智能机器学习']
for sent in tests:
    ws=[w for w in jieba.cut(re.sub(r'[^\u4e00-\u9fff]','',sent)) if len(w.strip())>=2]
    hit=sum(1 for w in ws if w in idx)
    miss=[w for w in ws if w not in idx]
    print(f'  {sent[:30]}: {hit}/{len(ws)} 缺:{miss if miss else "✅全部"}')

# ═══ 2. BERT嵌入 ═══
print('\n═══ 2. BERT嵌入 ═══')
tok = AutoTokenizer.from_pretrained('bert-base-chinese')
model = AutoModel.from_pretrained('bert-base-chinese'); model.eval()
emb = np.zeros((N,768),dtype=np.float32)
t0=time.time()
with torch.no_grad():
    for i,w in enumerate(vocab):
        emb[i]=model(**tok(f'这是一个{w}。',return_tensors='pt')).last_hidden_state[0,0].detach().cpu().numpy()
        if (i+1)%200==0: print(f'  {i+1}/{N}')
print(f'{time.time()-t0:.0f}s')

# ═══ 3. 余弦→J ═══
print('\n═══ 3. 余弦→J ═══')
from sklearn.metrics.pairwise import cosine_similarity
sim = cosine_similarity(emb)
np.fill_diagonal(sim,0)
J = np.zeros((N,N),dtype=np.float32)
for i in range(N):
    k=min(25,N-1); top=np.argpartition(-sim[i],k)[:k]
    for j in top:
        if sim[i,j]>0.25: J[i,j]=sim[i,j]
edges0 = int(np.count_nonzero(J))
print(f'{edges0}边, 稀疏{1-edges0/(N*N):.2%}')

# ═══ 4. O1领域聚类 ═══
print('\n═══ 4. O1领域聚类 ═══')
labels = AgglomerativeClustering(n_clusters=20).fit_predict(emb)
for d_id in range(20):
    ws=[vocab[i] for i in range(N) if labels[i]==d_id][:6]
    print(f'  D{d_id:02d}: {" ".join(ws)}')

# 同域增强
for i in range(N):
    for j in range(i+1,N):
        if J[i,j]>0 and labels[i]==labels[j]:
            J[i,j]=min(1.0,J[i,j]*1.15)

# ═══ 5. O7度优化 ═══
print('\n═══ 5. O7度优化 ═══')
J2=np.zeros_like(J)
for i in range(N):
    row=J[i].copy(); row[i]=0
    if (row>0).sum()==0: continue
    k=min(25,(row>0).sum()); th=np.sort(row[row>0])[-k] if k>0 else 0
    J2[i,row>=th]=row[row>=th]
e2=int(np.count_nonzero(J2))
print(f'{edges0}→{e2}边, 稀疏{1-e2/(N*N):.2%}')

# ═══ 6. 推理验证 ═══
print(f'\n═══ 6. 推理验证 ═══')
omega=1.0+0.02*np.random.randn(N)
def ks(p,o,J,K=0.5,dt=0.03):
    pd=p[None,:]-p[:,None]; return (p+dt*(o+K*(J*np.sin(pd)).sum(axis=1)))%(2*np.pi)

def decode(p,J,pert):
    act=np.zeros(N)
    for pi in pert:
        act[pi]=1.0; v={pi}; q=[(pi,1.0,0)]
        while q:
            n2,vv,d=q.pop(0)
            if d>=4: continue
            for j in range(N):
                if J[n2,j]>0.01 and j not in v:
                    v.add(j); al=np.cos(p[j]-p[n2])
                    if al>0.2: act[j]=max(act[j],vv*J[n2,j]*al*0.6); q.append((j,act[j],d+1))
    ps=set(pert)
    return [(vocab[i],round(float(act[i]),3)) for i in range(N) if i not in ps and act[i]>0.01]

# 收敛测试
p0=np.random.uniform(0,2*np.pi,N)
t1=time.time()
for _ in range(150): p0=ks(p0,omega,J2)
pd=p0[None,:]-p0[:,None]
energy=(J2*(1-np.cos(pd))).sum()/2
print(f'收敛: {time.time()-t1:.1f}s, 势能={energy:.1f}')

# 推理
for sent in tests:
    ws=[w for w in jieba.cut(re.sub(r'[^\u4e00-\u9fff]','',sent)) if len(w.strip())>=2 and w in idx]
    ph=np.zeros(N); pert=[idx[w] for w in ws]
    if len(pert)<2: continue
    for i in pert: ph[i]=np.pi*0.9
    t1=time.time()
    for _ in range(130): ph=ks(ph,omega,J2)
    dt=time.time()-t1
    out=decode(ph,J2,pert)
    tops=' '.join(f'{w}({s:.2f})' for w,s in out[:6])
    print(f'  {sent[:30]} → {dt:.2f}s | {tops}')

# ═══ 7. 保存 ═══
out={'vocab':vocab,'J':J2.tolist(),'n':N,'edges':e2,
     'domains':{vocab[i]:int(labels[i]) for i in range(N)},
     'source':'BERT_800_O1O7_optimized'}
json.dump(out,open('H:/otherproject/深渊实验室/prototype/aembr_800_topology.json','w',encoding='utf-8'),ensure_ascii=False)

print(f'\n✅ aembr_800_topology.json ({N}节点{e2}边) | 总耗时 {time.time()-t_total:.0f}s')
