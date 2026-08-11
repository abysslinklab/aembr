#!/usr/bin/env python3
"""AEMBR 拓扑优化: O1领域聚类 + O7度优化"""
import json, numpy as np, jieba, re, time
from sklearn.cluster import AgglomerativeClustering
from collections import Counter
from transformers import AutoTokenizer, AutoModel

# Load
d = json.loads(open('H:/otherproject/深渊实验室/prototype/aembr_bert_topology.json',encoding='utf-8').read())
vocab, J_raw = d['vocab'], np.array(d['J'], dtype=np.float32)
N = len(vocab)
print(f'当前: {N}节点 {int(np.count_nonzero(J_raw))}边')

# ── O1: 领域聚类 ──
print('\n═══ O1: 领域聚类 ═══')
tok = AutoTokenizer.from_pretrained('bert-base-chinese')
model = AutoModel.from_pretrained('bert-base-chinese'); model.eval()
import torch
import torch
with torch.no_grad():
    emb = np.array([model(**tok(f'这是一个{w}。', return_tensors='pt')).last_hidden_state[0,0].detach().cpu().numpy() for w in vocab])
n_clusters = 15
labels = AgglomerativeClustering(n_clusters=n_clusters).fit_predict(emb)
for d_id in range(n_clusters):
    ws = [vocab[i] for i in range(N) if labels[i]==d_id][:8]
    print(f'  D{d_id:02d}: {" ".join(ws)}')

J = J_raw.copy()
for i in range(N):
    for j in range(i+1, N):
        if J[i,j]>0 and labels[i]==labels[j]:
            J[i,j]=min(1.0,J[i,j]*1.15)  # 同域轻微增强
# 不做跨域衰减——保留跨域桥接边的推理价值

# ── O7: 度优化 ──
print('\n═══ O7: 度优化 ═══')
base = __import__('pathlib').Path('H:/文澜阁')
freq, n = Counter(), 0
for d2 in sorted(base.iterdir()):
    if not d2.is_dir() or not d2.name.startswith('0'): continue
    for m in sorted((d2/'papers').glob('*_chinesereadme.md')):
        try:
            t=m.read_text(encoding='utf-8')[:3000]; parts=t.split('---',2)
            b=re.sub(r'[^\u4e00-\u9fff]',' ',parts[2] if len(parts)>=3 else t).strip()
            freq.update(w.strip() for w in jieba.cut(b) if len(w.strip())>=2)
            n+=1
            if n>=300: break
        except: pass
    if n>=300: break

max_f=max(freq.get(w,1) for w in vocab)
J2=np.zeros_like(J)
for i in range(N):
    rk=freq.get(vocab[i],1)/max_f; topk=max(8,int(10+rk*20))
    row=J[i].copy(); row[i]=0
    if (row>0).sum()==0: continue
    k=min(topk,(row>0).sum()); th=np.sort(row[row>0])[-k] if k>0 else 0
    J2[i,row>=th]=row[row>=th]
e2=int(np.count_nonzero(J2))
print(f'优化后: {e2}边 (稀疏{1-e2/(N*N):.2%})')

# ── 推理对比 ──
omega=1.0+0.02*np.random.randn(N)
def ks(p,o,J,K=0.5,dt=0.03):
    pd=p[None,:]-p[:,None]; return (p+dt*(o+K*(J*np.sin(pd)).sum(axis=1)))%(2*np.pi)
def decode(p,J,pert):
    act=np.zeros(N)
    for pi in pert:
        act[pi]=1.0; v={pi}; q=[(pi,1.0,0)]
        while q:
            n2,vv,d=q.pop(0)
            if d>=3: continue
            for j in range(N):
                if J[n2,j]>0.01 and j not in v:
                    v.add(j); al=np.cos(p[j]-p[n2])
                    if al>0.2: act[j]=max(act[j],vv*J[n2,j]*al*0.6); q.append((j,act[j],d+1))
    ps=set(pert)
    return [(vocab[i],act[i]) for i in range(N) if i not in ps and act[i]>0.01][:4]

idx={w:i for i,w in enumerate(vocab)}
tests=['分布式系统共识协议','密码学数学安全','神经网络深度学习','区块链去中心化','量子计算纠错','人工智能机器学习']

print('\n═══ 推理对比 ═══')
for sent in tests:
    ws=[w.strip() for w in jieba.cut(re.sub(r'[^\u4e00-\u9fff]','',sent)) if len(w.strip())>=2]
    ph=np.zeros(N); pert=[idx[w] for w in ws if w in idx]
    if len(pert)<2: continue
    for i in pert: ph[i]=np.pi*0.9
    p1=ph.copy()
    for _ in range(130): p1=ks(p1,omega,J_raw)
    old=[w for w,_ in decode(p1,J_raw,pert)]
    p2=ph.copy()
    for _ in range(130): p2=ks(p2,omega,J2)
    new=[w for w,_ in decode(p2,J2,pert)]
    old_s = " ".join(old)
    new_s = " ".join(new)
    print(f'  {sent[:30]}')
    print(f'    旧: {old_s}')
    print(f'    新: {new_s}')

# Save
out={'vocab':vocab,'J':J2.tolist(),'n':N,'edges':e2,'domains':{vocab[i]:int(labels[i]) for i in range(N)},'source':'BERT_O1O7_optimized'}
json.dump(out,open('H:/otherproject/深渊实验室/prototype/aembr_optimized_topology.json','w',encoding='utf-8'),ensure_ascii=False)
print(f'\n✅ aembr_optimized_topology.json ({e2}边)')
