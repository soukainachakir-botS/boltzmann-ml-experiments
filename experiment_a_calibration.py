"""Experiment A: temperature scaling, calibration and entropy for three classifiers.
Usage: python experiment_a_calibration.py <dataset> <n_splits> <subset_size> <tag>
  digits : python experiment_a_calibration.py digits 100 0 digits
  fashion: python experiment_a_calibration.py fashion 30 10000 fashion
  cifar10: python experiment_a_calibration.py cifar10 5 0 cifar10   (official train/test partition)
Results are written to results/<tag>_per_split.csv and results/<tag>_curves.csv.
"""
import os
for _v in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'): os.environ.setdefault(_v,'1')  # single-threaded linear algebra: bit-for-bit reproducible results

import sys, time, gzip, numpy as np, pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_digits
GRID=[0.05,0.1,0.2,0.3,0.5,0.7,1,1.5,2,3,5,8,12,20,40]
def load(name):
    if name=='digits':
        d=load_digits(); return d.data.astype(float), d.target
    base='data/fashion-mnist/data/fashion/'
    def img(f):
        with gzip.open(base+f,'rb') as g: return np.frombuffer(g.read(),np.uint8,offset=16).reshape(-1,784)
    def lab(f):
        with gzip.open(base+f,'rb') as g: return np.frombuffer(g.read(),np.uint8,offset=8)
    X=np.vstack([img('train-images-idx3-ubyte.gz'),img('t10k-images-idx3-ubyte.gz')]).astype(float)
    y=np.concatenate([lab('train-labels-idx1-ubyte.gz'),lab('t10k-labels-idx1-ubyte.gz')])
    return X,y
def softmax(Z):
    return np.exp(Z-logsumexp(Z,axis=1,keepdims=True))
def probs_T(E,T): return softmax(-E/T)
def nll(p,y): return float(-np.mean(np.log(np.clip(p[np.arange(len(y)),y],1e-300,None))))
def nllE(E,T,y):
    L=-E/T; lp=L-logsumexp(L,axis=1,keepdims=True)
    return float(-np.mean(lp[np.arange(len(y)),y]))
def ece(p,y,nb=15):
    conf=p.max(1); pred=p.argmax(1); acc=(pred==y).astype(float)
    b=np.minimum((conf*nb).astype(int),nb-1); e=0.0
    for k in range(nb):
        m=b==k
        if m.any(): e+=m.mean()*abs(acc[m].mean()-conf[m].mean())
    return e
def entropy(p): return float(np.mean(-np.sum(p*np.log(np.clip(p,1e-300,None)),1)))
CFG={'C':1.0,'epochs':100,'wd':0.0}
def mlp_logits(Xtr,ytr,Xs,seed,epochs=100,bs=32,lr=1e-3):
    epochs=CFG['epochs']; wd=CFG['wd']
    rng=np.random.default_rng(seed); sizes=[Xtr.shape[1],64,32,10]
    W=[rng.normal(0,np.sqrt(2/sizes[i]),(sizes[i],sizes[i+1])) for i in range(3)]; b=[np.zeros(sizes[i+1]) for i in range(3)]
    mW=[np.zeros_like(w) for w in W]; vW=[np.zeros_like(w) for w in W]; mb=[np.zeros_like(x) for x in b]; vb=[np.zeros_like(x) for x in b]
    t=0; n=len(Xtr); Y=np.eye(10)[ytr]
    for ep in range(epochs):
        idx=rng.permutation(n)
        for s in range(0,n,bs):
            ii=idx[s:s+bs]; x=Xtr[ii]; y=Y[ii]
            a1=np.maximum(x@W[0]+b[0],0); a2=np.maximum(a1@W[1]+b[1],0); z=a2@W[2]+b[2]
            d=(softmax(z)-y)/len(ii)
            gW2=a2.T@d+wd*W[2]; gb2=d.sum(0); d2=(d@W[2].T)*(a2>0)
            gW1=a1.T@d2+wd*W[1]; gb1=d2.sum(0); d1=(d2@W[1].T)*(a1>0)
            gW0=x.T@d1+wd*W[0]; gb0=d1.sum(0)
            t+=1
            for i,(gw,gb) in enumerate([(gW0,gb0),(gW1,gb1),(gW2,gb2)]):
                for P,g,m,v in ((W,gw,mW,vW),(b,gb,mb,vb)):
                    m[i]=0.9*m[i]+0.1*g; v[i]=0.999*v[i]+0.001*g*g
                    P[i]-=lr*(m[i]/(1-0.9**t))/(np.sqrt(v[i]/(1-0.999**t))+1e-8)
    def f(X): return np.maximum(np.maximum(X@W[0]+b[0],0)@W[1]+b[1],0)@W[2]+b[2]
    return [f(X) for X in Xs]
OFFICIAL=None
def run_split(X,y,s,nsub):
    if OFFICIAL is not None:
        # official CIFAR-10 partition: test = the official 10,000 test images (fixed);
        # validation = a stratified 5,000-image part of the official training set (changes with the seed)
        Xa,ya,Xte,yte=OFFICIAL
        Xtr,Xva,ytr,yva=train_test_split(Xa,ya,test_size=5000,stratify=ya,random_state=s)
    else:
      if nsub:
        X,_,y,_=train_test_split(X,y,train_size=nsub,stratify=y,random_state=s)
      Xtr,Xr,ytr,yr=train_test_split(X,y,test_size=0.4,stratify=y,random_state=s)
      Xva,Xte,yva,yte=train_test_split(Xr,yr,test_size=0.5,stratify=yr,random_state=s)
    mu=Xtr.mean(0); sd=Xtr.std(0); sd[sd==0]=1
    Xtr,Xva,Xte=[(a-mu)/sd for a in (Xtr,Xva,Xte)]
    out={}
    lr=LogisticRegression(C=CFG['C'],max_iter=5000).fit(Xtr,ytr)
    out['LR']=(-lr.decision_function(Xva),-lr.decision_function(Xte))
    zv,zt=mlp_logits(Xtr,ytr,[Xva,Xte],s); out['MLP']=(-zv,-zt)
    rf=RandomForestClassifier(n_estimators=200,random_state=s,n_jobs=2).fit(Xtr,ytr)
    def rfE(X):
        p=np.maximum(rf.predict_proba(X),1e-4); p/=p.sum(1,keepdims=True); return -np.log(p)
    out['RF']=(rfE(Xva),rfE(Xte))
    rows=[];curves=[]
    for m,(Ev,Et) in out.items():
        res=minimize_scalar(lambda lt:nllE(Ev,np.exp(lt),yva),bounds=(np.log(0.01),np.log(100)),method='bounded')
        Ts=float(np.exp(res.x)); p1=probs_T(Et,1.0); ps=probs_T(Et,Ts)
        rows.append(dict(split=s,model=m,Tstar=Ts,acc=float((p1.argmax(1)==yte).mean()),ece1=ece(p1,yte),eceT=ece(ps,yte),
            nll1=nllE(Et,1.0,yte),nllT=nllE(Et,Ts,yte),conf1=float(p1.max(1).mean())))
        for T in GRID:
            pv=probs_T(Ev,T); pt=probs_T(Et,T)
            curves.append(dict(split=s,model=m,T=T,test_acc=float((pt.argmax(1)==yte).mean()),test_entropy=entropy(pt),val_nll=nllE(Ev,T,yva),test_ece=ece(pt,yte)))
    return rows,curves
if __name__=='__main__':
    name,ns,nsub,tag=sys.argv[1],int(sys.argv[2]),int(sys.argv[3]),sys.argv[4]
    if name=='cifar10': CFG.update({'C':0.01,'epochs':30,'wd':5e-4})
    if name=='cifar10':
        d=np.load('data/cifar10_official.npz'); OFFICIAL=(d['Xtr'].astype(float),d['ytr'],d['Xte'].astype(float),d['yte']); X=y=None
    else:
        X,y=load(name)
    R=[];C=[]
    for s in range(ns):
        t0=time.time(); r,c=run_split(X,y,s,nsub); R+=r; C+=c
        pd.DataFrame(R).to_csv(f'results/{tag}_per_split.csv',index=False); pd.DataFrame(C).to_csv(f'results/{tag}_curves.csv',index=False)
        print(s,round(time.time()-t0,1),[(x['model'],round(x['Tstar'],2)) for x in r],flush=True)
