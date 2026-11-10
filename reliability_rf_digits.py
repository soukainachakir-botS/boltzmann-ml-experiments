"""Digits, random forest: pooled reliability diagram (Fig. 12), pooled ECE, and sensitivity of T* to the floor
used for zero vote fractions. Uses the same splits, models and seeds as experiment_a_calibration.py.
Usage: python reliability_rf_digits.py   (about 5 minutes on two cores)"""
import os
for _v in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'): os.environ.setdefault(_v,'1')
import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from experiment_a_calibration import probs_T, nllE, ece
X,y=load_digits(return_X_y=True); X=X.astype(float)
FLOORS=(1e-6,1e-4,1e-2); P1=[];PT=[];Y=[];TS={f:[] for f in FLOORS}
for s in range(100):
    Xtr,Xr,ytr,yr=train_test_split(X,y,test_size=0.4,stratify=y,random_state=s)
    Xva,Xte,yva,yte=train_test_split(Xr,yr,test_size=0.5,stratify=yr,random_state=s)
    mu=Xtr.mean(0); sd=Xtr.std(0); sd[sd==0]=1; Xtr,Xva,Xte=[(a-mu)/sd for a in (Xtr,Xva,Xte)]
    rf=RandomForestClassifier(n_estimators=200,random_state=s,n_jobs=2).fit(Xtr,ytr)
    pv0,pt0=rf.predict_proba(Xva),rf.predict_proba(Xte)
    for f in FLOORS:
        def E(p): p=np.maximum(p,f); p=p/p.sum(1,keepdims=True); return -np.log(p)
        Ev,Et=E(pv0),E(pt0)
        T=float(np.exp(minimize_scalar(lambda lt:nllE(Ev,np.exp(lt),yva),bounds=(np.log(0.01),np.log(100)),method='bounded').x))
        TS[f].append(T)
        if f==1e-4: P1.append(probs_T(Et,1.0)); PT.append(probs_T(Et,T)); Y.append(yte)
P1=np.vstack(P1);PT=np.vstack(PT);Y=np.concatenate(Y)
print(f'{len(Y)} pooled test predictions; pooled ECE: T=1 {ece(P1,Y):.3f}, T=T* per split {ece(PT,Y):.3f}')
for f in FLOORS: print(f'floor {f:g}: geometric mean T* = {np.exp(np.mean(np.log(TS[f]))):.3f}')
for name,P in (('raw T=1',P1),('calibrated T*',PT)):
    conf=P.max(1); acc=(P.argmax(1)==Y).astype(float); b=np.minimum((conf*15).astype(int),14)
    pts=[(conf[b==k].mean(),acc[b==k].mean()) for k in range(15) if (b==k).sum()>=20]
    print(name,'bins (mean confidence, accuracy), bins with >=20 predictions:',' '.join(f'({c:.3f},{a:.3f})' for c,a in pts))
