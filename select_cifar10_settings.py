"""Selection of the CIFAR-10 settings on validation data only (the official test set is never read here).
The official training set is split once (seed 100, not used by any reported run) into 45,000 training and
5,000 validation images. Criterion: validation accuracy.
  logistic regression: C in {0.00001, 0.0001, 0.001, 0.01, 0.1}
  MLP: weight decay in {0, 5e-4, 5e-3} x epochs in {10, 20, 30}
The random forest has no tuned setting (200 trees, as for the other datasets).
Usage: python select_cifar10_settings.py     Output: results/cifar10_selection.csv
"""
import os
for _v in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'): os.environ.setdefault(_v,'1')
import time, numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import experiment_a_calibration as E
SEED=100
d=np.load('data/cifar10_official.npz')          # only the training part is used
Xtr,Xva,ytr,yva=train_test_split(d['Xtr'].astype(float),d['ytr'],test_size=5000,stratify=d['ytr'],random_state=SEED)
mu=Xtr.mean(0); sd=Xtr.std(0); sd[sd==0]=1; Xtr=(Xtr-mu)/sd; Xva=(Xva-mu)/sd
rows=[]
def save(): pd.DataFrame(rows).to_csv('results/cifar10_selection.csv',index=False)
for wd in (0.0,5e-4,5e-3):
    t0=time.time(); E.CFG.update({'epochs':30,'wd':wd})
    def cb(ep,f,wd=wd):
        if ep in (10,20,30):
            z=f(Xva); rows.append(dict(model='MLP',setting=f'wd={wd:g}, epochs={ep}',val_acc=float((z.argmax(1)==yva).mean()),val_nll=E.nllE(-z,1.0,yva))); save()
    E.CFG['on_epoch']=cb; E.mlp_logits(Xtr,ytr,[],SEED); E.CFG['on_epoch']=None
    print('MLP wd',wd,round(time.time()-t0),rows[-3:],flush=True)
for C in (1e-5,1e-4,0.001,0.01,0.1):
    t0=time.time(); lr=LogisticRegression(C=C,max_iter=5000).fit(Xtr,ytr); z=lr.decision_function(Xva)
    rows.append(dict(model='LR',setting=f'C={C:g}',val_acc=float((z.argmax(1)==yva).mean()),val_nll=E.nllE(-z,1.0,yva))); save()
    print('LR C',C,round(time.time()-t0),rows[-1],flush=True)
