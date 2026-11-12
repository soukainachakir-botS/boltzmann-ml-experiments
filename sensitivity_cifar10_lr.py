"""Sensitivity check: optimal temperature T* of logistic regression on CIFAR-10 for a weaker regularization
(C = 0.01 instead of the selected C = 0.001), same five runs as experiment_a_calibration.py.
T* is fitted on the validation set; the test set is not used. Output: results/cifar10_lr_sensitivity.csv
"""
import os
for _v in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'): os.environ.setdefault(_v,'1')
import numpy as np, pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from experiment_a_calibration import nllE
d=np.load('data/cifar10_official.npz'); Xa=d['Xtr'].astype(float); ya=d['ytr']; rows=[]
for s in range(5):
    Xtr,Xva,ytr,yva=train_test_split(Xa,ya,test_size=5000,stratify=ya,random_state=s)
    mu=Xtr.mean(0); sd=Xtr.std(0); sd[sd==0]=1; Xtr=(Xtr-mu)/sd; Xva=(Xva-mu)/sd
    Ev=-LogisticRegression(C=0.01,max_iter=5000).fit(Xtr,ytr).decision_function(Xva)
    T=float(np.exp(minimize_scalar(lambda lt:nllE(Ev,np.exp(lt),yva),bounds=(np.log(0.01),np.log(100)),method='bounded').x))
    rows.append(dict(split=s,C=0.01,Tstar=T,val_acc=float((Ev.argmin(1)==yva).mean()))); pd.DataFrame(rows).to_csv('results/cifar10_lr_sensitivity.csv',index=False); print(rows[-1],flush=True)
