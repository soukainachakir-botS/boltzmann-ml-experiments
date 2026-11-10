import os
for _v in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'): os.environ.setdefault(_v,'1')  # single-threaded linear algebra: bit-for-bit reproducible results
# Experiment B at N=60, everything from one code base:
# final energies (4,000 steps), mean trajectories, 16,000-step check, and a reference low energy per matrix.
import sys, time, numpy as np
from experiment_b_annealing import schedules
def make_J(N,seed):
    rng=np.random.default_rng(seed)
    A=rng.normal(size=(N,N)); J=(A+A.T)/(2*np.sqrt(N)); np.fill_diagonal(J,0); return J,rng
def anneal(J,rng,Tm,sched,R,steps,rec=None):
    N=J.shape[0]; S=rng.choice([-1.0,1.0],size=(R,N)); ar=np.arange(R); traj=[]
    for t in range(steps):
        if rec and t%rec==0: traj.append(-0.5*np.einsum('rn,nm,rm->r',S,J,S))
        i=rng.integers(N,size=R); h=np.einsum('rn,rn->r',J[i],S); si=S[ar,i]; dE=2*si*h
        T=Tm[sched,t] if Tm.ndim==2 else Tm[t]
        acc=(dE<=0)|(rng.random(R)<np.exp(-np.maximum(dE,0)/T)); S[ar,i]=np.where(acc,-si,si)
    E=-0.5*np.einsum('rn,nm,rm->r',S,J,S)
    if rec: traj.append(E)
    return S,E,(np.array(traj) if rec else None)
def greedy(J,S):
    S=S.copy()
    while True:
        dE=2*S*(S@J)            # energy change of flipping each spin
        k=dE.argmin(1); m=dE[np.arange(len(S)),k]<-1e-12
        if not m.any(): break
        S[np.where(m)[0],k[m]]*=-1
    return -0.5*np.einsum('rn,nm,rm->r',S,J,S)
if __name__=='__main__':
    N=60; M=50; trials=40; names=['quench','fast','slow','log']
    sched=np.repeat(np.arange(4),trials)
    F4=np.zeros((M,4,trials)); F16=np.zeros((M,4,trials)); TR=None; REF=np.zeros(M); t0=time.time()
    for m in range(M):
        J,rng=make_J(N,1000*N+m)   # same matrices as expB.py
        for steps,F in ((4000,F4),(16000,F16)):
            Ts=schedules(np.arange(steps).astype(float)); Tm=np.stack([Ts[n] for n in names])
            _,E,tr=anneal(J,rng,Tm,sched,4*trials,steps,rec=20 if steps==4000 else None)
            F[m]=E.reshape(4,trials)
            if tr is not None:
                tr=tr.reshape(len(tr),4,trials).mean(2); TR=tr if TR is None else TR+tr
        # reference: 200 long slow anneals (20,000 steps, T=2*0.9997^t, floor 1e-3) + greedy descent; best value
        Tl=np.maximum(2*0.9997**np.arange(20000),1e-3)
        S,_,_=anneal(J,rng,Tl,None,200,20000); REF[m]=greedy(J,S).min()
        print(m,round(time.time()-t0),flush=True)
    np.savez("results/b60_full.npz",F4=F4,F16=F16,traj=TR/M,ref=REF,rec=20)
