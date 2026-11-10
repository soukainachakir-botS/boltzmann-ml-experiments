import os
for _v in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'): os.environ.setdefault(_v,'1')  # single-threaded linear algebra: bit-for-bit reproducible results
import sys, time, numpy as np, pandas as pd
from scipy import stats
def schedules(tt):
    # tt: effective step index (float array), returns dict of T arrays
    return {'quench':np.full_like(tt,0.05),
            'fast':np.maximum(2*0.985**tt,1e-3),
            'slow':np.maximum(2*0.9985**tt,1e-3),
            'log':np.maximum(2/np.log(2+tt),1e-3)}
def run_matrix(N,steps,trials,seed):
    rng=np.random.default_rng(seed)
    A=rng.normal(size=(N,N)); J=(A+A.T)/(2*np.sqrt(N)); np.fill_diagonal(J,0)
    names=['quench','fast','slow','log']; R=4*trials
    S=rng.choice([-1.0,1.0],size=(R,N)); sched=np.repeat(np.arange(4),trials)
    scale=60.0/N
    tt=np.arange(steps)*scale
    Ts=schedules(tt); Tm=np.stack([Ts[n] for n in names])  # (4,steps)
    ar=np.arange(R)
    for t in range(steps):
        i=rng.integers(N,size=R)
        h=np.einsum('rn,rn->r',J[i],S)
        si=S[ar,i]; dE=2*si*h
        T=Tm[sched,t]
        acc=(dE<=0)|(rng.random(R)<np.exp(-np.maximum(dE,0)/T))
        S[ar,i]=np.where(acc,-si,si)
    E=-0.5*np.einsum('rn,nm,rm->r',S,J,S)
    return E.reshape(4,trials)  # rows in names order
if __name__=='__main__':
    N,steps,M,trials,tag=int(sys.argv[1]),int(sys.argv[2]),int(sys.argv[3]),int(sys.argv[4]),sys.argv[5]
    allE=[];t0=time.time()
    for m in range(M):
        allE.append(run_matrix(N,steps,trials,1000*N+m))
        if m%10==0: print(m,round(time.time()-t0,1),flush=True)
    allE=np.array(allE)  # (M,4,trials)
    np.save(f'results/{tag}_final.npy',allE)
    names=['quench','fast','slow','log']; pm=allE.mean(2)
    print('N',N,'steps',steps)
    for k,n in enumerate(names):
        x=pm[:,k]; print(n,'mean %.2f CI [%.2f,%.2f] SD %.2f'%(x.mean(),x.mean()-1.96*x.std(ddof=1)/np.sqrt(M),x.mean()+1.96*x.std(ddof=1)/np.sqrt(M),x.std(ddof=1)),'per spin %.4f'%(x.mean()/N))
    pairs=[('slow','fast'),('log','fast'),('slow','quench'),('quench','fast'),('log','quench'),('slow','log')]
    rows=[]
    for a,b in pairs:
        d=pm[:,names.index(a)]-pm[:,names.index(b)]
        t,p=stats.ttest_1samp(d,0); rows.append([a,b,d.mean(),d.mean()-1.96*d.std(ddof=1)/np.sqrt(M),d.mean()+1.96*d.std(ddof=1)/np.sqrt(M),d.mean()/d.std(ddof=1),p,(d<0).sum()])
    order=np.argsort([r[6] for r in rows]); adj=[0]*6; run=0
    for rank,k in enumerate(order):
        run=max(run,min(1,rows[k][6]*(6-rank))); adj[k]=run
    for r,a in zip(rows,adj): print('%s-%s  d=%.2f [%.2f,%.2f] dz=%.2f p_holm=%.1e first-lower %d/%d'%(r[0],r[1],r[2],r[3],r[4],r[5],a,r[7],M))
    # fast vs quench within matrix welch
    cnt=0
    for m in range(M):
        t,p=stats.ttest_ind(allE[m,1],allE[m,0],equal_var=False)
        if allE[m,1].mean()<allE[m,0].mean() and p<0.05: cnt+=1
    print('matrices where fast significantly lower than quench:',cnt)
