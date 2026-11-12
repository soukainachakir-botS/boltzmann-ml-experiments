"""Prints every number reported in Section IX from the files in results/."""
import numpy as np, pandas as pd
from scipy import stats

def holm(ps):
    ps=np.asarray(ps); o=np.argsort(ps); adj=np.empty(len(ps)); run=0
    for k,i in enumerate(o): run=max(run,min(1,ps[i]*(len(ps)-k))); adj[i]=run
    return adj

def experiment_a(tag, ratio):
    """ratio = n_heldout/n_train used in the Nadeau-Bengio correction (None: plain t-interval)."""
    d=pd.read_csv(f'results/{tag}_per_split.csv'); J=d.split.nunique()
    f=np.sqrt(1/J+(ratio or 0)); tq=stats.t.ppf(0.975,J-1)
    print(f'\n=== Experiment A: {tag} ({J} runs; correction factor {"none" if not ratio else f"1/J+{ratio:.3f}"})')
    for m,g in d.groupby('model'):
        l=np.log(g.Tstar); se=l.std(ddof=1)*f; x=g.ece1-g.eceT; sx=x.std(ddof=1)*f
        if not ratio:   # official CIFAR-10 partition: fixed test set, report mean and range over runs
            print(f'{m:4s} acc {g.acc.mean():.3f} | T* {np.exp(l.mean()):.2f} (range {g.Tstar.min():.2f}-{g.Tstar.max():.2f}) | '
                  f'ECE {g.ece1.mean():.3f}->{g.eceT.mean():.3f} improved {(x>0).sum()}/{J} | NLL {g.nll1.mean():.3f}->{g.nllT.mean():.3f}'); continue
        print(f'{m:4s} acc {g.acc.mean():.3f} conf(T=1) {g.conf1.mean():.3f} | T* {np.exp(l.mean()):.2f} '
              f'[{np.exp(l.mean()-tq*se):.2f},{np.exp(l.mean()+tq*se):.2f}] T*<1 in {100*(g.Tstar<1).mean():.0f}% | '
              f'ECE {g.ece1.mean():.3f}->{g.eceT.mean():.3f} change {-x.mean():+.3f} [{-x.mean()-tq*sx:+.3f},{-x.mean()+tq*sx:+.3f}] '
              f'improved {(x>0).sum()}/{J} | NLL {g.nll1.mean():.3f}->{g.nllT.mean():.3f}')
    p=d.pivot(index='split',columns='model',values='Tstar')
    if not ratio:
        print(f'RF lowest in {((p.RF<p.LR)&(p.RF<p.MLP)).sum()}/{J} runs; LR<MLP in {(p.LR<p.MLP).sum()}/{J} runs')
        q=pd.read_csv('results/cifar10_lr_sensitivity.csv').set_index('split').Tstar
        print(f'sensitivity, LR with C=0.01: T* {np.exp(np.log(q).mean()):.2f} (range {q.min():.2f}-{q.max():.2f}); below MLP in {(q<p.MLP).sum()}/{J} runs')
        sel=pd.read_csv('results/cifar10_selection.csv')
        for m in ('LR','MLP'):
            r=sel[sel.model==m]; r=r.loc[r.val_acc.idxmax()]; print(f'selected on validation data, {m}: {r.setting} (validation accuracy {r.val_acc:.3f})')
        return
    order=((p.RF<p.LR)&(p.LR<p.MLP)).sum(); ps=[]
    for a,b in (('RF','LR'),('LR','MLP'),('RF','MLP')):
        x=np.log(p[a])-np.log(p[b]); ps.append(2*stats.t.sf(abs(x.mean()/(x.std(ddof=1)*f)),J-1))
    print(f'order RF<LR<MLP in {order}/{J}; Holm-adjusted p (RF-LR, LR-MLP, RF-MLP):',['%.1e'%v for v in holm(ps)])

def experiment_b_n60():
    d=np.load('results/b60_full.npz'); F4=d['F4']; M=F4.shape[0]; pm=F4.mean(2); names=['quench','fast','slow','log']
    print('\n=== Experiment B, N=60 (4,000 steps)')
    for k,n in enumerate(names):
        x=pm[:,k]; h=stats.t.ppf(.975,M-1)*x.std(ddof=1)/np.sqrt(M)
        print(f'{n:6s} mean {x.mean():.2f} [{x.mean()-h:.2f},{x.mean()+h:.2f}] SD {x.std(ddof=1):.2f} gap to reference {x.mean()-d["ref"].mean():.2f}')
    print(f'reference low energy (best of 200 long anneals + greedy descent): mean {d["ref"].mean():.2f}')
    pairs(pm,names,M)
    p16=d['F16'].mean(2)
    for a,b in (('slow','log'),('quench','fast')):
        x=p16[:,names.index(a)]-p16[:,names.index(b)]; h=stats.t.ppf(.975,M-1)*x.std(ddof=1)/np.sqrt(M)
        print(f'16,000 steps: {a}-{b} {x.mean():+.2f} [{x.mean()-h:+.2f},{x.mean()+h:+.2f}], {a} lower on {(x<0).sum()}/{M}')
    tr=d['traj']; print(f"fast schedule mean energy: step 600 {tr[30,1]:.2f}, step 4000 {tr[-1,1]:.2f}")
    c=sum(1 for m in range(M) if F4[m,1].mean()<F4[m,0].mean() and stats.ttest_ind(F4[m,1],F4[m,0],equal_var=False).pvalue<0.05)
    print(f'matrices where fast is significantly lower than quench (Welch, p<0.05): {c}')

def pairs(pm,names,M):
    rows=[]
    for a,b in (('slow','fast'),('log','fast'),('slow','quench'),('quench','fast'),('log','quench'),('slow','log')):
        x=pm[:,names.index(a)]-pm[:,names.index(b)]; h=stats.t.ppf(.975,M-1)*x.std(ddof=1)/np.sqrt(M)
        rows.append((a,b,x.mean(),x.mean()-h,x.mean()+h,x.mean()/x.std(ddof=1),stats.ttest_1samp(x,0).pvalue,(x<0).sum()))
    adj=holm([r[6] for r in rows])
    for r,a in zip(rows,adj): print(f'{r[0]}-{r[1]}: {r[2]:+.2f} [{r[3]:+.2f},{r[4]:+.2f}] d_z {r[5]:.2f} Holm p {a:.1e} lower on {r[7]}/{M}')

def experiment_b_large(N):
    F=np.load(f'results/b{N}_final.npy'); M=F.shape[0]; pm=F.mean(2); names=['quench','fast','slow','log']
    print(f'\n=== Experiment B, N={N}'); print('mean energy per spin:',' '.join(f'{n} {pm[:,k].mean()/N:.3f}' for k,n in enumerate(names)))
    pairs(pm,names,M)
    c=sum(1 for m in range(M) if F[m,1].mean()<F[m,0].mean() and stats.ttest_ind(F[m,1],F[m,0],equal_var=False).pvalue<0.05)
    print(f'matrices where fast is significantly lower than quench (Welch, p<0.05): {c}')

if __name__=='__main__':
    experiment_a('digits',0.2/0.6); experiment_a('fashion',0.2/0.6); experiment_a('cifar10',None)
    experiment_b_n60(); experiment_b_large(120); experiment_b_large(200)
