"""Watch what later rounds do to the under-split collapse: run several rounds of
counterexample_driven_synthesis on the structured oracle (a seed that collapses to
accept-all in round 0) and print per-round states / phi / accept-rate / pool size /
indecisive-harvest size, to see whether the multi-round feedback improves the split."""
import argparse, numpy as np
from orthogonal_dfa.l_star.counterexample_synthesis import counterexample_driven_synthesis
from orthogonal_dfa.l_star.learn import build_pst
from orthogonal_dfa.l_star.structures import Oracle, _uniform_random
from orthogonal_dfa.superlanguage.sampler import SuperSampler
from orthogonal_dfa.superlanguage.vocabulary import KmerVocabulary

SIGNAL, DELTA, LENGTH = 0.30, 0.10, 36
def fstate(s):
    wc=0; f0=f1=False; first=0
    for c in s:
        if c>=3: wc+=1
        else:
            ph=wc%3
            if ph==0:
                if not f0 and not f1 and first==0: first=+1
                f0=True
            elif ph==1:
                if not f0 and not f1 and first==0: first=-1
                f1=True
    return f0,f1,first
class SO(Oracle):
    def __init__(self,seed): self.s=seed
    @property
    def alphabet_size(self): return 5
    def membership_query(self,x):
        f0,f1,first=fstate(x); p=(0.5-SIGNAL) if (f0 and f1) else (0.5+SIGNAL)
        p=min(max(p+DELTA*first,0.02),0.98); return _uniform_random(bytes(x),self.s)<p

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--seed",type=int,default=1)
    ap.add_argument("--rounds",type=int,default=6); args=ap.parse_args()
    vocab=KmerVocabulary(kmers=((3,0,2),(3,0,0),(3,2,0)),base_alphabet_size=4)
    pst=build_pst(lambda _n,s: SO(s), min_signal_strength=0.20, seed=args.seed,
                  sampler=SuperSampler(vocab,LENGTH))
    pst.config.fnr_limit=0.10
    samp=SuperSampler(vocab,LENGTH); rng=np.random.default_rng(args.seed+999)
    ev=[list(samp.sample(rng,vocab.alphabet_size)) for _ in range(2000)]
    tgt=np.array([0.0 if (fstate(w)[0] and fstate(w)[1]) else 1.0 for w in ev])
    def phi(a,b):
        a,b=np.asarray(a,float),np.asarray(b,float)
        return 0.0 if a.std()==0 or b.std()==0 else float(np.corrcoef(a,b)[0,1])
    print(f"seed {args.seed}: round | states | accept | phi(frame) | pool_prefixes | boundary")
    for i,(dfa,dt,ta,bd,cls) in enumerate(counterexample_driven_synthesis(pst, acc_threshold=0.98)):
        call=np.array([bool(dfa.accepts_input(w)) for w in ev],float)
        print(f"  round {i}: {len(dfa.states):2d} st | {call.mean():.3f} | {phi(call,tgt):+.3f} "
              f"| pool={pst.num_prefixes} | bd={pst.decision_boundary:.3f}", flush=True)
        if i+1>=args.rounds: break

if __name__=="__main__": main()
