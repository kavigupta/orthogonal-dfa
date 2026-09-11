"""Isolate the minimal decisive-classifier change that fixes the collapse.
Variant is a subset of {R,P,D}: R=routing (sift_and_boundary), P=population
(_classify), D=split-proposal (disagreement) use the decisive `side` instead of
the banded is_accept.  RPD == the known-working global monkeypatch."""
import argparse, numpy as np
from orthogonal_dfa.l_star import sifting, transition_resolver

def _apply(variant):
    if "R" in variant:
        def sab(self, seq):
            return self.tree.sift(seq, self.family.side)
        sifting.Sifter.sift_and_boundary = sab
    if "P" in variant:
        def classify(self, strings, midfix):
            self.family.prefill([list(s) + list(midfix) for s in strings])
            return [self.family.side(s, midfix) for s in strings]
        transition_resolver.TransitionResolver._classify = classify
    if "D" in variant:
        def disagreement(self, s, sprime, prefix):
            return self.tree.first_disagreement(s, sprime, self.family.side, prefix)
        sifting.Sifter.disagreement = disagreement

from orthogonal_dfa.l_star.counterexample_synthesis import counterexample_driven_synthesis
from orthogonal_dfa.l_star.learn import build_pst
from orthogonal_dfa.l_star.structures import Oracle, _uniform_random
from orthogonal_dfa.superlanguage.sampler import SuperSampler
from orthogonal_dfa.superlanguage.vocabulary import KmerVocabulary

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
    def __init__(self,sig,delta,seed): self.a=0.5+sig; self.r=0.5-sig; self.d=delta; self.s=seed
    @property
    def alphabet_size(self): return 5
    def membership_query(self,x):
        f0,f1,first=fstate(x); p=self.r if (f0 and f1) else self.a
        p=min(max(p+self.d*first,0.02),0.98); return _uniform_random(bytes(x),self.s)<p

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--variant", default="")
    ap.add_argument("--seed", type=int, default=0)
    args=ap.parse_args()
    _apply(args.variant)
    vocab=KmerVocabulary(kmers=((3,0,2),(3,0,0),(3,2,0)),base_alphabet_size=4)
    pst=build_pst(lambda _nm,s: SO(0.30,0.10,s), min_signal_strength=0.20,
                  seed=args.seed, sampler=SuperSampler(vocab,36))
    pst.config.fnr_limit=0.10
    samp=SuperSampler(vocab,36); rng=np.random.default_rng(args.seed+999)
    ev=[list(samp.sample(rng,vocab.alphabet_size)) for _ in range(3000)]
    tgt=np.array([0.0 if (fstate(w)[0] and fstate(w)[1]) else 1.0 for w in ev])
    def phi(a,b):
        a,b=np.asarray(a,float),np.asarray(b,float)
        return 0.0 if a.std()==0 or b.std()==0 else float(np.corrcoef(a,b)[0,1])
    for dfa,dt,ta,bd,_c in counterexample_driven_synthesis(pst, acc_threshold=0.88):
        call=np.array([bool(dfa.accepts_input(w)) for w in ev],float)
        print(f"[variant='{args.variant}' seed={args.seed}] {len(dfa.states)} states, "
              f"accept {call.mean():.3f}, phi {phi(call,tgt):+.3f}", flush=True); break

if __name__=="__main__": main()
