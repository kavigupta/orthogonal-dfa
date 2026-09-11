"""Why over-split->collapse instead of a clean larger DFA?  Compare two delta=0.10
runs that BOTH make 8 states but score wildly differently (seed 4: phi+1.0 crisp;
seed 0: phi+0.42 collapse).  Map each DFA's states to the true refined signature
(f0,f1,first,phase) and show whether it is a CONSISTENT refinement (each state one
coherent signature, correctly labeled) or a PARTIAL/INCONSISTENT one (a state mixes
signatures, or a clear-accept signature is labeled reject) -> misrouting/mislabel."""
import numpy as np, collections
from orthogonal_dfa.l_star.counterexample_synthesis import counterexample_driven_synthesis
from orthogonal_dfa.l_star.learn import build_pst
from orthogonal_dfa.l_star.structures import Oracle, _uniform_random
from orthogonal_dfa.superlanguage.sampler import SuperSampler
from orthogonal_dfa.superlanguage.vocabulary import KmerVocabulary

SIG, DELTA = 0.30, 0.10
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
    return (f0,f1,first,wc%3)
def frame_reject(s):
    f0,f1,_,_=fstate(s); return f0 and f1
class SO(Oracle):
    def __init__(self,seed): self.s=seed
    @property
    def alphabet_size(self): return 5
    def membership_query(self,x):
        f0,f1,first,_=fstate(x); p=(0.5-SIG) if (f0 and f1) else (0.5+SIG)
        p=min(max(p+DELTA*first,0.02),0.98); return _uniform_random(bytes(x),self.s)<p
vocab=KmerVocabulary(kmers=((3,0,2),(3,0,0),(3,2,0)),base_alphabet_size=4)

def run(seed):
    pst=build_pst(lambda _nm,s: SO(s), min_signal_strength=0.20, seed=seed,
                  sampler=SuperSampler(vocab,36))
    pst.config.fnr_limit=0.10
    for dfa,dt,ta,bd,_c in counterexample_driven_synthesis(pst, acc_threshold=0.88): break
    samp=SuperSampler(vocab,36); rng=np.random.default_rng(seed+999)
    ev=[list(samp.sample(rng,vocab.alphabet_size)) for _ in range(6000)]
    def walk(w):
        s=dfa.initial_state
        for c in w: s=dfa.transitions[s][c]
        return s
    byst=collections.defaultdict(collections.Counter)
    accfrac=collections.defaultdict(list)
    for w in ev:
        st=walk(w); byst[st][fstate(w)]+=1; accfrac[st].append(0.0 if frame_reject(w) else 1.0)
    print(f"\n### seed {seed}: {len(dfa.states)} states, non-accepting={set(dfa.states)-dfa.final_states}")
    print("  state  label   n    frame-accept-frac   #distinct-sigs (top sig share)")
    for st in sorted(byst):
        lab="ACC" if st in dfa.final_states else "REJ"
        n=sum(byst[st].values()); fa=np.mean(accfrac[st])
        top,topn=byst[st].most_common(1)[0]; nd=len(byst[st])
        flag=""
        if lab=="REJ" and fa>0.6: flag="  <-- rejects a mostly-ACCEPT state (mislabel)"
        if nd>2 and topn/n<0.7: flag+="  <-- mixes signatures (inconsistent routing)"
        print(f"   {st:2d}   {lab}  {n:5d}    {fa:.3f}             {nd:2d}  ({topn/n:.0%})  {top}{flag}")

for sd in (4, 0):   # 4 = crisp phi+1.0 ; 0 = collapse phi+0.42
    run(sd)
