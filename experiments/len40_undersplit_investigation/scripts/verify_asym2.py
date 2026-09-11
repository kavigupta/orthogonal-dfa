"""Get an actual COLLAPSE run (delta=0.04, noise seed matched to structured_oracle_run
so it reproduces the 8-state collapse), then for each DFA state measure the DECISION
value the asymmetric thresholds actually act on -- the accept-RATE over a suffix family,
not the string's own label -- and compare to accept_thresh.  Shows whether the collapse
comes from a borderline state crossing accept_thresh when it is split out."""
import numpy as np, collections
from orthogonal_dfa.l_star.counterexample_synthesis import counterexample_driven_synthesis
from orthogonal_dfa.l_star.learn import build_pst
from orthogonal_dfa.l_star.structures import Oracle, _uniform_random
from orthogonal_dfa.superlanguage.sampler import SuperSampler
from orthogonal_dfa.superlanguage.vocabulary import KmerVocabulary

DELTA, SEED = 0.04, 1
def frame_state(s):
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
class SOracle(Oracle):
    def __init__(self, seed): self.s=seed
    @property
    def alphabet_size(self): return 5
    def membership_query(self, s):
        f0,f1,first=frame_state(s)
        p=0.42 if (f0 and f1) else 0.60
        p=min(max(p+DELTA*first,0.02),0.98)
        return _uniform_random(bytes(s), self.s) < p

vocab=KmerVocabulary(kmers=((3,0,2),(3,0,0),(3,2,0)),base_alphabet_size=4)
pst=build_pst(lambda _nm,s: SOracle(s), min_signal_strength=0.05, seed=SEED,
              sampler=SuperSampler(vocab,36))
pst.config.fnr_limit=0.10
for dfa,dt,ta,bd,_c in counterexample_driven_synthesis(pst, acc_threshold=0.98): break
oracle=SOracle(SEED)
print(f"boundary={pst.decision_boundary:.4f} margin={pst.evidence_margin:.4f} "
      f"accept_thresh={pst.accept_thresh:.4f} reject_thresh={pst.reject_thresh:.4f}")
print(f"DFA: {len(dfa.states)} states, non-accepting={set(dfa.states)-dfa.final_states}")

samp=SuperSampler(vocab,36); rng=np.random.default_rng(SEED+999)
ev=[list(samp.sample(rng,vocab.alphabet_size)) for _ in range(4000)]
def walk(w):
    s=dfa.initial_state
    for c in w: s=dfa.transitions[s][c]
    return s
def truep(w):
    f0,f1,first=frame_state(w); p=0.42 if (f0 and f1) else 0.60
    return min(max(p+DELTA*first,0.02),0.98)
call=np.array([1.0 if walk(w) in dfa.final_states else 0.0 for w in ev])
tgt=np.array([0.0 if (frame_state(w)[0] and frame_state(w)[1]) else 1.0 for w in ev])
print(f"accept-rate {call.mean():.3f}, agree-with-frame-rule {np.mean(call==tgt):.3f}")

# DECISION value per state: accept-rate over a suffix family (what thresholds act on)
SUFF=[list(samp.sample(np.random.default_rng(9000+i),vocab.alphabet_size)) for i in range(40)]
byst=collections.defaultdict(list)
for w in ev[:1500]:
    st=walk(w)
    dec=np.mean([1.0 if oracle.membership_query(w+v) else 0.0 for v in SUFF])
    byst[st].append((dec, truep(w)))
print("\nstate label   n    mean-DECISION(over-suffixes)  mean-own-p   vs accept_thresh")
for s in sorted(byst):
    arr=np.array(byst[s]); lab="ACCEPT" if s in dfa.final_states else "reject"
    dec=arr[:,0].mean(); own=arr[:,1].mean()
    rel = "ACC>thr" if dec>=pst.accept_thresh else ("REJ<thr" if dec<pst.reject_thresh else "IN-BAND")
    print(f"  {s:2d}  {lab}  {len(arr):4d}   {dec:.3f}                       {own:.3f}      {rel}")
