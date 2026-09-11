"""Verify the corrected mechanism: after a delta=0.04 collapse run, is the
acceptance threshold (boundary+margin) BETWEEN the low substate rate (~0.56) and
the accept-region average (~0.60), and are the reject-labeled states exactly the
low-rate substates?  Uses the structured (super-string) oracle for speed."""
import numpy as np, collections
from orthogonal_dfa.l_star.counterexample_synthesis import counterexample_driven_synthesis
from orthogonal_dfa.l_star.learn import build_pst
from orthogonal_dfa.l_star.structures import Oracle, _uniform_random
from orthogonal_dfa.superlanguage.sampler import SuperSampler
from orthogonal_dfa.superlanguage.vocabulary import KmerVocabulary

def frame_state(s):
    wc = 0; f0 = f1 = False; first = 0
    for c in s:
        if c >= 3: wc += 1
        else:
            ph = wc % 3
            if ph == 0:
                if not f0 and not f1 and first == 0: first = +1
                f0 = True
            elif ph == 1:
                if not f0 and not f1 and first == 0: first = -1
                f1 = True
    return f0, f1, first

DELTA = 0.04
class StructuredOracle(Oracle):
    @property
    def alphabet_size(self): return 5
    def membership_query(self, s):
        f0, f1, first = frame_state(s)
        p = 0.42 if (f0 and f1) else 0.60
        p = min(max(p + DELTA * first, 0.02), 0.98)
        return _uniform_random(bytes(s), 0) < p

vocab = KmerVocabulary(kmers=((3,0,2),(3,0,0),(3,2,0)), base_alphabet_size=4)
SEED = 1
pst = build_pst(lambda _nm, s: StructuredOracle(), min_signal_strength=0.05, seed=SEED,
                sampler=SuperSampler(vocab, 36))
pst.config.fnr_limit = 0.10
dfa = dt = None
for dfa, dt, ta, bd, _c in counterexample_driven_synthesis(pst, acc_threshold=0.98):
    break
print(f"decision_boundary={pst.decision_boundary:.4f} evidence_margin={pst.evidence_margin:.4f}")
print(f"accept_thresh={pst.accept_thresh:.4f} reject_thresh={pst.reject_thresh:.4f}")
print(f"DFA: {len(dfa.states)} states, non-accepting={set(dfa.states)-dfa.final_states}")

# per DFA-state: true accept-PROBABILITY of its strings (avg p, not the noisy label),
# and the state's accept label
samp = SuperSampler(vocab, 36); rng = np.random.default_rng(SEED + 999)
ev = [list(samp.sample(rng, vocab.alphabet_size)) for _ in range(8000)]
def walk(w):
    s = dfa.initial_state
    for c in w: s = dfa.transitions[s][c]
    return s
def truep(w):
    f0, f1, first = frame_state(w)
    p = 0.42 if (f0 and f1) else 0.60
    return min(max(p + DELTA * first, 0.02), 0.98)
byst = collections.defaultdict(list)
for w in ev: byst[walk(w)].append(truep(w))
print("\nstate  label   n     mean-true-accept-prob   (vs accept_thresh)")
for s in sorted(byst):
    arr = np.array(byst[s]); lab = "ACCEPT" if s in dfa.final_states else "reject"
    flag = "  <-- reject-labeled but prob>%.3f?" % pst.accept_thresh if (lab=="reject" and arr.mean()>pst.accept_thresh) else ""
    below = "  <-- below accept_thresh" if arr.mean() < pst.accept_thresh else ""
    print(f"  {s:2d}   {lab}  {len(arr):5d}   {arr.mean():.3f}{below}{flag}")
