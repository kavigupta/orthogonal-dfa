import pickle, numpy as np
from orthogonal_dfa.data.exon import default_exon
from orthogonal_dfa.l_star.examples.gate_composition_residual import gate_residual_oracle
from orthogonal_dfa.spliceai.load_model import load_spliceai
from orthogonal_dfa.superlanguage import KmerVocabulary, SuperSampler

STOPS = {(3, 0, 2), (3, 0, 0), (3, 2, 0)}
def frame_closed(seq, ph):
    sub = seq[ph:]
    return any(tuple(sub[i:i+3]) in STOPS for i in range(0, len(sub)-2, 3))

def phi(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return 0.0 if a.std() == 0 or b.std() == 0 else float(np.corrcoef(a, b)[0, 1])

base = gate_residual_oracle(default_exon, load_spliceai(400, 0), length=40, len_lo=35, len_hi=85)
v = KmerVocabulary(kmers=((3, 0, 2), (3, 0, 0), (3, 2, 0)), base_alphabet_size=4)
d = pickle.load(open("/tmp/claude-25787/-mnt-md0-orthogonal-dfa-3/e4e9621c-a36b-4b7e-a24d-819fbb2cab69/scratchpad/round_dumps/round_00.pkl", "rb"))
dfa = d["dfa"]
s = SuperSampler(v, 36); rng = np.random.default_rng(2024)
N = 8000
ws = [s.sample(rng, v.alphabet_size) for _ in range(N)]
bs = v.compile_many(ws, [np.random.default_rng(i) for i in range(N)])
ora = np.asarray(base.membership_queries(bs)).astype(float)
fc = np.array([[frame_closed(b, ph) for ph in range(3)] for b in bs], dtype=int)  # (N,3)
call = np.array([bool(dfa.accepts_input(w)) for w in ws]).astype(float)

print(f"eval N={N}, oracle accept-rate {ora.mean():.3f}\n")
print("predicate correlations with the oracle (|phi| = strength):")
preds = {
 "frame0 closed":            fc[:,0],
 "frame1 closed":            fc[:,1],
 "frame2 closed":            fc[:,2],
 "frames {0,1} both closed": fc[:,0]&fc[:,1],   # <- DFA's exact reject predicate
 "frames {0,2} both closed": fc[:,0]&fc[:,2],
 "frames {1,2} both closed": fc[:,1]&fc[:,2],
 "ALL frames closed":        fc[:,0]&fc[:,1]&fc[:,2],
 "count frames closed":      fc.sum(1),
}
for name, p in preds.items():
    print(f"  phi(oracle, {name:26s}) = {phi(ora, p):+.3f}")
print()
print(f"  phi(oracle, DFA-accept)                 = {phi(ora, call):+.3f}")
print(f"  phi(oracle, NOT(frames 0&1 closed))     = {phi(ora, 1-(fc[:,0]&fc[:,1])):+.3f}   <- DFA's rule as a predicate")
