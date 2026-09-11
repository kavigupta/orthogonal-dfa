import pickle, numpy as np
from orthogonal_dfa.data.exon import default_exon
from orthogonal_dfa.l_star.examples.gate_composition_residual import gate_residual_oracle
from orthogonal_dfa.spliceai.load_model import load_spliceai
from orthogonal_dfa.superlanguage import KmerVocabulary, SuperSampler

STOPS = {(3, 0, 2), (3, 0, 0), (3, 2, 0)}
def frame_closed(seq, ph):
    sub = seq[ph:]
    return any(tuple(sub[i:i+3]) in STOPS for i in range(0, len(sub)-2, 3))

def mi_bits(x, y):
    """Mutual information I(x;y) in bits for integer-labeled arrays."""
    x = np.asarray(x); y = np.asarray(y)
    xs = {v: i for i, v in enumerate(np.unique(x))}
    ys = {v: i for i, v in enumerate(np.unique(y))}
    joint = np.zeros((len(xs), len(ys)))
    for xi, yi in zip(x, y):
        joint[xs[xi], ys[yi]] += 1
    joint /= joint.sum()
    px = joint.sum(1, keepdims=True); py = joint.sum(0, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = joint * (np.log2(joint) - np.log2(px) - np.log2(py))
    return float(np.nansum(t))

base = gate_residual_oracle(default_exon, load_spliceai(400, 0), length=40, len_lo=35, len_hi=85)
v = KmerVocabulary(kmers=((3, 0, 2), (3, 0, 0), (3, 2, 0)), base_alphabet_size=4)
d = pickle.load(open("/tmp/claude-25787/-mnt-md0-orthogonal-dfa-3/e4e9621c-a36b-4b7e-a24d-819fbb2cab69/scratchpad/round_dumps/round_00.pkl", "rb"))
dfa = d["dfa"]
s = SuperSampler(v, 36); rng = np.random.default_rng(2024)
N = 20000
ws = [s.sample(rng, v.alphabet_size) for _ in range(N)]
bs = v.compile_many(ws, [np.random.default_rng(i) for i in range(N)])
ora = np.asarray(base.membership_queries(bs)).astype(int)
fc = np.array([[frame_closed(b, ph) for ph in range(3)] for b in bs], dtype=int)
call = np.array([bool(dfa.accepts_input(w)) for w in ws]).astype(int)

print(f"eval N={N}, oracle accept-rate {ora.mean():.3f}\n")
preds = {
 "frame0 closed":            fc[:,0],
 "frame1 closed":            fc[:,1],
 "frame2 closed":            fc[:,2],
 "frames {0,1} both closed": fc[:,0]&fc[:,1],
 "frames {0,2} both closed": fc[:,0]&fc[:,2],
 "frames {1,2} both closed": fc[:,1]&fc[:,2],
 "ALL frames closed":        fc[:,0]&fc[:,1]&fc[:,2],
 "count frames closed(0-3)": fc.sum(1),
 "full (f0,f1,f2) 8-way":    fc[:,0]*4 + fc[:,1]*2 + fc[:,2],
 "DFA-accept":               call,
}
print("mutual information with the oracle (bits):")
for name, p in preds.items():
    print(f"  I(oracle; {name:26s}) = {mi_bits(ora, p):.5f}")
