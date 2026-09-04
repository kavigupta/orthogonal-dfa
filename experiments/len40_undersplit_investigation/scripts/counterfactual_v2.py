"""Do the GGTA/AGGT transitions help the round-0 v2 DFA correlate with the oracle,
or would treating them as wildcards be as good / better?

Same DFA, same eval strings, three readings:
  as-is        : GGTA/AGGT use their learned transitions (escape hatches to accept)
  ->XXXX       : each GGTA/AGGT symbol replaced by four wildcards (5,5,5,5) --
                 what parse would emit if the 4-mers were not in the vocabulary
  ->X (1 step) : each GGTA/AGGT replaced by a single wildcard (transition as X once)
"""
import pickle, numpy as np
from orthogonal_dfa.data.exon import default_exon
from orthogonal_dfa.l_star.examples.gate_composition_residual import gate_residual_oracle
from orthogonal_dfa.spliceai.load_model import load_spliceai
from orthogonal_dfa.superlanguage import KmerVocabulary, SuperSampler

STOPS = {(3, 0, 2), (3, 0, 0), (3, 2, 0)}
FOURMERS = {1, 4}   # GGTA, AGGT super-symbols
X = 5               # first wildcard

def frame_closed(seq, ph):
    return any(tuple(seq[i:i+3]) in STOPS for i in range(0, len(seq)-2, 3))

def phi(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return 0.0 if a.std() == 0 or b.std() == 0 else float(np.corrcoef(a, b)[0, 1])

def mi_bits(x, y):
    x, y = np.asarray(x), np.asarray(y)
    xs = {v: i for i, v in enumerate(np.unique(x))}; ys = {v: i for i, v in enumerate(np.unique(y))}
    j = np.zeros((len(xs), len(ys)))
    for a, b in zip(x, y): j[xs[a], ys[b]] += 1
    j /= j.sum(); px, py = j.sum(1, keepdims=True), j.sum(0, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = j * (np.log2(j) - np.log2(px) - np.log2(py))
    return float(np.nansum(t))

d = pickle.load(open("/tmp/claude-25787/-mnt-md0-orthogonal-dfa-3/e4e9621c-a36b-4b7e-a24d-819fbb2cab69/scratchpad/rounds_v2/round_00.pkl", "rb"))
dfa = d["dfa"]
V = KmerVocabulary(kmers=((3,0,2),(2,2,3,0),(3,2,0),(3,0,0),(0,2,2,3)), base_alphabet_size=4)
samp = SuperSampler(V, 36); rng = np.random.default_rng(2024)
N = 8000
ws = [samp.sample(rng, V.alphabet_size) for _ in range(N)]
bs = V.compile_many(ws, [np.random.default_rng(i) for i in range(N)])
ora = np.asarray(gate_residual_oracle(default_exon, load_spliceai(400,0), length=40, len_lo=35, len_hi=85).membership_queries(bs)).astype(int)
fc = np.array([[frame_closed(b, p) for p in range(3)] for b in bs], dtype=int)
afc = fc.all(1).astype(int)

def sub(w, rep):  # replace GGTA/AGGT with rep (a list)
    out = []
    for s in w:
        out.extend(rep) if s in FOURMERS else out.append(s)
    return out

variants = {
    "as-is        ": ws,
    "GGTA/AGGT->XXXX": [sub(w, [X, X, X, X]) for w in ws],
    "GGTA/AGGT->X   ": [sub(w, [X]) for w in ws],
}
print(f"eval N={N}, oracle accept {ora.mean():.3f}")
print(f"{'reading':16s}  accept   phi(DFA,oracle)   MI(bits)   phi(DFA,{{0,1}}closed)")
p01 = (fc[:,0] & fc[:,1])
for name, seqs in variants.items():
    call = np.array([bool(dfa.accepts_input(w)) for w in seqs], dtype=int)
    print(f"{name}  {call.mean():.3f}    {phi(call, ora):+.3f}          {mi_bits(call, ora):.5f}    {phi(call, p01):+.3f}")

# reject-by-frame-set for the ->XXXX reading (does removing escape hatches sharpen it?)
xcall = np.array([bool(dfa.accepts_input(sub(w, [X,X,X,X]))) for w in ws], dtype=int)
key = fc[:,0]*4 + fc[:,1]*2 + fc[:,2]
print("\nGGTA/AGGT->XXXX reading, reject-rate by closed-frame set:")
for k in range(8):
    m = key == k
    if m.sum():
        fs = "{" + ",".join(str(i) for i in range(3) if k & (4>>i)) + "}"
        print(f"  {fs:<9}: reject {1-xcall[m].mean():.3f} (n={int(m.sum())})")
