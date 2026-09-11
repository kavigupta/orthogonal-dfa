import pickle, numpy as np
from orthogonal_dfa.superlanguage import KmerVocabulary, SuperSampler
d = pickle.load(open("/tmp/claude-25787/-mnt-md0-orthogonal-dfa-3/e4e9621c-a36b-4b7e-a24d-819fbb2cab69/scratchpad/round_dumps/round_00.pkl", "rb"))
dfa = d["dfa"]
STOPS = {(3, 0, 2), (3, 0, 0), (3, 2, 0)}

def closed_frames(seq):
    fs = set()
    for ph in range(3):
        sub = seq[ph:]
        if any(tuple(sub[i:i+3]) in STOPS for i in range(0, len(sub)-2, 3)):
            fs.add(ph)
    return frozenset(fs)

v = KmerVocabulary(kmers=((3, 0, 2), (3, 0, 0), (3, 2, 0)), base_alphabet_size=4)
s = SuperSampler(v, 36)
rng = np.random.default_rng(11)
ws = [s.sample(rng, v.alphabet_size) for _ in range(20000)]
bs = v.compile_many(ws, [np.random.default_rng(i) for i in range(len(ws))])
call = np.array([bool(dfa.accepts_input(w)) for w in ws])
cf = [closed_frames(b) for b in bs]
groups = {}
for c, a in zip(cf, call):
    groups.setdefault(c, []).append(a)
print("DFA reject-rate by EXACT set of closed frames:")
for k in sorted(groups, key=lambda s: (len(s), sorted(s))):
    a = np.array(groups[k])
    label = "{" + ",".join(str(x) for x in sorted(k)) + "}"
    print(f"  frames closed = {label:<9}: reject {1-a.mean():.3f}  accept {a.mean():.3f}  (n={len(a)})")
