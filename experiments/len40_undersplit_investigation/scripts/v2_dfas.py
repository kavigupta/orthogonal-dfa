"""Characterize every v2 (TAG,GGTA,TGA,TAA,AGGT) DFA we have: states, accept-rate,
phi(DFA,oracle), and reject-rate by exact closed-frame set."""
import pickle, glob, os, numpy as np
from orthogonal_dfa.data.exon import default_exon
from orthogonal_dfa.l_star.examples.gate_composition_residual import gate_residual_oracle
from orthogonal_dfa.spliceai.load_model import load_spliceai
from orthogonal_dfa.superlanguage.vocabulary import KmerVocabulary
from orthogonal_dfa.superlanguage.sampler import SuperSampler

D = "/tmp/claude-25787/-mnt-md0-orthogonal-dfa-3/e4e9621c-a36b-4b7e-a24d-819fbb2cab69/scratchpad"
STOPS = {(3, 0, 2), (3, 0, 0), (3, 2, 0)}
V = KmerVocabulary(kmers=((3, 0, 2), (2, 2, 3, 0), (3, 2, 0), (3, 0, 0), (0, 2, 2, 3)), base_alphabet_size=4)
base = gate_residual_oracle(default_exon, load_spliceai(400, 0), length=40, len_lo=35, len_hi=85)

def phi(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return 0.0 if a.std() == 0 or b.std() == 0 else float(np.corrcoef(a, b)[0, 1])

samp = SuperSampler(V, 36); rng = np.random.default_rng(2024)
N = 8000
ws = [samp.sample(rng, V.alphabet_size) for _ in range(N)]
bs = V.compile_many(ws, [np.random.default_rng(i) for i in range(N)])
ora = np.asarray(base.membership_queries(bs)).astype(float)
key = np.array([sum((1 if any(tuple(b[p:][i:i+3]) in STOPS for i in range(0, len(b[p:])-2, 3)) else 0) << (2 - p)
                    for p in range(3)) for b in bs])
afc = (key == 7).astype(float)
print(f"eval N={N}, oracle accept {ora.mean():.3f}, phi(oracle,afc)={phi(ora,afc):+.3f} [baseline]\n")

for tag, path in [("v2_s0 r0", "rounds_v2_s0/round_00.pkl"),
                  ("v2_s1 r0", "rounds_v2_s1/round_00.pkl"),
                  ("v2_s2 r0", "rounds_v2_s2/round_00.pkl"),
                  ("v2_s2 r1", "rounds_v2_s2/round_01.pkl")]:
    fp = f"{D}/{path}"
    if not os.path.exists(fp):
        print(f"{tag}: (missing)"); continue
    dd = pickle.load(open(fp, "rb")); dfa = dd["dfa"]
    call = np.array([bool(dfa.accepts_input(w)) for w in ws], float)
    parts = []
    for k in range(8):
        m = key == k
        if m.sum() >= 30:
            fs = "{" + ",".join(str(i) for i in range(3) if k & (4 >> i)) + "}"
            parts.append(f"{fs}:{1-call[m].mean():.2f}")
    print(f"{tag}: {dd['num_states']} states, accept {call.mean():.3f}, est {dd['est']:.3f}, "
          f"phi(DFA,ora) {phi(call, ora):+.3f}")
    print(f"    reject-rate by frames-closed: {'  '.join(parts)}", flush=True)
