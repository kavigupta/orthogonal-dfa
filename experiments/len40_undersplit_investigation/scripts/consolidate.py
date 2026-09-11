"""One table: every available round-0/round-1 DFA across both versions and seeds.
For each: states, accept-rate, phi(DFA,oracle), and the reject rule (which sets of
closed reading frames it rejects)."""
import pickle, glob, os, numpy as np
from orthogonal_dfa.data.exon import default_exon
from orthogonal_dfa.l_star.examples.gate_composition_residual import gate_residual_oracle
from orthogonal_dfa.spliceai.load_model import load_spliceai
from orthogonal_dfa.superlanguage.vocabulary import KmerVocabulary
from orthogonal_dfa.superlanguage.sampler import SuperSampler

D = "/tmp/claude-25787/-mnt-md0-orthogonal-dfa-3/e4e9621c-a36b-4b7e-a24d-819fbb2cab69/scratchpad"
STOPS = {(3, 0, 2), (3, 0, 0), (3, 2, 0)}
base = gate_residual_oracle(default_exon, load_spliceai(400, 0), length=40, len_lo=35, len_hi=85)

def closed(seq):
    return frozenset(p for p in range(3)
                     if any(tuple(seq[p:][i:i+3]) in STOPS for i in range(0, len(seq[p:])-2, 3)))
def phi(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return 0.0 if a.std() == 0 or b.std() == 0 else float(np.corrcoef(a, b)[0, 1])

def rule(call, cf):
    # which closed-frame sets are rejected (>=0.5)?  report the crisp ones.
    groups = {}
    for c, k in zip(call, cf): groups.setdefault(k, []).append(c)
    rej = sorted(("{" + ",".join(map(str, sorted(k))) + "}", 1 - np.mean(v))
                 for k, v in groups.items() if len(v) >= 30)
    rejected = [f for f, r in rej if r >= 0.5]
    return "reject " + ("/".join(rejected) if rejected else "≈nothing")

CONFIGS = [
    ("v1", "TAG,TAA,TGA", [("s0", "rounds_v1_s0"), ("s1", "rounds_v1_s1"), ("s2", "rounds_v1_s2")]),
    ("v2", "TAG,GGTA,TGA,TAA,AGGT", [("s0", "rounds_v2")]),
]
print(f"{'cfg':10} {'rnd':4} {'states':6} {'accept':7} {'phi(DFA,ora)':12} rule")
for ver, kmers_s, dirs in CONFIGS:
    kmers = tuple(tuple("ACGT".index(c) for c in k) for k in kmers_s.split(","))
    V = KmerVocabulary(kmers=kmers, base_alphabet_size=4)
    samp = SuperSampler(V, 36); rng = np.random.default_rng(2024)
    ws = [samp.sample(rng, V.alphabet_size) for _ in range(6000)]
    bs = V.compile_many(ws, [np.random.default_rng(i) for i in range(6000)])
    ora = np.asarray(base.membership_queries(bs)).astype(float)
    cf = [closed(b) for b in bs]
    afc = np.array([len(k) == 3 for k in cf], float)
    for seed, d in dirs:
        for rp in sorted(glob.glob(f"{D}/{d}/round_*.pkl")):
            dd = pickle.load(open(rp, "rb")); dfa = dd["dfa"]
            call = np.array([bool(dfa.accepts_input(w)) for w in ws], float)
            rn = os.path.basename(rp).replace("round_", "r").replace(".pkl", "")
            print(f"{ver+' '+seed:10} {rn:4} {dd['num_states']:<6} {call.mean():<7.3f} "
                  f"{phi(call, ora):+.3f}       {rule(call, cf)}", flush=True)
print(f"\nbaseline: phi(oracle, all-frames-closed) = {phi(ora, afc):+.3f}")
