"""Clinching test: is v1_s1's ACTUAL stop-heavy round-1 family a tighter cluster
(lower identify_cluster_around loss) than a wildcard family or a base-rate family,
on the round-1 prefixes?  If wildcards/base-rate are tighter, the stop-heavy family
was NOT the clustering's free choice -- it was forced by what the candidate pool
contained.
"""
import pickle, numpy as np
from orthogonal_dfa.data.exon import default_exon
from orthogonal_dfa.l_star.examples.gate_composition_residual import gate_residual_oracle
from orthogonal_dfa.spliceai.load_model import load_spliceai
from orthogonal_dfa.superlanguage.vocabulary import KmerVocabulary
from orthogonal_dfa.superlanguage.sampler import SuperSampler

D = "/tmp/claude-25787/-mnt-md0-orthogonal-dfa-3/e4e9621c-a36b-4b7e-a24d-819fbb2cab69/scratchpad"
V = KmerVocabulary(kmers=((3, 0, 2), (3, 0, 0), (3, 2, 0)), base_alphabet_size=4)
Xs = V.wildcard_symbols
base = gate_residual_oracle(default_exon, load_spliceai(400, 0), length=40, len_lo=35, len_hi=85)
r1 = pickle.load(open(f"{D}/rounds_v1_s1/round_01.pkl", "rb"))
boundary = float(r1["decision_boundary"])
reps = [p for p, rep in zip(r1["prefixes"], r1["representative"]) if rep]
rng = np.random.default_rng(1)
reps = [reps[i] for i in rng.choice(len(reps), size=min(200, len(reps)), replace=False)]
fam = r1["dt"].base_family
K = 500

def cols(suffixes, off):
    m = []
    for j, s in enumerate(suffixes):
        strings = [V.compile(list(p) + list(s), np.random.default_rng((off + j) * 991 + i))
                   for i, p in enumerate(reps)]
        m.append(np.asarray(base.membership_queries(strings)).astype(int))
    return np.array(m)

def tightness(suffixes, off, tag):
    M = cols(suffixes, off)                    # (K, num_prefix)
    center = M.mean(0) > boundary              # the cluster's own binary center
    loss = (M != center).sum(1)                # per-suffix disagreement with center
    ns = np.array([sum(x < 3 for x in s) for s in suffixes])
    print(f"{tag:22} mean per-suffix loss {loss.mean():.1f} (of {len(reps)} prefixes)  "
          f"| pure-wild {float((ns==0).mean()):.0%}, mean stops {ns.mean():.2f}", flush=True)
    return loss.mean()

# actual family (subsample K), a wildcard family, a base-rate family
actual = [fam[i] for i in rng.choice(len(fam), size=K, replace=False)]
wild = [[int(rng.choice(Xs)) for _ in range(36)] for _ in range(K)]
samp = SuperSampler(V, 36); baserate = [samp.sample(rng, V.alphabet_size) for _ in range(K)]
print(f"round-1 prefixes {len(reps)}, boundary {boundary:.4f}, K={K} suffixes each\n")
la = tightness(actual, 0, "ACTUAL (stop-heavy)")
lw = tightness(wild, 100000, "wildcard family")
lb = tightness(baserate, 200000, "base-rate family")
print(f"\nlower loss = tighter cluster = what identify_cluster_around prefers.")
print(f"actual {la:.1f} vs wildcard {lw:.1f} vs base-rate {lb:.1f}")
print("If wildcard/base-rate < actual: the stop-heavy family was NOT the clustering's free choice.")
