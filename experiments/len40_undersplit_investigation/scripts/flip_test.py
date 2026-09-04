"""Does a stop-heavy suffix disagree with the empty-seed column more than a
wildcard suffix does?  That disagreement (loss) is exactly what
identify_cluster_around minimizes.  If stop-heavy ~ wildcard, the clustering
can't select against stops (-> base-rate family); if stop-heavy loss is higher,
it should deplete stops (contradicting round 0).
"""
import numpy as np
from orthogonal_dfa.data.exon import default_exon
from orthogonal_dfa.l_star.examples.gate_composition_residual import gate_residual_oracle
from orthogonal_dfa.spliceai.load_model import load_spliceai
from orthogonal_dfa.superlanguage import KmerVocabulary, SuperSampler

V = KmerVocabulary(kmers=((3, 0, 2), (3, 0, 0), (3, 2, 0)), base_alphabet_size=4)
X, Y = V.wildcard_symbols
base = gate_residual_oracle(default_exon, load_spliceai(400, 0), length=40, len_lo=35, len_hi=85)
samp = SuperSampler(V, 36)
rng = np.random.default_rng(0)

P = 400
prefixes = [samp.sample(rng, V.alphabet_size) for _ in range(P)]

def n_stops(w):
    return sum(s < 3 for s in w)

# 30 pure-wildcard and 30 stop-heavy suffixes, all length 36
wild = [[int(rng.choice([X, Y])) for _ in range(36)] for _ in range(30)]
stopheavy = []
while len(stopheavy) < 30:
    w = samp.sample(rng, V.alphabet_size)
    if n_stops(w) >= 3:
        stopheavy.append(w)

def compile1(w, seed):
    return V.compile(w, np.random.default_rng(seed))

# empty-seed column: membership(prefix)  (num_compilations=1, deterministic seed)
e = np.asarray(base.membership_queries([compile1(p, i) for i, p in enumerate(prefixes)])).astype(int)
print(f"empty-seed column: accept-rate {e.mean():.3f}  (P={P} prefixes)")

def losses(suffixes, tag):
    lo = []
    means = []
    for j, s in enumerate(suffixes):
        col = np.asarray(base.membership_queries(
            [compile1(list(p) + list(s), 10000 + j * P + i) for i, p in enumerate(prefixes)]
        )).astype(int)
        lo.append(float((col != e).mean()))   # disagreement with empty seed
        means.append(float(col.mean()))
    print(f"{tag}: disagreement-with-seed mean {np.mean(lo):.3f} (std {np.std(lo):.3f}), "
          f"col accept-rate mean {np.mean(means):.3f}")
    return lo

lw = losses(wild, "pure-wildcard suffixes")
ls = losses(stopheavy, "stop-heavy suffixes (>=3 stops)")
print(f"\ndifference in disagreement (stop - wild): {np.mean(ls) - np.mean(lw):+.3f}")
print("If ~0: clustering can't tell them apart -> base-rate family (round 0).")
print("If >0: stop-heavy disagree more -> should be selected against.")
