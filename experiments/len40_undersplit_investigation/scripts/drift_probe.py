"""Does identify_cluster_around, seeded on empty, find EMPTY'S natural cluster, or
drift to a globally-tighter (stop-heavy) one and leave empty an outlier?

Run the EXACT algorithm on a realistic candidate pool seeded on empty, then report:
  - final family composition (wildcard vs stop-heavy)
  - empty seed's loss-percentile in the final family (low = natural fit; high = drift)
  - the natural-around-empty cluster (suffixes agreeing with empty's own column):
    its composition and whether it's a distinct tight cluster.

Compares two pools: base-rate, and a deliberately stop-heavy pool, to see whether
the stop-heavy family the real run produced requires a stop-heavy pool.
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
rng = np.random.default_rng(2)
reps = [reps[i] for i in rng.choice(len(reps), size=200, replace=False)]

def cols(suffixes, off):
    return np.array([
        np.asarray(base.membership_queries(
            [V.compile(list(p) + list(s), np.random.default_rng((off + j) * 991 + i))
             for i, p in enumerate(reps)])).astype(int)
        for j, s in enumerate(suffixes)])

def cluster_around(masks, count):
    """Exact identify_cluster_around, seed = index 0 (empty)."""
    cl = np.array([0]); loss = np.inf
    for _ in range(30):
        center = masks[cl].mean(0) > boundary
        losses = (masks != center).sum(1)
        cl = losses.argsort()[:count]
        if 0 not in cl:
            cl = np.concatenate([[0], cl[:count - 1]])
        nl = losses[cl].sum()
        if nl >= loss:
            break
        loss = nl
    return cl, center, losses

samp = SuperSampler(V, 36)
def build(pool_kind, n):
    pool = [[]]  # empty seed at 0
    while len(pool) < n:
        if pool_kind == "base":
            pool.append(samp.sample(rng, V.alphabet_size))
        else:  # stop-heavy
            w = samp.sample(rng, V.alphabet_size)
            if sum(s < 3 for s in w) >= 3:
                pool.append(w)
    return pool

for kind in ("base", "stopheavy"):
    pool = build(kind, 500)
    M = cols(pool, {"base": 0, "stopheavy": 900000}[kind])
    ns = np.array([sum(s < 3 for s in p) for p in pool])
    cl, center, losses = cluster_around(M, 250)
    sel = ns[cl]
    empty_loss = losses[0]
    empty_pct = (losses[cl] < empty_loss).mean()
    print(f"\n=== pool={kind} (pure-wild {float((ns==0).mean()):.0%}, mean stops {ns.mean():.2f}) ===")
    print(f"  SELECTED family: pure-wild {float((sel==0).mean()):.0%}, mean stops {sel.mean():.2f}")
    print(f"  final center accept-frac {center.mean():.3f}; empty column accept-frac {M[0].mean():.3f}; "
          f"agree {(center==M[0]).mean():.3f}")
    print(f"  empty-seed loss {empty_loss} vs family median {int(np.median(losses[cl]))}; "
          f"empty is at {empty_pct:.0%} percentile (0%=best fit, 100%=worst)")
    verdict = "DRIFTED (empty is outlier)" if empty_pct > 0.6 else "found empty's natural cluster"
    print(f"  -> {verdict}")
