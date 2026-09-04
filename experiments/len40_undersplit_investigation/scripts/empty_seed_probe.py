"""Is the empty seed representative of v1_s1's stop-heavy round-1 family, or a
forced-in outlier?  identify_cluster_around seeds on empty and force-includes it,
but the cluster CENTER is the members' consensus (majority over the 200 prefixes),
which the 2408 stop-heavy members dominate.  Measure:
  - center = family consensus column;  empty-seed column
  - how much the empty seed disagrees with the family center (its loss)
  - vs the median family member's loss
If the empty seed's loss >> median, it's a forced outlier -- the family self-agrees
without agreeing with empty, which is how a stop-heavy family stays 'anchored' on
the empty seed.
"""
import pickle, numpy as np
from orthogonal_dfa.data.exon import default_exon
from orthogonal_dfa.l_star.examples.gate_composition_residual import gate_residual_oracle
from orthogonal_dfa.spliceai.load_model import load_spliceai
from orthogonal_dfa.superlanguage.vocabulary import KmerVocabulary

D = "/tmp/claude-25787/-mnt-md0-orthogonal-dfa-3/e4e9621c-a36b-4b7e-a24d-819fbb2cab69/scratchpad"
V = KmerVocabulary(kmers=((3, 0, 2), (3, 0, 0), (3, 2, 0)), base_alphabet_size=4)
base = gate_residual_oracle(default_exon, load_spliceai(400, 0), length=40, len_lo=35, len_hi=85)
r1 = pickle.load(open(f"{D}/rounds_v1_s1/round_01.pkl", "rb"))
boundary = float(r1["decision_boundary"])
reps = [p for p, rep in zip(r1["prefixes"], r1["representative"]) if rep]
rng = np.random.default_rng(1)
reps = [reps[i] for i in rng.choice(len(reps), size=200, replace=False)]
fam = r1["dt"].base_family

# sample K members of the actual family (guaranteed to include the empty seed)
K = 400
members = [[]] + [fam[i] for i in rng.choice(len(fam), size=K - 1, replace=False)]
assert any(len(m) == 0 for m in members)

def col(s, off):
    return np.asarray(base.membership_queries(
        [V.compile(list(p) + list(s), np.random.default_rng(off * 991 + i)) for i, p in enumerate(reps)]
    )).astype(int)

M = np.array([col(s, j) for j, s in enumerate(members)])   # (K, 200)
center = M.mean(0) > boundary                              # family consensus
empty_col = M[0]                                           # empty seed column
loss = (M != center).sum(1)                                # each member's loss vs center
empty_loss = loss[0]
ns = np.array([sum(x < 3 for x in s) for s in members])

print(f"round-1 prefixes {len(reps)}, boundary {boundary:.4f}, K={K} family members")
print(f"family consensus center accept-frac: {center.mean():.3f}")
print(f"empty-seed column accept-frac:        {empty_col.mean():.3f}")
print(f"agreement(center, empty-seed): {(center == empty_col).mean():.3f}")
print()
print(f"empty-seed loss vs family center: {empty_loss} / {len(reps)}  ({empty_loss/len(reps):.3f})")
print(f"family member loss: median {np.median(loss):.0f}, mean {loss.mean():.1f}, "
      f"90th pct {np.percentile(loss,90):.0f}, max {loss.max()}")
print(f"empty-seed percentile within family losses: "
      f"{(loss < empty_loss).mean():.0%} of members have lower loss")
print()
# loss by stop-count: do stop-heavy members agree with the (stop-heavy) center?
for lo, hi, tag in [(0, 1, "pure-wild (0 stops)"), (1, 3, "1-2 stops"), (3, 99, ">=3 stops")]:
    m = (ns >= lo) & (ns < hi)
    if m.sum():
        print(f"  {tag:20}: n={int(m.sum())}, mean loss vs center {loss[m].mean():.1f}")
