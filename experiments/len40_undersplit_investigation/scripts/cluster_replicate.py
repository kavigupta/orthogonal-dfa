"""Why does round 1 enrich stop-heavy suffixes when wildcards agree more with the
empty seed?  Replicate identify_cluster_around on v1_s1's round-1 representative
prefixes with a realistic (base-rate) candidate pool, and watch the selection.
"""
import pickle, numpy as np
from orthogonal_dfa.data.exon import default_exon
from orthogonal_dfa.l_star.examples.gate_composition_residual import gate_residual_oracle
from orthogonal_dfa.spliceai.load_model import load_spliceai
from orthogonal_dfa.superlanguage.vocabulary import KmerVocabulary
from orthogonal_dfa.superlanguage.sampler import SuperSampler

D = "/tmp/claude-25787/-mnt-md0-orthogonal-dfa-3/e4e9621c-a36b-4b7e-a24d-819fbb2cab69/scratchpad"
V = KmerVocabulary(kmers=((3, 0, 2), (3, 0, 0), (3, 2, 0)), base_alphabet_size=4)
base = gate_residual_oracle(default_exon, load_spliceai(400, 0), length=40, len_lo=35, len_hi=85)
r1 = pickle.load(open(f"{D}/rounds_v1_s1/round_01.pkl", "rb"))
boundary = float(r1["decision_boundary"])
reps = [p for p, rep in zip(r1["prefixes"], r1["representative"]) if rep]
rng = np.random.default_rng(0)
reps = [reps[i] for i in rng.choice(len(reps), size=min(300, len(reps)), replace=False)]
print(f"round-1 representative prefixes used: {len(reps)}, decision_boundary {boundary:.4f}", flush=True)

samp = SuperSampler(V, 36)
# candidate suffix pool at the sampler base rate (like sample_more_suffixes draws),
# plus the empty seed at index 0
cands = [[]] + [samp.sample(rng, V.alphabet_size) for _ in range(600)]
nstop = np.array([sum(s < 3 for s in c) for c in cands])
def col(suffix, off):
    strings = [V.compile(list(p) + list(suffix), np.random.default_rng(off * 997 + i)) for i, p in enumerate(reps)]
    return np.asarray(base.membership_queries(strings)).astype(int)
masks = np.array([col(c, j) for j, c in enumerate(cands)])  # (num_cands, num_prefix)
print(f"candidate pool: {len(cands)} (pure-wild {int((nstop==0).sum())}={float((nstop==0).mean()):.1%}, mean stops {nstop.mean():.2f})", flush=True)

# identify_cluster_around loop (seed = empty = index 0)
count = 300
cluster = np.array([0]); loss = np.inf
for it in range(20):
    center = masks[cluster].mean(0) > boundary
    losses = (masks != center).sum(1)
    cluster = losses.argsort()[:count]
    if 0 not in cluster:
        cluster = np.concatenate([[0], cluster[:count-1]])
    nl = losses[cluster].sum()
    if nl >= loss: break
    loss = nl
sel = nstop[cluster]
print(f"\nSELECTED {len(cluster)} suffixes: pure-wild {int((sel==0).sum())} ({float((sel==0).mean()):.1%}), mean stops {sel.mean():.2f}")
print(f"  (candidate pool was pure-wild {float((nstop==0).mean()):.1%}, mean stops {nstop.mean():.2f})")
print(f"cluster_center accept-fraction over prefixes: {center.mean():.3f}")
print(f"mean loss  -- pure-wild cands: {losses[nstop==0].mean():.1f}  |  stop>=3 cands: {losses[nstop>=3].mean():.1f}  (lower loss = selected)")
