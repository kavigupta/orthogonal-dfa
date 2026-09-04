"""Score a suffix family's discriminating power for the round-0 frame rule
(reject iff f0 AND f1 closed).  For T test prefixes with known frame target and
K sampled family suffixes v, compute oracle(compile(p.v)); a suffix that tracks
the prefix's frame target is a good distinguisher.  Reports the best and the
aggregate discriminating power.  Oracle is identical across commits, so all
families are scored on the same footing."""
import sys, numpy as np
from orthogonal_dfa.data.exon import default_exon
from orthogonal_dfa.l_star.examples.gate_composition_residual import gate_residual_oracle
from orthogonal_dfa.spliceai.load_model import load_spliceai
from orthogonal_dfa.superlanguage.sampler import SuperSampler
from orthogonal_dfa.superlanguage.vocabulary import KmerVocabulary

STOPS = {(3, 0, 2), (3, 0, 0), (3, 2, 0)}
def frames01_closed(seq):
    def closed(ph): return any(tuple(seq[ph:][i:i+3]) in STOPS for i in range(0, len(seq[ph:])-2, 3))
    return closed(0) and closed(1)
def phi(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return 0.0 if a.std()==0 or b.std()==0 else float(np.corrcoef(a, b)[0, 1])

famfile, label = sys.argv[1], sys.argv[2]
fam = [list(map(int, ln.split(","))) for ln in open(famfile) if ln.strip()]
rng = np.random.default_rng(12345)                 # FIXED across families -> comparable
vocab = KmerVocabulary(kmers=((3,0,2),(3,0,0),(3,2,0)), base_alphabet_size=4)
samp = SuperSampler(vocab, 12)                      # test "prefixes" length 12
T, K = 240, 80
prefixes = [list(samp.sample(rng, vocab.alphabet_size)) for _ in range(T)]
pbases = vocab.compile_many(prefixes, [np.random.default_rng(10_000+i) for i in range(T)])
target = np.array([0.0 if frames01_closed(b) else 1.0 for b in pbases])   # accept target
vidx = rng.choice(len(fam), size=min(K, len(fam)), replace=False)
vs = [fam[i] for i in vidx]
base = gate_residual_oracle(default_exon, load_spliceai(400, 0), length=40, len_lo=35, len_hi=85)

# batch all p.v
combos, rngs = [], []
for j, v in enumerate(vs):
    for i, p in enumerate(prefixes):
        combos.append(p + v); rngs.append(np.random.default_rng(50_000 + j*T + i))
cbytes = [bytes(b) for b in vocab.compile_many(combos, rngs)]
resp = np.asarray(base.membership_queries(cbytes), float).reshape(len(vs), T)   # (K, T)
phis = np.array([phi(resp[j], target) for j in range(len(vs))])
avg = resp.mean(0)
print(f"{label}: target reject-frac={1-target.mean():.3f} | "
      f"per-suffix phi(v,target): max={np.max(np.abs(phis)):.3f} "
      f"top5mean={np.mean(np.sort(np.abs(phis))[-5:]):.3f} "
      f"median={np.median(np.abs(phis)):.3f} | phi(family-avg,target)={phi(avg,target):+.3f}")
