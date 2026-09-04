"""Is the over-split REAL signal or noise?  For frame-equivalent prefix groups
(signatures the true 8-state DFA merges), measure the spliceai oracle's mean
accept rate per group over a common suffix distribution.  A systematic gap =>
the oracle really distinguishes them (the split is real, the frame rule is lossy);
no gap => the split would be noise.  Contrast with a clean Bernoulli oracle whose
label is exactly the frame rule + per-string hash noise."""
import numpy as np, collections
from orthogonal_dfa.data.exon import default_exon
from orthogonal_dfa.l_star.examples.gate_composition_residual import gate_residual_oracle
from orthogonal_dfa.l_star.structures import SymmetricBernoulli
from orthogonal_dfa.spliceai.load_model import load_spliceai
from orthogonal_dfa.superlanguage.sampler import SuperSampler
from orthogonal_dfa.superlanguage.vocabulary import KmerVocabulary

STOPS = {(3, 0, 2), (3, 0, 0), (3, 2, 0)}
def sig(w):
    wc = 0; f0 = f1 = False
    for c in w:
        if c >= 3: wc += 1
        else:
            ph = wc % 3
            if ph == 0: f0 = True
            elif ph == 1: f1 = True
    return "SINK" if (f0 and f1) else (f0, f1, wc % 3)
def frame01_reject(seq):   # true label on compiled bases: reject iff f0 and f1 closed
    c = lambda ph: any(tuple(seq[ph:][i:i+3]) in STOPS for i in range(0, len(seq[ph:])-2, 3))
    return c(0) and c(1)

vocab = KmerVocabulary(kmers=((3,0,2),(3,0,0),(3,2,0)), base_alphabet_size=4)
psamp = SuperSampler(vocab, 12)      # prefixes
ssamp = SuperSampler(vocab, 24)      # suffixes (common distribution)
rng = np.random.default_rng(11)

# bucket prefixes by signature
buckets = collections.defaultdict(list)
for _ in range(40000):
    w = list(psamp.sample(rng, vocab.alphabet_size))
    buckets[sig(w)].append(w)

SUFF = [list(ssamp.sample(rng, vocab.alphabet_size)) for _ in range(60)]
base = gate_residual_oracle(default_exon, load_spliceai(400, 0), length=40, len_lo=35, len_hi=85)
bern = SymmetricBernoulli(p_correct=0.72)   # clean oracle: frame rule + per-string hash noise

def group_rate(sigkey, npref, oracle_kind):
    prefs = buckets[sigkey][:npref]
    combos, rngs = [], []
    for pi, p in enumerate(prefs):
        for si, v in enumerate(SUFF):
            combos.append(p + v); rngs.append(np.random.default_rng(1000 + pi*len(SUFF) + si))
    cb = vocab.compile_many(combos, rngs)
    if oracle_kind == "spliceai":
        lab = np.asarray(base.membership_queries([bytes(b) for b in cb]), float)
    else:  # clean bernoulli: true frame label XOR hash noise
        lab = np.array([float(bern.apply_noise(not frame01_reject(b), bytes(b), 0)) for b in cb])
    return lab.reshape(len(prefs), len(SUFF)).mean(1)   # per-prefix accept rate

pairs = [((True,False,1),(False,True,0)), ((True,False,0),(False,True,2)), ((False,True,1),(True,False,2))]
for kind in ("spliceai", "bernoulli"):
    print(f"\n===== {kind} =====")
    for x, y in pairs:
        rx, ry = group_rate(x, 60, kind), group_rate(y, 60, kind)
        # Welch effect size
        d = (rx.mean()-ry.mean()) / np.sqrt((rx.var()+ry.var())/2 + 1e-9)
        print(f"  merged pair {x} vs {y}: mean acc {rx.mean():.3f} vs {ry.mean():.3f} "
              f"(gap {rx.mean()-ry.mean():+.3f}, cohen-d {d:+.2f})")
