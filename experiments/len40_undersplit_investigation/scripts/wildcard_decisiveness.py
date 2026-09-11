"""Fast, direct probe: on the real deconfounded oracle, is a wildcard-only suffix
family decisive per prefix, or indecisive?

For each of P prefixes we query the RAW oracle (no framework noise) on
prefix + suffix for S distinct pure-X/Y suffixes, and look at the per-prefix
accept fraction ``decision[p]``.  Decisive => concentrated at 0/1; indecisive =>
piled near 1/2.  FNR is (roughly) the fraction of prefixes whose decision lands in
the accept/reject dead-band, so this shows what drives it -- without the
N=2408 x 200 full-family cost.
"""
import numpy as np
from orthogonal_dfa.data.exon import default_exon
from orthogonal_dfa.l_star.examples.gate_composition_residual import gate_residual_oracle
from orthogonal_dfa.spliceai.load_model import load_spliceai
from orthogonal_dfa.superlanguage import KmerVocabulary, SuperSampler

TAG, TAA, TGA = (3, 0, 2), (3, 0, 0), (3, 2, 0)
base = gate_residual_oracle(default_exon, load_spliceai(400, 0), length=40, len_lo=35, len_hi=85)
vocab = KmerVocabulary(kmers=(TAG, TAA, TGA), base_alphabet_size=4)
X, Y = vocab.wildcard_symbols

P, S = 200, 64
rng = np.random.default_rng(0)
# prefixes: real sampled super-strings; suffixes: pure X/Y of length ~8
samp = SuperSampler(vocab, 30)
prefixes = [samp.sample(rng, vocab.alphabet_size) for _ in range(P)]
suffixes = [[int(rng.choice([X, Y])) for _ in range(int(rng.integers(4, 12)))] for _ in range(S)]

# one flat batch: P*S compiled base strings
flat = [vocab.compile(list(p) + list(s), rng) for p in prefixes for s in suffixes]
labels = np.asarray(base.membership_queries(flat)).reshape(P, S).astype(float)
decision = labels.mean(1)  # per-prefix accept fraction across wildcard tails

print(f"prefixes={P}, wildcard suffixes each={S}")
print(f"per-prefix decision (accept-frac across wildcard tails):")
print(f"  mean {decision.mean():.3f}, std {decision.std():.3f}")
for lo, hi, name in [(0.0, 0.1, "decisive REJECT [0,0.1)"),
                     (0.1, 0.45, "leaning reject"),
                     (0.45, 0.55, "INDECISIVE [0.45,0.55)"),
                     (0.55, 0.9, "leaning accept"),
                     (0.9, 1.01, "decisive ACCEPT [0.9,1]")]:
    frac = ((decision >= lo) & (decision < hi)).mean()
    print(f"  {name:28s}: {frac:.3f}")
# how many prefixes have the label actually FLIP across wildcard tails at all
flips = ((labels.min(1) == 0) & (labels.max(1) == 1)).mean()
print(f"fraction of prefixes whose label FLIPS across the {S} wildcard tails: {flips:.3f}")
print(f"(if wildcard tails were neutral this would be ~0 -> decisive; high -> indecisive)")
