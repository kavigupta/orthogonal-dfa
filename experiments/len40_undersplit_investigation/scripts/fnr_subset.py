"""Actually compute compute_fnr for a pure-X/Y family on the REAL oracle (no
synthetic noise), restricted to a small subset of prefixes so it's tractable.

Same computation as PrefixSuffixTracker.compute_fnr, but compute_decision is run
over a `subset_mask` of ~40 representative prefixes instead of all 200 -- so the
cost is 2408 suffixes x 40 prefixes rather than x 200.  The suffix family is the
full configured size (2408) so decision[p] is properly concentrated.
"""
import numpy as np
from orthogonal_dfa.data.exon import default_exon
from orthogonal_dfa.l_star.examples.gate_composition_residual import gate_residual_oracle
from orthogonal_dfa.spliceai.load_model import load_spliceai
from orthogonal_dfa.l_star.learn import build_pst
from orthogonal_dfa.superlanguage import KmerVocabulary, SuperSampler, LiftedOracle

TAG, TAA, TGA = (3, 0, 2), (3, 0, 0), (3, 2, 0)
base = gate_residual_oracle(default_exon, load_spliceai(400, 0), length=40, len_lo=35, len_hi=85)
vocab = KmerVocabulary(kmers=(TAG, TAA, TGA), base_alphabet_size=4)
X, Y = vocab.wildcard_symbols


def oc(_nm, s):  # ignore nm: real oracle, NO synthetic noise (matches the fix)
    return LiftedOracle(base, vocab, num_compilations=1, seed=s, noise_model=None)


pst = build_pst(oc, min_signal_strength=0.05, seed=0, sample_length=36,
                sampler=SuperSampler(vocab, 36), fnr_limit=0.02)
N = pst.config.suffix_family_size
rep_idx = np.flatnonzero(pst.table.representative)
SUB = 30
sub_idx = rep_idx[:SUB]
mask = np.zeros(pst.num_prefixes, dtype=bool)
mask[sub_idx] = True
print(f"N(suffix_family)={N}  rep={len(rep_idx)}  subset={SUB}  "
      f"accept_thresh={pst.accept_thresh:.4f} reject_thresh={pst.reject_thresh:.4f}", flush=True)


def fnr_on(vs, mask):
    dec = pst.compute_decision(vs, mask)
    below = (dec < pst.reject_thresh).mean()
    above = (dec >= pst.accept_thresh).mean()
    arr_min = min(below, above)
    fnr = 1.0 if arr_min == 0 else 1.0 - below - above
    return fnr, dec, below, above


rng = np.random.default_rng(0)
wild = set()
while len(wild) < N:
    L = int(rng.integers(1, 11))
    wild.add(tuple(int(rng.choice([X, Y])) for _ in range(L)))
vs = [pst.table.intern_suffix(list(w)) for w in wild]
fnr, dec, below, above = fnr_on(vs, mask)
print(f"pure-X/Y: decision mean {dec.mean():.3f} std {dec.std():.3f} "
      f"(reject {below:.3f} / accept {above:.3f} / indecisive {1-below-above:.3f})", flush=True)
print(f"==> FNR pure-X/Y family (real oracle, no noise, {SUB} prefixes) = {fnr:.4f}", flush=True)

samp = SuperSampler(vocab, 8)
kf = set()
while len(kf) < N:
    kf.add(tuple(samp.sample(rng, vocab.alphabet_size)))
vs2 = [pst.table.intern_suffix(list(w)) for w in kf]
fnr2, dec2, b2, a2 = fnr_on(vs2, mask)
print(f"kmer-bearing: decision mean {dec2.mean():.3f} std {dec2.std():.3f} "
      f"(reject {b2:.3f} / accept {a2:.3f} / indecisive {1-b2-a2:.3f})", flush=True)
print(f"==> FNR kmer-bearing family (real oracle, no noise, {SUB} prefixes) = {fnr2:.4f}", flush=True)
