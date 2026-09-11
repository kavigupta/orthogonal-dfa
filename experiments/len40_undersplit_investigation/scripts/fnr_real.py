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


def oc(nm, s):
    return LiftedOracle(base, vocab, num_compilations=1, seed=s, noise_model=nm)


pst = build_pst(oc, min_signal_strength=0.05, seed=0, sample_length=36,
                sampler=SuperSampler(vocab, 36), fnr_limit=0.02)
N = pst.config.suffix_family_size
print("suffix_family_size N =", N, " rep prefixes =", int(pst.table.representative.sum()),
      " accept_thresh %.4f reject_thresh %.4f" % (pst.accept_thresh, pst.reject_thresh), flush=True)

rng = np.random.default_rng(0)
wild = set()
while len(wild) < N:
    L = int(rng.integers(1, 11))
    wild.add(tuple(int(rng.choice([X, Y])) for _ in range(L)))
vs = [pst.table.intern_suffix(list(w)) for w in wild]
dec = pst.compute_decision(vs, pst.table.representative)
undecided = ((dec >= pst.reject_thresh) & (dec < pst.accept_thresh)).mean()
print("pure-X/Y decision over rep prefixes: mean %.3f  min %.3f  max %.3f  undecided-frac %.3f"
      % (dec.mean(), dec.min(), dec.max(), undecided), flush=True)
print("FNR pure-X/Y family (N filled) =", round(float(pst.compute_fnr(vs)), 4), flush=True)
print("FNR empty-suffix only          =", round(float(pst.compute_fnr([pst.table.intern_suffix([])])), 4), flush=True)

samp = SuperSampler(vocab, 8)
kf = set()
while len(kf) < N:
    kf.add(tuple(samp.sample(rng, vocab.alphabet_size)))
vs2 = [pst.table.intern_suffix(list(w)) for w in kf]
print("FNR kmer-bearing family        =", round(float(pst.compute_fnr(vs2)), 4), flush=True)
