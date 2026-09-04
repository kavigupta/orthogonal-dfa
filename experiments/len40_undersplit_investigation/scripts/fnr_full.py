"""Full compute_fnr for a pure-X/Y family vs a kmer-bearing family, on the REAL
oracle (no synthetic noise), fast compiler (compile_many).  Uses ALL 200
representative prefixes -- no subset approximation needed now.
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


def oc(_nm, s):  # ignore nm: real oracle, NO synthetic noise
    return LiftedOracle(base, vocab, num_compilations=1, seed=s, noise_model=None)


pst = build_pst(oc, min_signal_strength=0.05, seed=0, sample_length=36,
                sampler=SuperSampler(vocab, 36), fnr_limit=0.02)
N = pst.config.suffix_family_size
print(f"N(suffix_family)={N}  rep={int(pst.table.representative.sum())}  "
      f"accept_thresh={pst.accept_thresh:.4f} reject_thresh={pst.reject_thresh:.4f}", flush=True)

rng = np.random.default_rng(0)


def show(vs, name):
    dec = pst.compute_decision(vs, pst.table.representative)
    below = (dec < pst.reject_thresh).mean()
    above = (dec >= pst.accept_thresh).mean()
    fnr = float(pst.compute_fnr(vs))
    print(f"{name}: decision mean {dec.mean():.3f} std {dec.std():.3f} | "
          f"reject {below:.3f} / accept {above:.3f} / indecisive {1-below-above:.3f} "
          f"==> FNR = {fnr:.4f}", flush=True)


# length 1..20 -> sum_{L=1}^{20} 2^L ~= 2M distinct wildcard strings, far more
# than N=2408 (length<=10 only gives 2046, which cannot fill the family -> hang).
wild = set()
attempts = 0
while len(wild) < N:
    L = int(rng.integers(1, 21))
    wild.add(tuple(int(rng.choice([X, Y])) for _ in range(L)))
    attempts += 1
    assert attempts < 200 * N, "cannot draw enough distinct wildcard strings"
print(f"pure-X/Y family: {len(wild)} distinct wildcard suffixes drawn", flush=True)
show([pst.table.intern_suffix(list(w)) for w in wild], "pure-X/Y   ")

samp = SuperSampler(vocab, 8)
kf = set()
while len(kf) < N:
    kf.add(tuple(samp.sample(rng, vocab.alphabet_size)))
show([pst.table.intern_suffix(list(w)) for w in kf], "kmer-bearing")
