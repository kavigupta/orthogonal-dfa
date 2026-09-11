"""Experiment 1: exact suffix-family FNR on the length-40 deconfounded oracle,
over the stop-codon superlanguage.

Question: E-L*'s FNR gate (``sample_suffix_family``) loops until the family's
false-negative rate drops to ``fnr_limit`` (0.02).  On the real deconfounded
SpliceAI oracle it never does -- why?  This measures the exact ``compute_fnr`` a
suffix family achieves, for two family types:

  * pure-wildcard (X/Y only) -- the family the clustering locks onto (wildcard-only
    suffixes share the empty-seed column), and
  * kmer-bearing (drawn from SuperSampler, so containing stop codons).

Setup mirrors the learn run: real oracle, NO synthetic noise (the compilation
fiber is the only stochasticity; ``min_signal_strength`` sizes the search only,
exactly as run_len40.py passes ``lambda _nm,_s: oracle``), ``num_compilations=1``.

Result (seed 0): pure-X/Y FNR = 0.085, kmer-bearing FNR = 0.065 -- both far above
0.02.  ~6-9% of representative prefixes sit inside the +/-eps decision band no
matter the family, because the oracle's label flips across wildcard compilations;
so no family drives the FNR to 0.02 and the strict-limit run cannot converge.
(Contrast: on the synthetic AllFramesClosedOracle a wildcard tail is neutral, so
the pure-X/Y family is decisive and its FNR is ~0.)
"""
import argparse
import numpy as np

from orthogonal_dfa.data.exon import default_exon
from orthogonal_dfa.l_star.examples.gate_composition_residual import gate_residual_oracle
from orthogonal_dfa.l_star.learn import build_pst
from orthogonal_dfa.spliceai.load_model import load_spliceai
from orthogonal_dfa.superlanguage import KmerVocabulary, SuperSampler, LiftedOracle

TAG, TAA, TGA = (3, 0, 2), (3, 0, 0), (3, 2, 0)  # A=0 C=1 G=2 T=3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--length", type=int, default=40)
    ap.add_argument("--len-lo", type=int, default=35)
    ap.add_argument("--len-hi", type=int, default=85)
    ap.add_argument("--num-symbols", type=int, default=36)
    ap.add_argument("--min-signal-strength", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    base = gate_residual_oracle(
        default_exon, load_spliceai(400, 0),
        length=args.length, len_lo=args.len_lo, len_hi=args.len_hi,
    )
    vocab = KmerVocabulary(kmers=(TAG, TAA, TGA), base_alphabet_size=4)
    X, Y = vocab.wildcard_symbols

    # No synthetic noise: ignore the framework noise model (_nm), like run_len40.py.
    def oracle_creator(_nm, s):
        return LiftedOracle(base, vocab, num_compilations=1, seed=s, noise_model=None)

    pst = build_pst(
        oracle_creator, min_signal_strength=args.min_signal_strength, seed=args.seed,
        sample_length=args.num_symbols, sampler=SuperSampler(vocab, args.num_symbols),
        fnr_limit=0.02,
    )
    N = pst.config.suffix_family_size
    print(f"suffix_family_size N={N}  representative prefixes={int(pst.table.representative.sum())}  "
          f"accept_thresh={pst.accept_thresh:.4f} reject_thresh={pst.reject_thresh:.4f}", flush=True)

    rng = np.random.default_rng(args.seed)

    def report(vs, name):
        dec = pst.compute_decision(vs, pst.table.representative)
        below = float((dec < pst.reject_thresh).mean())
        above = float((dec >= pst.accept_thresh).mean())
        fnr = float(pst.compute_fnr(vs))
        print(f"{name}: decision mean {dec.mean():.3f} std {dec.std():.3f} | "
              f"reject {below:.3f} / accept {above:.3f} / indecisive {1-below-above:.3f} "
              f"==> FNR = {fnr:.4f}", flush=True)

    # pure-wildcard family: distinct X/Y strings.  Length 1..20 so there are
    # sum_{L=1}^{20} 2^L ~= 2M distinct strings, far more than N (length<=10 gives
    # only 2046 and the draw loop would never fill N=2408).
    wild, attempts = set(), 0
    while len(wild) < N:
        L = int(rng.integers(1, 21))
        wild.add(tuple(int(rng.choice([X, Y])) for _ in range(L)))
        attempts += 1
        assert attempts < 200 * N, "cannot draw enough distinct wildcard strings"
    report([pst.table.intern_suffix(list(w)) for w in wild], "pure-X/Y    ")

    # kmer-bearing family: sampled super-strings (contain stop codons).
    samp = SuperSampler(vocab, 8)
    kf = set()
    while len(kf) < N:
        kf.add(tuple(samp.sample(rng, vocab.alphabet_size)))
    report([pst.table.intern_suffix(list(w)) for w in kf], "kmer-bearing")


if __name__ == "__main__":
    main()
