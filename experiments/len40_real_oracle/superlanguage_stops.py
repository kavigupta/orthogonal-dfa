"""E-L* over the stop-codon kmer language on the length-40 composition-deconfounded
SpliceAI oracle.

run_len40.py's raw-ACGT direct-L* either shatters (phi~0) or collapses to
accept-all -- it never captures the reading-frame signal the oracle is known to
carry.  Here we swap only the *alphabet*: the learner works over the superlanguage
whose symbols are the three stop codons (TAG/TAA/TGA) plus interchangeable
wildcards.  compile/parse are inverses (a wildcard never forges a stop codon), so
the lifted oracle is well-defined and the learned DFA is scored with phi
(correlation), NOT est, exactly as analyze_rounds.py insists.

  --probe-only   cheap: does the stop-codon alphabet expose any oracle signal?
  (default)      also run direct-L* FNR synthesis over the super alphabet and
                 phi-score the learned DFA vs oracle and vs all-frames-closed.
"""
import argparse
import time

import numpy as np

from orthogonal_dfa.data.exon import default_exon
from orthogonal_dfa.l_star.examples.gate_composition_residual import gate_residual_oracle
from orthogonal_dfa.l_star.learn import build_pst
from orthogonal_dfa.l_star.counterexample_synthesis import synthesize_direct_lstar_fnr
from orthogonal_dfa.spliceai.load_model import load_spliceai
from orthogonal_dfa.superlanguage import KmerVocabulary, LiftedOracle, SuperSampler

# A=0, C=1, G=2, T=3
TAG, TAA, TGA = (3, 0, 2), (3, 0, 0), (3, 2, 0)
STOPS = {TAG, TAA, TGA}


def n_frames_closed(seq):
    n = 0
    for phase in range(3):
        sub = seq[phase:]
        if any(tuple(sub[i : i + 3]) in STOPS for i in range(0, len(sub) - 2, 3)):
            n += 1
    return n


def phi(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return 0.0 if a.std() == 0 or b.std() == 0 else float(np.corrcoef(a, b)[0, 1])


def eval_set(vocab, num_symbols, n_eval, base_oracle, seed):
    """Sample super-strings, compile each once to a base string, and return
    (super_strings, oracle_label, n_frames_closed, base_len) aligned on that base
    string."""
    sampler = SuperSampler(vocab, num_symbols)
    rng = np.random.default_rng(seed)
    supers, bases = [], []
    for _ in range(n_eval):
        w = sampler.sample(rng, vocab.alphabet_size)
        supers.append(w)
        bases.append(vocab.compile(w, rng))
    ora = np.asarray(base_oracle.membership_queries(bases)).astype(float)
    nfr = np.array([n_frames_closed(b) for b in bases])
    blen = np.array([len(b) for b in bases])
    return supers, ora, nfr, blen


def probe(vocab, num_symbols, n_eval, base_oracle, seed):
    supers, ora, nfr, blen = eval_set(vocab, num_symbols, n_eval, base_oracle, seed)
    afc = (nfr == 3).astype(float)
    n_kmer = np.array([sum(not vocab.is_unknown(s) for s in w) for w in supers])
    print(f"  compiled base length: mean {blen.mean():.1f} (min {blen.min()}, max {blen.max()})")
    print(f"  stop-codon symbols per string: mean {n_kmer.mean():.2f}")
    print(f"  oracle accept-rate overall: {ora.mean():.3f}")
    print(f"  phi(oracle, all-frames-closed) = {phi(ora, afc):+.3f}")
    print(f"  phi(oracle, #frames-closed)    = {phi(ora, nfr):+.3f}")
    print(f"  phi(oracle, #stop-codon-symbols) = {phi(ora, n_kmer):+.3f}")
    parts = []
    for k in range(4):
        m = nfr == k
        if m.any():
            parts.append(f"nfr={k}: ora {ora[m].mean():.3f} (n={m.sum()})")
    print("  oracle accept-rate by #frames-closed:  " + "  ".join(parts))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--length", type=int, default=40)
    ap.add_argument("--len-lo", type=int, default=35)
    ap.add_argument("--len-hi", type=int, default=85)
    ap.add_argument("--num-symbols", type=int, default=36)
    ap.add_argument("--num-compilations", type=int, default=5)
    ap.add_argument("--min-signal-strength", type=float, default=0.05)
    ap.add_argument("--max-rounds", type=int, default=8)
    ap.add_argument("--fnr-limit", type=float, default=0.02)
    ap.add_argument("--acc-threshold", type=float, default=0.98)
    ap.add_argument("--n-eval", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--probe-only", action="store_true")
    args = ap.parse_args()

    print(f"building length-{args.length} deconfounded oracle (band {args.len_lo}-{args.len_hi})...",
          flush=True)
    base_oracle = gate_residual_oracle(
        default_exon, load_spliceai(400, 0),
        length=args.length, len_lo=args.len_lo, len_hi=args.len_hi,
    )
    vocab = KmerVocabulary(kmers=(TAG, TAA, TGA), base_alphabet_size=4)
    print(f"vocab: kmers={vocab.kmers}, alphabet_size={vocab.alphabet_size} "
          f"(wildcards {vocab.wildcard_symbols}), probs={np.round(vocab.probabilities(), 4)}",
          flush=True)

    print("\n=== PROBE (no learning) ===", flush=True)
    probe(vocab, args.num_symbols, args.n_eval, base_oracle, args.seed + 777)
    if args.probe_only:
        return

    print("\n=== LEARN (direct-L* FNR over the stop-codon superlanguage) ===", flush=True)

    # IGNORE nm: the real SpliceAI-derived oracle must NOT be corrupted with
    # synthetic SymmetricBernoulli noise (run_len40.py does the same -- it passes
    # `lambda _nm, _s: oracle`).  min_signal_strength is used only to size the
    # search; the oracle's own variation (compilation fiber + near-threshold
    # neural response) is the noise.
    def oracle_creator(_nm, s):
        return LiftedOracle(
            base_oracle, vocab, num_compilations=args.num_compilations, seed=s,
            noise_model=None,
        )

    t0 = time.time()
    pst = build_pst(
        oracle_creator,
        min_signal_strength=args.min_signal_strength,
        seed=args.seed,
        sample_length=args.num_symbols,
        sampler=SuperSampler(vocab, args.num_symbols),
        fnr_limit=args.fnr_limit,
    )
    print(f"pst built in {time.time() - t0:.0f}s; synthesizing (max_rounds={args.max_rounds})...",
          flush=True)
    dfa, _tree = synthesize_direct_lstar_fnr(
        pst, acc_threshold=args.acc_threshold, max_rounds=args.max_rounds
    )
    if dfa is None:
        print("[done] synthesis produced no DFA")
        return
    print(f"[done] {len(dfa.states)} states in {time.time() - t0:.0f}s total", flush=True)

    supers, ora, nfr, _ = eval_set(vocab, args.num_symbols, args.n_eval, base_oracle, args.seed + 999)
    afc = (nfr == 3).astype(float)
    call = np.array([bool(dfa.accepts_input(w)) for w in supers]).astype(float)
    tag = ""
    if call.std() == 0:
        tag = "ACCEPT-ALL" if call.mean() > 0.5 else "REJECT-ALL"
    print(f"\nlearned DFA: {len(dfa.states)} states, accept-rate {call.mean():.3f} {tag}")
    print(f"  phi(DFA, oracle) = {phi(call, ora):+.3f}")
    print(f"  phi(DFA, all-frames-closed) = {phi(call, afc):+.3f}")
    print(f"  phi(oracle, all-frames-closed) = {phi(ora, afc):+.3f}  (baseline ceiling)")
    parts = []
    for k in range(4):
        m = nfr == k
        if m.any():
            parts.append(f"nfr={k}: DFA {call[m].mean():.2f}/ora {ora[m].mean():.2f}")
    print("  accept-rate by #frames-closed:  " + "   ".join(parts))


if __name__ == "__main__":
    main()
