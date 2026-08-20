"""Experiment 2: learn a DFA over the stop-codon superlanguage from the length-40
deconfounded oracle, and see what frame structure it recovers.

The alphabet is the three stop codons (TAG/TAA/TGA) plus interchangeable wildcards
(X, Y); compile/parse are inverses so a wildcard never forges a stop.  Real oracle,
NO synthetic noise, ``num_compilations=1`` (deterministic per seed).

``--fnr-limit`` defaults to 0.10.  Experiment 1 shows the family-FNR floor on this
oracle is ~0.06-0.085, so the strict 0.02 gate never converges (the loop spins);
0.10 sits just above the floor so the gate clears and synthesis produces a DFA.
This is the knob run_len40.py deliberately holds at 0.02 -- relaxing it can
manufacture shattering, so score with phi (analyze.py), never est.

Set ``--dump-dir`` to export ``round_NN.pkl`` per round (dfa/dt/est/boundary) for
analyze.py.  Prints ``phi(DFA, oracle)`` and ``phi(DFA, all-frames-closed)`` on a
fresh eval set at the end.

Result (seed 0, fnr-limit 0.10): round 0 = 8-state DFA, accept-rate 0.78,
phi(DFA, oracle) = +0.167; round 1 drifts toward accept-all (0.93), phi +0.077.
The round-0 DFA rejects exactly when reading frames 0 AND 1 are both closed.
"""
import argparse
import os
import numpy as np

from orthogonal_dfa.data.exon import default_exon
from orthogonal_dfa.l_star.counterexample_synthesis import synthesize_direct_lstar_fnr
from orthogonal_dfa.l_star.examples.gate_composition_residual import gate_residual_oracle
from orthogonal_dfa.l_star.learn import build_pst
from orthogonal_dfa.spliceai.load_model import load_spliceai
from orthogonal_dfa.superlanguage import KmerVocabulary, SuperSampler, LiftedOracle

TAG, TAA, TGA = (3, 0, 2), (3, 0, 0), (3, 2, 0)
STOPS = {TAG, TAA, TGA}


def n_frames_closed(seq):
    n = 0
    for ph in range(3):
        sub = seq[ph:]
        if any(tuple(sub[i:i+3]) in STOPS for i in range(0, len(sub)-2, 3)):
            n += 1
    return n


def phi(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return 0.0 if a.std() == 0 or b.std() == 0 else float(np.corrcoef(a, b)[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--length", type=int, default=40)
    ap.add_argument("--len-lo", type=int, default=35)
    ap.add_argument("--len-hi", type=int, default=85)
    ap.add_argument("--num-symbols", type=int, default=36)
    ap.add_argument("--min-signal-strength", type=float, default=0.05)
    ap.add_argument("--fnr-limit", type=float, default=0.10)
    ap.add_argument("--max-rounds", type=int, default=8)
    ap.add_argument("--acc-threshold", type=float, default=0.98)
    ap.add_argument("--n-eval", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dump-dir", default=None, help="export round_NN.pkl per round")
    args = ap.parse_args()
    if args.dump_dir:
        os.makedirs(args.dump_dir, exist_ok=True)
        os.environ["DLSTAR_DUMP_DIR"] = args.dump_dir

    base = gate_residual_oracle(
        default_exon, load_spliceai(400, 0),
        length=args.length, len_lo=args.len_lo, len_hi=args.len_hi,
    )
    vocab = KmerVocabulary(kmers=(TAG, TAA, TGA), base_alphabet_size=4)
    print(f"vocab kmers={vocab.kmers}, alphabet_size={vocab.alphabet_size} "
          f"(wildcards {vocab.wildcard_symbols})", flush=True)

    # No synthetic noise on the real oracle (ignore _nm).
    def oracle_creator(_nm, s):
        return LiftedOracle(base, vocab, num_compilations=1, seed=s, noise_model=None)

    pst = build_pst(
        oracle_creator, min_signal_strength=args.min_signal_strength, seed=args.seed,
        sample_length=args.num_symbols, sampler=SuperSampler(vocab, args.num_symbols),
        fnr_limit=args.fnr_limit,
    )
    print(f"synthesizing (fnr_limit={args.fnr_limit}, max_rounds={args.max_rounds})...", flush=True)
    dfa, _ = synthesize_direct_lstar_fnr(
        pst, acc_threshold=args.acc_threshold, max_rounds=args.max_rounds
    )
    if dfa is None:
        print("[done] no DFA produced")
        return

    # phi-score the learned super-DFA on a fresh eval set (score with phi, not est).
    samp = SuperSampler(vocab, args.num_symbols)
    rng = np.random.default_rng(args.seed + 999)
    supers = [samp.sample(rng, vocab.alphabet_size) for _ in range(args.n_eval)]
    bases = vocab.compile_many(supers, [np.random.default_rng(i) for i in range(args.n_eval)])
    ora = np.asarray(base.membership_queries(bases)).astype(float)
    afc = np.array([n_frames_closed(b) == 3 for b in bases], dtype=float)
    call = np.array([bool(dfa.accepts_input(w)) for w in supers], dtype=float)
    tag = (" ACCEPT-ALL" if call.mean() > 0.5 else " REJECT-ALL") if call.std() == 0 else ""
    print(f"\n[done] final DFA: {len(dfa.states)} states, accept-rate {call.mean():.3f}{tag}")
    print(f"  phi(DFA, oracle)            = {phi(call, ora):+.3f}")
    print(f"  phi(DFA, all-frames-closed) = {phi(call, afc):+.3f}")
    print(f"  phi(oracle, all-frames-closed) = {phi(ora, afc):+.3f}  [baseline ceiling]")


if __name__ == "__main__":
    main()
