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

Result on merged main (seed 0, fnr-limit 0.10): round 0 = 8-state DFA, accept-rate
0.792, phi(DFA, oracle) = +0.164.  Frame closure (f0,f1,f2) *fully determines* its
output -- every cross-tab cell is 1.0 or 0.0 -- so round 0's language is EXACTLY the
predicate "reject iff frames 0 AND 1 both carry a stop codon" (frame 2 ignored).
Round 1 = 12 states (11 reachable/minimal), accept-rate 0.601, phi(DFA,oracle)
+0.121, strictly weaker.  Its per-frame-pattern cross-tab looks fuzzy (cells like
0.323 / 0.816 = the *share* of same-pattern strings accepted; the DFA is
deterministic, the frame projection is lossy), but the DFA is interpretable: it
EXACTLY tracks reading phase g = (#wildcards mod 3) -- every wildcard edge does
g->g+1 with zero violations -- and behaves as an approximate detector of "a stop
codon in reading frame 0" (agrees 89%, phi +0.787 with that predicate; cf. round 0
= frame 0 AND frame 1).  It is only approximate because it is blind to the random
wildcard bases that also create frame-0 stops.  Single non-accepting state 1 is an
absorbing trap; accept iff the run avoids it.  The best-round wrapper keeps round 0.
Round 1 does NOT collapse to accept-all here; the ~0.93 accept-all was a pre-merge
branch artifact main's accept-preserving machinery removes.  Baseline
phi(oracle, all-frames-closed) = -0.087.
"""
import argparse
import os
import numpy as np

from orthogonal_dfa.data.exon import default_exon
from orthogonal_dfa.l_star.counterexample_synthesis import counterexample_driven_synthesis
from orthogonal_dfa.l_star.examples.gate_composition_residual import gate_residual_oracle
from orthogonal_dfa.l_star.learn import build_pst
from orthogonal_dfa.spliceai.load_model import load_spliceai
# main's #209 superlanguage keeps __init__ empty; import from the submodules.
from orthogonal_dfa.superlanguage.oracle import LiftedOracle
from orthogonal_dfa.superlanguage.sampler import SuperSampler
from orthogonal_dfa.superlanguage.vocabulary import KmerVocabulary

TAG, TAA, TGA = (3, 0, 2), (3, 0, 0), (3, 2, 0)
STOPS = {TAG, TAA, TGA}


def frames_closed(seq):
    """(f0, f1, f2): whether each reading frame contains a stop codon."""
    out = []
    for ph in range(3):
        sub = seq[ph:]
        out.append(any(tuple(sub[i:i+3]) in STOPS for i in range(0, len(sub)-2, 3)))
    return tuple(out)


def n_frames_closed(seq):
    return sum(frames_closed(seq))


def phi(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return 0.0 if a.std() == 0 or b.std() == 0 else float(np.corrcoef(a, b)[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--length", type=int, default=40)
    ap.add_argument("--len-lo", type=int, default=35)
    ap.add_argument("--len-hi", type=int, default=85)
    ap.add_argument("--num-symbols", type=int, default=36)
    ap.add_argument("--kmers", default="TAG,TAA,TGA",
                    help="comma-separated ACGT kmers (prefix-free)")
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
    kmers = tuple(tuple("ACGT".index(c) for c in k) for k in args.kmers.split(","))
    vocab = KmerVocabulary(kmers=kmers, base_alphabet_size=4)
    print(f"vocab kmers={args.kmers} = {vocab.kmers}, alphabet_size={vocab.alphabet_size} "
          f"(wildcards {tuple(range(vocab.num_kmers, vocab.alphabet_size))})", flush=True)

    # No synthetic noise on the real oracle (ignore _nm).  Main's #251 moved noise to
    # a separate wrapper and #209's LiftedOracle no longer takes num_compilations /
    # noise_model, so a bare LiftedOracle is the deterministic num_compilations=1 case.
    def oracle_creator(_nm, s):
        return LiftedOracle(base, vocab, seed=s)

    pst = build_pst(
        oracle_creator, min_signal_strength=args.min_signal_strength, seed=args.seed,
        sampler=SuperSampler(vocab, args.num_symbols),
    )
    # main hardcodes SearchConfig.fnr_limit=0.02 and build_pst exposes no knob; this
    # oracle's family-FNR floor is ~0.06-0.085, so sample_suffix_family's uncapped
    # `while True` never clears 0.02 and spins to OOM.  Give it the same headroom the
    # branch used so round 0 can actually form (score with phi, not est).
    pst.config.fnr_limit = args.fnr_limit
    print(f"synthesizing (fnr_limit={args.fnr_limit}, max_rounds={args.max_rounds}) via "
          f"main's counterexample_driven_synthesis...", flush=True)

    # Fresh eval set, shared across rounds (score with phi, not est).
    samp = SuperSampler(vocab, args.num_symbols)
    rng = np.random.default_rng(args.seed + 999)
    supers = [samp.sample(rng, vocab.alphabet_size) for _ in range(args.n_eval)]
    bases = vocab.compile_many(supers, [np.random.default_rng(i) for i in range(args.n_eval)])
    # #219 made oracles answer in bytes; compile_many still returns int lists.
    ora = np.asarray(base.membership_queries([bytes(b) for b in bases])).astype(float)
    fpat = [frames_closed(b) for b in bases]                 # (f0,f1,f2) per eval string
    nfc = np.array([sum(p) for p in fpat])
    afc = (nfc == 3).astype(float)
    dump_dir = args.dump_dir or "postmerge_dumps"
    os.makedirs(dump_dir, exist_ok=True)

    def evaluate(dfa, dt, boundary, i, label):
        import pickle
        call = np.array([bool(dfa.accepts_input(w)) for w in supers], dtype=float)
        tag = (" ACCEPT-ALL" if call.mean() > 0.5 else " REJECT-ALL") if call.std() == 0 else ""
        print(f"\n[{label}] DFA: {len(dfa.states)} states, accept-rate {call.mean():.3f}{tag}")
        print(f"  phi(DFA, oracle)            = {phi(call, ora):+.3f}")
        print(f"  phi(DFA, all-frames-closed) = {phi(call, afc):+.3f}")
        # Cross-tab the DFA's rule against reading-frame closure.  For each frame
        # pattern (f0,f1,f2), report how often the DFA accepts vs the oracle -- the
        # DFA-accept column, read off, IS the learned rule in frame terms.
        print("  rule by frame pattern (f0f1f2): count  DFA-acc  oracle-acc")
        for pat in sorted({p for p in fpat}, key=lambda p: (sum(p), p)):
            m = np.array([p == pat for p in fpat])
            if m.sum() == 0:
                continue
            bits = "".join("C" if f else "." for f in pat)
            print(f"    {bits}  ({sum(pat)}closed): {int(m.sum()):5d}  "
                  f"{call[m].mean():.3f}    {ora[m].mean():.3f}")
        print("  rule by #frames closed: n  count  DFA-acc  oracle-acc")
        for n in range(4):
            m = nfc == n
            if m.sum():
                print(f"    n={n}: {int(m.sum()):5d}  {call[m].mean():.3f}    {ora[m].mean():.3f}")
        with open(os.path.join(dump_dir, f"round_{i:02d}.pkl"), "wb") as f:
            pickle.dump(dict(dfa=dfa, dt=dt, boundary=float(boundary),
                             call=call, fpat=fpat, ora=ora), f)

    # main's generator yields (dfa, dt, true_acc, boundary, classifier) per round;
    # round 0 is the first yield.
    got = False
    for i, (dfa, dt, true_acc, boundary, _cls) in enumerate(
        counterexample_driven_synthesis(pst, acc_threshold=args.acc_threshold)
    ):
        got = True
        evaluate(dfa, dt, boundary, i, f"round {i} (est={true_acc:.3f})")
        if i + 1 >= args.max_rounds:
            break
    if not got:
        print("[done] no DFA produced")
        return
    print(f"\n  phi(oracle, all-frames-closed) = {phi(ora, afc):+.3f}  [baseline ceiling]")


if __name__ == "__main__":
    main()
