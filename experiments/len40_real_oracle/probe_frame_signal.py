"""Direct probe: is a reading-frame signal present in the deconfounded oracle?

Independent of any learned DFA.  Measures the marginal effect of appending a
distinguisher d at absolute position L:

    g(d, L) = accept_rate(random length-L string + d) - accept_rate(length-L string)

Appended after a length-L string, d sits in reading frame ``L mod 3``, so a stop-codon
d (TAA) closes *that* frame.  If the oracle carries a reading-frame signal, g(TAA, L)
is period-3 in L (a stop matters in the frame it lands in), while a non-stop control
(AAA) is not.  The period-3 spread (max-min of mean g across the three frames)
quantifies it; the control's spread is the null.

This is the "is the target signal even there" sanity check that must pass before any
DFA-recovery result (analyze_rounds.py) is meaningful.  Note the caveat documented in
the README: both TAA and AAA also swing aperiodically length-to-length -- that is the
oracle's dominant, non-frame positionality; the frame period-3 is a ripple on top.
"""
import argparse

import numpy as np

from orthogonal_dfa.data.exon import default_exon
from orthogonal_dfa.l_star.examples.gate_composition_residual import gate_residual_oracle
from orthogonal_dfa.spliceai.load_model import load_spliceai

ENCODE = {c: i for i, c in enumerate("ACGT")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--length", type=int, default=40)
    ap.add_argument("--len-lo", type=int, default=35)
    ap.add_argument("--len-hi", type=int, default=85)
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--l-min", type=int, default=24)
    ap.add_argument("--l-max", type=int, default=42)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    oracle = gate_residual_oracle(
        default_exon, load_spliceai(400, 0),
        length=args.length, len_lo=args.len_lo, len_hi=args.len_hi,
    )
    mq = oracle.membership_queries
    rng = np.random.default_rng(args.seed)

    def g(d, L):
        S = [rng.integers(0, 4, L).tolist() for _ in range(args.n)]
        base = mq(S).astype(float).mean()
        withd = mq([s + d for s in S]).astype(float).mean()
        return withd - base

    print("g(d,L) = accept_rate(len-L string + d) - accept_rate(len-L string)")
    print(f"frame that d lands in = L mod 3;  averaged over {args.n} strings per L\n")
    for name, s in [("TAA (stop codon)", "TAA"), ("AAA (non-stop control)", "AAA")]:
        d = [ENCODE[c] for c in s]
        vals = {L: g(d, L) for L in range(args.l_min, args.l_max + 1)}
        by_frame = {r: float(np.mean([vals[L] for L in vals if L % 3 == r]))
                    for r in range(3)}
        spread = max(by_frame.values()) - min(by_frame.values())
        print(f"=== {name} ===")
        print(f"  mean g by frame:  "
              f"frame0={by_frame[0]:+.3f}  frame1={by_frame[1]:+.3f}  "
              f"frame2={by_frame[2]:+.3f}")
        print(f"  period-3 spread (max-min across frames): {spread:.3f}\n")


if __name__ == "__main__":
    main()
