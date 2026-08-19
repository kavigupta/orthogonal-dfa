"""Round-by-round signal-recovery analysis for the length-40 real-oracle run.

For each round_NN.pkl dumped by run_len40.py (with --dump-dir), measure whether that
round's DFA actually recovers the reading-frame signal -- using phi (correlation), NOT
est or accuracy.  est rewards a trivial accept-all DFA (its tree agrees with itself);
phi against the oracle and against the ground-truth all-frames-closed predicate is the
metric that tells you whether real regular structure was recovered.

Key columns:
  phi(DFA, oracle)  -- does the round DFA correlate with the (deconfounded) oracle?
  phi(DFA, afc)     -- does it correlate with all-frames-closed (the frame signal)?
  phi(oracle, afc)  -- is the frame signal even present in the oracle at this length?

A round that shatters to many states but has phi(DFA, oracle) ~ 0 recovered nothing;
a round that collapses to accept-all has DFA.std() == 0 so phi is 0 by construction.
"""
import argparse
import glob
import os
import pickle

import numpy as np

from orthogonal_dfa.data.exon import default_exon
from orthogonal_dfa.l_star.examples.gate_composition_residual import gate_residual_oracle
from orthogonal_dfa.spliceai.load_model import load_spliceai

STOPS = {(3, 0, 0), (3, 0, 2), (3, 2, 0)}  # TAA, TAG, TGA with A=0,C=1,G=2,T=3


def n_frames_closed(seq):
    """Number of the 3 reading frames that contain a stop codon."""
    n = 0
    for phase in range(3):
        sub = seq[phase:]
        if any(tuple(sub[i:i + 3]) in STOPS for i in range(0, len(sub) - 2, 3)):
            n += 1
    return n


def phi(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return 0.0 if a.std() == 0 or b.std() == 0 else float(np.corrcoef(a, b)[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump-dir", required=True)
    ap.add_argument("--length", type=int, default=40)
    ap.add_argument("--len-lo", type=int, default=35)
    ap.add_argument("--len-hi", type=int, default=85)
    ap.add_argument("--n-eval", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=12345)
    args = ap.parse_args()

    oracle = gate_residual_oracle(
        default_exon, load_spliceai(400, 0),
        length=args.length, len_lo=args.len_lo, len_hi=args.len_hi,
    )
    rng = np.random.default_rng(args.seed)
    S = rng.integers(0, 4, (args.n_eval, args.length))
    rows = [s.tolist() for s in S]
    ora = np.array(oracle.membership_queries(rows)).astype(float)
    nfr = np.array([n_frames_closed(r) for r in rows])
    afc = (nfr == 3).astype(float)

    print(f"eval set: {args.n_eval} random length-{args.length} strings")
    print(f"phi(oracle, all-frames-closed) = {phi(ora, afc):+.3f}   "
          "(is the frame signal present in the oracle at this length?)\n")

    for path in sorted(glob.glob(os.path.join(args.dump_dir, "round_*.pkl"))):
        with open(path, "rb") as f:
            d = pickle.load(f)
        if "dfa" not in d:
            print(f"{os.path.basename(path)}: no DFA ({d.get('dump_error')})")
            continue
        dfa = d["dfa"]
        call = np.array([bool(dfa.accepts_input(r)) for r in rows]).astype(float)
        if call.std() == 0:
            tag = "ACCEPT-ALL" if call.mean() > 0.5 else "REJECT-ALL"
        else:
            tag = ""
        print(f"{os.path.basename(path)}: round {d.get('round')}, "
              f"{len(dfa.states)} states, est {d.get('est'):.3f}, "
              f"accept-rate {call.mean():.3f} {tag}")
        print(f"    phi(DFA, oracle) = {phi(call, ora):+.3f}   "
              f"phi(DFA, afc) = {phi(call, afc):+.3f}")
        parts = []
        for k in range(4):
            m = nfr == k
            if m.any():
                parts.append(f"nfr={k}: DFA {call[m].mean():.2f}/ora {ora[m].mean():.2f}")
        print("    accept-rate by #frames-closed:  " + "   ".join(parts))


if __name__ == "__main__":
    main()
