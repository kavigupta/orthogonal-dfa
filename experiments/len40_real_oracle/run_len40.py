"""Direct-L* FNR synthesis on the length-40 composition-deconfounded SpliceAI oracle.

This is the reproducible driver for the "does direct-L* recover a reading-frame
signal from the controlled neural oracle" experiment.  See README.md for the full
writeup, the findings, and the metric caveats (use phi, not est).

Oracle: ``gate_residual_oracle`` = SpliceAI exon score minus a monotonic-gate
bag-of-k-mers composition model, median-thresholded -- i.e. the SpliceAI signal with
its (confounding) base-composition component regressed out.  Length 40 means the
learner samples length-40 prefixes and queries prefix+suffix (~40-84 nt); the gate is
fit on a band [len_lo, len_hi) that covers those query lengths.

Env:
  DLSTAR_DUMP_DIR   if set, each round dumps round_NN.pkl (dfa/dt/boundary) for the
                    round-by-round phi analysis in analyze_rounds.py.

Config knobs below are the "leaned" settings that keep peak RSS ~17 GB (the original,
heavier config OOM'd in round 2).  fnr_limit STAYS 0.02 -- loosening it manufactures
artificial shattering and invalidates the experiment.
"""
import argparse
import os
import resource
import threading
import time

from orthogonal_dfa.data.exon import default_exon
from orthogonal_dfa.l_star.examples.gate_composition_residual import gate_residual_oracle
from orthogonal_dfa.l_star.learn import build_pst
from orthogonal_dfa.l_star.counterexample_synthesis import synthesize_direct_lstar_fnr
from orthogonal_dfa.spliceai.load_model import load_spliceai


def _watch_memory():
    while True:
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
        print(f"[mem] peak RSS {rss:.1f} GB", flush=True)
        time.sleep(60)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--length", type=int, default=40)
    ap.add_argument("--len-lo", type=int, default=35)
    ap.add_argument("--len-hi", type=int, default=85)
    ap.add_argument("--min-signal-strength", type=float, default=0.05)
    ap.add_argument("--fnr-limit", type=float, default=0.02)  # DO NOT loosen
    ap.add_argument("--max-rounds", type=int, default=10)
    ap.add_argument("--acc-threshold", type=float, default=0.98)
    ap.add_argument("--counterexample-probes", type=int, default=250)
    ap.add_argument("--per-state", type=int, default=12)
    ap.add_argument("--min-indecisive", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dump-dir", default=None,
                    help="if set, exports DLSTAR_DUMP_DIR so each round dumps a pkl")
    args = ap.parse_args()

    assert abs(args.fnr_limit - 0.02) < 1e-9, (
        "fnr_limit must stay 0.02 -- loosening it manufactures artificial shattering"
    )
    if args.dump_dir:
        os.environ["DLSTAR_DUMP_DIR"] = args.dump_dir

    threading.Thread(target=_watch_memory, daemon=True).start()

    print(f"building length-{args.length} deconfounded oracle "
          f"(band {args.len_lo}-{args.len_hi})...", flush=True)
    oracle = gate_residual_oracle(
        default_exon, load_spliceai(400, 0),
        length=args.length, len_lo=args.len_lo, len_hi=args.len_hi,
    )
    pst = build_pst(
        lambda _nm, _s: oracle,
        min_signal_strength=args.min_signal_strength,
        seed=args.seed,
        sample_length=args.length,
        fnr_limit=args.fnr_limit,
    )
    print(f"pst built; synthesizing (probes={args.counterexample_probes}, "
          f"per_state={args.per_state}, min_indecisive={args.min_indecisive}, "
          f"max_rounds={args.max_rounds})", flush=True)
    dfa, _tree = synthesize_direct_lstar_fnr(
        pst,
        acc_threshold=args.acc_threshold,
        max_rounds=args.max_rounds,
        counterexample_probes=args.counterexample_probes,
        per_state=args.per_state,
        min_indecisive=args.min_indecisive,
    )
    print(f"[LEN40] done: final DFA has {len(dfa.states)} states", flush=True)


if __name__ == "__main__":
    main()
