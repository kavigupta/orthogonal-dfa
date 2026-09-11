"""Instrumented multi-round synthesis on the len40 spliceai oracle, via the tracker.

Runs ``counterexample_driven_synthesis`` with a ``RecordingTracker`` (no monkeypatching,
no yield loop) and writes one ``round_{i}.pkl`` per round from the tracker's streams:
dfa, dt, band thresholds, est, prefix pool, harvested indecisive, plus eval
phi(DFA, oracle) / phi(DFA, frame-rule) on a fixed eval set.

Depends only on the public tracker interface (``on_pool_resolved`` carries the pool and
the harvested boundary strings), so it works off main.
"""
import argparse
import os
import pickle

import numpy as np

from orthogonal_dfa.data.exon import default_exon
from orthogonal_dfa.l_star.counterexample_synthesis import counterexample_driven_synthesis
from orthogonal_dfa.l_star.examples.gate_composition_residual import gate_residual_oracle
from orthogonal_dfa.l_star.learn import build_pst
from orthogonal_dfa.l_star.tracker import RecordingTracker
from orthogonal_dfa.spliceai.load_model import load_spliceai
from orthogonal_dfa.superlanguage.oracle import LiftedOracle
from orthogonal_dfa.superlanguage.sampler import SuperSampler
from orthogonal_dfa.superlanguage.vocabulary import KmerVocabulary

STOPS = {(3, 0, 2), (3, 0, 0), (3, 2, 0)}


def frames_closed(seq):
    return tuple(
        any(tuple(seq[ph:][i:i + 3]) in STOPS for i in range(0, len(seq[ph:]) - 2, 3))
        for ph in range(3)
    )


def phi(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return 0.0 if a.std() == 0 or b.std() == 0 else float(np.corrcoef(a, b)[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rounds", type=int, default=8)
    ap.add_argument("--acc-threshold", type=float, default=0.98)
    ap.add_argument("--fnr-limit", type=float, default=0.10)
    ap.add_argument("--min-signal-strength", type=float, default=0.05)
    ap.add_argument("--n-eval", type=int, default=4000)
    ap.add_argument("--dump-dir", required=True)
    args = ap.parse_args()
    os.makedirs(args.dump_dir, exist_ok=True)

    base = gate_residual_oracle(
        default_exon, load_spliceai(400, 0), length=40, len_lo=35, len_hi=85
    )
    vocab = KmerVocabulary(kmers=((3, 0, 2), (3, 0, 0), (3, 2, 0)), base_alphabet_size=4)

    def oracle_creator(_nm, s):
        return LiftedOracle(base, vocab, seed=s)

    pst = build_pst(
        oracle_creator, min_signal_strength=args.min_signal_strength, seed=args.seed,
        sampler=SuperSampler(vocab, 36),
    )
    pst.config.fnr_limit = args.fnr_limit

    # fixed eval set (same for every round): oracle labels, frame patterns, frame rule
    samp = SuperSampler(vocab, 36)
    rng = np.random.default_rng(args.seed + 999)
    supers = [samp.sample(rng, vocab.alphabet_size) for _ in range(args.n_eval)]
    bases = vocab.compile_many(supers, [np.random.default_rng(i) for i in range(args.n_eval)])
    ora = np.asarray(base.membership_queries([bytes(b) for b in bases])).astype(float)
    fpat = [frames_closed(b) for b in bases]
    f01 = np.array([1.0 if not (p[0] and p[1]) else 0.0 for p in fpat])  # accept iff not f0&f1
    afc = np.array([1.0 if sum(p) == 3 else 0.0 for p in fpat])
    print(f"[seed {args.seed}] eval ready (n={args.n_eval}); "
          f"phi(oracle,frame01)={phi(ora, f01):+.3f} phi(oracle,allframes)={phi(ora, afc):+.3f}", flush=True)

    tracker = RecordingTracker()
    counterexample_driven_synthesis(
        pst, acc_threshold=args.acc_threshold, tracker=tracker, max_rounds=args.rounds
    )

    print(f"[seed {args.seed}] round | states | accept | phi(DFA,oracle) | phi(DFA,frame01) | "
          f"est | pool | |indec| | boundary", flush=True)
    for i in range(len(tracker.consistency)):
        dfa, dt = tracker.hypotheses[i]
        _, boundary = tracker.families[i]
        clf = tracker.classifiers[i]
        prefixes, indecisive = tracker.pools[i]
        call = np.array([bool(dfa.accepts_input(w)) for w in supers], dtype=float)
        margin = (clf.accept_thresh - clf.reject_thresh) / 2
        rec = dict(
            round=i, seed=args.seed,
            dfa=dfa, dt=dt,
            boundary=float(boundary),
            evidence_margin=float(margin),
            accept_thresh=float(clf.accept_thresh),
            reject_thresh=float(clf.reject_thresh),
            true_acc=float(tracker.consistency[i]),
            num_prefixes=len(prefixes),
            prefixes=list(prefixes),
            indecisive=list(indecisive),
            n_states=len(dfa.states),
            call=call, ora=ora, fpat=fpat,
            phi_oracle=phi(call, ora), phi_frame01=phi(call, f01),
            accept_rate=float(call.mean()),
        )
        with open(os.path.join(args.dump_dir, f"round_{i:02d}.pkl"), "wb") as fh:
            pickle.dump(rec, fh)
        print(f"  round {i}: {len(dfa.states):2d} | {call.mean():.3f} | {phi(call, ora):+.3f} | "
              f"{phi(call, f01):+.3f} | {tracker.consistency[i]:.3f} | {len(prefixes)} | "
              f"{len(indecisive)} | {boundary:.3f}", flush=True)
    print(f"[seed {args.seed}] DONE", flush=True)


if __name__ == "__main__":
    main()
