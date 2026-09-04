"""Instrumented multi-round synthesis on the len40 spliceai oracle.

Runs the FULL counterexample_driven_synthesis (unfixed baseline) for several rounds
and dumps a rich pickle per round so we can investigate later whether -- and how --
later rounds improve the split.  Captures per round: the DFA, the discrimination
tree, decision boundary / evidence margin / accept+reject thresholds, est accuracy,
the prefix pool, the harvested `indecisive` boundary strings (the multi-round
feedback), the round's suffix family, and the DFA's call / oracle labels / frame
patterns on a FIXED eval set with phi(DFA,oracle) and phi(DFA,frame-rule).
"""
import argparse, os, pickle, numpy as np

from orthogonal_dfa.data.exon import default_exon
from orthogonal_dfa.l_star.counterexample_synthesis import counterexample_driven_synthesis
from orthogonal_dfa.l_star.examples.gate_composition_residual import gate_residual_oracle
from orthogonal_dfa.l_star.learn import build_pst
from orthogonal_dfa.l_star.transition_resolver import TransitionResolver
from orthogonal_dfa.spliceai.load_model import load_spliceai
from orthogonal_dfa.superlanguage.oracle import LiftedOracle
from orthogonal_dfa.superlanguage.sampler import SuperSampler
from orthogonal_dfa.superlanguage.vocabulary import KmerVocabulary

STOPS = {(3, 0, 2), (3, 0, 0), (3, 2, 0)}

# ---- capture each round's resolver so we can read its harvested `indecisive` ----
_RESOLVERS = []
_orig_init = TransitionResolver.__init__
def _capturing_init(self, pst, vs):
    _orig_init(self, pst, vs)
    _RESOLVERS.append(self)
TransitionResolver.__init__ = _capturing_init


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

    # fixed eval set (same every round): oracle labels, frame patterns, frame rule
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

    print(f"[seed {args.seed}] round | states | accept | phi(DFA,oracle) | phi(DFA,frame01) | "
          f"est | pool | |indec| | boundary | margin", flush=True)
    for i, (dfa, dt, true_acc, boundary, classifier) in enumerate(
        counterexample_driven_synthesis(pst, acc_threshold=args.acc_threshold)
    ):
        call = np.array([bool(dfa.accepts_input(w)) for w in supers], dtype=float)
        resolver = _RESOLVERS[-1] if _RESOLVERS else None
        indec = set(resolver.indecisive) if resolver is not None else set()
        rec = dict(
            round=i, seed=args.seed,
            dfa=dfa, dt=dt,
            boundary=float(pst.decision_boundary),
            evidence_margin=float(pst.evidence_margin),
            accept_thresh=float(pst.accept_thresh),
            reject_thresh=float(pst.reject_thresh),
            true_acc=float(true_acc),
            num_prefixes=int(pst.num_prefixes),
            prefixes=[bytes(p) for p in pst.table.prefixes],
            indecisive=[bytes(x) for x in indec],
            n_states=len(dfa.states),
            call=call, ora=ora, fpat=fpat,
            phi_oracle=phi(call, ora), phi_frame01=phi(call, f01),
            accept_rate=float(call.mean()),
        )
        with open(os.path.join(args.dump_dir, f"round_{i:02d}.pkl"), "wb") as fh:
            pickle.dump(rec, fh)
        print(f"  round {i}: {len(dfa.states):2d} | {call.mean():.3f} | {phi(call, ora):+.3f} | "
              f"{phi(call, f01):+.3f} | {true_acc:.3f} | {pst.num_prefixes} | {len(indec)} | "
              f"{pst.decision_boundary:.3f} | {pst.evidence_margin:.3f}", flush=True)
        if i + 1 >= args.rounds:
            break
    print(f"[seed {args.seed}] DONE", flush=True)


if __name__ == "__main__":
    main()
