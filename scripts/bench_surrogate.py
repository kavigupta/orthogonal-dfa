#!/usr/bin/env python
"""Run the surrogate learner across a spread of target DFAs and report accuracy vs queries.

    python scripts/bench_surrogate.py                 # hand-written + generated
    python scripts/bench_surrogate.py --generated-only --seeds 8
    python scripts/bench_surrogate.py --signal 0.2    # harder noise

Accuracy is measured by the same harness the L* tests use: 10k uniform length-40 strings
against a NOISELESS oracle.
"""

import argparse
import json
import os
import sys
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from orthogonal_dfa.l_star.examples.benchmark_generator import (
    DFAOracle,
    sample_balanced_benchmark,
)
from orthogonal_dfa.l_star.examples.bernoulli_parity import (
    BernoulliParityOracle,
    BernoulliRegex,
)
from orthogonal_dfa.l_star.neural.surrogate import SurrogateConfig, learn_dfa
from orthogonal_dfa.l_star.structures import Oracle, SymmetricBernoulli

HAND_WRITTEN = {
    "parity": (lambda nm, s: BernoulliParityOracle(nm, s), 2),
    "modulo9": (
        lambda nm, s: BernoulliParityOracle(nm, s, modulo=9, allowed_moduluses=(3, 6)),
        9,
    ),
    "subseq": (lambda nm, s: BernoulliRegex(nm, s, regex=r".*1010101.*"), 8),
    "two_subseq": (lambda nm, s: BernoulliRegex(nm, s, regex=r".*1111.*1111.*"), 9),
    "alternating": (lambda nm, s: BernoulliRegex(nm, s, regex=r"(10)*1?"), 3),
    "endswith": (lambda nm, s: BernoulliRegex(nm, s, regex=r".*110"), 4),
}


class CountingOracle(Oracle):
    def __init__(self, inner):
        self._inner = inner
        self.count = 0
        self.distinct = set()

    @property
    def alphabet_size(self):
        return self._inner.alphabet_size

    def membership_query(self, string):
        self.count += 1
        self.distinct.add(tuple(string))
        return self._inner.membership_query(string)

    def membership_queries(self, strings):
        self.count += len(strings)
        self.distinct.update(tuple(s) for s in strings)
        return self._inner.membership_queries(strings)


# (inner, outer) pairs the existing L* tests are known to be able to sample.
GENERATED_SHAPES = ((12, 10), (20, 18))


def generated(seed, num_inner_states, num_outer_states):
    outer, _, _ = sample_balanced_benchmark(
        seed,
        alphabet_size=2,
        num_inner_states=num_inner_states,
        num_outer_states=num_outer_states,
        probe_length=40,
        min_accept_or_reject=0.15,
        max_attempts=50000,
    )
    return (lambda nm, s: DFAOracle(nm, s, outer)), len(outer.states)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--signal", type=float, default=0.3)
    parser.add_argument("--seeds", type=int, default=4)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--states", type=int, default=32)
    parser.add_argument("--prefixes", type=int, default=1200)
    parser.add_argument("--generated-only", action="store_true")
    parser.add_argument("--hand-only", action="store_true")
    args = parser.parse_args()

    from tests.test_lstar import evaluate_accuracy

    noise = SymmetricBernoulli(p_correct=0.5 + args.signal)
    tasks = []
    if not args.generated_only:
        tasks += [(name, creator, n) for name, (creator, n) in HAND_WRITTEN.items()]
    if not args.hand_only:
        for num_inner, num_outer in GENERATED_SHAPES:
            for seed in range(args.seeds):
                try:
                    creator, n = generated(seed, num_inner, num_outer)
                except RuntimeError as exc:  # infeasible shape/seed; skip, don't abort
                    print(f"skip gen{num_outer}s{seed}: {exc}", flush=True)
                    continue
                tasks.append((f"gen{num_outer}s{seed}", creator, n))

    results = []
    for name, creator, true_states in tasks:
        oracle = CountingOracle(creator(noise, 0))
        cfg = SurrogateConfig(
            num_states=args.states,
            num_prefixes=args.prefixes,
            rounds=args.rounds,
        )
        started = time.time()
        try:
            dfa, info = learn_dfa(oracle, cfg, log=lambda *_: None)
            accuracy = evaluate_accuracy(dfa, creator, symbols=oracle.alphabet_size)
        except Exception:  # keep the sweep going; report the failure
            traceback.print_exc()
            results.append({"task": name, "error": True})
            continue
        row = {
            "task": name,
            "true_states": true_states,
            "learned_states": info["states"],
            "accuracy": round(accuracy, 4),
            "queries": len(oracle.distinct),
            "seconds": round(time.time() - started, 1),
        }
        results.append(row)
        print(json.dumps(row), flush=True)

    ok = [r for r in results if not r.get("error")]
    if ok:
        good = [r for r in ok if r["accuracy"] >= 0.97]
        print("\n===== SUMMARY =====")
        print(f"tasks: {len(ok)}   >=0.97 accuracy: {len(good)}")
        print(f"median accuracy: {sorted(r['accuracy'] for r in ok)[len(ok) // 2]}")
        print(f"median queries:  {sorted(r['queries'] for r in ok)[len(ok) // 2]:,}")


if __name__ == "__main__":
    main()
