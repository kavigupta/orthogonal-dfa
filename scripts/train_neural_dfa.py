#!/usr/bin/env python
"""Run the amortised-state-predictor learner on the L* benchmarks.

    python scripts/train_neural_dfa.py parity
    python scripts/train_neural_dfa.py subseq modulo --signal 0.3 --states 48

Reports noiseless accuracy against the same harness the L* tests use, plus the distinct
oracle query count so the result is comparable to `scripts/count_queries.py`.
"""

import argparse
import json
import os
import sys
import time

# Running this as a script only puts scripts/ on the path; the eval harness lives in
# tests/. Same approach as scripts/count_queries.py.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from orthogonal_dfa.l_star.examples.benchmark_generator import (
    DFAOracle,
    sample_balanced_benchmark,
)
from orthogonal_dfa.l_star.examples.bernoulli_parity import (
    BernoulliParityOracle,
    BernoulliRegex,
)
from orthogonal_dfa.l_star.neural.train import NeuralConfig, train_neural_dfa
from orthogonal_dfa.l_star.structures import (
    AsymmetricBernoulli,
    Oracle,
    SymmetricBernoulli,
)

BENCHMARKS = {
    "parity": lambda nm, s: BernoulliParityOracle(nm, s),
    "modulo": lambda nm, s: BernoulliParityOracle(
        nm, s, modulo=9, allowed_moduluses=(3, 6)
    ),
    "subseq": lambda nm, s: BernoulliRegex(nm, s, regex=r".*1010101.*"),
    "two_subseq": lambda nm, s: BernoulliRegex(nm, s, regex=r".*1111.*1111.*"),
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


def generated_benchmark(seed, num_outer_states):
    outer, _, _ = sample_balanced_benchmark(
        seed,
        alphabet_size=2,
        num_inner_states=4,
        num_outer_states=num_outer_states,
        probe_length=40,
        min_accept_or_reject=0.2,
    )
    return lambda nm, s: DFAOracle(nm, s, outer)


def main():
    # Imported here so the sys.path fix above is in effect and isort stays happy.
    from tests.test_lstar import evaluate_accuracy

    parser = argparse.ArgumentParser()
    parser.add_argument("tasks", nargs="*", default=["parity"])
    parser.add_argument("--signal", type=float, default=0.3)
    parser.add_argument("--asymmetric", nargs=2, type=float, metavar=("P0", "P1"))
    parser.add_argument("--states", type=int, default=32)
    parser.add_argument("--strings", type=int, default=3000)
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lags", type=int, default=NeuralConfig.num_lags)
    parser.add_argument(
        "--lambda-ext", type=float, default=NeuralConfig.lambda_external
    )
    parser.add_argument(
        "--warmup", type=int, default=NeuralConfig.internal_warmup_rounds
    )
    parser.add_argument("--beta", type=float, default=NeuralConfig.beta_balance)
    parser.add_argument("--error-boost", type=float, default=NeuralConfig.error_boost)
    parser.add_argument("--generated", type=int, metavar="N_STATES")
    args = parser.parse_args()

    noise = (
        AsymmetricBernoulli(*args.asymmetric)
        if args.asymmetric
        else SymmetricBernoulli(p_correct=0.5 + args.signal)
    )

    tasks = args.tasks
    if args.generated is not None:
        tasks = [f"generated{args.generated}"]

    for name in tasks:
        if name.startswith("generated"):
            creator = generated_benchmark(args.seed, int(name[len("generated") :]))
        else:
            creator = BENCHMARKS[name]
        oracle = CountingOracle(creator(noise, args.seed))
        cfg = NeuralConfig(
            num_states=args.states,
            num_strings=args.strings,
            rounds=args.rounds,
            seed=args.seed,
            num_lags=args.lags,
            lambda_external=args.lambda_ext,
            internal_warmup_rounds=args.warmup,
            beta_balance=args.beta,
            error_boost=args.error_boost,
        )
        started = time.time()
        dfa, info = train_neural_dfa(oracle, cfg)
        info.update(
            task=name,
            accuracy=evaluate_accuracy(dfa, creator, symbols=oracle.alphabet_size),
            oracle_queries=oracle.count,
            oracle_distinct=len(oracle.distinct),
            seconds=round(time.time() - started, 1),
        )
        print(json.dumps(info, indent=2, default=str))


if __name__ == "__main__":
    main()
