"""Count oracle membership queries for a few ~1min L* synthesis benchmarks.

Wraps each benchmark's oracle in a counter and runs the exact synthesis path the
tests use (``tests.test_lstar.compute_dfa_for_oracle``).  ``membership_query`` is a
pure function of the string (noise is a stable hash, no RNG), so the synthesis path
is deterministic given the seed and independent of how many queries we skip via
caching; running this under two git states therefore isolates the query delta.

Usage:
    python scripts/count_queries.py                 # run the default set
    python scripts/count_queries.py modulo subseq   # run named benchmarks only
"""

import sys
import time

from automata.fa.dfa import DFA

from orthogonal_dfa.l_star.examples.benchmark_generator import DFAOracle
from orthogonal_dfa.l_star.examples.bernoulli_parity import (
    BernoulliParityOracle,
    BernoulliRegex,
)
from orthogonal_dfa.l_star.structures import Oracle

# Reuse the exact test harness so configs match the ~1min regression benchmarks.
from tests.test_lstar import compute_dfa_for_oracle, evaluate_accuracy


class CountingOracle(Oracle):
    """Delegates to an inner oracle and counts membership_query calls."""

    def __init__(self, inner: Oracle):
        self._inner = inner
        self.count = 0

    @property
    def alphabet_size(self) -> int:
        return self._inner.alphabet_size

    def membership_query(self, string):
        self.count += 1
        return self._inner.membership_query(string)


_ANOTHER_POOR_DFA = DFA(
    states=set(range(10)),
    input_symbols={0, 1},
    transitions={
        0: {1: 8, 0: 0}, 1: {1: 1, 0: 1}, 2: {1: 1, 0: 6}, 3: {1: 9, 0: 2},
        4: {1: 3, 0: 8}, 5: {1: 8, 0: 4}, 6: {1: 3, 0: 9}, 7: {1: 8, 0: 6},
        8: {1: 8, 0: 5}, 9: {1: 3, 0: 7},
    },
    initial_state=0,
    final_states={1},
    allow_partial=False,
)


# name -> (oracle_creator, min_signal_strength, symbols)
BENCHMARKS = {
    "modulo": (
        lambda nm, s: BernoulliParityOracle(nm, s, modulo=9, allowed_moduluses=(3, 6)),
        0.3,
        2,
    ),
    "subseq": (
        lambda nm, s: BernoulliRegex(nm, s, regex=r".*1010101.*"),
        0.3,
        2,
    ),
    "two_subseq": (
        lambda nm, s: BernoulliRegex(nm, s, regex=r".*1111.*1111.*"),
        0.3,
        2,
    ),
    "poor_case": (
        lambda nm, s: DFAOracle(nm, s, _ANOTHER_POOR_DFA),
        0.3,
        2,
    ),
}


def run_benchmark(name, seed=0):
    oracle_creator, signal, symbols = BENCHMARKS[name]
    counters = []

    def counting_creator(noise_model, s):
        oracle = CountingOracle(oracle_creator(noise_model, s))
        counters.append(oracle)
        return oracle

    start = time.time()
    _, dfa, _ = compute_dfa_for_oracle(
        counting_creator, min_signal_strength=signal, seed=seed
    )
    elapsed = time.time() - start
    # The synthesis oracle is the first (and only) one created during synthesis;
    # accuracy evaluation below creates its own noiseless oracle, uncounted here.
    queries = sum(c.count for c in counters)
    acc = evaluate_accuracy(dfa, oracle_creator, symbols=symbols)
    return {
        "name": name,
        "queries": queries,
        "elapsed": elapsed,
        "num_states": len(dfa.states),
        "accuracy": acc,
    }


def main():
    names = sys.argv[1:] or list(BENCHMARKS)
    rows = [run_benchmark(n) for n in names]
    print("\n\n===== QUERY COUNT SUMMARY =====")
    header = f"{'benchmark':<14}{'queries':>12}{'states':>8}{'acc':>8}{'sec':>8}"
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r['name']:<14}{r['queries']:>12,}{r['num_states']:>8}"
            f"{r['accuracy']:>8.3f}{r['elapsed']:>8.1f}"
        )
    print(f"\nTOTAL queries: {sum(r['queries'] for r in rows):,}")


if __name__ == "__main__":
    main()
