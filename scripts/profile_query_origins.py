"""Attribute every oracle membership query to its origin in the L* pipeline.

Wraps a benchmark oracle and, on each membership_query, walks the call stack to
bucket the query by (a) its direct call site and (b) the high-level phase driving
it.  Prints a breakdown so we can see where queries actually come from before
deciding what else is worth caching.

Usage:
    python scripts/profile_query_origins.py            # default: subseq
    python scripts/profile_query_origins.py poor_case
"""

import sys
from collections import Counter

from automata.fa.dfa import DFA

from orthogonal_dfa.l_star.examples.benchmark_generator import DFAOracle
from orthogonal_dfa.l_star.examples.bernoulli_parity import (
    BernoulliParityOracle,
    BernoulliRegex,
)
from orthogonal_dfa.l_star.structures import Oracle
from tests.test_lstar import compute_dfa_for_oracle, evaluate_accuracy

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
        0.3, 2,
    ),
    "subseq": (
        lambda nm, s: BernoulliRegex(nm, s, regex=r".*1010101.*"), 0.3, 2,
    ),
    "two_subseq": (
        lambda nm, s: BernoulliRegex(nm, s, regex=r".*1111.*1111.*"), 0.3, 2,
    ),
    "poor_case": (
        lambda nm, s: DFAOracle(nm, s, _ANOTHER_POOR_DFA), 0.3, 2,
    ),
}

# Direct call sites: the function whose body contains the membership_query call.
DIRECT_SITES = {
    "compute_mask": "matrix col: new suffix over all prefixes (compute_mask)",
    "add_prefixes": "matrix row: new prefix over all suffixes (add_prefixes)",
    "relabel": "denoise: fresh samples per state (relabel)",
    "predict": "DT classify one string (TriPredicate.predict)",
}

# For predict, the enclosing high-level phase (searched outward, first match wins).
PREDICT_PHASES = [
    "estimate_agreement_rate",
    "generate_counterexamples",
    "enrich_underrepresented_leaves",
]

# For compute_mask, what triggered the new suffix (searched outward).
MASK_TRIGGERS = [
    "prepend_to_all",          # building child/transition predicates during discovery
    "sample_more_suffixes",    # growing the suffix family
    "discover_states",         # the initial empty-suffix / seed recording
]


class ProfilingOracle(Oracle):
    def __init__(self, inner: Oracle):
        self._inner = inner
        self.buckets = Counter()
        self.total = 0

    @property
    def alphabet_size(self) -> int:
        return self._inner.alphabet_size

    def _attribute(self):
        # Collect (co_name, lineno) on the stack, innermost first, keeping the
        # locate_incorrect_point frame so predict queries can be sub-attributed to
        # the exact call site (empty-string s0 vs full-string check vs binary search).
        names = []
        locate_line = None
        frame = sys._getframe(2)  # skip _attribute + membership_query
        depth = 0
        while frame is not None and depth < 60:
            names.append(frame.f_code.co_name)
            if frame.f_code.co_name == "locate_incorrect_point":
                locate_line = frame.f_lineno
            frame = frame.f_back
            depth += 1
        name_set = set(names)
        for site, label in DIRECT_SITES.items():
            if site in name_set:
                if site == "predict":
                    phase = next((p for p in PREDICT_PHASES if p in name_set), "other")
                    sub = f" [locate L{locate_line}]" if locate_line else ""
                    return f"{label} <- {phase}{sub}"
                if site == "compute_mask":
                    trig = next((t for t in MASK_TRIGGERS if t in name_set), "other")
                    return f"{label} <- {trig}"
                return label
        return f"UNATTRIBUTED ({names[:4]})"

    def membership_query(self, string):
        self.total += 1
        self.buckets[self._attribute()] += 1
        return self._inner.membership_query(string)


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "subseq"
    oracle_creator, signal, symbols = BENCHMARKS[name]
    holder = {}

    def creator(noise_model, s):
        o = ProfilingOracle(oracle_creator(noise_model, s))
        holder["o"] = o
        return o

    _, dfa, _ = compute_dfa_for_oracle(creator, min_signal_strength=signal, seed=0)
    acc = evaluate_accuracy(dfa, oracle_creator, symbols=symbols)
    o = holder["o"]

    print(f"\n\n===== QUERY ORIGINS: {name} "
          f"(states={len(dfa.states)}, acc={acc:.3f}) =====")
    print(f"total queries: {o.total:,}\n")
    for label, n in o.buckets.most_common():
        print(f"{n:>12,}  {100*n/o.total:5.1f}%  {label}")


if __name__ == "__main__":
    main()
