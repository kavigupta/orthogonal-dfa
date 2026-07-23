"""Attribute every oracle membership query to its origin in the L* pipeline.

Wraps a benchmark oracle and, on each membership_query, walks the call stack to
bucket the query by (a) its direct call site and (b) the high-level phase driving
it.  Prints a breakdown so we can see where queries actually come from before
deciding what else is worth caching.

Usage:
    python scripts/profile_query_origins.py                  # default: modulo subseq
    python scripts/profile_query_origins.py poor_case        # one benchmark
    python scripts/profile_query_origins.py modulo subseq    # several, in one run
"""

import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

# Profile THIS working tree, not whatever `orthogonal_dfa` is pip-installed
# (it's an editable install pointing at a different clone). Insert before the
# library imports so the local source wins.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

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

# The bulk prefix x suffix queries live in MaskTable. We support both layouts
# it has had: the EAGER one (queries in `_query`, reached via `intern_suffix`
# for a new suffix column or `add_prefixes` for a new prefix row) and the LAZY
# one (per-cell queries in `_ensure`, reached via `observed_masks`/`column`;
# `add_prefixes` queries fully-observed family columns directly). `predict` and
# `relabel` query the oracle directly in both. Attribution below is by role, so
# it survives either layout.

# A new-suffix-column query goes through one of these functions.
COL_SITES = ("_query", "intern_suffix", "_ensure", "observed_masks", "column")

# For predict, the enclosing high-level phase (searched outward, first match wins).
PREDICT_PHASES = [
    "estimate_agreement_rate",
    "generate_counterexamples",
    "enrich_underrepresented_leaves",
]

# For a new-suffix column, the method that needed the suffix (searched outward,
# first match wins). Ordered most-specific first, because the low-level helpers
# (compute_decision*) are reached from several distinct phases — matching the
# outer method before them splits the otherwise-broad decision bucket by caller.
SUFFIX_TRIGGERS = [
    "identify_cluster_around",
    "compute_fnr",
    "_give_up_check",
    "_resolve",
    "_split",
    "classify_states_with_decision_tree",
    "_sample_suffix",
    "sample_more_suffixes",
    "sample_suffix_family",
    "build",
    "discover_states",
    "compute_decision_from_strings",
    "compute_decision",
]

# For a new-prefix row (add_prefixes), what added the prefixes.
ROW_TRIGGERS = [
    "add_counterexample_prefixes",
    "enrich_underrepresented_leaves",
    "sample_more_prefixes",
]


class ProfilingOracle(Oracle):
    def __init__(self, inner: Oracle):
        self._inner = inner
        # label -> Counter(batch size -> calls at that size). Forward passes
        # need the sizes, not just the totals.
        self.sizes = defaultdict(Counter)
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
        frame = sys._getframe(3)  # skip _attribute + _record + the query method
        depth = 0
        while frame is not None and depth < 60:
            names.append(frame.f_code.co_name)
            if frame.f_code.co_name == "locate_incorrect_point":
                locate_line = frame.f_lineno
            frame = frame.f_back
            depth += 1
        name_set = set(names)
        # Attribute by role, checking in priority order so the layout doesn't
        # matter. DT classify and denoise query the oracle directly. A new prefix
        # row is checked before the column sites because in the eager layout the
        # row query still passes through `_query` (add_prefixes -> _query).
        if name_set & {"predict", "classify_many"}:
            phase = next((p for p in PREDICT_PHASES if p in name_set), "other")
            sub = f" [locate L{locate_line}]" if locate_line else ""
            how = "batched" if "classify_many" in name_set else "one string"
            return f"DT classify {how} <- {phase}{sub}"
        if "relabel" in name_set:
            return "denoise: fresh samples per state (relabel)"
        if "add_prefixes" in name_set:
            trig = next((t for t in ROW_TRIGGERS if t in name_set), "other")
            return f"matrix row: new prefix over all suffixes (add_prefixes) <- {trig}"
        if any(s in name_set for s in COL_SITES):
            trig = next((t for t in SUFFIX_TRIGGERS if t in name_set), "other")
            return f"matrix col: new suffix over all prefixes <- {trig}"
        return f"UNATTRIBUTED ({names[:4]})"

    def _record(self, n):
        self.total += n
        self.sizes[self._attribute()][n] += 1

    def membership_query(self, string):
        self._record(1)
        return self._inner.membership_query(string)

    def membership_queries(self, strings):
        self._record(len(strings))
        return self._inner.membership_queries(strings)


def profile_one(name: str) -> None:
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
    # A call of n strings costs ceil(n / cap) passes, so a site issuing many
    # small calls costs far more than its query share suggests. Sorted by fp at
    # the largest cap, where under-filled batches hurt most.
    caps = (32, 128, 1024)
    fp = {lab: {c: sum(math.ceil(n / c) * k for n, k in sz.items()) for c in caps}
          for lab, sz in o.sizes.items()}
    head = (f"{'queries':>12}{'':7}{'calls':>9}{'avg sz':>8}"
            + "".join(f"{f'fp@{c}':>10}" for c in caps) + "  site")
    print(head)
    for label in sorted(o.sizes, key=lambda lab: -fp[lab][caps[-1]]):
        sz = o.sizes[label]
        strings, calls = sum(n * k for n, k in sz.items()), sum(sz.values())
        print(f"{strings:>12,}{100 * strings / o.total:6.1f}%{calls:>9,}"
              f"{strings / calls:>8.0f}"
              + "".join(f"{fp[label][c]:>10,}" for c in caps) + f"  {label}")
    for c in caps:
        actual = sum(fp[lab][c] for lab in fp)
        ideal = math.ceil(o.total / c)
        print(f"fp@{c}: {actual:,} (ideal {ideal:,}, "
              f"{100 * ideal / actual:.0f}% packed)")


def main():
    # One or more benchmark names (default: a representative couple).
    names = sys.argv[1:] or ["modulo", "subseq"]
    for name in names:
        profile_one(name)


if __name__ == "__main__":
    main()
