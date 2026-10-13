"""Hand-built hypotheses and scripted reads for the counterexample search's tests."""

from types import SimpleNamespace

from orthogonal_dfa.l_star.counterexample_search import CounterexampleSearch
from orthogonal_dfa.l_star.midfix_tree import MidfixTree

EVEN, ODD = (False,), (True,)

#: Parity of the ones, which the root's cut alone reads.
PARITY_EDGES = {
    (EVEN, 0): (EVEN, b""),
    (EVEN, 1): (ODD, b""),
    (ODD, 0): (ODD, b"\x01"),
    (ODD, 1): (EVEN, b"\x01"),
}

#: Parity with the edge out of an odd leaf by a 0 pointing the wrong way.
WRONG_ODD_0 = {**PARITY_EDGES, (ODD, 0): (EVEN, b"\x01")}


def parity_cut(undecided):
    """Accepts an odd count of ones, and leaves ``undecided`` undecided."""
    undecided = set(undecided)

    def cut(z):
        return None if z in undecided else sum(z) % 2 == 1

    return cut


def mod3_cut(z):
    """Accepts a count of ones divisible by 3: the root's leaves conflate 1 and 2."""
    return sum(z) % 3 == 0


def quiet_config(**changes):
    """What the round reads of a config, set so no test fires and a fix takes two
    records."""
    quiet = {
        "k": 0,
        "m": 2,
        "max_leaves": 10,
        "start_rate": 1.0,
        "edge_rate": 0.0,
        "middle_rate": 1.0,
        "disagreement_rate": 0.0,
        "level": 0.01,
        "first_look": 1,
        "edge_excess": 1e9,
        "edge_traffic": 0.0,
        "search_share": 1e9,
        "max_probes": 100,
    }
    config = {**quiet, **changes}
    start_rate, middle_rate = config["start_rate"], config["middle_rate"]
    config["start_rate"] = lambda depth: start_rate
    config["middle_rate"] = lambda queries_per_search: middle_rate
    return SimpleNamespace(**config)


def quiet_search(cut, edges, **changes):
    search = CounterexampleSearch(quiet_config(**changes), cut, MidfixTree([]))
    search.hypothesis.edges = dict(edges)
    return search
