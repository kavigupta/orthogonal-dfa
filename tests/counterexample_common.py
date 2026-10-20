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


def _constant(value):
    return lambda _: value


def quiet_config(**changes):
    """What the round reads of a config, set so no test fires."""
    quiet = {
        "k": 0,
        "ends_rate": 1.0,
        "ends_alternative": 1.0,
        "stuck_rate": 1.0,
        "stuck_alternative": 1.0,
        "middle_rate": 1.0,
        "middle_alternative": 1.0,
        "disagreement_rate": 0.0,
        "disagreement_alternative": 0.0,
        "level": 0.01,
        "search_share": 1e9,
        "max_probes": 100,
    }
    config = {**quiet, **changes}
    for per_unit in (
        "ends_rate",
        "ends_alternative",
        "middle_rate",
        "middle_alternative",
    ):
        config[per_unit] = _constant(config[per_unit])
    return SimpleNamespace(**config)


def quiet_search(cut, edges, **changes):
    search = CounterexampleSearch(quiet_config(**changes), cut, MidfixTree([]))
    search.hypothesis.edges = dict(edges)
    return search
