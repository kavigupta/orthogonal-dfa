import unittest
from types import SimpleNamespace
from unittest import mock

from orthogonal_dfa.l_star import cluster
from orthogonal_dfa.l_star.cluster import ADMITTED, Judged, sample_suffix_family

FAMILY_SIZE = 4
FNR_LIMIT = 0.05


def _search(fnrs, patience):
    """The family each round returns is [round] * FAMILY_SIZE, judged at fnrs[round], with boundary float(round)."""
    rounds = iter(range(len(fnrs)))
    pst = SimpleNamespace(
        config=SimpleNamespace(
            min_signal_strength=0.3,
            fnr_plateau_patience=patience,
            fnr_plateau_tol=0.01,
        ),
        decision_boundary=0.5,
        fnr_limit=FNR_LIMIT,
        evidence_margin=0.0,
        suffix_pool=[],
        sample_more_suffixes=lambda amount, reference: (amount, amount),
        sample_more_prefixes=lambda: None,
    )

    def identify(_pst, _v, size, _boundary, _anchored):
        index = next(rounds)
        return [index] * size, float(index)

    def judge(_pst, _gate, _v, vs, _size):
        return Judged(vs, fnrs[vs[0]], "too high", ADMITTED, None)

    with mock.patch.multiple(
        cluster,
        identify_cluster_around=identify,
        judge_family=judge,
        smallest_readable_family=lambda signal, boundary, rates: FAMILY_SIZE,
        read_rates=lambda pst, boundary: None,
        AcceptPreservingGate=lambda config, state: None,
    ), mock.patch("builtins.print"):
        return sample_suffix_family(pst, 0, None)


class TestFnrPlateau(unittest.TestCase):
    def test_returns_the_best_family_and_its_boundary(self):
        vs, boundary = _search([0.5, 0.3, 0.295, 0.3, 0.29, 0.1], patience=3)
        self.assertEqual([1] * FAMILY_SIZE, vs)
        self.assertEqual(1.0, boundary)

    def test_an_improving_search_keeps_going_until_the_limit(self):
        vs, _ = _search([0.5, 0.4, 0.3, 0.2, 0.1, 0.04], patience=2)
        self.assertEqual([5] * FAMILY_SIZE, vs)

    def test_unset_patience_waits_for_the_limit(self):
        vs, _ = _search([0.3] * 10 + [0.04], patience=None)
        self.assertEqual([10] * FAMILY_SIZE, vs)


if __name__ == "__main__":
    unittest.main()
