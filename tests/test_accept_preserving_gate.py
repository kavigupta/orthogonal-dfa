"""The gate holds a family to every population: it admits only when the share
of each population's distribution the family's cut misclassifies is bounded,
by max_coverage_error on the uniform pool and by a half elsewhere, and it names
the population that stops it."""

import itertools
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

from orthogonal_dfa.l_star import cluster
from orthogonal_dfa.l_star.cluster import (
    ADMITTED,
    DRIFTED,
    UNCERTIFIED,
    alignment_size,
    drift_verdict,
    misclassified_bounds,
    prefixes_to_certify,
)
from orthogonal_dfa.l_star.mask_table import UNIFORM
from orthogonal_dfa.l_star.suffix_groups import aligned_suffixes

#: The rates the round reads at, boundary -/+ the signal.
P_0, P_1 = 0.35, 0.95
LEVEL = 0.05

_PST = SimpleNamespace(
    decision_boundary=(P_0 + P_1) / 2,
    config=SimpleNamespace(
        min_signal_strength=(P_1 - P_0) / 2, max_coverage_error=1 / 3
    ),
)


def _read_right(n):
    """n prefixes a side, each side read at its own class's rate."""
    return ((round(P_1 * n), n), (round(P_0 * n), n))


class TestEveryPopulationIsHeld(unittest.TestCase):
    def test_a_family_reading_every_population_right_is_admitted(self):
        counts = {UNIFORM: _read_right(500), ("state", 0): _read_right(500)}

        self.assertEqual((ADMITTED, None), drift_verdict(_PST, counts, LEVEL))

    def test_a_one_class_population_read_as_its_class_is_admitted(self):
        counts = {
            UNIFORM: _read_right(500),
            ("state", 0): ((round(P_1 * 500), 500), (0, 0)),
        }

        self.assertEqual(ADMITTED, drift_verdict(_PST, counts, LEVEL)[0])

    def test_a_one_class_population_is_read_at_the_offset_the_others_pin(self):
        # Alone, a side reading 0.35 could be all rejecting at p_0 = 0.35 or
        # mostly accepting at a lower p_0; the pool's two sides say which.
        counts = {
            UNIFORM: _read_right(500),
            ("state", 1): ((0, 0), (round(P_0 * 500), 500)),
        }

        self.assertEqual(ADMITTED, drift_verdict(_PST, counts, LEVEL)[0])

    def test_a_population_read_backwards_is_refused_by_name(self):
        # Every prefix the family rejects reads at the accepting rate.
        counts = {
            UNIFORM: _read_right(500),
            ("state", 3): ((0, 0), (round(P_1 * 500), 500)),
        }

        self.assertEqual((DRIFTED, ("state", 3)), drift_verdict(_PST, counts, LEVEL))

    def test_a_population_misread_past_a_half_is_refused(self):
        # Seven in ten of what the family rejects is accepting.
        mostly = round((0.3 * P_0 + 0.7 * P_1) * 500)
        counts = {UNIFORM: _read_right(500), "mixed": ((0, 0), (mostly, 500))}

        self.assertEqual((DRIFTED, "mixed"), drift_verdict(_PST, counts, LEVEL))

    def test_a_population_need_only_be_read_the_right_way_round(self):
        # Two in five misread: past max_coverage_error, short of a half.
        some = round((0.6 * P_0 + 0.4 * P_1) * 2000)
        counts = {UNIFORM: _read_right(2000), "mixed": ((0, 0), (some, 2000))}

        self.assertEqual(ADMITTED, drift_verdict(_PST, counts, LEVEL)[0])

    def test_the_pool_is_held_to_max_coverage_error(self):
        # The same two in five, now the whole pool; a state read right pins p_0.
        some = round((0.6 * P_0 + 0.4 * P_1) * 2000)
        counts = {UNIFORM: ((0, 0), (some, 2000)), ("state", 0): _read_right(2000)}

        self.assertEqual((DRIFTED, UNIFORM), drift_verdict(_PST, counts, LEVEL))

    def test_too_few_prefixes_leaves_the_family_uncertified(self):
        counts = {UNIFORM: _read_right(500), "small": _read_right(3)}

        self.assertEqual((UNCERTIFIED, "small"), drift_verdict(_PST, counts, LEVEL))

    def test_nothing_drawn_is_uncertified(self):
        self.assertEqual(
            (UNCERTIFIED, UNIFORM),
            drift_verdict(_PST, {UNIFORM: ((0, 0), (0, 0))}, LEVEL),
        )


class TestTheBoundHolds(unittest.TestCase):
    def test_the_bound_covers_the_misclassified_share(self):
        # The family accepts 60% of the population; a fifth of that is in fact
        # rejecting, and a tenth of what it rejects is accepting.
        rng = np.random.default_rng(0)
        accepted, wrong_in, wrong_out, n = 0.6, 0.2, 0.1, 400
        truth = accepted * wrong_in + (1 - accepted) * wrong_out
        missed = 0
        for _ in range(500):
            n_a = rng.binomial(n, accepted)
            hits_a = rng.binomial(n_a, P_1 - (P_1 - P_0) * wrong_in)
            hits_r = rng.binomial(n - n_a, P_0 + (P_1 - P_0) * wrong_out)
            counts = {UNIFORM: ((hits_a, n_a), (hits_r, n - n_a))}
            bound, _ = misclassified_bounds(_PST, counts, LEVEL)[UNIFORM]
            missed += bound < truth

        self.assertLessEqual(missed, 500 * LEVEL)


class TestHowMuchIsDrawn(unittest.TestCase):
    def test_a_cut_reading_right_certifies_at_the_size_and_not_half_of_it(self):
        size = alignment_size(_PST, 1, 1 / 3)
        level = cluster.ACCEPT_PRESERVING_ERROR_RATE / 2

        self.assertEqual(
            ADMITTED, drift_verdict(_PST, {UNIFORM: _read_right(size // 2)}, level)[0]
        )
        self.assertNotEqual(
            ADMITTED, drift_verdict(_PST, {UNIFORM: _read_right(size // 4)}, level)[0]
        )

    def test_the_top_up_is_the_first_that_settles_the_named_population(self):
        pst = SimpleNamespace(**vars(_PST), suffix_pool=list(range(8)))
        pst.config = SimpleNamespace(**vars(_PST.config), num_addtl_prefixes=2000)
        other = ((1, 6), (0, 0))
        drawn = 20

        def settled(n):
            counts = {"small": _read_right(n), "other": other}
            bound, at_rates = misclassified_bounds(pst, counts, LEVEL / 2)["small"]
            return bound <= 1 / 2 or at_rates > 1 / 2

        wanted = prefixes_to_certify(
            pst,
            {"small": _read_right(drawn), "other": other},
            "small",
            2 * drawn,
            range(8),
        )

        self.assertTrue(settled(drawn + wanted // 2))
        self.assertFalse(settled(drawn + wanted // 2 - drawn))


class TestARefusalRedrawsItsSample(unittest.TestCase):
    """A refusal scores the population's own reads, so a sample kept after it
    refuses would refuse every later family on the same bits."""

    def test_the_refusing_population_is_redrawn_and_the_rest_kept(self):
        draws = itertools.count()
        verdicts = iter([(DRIFTED, "state"), (ADMITTED, None)])
        read = []
        with mock.patch.multiple(
            cluster,
            population_labels=lambda state: [UNIFORM, "state"],
            prefixes_for_split=lambda pst, state, label, n: [next(draws)],
            certification_budget=lambda pst, vs: 1,
            alignment_size=lambda pst, populations, limit: 1,
            certification_sample=lambda pst, vs, prefixes: read.append(dict(prefixes)),
            _split_counts=lambda pst, reads: {},
            drift_verdict=lambda pst, counts, level: next(verdicts),
        ):
            gate = cluster.AcceptPreservingGate(
                SimpleNamespace(require_accept_preserving=True), state=None
            )
            gate.verdict(_PST, 0, [1, 2])
            gate.verdict(_PST, 0, [1, 2])

        self.assertEqual({UNIFORM: [0], "state": [2]}, read[1])


class TestAlignedSuffixes(unittest.TestCase):
    def test_a_row_reading_like_the_anchor_is_kept_and_one_that_does_not_is_not(self):
        rng = np.random.default_rng(0)
        n = 20000
        accepting = rng.random(n) < 0.3
        # The second row reads a tenth of the prefixes in the other class.
        moved = accepting ^ (rng.random(n) < 0.1)
        read = lambda classes: (rng.random(n) < np.where(classes, P_1, P_0)) * 1.0
        rows = np.array([read(accepting), read(moved)])

        kept = aligned_suffixes(
            rows,
            read(accepting),
            [np.ones(n, dtype=bool)],
            (P_0, P_1),
            epsilon=0.1,
            alpha=LEVEL,
        )

        self.assertEqual([0], kept.tolist())


if __name__ == "__main__":
    unittest.main()
