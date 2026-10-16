"""The gate holds a family to every population: it admits only when a bound on
the share of each population's distribution the family's cut misclassifies is
within max_coverage_error on the uniform pool and a half elsewhere, the noise
rates read off the uniform pool's own sample, and it names the population that
stops it."""

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
    Sample,
    Side,
    alignment_size,
    calibrate,
    drift_verdict,
    family_level,
    misclassified_bound,
    prefixes_to_certify,
    read_right,
    size_level,
    uniform_rates,
)
from orthogonal_dfa.l_star.mask_table import UNIFORM

#: The rates the round reads at, boundary -/+ the signal.
P_0, P_1 = 0.2, 0.8
#: The noise rates in and out of the language.
ETA_IN, ETA_OUT = 1 - P_1, P_0
LEVEL = family_level(3, 0)

_PST = SimpleNamespace(
    decision_boundary=(P_0 + P_1) / 2,
    config=SimpleNamespace(
        min_signal_strength=(P_1 - P_0) / 2, max_coverage_error=1 / 3
    ),
)


def _side(n, wrong, right_rate, wrong_rate):
    """n prefixes, a share wrong of them misplaced, every count at its mean."""
    k = round(wrong * n)
    against = (n - k) * right_rate + k * (1 - wrong_rate)
    apart = 2 * (
        (n - k) * right_rate * (1 - right_rate) + k * wrong_rate * (1 - wrong_rate)
    )
    return Side(n, round(against), round(apart))


def _sample(n_accept, n_reject, wrong_accept=0.0, wrong_reject=0.0):
    return Sample(
        _side(n_accept, wrong_accept, ETA_IN, ETA_OUT),
        _side(n_reject, wrong_reject, ETA_OUT, ETA_IN),
        n_accept + n_reject,
    )


RIGHT = _sample(4096, 4096)
RATES = uniform_rates(_PST, RIGHT, LEVEL)


class TestEveryPopulationIsHeld(unittest.TestCase):
    def test_a_family_reading_every_population_right_is_admitted(self):
        samples = {UNIFORM: RIGHT, ("state", 0): RIGHT}

        self.assertEqual((ADMITTED, None), drift_verdict(_PST, samples, RATES, LEVEL))

    def test_one_class_populations_read_as_their_class_are_admitted(self):
        samples = {
            UNIFORM: RIGHT,
            ("state", 0): _sample(8192, 0),
            ("state", 1): _sample(0, 8192),
        }

        self.assertEqual(ADMITTED, drift_verdict(_PST, samples, RATES, LEVEL)[0])

    def test_a_population_read_backwards_is_refused_by_name(self):
        # Every prefix the family rejects is accepting.
        samples = {UNIFORM: RIGHT, ("state", 3): _sample(0, 8192, wrong_reject=1)}

        self.assertEqual(
            (DRIFTED, ("state", 3)), drift_verdict(_PST, samples, RATES, LEVEL)
        )

    def test_a_population_misread_past_a_half_is_refused(self):
        # Seven in ten of what the family rejects is accepting.
        samples = {UNIFORM: RIGHT, "mixed": _sample(0, 8192, wrong_reject=0.7)}

        self.assertEqual((DRIFTED, "mixed"), drift_verdict(_PST, samples, RATES, LEVEL))

    def test_the_pool_is_held_to_max_coverage_error(self):
        # Two in five misread: past max_coverage_error, short of a half.
        misread = _sample(4096, 4096, wrong_reject=0.8)
        as_pool = drift_verdict(_PST, {UNIFORM: misread}, RATES, LEVEL)
        as_other = drift_verdict(_PST, {UNIFORM: RIGHT, "mixed": misread}, RATES, LEVEL)

        self.assertEqual((DRIFTED, UNIFORM), as_pool)
        self.assertNotEqual(DRIFTED, as_other[0])

    def test_too_few_prefixes_leaves_the_family_uncertified(self):
        samples = {UNIFORM: RIGHT, "small": _sample(3, 3)}

        self.assertEqual(
            (UNCERTIFIED, "small"), drift_verdict(_PST, samples, RATES, LEVEL)
        )

    def test_nothing_drawn_is_uncertified(self):
        self.assertEqual((UNCERTIFIED, UNIFORM), drift_verdict(_PST, {}, None, LEVEL))

    def test_redrawn_prefixes_count_against_the_family(self):
        # The same sample, found in twice as many draws: half of them count as
        # misclassified, though none of the prefixes is.
        redrawn = Sample(RIGHT.accept, RIGHT.reject, 2 * RIGHT.drawn)
        samples = {UNIFORM: RIGHT, "repeats": redrawn}

        self.assertEqual(
            (UNCERTIFIED, "repeats"), drift_verdict(_PST, samples, RATES, LEVEL)
        )


class TestTheBoundHolds(unittest.TestCase):
    def test_the_bound_covers_the_misclassified_share(self):
        # The family accepts 60% of the population; a fifth of that is in fact
        # rejecting, and a tenth of what it rejects is accepting.
        rng = np.random.default_rng(0)
        accepted, wrong_in, wrong_out, n = 0.6, 0.2, 0.1, 2000
        truth = accepted * wrong_in + (1 - accepted) * wrong_out

        def draw(accepted, wrong_in, wrong_out):
            on_accept = rng.random(n) < accepted
            wrong = rng.random(n) < np.where(on_accept, wrong_in, wrong_out)
            label = on_accept != wrong
            rate = np.where(label, ETA_IN, ETA_OUT)
            seed = label != (rng.random(n) < rate)
            reference = label != (rng.random(n) < rate)

            def side(on, against):
                return Side(
                    int(on.sum()),
                    int((seed[on] == against).sum()),
                    int((reference[on] != seed[on]).sum()),
                )

            return Sample(side(on_accept, 0), side(~on_accept, 1), n)

        level = 0.05 / cluster.GATE_TAILS
        missed = 0
        for _ in range(500):
            uniform = draw(0.5, 0.0, 0.0)
            rates = calibrate(_PST, uniform, size_level(level, uniform.size))
            sample = draw(accepted, wrong_in, wrong_out)
            missed += misclassified_bound(sample, level, rates) < truth

        self.assertLessEqual(missed, 500 * 0.05)


class TestHowMuchIsDrawn(unittest.TestCase):
    def test_a_cut_reading_right_certifies_at_the_size_and_not_a_quarter_of_it(self):
        size = alignment_size(_PST, 1, 1 / 3)
        level = family_level(1, 0)

        def verdict(n):
            sample = read_right(_PST, n)
            rates = uniform_rates(_PST, sample, level)
            return drift_verdict(_PST, {UNIFORM: sample}, rates, level)[0]

        self.assertEqual(ADMITTED, verdict(size))
        self.assertNotEqual(ADMITTED, verdict(size // 4))

    def test_the_top_up_is_the_first_that_settles_the_named_population(self):
        drawn = 512

        def settled(n):
            samples = {"small": _sample(n // 2, n // 2)}
            return drift_verdict(_PST, samples, RATES, LEVEL)[0] is not UNCERTIFIED

        wanted = prefixes_to_certify(
            _PST,
            _sample(drawn // 2, drawn // 2),
            RATES,
            label="small",
            level=LEVEL,
            budget=10**6,
        )

        self.assertTrue(settled(drawn + wanted))
        self.assertFalse(settled(wanted))


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
            prefixes_for_split=lambda pst, state, label, n, held: ([next(draws)], 1),
            alignment_size=lambda pst, populations, limit: 1,
            certification_sample=lambda pst, vs, refs, prefixes: read.append(
                dict(prefixes)
            )
            or {},
            drift_verdict=lambda pst, samples, calibration, level: next(verdicts),
        ):
            gate = cluster.AcceptPreservingGate(
                SimpleNamespace(require_accept_preserving=True), state=None
            )
            pst = SimpleNamespace(**vars(_PST), suffix_pool=[0, 1, 2, 3])
            gate.verdict(pst, 0, [1, 2])
            gate.verdict(pst, 0, [1, 2])

        self.assertEqual({UNIFORM: [0], "state": [2]}, read[1])


if __name__ == "__main__":
    unittest.main()
