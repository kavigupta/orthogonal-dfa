"""The certificate that ends synthesis: its bound on a DFA's error from each
state's rate, and its verdicts."""

import unittest

import numpy as np
from automata.fa.dfa import DFA
from parameterized import parameterized

from orthogonal_dfa.l_star.certificate import certifies, error_bound
from orthogonal_dfa.l_star.examples.benchmark_generator import DFAOracle
from orthogonal_dfa.l_star.learn import build_pst
from orthogonal_dfa.l_star.sampler import UniformSampler
from orthogonal_dfa.l_star.structures import AsymmetricBernoulli, NoisyOracle

LEVEL = 0.05


def _ones_mod_3(accepting):
    return DFA(
        states={0, 1, 2},
        input_symbols={0, 1},
        transitions={q: {0: q, 1: (q + 1) % 3} for q in range(3)},
        initial_state=0,
        final_states=set(accepting),
    )


TARGET = _ones_mod_3({0})
#: Residue 1, a third of the strings, read as accepting.
MERGED = _ones_mod_3({0, 1})


def _rates(low_rate, gap, contaminated):
    """r_S = p_0 + gap c_S for c_S the share of accepting strings in each state."""
    return low_rate + gap * np.asarray(contaminated, dtype=float)


class TestErrorBound(unittest.TestCase):
    def test_never_below_the_error_while_the_intervals_hold(self):
        rng = np.random.default_rng(0)
        for _ in range(500):
            states = int(rng.integers(1, 6))
            masses = rng.dirichlet(np.ones(states))
            accepting = rng.random(states) < 0.5
            gap = float(rng.uniform(0.1, 0.9))
            low_rate = float(rng.uniform(0, 1 - gap))
            shares = rng.random(states) * (rng.random(states) < 0.5)
            accepted_share = np.where(accepting, 1 - shares, shares)
            rates = _rates(low_rate, gap, accepted_share)
            low = np.clip(rates - rng.random(states) * 0.1, 0, 1)
            high = np.clip(rates + rng.random(states) * 0.1, 0, 1)
            spread = rng.random(states) * 0.05
            mass_intervals = (
                np.clip(masses - spread, 0, 1),
                np.clip(masses + spread, 0, 1),
            )
            self.assertGreaterEqual(
                error_bound(mass_intervals, (low, high), accepting, gap),
                float(masses @ shares) - 1e-12,
            )

    def test_reads_apart_while_the_intervals_hold(self):
        rng = np.random.default_rng(0)
        for _ in range(500):
            states = int(rng.integers(1, 6))
            masses = rng.dirichlet(np.ones(states))
            accepting = rng.random(states) < 0.5
            gap = float(rng.uniform(0.02, 0.5))
            rates = rng.random(states)
            low = np.clip(rates - rng.random(states) * 0.1, 0, 1)
            high = np.clip(rates + rng.random(states) * 0.1, 0, 1)
            spread = rng.random(states) * 0.05
            mass_intervals = (
                np.clip(masses - spread, 0, 1),
                np.clip(masses + spread, 0, 1),
            )
            self.assertReadsApart(
                masses,
                rates,
                accepting,
                gap,
                bound=error_bound(mass_intervals, (low, high), accepting, gap),
            )

    def test_a_state_read_past_the_band_counts_past_its_mass(self):
        masses = np.array([0.058, 0.068, 0.122, 0.744, 0.008])
        accepting = np.array([False, False, False, True, True])
        rates = np.array([0.5696, 0.5723, 0.5095, 0.5483, 0.4916])
        low = np.array([0.5411, 0.5257, 0.4804, 0.5469, 0.4406])
        high = np.array([0.5859, 0.5911, 0.5095, 0.5796, 0.5208])
        mass_intervals = (
            np.array([0.0267, 0.0212, 0.065, 0.7072, 0.0]),
            np.array([0.0688, 0.0679, 0.1217, 0.7578, 0.011]),
        )
        gap = 0.0352
        self.assertReadsApart(
            masses,
            rates,
            accepting,
            gap,
            bound=error_bound(mass_intervals, (low, high), accepting, gap),
        )

    def assertReadsApart(self, masses, rates, accepting, gap, *, bound):
        """m_R m_A Delta >= gap (m_A m_R - bound) wherever m_A m_R >= bound."""
        accepted = float(masses[accepting].sum())
        rejected = 1 - accepted
        if accepted * rejected < bound:
            return
        apart = rejected * float(
            masses[accepting] @ rates[accepting]
        ) - accepted * float(masses[~accepting] @ rates[~accepting])
        self.assertGreaterEqual(apart, gap * (accepted * rejected - bound) - 1e-12)

    def test_exact_with_a_clean_state_of_each_label(self):
        masses = [0.88, 0.06, 0.06]
        accepting = [True, False, False]
        rates = _rates(0.3, 0.4, [1.0, 0.0, 0.2])
        self.assertAlmostEqual(
            error_bound((masses, masses), (rates, rates), accepting, 0.4), 0.06 * 0.2
        )


def _pst(signal, length):
    return build_pst(
        lambda nm, s: NoisyOracle(DFAOracle(TARGET), nm, s),
        min_signal_strength=signal,
        seed=0,
        sampler=UniformSampler(length),
        noise_model=AsymmetricBernoulli(p_0=0.2, p_1=0.8),
    )


class TestVerdicts(unittest.TestCase):
    # The band's signal is 0.3: stated exactly, and understated.
    @parameterized.expand([(0.3,), (0.2,)])
    def test_the_target_is_certified(self, signal):
        self.assertTrue(certifies(_pst(signal, 20), [TARGET], alpha=LEVEL).certified)

    @parameterized.expand([(0.3,), (0.2,)])
    def test_a_merge_past_the_slack_is_not(self, signal):
        self.assertFalse(certifies(_pst(signal, 20), [MERGED], alpha=LEVEL).certified)

    def test_a_dfa_with_one_label_is_not(self):
        self.assertFalse(
            certifies(_pst(0.3, 20), [_ones_mod_3(set())], alpha=LEVEL).certified
        )

    def test_among_starts_the_one_that_certifies_is_the_verdict(self):
        verdict = certifies(_pst(0.3, 20), [MERGED, TARGET], alpha=LEVEL)

        self.assertTrue(verdict.certified)
        self.assertIs(TARGET, verdict.dfa)


if __name__ == "__main__":
    unittest.main()
