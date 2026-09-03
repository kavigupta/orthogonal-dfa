"""The fast lstar tests, run as a single CI shard.

These are dominated by CI job setup rather than test time, so splitting them
one-job-per-test (as tests/test_lstar.py is) wastes a runner slot each. The
slower lstar tests stay in test_lstar.py, where per-test sharding pays off.
"""

import unittest

from automata.fa.dfa import DFA

from orthogonal_dfa.l_star.examples.benchmark_generator import DFAOracle
from orthogonal_dfa.l_star.examples.bernoulli_parity import (
    BernoulliParityOracle,
    BernoulliRegex,
)
from orthogonal_dfa.l_star.structures import AsymmetricBernoulli, NoisyOracle
from tests.lstar_common import assertDFA, assertDoesNotMeetProperty
from tests.lstar_common import learn_dfa_verified as learn_dfa


class TestLStarFast(unittest.TestCase):
    def test_modulo(self):
        oracle_creator = lambda noise_model, seed: NoisyOracle(
            BernoulliParityOracle(modulo=9, allowed_moduluses=(3, 6)), noise_model, seed
        )
        dfa = learn_dfa(oracle_creator, min_signal_strength=0.3, seed=0)
        assertDFA(self, dfa, oracle_creator)

    def test_specific_subsequence(self):
        oracle_creator = lambda noise_model, seed: NoisyOracle(
            BernoulliRegex(regex=r".*1010101.*"), noise_model, seed
        )
        dfa = learn_dfa(oracle_creator, min_signal_strength=0.3, seed=0)
        assertDFA(self, dfa, oracle_creator)

    def test_two_subsequences(self):
        oracle_creator = lambda noise_model, seed: NoisyOracle(
            BernoulliRegex(regex=r".*1111.*1111.*"), noise_model, seed
        )
        dfa = learn_dfa(oracle_creator, min_signal_strength=0.3, seed=0)
        assertDFA(self, dfa, oracle_creator)

    def test_specific_alternation(self):
        oracle_creator = lambda noise_model, seed: NoisyOracle(
            BernoulliRegex(regex=r".*(1111|0000)11.*"), noise_model, seed
        )
        dfa = learn_dfa(oracle_creator, min_signal_strength=0.3, seed=0)
        assertDFA(
            self, dfa, oracle_creator, exclude_pattern=lambda s: s[:5] == bytes([1] * 5)
        )

    def test_specific_alternation_with_nothing_at_end_does_not_meet_property(self):
        oracle_creator = lambda noise_model, seed: NoisyOracle(
            BernoulliRegex(regex=r".*(11111|00000).*"), noise_model, seed
        )

        def counterexample_generator(suffix):
            if suffix[0] == 1:
                return bytes([1]) * 4
            return bytes([0]) * 4

        assertDoesNotMeetProperty(self, oracle_creator, counterexample_generator)

    def test_specific_alternation_with_only_one_at_end_does_not_meet_property(self):
        oracle_creator = lambda noise_model, seed: NoisyOracle(
            BernoulliRegex(regex=r".*(11111|00000)1.*"), noise_model, seed
        )

        def counterexample_generator(suffix):
            if suffix[0] == 1:
                return bytes([1]) * 5
            return bytes([0]) * 4

        assertDoesNotMeetProperty(self, oracle_creator, counterexample_generator)

    def test_transient_states_terminate(self):
        # Regression for issue #128. This target -- {w : |w| >= 3 and
        # w[2] == '0'} -- has transient states (0, 1, 2) that a fixed-length
        # prefix sampler never lands on, so they are unresolvable and the DFA
        # is not learnable with this sampler. Synthesis must still *terminate*
        # rather than grow the prefix set forever. We assert only that it
        # returns within a generous timeout, not what it returns: the learned
        # DFA is expected to be imperfect, so there is no correct output to
        # check.
        dfa = DFA(
            states={0, 1, 2, 3, 4},
            input_symbols={0, 1},
            transitions={
                0: {0: 1, 1: 1},
                1: {0: 2, 1: 2},
                2: {0: 3, 1: 4},
                3: {0: 3, 1: 3},
                4: {0: 4, 1: 4},
            },
            initial_state=0,
            final_states={3},
            allow_partial=False,
        )
        oracle_creator = lambda nm, s, _dfa=dfa: NoisyOracle(DFAOracle(_dfa), nm, s)

        # Imported locally: a module-level `import signal` would shadow the
        # `signal` (signal-strength) parameter other tests in this file use.
        import signal

        def _timeout(signum, frame):
            raise AssertionError(
                "synthesis did not terminate within the timeout (issue #128)"
            )

        previous = signal.signal(signal.SIGALRM, _timeout)
        signal.alarm(300)
        try:
            learn_dfa(oracle_creator, min_signal_strength=0.45, seed=0)
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, previous)


class TestLStarAsymmetricFast(unittest.TestCase):
    def test_modulo_asymmetric(self):
        oracle_creator = lambda noise_model, seed: NoisyOracle(
            BernoulliParityOracle(modulo=9, allowed_moduluses=(3, 6)), noise_model, seed
        )
        noise_model = AsymmetricBernoulli(p_0=0.05, p_1=0.85)
        # signal = (0.85 - 0.05) / 2 = 0.4, but for now we're using 0.35 to be safe.
        dfa = learn_dfa(
            oracle_creator, min_signal_strength=0.35, seed=0, noise_model=noise_model
        )
        assertDFA(self, dfa, oracle_creator)

    def test_modulo_asymmetric_skewed(self):
        oracle_creator = lambda noise_model, seed: NoisyOracle(
            BernoulliParityOracle(modulo=9, allowed_moduluses=(3, 6)), noise_model, seed
        )
        noise_model = AsymmetricBernoulli(p_0=0.25, p_1=0.95)
        # signal = (0.95 - 0.25) / 2 = 0.35, but for now we're using 0.25 to be safe.
        dfa = learn_dfa(
            oracle_creator, min_signal_strength=0.25, seed=0, noise_model=noise_model
        )
        assertDFA(self, dfa, oracle_creator)

    def test_rare_accept_class(self):
        """Only 1 of 7 states is accepting, so boundary estimation sees mostly rejects."""
        oracle_creator = lambda noise_model, seed: NoisyOracle(
            BernoulliParityOracle(modulo=7, allowed_moduluses=(3,)), noise_model, seed
        )
        noise_model = AsymmetricBernoulli(p_0=0.15, p_1=0.75)
        # signal = 0.30, boundary = 0.45
        dfa = learn_dfa(
            oracle_creator, min_signal_strength=0.25, seed=0, noise_model=noise_model
        )
        assertDFA(self, dfa, oracle_creator)

    @unittest.skip(
        "Trimodal over seeds -- 4/8 resolve 9 states, 2/8 collapse to 3, 2/8 raise "
        "from synthesis -- so no single-seed assertion holds either way. See #230."
    )
    def test_boundary_near_zero(self):
        """Both noise rates near 0, boundary far from 0.5.
        The mode this pins: finds only 3 states instead of 9. With the true
        boundary at 0.22, the clustering threshold is so low that true-reject
        prefixes (mean ~0.02) get mixed into the "accept" group on noisy suffix
        samples, contaminating the boundary estimate downward to ~0.11."""
        oracle_creator = lambda noise_model, seed: NoisyOracle(
            BernoulliParityOracle(modulo=9, allowed_moduluses=(3, 6)), noise_model, seed
        )
        noise_model = AsymmetricBernoulli(p_0=0.02, p_1=0.42)
        # signal = 0.20, boundary = 0.22
        dfa = learn_dfa(
            oracle_creator, min_signal_strength=0.15, seed=0, noise_model=noise_model
        )
        assertDFA(self, dfa, oracle_creator)


if __name__ == "__main__":
    unittest.main()
