import unittest

import numpy as np
from automata.fa.dfa import DFA
from parameterized import parameterized

from orthogonal_dfa.l_star.examples.benchmark_generator import (
    DFAOracle,
    sample_balanced_benchmark,
)
from orthogonal_dfa.l_star.examples.bernoulli_parity import (
    AllFramesClosedOracle,
    BernoulliParityOracle,
    BernoulliRegex,
)
from orthogonal_dfa.l_star.learn import learn_dfa
from orthogonal_dfa.l_star.structures import AsymmetricBernoulli
from tests.lstar_common import assertDFA, assertion_allowed_error, compute_dfa_accuracy


class TestLStar(unittest.TestCase):
    def test_modulo_harder(self):
        oracle_creator = lambda noise_model, seed: BernoulliParityOracle(
            noise_model, seed, modulo=9, allowed_moduluses=(3, 6)
        )
        dfa = learn_dfa(oracle_creator, min_signal_strength=0.2, seed=0)
        assertDFA(self, dfa, oracle_creator)

    def test_modulo_even_harder(self):
        oracle_creator = lambda noise_model, seed: BernoulliParityOracle(
            noise_model, seed, modulo=9, allowed_moduluses=(3, 6)
        )
        dfa = learn_dfa(oracle_creator, min_signal_strength=0.1, seed=0)
        assertDFA(self, dfa, oracle_creator)

    def test_two_subsequences_with_alternation(self):
        oracle_creator = lambda noise_model, seed: BernoulliRegex(
            noise_model, seed, regex=r".*1111.*(1111|0000)11.*"
        )
        dfa = learn_dfa(oracle_creator, min_signal_strength=0.3, seed=0)
        assertDFA(self, dfa, oracle_creator)

    def test_specific_alternation_with_nothing_at_end_3_syms(self):
        oracle_creator = lambda noise_model, seed: BernoulliRegex(
            noise_model, seed, regex=r".*(111|000).*", alphabet_size=3
        )
        dfa = learn_dfa(oracle_creator, min_signal_strength=0.3, seed=0)
        assertDFA(self, dfa, oracle_creator, symbols=3)

    def test_counterexample_poor_case(self):
        dfa = DFA(
            states={0, 1, 2, 3, 4, 5, 6, 7, 8, 9},
            input_symbols={0, 1},
            transitions={
                0: {1: 9, 0: 9},
                1: {1: 1, 0: 1},
                2: {1: 1, 0: 8},
                3: {1: 2, 0: 8},
                4: {1: 5, 0: 3},
                5: {1: 6, 0: 3},
                6: {1: 1, 0: 3},
                7: {1: 4, 0: 8},
                8: {1: 7, 0: 8},
                9: {1: 8, 0: 8},
            },
            initial_state=0,
            final_states={1},
            allow_partial=False,
        )
        oracle_creator = lambda nm, s, _dfa=dfa: DFAOracle(nm, s, _dfa)
        dfa = learn_dfa(oracle_creator, min_signal_strength=0.3, seed=0)
        assertDFA(self, dfa, oracle_creator)

    def test_another_countexample_poor_case(self):
        dfa = DFA(
            states={0, 1, 2, 3, 4, 5, 6, 7, 8, 9},
            input_symbols={0, 1},
            transitions={
                0: {1: 8, 0: 0},
                1: {1: 1, 0: 1},
                2: {1: 1, 0: 6},
                3: {1: 9, 0: 2},
                4: {1: 3, 0: 8},
                5: {1: 8, 0: 4},
                6: {1: 3, 0: 9},
                7: {1: 8, 0: 6},
                8: {1: 8, 0: 5},
                9: {1: 3, 0: 7},
            },
            initial_state=0,
            final_states={1},
            allow_partial=False,
        )
        oracle_creator = lambda nm, s, _dfa=dfa: DFAOracle(nm, s, _dfa)
        dfa = learn_dfa(oracle_creator, min_signal_strength=0.3, seed=0)
        assertDFA(self, dfa, oracle_creator)


class TestLStarAsymmetric(unittest.TestCase):
    def test_regex_asymmetric(self):
        oracle_creator = lambda noise_model, seed: BernoulliRegex(
            noise_model, seed, regex=r".*1010101.*"
        )
        noise_model = AsymmetricBernoulli(p_0=0.15, p_1=0.7)
        # signal = (0.7 - 0.15) / 2 = 0.275, but for now we're using 0.2 to be safe.
        dfa = learn_dfa(
            oracle_creator, min_signal_strength=0.2, seed=0, noise_model=noise_model
        )
        assertDFA(self, dfa, oracle_creator)

    @parameterized.expand([(0.10, 0.40), (0.60, 0.90)])
    def test_modulo_asymmetric_non_straddling(self, p_0, p_1):
        oracle_creator = lambda noise_model, seed: BernoulliParityOracle(
            noise_model, seed, modulo=9, allowed_moduluses=(3, 6)
        )
        noise_model = AsymmetricBernoulli(p_0=p_0, p_1=p_1)
        # signal = (p_1 - p_0) / 2, so 0.15 in both cases.
        dfa = learn_dfa(
            oracle_creator, min_signal_strength=0.15, seed=0, noise_model=noise_model
        )
        assertDFA(self, dfa, oracle_creator)

    def test_one_sided_noise(self):
        """One class is pure coin-flip (p_0=0.50), only the other carries signal."""
        oracle_creator = lambda noise_model, seed: BernoulliParityOracle(
            noise_model, seed, modulo=9, allowed_moduluses=(3, 6)
        )
        noise_model = AsymmetricBernoulli(p_0=0.50, p_1=0.80)
        # signal = 0.15, boundary = 0.65
        dfa = learn_dfa(
            oracle_creator, min_signal_strength=0.15, seed=0, noise_model=noise_model
        )
        assertDFA(self, dfa, oracle_creator)


class TestLStarORF(unittest.TestCase):
    @parameterized.expand([(signal,) for signal in (0.3, 0.2)])
    def test_no_orf(self, signal):
        oracle_creator = AllFramesClosedOracle
        dfa = learn_dfa(oracle_creator, min_signal_strength=signal, seed=0)
        assertDFA(self, dfa, oracle_creator, symbols=4)


class TestLStarOnGeneratedBenchmarks(unittest.TestCase):
    @parameterized.expand([(seed,) for seed in range(10)])
    def test_generated_benchmark(self, seed):
        outer, _, _ = sample_balanced_benchmark(
            seed,
            alphabet_size=2,
            num_inner_states=12,
            num_outer_states=10,
            probe_length=40,
            min_accept_or_reject=0.15,
        )
        print(outer)
        oracle_creator = lambda nm, s, _dfa=outer: DFAOracle(nm, s, _dfa)
        dfa = learn_dfa(oracle_creator, min_signal_strength=0.3, seed=0)
        accuracy, fp, fn = compute_dfa_accuracy(dfa, oracle_creator)
        if accuracy < 1 - assertion_allowed_error:
            self.fail(
                f"DFA incorrect (accuracy {accuracy:.3f}). "
                f"FP: {len(fp)}, FN: {len(fn)}"
            )


class TestLStarOnLargeGeneratedBenchmarks(unittest.TestCase):
    """Like ``TestLStarOnGeneratedBenchmarks`` but with larger 18-state outer
    DFAs (vs 10). Balanced ``Sigma*LSigma*`` benchmarks of this size are rarer to
    sample (hence the raised ``max_attempts``) and slower to synthesise (~2-5 min
    each), yet still learn to >= 1 - assertion_allowed_error accuracy. Guards that
    counterexample-driven synthesis scales past the 10-state benchmarks.
    """

    @parameterized.expand([(seed,) for seed in range(5)])
    def test_large_generated_benchmark(self, seed):
        outer, _, _ = sample_balanced_benchmark(
            seed,
            alphabet_size=2,
            num_inner_states=20,
            num_outer_states=18,
            probe_length=40,
            min_accept_or_reject=0.15,
            max_attempts=50000,
        )
        print(outer)
        oracle_creator = lambda nm, s, _dfa=outer: DFAOracle(nm, s, _dfa)
        dfa = learn_dfa(oracle_creator, min_signal_strength=0.3, seed=0)
        accuracy, fp, fn = compute_dfa_accuracy(dfa, oracle_creator)
        if accuracy < 1 - assertion_allowed_error:
            self.fail(
                f"DFA incorrect (accuracy {accuracy:.3f}). "
                f"FP: {len(fp)}, FN: {len(fn)}"
            )


class TestLStarBimodalReproducer(unittest.TestCase):
    """A hand-constructed (not sampled) explicit DFA that pins the spurious-accept
    failure mode behind the rare flakes in ``TestLStarOnGeneratedBenchmarks``, and
    guards the fix for it (``denoise_accept_labels``).

    Structure (alphabet {0,1}, init 0, single absorbing accept state 9):

      * entry funnel 0 -> 1 -> (cluster): skip to the first 1.
      * recurrent confusable reject cluster {2,3,4,5}: "00" advances toward the
        pocket, "11" launches into it; states share a continuation-accept rate
        ~0.6, so they are hard to tell apart under noise.
      * rare pocket state 6: REJECT, shortest path 6 (so the length<=4
        prefix-closed core never reaches it) and landing probability ~0.022, i.e.
        only ~4 of the ~200 random prefixes land in it -- right at the discovery
        threshold.  It is accept-adjacent: ``6 --0--> 9`` (accept).
      * feedback 6 --1--> 7 --*--> 8 --*--> 5: a clean linear return into the
        cluster.  This recurrence is essential -- without it the pocket is
        resolved reliably (a clean "contains-pattern" chain is learned perfectly);
        it is what keeps the pocket's split marginal.

    On a fraction of synthesis trajectories the pocket's ~4 noisy prefixes fail to
    clear the split threshold and the (pure-reject) pocket state is mislabelled
    accept, silently accepting every string ending there -- a ~2% false-positive
    leak (realized ~0.965 vs ~0.99), bimodal across the synthesis RNG and the
    environment.  ``denoise_accept_labels`` re-derives each state's label by a
    majority vote over many fresh samples and corrects the mislabel, so synthesis
    now clears the 0.97 floor here regardless of branch; if that pass regresses,
    this test fails.
    """

    DFA = DFA(
        states=set(range(10)),
        input_symbols={0, 1},
        transitions={
            0: {0: 1, 1: 1},
            1: {0: 1, 1: 2},
            2: {0: 3, 1: 2},
            3: {0: 4, 1: 2},
            4: {0: 3, 1: 5},
            5: {0: 3, 1: 6},
            6: {0: 9, 1: 7},  # rare accept-adjacent reject pocket
            7: {0: 8, 1: 8},
            8: {0: 5, 1: 5},
            9: {0: 9, 1: 9},  # absorbing accept
        },
        initial_state=0,
        final_states={9},
        allow_partial=False,
    )

    def test_bimodal_reproducer(self):
        oracle_creator = lambda nm, s, _dfa=self.DFA: DFAOracle(nm, s, _dfa)
        dfa = learn_dfa(oracle_creator, min_signal_strength=0.3, seed=0)
        accuracy, fp, fn = compute_dfa_accuracy(dfa, oracle_creator)
        if accuracy < 1 - assertion_allowed_error:
            self.fail(
                f"DFA incorrect (accuracy {accuracy:.3f}). "
                f"FP: {len(fp)}, FN: {len(fn)}"
            )


class TestLStarDeepCounter(unittest.TestCase):
    """``Sigma* 0^k Sigma*`` — "contains a run of k zeros".

    State i counts i consecutive zeros, so its only access string is ``0^i``: for
    k > 4 the deepest counter states sit *beyond* the short prefix-closed core and
    are reached only via long, specific paths that random length-L probes almost
    never hit. They are nonetheless recurrent and on the critical path to
    acceptance. This guards that counterexample-driven discovery still finds and
    enriches them — a misclassified ``0^k`` string yields a discriminating prefix
    ending exactly at a deep state, which seeds it. (Regression guard for deep
    recurrent-state learning; complements the prefix-closed core, which only
    reaches shallow states.)
    """

    @parameterized.expand([(k,) for k in (6, 7)])
    def test_contains_run_of_k_zeros(self, k):
        transitions = {i: {0: i + 1, 1: 0} for i in range(k)}
        transitions[k] = {0: k, 1: k}  # absorbing accept once k zeros are seen
        dfa = DFA(
            states=set(range(k + 1)),
            input_symbols={0, 1},
            transitions=transitions,
            initial_state=0,
            final_states={k},
            allow_partial=False,
        )
        oracle_creator = lambda nm, s, _d=dfa: DFAOracle(nm, s, _d)
        learned = learn_dfa(oracle_creator, min_signal_strength=0.3, seed=0)
        assertDFA(self, learned, oracle_creator)


def _worst_pair_distinguishing_fraction(dfa, length=40):
    """Min over state pairs of P[a uniform length-``length`` suffix distinguishes
    them], computed exactly by propagating the joint distribution on the product
    automaton. A tiny value means some pair is nearly indistinguishable at the
    probe length."""
    states = sorted(dfa.states, key=str)
    idx = {s: i for i, s in enumerate(states)}
    n = len(states)
    syms = sorted(dfa.input_symbols)
    finals = np.array([s in dfa.final_states for s in states])
    delta = np.array([[idx[dfa.transitions[s][c]] for c in syms] for s in states])
    diff = finals[:, None] != finals[None, :]
    worst = 1.0
    for i in range(n):
        for j in range(i + 1, n):
            m = np.zeros((n, n))
            m[i, j] = 1.0
            for _ in range(length):
                nxt = np.zeros((n, n))
                for c in range(len(syms)):
                    np.add.at(
                        nxt,
                        (delta[:, c][:, None], delta[:, c][None, :]),
                        m / len(syms),
                    )
                m = nxt
            worst = min(worst, float(m[diff].sum()))
    return worst


class TestLStarIndistinguishablePair(unittest.TestCase):
    """A balanced, class-preserving benchmark can still contain a state pair far
    harder to separate at length 40 than any in the seed 0..9 suite, yet main
    learns it.

    The accuracy cost of merging a pair is bounded by its distinguishing
    fraction, so a hard-to-separate pair is also a cheap-to-miss one —
    ``min_class_preserving_frac`` (a global check) need not bound worst-case
    *pairwise* distinguishability. A benchmark whose hardest pair is distinguished
    by < ~0.3% of length-40 suffixes (vs ~0.96% for seed 5) is found by scanning,
    and synthesis still clears the 0.97 bar on it.
    """

    # A pair distinguishable by less than this fraction of length-40 suffixes is
    # far harder than anything in the seed 0..9 suite.
    HARD_PAIR_FRACTION = 0.003

    def test_extreme_worst_pair_still_learned(self):
        # Don't hard-code a seed (fragile to generator changes): scan until a
        # benchmark has a near-indistinguishable pair, and fail loudly if the
        # generator no longer produces one.
        for seed in range(10000):
            outer, _, _ = sample_balanced_benchmark(
                seed,
                alphabet_size=2,
                num_inner_states=12,
                num_outer_states=10,
                probe_length=40,
                min_accept_or_reject=0.15,
            )
            if _worst_pair_distinguishing_fraction(outer) < self.HARD_PAIR_FRACTION:
                break
        else:
            self.fail(
                "no benchmark with a pair distinguishable by < "
                f"{self.HARD_PAIR_FRACTION} of length-40 suffixes in 10000 seeds"
            )
        oracle_creator = lambda nm, s, _d=outer: DFAOracle(nm, s, _d)
        learned = learn_dfa(oracle_creator, min_signal_strength=0.3, seed=0)
        assertDFA(self, learned, oracle_creator)
