import unittest

import numpy as np
from automata.fa.dfa import DFA
from parameterized import parameterized

from orthogonal_dfa.l_star.cluster import GaveUpOnSuffixSearch
from orthogonal_dfa.l_star.examples.benchmark_generator import (
    DFAOracle,
    sample_balanced_benchmark,
)
from orthogonal_dfa.l_star.examples.bernoulli_parity import (
    AllFramesClosedOracle,
    BernoulliParityOracle,
    BernoulliRegex,
)
from orthogonal_dfa.l_star.lstar import do_counterexample_driven_synthesis
from orthogonal_dfa.l_star.prefix_suffix_tracker import (
    PrefixSuffixTracker,
    SearchConfig,
)
from orthogonal_dfa.l_star.sampler import UniformSampler
from orthogonal_dfa.l_star.statistics import (
    compute_prefix_set_size,
    compute_suffix_size_counterexample_gen,
    give_up_check,
    population_size_and_evidence_margin,
)
from orthogonal_dfa.l_star.structures import AsymmetricBernoulli, SymmetricBernoulli

us = UniformSampler(40)

allowed_error = 0.02
# assertDFA tolerance — slightly looser than the synthesis target so we don't
# flake when synthesis converges near the threshold.  See GitHub issue on
# tightening synthesis output.
assertion_allowed_error = 0.03


def sample_with_exclusion(exclude_pattern, *, symbols, count):
    rng = np.random.default_rng(0x1234)
    results = []
    while len(results) < count:
        s = us.sample(rng, symbols)
        if exclude_pattern is None or not exclude_pattern(s):
            results.append(s)
    return results


def compute_dfa_accuracy(
    dfa, oracle_creator, exclude_pattern=None, symbols=2, count=10_000
):
    """Evaluate dfa against a noiseless oracle. Returns (accuracy, false_positives, false_negatives)."""
    oracle = oracle_creator(SymmetricBernoulli(p_correct=1.0), 0)
    false_positives, false_negatives = [], []
    for s in sample_with_exclusion(exclude_pattern, symbols=symbols, count=count):
        expected = oracle.membership_query(s)
        actual = dfa.accepts_input(s)
        if expected and not actual:
            false_negatives.append(s)
        elif not expected and actual:
            false_positives.append(s)
    accuracy = 1 - (len(false_positives) + len(false_negatives)) / count
    return accuracy, false_positives, false_negatives


def evaluate_accuracy(
    dfa, oracle_creator, exclude_pattern=None, symbols=2, count=10_000
):
    """Return accuracy of dfa against a noiseless oracle."""
    accuracy, _, _ = compute_dfa_accuracy(
        dfa, oracle_creator, exclude_pattern, symbols, count
    )
    return accuracy


def assertDFA(
    testcase, dfa, oracle_creator, exclude_pattern=None, symbols=2, *, count=10_000
):
    accuracy, false_positives, false_negatives = compute_dfa_accuracy(
        dfa, oracle_creator, exclude_pattern, symbols, count
    )
    if accuracy < 1 - assertion_allowed_error:
        print("DFA is incorrect!")
        print(dfa)
        print(f"False positives: {false_positives}")
        print(f"False negatives: {false_negatives}")
        testcase.fail(
            f"DFA incorrect. False positives: {len(false_positives)}, False negatives: {len(false_negatives)}"
        )


def compute_dfa_for_oracle(
    oracle_creator,
    *,
    min_signal_strength,
    seed,
    noise_model=None,
    min_suffix_frequency=0.05,
):
    pst = compute_pst(
        oracle_creator,
        min_signal_strength,
        seed,
        noise_model=noise_model,
        min_suffix_frequency=min_suffix_frequency,
    )
    dfa, dt = do_counterexample_driven_synthesis(
        pst, additional_counterexamples=200, acc_threshold=1 - allowed_error
    )
    return pst, dfa, dt


def compute_pst(
    oracle_creator,
    min_signal_strength,
    seed,
    *,
    use_dynamic=True,
    noise_model=None,
    min_suffix_frequency=0.05,
):
    effective_p_acc = 0.5 + min_signal_strength
    if noise_model is None:
        noise_model = SymmetricBernoulli(p_correct=effective_p_acc)
    oracle = oracle_creator(noise_model, seed)
    n, eps = population_size_and_evidence_margin(
        signal_strength=min_signal_strength, acceptable_fpr=0.01, acceptable_fnr=0.01
    )
    k = compute_prefix_set_size(0.05, effective_p_acc, 0.05)
    suffix_size = compute_suffix_size_counterexample_gen(0.01, effective_p_acc)
    num_prefixes = 200 if use_dynamic else k
    config = SearchConfig(
        suffix_family_size=n,
        evidence_margin=eps,
        decision_rule_fpr=0.01,
        suffix_size_counterexample_gen=suffix_size,
        min_signal_strength=min_signal_strength,
        num_addtl_prefixes=200 if use_dynamic else None,
        min_suffix_frequency=min_suffix_frequency,
    )
    print(
        f"Using suffix population size {n}, eps {eps}, and {k} prefixes "
        f"(signal strength {min_signal_strength})."
    )
    pst = PrefixSuffixTracker.create(
        us,
        np.random.default_rng(0),
        oracle,
        config,
        num_prefixes=num_prefixes,
    )

    return pst


def assertDoesNotMeetProperty(
    testcase, oracle_creator, counterexample_generator, count=10_000
):
    rng = np.random.default_rng(0)
    oracle = oracle_creator(SymmetricBernoulli(p_correct=1.0), 0)
    valid = []
    for _ in range(count):
        suffix = us.sample(rng, 2)
        prefix = counterexample_generator(suffix)
        s = prefix + suffix
        if oracle.membership_query(s) == oracle.membership_query(prefix):
            valid.append((suffix, prefix))
    if len(valid) / count < 0.001:
        return
    for suffix, prefix in valid[:10]:
        print(f"Counterexample: prefix={prefix}, suffix={suffix}")
    testcase.fail(
        f"Oracle meets property; found {len(valid)} / {count} counterexamples."
    )


class TestLStar(unittest.TestCase):
    def test_modulo(self):
        oracle_creator = lambda noise_model, seed: BernoulliParityOracle(
            noise_model, seed, modulo=9, allowed_moduluses=(3, 6)
        )
        _, dfa, _ = compute_dfa_for_oracle(
            oracle_creator, min_signal_strength=0.3, seed=0
        )
        assertDFA(self, dfa, oracle_creator)

    def test_modulo_harder(self):
        oracle_creator = lambda noise_model, seed: BernoulliParityOracle(
            noise_model, seed, modulo=9, allowed_moduluses=(3, 6)
        )
        _, dfa, _ = compute_dfa_for_oracle(
            oracle_creator, min_signal_strength=0.2, seed=0
        )
        assertDFA(self, dfa, oracle_creator)

    def test_modulo_even_harder(self):
        oracle_creator = lambda noise_model, seed: BernoulliParityOracle(
            noise_model, seed, modulo=9, allowed_moduluses=(3, 6)
        )
        _, dfa, _ = compute_dfa_for_oracle(
            oracle_creator, min_signal_strength=0.1, seed=0
        )
        assertDFA(self, dfa, oracle_creator)

    def test_specific_subsequence(self):
        oracle_creator = lambda noise_model, seed: BernoulliRegex(
            noise_model, seed, regex=r".*1010101.*"
        )
        _, dfa, _ = compute_dfa_for_oracle(
            oracle_creator, min_signal_strength=0.3, seed=0
        )
        assertDFA(self, dfa, oracle_creator)

    def test_two_subsequences(self):
        oracle_creator = lambda noise_model, seed: BernoulliRegex(
            noise_model, seed, regex=r".*1111.*1111.*"
        )
        _, dfa, _ = compute_dfa_for_oracle(
            oracle_creator, min_signal_strength=0.3, seed=0
        )
        assertDFA(self, dfa, oracle_creator)

    def test_two_subsequences_with_alternation(self):
        oracle_creator = lambda noise_model, seed: BernoulliRegex(
            noise_model, seed, regex=r".*1111.*(1111|0000)11.*"
        )
        _, dfa, _ = compute_dfa_for_oracle(
            oracle_creator, min_signal_strength=0.3, seed=0
        )
        assertDFA(self, dfa, oracle_creator)

    def test_specific_alternation(self):
        oracle_creator = lambda noise_model, seed: BernoulliRegex(
            noise_model, seed, regex=r".*(1111|0000)11.*"
        )
        _, dfa, _ = compute_dfa_for_oracle(
            oracle_creator, min_signal_strength=0.3, seed=0
        )
        assertDFA(self, dfa, oracle_creator, exclude_pattern=lambda s: s[:5] == [1] * 5)

    def test_specific_alternation_with_nothing_at_end_3_syms(self):
        oracle_creator = lambda noise_model, seed: BernoulliRegex(
            noise_model, seed, regex=r".*(111|000).*", alphabet_size=3
        )
        _, dfa, _ = compute_dfa_for_oracle(
            oracle_creator, min_signal_strength=0.3, seed=0
        )
        assertDFA(self, dfa, oracle_creator, symbols=3)

    def test_specific_alternation_with_nothing_at_end_does_not_meet_property(self):
        oracle_creator = lambda noise_model, seed: BernoulliRegex(
            noise_model, seed, regex=r".*(11111|00000).*"
        )

        def counterexample_generator(suffix):
            if suffix[0] == 1:
                return [1, 1, 1, 1]
            return [0, 0, 0, 0]

        assertDoesNotMeetProperty(self, oracle_creator, counterexample_generator)

    def test_specific_alternation_with_only_one_at_end_does_not_meet_property(self):
        oracle_creator = lambda noise_model, seed: BernoulliRegex(
            noise_model, seed, regex=r".*(11111|00000)1.*"
        )

        def counterexample_generator(suffix):
            if suffix[0] == 1:
                return [1, 1, 1, 1, 1]
            return [0, 0, 0, 0]

        assertDoesNotMeetProperty(self, oracle_creator, counterexample_generator)

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
        _, dfa, _ = compute_dfa_for_oracle(
            oracle_creator, min_signal_strength=0.3, seed=0
        )
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
        _, dfa, _ = compute_dfa_for_oracle(
            oracle_creator, min_signal_strength=0.3, seed=0
        )
        assertDFA(self, dfa, oracle_creator)


class TestLStarAsymmetric(unittest.TestCase):
    def test_modulo_asymmetric(self):
        oracle_creator = lambda noise_model, seed: BernoulliParityOracle(
            noise_model, seed, modulo=9, allowed_moduluses=(3, 6)
        )
        noise_model = AsymmetricBernoulli(p_0=0.05, p_1=0.85)
        # signal = (0.85 - 0.05) / 2 = 0.4, but for now we're using 0.35 to be safe.
        _, dfa, _ = compute_dfa_for_oracle(
            oracle_creator, min_signal_strength=0.35, seed=0, noise_model=noise_model
        )
        assertDFA(self, dfa, oracle_creator)

    def test_modulo_asymmetric_skewed(self):
        oracle_creator = lambda noise_model, seed: BernoulliParityOracle(
            noise_model, seed, modulo=9, allowed_moduluses=(3, 6)
        )
        noise_model = AsymmetricBernoulli(p_0=0.25, p_1=0.95)
        # signal = (0.95 - 0.25) / 2 = 0.35, but for now we're using 0.25 to be safe.
        _, dfa, _ = compute_dfa_for_oracle(
            oracle_creator, min_signal_strength=0.25, seed=0, noise_model=noise_model
        )
        assertDFA(self, dfa, oracle_creator)

    def test_regex_asymmetric(self):
        oracle_creator = lambda noise_model, seed: BernoulliRegex(
            noise_model, seed, regex=r".*1010101.*"
        )
        noise_model = AsymmetricBernoulli(p_0=0.15, p_1=0.7)
        # signal = (0.7 - 0.15) / 2 = 0.275, but for now we're using 0.2 to be safe.
        _, dfa, _ = compute_dfa_for_oracle(
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
        _, dfa, _ = compute_dfa_for_oracle(
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
        _, dfa, _ = compute_dfa_for_oracle(
            oracle_creator, min_signal_strength=0.15, seed=0, noise_model=noise_model
        )
        assertDFA(self, dfa, oracle_creator)

    def test_rare_accept_class(self):
        """Only 1 of 7 states is accepting, so boundary estimation sees mostly rejects."""
        oracle_creator = lambda noise_model, seed: BernoulliParityOracle(
            noise_model, seed, modulo=7, allowed_moduluses=(3,)
        )
        noise_model = AsymmetricBernoulli(p_0=0.15, p_1=0.75)
        # signal = 0.30, boundary = 0.45
        _, dfa, _ = compute_dfa_for_oracle(
            oracle_creator, min_signal_strength=0.25, seed=0, noise_model=noise_model
        )
        assertDFA(self, dfa, oracle_creator)

    @unittest.expectedFailure
    def test_boundary_near_zero(self):
        """Both noise rates near 0, boundary far from 0.5.
        Fails: finds only 3 states instead of 9. With the true boundary at
        0.22, the clustering threshold is so low that true-reject prefixes
        (mean ~0.02) get mixed into the "accept" group on noisy suffix
        samples, contaminating the boundary estimate downward to ~0.11."""
        oracle_creator = lambda noise_model, seed: BernoulliParityOracle(
            noise_model, seed, modulo=9, allowed_moduluses=(3, 6)
        )
        noise_model = AsymmetricBernoulli(p_0=0.02, p_1=0.42)
        # signal = 0.20, boundary = 0.22
        _, dfa, _ = compute_dfa_for_oracle(
            oracle_creator, min_signal_strength=0.15, seed=0, noise_model=noise_model
        )
        assertDFA(self, dfa, oracle_creator)


class TestGiveUpThreshold(unittest.TestCase):

    @parameterized.expand(
        [
            # (signal, P, r, min_acc_rej, center)
            (0.25, 200, 0.10, 0.5, 0.5),
            (0.30, 200, 0.10, 0.5, 0.5),
            (0.30, 100, 0.10, 0.5, 0.5),
            # Asymmetric: center=0.65, min_acc_rej=0.2
            (0.30, 200, 0.10, 0.2, 0.65),
        ]
    )
    def test_rarely_gives_up_when_evidence_present(  # pylint: disable=too-many-positional-arguments
        self, signal_strength, num_prefixes, r, min_acc_rej, center
    ):
        """Empirically validate that the give-up check matches its claimed
        failure probability. Under signal, the top-k mean agreement should
        almost always exceed the threshold."""
        failure_prob = 0.05
        num_suffixes = 200
        p_accept = center + signal_strength
        p_reject = center - signal_strength
        empirical_pos = min_acc_rej * p_accept + (1 - min_acc_rej) * p_reject

        result = give_up_check(
            signal_strength,
            num_prefixes,
            num_suffixes,
            r,
            min_acc_rej,
            empirical_pos,
            failure_prob=failure_prob,
        )
        self.assertIsNotNone(result, "k too small")
        k, threshold = result

        num_trials = 5_000
        rng = np.random.default_rng(42)
        failures = 0

        for _ in range(num_trials):
            true_labels = rng.random(num_prefixes) < min_acc_rej
            p_per_prefix = np.where(true_labels, p_accept, p_reject)
            seed_obs = rng.random(num_prefixes) < p_per_prefix

            is_idempotent = rng.random(num_suffixes) < r
            all_obs = rng.random((num_suffixes, num_prefixes))
            thresh = np.where(
                is_idempotent[:, None],
                p_per_prefix[None, :],
                empirical_pos,
            )
            suffix_obs = all_obs < thresh
            agreements = (suffix_obs == seed_obs[None, :]).mean(axis=1)
            top_k_mean = np.sort(agreements)[-k:].mean()

            if top_k_mean <= threshold:
                failures += 1

        empirical_failure_rate = failures / num_trials
        print(
            f"s={signal_strength}, P={num_prefixes}, r={r}, "
            f"min_acc_rej={min_acc_rej}, T={num_suffixes}: "
            f"k={k}, threshold={threshold:.4f}, "
            f"empirical_failure={empirical_failure_rate:.4f}, "
            f"target={failure_prob}"
        )

        # The bound is conservative (top-k >= random idempotent
        # suffixes), so we only check the upper bound.
        self.assertLess(empirical_failure_rate, failure_prob + 0.02)

    def test_gives_up_with_no_signal(self):
        """With p_0 = p_1 = 0.5 (pure coin-flip), the oracle has no signal.
        The give-up mechanism should detect this and raise GaveUpOnSuffixSearch."""
        oracle_creator = lambda noise_model, seed: BernoulliParityOracle(
            noise_model, seed, modulo=9, allowed_moduluses=(3, 6)
        )
        noise_model = AsymmetricBernoulli(p_0=0.5, p_1=0.5)
        with self.assertRaises(GaveUpOnSuffixSearch):
            compute_dfa_for_oracle(
                oracle_creator,
                min_signal_strength=0.3,
                seed=0,
                noise_model=noise_model,
            )


class TestLStarORF(unittest.TestCase):
    @parameterized.expand([(signal,) for signal in (0.3, 0.2)])
    def test_no_orf(self, signal):
        oracle_creator = AllFramesClosedOracle
        _, dfa, _ = compute_dfa_for_oracle(
            oracle_creator, min_signal_strength=signal, seed=0
        )
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
        _, dfa, _ = compute_dfa_for_oracle(
            oracle_creator, min_signal_strength=0.3, seed=0
        )
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
        _, dfa, _ = compute_dfa_for_oracle(
            oracle_creator, min_signal_strength=0.3, seed=0
        )
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
        _, learned, _ = compute_dfa_for_oracle(
            oracle_creator, min_signal_strength=0.3, seed=0
        )
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
        _, learned, _ = compute_dfa_for_oracle(
            oracle_creator, min_signal_strength=0.3, seed=0
        )
        assertDFA(self, learned, oracle_creator)
