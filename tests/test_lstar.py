import unittest

import numpy as np
import pytest
from automata.fa.dfa import DFA
from parameterized import parameterized

from orthogonal_dfa.l_star import preconditions as P
from orthogonal_dfa.l_star.examples.benchmark_generator import (
    DFAOracle,
    sample_balanced_benchmark,
)
from orthogonal_dfa.l_star.examples.bernoulli_parity import (
    AllFramesClosedOracle,
    BernoulliParityOracle,
    BernoulliRegex,
)
from orthogonal_dfa.l_star.learn import learn_dfa as learn_dfa_unchecked
from orthogonal_dfa.l_star.sampler import UniformSampler
from orthogonal_dfa.l_star.structures import AsymmetricBernoulli, NoisyOracle
from orthogonal_dfa.superlanguage.sampler import SuperSampler
from orthogonal_dfa.superlanguage.vocabulary import KmerVocabulary
from tests.lstar_common import (
    assert_not_merged,
    DEFAULT_SAMPLER,
    assert_terminates,
    assertDFA,
    assertion_allowed_error,
    compute_dfa_accuracy,
    endpoint_mass,
)
from tests.lstar_common import learn_dfa_verified as learn_dfa


def _confounded_frame_product():
    """A 28-state target: the product of a phase/frame automaton and a parity
    confounder over a 5-symbol alphabet.

    frame:  symbols 3,4 advance a phase (mod 3); symbols 0,1,2 close frame f0 at
            phase 0 and f1 at phase 1.  Reject once both are closed.
    parity: (count of 3,4 is odd) AND (count of 3 is odd).
    accept = (not both frames closed) XOR parity.

    State ``(phase6, f0, f1, ones3)`` -- ``phase6`` is the 3,4 count mod 6 (its
    mod-3 is the frame phase, its parity feeds the confounder); ``ones3`` is the
    count of 3s mod 2.  Minimizes to 28 states.
    """

    def step(state, symbol):
        phase6, f0, f1, ones3 = state
        if symbol >= 3:
            return ((phase6 + 1) % 6, f0, f1, ones3 ^ (symbol == 3))
        phase = phase6 % 3
        if phase == 0:
            return (phase6, True, f1, ones3)
        if phase == 1:
            return (phase6, f0, True, ones3)
        return state

    def accepts(state):
        phase6, f0, f1, ones3 = state
        parity = (phase6 % 2 == 1) and ones3
        return (not (f0 and f1)) ^ parity

    start = (0, False, False, False)
    states, transitions, frontier = {start}, {}, [start]
    while frontier:
        state = frontier.pop()
        transitions[state] = {}
        for symbol in range(5):
            nxt = step(state, symbol)
            transitions[state][symbol] = nxt
            if nxt not in states:
                states.add(nxt)
                frontier.append(nxt)
    return DFA(
        states=set(states),
        input_symbols=set(range(5)),
        transitions=transitions,
        initial_state=start,
        final_states={s for s in states if accepts(s)},
        allow_partial=False,
    ).minify()


#: Phase-structured sampling keeps the frame substructure live (under uniform
#: sampling both frames close at once, only the ~4-state parity survives).
_CONFOUNDED_SAMPLER = SuperSampler(
    KmerVocabulary(kmers=((3, 0, 2), (3, 0, 0), (3, 2, 0)), base_alphabet_size=4), 36
)


def _poor_case_target(transitions):
    return DFA(
        states={0, 1, 2, 3, 4, 5, 6, 7, 8, 9},
        input_symbols={0, 1},
        transitions=transitions,
        initial_state=0,
        final_states={1},
        allow_partial=False,
    )


POOR_CASE_TARGET = _poor_case_target(
    {
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
    }
)

ANOTHER_POOR_CASE_TARGET = _poor_case_target(
    {
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
    }
)

#: Targets E-L* cannot learn to the default threshold, kept because they are
#: the counterexamples that found the behaviour, not because they are typical.
POOR_CASE_TARGETS = [
    ("counterexample", POOR_CASE_TARGET),
    ("another_counterexample", ANOTHER_POOR_CASE_TARGET),
]


def _assert_bar_is_what_the_target_allows(testcase, target):
    """Why these targets get a lowered threshold, and why 0.97 in particular.

    No other state transitions into state 0, so a prefix ends there only by
    never leaving it, and none of length 40 does.  E-L* names a state by the
    prefixes that end in it, so it cannot anchor one here and must re-root
    where it can, misclassifying whatever state 0 decides differently.

    What is left caps short of the 0.99 the preconditions ask for, and close
    enough to the default 0.98 that the rounds spent reaching for it land or
    not by noise -- but comfortably above 0.97, which is where assertDFA's own
    tolerance already sat.
    """
    ceiling = P.covered_accuracy_ceiling(target, length=40)
    testcase.assertLess(ceiling, 0.99, "target would be admissible; no need to lower")
    testcase.assertGreater(ceiling, 0.975, "0.97 is not the bar this target allows")


class TestLStar(unittest.TestCase):
    def test_modulo_harder(self):
        oracle_creator = lambda noise_model, seed: NoisyOracle(
            BernoulliParityOracle(modulo=9, allowed_moduluses=(3, 6)), noise_model, seed
        )
        dfa = learn_dfa(oracle_creator, min_signal_strength=0.2, seed=0)
        assertDFA(self, dfa, oracle_creator, sampler=DEFAULT_SAMPLER)

    @pytest.mark.slow
    def test_modulo_even_harder(self):
        oracle_creator = lambda noise_model, seed: NoisyOracle(
            BernoulliParityOracle(modulo=9, allowed_moduluses=(3, 6)), noise_model, seed
        )
        dfa = learn_dfa(oracle_creator, min_signal_strength=0.1, seed=0)
        assertDFA(self, dfa, oracle_creator, sampler=DEFAULT_SAMPLER)

    def test_confounded_frame_product(self):
        """A confounder turns a phase/frame automaton into a 28-state product
        whose class-preserving suffixes are rare.  The learner cannot resolve it
        in one round: it grows the pool over a couple of rounds until the
        partition stabilizes, then converges on all 28 states."""
        target = _confounded_frame_product()
        self.assertEqual(len(target.states), 28)
        oracle_creator = lambda noise_model, seed: NoisyOracle(
            DFAOracle(target), noise_model, seed
        )
        dfa = learn_dfa_unchecked(
            oracle_creator,
            min_signal_strength=0.45,
            seed=0,
            sampler=_CONFOUNDED_SAMPLER,
        )
        self.assertEqual(len(dfa.states), 28)
        assertDFA(self, dfa, oracle_creator, symbols=5, sampler=_CONFOUNDED_SAMPLER)

    def test_two_subsequences_with_alternation(self):
        oracle_creator = lambda noise_model, seed: NoisyOracle(
            BernoulliRegex(regex=r".*1111.*(1111|0000)11.*"), noise_model, seed
        )
        dfa = learn_dfa(oracle_creator, min_signal_strength=0.3, seed=0)
        assertDFA(self, dfa, oracle_creator, sampler=DEFAULT_SAMPLER)

    def test_specific_alternation_with_nothing_at_end_3_syms(self):
        oracle_creator = lambda noise_model, seed: NoisyOracle(
            BernoulliRegex(regex=r".*(111|000).*", alphabet_size=3), noise_model, seed
        )
        dfa = learn_dfa(oracle_creator, min_signal_strength=0.3, seed=0)
        assertDFA(self, dfa, oracle_creator, symbols=3, sampler=DEFAULT_SAMPLER)

    @parameterized.expand(POOR_CASE_TARGETS)
    def test_poor_case_learned_at_a_reachable_bar(self, _name, target):
        _assert_bar_is_what_the_target_allows(self, target)
        oracle_creator = lambda nm, s, _dfa=target: NoisyOracle(DFAOracle(_dfa), nm, s)
        dfa = learn_dfa(
            oracle_creator, min_signal_strength=0.3, seed=0, acc_threshold=0.97
        )
        assertDFA(self, dfa, oracle_creator, sampler=DEFAULT_SAMPLER)

    @parameterized.expand(POOR_CASE_TARGETS)
    def test_poor_case_terminates_at_the_default_bar(self, _name, target):
        # The lowered threshold is what these targets can reach, not what the
        # learner needs to survive: asked for accuracy the target does not
        # have, synthesis must run out of patience and return rather than keep
        # buying prefixes for a gate no family will satisfy. Only termination
        # is asserted -- the DFA it settles on is expected to be imperfect, so
        # there is nothing correct to check it against.
        oracle_creator = lambda nm, s, _dfa=target: NoisyOracle(DFAOracle(_dfa), nm, s)

        assert_terminates(
            lambda: learn_dfa(oracle_creator, min_signal_strength=0.3, seed=0),
            seconds=300,
            message="synthesis did not terminate within the timeout",
        )


class TestLStarAsymmetric(unittest.TestCase):
    def test_regex_asymmetric(self):
        oracle_creator = lambda noise_model, seed: NoisyOracle(
            BernoulliRegex(regex=r".*1010101.*"), noise_model, seed
        )
        noise_model = AsymmetricBernoulli(p_0=0.15, p_1=0.7)
        # signal = (0.7 - 0.15) / 2 = 0.275, but for now we're using 0.2 to be safe.
        dfa = learn_dfa(
            oracle_creator, min_signal_strength=0.2, seed=0, noise_model=noise_model
        )
        assertDFA(self, dfa, oracle_creator, sampler=DEFAULT_SAMPLER)

    @parameterized.expand([(0.10, 0.40), (0.60, 0.90)])
    def test_modulo_asymmetric_non_straddling(self, p_0, p_1):
        oracle_creator = lambda noise_model, seed: NoisyOracle(
            BernoulliParityOracle(modulo=9, allowed_moduluses=(3, 6)), noise_model, seed
        )
        noise_model = AsymmetricBernoulli(p_0=p_0, p_1=p_1)
        # signal = (p_1 - p_0) / 2, so 0.15 in both cases.
        dfa = learn_dfa(
            oracle_creator, min_signal_strength=0.15, seed=0, noise_model=noise_model
        )
        assertDFA(self, dfa, oracle_creator, sampler=DEFAULT_SAMPLER)

    def test_one_sided_noise(self):
        """One class is pure coin-flip (p_0=0.50), only the other carries signal."""
        oracle_creator = lambda noise_model, seed: NoisyOracle(
            BernoulliParityOracle(modulo=9, allowed_moduluses=(3, 6)), noise_model, seed
        )
        noise_model = AsymmetricBernoulli(p_0=0.50, p_1=0.80)
        # signal = 0.15, boundary = 0.65
        dfa = learn_dfa(
            oracle_creator, min_signal_strength=0.15, seed=0, noise_model=noise_model
        )
        assertDFA(self, dfa, oracle_creator, sampler=DEFAULT_SAMPLER)


@pytest.mark.slow
class TestLStarORF(unittest.TestCase):
    @parameterized.expand([(signal,) for signal in (0.3, 0.2)])
    def test_no_orf(self, signal):
        oracle_creator = lambda nm, s: NoisyOracle(AllFramesClosedOracle(), nm, s)
        dfa = learn_dfa(oracle_creator, min_signal_strength=signal, seed=0)
        assertDFA(self, dfa, oracle_creator, symbols=4, sampler=DEFAULT_SAMPLER)


class TestLStarOnGeneratedBenchmarks(unittest.TestCase):
    @parameterized.expand([(seed,) for seed in range(10)])
    def test_generated_benchmark(self, seed):
        outer, _, _ = sample_balanced_benchmark(
            seed,
            alphabet_size=2,
            num_inner_states=12,
            num_outer_states=10,
            probe_length=40,
        )
        print(outer)
        oracle_creator = lambda nm, s, _dfa=outer: NoisyOracle(DFAOracle(_dfa), nm, s)
        dfa = learn_dfa(oracle_creator, min_signal_strength=0.3, seed=0)
        accuracy, fp, fn = compute_dfa_accuracy(
            dfa, oracle_creator, sampler=DEFAULT_SAMPLER
        )
        if accuracy < 1 - assertion_allowed_error:
            self.fail(
                f"DFA incorrect (accuracy {accuracy:.3f}). "
                f"FP: {len(fp)}, FN: {len(fn)}"
            )


@pytest.mark.slow
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
            max_attempts=200000,
        )
        print(outer)
        oracle_creator = lambda nm, s, _dfa=outer: NoisyOracle(DFAOracle(_dfa), nm, s)
        dfa = learn_dfa(oracle_creator, min_signal_strength=0.3, seed=0)
        accuracy, fp, fn = compute_dfa_accuracy(
            dfa, oracle_creator, sampler=DEFAULT_SAMPLER
        )
        if accuracy < 1 - assertion_allowed_error:
            self.fail(
                f"DFA incorrect (accuracy {accuracy:.3f}). "
                f"FP: {len(fp)}, FN: {len(fn)}"
            )


@unittest.skip(
    "The target fails the learnability preconditions: no suffix preserves the "
    "accept/reject classes (class_preserving_fraction 0.0) and an uncovered state "
    "carries a decision. What the learner does here is not a property to hold it to."
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
      * rare pocket state 6: REJECT, shortest path 6 and landing probability
        ~0.022, i.e.
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
        oracle_creator = lambda nm, s, _dfa=self.DFA: NoisyOracle(
            DFAOracle(_dfa), nm, s
        )
        # Accept state 9 is absorbing and reachable from the reject cluster, so a
        # random suffix almost always drives a reject prefix into it: no suffix is
        # accept-preserving, so no accept-preserving family exists for the round
        # check to recover. Synthesis instead settles on a coherent
        # non-accept-preserving cut and relies on denoise_accept_labels to fix the
        # resulting labels -- hence learn_dfa_unchecked rather than the verified
        # learner the other tests use.
        self.assertLess(P.class_preserving_fraction(self.DFA, length=40), 0.01)
        dfa = learn_dfa_unchecked(oracle_creator, min_signal_strength=0.3, seed=0)
        accuracy, fp, fn = compute_dfa_accuracy(
            dfa, oracle_creator, sampler=DEFAULT_SAMPLER
        )
        if accuracy < 1 - assertion_allowed_error:
            self.fail(
                f"DFA incorrect (accuracy {accuracy:.3f}). "
                f"FP: {len(fp)}, FN: {len(fn)}"
            )


class TestLStarDeepCounter(unittest.TestCase):
    """``Sigma* 0^k Sigma*`` — "contains a run of k zeros".

    State i counts i consecutive zeros, so its only access string is ``0^i``, and
    the deepest counter states are reached only via long, specific paths that
    random length-L probes almost never hit. They are nonetheless recurrent and
    on the critical path to acceptance. This guards that counterexample-driven
    discovery still finds and enriches them — a misclassified ``0^k`` string
    yields a discriminating prefix ending exactly at a deep state, which seeds it.
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
        oracle_creator = lambda nm, s, _d=dfa: NoisyOracle(DFAOracle(_d), nm, s)
        learned = learn_dfa(oracle_creator, min_signal_strength=0.3, seed=0)
        assertDFA(self, learned, oracle_creator, sampler=DEFAULT_SAMPLER)


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
            )
            if _worst_pair_distinguishing_fraction(outer) < self.HARD_PAIR_FRACTION:
                break
        else:
            self.fail(
                "no benchmark with a pair distinguishable by < "
                f"{self.HARD_PAIR_FRACTION} of length-40 suffixes in 10000 seeds"
            )
        oracle_creator = lambda nm, s, _d=outer: NoisyOracle(DFAOracle(_d), nm, s)
        learned = learn_dfa(oracle_creator, min_signal_strength=0.3, seed=0)
        assertDFA(self, learned, oracle_creator, sampler=DEFAULT_SAMPLER)


# -- a rejecting state folded into the accepting one --------------------------
#
# ``Q`` is non-final but reaches the absorbing accept state ``A`` on every symbol it does
# not hold on, so all but a thin slice of suffixes carry it to an accepting endpoint and a
# family built from the rest votes it accept.  The tree inherits that verdict and the
# hypothesis is built from the tree, so the two AGREE on ``Q``: DFA/DT consistency clears
# its target and synthesis stops with ``Q`` inside ``A``.
#
# The merge is expensive here where a merge of two REJECTING classes is not: ``Q`` rejects
# and ``A`` accepts, so every string truly ending in ``Q`` is accepted by the hypothesis.

ARMED_ALPHABET = 12
#: Symbols that arm the trap.  One: two of them let the round separate ``Q`` and the target
#: is learned exactly.
ARMED_ARMS = 1
#: Symbols ``Q`` holds on.  The rest reach ``A``, so this sets how many suffixes preserve
#: ``Q``'s class -- `holds/alphabet` to the sampler's length, which has to clear the 2% the
#: preconditions ask for without clearing it by much.
ARMED_HOLDS = 10
ARMED_LENGTH = 20
ARMED_SIGNAL = 0.3

#: Seeds each merge rate is measured over, one test apiece: which of them merge is a
#: property of the suffixes a round draws, so a bar on one seed says nothing.
MERGE_SEEDS = 10


def build_armed_target() -> DFA:
    arm = set(range(ARMED_ALPHABET - ARMED_ARMS, ARMED_ALPHABET))
    return DFA(
        states={"c0", "Q", "A"},
        input_symbols=set(range(ARMED_ALPHABET)),
        transitions={
            "c0": {c: ("Q" if c in arm else "c0") for c in range(ARMED_ALPHABET)},
            "Q": {c: ("Q" if c < ARMED_HOLDS else "A") for c in range(ARMED_ALPHABET)},
            "A": {c: "A" for c in range(ARMED_ALPHABET)},
        },
        initial_state="c0",
        final_states={"A"},
        allow_partial=False,
    )


class TestArmedMergeTarget(unittest.TestCase):
    """The construction, not the learner: cheap, and guards the stressor."""

    def test_admitted_and_learnable(self):
        target = build_armed_target()
        sampler = UniformSampler(ARMED_LENGTH)
        report = P.satisfies_preconditions(
            target, length=ARMED_LENGTH, short_circuit=False, sampler=sampler
        )
        self.assertTrue(report.satisfied, report.reasons)
        # The merge is only reachable while the suffixes that preserve ``Q`` are scarce,
        # and only costly while ``Q`` carries mass.  Either drifting leaves the test below
        # passing without exercising anything.
        self.assertLess(report.class_preserving_fraction, 0.05)
        # Nothing about the sampler stops this being learned exactly, so the shortfall
        # below is the learner's and not the target's.
        self.assertAlmostEqual(
            P.covered_accuracy_ceiling(target, length=ARMED_LENGTH, sampler=sampler),
            1.0,
        )


class TestArmedMergeLearned(unittest.TestCase):
    @parameterized.expand([(seed,) for seed in range(MERGE_SEEDS)])
    def test_the_armed_state_is_not_merged(self, seed):
        target = build_armed_target()
        oracle_creator = lambda nm, s, _d=target: NoisyOracle(DFAOracle(_d), nm, s)
        sampler = UniformSampler(ARMED_LENGTH)
        dfa = learn_dfa_unchecked(
            oracle_creator,
            min_signal_strength=ARMED_SIGNAL,
            seed=seed,
            sampler=sampler,
        )
        # Graded at the length it was learned at: a hypothesis that merges ``Q`` still
        # reads well on strings long enough that almost all of them reach ``A`` anyway.
        assert_not_merged(
            self,
            dfa,
            target,
            oracle_creator=oracle_creator,
            symbols=ARMED_ALPHABET,
            sampler=sampler,
        )


# -- the same merge with an escape route --------------------------------------

TRAP_LENGTH = 40
TRAP_SIGNAL = 0.2


def build_trap(alphabet: int, arms: int, disarm: int) -> DFA:
    """``arms`` symbols reach ``Q``; from ``Q``, ``disarm`` escapes and the symbols above
    it leak into the absorbing accept state."""
    arm = range(alphabet - arms, alphabet)
    transitions = {
        "c0": {c: ("Q" if c in arm else "c0") for c in range(alphabet)},
        "Q": {
            c: ("Q" if c < disarm else "e1" if c == disarm else "A")
            for c in range(alphabet)
        },
        "e1": {c: "c0" for c in range(alphabet)},
        "A": {c: "A" for c in range(alphabet)},
    }
    return DFA(
        states={"c0", "Q", "e1", "A"},
        input_symbols=set(range(alphabet)),
        transitions=transitions,
        initial_state="c0",
        final_states={"A"},
        allow_partial=False,
    )


#: ``(alphabet, arms, disarm)``.
TRAPS = [(200, 4, 170), (200, 4, 180)]


class TestTrapTargets(unittest.TestCase):
    @parameterized.expand(list(TRAPS))
    def test_admitted_and_still_heavy(self, alphabet, arms, disarm):
        target = build_trap(alphabet, arms, disarm)
        report = P.satisfies_preconditions(
            target, length=TRAP_LENGTH, short_circuit=False
        )
        self.assertTrue(report.satisfied, report.reasons)
        # A merge only costs accuracy while the merged class carries mass, and is only
        # reachable while class-preserving suffixes are scarce.
        self.assertGreater(endpoint_mass(target, TRAP_LENGTH)["Q"], 0.05)
        self.assertLess(report.class_preserving_fraction, 0.10)


class TestTrapLearned(unittest.TestCase):
    """``Q`` is rejecting and carries 7% of the endpoint mass, so a family that votes it
    into accepting ``A`` costs that much accuracy at the distribution the DFA is graded
    on -- unlike ``e1``, which merges into a rejecting state and costs only its own
    routing."""

    @parameterized.expand([(seed,) for seed in range(MERGE_SEEDS)])
    def test_the_armed_state_is_not_merged(self, seed):
        alphabet, arms, disarm = TRAPS[0]
        target = build_trap(alphabet, arms, disarm)
        oracle_creator = lambda nm, s, _d=target: NoisyOracle(DFAOracle(_d), nm, s)
        sampler = UniformSampler(TRAP_LENGTH)
        dfa = learn_dfa_unchecked(
            oracle_creator, min_signal_strength=TRAP_SIGNAL, seed=seed, sampler=sampler
        )
        assert_not_merged(
            self,
            dfa,
            target,
            oracle_creator=oracle_creator,
            symbols=alphabet,
            sampler=sampler,
        )
