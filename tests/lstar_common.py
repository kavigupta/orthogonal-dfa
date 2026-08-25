import numpy as np
import scipy.stats
from automata.fa.dfa import DFA
from automata.fa.nfa import NFA

from orthogonal_dfa.l_star.learn import learn_dfa
from orthogonal_dfa.l_star.sampler import UniformSampler
from orthogonal_dfa.l_star.statistics import binomial_side_of_boundary
from orthogonal_dfa.l_star.structures import SymmetricBernoulli
from orthogonal_dfa.utils.dfa import al_dfa_symbols_to_int

us = UniformSampler(40)

#: Stands in for "ran off the end of a partial DFA".
_DEAD = object()

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


# Every synthesis round's family is seeded at the empty suffix, so its decisive
# classifications should realise the accept-preserving split: the noiseless
# membership 1[x in L]. learn_dfa returns each round's RoundClassifier, and
# learn_dfa_verified checks it over the prefixes the round decides (indecisive ones
# are boundary strings, excluded).
#
# A round fails only when its misclassification count is, by a binomial test,
# significantly above the calibration's own per-decision error budget -- not against
# a hand-picked fraction. This is sample-size-aware: a handful of near-boundary
# misses on a few-hundred-prefix pool is within the fpr floor, while a family that
# computes the wrong cut misclassifies far more than the floor predicts.
round_verify_fpr = 0.01  # matches acceptable_fpr in learn.build_pst
round_verify_alpha = 1e-4  # binomial significance for flagging a round


#: A state is *common in prefixes* when enough of the round's prefixes reach it
#: for its accept/reject label to be more than a coin flip.  We decide that from
#: the count alone, so this is the rate at which we call a state common in
#: prefixes while its label is in fact undetermined.
common_in_prefixes_fpr = 0.01


def _common_in_prefixes_threshold(signal: float, fpr: float) -> float:
    """Prefixes a state needs before it counts as common in prefixes.

    A family that labels state q correctly and one that does not differ only on
    the prefixes that *reach* q -- everywhere else both predict the same thing.
    Each such prefix votes correctly with probability 1/2 + signal, so q's label
    is a binomial vote over m_q of them, and lands the wrong way more often than
    ``fpr`` unless

        m_q >= z^2 (1/4 - signal^2) / signal^2,   z = Phi^-1(1 - fpr)
    """
    z = scipy.stats.norm.ppf(1 - fpr)
    return z**2 * (0.25 - signal**2) / signal**2


def _target_dfa(oracle):
    """The target's DFA, where the oracle can hand one over.

    A DFA-backed oracle already holds it.  A regex one is compiled the same way
    ``capal_official.build_regex_dfa`` does, over the same int-as-str symbols
    ``BernoulliRegex`` matches on.  Anything else has no states to offer.
    """
    held = getattr(oracle, "_dfa", None)
    if held is not None:
        return held
    regex = getattr(oracle, "regex", None)
    if regex is None:
        return None
    symbols = {str(i) for i in range(oracle.alphabet_size)}
    nfa = NFA.from_regex(regex, input_symbols=symbols)
    return al_dfa_symbols_to_int(DFA.from_nfa(nfa, minify=True))


def _reached_states(prefixes, true_dfa):
    """The state each prefix reaches in ``true_dfa``."""

    def end(prefix):
        state = true_dfa.initial_state
        for symbol in prefix:
            # A compiled regex DFA can be partial; a missing edge is a dead end,
            # which is a state like any other for counting purposes.
            state = true_dfa.transitions[state].get(symbol, _DEAD)
            if state is _DEAD:
                break
        return state

    return [end(p) for p in prefixes]


def _round_accept_preserving_counts(classifier, truth_oracle, signal, true_dfa):
    """``(decisive, misclassified)`` counts over the calibrated-population prefixes
    ``classifier`` decides, or None when it decides none of them. Off-length prefixes
    (boundary strings, per-state samples) are excluded: the family was never
    calibrated on them, so its cut is not expected to hold there.

    Only states that are *common in prefixes* are held to the cut, and states
    that are not are excused only where the round is consistent on them.  Too
    few prefixes reach a rare state for the evidence separating a correct cut
    from a wrong one to be anything but a coin flip, so holding the round to it
    tests the sampler rather than the learner.  A round that cuts one such state
    both ways has no such excuse and still counts.  This needs the target's
    states, so without ``true_dfa`` every decisive prefix is held, as before.
    """
    decisive = classifier.decisive & classifier.calibrated
    if not decisive.any():
        return None
    truth = np.array(
        [bool(truth_oracle.membership_query(p)) for p in classifier.prefixes]
    )
    if true_dfa is None:
        return int(decisive.sum()), int(
            np.sum(classifier.accept[decisive] != truth[decisive])
        )
    states = _reached_states(classifier.prefixes, true_dfa)
    threshold = _common_in_prefixes_threshold(signal, common_in_prefixes_fpr)
    held = np.zeros(len(states), dtype=bool)
    for state in set(states):
        reaching = decisive & np.array([s == state for s in states])
        common_in_prefixes = reaching.sum() >= threshold
        consistent = len(set(classifier.accept[reaching])) <= 1
        if common_in_prefixes or not consistent:
            held |= reaching
    if not held.any():
        return None
    return int(held.sum()), int(np.sum(classifier.accept[held] != truth[held]))


def learn_dfa_verified(oracle_creator, *, true_dfa=None, **kwargs):
    """``learn_dfa``, asserting the per-round accept-preserving invariant.

    ``true_dfa`` lets the check see which state each prefix reaches, so it can
    skip the states too rare in the pool to label; an oracle carrying its own DFA
    supplies it without the caller saying anything."""
    dfa, classifiers = learn_dfa(oracle_creator, **kwargs)
    truth_oracle = oracle_creator(SymmetricBernoulli(p_correct=1.0), 0)
    if true_dfa is None:
        true_dfa = _target_dfa(truth_oracle)
    signal = kwargs["min_signal_strength"]
    for classifier in classifiers:
        counts = _round_accept_preserving_counts(
            classifier, truth_oracle, signal, true_dfa
        )
        if counts is None:
            continue
        decisive, wrong = counts
        if binomial_side_of_boundary(
            wrong, decisive, round_verify_fpr, failure_prob=round_verify_alpha
        ):
            raise AssertionError(
                f"a synthesis round misclassified {wrong}/{decisive} of its decisive "
                f"prefixes against the noiseless accept-preserving split -- "
                f"significantly above the fpr budget {round_verify_fpr} "
                f"(binomial test, alpha={round_verify_alpha})"
            )
    return dfa
