import numpy as np

from orthogonal_dfa.l_star.learn import learn_dfa
from orthogonal_dfa.l_star.sampler import UniformSampler
from orthogonal_dfa.l_star.statistics import binomial_side_of_boundary
from orthogonal_dfa.l_star.structures import SymmetricBernoulli

us = UniformSampler(40)

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


def _round_accept_preserving_counts(classifier, truth_oracle):
    """``(decisive, misclassified)`` counts over the calibrated-population prefixes
    ``classifier`` decides, or None when it decides none of them. Off-length prefixes
    (boundary strings, per-state samples) are excluded: the family was never
    calibrated on them, so its cut is not expected to hold there."""
    decisive = classifier.decisive & classifier.calibrated
    if not decisive.any():
        return None
    truth = np.array(
        [bool(truth_oracle.membership_query(p)) for p in classifier.prefixes]
    )
    return int(decisive.sum()), int(
        np.sum(classifier.accept[decisive] != truth[decisive])
    )


def learn_dfa_verified(oracle_creator, **kwargs):
    """``learn_dfa``, asserting the per-round accept-preserving invariant."""
    dfa, classifiers = learn_dfa(oracle_creator, **kwargs)
    truth_oracle = oracle_creator(SymmetricBernoulli(p_correct=1.0), 0)
    for classifier in classifiers:
        counts = _round_accept_preserving_counts(classifier, truth_oracle)
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
