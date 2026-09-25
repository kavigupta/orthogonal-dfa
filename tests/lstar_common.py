import signal
from types import SimpleNamespace

import numpy as np

from orthogonal_dfa.l_star.learn import (
    DEFAULT_MAX_COVERAGE_ERROR,
    DEFAULT_SAMPLE_LENGTH,
    learn_dfa,
)
from orthogonal_dfa.l_star.mask_table import UNIFORM
from orthogonal_dfa.l_star.sampler import UniformSampler
from orthogonal_dfa.l_star.statistics import binomial_side_of_boundary
from orthogonal_dfa.l_star.structures import SymmetricBernoulli
from orthogonal_dfa.l_star.tracker import RecordingTracker

DEFAULT_SAMPLER = UniformSampler(DEFAULT_SAMPLE_LENGTH)

# How far a learned DFA may sit from the target before a test calls it wrong.
assertion_allowed_error = 0.05


def sample_with_exclusion(exclude_pattern, *, symbols, count, sampler):
    """Draws from ``sampler``.  A hypothesis read on longer strings than it was learned
    on can score well while being wrong everywhere the learner looked."""
    rng = np.random.default_rng(0x1234)
    results = []
    while len(results) < count:
        s = sampler.sample(rng, symbols)
        if exclude_pattern is None or not exclude_pattern(s):
            results.append(s)
    return results


def compute_dfa_accuracy(
    dfa, oracle_creator, *, sampler, exclude_pattern=None, symbols=2, count=10_000
):
    """Evaluate dfa against a noiseless oracle. Returns (accuracy, false_positives, false_negatives)."""
    oracle = oracle_creator(SymmetricBernoulli(p_correct=1.0), 0)
    false_positives, false_negatives = [], []
    for s in sample_with_exclusion(
        exclude_pattern, symbols=symbols, count=count, sampler=sampler
    ):
        expected = oracle.membership_query(s)
        actual = dfa.accepts_input(s)
        if expected and not actual:
            false_negatives.append(s)
        elif not expected and actual:
            false_positives.append(s)
    accuracy = 1 - (len(false_positives) + len(false_negatives)) / count
    return accuracy, false_positives, false_negatives


def evaluate_accuracy(
    dfa, oracle_creator, exclude_pattern=None, symbols=2, count=10_000, *, sampler
):
    """Return accuracy of dfa against a noiseless oracle."""
    accuracy, _, _ = compute_dfa_accuracy(
        dfa,
        oracle_creator,
        exclude_pattern=exclude_pattern,
        symbols=symbols,
        count=count,
        sampler=sampler,
    )
    return accuracy


def assertDFA(
    testcase,
    dfa,
    oracle_creator,
    exclude_pattern=None,
    symbols=2,
    *,
    sampler,
    count=10_000,
):
    accuracy, false_positives, false_negatives = compute_dfa_accuracy(
        dfa,
        oracle_creator,
        exclude_pattern=exclude_pattern,
        symbols=symbols,
        count=count,
        sampler=sampler,
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
    testcase, oracle_creator, counterexample_generator, *, sampler, count=10_000
):
    rng = np.random.default_rng(0)
    oracle = oracle_creator(SymmetricBernoulli(p_correct=1.0), 0)
    valid = []
    for _ in range(count):
        suffix = sampler.sample(rng, 2)
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
# membership 1[x in L]. learn_dfa_verified reads each round's RoundClassifier
# off a tracker and checks it over the prefixes the round decides (indecisive ones
# are boundary strings, excluded).
#
# These two bound the *within-state* disagreement: a round is entitled to
# `round_verify_fpr` wrong decisions per prefix, so a state whose minority side is
# larger than that explains -- by a binomial test at `round_verify_alpha` -- was
# not cut by a family holding one opinion about it.
round_verify_fpr = 0.01  # matches acceptable_fpr in learn.build_pst
round_verify_alpha = 1e-4  # binomial significance for flagging a state


#: Rate at which one verified run is expected to fail spuriously, divided by the
#: rounds it checks to get the rate any single round is held to.
round_check_run_fpr = 0.01


def _reached_states(prefixes, true_dfa):
    """The state each prefix reaches in ``true_dfa``."""

    def end(prefix):
        state = true_dfa.initial_state
        for symbol in prefix:
            state = true_dfa.transitions[state][symbol]
        return state

    return [end(p) for p in prefixes]


def _state_cuts(classifier, true_dfa):
    """How the round cut each state: ``state -> (accepted, rejected)`` counts.

    Every prefix reaching a state is the same string as far as the language is
    concerned, so the round should cut them all the same way, and the way it
    should cut them is whether the state accepts.  Prefixes the round left
    undecided are boundary strings; off-length ones reach the family outside its
    calibration.  Neither says anything about the cut, so neither is counted.
    """
    counted = classifier.decisive & classifier.calibrated
    cuts = {}
    for state, keep, accept in zip(
        _reached_states(classifier.prefixes, true_dfa), counted, classifier.accept
    ):
        if keep:
            tally = cuts.setdefault(state, [0, 0])
            tally[0 if accept else 1] += 1
    return cuts


def _split_states(cuts):
    """States the round cut both ways by more than its own error budget allows.

    A round is entitled to ``round_verify_fpr`` wrong decisions per prefix, so a
    state whose minority side is bigger than that explains was not cut by a
    family with one opinion about it.
    """
    return [
        (state, accepted, rejected)
        for state, (accepted, rejected) in cuts.items()
        if binomial_side_of_boundary(
            min(accepted, rejected),
            accepted + rejected,
            round_verify_fpr,
            failure_prob=round_verify_alpha,
        )
    ]


def _miscut(cuts, true_dfa):
    """Prefixes the round decided against the language, and prefixes it decided."""
    against = sum(
        rejected if state in true_dfa.final_states else accepted
        for state, (accepted, rejected) in cuts.items()
    )
    return against, sum(accepted + rejected for accepted, rejected in cuts.values())


def assert_rounds_accept_preserving(classifiers, true_dfa, max_coverage_error):
    """The per-round accept-preserving invariant.

    Each round's family is seeded at the empty suffix, so its decisive
    classifications should realise the accept-preserving split -- to within the
    ``max_coverage_error`` the gate certifies it at, which is a share of the prefixes
    it decides and not a promise about any one state: a state light enough can be
    cut against the language whole.  Checked a round at a time: require it to have
    had a single opinion about each state, and the share it decided against the
    language not to be significantly above what the gate allows.
    """
    assert classifiers, "no rounds recorded -- did the tracker reach synthesis?"
    per_round = [_state_cuts(c, true_dfa) for c in classifiers]
    for cuts in per_round:

        split = _split_states(cuts)
        if split:
            state, accepted, rejected = split[0]
            raise AssertionError(
                f"a synthesis round cut state {state} both ways "
                f"({accepted} accept / {rejected} reject) -- more disagreement than "
                f"its {round_verify_fpr} per-prefix budget explains, so the family "
                f"had no single opinion about the state"
            )

        against, decided = _miscut(cuts, true_dfa)
        if binomial_side_of_boundary(
            against,
            decided,
            max_coverage_error,
            failure_prob=round_check_run_fpr / len(per_round),
        ):
            raise AssertionError(
                f"a synthesis round decided {against} of {decided} prefixes against "
                f"the language -- significantly more than the {max_coverage_error:.3f} "
                f"share its gate certifies"
            )


def learn_dfa_verified(oracle_creator, **kwargs):
    """``learn_dfa``, asserting the per-round accept-preserving invariant."""
    tracker = RecordingTracker()
    dfa = learn_dfa(oracle_creator, tracker=tracker, **kwargs)
    truth_oracle = oracle_creator(SymmetricBernoulli(p_correct=1.0), 0)
    assert_rounds_accept_preserving(
        tracker.classifiers,
        truth_oracle.target_dfa(),
        kwargs.get("max_coverage_error", DEFAULT_MAX_COVERAGE_ERROR),
    )
    return dfa


def assert_terminates(call, *, seconds: int, message: str):
    """Run ``call``, failing if it has not returned within ``seconds``."""

    def expired(signum, frame):
        raise AssertionError(message)

    previous = signal.signal(signal.SIGALRM, expired)
    signal.alarm(seconds)
    try:
        return call()
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)


class _ClusterTable:
    """The reads ``identify_cluster_around`` makes, over a mask block whose rows
    are suffixes and columns prefixes."""

    def __init__(self, masks):
        self._masks = masks
        self.representative = np.ones(masks.shape[1], dtype=bool)

    def fully_observed(self):
        return np.arange(self._masks.shape[0])

    def observed_masks(self, rows, prefixes):
        return self._masks[np.asarray(rows)][:, prefixes]

    def population_masks(self):
        return {UNIFORM: self.representative}


def cluster_pst(masks, min_signal_strength):
    return SimpleNamespace(
        table=_ClusterTable(masks),
        config=SimpleNamespace(min_signal_strength=min_signal_strength),
    )
