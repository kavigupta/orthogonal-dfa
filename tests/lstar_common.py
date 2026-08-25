import numpy as np
import scipy.stats

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
# learn_dfa_verified checks it a state at a time over the prefixes the round
# decides (indecisive ones are boundary strings, excluded).
#
# These two bound the *within-state* disagreement: a round is entitled to
# `round_verify_fpr` wrong decisions per prefix, so a state whose minority side is
# larger than that explains -- by a binomial test at `round_verify_alpha` -- was
# not cut by a family holding one opinion about it.  Neither is a budget over the
# pool as a whole; nothing sums misclassifications across states any more.
round_verify_fpr = 0.01  # matches acceptable_fpr in learn.build_pst
round_verify_alpha = 1e-4  # binomial significance for flagging a state


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


def _wrongly_cut(cuts, true_dfa):
    """``state -> prefixes reaching it`` for the states cut against the language."""
    return {
        state: accepted + rejected
        for state, (accepted, rejected) in cuts.items()
        if (accepted >= rejected) != (state in true_dfa.final_states)
    }


def _wrongly_cut_states(cuts, true_dfa, threshold):
    """States the round cut against the language, that it had the prefixes to know.

    Below ``threshold`` prefixes the state's label is a coin flip whichever way
    the round called it, so being wrong there is the sampler's doing.  Above it,
    the round had the evidence and still cut the other way.
    """
    return [
        (state, reached)
        for state, reached in _wrongly_cut(cuts, true_dfa).items()
        if reached >= threshold
    ]


def _excused_wrong_prefixes(cuts, true_dfa, threshold):
    """``(excused, total)`` prefixes: those in states cut wrong but too rare to hold.

    Excusing a rare state one at a time is right -- its label was a coin flip.
    Excusing every one of them is not: a family that inverts several states at
    once is systematically wrong, not unlucky nine times over, and the prefixes
    it cost add up even though no single state carries enough to convict.  So
    they stay excused individually and budgeted together.
    """
    excused = sum(
        reached
        for reached in _wrongly_cut(cuts, true_dfa).values()
        if reached < threshold
    )
    return excused, sum(accepted + rejected for accepted, rejected in cuts.values())


def learn_dfa_verified(oracle_creator, **kwargs):
    """``learn_dfa``, asserting the per-round accept-preserving invariant.

    Each round's family is seeded at the empty suffix, so its decisive
    classifications should realise the accept-preserving split.  Checked a state
    at a time: tally how the round cut each one, require it to have had a single
    opinion about each, and require the ones it got backwards to be states its
    prefixes barely reached.
    """
    dfa, classifiers = learn_dfa(oracle_creator, **kwargs)
    truth_oracle = oracle_creator(SymmetricBernoulli(p_correct=1.0), 0)
    true_dfa = truth_oracle.target_dfa()
    threshold = _common_in_prefixes_threshold(
        kwargs["min_signal_strength"], common_in_prefixes_fpr
    )
    for classifier in classifiers:
        cuts = _state_cuts(classifier, true_dfa)

        split = _split_states(cuts)
        if split:
            state, accepted, rejected = split[0]
            raise AssertionError(
                f"a synthesis round cut state {state} both ways "
                f"({accepted} accept / {rejected} reject) -- more disagreement than "
                f"its {round_verify_fpr} per-prefix budget explains, so the family "
                f"had no single opinion about the state"
            )

        wrong = _wrongly_cut_states(cuts, true_dfa, threshold)
        if wrong:
            state, reached = wrong[0]
            raise AssertionError(
                f"a synthesis round cut state {state} against the language, on "
                f"{reached} prefixes -- at or above the {threshold:.1f} needed for "
                f"the state's label to be more than a coin flip, so the round had "
                f"the evidence and still cut the other way"
            )

        excused, total = _excused_wrong_prefixes(cuts, true_dfa, threshold)
        if total and binomial_side_of_boundary(
            excused, total, round_verify_fpr, failure_prob=round_verify_alpha
        ):
            raise AssertionError(
                f"a synthesis round cut {excused}/{total} of its decisive prefixes "
                f"against the language across states too rare to convict one at a "
                f"time -- together still past the {round_verify_fpr} budget, so the "
                f"family is wrong about the target rather than unlucky on one state"
            )
    return dfa
