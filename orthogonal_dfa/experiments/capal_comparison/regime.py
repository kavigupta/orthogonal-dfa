"""E-L*'s designed operating regime: the thresholds, and the verdict for one DFA.

A target outside this regime is one this repo's own benchmark generator would
have discarded, so measuring E-L* on it says more about the benchmark than
about the learner. `report` returns the measured values alongside the verdict,
so an exclusion recorded in an experiment JSON can be audited without rerunning
anything.
"""

from __future__ import annotations

from typing import Any, Dict

from orthogonal_dfa.l_star import preconditions

#: E-L*'s word-sampling length, fixed: it is `compute_pst`'s default, and every
#: regime measurement here is taken at it.
SAMPLE_LENGTH = 40

#: How many words each rate is estimated from.
NUM_SAMPLES = 2000

#: Taken from the filters `sample_balanced_benchmark` applies, with the
#: threshold values its callers in tests/test_lstar.py actually pass.
MIN_ACCEPT_OR_REJECT = 0.15  # tests/test_lstar.py passes this
MIN_CLASS_PRESERVING_FRAC = 0.05  # sample_balanced_benchmark default

#: A covered-states-only classifier (all E-L* can build) must reach at least
#: this accuracy, else the target has a state the prefix sampler never lands in
#: that carries a decision the learner cannot represent (e.g. Difficult10).
#: This closely predicts E-L*'s ceiling; it subsumes the weaker structural
#: "no transient non-start state" check.
MIN_COVERED_ACCURACY = 0.99


def report(aut: Any) -> Dict[str, Any]:
    """Is `aut` inside E-L*'s designed regime, and if not, why not?

    Applies the three conditions of preconditions.satisfies_preconditions:
    acceptance balance, class-preservation, and the covered-accuracy ceiling,
    all at SAMPLE_LENGTH.
    """
    length = SAMPLE_LENGTH
    rate = preconditions.acceptance_rate(aut, length=length, num_samples=NUM_SAMPLES)
    cp = preconditions.class_preserving_fraction(
        aut, length=length, num_samples=NUM_SAMPLES
    )
    ceiling = preconditions.covered_accuracy_ceiling(aut, length=length)
    covered = preconditions.covered_states(aut, length=length)
    uncovered = sorted(str(q) for q in aut.states if q not in covered)
    reasons = []
    if not MIN_ACCEPT_OR_REJECT <= rate <= 1 - MIN_ACCEPT_OR_REJECT:
        reasons.append(
            f"acceptance rate {rate:.3f} outside "
            f"[{MIN_ACCEPT_OR_REJECT}, {1 - MIN_ACCEPT_OR_REJECT}]"
        )
    if cp < MIN_CLASS_PRESERVING_FRAC:
        reasons.append(
            f"class-preserving fraction {cp:.3f} below {MIN_CLASS_PRESERVING_FRAC}"
        )
    if ceiling < MIN_COVERED_ACCURACY:
        reasons.append(
            f"covered-accuracy ceiling {ceiling:.3f} below "
            f"{MIN_COVERED_ACCURACY} (an uncovered state carries a decision)"
        )
    return {
        "sample_length": length,
        "accept_rate_at_sample_length": round(rate, 4),
        "class_preserving_frac": round(cp, 4),
        "covered_accuracy_ceiling": round(ceiling, 4),
        "uncovered_states": uncovered,
        "in_regime": not reasons,
        "excluded_because": reasons,
    }
