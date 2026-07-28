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

    `preconditions.satisfies_preconditions` at this repo's thresholds, flattened
    for the experiment JSON. Every reason is collected rather than just the
    first, so an exclusion records everything that disqualified the target.
    """
    r = preconditions.satisfies_preconditions(
        aut,
        length=SAMPLE_LENGTH,
        min_accept_or_reject=MIN_ACCEPT_OR_REJECT,
        min_class_preserving_frac=MIN_CLASS_PRESERVING_FRAC,
        min_covered_accuracy=MIN_COVERED_ACCURACY,
        num_samples=NUM_SAMPLES,
        short_circuit=False,
    )
    return {
        "sample_length": r.length,
        "accept_rate_at_sample_length": round(r.acceptance_rate, 4),
        "class_preserving_frac": round(r.class_preserving_fraction, 4),
        "covered_accuracy_ceiling": round(r.covered_accuracy_ceiling, 4),
        "uncovered_states": r.uncovered_states,
        "in_regime": r.satisfied,
        "excluded_because": list(r.reasons),
    }
