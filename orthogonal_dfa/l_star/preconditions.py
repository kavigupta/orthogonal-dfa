"""
Preconditions for E-L* learnability of a target DFA.

satisfies_preconditions(dfa, *, length, ...) is the main check, checking
the following conditions, for a particular length of uniform sampling:

- acceptance_rate: the language does not accept or reject nearly all strings
- class_preserving_fraction: some fraction of suffixes map all accept
  states to an accept state and all reject states to a reject state
- covered_accuracy_ceiling: the best a covered-states-only classifier can
  do is near-perfect, i.e. the states the length-``length`` prefix sampler
  never lands in (which the learner cannot build) carry no classification
  decision it would miss
"""

from collections import Counter
from typing import List

import numpy as np
from automata.fa.dfa import DFA

DEFAULT_NUM_SAMPLES = 2000

#: Bar for coverage by prefixes of the given length before we consider a state
#: "covered" by it.
DEFAULT_MIN_COVERAGE = 0.01


def _random_string(dfa: DFA, length: int, rng: np.random.Generator) -> List[int]:
    """A uniform random string of ``length`` symbols over the DFA's alphabet."""
    return rng.choice(sorted(dfa.input_symbols), size=length).tolist()


def _endpoint(dfa: DFA, string: List[int], start=None):
    """The state reached by running ``string`` from ``start`` (default q0)."""
    q = dfa.initial_state if start is None else start
    for c in string:
        q = dfa.transitions[q][c]
    return q


def acceptance_rate(
    dfa: DFA, *, length: int, num_samples: int = DEFAULT_NUM_SAMPLES
) -> float:
    """Fraction of random length-``length`` strings the DFA accepts."""
    rng = np.random.default_rng(0)
    accepts = sum(
        _endpoint(dfa, _random_string(dfa, length, rng)) in dfa.final_states
        for _ in range(num_samples)
    )
    return accepts / num_samples


def class_preserving_fraction(
    dfa: DFA, *, length: int, num_samples: int = DEFAULT_NUM_SAMPLES
) -> float:
    """Fraction of random length-``length`` strings ``s`` for which *every*
    state ``q`` satisfies ``(q in F) == (delta*(q, s) in F)`` -- the suffixes
    that reset the whole state set into a single accept/reject class."""
    rng = np.random.default_rng(0)
    finals = dfa.final_states
    states = list(dfa.states)
    preserving = sum(
        all((q in finals) == (_endpoint(dfa, s, q) in finals) for q in states)
        for s in (_random_string(dfa, length, rng) for _ in range(num_samples))
    )
    return preserving / num_samples


def covered_states(
    dfa: DFA,
    *,
    length: int,
    num_samples: int = DEFAULT_NUM_SAMPLES,
    min_coverage: float = DEFAULT_MIN_COVERAGE,
) -> set:
    """The states the learner can actually build: those reached as the endpoint
    of at least ``min_coverage`` of random length-``length`` strings.

    E-L* discovers states from where its sampled prefixes end, so a state no
    prefix lands in cannot be built -- regardless of whether it is structurally
    reachable or even on a cycle. This is the empirical, length-dependent notion
    that predicts learnability; structural reachability over-counts it (issue
    #128, and the [336]/[377] false positives, are exactly recurrent-but-uncovered
    states).
    """
    rng = np.random.default_rng(0)
    counts = Counter(
        _endpoint(dfa, _random_string(dfa, length, rng)) for _ in range(num_samples)
    )
    return {q for q, c in counts.items() if c / num_samples >= min_coverage}


def covered_accuracy_ceiling(
    dfa: DFA,
    *,
    length: int,
    num_samples: int = DEFAULT_NUM_SAMPLES,
    min_coverage: float = DEFAULT_MIN_COVERAGE,
) -> float:
    """Best accuracy any covered-states-only classifier reaches on random
    length-``length`` strings.

    The learner can build only *covered* states (``covered_states``) -- an
    uncovered state gets essentially no sampled prefixes to aggregate over. So
    its achievable accuracy is capped by the best classifier the covered
    states allow: run each test string through the true transitions from the
    single best covered start state and read off that endpoint's accept label.
    ``1 - ceiling`` is the mass of strings it must misclassify because telling
    them apart needs an uncovered state it cannot build (issue #128) -- e.g. an
    initial state that routes, by an early character, into covered states of
    differing acceptance (then the ceiling is a coin flip).

    This tracks E-L*'s achievable accuracy closely (measured 0.738 vs actual
    0.751 on the [336] false positive), so it subsumes the weaker structural
    "every non-start state is infinitely reachable" check that admitted it.
    """
    rng = np.random.default_rng(0)
    strings = [_random_string(dfa, length, rng) for _ in range(num_samples)]
    truth = [_endpoint(dfa, s) in dfa.final_states for s in strings]
    counts = Counter(_endpoint(dfa, s) for s in strings)
    covered = {q for q, c in counts.items() if c / num_samples >= min_coverage}
    best = 0.0
    for start in covered:
        correct = sum(
            (_endpoint(dfa, s, start) in dfa.final_states) == t
            for s, t in zip(strings, truth)
        )
        best = max(best, correct / num_samples)
    return best


def satisfies_preconditions(
    dfa: DFA,
    *,
    length: int,
    min_accept_or_reject: float = 0.15,
    min_class_preserving_frac: float = 0.05,
    min_covered_accuracy: float = 0.99,
    num_samples: int = DEFAULT_NUM_SAMPLES,
) -> bool:
    """True iff ``dfa`` meets every learnability precondition, all under
    length-``length`` uniform sampling:

    - acceptance rate in ``[min_accept_or_reject, 1 - min_accept_or_reject]``
      (there is signal to separate accept from reject);
    - class-preserving fraction at least ``min_class_preserving_frac`` (some
      suffix separates the states);
    - covered-accuracy ceiling at least ``min_covered_accuracy`` (the
      uncovered states carry no classification decision the learner would miss).

    Checks run in increasing cost and short-circuit on the first failure.
    """
    rate = acceptance_rate(dfa, length=length, num_samples=num_samples)
    if not min_accept_or_reject <= rate <= 1 - min_accept_or_reject:
        return False
    cp = class_preserving_fraction(dfa, length=length, num_samples=num_samples)
    if cp < min_class_preserving_frac:
        return False
    ceiling = covered_accuracy_ceiling(dfa, length=length, num_samples=num_samples)
    return ceiling >= min_covered_accuracy
