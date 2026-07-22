"""
Preconditions for E-L* learnability of a target DFA.

satisfies_preconditions(dfa, *, length, ...) is the main check, checking
the following conditions, for a particular length of uniform sampling:

- acceptance_rate: the language does not accept or reject nearly all strings
- class_preserving_fraction: some fraction of suffixes map all accept
  states to an accept state and all reject states to a reject state
"""

from typing import List

import numpy as np
from automata.fa.dfa import DFA

DEFAULT_NUM_SAMPLES = 2000


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


def satisfies_preconditions(
    dfa: DFA,
    *,
    length: int,
    min_accept_or_reject: float = 0.15,
    min_class_preserving_frac: float = 0.05,
    num_samples: int = DEFAULT_NUM_SAMPLES,
) -> bool:
    """True iff ``dfa`` meets every learnability precondition, under
    length-``length`` uniform sampling:

    - acceptance rate in ``[min_accept_or_reject, 1 - min_accept_or_reject]``;
    - class-preserving fraction at least ``min_class_preserving_frac``.

    Checks run in increasing cost and short-circuit on the first failure.
    """
    rate = acceptance_rate(dfa, length=length, num_samples=num_samples)
    if not min_accept_or_reject <= rate <= 1 - min_accept_or_reject:
        return False
    cp = class_preserving_fraction(dfa, length=length, num_samples=num_samples)
    return cp >= min_class_preserving_frac
