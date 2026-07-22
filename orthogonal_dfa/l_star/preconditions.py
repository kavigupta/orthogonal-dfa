"""
Preconditions for E-L* learnability of a target DFA.

satisfies_preconditions(dfa, *, length, ...) is the main check, checking
the following conditions, for a particular length of uniform sampling:

- acceptance_rate: the language does not accept or reject nearly all strings
- class_preserving_fraction: some fraction of suffixes map all accept
  states to an accept state and all reject states to a reject state
- infinitely_reachable_states: every non-start state is reached by infinitely
  many strings, so the fixed-length sampler can land in it
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


def _reachable_from(dfa: DFA, sources) -> set:
    """States reachable from any of ``sources`` by following transitions."""
    seen = set(sources)
    stack = list(seen)
    while stack:
        s = stack.pop()
        for c in dfa.input_symbols:
            t = dfa.transitions[s][c]
            if t not in seen:
                seen.add(t)
                stack.append(t)
    return seen


def infinitely_reachable_states(dfa: DFA) -> set:
    """The states reached by infinitely many strings from the start.

    A state is reached by infinitely many strings iff it is forward-reachable
    from a state that lies on a cycle: pump the cycle for unboundedly many
    prefixes, then walk on to the state. Every other reachable state is reached
    by only finitely many (short) strings -- a transient state a fixed-length
    prefix sampler almost never lands in, so the learner cannot build it.
    """
    reachable = _reachable_from(dfa, [dfa.initial_state])
    on_cycle = {
        q
        for q in reachable
        if q in _reachable_from(dfa, [dfa.transitions[q][c] for c in dfa.input_symbols])
    }
    return _reachable_from(dfa, on_cycle)


def satisfies_preconditions(
    dfa: DFA,
    *,
    length: int,
    min_accept_or_reject: float = 0.15,
    min_class_preserving_frac: float = 0.05,
    num_samples: int = DEFAULT_NUM_SAMPLES,
) -> bool:
    """True iff ``dfa`` meets every learnability precondition:

    - acceptance rate in ``[min_accept_or_reject, 1 - min_accept_or_reject]``;
    - class-preserving fraction at least ``min_class_preserving_frac``;
    - every state other than the start is reached by infinitely many strings
      (the start is exempt -- it is always accessible via the empty string).

    The first two are sampled under length-``length`` uniform sampling; the last
    is an exact graph property. Checks run in increasing cost and short-circuit
    on the first failure.
    """
    rate = acceptance_rate(dfa, length=length, num_samples=num_samples)
    if not min_accept_or_reject <= rate <= 1 - min_accept_or_reject:
        return False
    cp = class_preserving_fraction(dfa, length=length, num_samples=num_samples)
    if cp < min_class_preserving_frac:
        return False
    infinite = infinitely_reachable_states(dfa)
    return all(q in infinite for q in dfa.states if q != dfa.initial_state)
