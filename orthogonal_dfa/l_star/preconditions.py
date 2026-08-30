"""
Preconditions for E-L* learnability of a target DFA.

satisfies_preconditions(dfa, *, length, ...) is the main check. It returns a
PreconditionReport -- truthy iff the DFA is admitted, and carrying the measured
values and a reason per failed condition. The conditions, for a particular
length and a particular way of sampling it -- uniform unless a sampler says
otherwise, since a condition is a property of the target under the distribution
the learner will draw from:

- acceptance_rate: the sampled strings are not all accepted or all rejected
- class_preserving_fraction: some fraction of suffixes map all accept
  states to an accept state and all reject states to a reject state
- covered_accuracy_ceiling: re-rooting the target at the best *covered* start
  state (all the learner can anchor to) still classifies almost every string
"""

from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from typing import List, Optional, Tuple

import numpy as np
from automata.fa.dfa import DFA

from .sampler import Sampler, UniformSampler

DEFAULT_NUM_SAMPLES = 2000

#: Bar for coverage by prefixes of the given length before we consider a state
#: "covered" by it.
DEFAULT_MIN_COVERAGE = 0.01


@lru_cache(maxsize=None)
def _sample_strings(
    sampler: Sampler, symbols: Tuple[int, ...], num_samples: int
) -> Tuple[Tuple[int, ...], ...]:
    """The sample every check below reads, as ``sampler`` draws it from a seed
    of 0.

    Held onto rather than redrawn, which each check used to do per DFA and was
    most of the cost of screening a population.  Keyed on the sampler, so a
    caller supplying one is spared the redrawing too.
    """
    rng = np.random.default_rng(0)
    return tuple(
        tuple(symbols[i] for i in sampler.sample(rng, len(symbols)))
        for _ in range(num_samples)
    )


def _samples(
    dfa: DFA, length: int, num_samples: int, sampler: Optional[Sampler] = None
) -> Tuple[Tuple[int, ...], ...]:
    """The strings every check below reads, over the DFA's own symbols.

    ``sampler`` is for a target whose learner does not draw uniformly: the checks
    have to ask about the distribution it will actually see, or they measure a
    language nobody learns.  Leaving it off asks for the uniform one, which is
    the same call and cannot answer differently.
    """
    if sampler is None:
        sampler = UniformSampler(length)
    assert sampler.length == length, (
        f"sampler draws {sampler.length} symbols but the preconditions were "
        f"asked for length {length}"
    )
    return _sample_strings(sampler, tuple(sorted(dfa.input_symbols)), num_samples)


def _endpoint(dfa: DFA, string: List[int], start=None):
    """The state reached by running ``string`` from ``start`` (default q0)."""
    q = dfa.initial_state if start is None else start
    for c in string:
        q = dfa.transitions[q][c]
    return q


def acceptance_rate(
    dfa: DFA,
    *,
    length: int,
    num_samples: int = DEFAULT_NUM_SAMPLES,
    sampler: Optional[Sampler] = None,
) -> float:
    """Fraction of random length-``length`` strings the DFA accepts."""
    accepts = sum(
        _endpoint(dfa, s) in dfa.final_states
        for s in _samples(dfa, length, num_samples, sampler)
    )
    return accepts / num_samples


def class_preserving_fraction(
    dfa: DFA,
    *,
    length: int,
    num_samples: int = DEFAULT_NUM_SAMPLES,
    sampler: Optional[Sampler] = None,
) -> float:
    """Fraction of random length-``length`` strings ``s`` for which *every*
    state ``q`` satisfies ``(q in F) == (delta*(q, s) in F)`` -- the suffixes
    that reset the whole state set into a single accept/reject class."""
    finals = dfa.final_states
    states = list(dfa.states)
    preserving = sum(
        all((q in finals) == (_endpoint(dfa, s, q) in finals) for q in states)
        for s in _samples(dfa, length, num_samples, sampler)
    )
    return preserving / num_samples


def covered_states(
    dfa: DFA,
    *,
    length: int,
    num_samples: int = DEFAULT_NUM_SAMPLES,
    min_coverage: float = DEFAULT_MIN_COVERAGE,
    sampler: Optional[Sampler] = None,
) -> set:
    """
    The states reached as the endpoint of at least ``min_coverage`` of random length-``length`` strings.
    """
    counts = Counter(
        _endpoint(dfa, s) for s in _samples(dfa, length, num_samples, sampler)
    )
    return {q for q, c in counts.items() if c / num_samples >= min_coverage}


def covered_accuracy_ceiling(
    dfa: DFA,
    *,
    length: int,
    num_samples: int = DEFAULT_NUM_SAMPLES,
    min_coverage: float = DEFAULT_MIN_COVERAGE,
    sampler: Optional[Sampler] = None,
) -> float:
    """
    Best accuracy reachable when the classifier may only be started from a
    covered state.

    E-L* discovers states from where its sampled prefixes land, so it can only
    anchor its automaton at covered state; if the true initial state is uncovered
    it cannot represent it. Only the start is constrained, from there we follow
    the target's true transitions and read off the endpoint's true accept label.
    """
    strings = _samples(dfa, length, num_samples, sampler)
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


@dataclass(frozen=True)
class PreconditionReport:
    """The verdict, plus the measurements and the reasons behind it.

    Truthy iff every precondition holds, so ``if satisfies_preconditions(...)``
    reads the same as when this was a bare bool. A measurement is ``None`` when
    short-circuiting meant it was never taken.
    """

    length: int
    acceptance_rate: float
    class_preserving_fraction: Optional[float] = None
    covered_accuracy_ceiling: Optional[float] = None
    #: States no sampled prefix lands in, as strings -- populated when the
    #: ceiling is measured, since that is the check they explain.
    uncovered_states: Optional[List[str]] = None
    #: One entry per failed precondition, naming the measured value and the
    #: threshold it missed. Empty iff the DFA is admitted.
    reasons: Tuple[str, ...] = ()

    @property
    def satisfied(self) -> bool:
        return not self.reasons

    def __bool__(self) -> bool:
        return self.satisfied


def satisfies_preconditions(
    dfa: DFA,
    *,
    length: int,
    min_class_preserving_frac: float = 0.02,
    min_covered_accuracy: float = 0.99,
    num_samples: int = DEFAULT_NUM_SAMPLES,
    short_circuit: bool = True,
    sampler: Optional[Sampler] = None,
) -> PreconditionReport:
    """Does ``dfa`` meet every learnability precondition, and if not, why not?

    All under length-``length`` sampling, uniform unless ``sampler`` says otherwise:

    - acceptance rate strictly between 0 and 1;
    - class-preserving fraction at least ``min_class_preserving_frac``;
    - covered-accuracy ceiling at least ``min_covered_accuracy``

    The acceptance-rate check only rejects degeneracy -- a language that is
    constant over the sampled strings, which E-L* cannot get signal from and
    which the other two checks pass trivially. It carries no balance
    requirement: an imbalanced language is the class-preserving check's business.

    Checks run in increasing cost. By default they stop at the first failure,
    which is what a caller wanting only a verdict should do; pass
    ``short_circuit=False`` to measure everything and collect every reason.
    """
    reasons: List[str] = []
    rate = acceptance_rate(dfa, length=length, num_samples=num_samples, sampler=sampler)
    if rate in (0.0, 1.0):
        reasons.append(
            f"acceptance rate {rate} degenerate: every sampled string of length "
            f"{length} has the same label"
        )
        if short_circuit:
            return PreconditionReport(length, rate, reasons=tuple(reasons))

    cp = class_preserving_fraction(
        dfa, length=length, num_samples=num_samples, sampler=sampler
    )
    if cp < min_class_preserving_frac:
        reasons.append(
            f"class-preserving fraction {cp:.3f} below {min_class_preserving_frac}"
        )
        if short_circuit:
            return PreconditionReport(length, rate, cp, reasons=tuple(reasons))

    ceiling = covered_accuracy_ceiling(
        dfa, length=length, num_samples=num_samples, sampler=sampler
    )
    if ceiling < min_covered_accuracy:
        reasons.append(
            f"covered-accuracy ceiling {ceiling:.3f} below "
            f"{min_covered_accuracy} (an uncovered state carries a decision)"
        )
    covered = covered_states(
        dfa, length=length, num_samples=num_samples, sampler=sampler
    )
    uncovered = sorted(str(q) for q in dfa.states if q not in covered)
    return PreconditionReport(length, rate, cp, ceiling, uncovered, tuple(reasons))
