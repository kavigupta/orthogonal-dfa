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
- dominated_accuracy_ceiling: re-rooting the target at some start state still
  classifies almost every string, with the re-rooted run only passing through
  states the sampled prefixes land in at a comparable rate
"""

import math
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from typing import List, Optional, Tuple

import numpy as np
from automata.fa.dfa import DFA

from .sampler import Sampler, UniformSampler

DEFAULT_NUM_SAMPLES = 2000

MAX_VISIT_RATIO = 2

DEFAULT_MIN_DOMINATED_ACCURACY = 0.99


@lru_cache(maxsize=None)
def _sample_strings(
    sampler: Sampler, symbols: Tuple[int, ...], num_samples: int
) -> Tuple[bytes, ...]:
    """The sample every check below reads, as ``sampler`` draws it from a seed
    of 0.

    Held onto rather than redrawn, which each check used to do per DFA and was
    most of the cost of screening a population.  Keyed on the sampler, so a
    caller supplying one is spared the redrawing too.
    """
    rng = np.random.default_rng(0)
    return tuple(
        bytes(symbols[i] for i in sampler.sample(rng, len(symbols)))
        for _ in range(num_samples)
    )


def _samples(
    dfa: DFA, length: int, num_samples: int, sampler: Optional[Sampler] = None
) -> Tuple[bytes, ...]:
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
    symbols = tuple(sorted(dfa.input_symbols))
    # A string here is a byte per symbol, so that is what the DFA's symbols have to
    # be, whatever else a DFA is allowed to be labelled with.
    assert all(isinstance(s, int) and 0 <= s < 256 for s in symbols), symbols
    return _sample_strings(sampler, symbols, num_samples)


def _endpoint(dfa: DFA, string: bytes, start=None):
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


def start_length(length: int) -> int:
    """The first position at which a state's visits count toward covering it."""
    return math.ceil(length / 2)


def _run(dfa: DFA, string: bytes, start=None) -> list:
    """The states visited running ``string`` from ``start`` (default q0), the
    start included."""
    q = dfa.initial_state if start is None else start
    states = [q]
    for c in string:
        q = dfa.transitions[q][c]
        states.append(q)
    return states


def _window_visits(dfa: DFA, strings, length: int) -> Counter:
    """Per state, how many of ``strings`` visit it, run from q0, somewhere from
    ``start_length(length)`` on."""
    k = start_length(length)
    return Counter(q for s in strings for q in set(_run(dfa, s)[k:]))


def _visit_ratio(length: int) -> float:
    """``MAX_VISIT_RATIO`` per position: a run has ``length + 1`` of them, the
    window ``length + 1 - start_length(length)``."""
    return MAX_VISIT_RATIO * (length + 1) / (length + 1 - start_length(length))


def _start_floor(dfa: DFA, length: int, min_dominated_accuracy: float) -> float:
    return (1 - min_dominated_accuracy) / (2 * _visit_ratio(length) * len(dfa.states))


def _dominated(dfa: DFA, start, window: Counter, runs, length: int) -> set:
    rerooted = Counter(q for run in runs for q in set(run))
    ratio = _visit_ratio(length)
    return {start} | {q for q in dfa.states if rerooted[q] <= ratio * window[q]}


def dominated_states(
    dfa: DFA,
    start,
    *,
    length: int,
    num_samples: int = DEFAULT_NUM_SAMPLES,
    sampler: Optional[Sampler] = None,
) -> set:
    """
    ``start`` and the states q' that runs from it visit at most C times as often
    as runs from q0 do in the window:

        r_start(q') <= C * w(q'),   C = MAX_VISIT_RATIO * (L + 1) / (L + 1 - start_length(L))

    r counting a visit at any of the L + 1 positions, w one at any from
    ``start_length(L)`` on, so C compares visits per position.
    """
    strings = _samples(dfa, length, num_samples, sampler)
    runs = [_run(dfa, s, start) for s in strings]
    return _dominated(dfa, start, _window_visits(dfa, strings, length), runs, length)


def unsampled_states(
    dfa: DFA,
    *,
    length: int,
    min_dominated_accuracy: float = DEFAULT_MIN_DOMINATED_ACCURACY,
    num_samples: int = DEFAULT_NUM_SAMPLES,
    sampler: Optional[Sampler] = None,
) -> set:
    """
    The states too rarely in the window to start a re-rooted run from, those with

        w(q) < (1 - min_dominated_accuracy) / (2 C |Q|)
    """
    strings = _samples(dfa, length, num_samples, sampler)
    window = _window_visits(dfa, strings, length)
    floor = _start_floor(dfa, length, min_dominated_accuracy)
    return {q for q in dfa.states if window[q] < floor * num_samples}


def dominated_accuracy_ceiling(
    dfa: DFA,
    *,
    length: int,
    min_dominated_accuracy: float = DEFAULT_MIN_DOMINATED_ACCURACY,
    num_samples: int = DEFAULT_NUM_SAMPLES,
    sampler: Optional[Sampler] = None,
) -> float:
    """
    Best accuracy of re-rooting the target at a state not in ``unsampled_states``,
    counting a string only if its run from there stays within
    ``dominated_states(start)``.

    E-L* only sees a state through the prefixes that land in it, so whatever a
    re-rooted run passes through has to be landed in, in the window, at a
    comparable rate. Only the start is chosen; from there we follow the target's
    true transitions and read off the endpoint's true accept label.
    """
    strings = _samples(dfa, length, num_samples, sampler)
    window = _window_visits(dfa, strings, length)
    floor = _start_floor(dfa, length, min_dominated_accuracy)
    truth = [_endpoint(dfa, s) in dfa.final_states for s in strings]
    best = 0.0
    for start in dfa.states:
        if window[start] < floor * num_samples:
            continue
        runs = [_run(dfa, s, start) for s in strings]
        dominated = _dominated(dfa, start, window, runs, length)
        correct = sum(
            all(q in dominated for q in run) and (run[-1] in dfa.final_states) == t
            for run, t in zip(runs, truth)
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
    dominated_accuracy_ceiling: Optional[float] = None
    #: ``unsampled_states``, as strings -- populated when the ceiling is measured,
    #: since that is the check they explain.
    unsampled_states: Optional[List[str]] = None
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
    min_dominated_accuracy: float = DEFAULT_MIN_DOMINATED_ACCURACY,
    num_samples: int = DEFAULT_NUM_SAMPLES,
    short_circuit: bool = True,
    sampler: Optional[Sampler] = None,
) -> PreconditionReport:
    """Does ``dfa`` meet every learnability precondition, and if not, why not?

    All under length-``length`` sampling, uniform unless ``sampler`` says otherwise:

    - acceptance rate strictly between 0 and 1;
    - class-preserving fraction at least ``min_class_preserving_frac``;
    - dominated-accuracy ceiling at least ``min_dominated_accuracy``

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

    ceiling = dominated_accuracy_ceiling(
        dfa,
        length=length,
        min_dominated_accuracy=min_dominated_accuracy,
        num_samples=num_samples,
        sampler=sampler,
    )
    if ceiling < min_dominated_accuracy:
        reasons.append(
            f"dominated-accuracy ceiling {ceiling:.3f} below {min_dominated_accuracy} "
            "(a decision rides on a state the window rarely lands in)"
        )
    unsampled = unsampled_states(
        dfa,
        length=length,
        min_dominated_accuracy=min_dominated_accuracy,
        num_samples=num_samples,
        sampler=sampler,
    )
    unsampled = sorted(str(q) for q in unsampled)
    return PreconditionReport(length, rate, cp, ceiling, unsampled, tuple(reasons))
