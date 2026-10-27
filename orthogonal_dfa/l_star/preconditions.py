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
- non_start_zone_accuracy_ceiling: re-rooting the target at the best start state
  (a state in its own non-start zone) still classifies almost every string,
  without the re-rooted run leaving that zone -- at every string length from
  ``ceil(length / 2)`` to twice ``length``, taking the worst length for each start

Under uniform sampling every condition is computed exactly, as a chain over
states, pairs of states or sets of states; under any other sampler it is
estimated from ``num_samples`` draws.
"""

import copy
import itertools
import math
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from typing import List, Optional, Tuple

import numpy as np
from automata.fa.dfa import DFA

from .sampler import Sampler, UniformSampler

DEFAULT_NUM_SAMPLES = 2000

MAX_VISIT_RATIO = 10

MIN_CLASS_PRESERVING_FRAC = 0.02

#: A rate below this counts as zero, which is all a sample of
#: ``DEFAULT_NUM_SAMPLES`` strings could resolve.
MIN_RESOLVED_RATE = 1 / DEFAULT_NUM_SAMPLES


def _exact(length: int, sampler: Optional[Sampler]) -> bool:
    """Whether ``sampler`` draws uniformly, so the checks need no sample."""
    if sampler is None:
        return True
    assert sampler.length == length, (
        f"sampler draws {sampler.length} symbols but the preconditions were "
        f"asked for length {length}"
    )
    return type(sampler) is UniformSampler  # pylint: disable=unidiomatic-typecheck


def _tables(dfa: DFA):
    """``(states, table, final, initial)``, with states indexed by position in
    ``states``, ``table[q, c]`` the successor of ``q`` on the ``c``-th symbol."""
    states = sorted(dfa.states, key=str)
    index = {q: i for i, q in enumerate(states)}
    symbols = sorted(dfa.input_symbols)
    table = np.array([[index[dfa.transitions[q][c]] for c in symbols] for q in states])
    final = np.array([q in dfa.final_states for q in states])
    return states, table, final, index[dfa.initial_state]


def _step(dist: np.ndarray, table: np.ndarray) -> np.ndarray:
    """``dist``, a distribution over states along its last axis, after one more
    uniformly drawn symbol."""
    n, k = table.shape
    flat = dist.reshape(-1, n)
    rows = np.arange(len(flat))[:, None] * n
    out = sum(
        np.bincount(
            (rows + table[:, c]).ravel(), weights=flat.ravel(), minlength=flat.size
        )
        for c in range(k)
    )
    return (out / k).reshape(dist.shape)


def _distribution_at(table: np.ndarray, initial: int, length: int) -> np.ndarray:
    dist = np.zeros(len(table))
    dist[initial] = 1.0
    for _ in range(length):
        dist = _step(dist, table)
    return dist


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
    if _exact(length, sampler):
        _, table, final, initial = _tables(dfa)
        return float(_distribution_at(table, initial, length) @ final)
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
    if _exact(length, sampler):
        _, table, final, _ = _tables(dfa)
        return _exact_class_preserving_fraction(table, final, length)
    finals = dfa.final_states
    states = list(dfa.states)
    preserving = sum(
        all((q in finals) == (_endpoint(dfa, s, q) in finals) for q in states)
        for s in _samples(dfa, length, num_samples, sampler)
    )
    return preserving / num_samples


def _exact_class_preserving_fraction(
    table: np.ndarray, final: np.ndarray, length: int
) -> float:
    """
    ``s`` preserves classes iff ``{q : delta(q, s) in F} == F``. Reading ``s``
    last symbol first, that set is a chain over subsets of states starting at
    ``F``: prepending ``c`` takes the set ``S`` to ``{q : delta(q, c) in S}``.
    """
    subsets, probs = final[None, :], np.ones(1)
    k = table.shape[1]
    for _ in range(length):
        subsets = np.concatenate([subsets[:, table[:, c]] for c in range(k)])
        probs = np.tile(probs, k) / k
        subsets, inverse = np.unique(subsets, axis=0, return_inverse=True)
        probs = np.bincount(inverse.ravel(), weights=probs)
    return float(probs[(subsets == final).all(axis=1)].sum())


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


def _window_visits(
    dfa: DFA, *, length: int, num_samples: int, sampler: Optional[Sampler]
) -> Counter:
    """Per state, the sampled runs from q0 visiting it at positions
    ``start_length(length)`` to ``length``."""
    k = start_length(length)
    return Counter(
        q
        for s in _samples(dfa, length, num_samples, sampler)
        for q in set(_run(dfa, s)[k:])
    )


def _exact_visits(table: np.ndarray, dist: np.ndarray, steps: int) -> np.ndarray:
    """Per state ``q``, the probability a run whose distribution is ``dist`` (over
    the last axis) visits ``q`` within the next ``steps`` steps, its current state
    included."""
    n = len(table)
    # avoiding[..., q, x]: the run at x, with the mass that has visited q removed
    avoiding = np.repeat(dist[..., None, :], n, axis=-2)
    diagonal = (..., np.arange(n), np.arange(n))
    visits = avoiding[diagonal].copy()
    avoiding[diagonal] = 0
    for _ in range(steps):
        avoiding = _step(avoiding, table)
        visits += avoiding[diagonal]
        avoiding[diagonal] = 0
    return visits


def _exact_window_visits(table: np.ndarray, initial: int, length: int) -> np.ndarray:
    """``_window_visits`` as probabilities."""
    k = start_length(length)
    return _exact_visits(table, _distribution_at(table, initial, k), length - k)


def non_start_zone_states(
    dfa: DFA,
    *,
    length: int,
    num_samples: int = DEFAULT_NUM_SAMPLES,
    sampler: Optional[Sampler] = None,
) -> dict:
    """
    Per state s in its own non-start zone, that zone

        {q : r_s(q) <= C * w(q)},   C = MAX_VISIT_RATIO * (L + 1) / (L + 1 - k)

    with w(q) the runs from q0 visiting q at positions k = start_length(L) to L, and
    r_s(q) the runs from s visiting q at positions 0 to L.
    """
    k = start_length(length)
    ratio = MAX_VISIT_RATIO * (length + 1) / (length + 1 - k)
    if _exact(length, sampler):
        states, table, _, initial = _tables(dfa)
        window = _exact_window_visits(table, initial, length)
        rerooted = _exact_visits(table, np.eye(len(states)), length)
        zone = rerooted <= ratio * window
        return {
            states[s]: {q for q, inside in zip(states, zone[s]) if inside}
            for s in range(len(states))
            if window[s] > 0 and zone[s, s]
        }
    strings = _samples(dfa, length, num_samples, sampler)
    window = _window_visits(
        dfa, length=length, num_samples=num_samples, sampler=sampler
    )
    zones = {}
    for start in window:
        rerooted = Counter(q for s in strings for q in set(_run(dfa, s, start)))
        zone = {q for q in dfa.states if rerooted[q] <= ratio * window[q]}
        if start in zone:
            zones[start] = zone
    return zones


@lru_cache(maxsize=None)
def _sampler_at_length(sampler: Sampler, length: int) -> Sampler:
    """``sampler`` drawing ``length`` symbols instead; its other settings stay."""
    if sampler.length == length:
        return sampler
    other = copy.copy(sampler)
    object.__setattr__(other, "length", length)
    return other


def _strings_at_length(
    dfa: DFA, length: int, num_samples: int, sampler: Optional[Sampler]
) -> np.ndarray:
    """Strings of ``length`` as symbol indices, one per row: every string when
    uniform and there are at most ``num_samples``, else a sample."""
    symbols = tuple(sorted(dfa.input_symbols))
    if length == 0:
        return np.zeros((1, 0), dtype=int)
    if sampler is None and len(symbols) ** length <= num_samples:
        return np.array(
            list(itertools.product(range(len(symbols)), repeat=length)), dtype=int
        ).reshape(-1, length)
    if sampler is None:
        sampler = UniformSampler(length)
    else:
        sampler = _sampler_at_length(sampler, length)
    strings = _samples(dfa, length, num_samples, sampler)
    index = {c: i for i, c in enumerate(symbols)}
    return np.array([[index[c] for c in s] for s in strings], dtype=int).reshape(
        -1, length
    )


def _worst_start_accuracies(
    dfa: DFA,
    *,
    length: int,
    string_lengths: range,
    num_samples: int,
    sampler: Optional[Sampler],
) -> np.ndarray:
    """Per start of ``non_start_zone_states`` at ``length``, its lowest accuracy
    over ``string_lengths``."""
    zones = non_start_zone_states(
        dfa, length=length, num_samples=num_samples, sampler=sampler
    )
    states, table, final, initial = _tables(dfa)
    index = {q: i for i, q in enumerate(states)}
    starts = np.array(sorted(index[q] for q in zones), dtype=int)
    allowed = np.array(
        [[q in zones[states[s]] for q in states] for s in starts], dtype=bool
    ).reshape(len(starts), len(states))
    if _exact(length, sampler):
        return _exact_worst_start_accuracies(
            table,
            final,
            initial,
            starts=starts,
            allowed=allowed,
            string_lengths=string_lengths,
        )
    worst = np.ones(len(starts))
    for ell in string_lengths:
        strings = _strings_at_length(dfa, ell, num_samples, sampler)
        initial_end, _ = _run_from(
            table,
            np.ones((1, len(states)), dtype=bool),
            strings,
            np.array([initial]),
        )
        truth = final[initial_end[0]]
        ends, stayed = _run_from(table, allowed, strings, starts)
        accuracy = (stayed & (final[ends] == truth)).mean(axis=1)
        worst = np.minimum(worst, accuracy)
    return worst


def _exact_worst_start_accuracies(
    table, final, initial, *, starts, allowed, string_lengths: range
) -> np.ndarray:
    """``_worst_start_accuracies`` under uniform sampling, from the joint
    distribution of (run from the initial state, run from each start)."""
    n = len(table)
    pair_table = (table[:, None, :] * n + table[None, :, :]).reshape(n * n, -1)
    # joint[s, x, y]: x the run from the initial state, y the run from starts[s],
    # with the mass whose y-run has left the start's zone dropped
    joint = np.zeros((len(starts), n, n))
    joint[np.arange(len(starts)), initial, starts] = 1.0
    agree = final[:, None] == final[None, :]
    worst = np.ones(len(starts))
    for ell in range(string_lengths.stop):
        if ell in string_lengths:
            worst = np.minimum(worst, (joint * agree).sum(axis=(1, 2)))
        joint = _step(joint.reshape(len(starts), n * n), pair_table)
        joint = joint.reshape(len(starts), n, n) * allowed[:, None, :]
    return worst


def non_start_zone_accuracy_at(
    dfa: DFA,
    *,
    length: int,
    num_samples: int = DEFAULT_NUM_SAMPLES,
    sampler: Optional[Sampler] = None,
) -> float:
    """``non_start_zone_accuracy_ceiling`` on strings of exactly ``length``."""
    worst = _worst_start_accuracies(
        dfa,
        length=length,
        string_lengths=range(length, length + 1),
        num_samples=num_samples,
        sampler=sampler,
    )
    return float(worst.max(initial=0.0))


def non_start_zone_accuracy_ceiling(
    dfa: DFA,
    *,
    length: int,
    num_samples: int = DEFAULT_NUM_SAMPLES,
    sampler: Optional[Sampler] = None,
) -> float:
    """
    Best accuracy reachable when the classifier may only be started from a start
    of ``non_start_zone_states``, counting a string only if the run from there
    stays within that start's zone.

    E-L* discovers states from where its sampled prefixes land, so it can only
    anchor its automaton at a state they land in often. Only the start is
    constrained, from there we follow the target's true transitions and read off
    the endpoint's true accept label. A run through a state the window lands in
    far less often than the run does uses transitions the learner rarely checks.

    A start's accuracy is its minimum over string lengths ``ceil(length / 2)`` to
    2 * ``length``; the ceiling is the best start's. A custom ``sampler`` is asked
    for the other lengths by changing only its ``length``, and is always sampled,
    never enumerated.
    """
    worst = _worst_start_accuracies(
        dfa,
        length=length,
        string_lengths=range((length + 1) // 2, 2 * length + 1),
        num_samples=num_samples,
        sampler=sampler,
    )
    return float(worst.max(initial=0.0))


def _run_from(table, allowed, strings, starts):
    """Endpoints of running each row of ``strings`` from each of ``starts``, and
    whether the run stayed in its start's row of ``allowed``."""
    rows = np.arange(len(starts))[:, None]
    current = np.repeat(starts[:, None], len(strings), axis=1)
    stayed = allowed[rows, current]
    for j in range(strings.shape[1]):
        current = table[current, strings[:, j]]
        stayed &= allowed[rows, current]
    return current, stayed


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
    non_start_zone_accuracy_ceiling: Optional[float] = None
    #: States no sampled prefix lands in, as strings -- populated when the
    #: ceiling is measured, since that is the check they explain.
    unvisited_states: Optional[List[str]] = None
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
    min_class_preserving_frac: float = MIN_CLASS_PRESERVING_FRAC,
    min_non_start_zone_accuracy: float = 0.99,
    num_samples: int = DEFAULT_NUM_SAMPLES,
    short_circuit: bool = True,
    sampler: Optional[Sampler] = None,
) -> PreconditionReport:
    """Does ``dfa`` meet every learnability precondition, and if not, why not?

    All under length-``length`` sampling, uniform unless ``sampler`` says otherwise:

    - acceptance rate strictly between 0 and 1;
    - class-preserving fraction at least ``min_class_preserving_frac``;
    - non-start-zone accuracy ceiling at least ``min_non_start_zone_accuracy``

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
    if min(rate, 1 - rate) < MIN_RESOLVED_RATE:
        reasons.append(
            f"acceptance rate {rate} degenerate: almost every string of length "
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

    ceiling = non_start_zone_accuracy_ceiling(
        dfa, length=length, num_samples=num_samples, sampler=sampler
    )
    if ceiling < min_non_start_zone_accuracy:
        reasons.append(
            f"non-start-zone accuracy ceiling {ceiling:.3f} below "
            f"{min_non_start_zone_accuracy} (a decision rides on a state the window "
            "lands in far less than the run does)"
        )
    if _exact(length, sampler):
        states, table, _, initial = _tables(dfa)
        window = _exact_window_visits(table, initial, length)
        visited = {q for q, w in zip(states, window) if w >= MIN_RESOLVED_RATE}
    else:
        visited = _window_visits(
            dfa, length=length, num_samples=num_samples, sampler=sampler
        )
    unvisited = sorted(str(q) for q in dfa.states if q not in visited)
    return PreconditionReport(length, rate, cp, ceiling, unvisited, tuple(reasons))
