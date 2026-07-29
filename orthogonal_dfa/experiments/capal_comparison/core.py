"""Shared measurement machinery for the CAPAL vs E-L* experiments.

The point of this module is that both learners are measured identically:
same evaluation word list, same accuracy definition, same query accounting.
Anything that differs between the two learners is a property of the learner,
not of the harness.

Noise. CAPAL parameterises it as `eta` (flip probability); this repo parameterises
it as `p_correct = 1 - eta`, with E-L* additionally told `min_signal_strength =
0.5 - eta` so it can size its suffix population.

Both learners are therefore told the true noise rate, but CAPAL discards part
of it: upstream floors its working estimate at
    `eta_hat = min(0.49, max(eta, 0.15))` (capal.py:931)
This is the choice of the CAPAL authors, not this harness.

Equivalence queries are recorded alongside membership queries, because the two
learners do not have the same oracles: CAPAL is given a perfect EQ (the paper's
pMAT assumption), and its counterexamples come back as gold labels that also
shadow the MQ, so those strings cost it nothing thereafter. E-L* has no EQ at
all and manufactures its counterexamples out of membership queries, so its EQ
count is 0 by construction.
"""

from __future__ import annotations

import contextlib
import io
import json
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

from orthogonal_dfa.capal_official import fit_with_fallback, make_learner
from orthogonal_dfa.l_star.learn import learn_dfa

#: Bump when the emitted record shape changes incompatibly. 4: queries_total is
#: populated for CAPAL too, so it no longer doubles as "this row is E-L*".
SCHEMA_VERSION = 4

LEARNER_CAPAL = "CAPAL"
LEARNER_ELSTAR = "E-L*"

REPO_ROOT = Path(__file__).resolve().parents[3]

#: Shared evaluation settings. Both learners' hypotheses are scored on the very
#: same sampled word list, so accuracies are directly comparable.
EVAL_COUNT = 5000
EVAL_MAX_LEN = 40
EVAL_SEED = 0x1234


def eta_to_signal_strength(eta: float) -> float:
    """E-L* is configured by signal strength; CAPAL by eta. They are the same
    knob: `p_correct = 0.5 + signal = 1 - eta`."""
    return 0.5 - eta


# -- shared evaluation --------------------------------------------------------


def eval_words(
    symbols: int,
    *,
    count: int = EVAL_COUNT,
    max_len: int = EVAL_MAX_LEN,
    seed: int = EVAL_SEED,
) -> List[List[int]]:
    """The fixed word list every hypothesis is scored on, as symbol indices."""
    rng = random.Random(seed)
    return [
        [rng.randrange(symbols) for _ in range(rng.randint(1, max_len))]
        for _ in range(count)
    ]


def accuracy(
    predict: Callable[[List[int]], bool],
    truth: Callable[[List[int]], bool],
    words: Sequence[List[int]],
) -> float:
    """Fraction of `words` on which `predict` agrees with the noiseless truth."""
    return sum(bool(predict(w)) == bool(truth(w)) for w in words) / len(words)


# -- uniform result record ----------------------------------------------------


@dataclass
class Cell:
    """
    One (benchmark, learner, eta, seed) measurement.

    The two membership columns answer different questions.

    `queries_distinct` is the algorithmic cost: distinct strings whose label was
    drawn from the noisy oracle. Under persistent noise a repeat carries no new
    information, so this is what the two learners can be compared on.

    `queries_total` is the engineering cost: calls each implementation actually
    issues today. CAPAL's equals its distinct count because upstream memoises
    above the MQ (SameStateOracle._label), so a repeat never reaches the oracle.
    E-L* issues its repeats for real -- it draws in batches, so it cannot simply
    put a cache in front -- and the gap between its two columns is headroom, not
    a property of the algorithm. Do not read it as a cost CAPAL avoids: measured
    at the matching layer (`ss._label`) CAPAL re-asks about 2.8x, it just
    re-asks into a dict.

    `equivalence_queries` is the other half of the oracle cost -- see the module
    docstring on why it has to be read alongside `queries_distinct`.

    `converged` is the one column that is *not* the same claim on both sides.
    CAPAL's comes from its PerfectEQ and means exact equality with the target.
    E-L* has no such signal, so it gets `accuracy == 1.0` on the sampled word
    list -- an upper bound, which a hypothesis differing only off the sample
    passes. Do not read an E-L* `converged` as exactness.
    """

    benchmark: str
    family: str
    learner: str
    eta: float
    seed: int
    target_states: Optional[int] = None
    alphabet_size: Optional[int] = None
    learner_config: Dict[str, Any] = field(default_factory=dict)
    learned_states: Optional[int] = None
    accuracy: Optional[float] = None
    converged: Optional[bool] = None
    queries_total: Optional[int] = None
    queries_distinct: Optional[int] = None
    equivalence_queries: Optional[int] = None
    seconds: Optional[float] = None
    #: Exception class name, so the report can group failure modes without
    #: parsing prose. E.g. E-L* raises GaveUpOnSuffixSearch when no suffix
    #: family clears the signal threshold -- a learner outcome, not a crash.
    error_type: Optional[str] = None
    error: Optional[str] = None

    def finalize(self) -> "Cell":
        """Round the float fields, for digits nobody reads. Called by the
        drivers once a cell is fully populated."""
        self.seconds = None if self.seconds is None else round(self.seconds, 3)
        self.accuracy = None if self.accuracy is None else round(self.accuracy, 6)
        return self


# -- learner drivers ----------------------------------------------------------


def run_capal_cell(
    target: Any,
    *,
    benchmark: str,
    family: str,
    eta: float,
    seed: int,
    words: Sequence[List[int]],
    truth: Callable[[List[int]], bool],
    alphabet: Sequence[str],
) -> Cell:
    """Run upstream CAPAL on `target` and score it on the shared word list."""
    learner = make_learner(target, eta, seed=seed)
    cell = Cell(
        benchmark=benchmark,
        family=family,
        learner=LEARNER_CAPAL,
        eta=eta,
        seed=seed,
        target_states=target.num_states,
        alphabet_size=len(target.alphabet),
        # Read off the learner rather than restated here, so this records what
        # CAPAL ran with even when `make_learner`'s defaults move. `eta_hat` is
        # not a knob at all: upstream derives it from eta and floors it, see the
        # module docstring.
        learner_config={
            "max_iters": learner.cfg.max_iters,
            "max_same_samples": learner.ss.cfg.max_samples,
            "suffix_pool_len_max": learner.ss.cfg.pool_len_max,
            "alpha": learner.ss.cfg.alpha,
            "tau_cap": learner.ss.cfg.tau_cap,
            "suffix_pool_init": learner.ss.cfg.pool_init,
            "enum_depth": learner.ss.cfg.enum_depth,
            "extra_len_max": learner.ss.cfg.extra_len_max,
            "eta_hat": learner.ss.eta_hat,
        },
    )

    # Every EQ call is a perfect-information answer E-L* has no access to, so
    # it is counted rather than left implicit.
    eq_calls = {"n": 0}
    inner_eq = learner.eq.query

    def counting_eq(hyp: Any) -> Any:
        eq_calls["n"] += 1
        return inner_eq(hyp)

    learner.eq.query = counting_eq  # type: ignore[method-assign]

    t0 = time.time()
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            dfa, converged = fit_with_fallback(learner)
        cell.converged = converged
    # Broad: any learner failure is a recorded outcome for this cell, not a
    # reason to abandon the rest of the sweep.
    except Exception as exc:  # pylint: disable=broad-exception-caught
        cell.error_type = type(exc).__name__
        cell.error = f"{type(exc).__name__}: {exc}"
        dfa = None
    cell.seconds = time.time() - t0
    # mq.cache is upstream's persistence dict, keyed by string, so its size is
    # the distinct count. Totals equal it: SameStateOracle._label memoises above
    # the MQ, so a repeat never reaches the oracle. See the Cell docstring.
    cell.queries_distinct = len(learner.mq.cache)
    cell.queries_total = cell.queries_distinct
    cell.equivalence_queries = eq_calls["n"]

    if dfa is not None:
        cell.learned_states = dfa.num_states
        cell.accuracy = accuracy(
            lambda w: dfa.run("".join(alphabet[i] for i in w)), truth, words
        )
    elif cell.error is None:
        cell.error_type = "NoHypothesis"
        cell.error = "no hypothesis produced"
    return cell.finalize()


def run_elstar_cell(
    oracle_creator: Callable[[Any, int], Any],
    *,
    benchmark: str,
    family: str,
    eta: float,
    seed: int,
    symbols: int,
    words: Sequence[List[int]],
    truth: Callable[[List[int]], bool],
    target_states: Optional[int] = None,
    min_suffix_frequency: float = 0.05,
) -> Cell:
    """Run this repo's E-L* on `oracle_creator` and score it identically.

    Word sampling is left at `learn_dfa`'s default length, which is also the
    length `Benchmark.regime_report` measures at.
    """
    from orthogonal_dfa.l_star.structures import Oracle

    signal = eta_to_signal_strength(eta)
    cell = Cell(
        benchmark=benchmark,
        family=family,
        learner=LEARNER_ELSTAR,
        eta=eta,
        seed=seed,
        target_states=target_states,
        alphabet_size=symbols,
        learner_config={
            "min_signal_strength": signal,
            "min_suffix_frequency": min_suffix_frequency,
        },
        # E-L* has no equivalence oracle: its counterexamples are built out of
        # membership queries, which queries_distinct already charges it for.
        equivalence_queries=0,
    )

    class CountingOracle(Oracle):
        """Counts total and distinct membership queries."""

        def __init__(self, inner: Any) -> None:
            self._inner = inner
            self.count = 0
            self.distinct: set = set()

        @property
        def alphabet_size(self) -> int:
            return self._inner.alphabet_size

        def membership_query(self, string: List[int]) -> bool:
            self.count += 1
            self.distinct.add(tuple(string))
            return self._inner.membership_query(string)

    counters: List[CountingOracle] = []

    def counting_creator(noise_model: Any, s: int) -> CountingOracle:
        o = CountingOracle(oracle_creator(noise_model, s))
        counters.append(o)
        return o

    t0 = time.time()
    dfa = None
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
            io.StringIO()
        ):
            dfa = learn_dfa(
                counting_creator,
                min_signal_strength=signal,
                seed=seed,
                min_suffix_frequency=min_suffix_frequency,
            )
    # Broad, for the same reason as the CAPAL driver above.
    except Exception as exc:  # pylint: disable=broad-exception-caught
        cell.error_type = type(exc).__name__
        cell.error = f"{type(exc).__name__}: {exc}"
    cell.seconds = time.time() - t0
    cell.queries_total = sum(c.count for c in counters)
    cell.queries_distinct = (
        len(set().union(*[c.distinct for c in counters])) if counters else 0
    )

    if dfa is not None:
        cell.learned_states = len(dfa.states)
        cell.accuracy = accuracy(dfa.accepts_input, truth, words)
        # E-L* has no convergence flag: it always returns a hypothesis. Treat
        # an exact-accuracy hypothesis as converged so the column is comparable.
        cell.converged = cell.accuracy == 1.0
    return cell.finalize()


# -- emit ---------------------------------------------------------------------


def write_experiment(
    path: Path,
    *,
    experiment: str,
    generated_by: str,
    description: str,
    config: Dict[str, Any],
    cells: Sequence[Cell],
    complete: bool,
) -> Path:
    """Write one experiment's JSON: the config that produced it and every
    cell, so the report generator needs nothing but this file.

    Sweeps rewrite this file after every cell so a crash costs one cell rather
    than the run, which means a partial file is as well-formed as a finished
    one. `complete` is what tells them apart: anything reading these numbers
    must refuse a file where it is false, or it will report a truncated sweep
    as the whole of one.
    """
    payload = {
        "schema_version": SCHEMA_VERSION,
        "experiment": experiment,
        "description": description,
        "complete": complete,
        "generated_by": generated_by,
        "config": config,
        "cells": [asdict(c) for c in cells],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")
    return path
