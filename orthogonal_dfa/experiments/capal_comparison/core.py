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

Equivalence queries are recorded alongside membership queries, because the two
learners do not have the same oracles: CAPAL is given a perfect EQ (the paper's
pMAT assumption), and its counterexamples come back as gold labels that also
shadow the MQ, so those strings cost it nothing thereafter. E-L* has no EQ at
all and manufactures its counterexamples out of membership queries, so its EQ
count is 0 by construction.
"""

from __future__ import annotations

import contextlib
import inspect
import io
import json
import random
import signal
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

from orthogonal_dfa.capal_official import fit_with_fallback, make_learner
from orthogonal_dfa.l_star.learn import learn_dfa

#: Bump when the emitted record shape changes incompatibly.
SCHEMA_VERSION = 5

#: Wall-clock a single cell may take. Neither learner has an internal bound on
#: how long it searches, so without this one target stalls the whole sweep.
#: A backstop against that, not a budget: Simple05 at eta=0.30 was still
#: searching at 53.7M queries when a 1800s cap cut it, with no sign of being
#: stuck, so the cap has to sit well clear of the slowest cell that converges.
CELL_TIMEOUT_SECONDS = 7200

#: Read off `learn_dfa` rather than restated here, for the same reason the CAPAL
#: side reads its config off the learner: E-L* must be measured at its own
#: default, and `min_suffix_frequency` has to stay in step with the
#: `min_class_preserving_frac` that decides which targets are in regime.
DEFAULT_MIN_SUFFIX_FREQUENCY = (
    inspect.signature(learn_dfa).parameters["min_suffix_frequency"].default
)

LEARNER_CAPAL = "CAPAL"
LEARNER_ELSTAR = "E-L*"

REPO_ROOT = Path(__file__).resolve().parents[3]

#: Shared evaluation settings, for the word list both learners are scored on.
EVAL_COUNT = 5000
EVAL_MAX_LEN = 40
EVAL_SEED = 0x1234


class BudgetExhausted(BaseException):
    """Raised when a cell has issued its allowance of distinct membership queries.

    BaseException for the same reason as `CellTimeout`: it is a stopping rule
    imposed from outside, not a learner failure, so the drivers' broad
    `except Exception` must not swallow it.
    """


class CellTimeout(BaseException):
    """Raised when a cell outruns `CELL_TIMEOUT_SECONDS`.

    Derives from BaseException so the broad `except Exception` each driver uses
    to record learner failures cannot swallow it.
    """


@contextlib.contextmanager
def time_limit(seconds: int):
    """Raise `CellTimeout` in the calling thread after `seconds`."""

    def _fire(*_):
        raise CellTimeout

    previous = signal.signal(signal.SIGALRM, _fire)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)


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

    `queries_total` is the membership cost: calls each implementation actually
    issues.

    `equivalence_queries` is the other half of the oracle cost -- see the module
    docstring on why the two have to be read together.

    `converged` is the one column that is *not* the same claim on both sides.
    CAPAL's comes from its PerfectEQ and means exact equality with the target.
    E-L* has no such signal, so it gets `accuracy == 1.0` on the sampled word
    list, which is an approximation to the same claim.
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
    equivalence_queries: Optional[int] = None
    seconds: Optional[float] = None
    #: Exception class name, so the report can group failure modes without
    #: parsing prose. "Timeout" means the cell hit `CELL_TIMEOUT_SECONDS`
    #: without producing a hypothesis.
    error_type: Optional[str] = None
    error: Optional[str] = None

    def finalize(self) -> "Cell":
        """Round the float fields. Called by the drivers once a cell is fully
        populated."""
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
    timeout: int = CELL_TIMEOUT_SECONDS,
    query_budget: Optional[int] = None,
    **learner_kwargs: Any,
) -> Cell:
    """Run upstream CAPAL on `target` and score it on the shared word list.

    learner_kwargs override make_learner's defaults, which are upstream's own
    benchmark settings.

    query_budget stops the fit after that many distinct membership queries and
    scores the hypothesis reached, so CAPAL can be compared against E-L* at
    equal spend. Stalled means it ran out of iterations under budget, so the
    budget never bound and the cell is not a matched-budget measurement.
    """
    learner = make_learner(target, eta, seed=seed, **learner_kwargs)
    cell = Cell(
        benchmark=benchmark,
        family=family,
        learner=LEARNER_CAPAL,
        eta=eta,
        seed=seed,
        target_states=target.num_states,
        alphabet_size=len(target.alphabet),
        # Read off the learner rather than restated here, so this records what
        # CAPAL ran with even when `make_learner`'s defaults move.
        learner_config={
            "max_iters": learner.cfg.max_iters,
            "max_same_samples": learner.ss.cfg.max_samples,
            "suffix_pool_len_max": learner.ss.cfg.pool_len_max,
            "alpha": learner.ss.cfg.alpha,
            "tau_cap": learner.ss.cfg.tau_cap,
            "suffix_pool_init": learner.ss.cfg.pool_init,
            "discr_search_max_len": learner.cfg.discr_search_max_len,
            "discr_search_random": learner.cfg.discr_search_random,
            "enum_depth": learner.ss.cfg.enum_depth,
            "extra_len_max": learner.ss.cfg.extra_len_max,
            "eta_hat": learner.ss.eta_hat,
        },
    )

    if query_budget is not None:
        inner_mq = learner.mq.query

        def budgeted_query(string: str) -> bool:
            answer = inner_mq(string)
            if len(learner.mq.cache) >= query_budget:
                raise BudgetExhausted()
            return answer

        learner.mq.query = budgeted_query  # type: ignore[method-assign]

    eq_calls = {"n": 0}
    inner_eq = learner.eq.query

    def counting_eq(hyp: Any) -> Any:
        eq_calls["n"] += 1
        return inner_eq(hyp)

    learner.eq.query = counting_eq  # type: ignore[method-assign]

    t0 = time.time()
    try:
        with time_limit(timeout), contextlib.redirect_stdout(io.StringIO()):
            dfa, converged = fit_with_fallback(learner)
        cell.converged = converged
    except BudgetExhausted:
        # The last hypothesis is what CAPAL would have answered had it been
        # stopped here, so it is the thing to score -- not a failure.
        last = getattr(learner, "_last_hyp", None)
        dfa = getattr(last, "dfa", None)
        cell.converged = False
        cell.error_type = "BudgetExhausted"
        cell.error = f"stopped at the {query_budget} distinct-query budget"
    except CellTimeout:
        cell.error_type = "Timeout"
        cell.error = f"no hypothesis within {timeout}s"
        dfa = None
    # Broad: any learner failure is a recorded outcome for this cell, not a
    # reason to abandon the rest of the sweep.
    except Exception as exc:  # pylint: disable=broad-exception-caught
        cell.error_type = type(exc).__name__
        cell.error = f"{type(exc).__name__}: {exc}"
        dfa = None
    cell.seconds = time.time() - t0
    # mq.cache is upstream's persistence dict, keyed by string, so its size is
    # every call that reached the oracle: SameStateOracle._label memoises above
    # the MQ, so repeats never get here.
    cell.queries_total = len(learner.mq.cache)
    cell.equivalence_queries = eq_calls["n"]

    if (
        query_budget is not None
        and cell.error_type is None
        and not cell.converged
        and cell.queries_total < query_budget
    ):
        cell.error_type = "Stalled"
        cell.error = (
            f"stopped after {cell.equivalence_queries} iterations having used "
            f"{cell.queries_total} of the {query_budget} query budget"
        )

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
    min_suffix_frequency: float = DEFAULT_MIN_SUFFIX_FREQUENCY,
    timeout: int = CELL_TIMEOUT_SECONDS,
) -> Cell:
    """Run this repo's E-L* on `oracle_creator` and score it identically."""
    from orthogonal_dfa.l_star.structures import Oracle

    signal_strength = eta_to_signal_strength(eta)
    cell = Cell(
        benchmark=benchmark,
        family=family,
        learner=LEARNER_ELSTAR,
        eta=eta,
        seed=seed,
        target_states=target_states,
        alphabet_size=symbols,
        learner_config={
            "min_signal_strength": signal_strength,
            "min_suffix_frequency": min_suffix_frequency,
        },
        equivalence_queries=0,
    )

    class CountingOracle(Oracle):
        def __init__(self, inner: Any) -> None:
            self._inner = inner
            self.count = 0

        @property
        def alphabet_size(self) -> int:
            return self._inner.alphabet_size

        def membership_query(self, string: List[int]) -> bool:
            self.count += 1
            return self._inner.membership_query(string)

    counters: List[CountingOracle] = []

    def counting_creator(noise_model: Any, s: int) -> CountingOracle:
        o = CountingOracle(oracle_creator(noise_model, s))
        counters.append(o)
        return o

    t0 = time.time()
    dfa = None
    try:
        with time_limit(timeout), contextlib.redirect_stdout(
            io.StringIO()
        ), contextlib.redirect_stderr(io.StringIO()):
            dfa = learn_dfa(
                counting_creator,
                min_signal_strength=signal_strength,
                seed=seed,
                min_suffix_frequency=min_suffix_frequency,
            )
    except CellTimeout:
        cell.error_type = "Timeout"
        cell.error = f"no hypothesis within {timeout}s"
    # Broad, for the same reason as the CAPAL driver above.
    except Exception as exc:  # pylint: disable=broad-exception-caught
        cell.error_type = type(exc).__name__
        cell.error = f"{type(exc).__name__}: {exc}"
    cell.seconds = time.time() - t0
    cell.queries_total = sum(c.count for c in counters)

    if dfa is not None:
        cell.learned_states = len(dfa.states)
        cell.accuracy = accuracy(dfa.accepts_input, truth, words)
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

    A partial file is as well-formed as a finished one; `complete` is what
    tells them apart. Written via a temporary file so an interrupted write
    cannot leave a half-written JSON behind.
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
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")
    tmp.replace(path)
    return path
