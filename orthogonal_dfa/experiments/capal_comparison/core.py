"""Shared measurement machinery for the CAPAL vs E-L* experiments.

The point of this module is that **both learners are measured identically**:
same evaluation word list, same accuracy definition, same query accounting,
same result record. Anything that differs between the two learners is a
property of the learner, not of the harness.

Noise. Both sides model *persistent* noise -- a given string's label is fixed
the first time it is asked, so repeated queries are free and `distinct` queries
are the honest oracle cost. CAPAL parameterises it as `eta` (flip probability);
this repo parameterises it as `p_correct = 1 - eta`, with E-L* additionally
told `min_signal_strength = 0.5 - eta` so it can size its suffix population.
Both learners therefore know the true noise rate.
"""

from __future__ import annotations

import contextlib
import io
import json
import platform
import random
import subprocess
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

from orthogonal_dfa.capal_official import PINNED_COMMIT, fit_with_fallback, make_learner

#: Bump when the emitted record shape changes incompatibly.
SCHEMA_VERSION = 1

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


# -- provenance ---------------------------------------------------------------


def _git(*args: str) -> Optional[str]:
    try:
        out = subprocess.run(
            ["git", "-C", str(REPO_ROOT), *args],
            capture_output=True,
            text=True,
            check=True,
        )
    except Exception:  # noqa: BLE001 -- provenance is best-effort
        return None
    return out.stdout.strip()


def provenance() -> Dict[str, Any]:
    """Everything needed to know what produced a JSON, and whether to trust it.

    `repo_dirty` matters: a dirty tree means the numbers cannot be tied to a
    commit, which is the same reproducibility standard we hold upstream to.
    """
    return {
        "repo_commit": _git("rev-parse", "HEAD"),
        "repo_dirty": bool(_git("status", "--porcelain")),
        "capal_commit": PINNED_COMMIT,
        "python": platform.python_version(),
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


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
    """One (benchmark, learner, eta, seed) measurement.

    `queries_distinct` is the comparable oracle cost under persistent noise;
    `queries_total` is kept so the report can show cache effectiveness.
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
    seconds: Optional[float] = None
    #: Exception class name, so the report can group failure modes without
    #: parsing prose. E.g. E-L* raises GaveUpOnSuffixSearch when no suffix
    #: family clears the signal threshold -- a learner outcome, not a crash.
    error_type: Optional[str] = None
    error: Optional[str] = None

    def finalize(self) -> "Cell":
        """Round the float fields.

        Full precision on timings would make every re-run a noisy diff on a
        checked-in file, for digits nobody reads. Called by the drivers once a
        cell is fully populated.
        """
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
    max_iters: int = 200,
    k_pos: int = 10,
    k_neg: int = 10,
    max_same_samples: int = 60,
    suffix_pool_len_max: int = 8,
    alpha: float = 1e-3,
    tau_cap: float = 0.2,
    suffix_pool_init: int = 32,
    enum_depth: int = 3,
    extra_len_max: int = 8,
) -> Cell:
    """Run upstream CAPAL on `target` and score it on the shared word list.

    ``enum_depth`` / ``extra_len_max`` control how many and how long the SAMESTATE
    suffixes are; raising them is the section-10 "matched query budget" probe.
    """
    cell = Cell(
        benchmark=benchmark,
        family=family,
        learner=LEARNER_CAPAL,
        eta=eta,
        seed=seed,
        target_states=target.num_states,
        alphabet_size=len(target.alphabet),
        learner_config={
            "K_pos": k_pos,
            "K_neg": k_neg,
            "max_iters": max_iters,
            "max_same_samples": max_same_samples,
            "suffix_pool_len_max": suffix_pool_len_max,
            "alpha": alpha,
            "tau_cap": tau_cap,
            "suffix_pool_init": suffix_pool_init,
            "enum_depth": enum_depth,
            "extra_len_max": extra_len_max,
        },
    )

    learner = make_learner(
        target,
        eta,
        max_iters=max_iters,
        seed=seed,
        verbose=False,
        k_pos=k_pos,
        k_neg=k_neg,
        max_same_samples=max_same_samples,
        tau_cap=tau_cap,
        suffix_pool_init=suffix_pool_init,
        suffix_pool_len_max=suffix_pool_len_max,
        alpha=alpha,
        enum_depth=enum_depth,
        extra_len_max=extra_len_max,
    )
    # Upstream's PersistentNoisyMQ caches but does not count; wrap it so we can
    # report total alongside distinct.
    totals = {"n": 0}
    inner_query = learner.mq.query

    def counting_query(s: str) -> bool:
        totals["n"] += 1
        return inner_query(s)

    learner.mq.query = counting_query  # type: ignore[method-assign]

    t0 = time.time()
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            dfa, converged = fit_with_fallback(learner)
        cell.converged = converged
    except Exception as exc:  # noqa: BLE001
        cell.error_type = type(exc).__name__
        cell.error = f"{type(exc).__name__}: {exc}"
        dfa = None
    cell.seconds = time.time() - t0
    cell.queries_total = totals["n"]
    cell.queries_distinct = len(getattr(learner.mq, "cache", {}))

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
    additional_counterexamples: int = 200,
    sample_length: int = 40,
    accept_rate_at_sample_length: Optional[float] = None,
) -> Cell:
    """Run this repo's E-L* on `oracle_creator` and score it identically.

    `sample_length` is E-L*'s word-sampling length -- a real hyperparameter,
    since every bit of its signal comes from words drawn at that length. See
    `Benchmark.tune_sample_length`.
    """
    from orthogonal_dfa.l_star.sampler import UniformSampler
    from orthogonal_dfa.l_star.structures import Oracle

    # Imported lazily: this pulls in the test harness, which is also where the
    # canonical synthesis entry point lives.
    from tests.test_lstar import compute_dfa_for_oracle

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
            "additional_counterexamples": additional_counterexamples,
            "sample_length": sample_length,
            "accept_rate_at_sample_length": accept_rate_at_sample_length,
        },
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
            _, dfa, _ = compute_dfa_for_oracle(
                counting_creator,
                min_signal_strength=signal,
                seed=seed,
                min_suffix_frequency=min_suffix_frequency,
                sampler=UniformSampler(sample_length),
            )
    except Exception as exc:  # noqa: BLE001
        cell.error_type = type(exc).__name__
        cell.error = f"{type(exc).__name__}: {exc}"
    cell.seconds = time.time() - t0
    cell.queries_total = sum(c.count for c in counters)
    cell.queries_distinct = (
        len(set().union(*[c.distinct for c in counters])) if counters else 0
    )

    if dfa is not None:
        cell.learned_states = len(dfa.states)
        cell.accuracy = accuracy(lambda w: dfa.accepts_input(w), truth, words)
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
) -> Path:
    """Write one experiment's JSON. Self-contained by design: provenance, the
    config that produced it, and every cell -- so the report generator needs
    nothing but this file."""
    payload = {
        "schema_version": SCHEMA_VERSION,
        "experiment": experiment,
        "description": description,
        "generated_by": generated_by,
        "provenance": provenance(),
        "config": config,
        "cells": [asdict(c) for c in cells],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(payload, f, indent=2, sort_keys=False)
        f.write("\n")
    return path


def _round(value: Optional[float], places: int) -> Optional[float]:
    return None if value is None else round(value, places)
