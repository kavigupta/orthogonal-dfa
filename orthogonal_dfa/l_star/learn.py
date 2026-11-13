"""Run E-L* against an oracle: the configuration entry point.

`counterexample_driven_synthesis` in `lstar` learns from a populated
PrefixSuffixTracker. Getting to one means sizing the suffix population, the
evidence margin and the prefix count from the noise level, which is what this
module does -- so a caller only has to say how much signal the oracle carries.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

import numpy as np

from .counterexample_synthesis import do_counterexample_driven_synthesis
from .prefix_suffix_tracker import PrefixSuffixTracker, SearchConfig
from .sampler import Sampler, UniformSampler
from .structures import SymmetricBernoulli
from .tracker import SynthesisTracker

#: All of E-L*'s signal comes from words drawn at this length.
DEFAULT_SAMPLE_LENGTH = 40

#: Accuracy the synthesis loop drives the hypothesis to before stopping.
DEFAULT_ACC_THRESHOLD = 0.98

#: Prefixes to start from.
NUM_PREFIXES = 200

#: Share of the prefixes a family decides that it may decide against the language.
DEFAULT_MAX_COVERAGE_ERROR = 1 / 3

#: The handling of bands not centred on a half (#319), each switchable to ablate it.
CENTRE_GRID = True
CENTRED_SCREEN = True
CENTRED_SUPPLY = True
#: Off: ranked row by row, the misread leaves most of a family's suffixes
#: outside the accept-preserving class at (0.60, 0.90) even when the pool holds
#: enough of them.
ORDER_BY_MISREAD = False

#: Lean's β: the least share of the uniform pool either class holds.
CLASS_FLOOR = 0.01


def build_pst(
    oracle_creator: Callable[[Any, int], Any],
    *,
    min_signal_strength: float,
    seed: int,
    noise_model: Optional[Any] = None,
    min_suffix_frequency: float = 0.02,
    sampler: Sampler = UniformSampler(DEFAULT_SAMPLE_LENGTH),
    require_accept_preserving: bool = True,
    cross_limit: float = 1.5e-7,
    acceptable_fnr: float = 0.01,
    max_coverage_error: float = DEFAULT_MAX_COVERAGE_ERROR,
) -> PrefixSuffixTracker:
    """A PrefixSuffixTracker for an oracle whose two classes read at least
    `2 * min_signal_strength` apart.

    `sampler` supplies the probe strings; pass one to learn over a different
    string distribution, or to vary the length.
    """
    if noise_model is None:
        noise_model = SymmetricBernoulli(p_correct=0.5 + min_signal_strength)
    oracle = oracle_creator(noise_model, seed)
    config = SearchConfig(
        min_signal_strength=min_signal_strength,
        centre_grid=CENTRE_GRID,
        centred_screen=CENTRED_SCREEN,
        order_by_misread=ORDER_BY_MISREAD,
        centred_supply=CENTRED_SUPPLY,
        class_floor=CLASS_FLOOR,
        num_addtl_prefixes=NUM_PREFIXES,
        min_suffix_frequency=min_suffix_frequency,
        require_accept_preserving=require_accept_preserving,
        cross_limit=cross_limit,
        acceptable_fnr=acceptable_fnr,
        max_coverage_error=max_coverage_error,
    )
    return PrefixSuffixTracker.create(
        sampler,
        np.random.default_rng(0),
        oracle,
        config,
        num_prefixes=NUM_PREFIXES,
    )


def learn_dfa(
    oracle_creator: Callable[[Any, int], Any],
    *,
    min_signal_strength: float,
    seed: int,
    noise_model: Optional[Any] = None,
    min_suffix_frequency: float = 0.02,
    sampler: Sampler = UniformSampler(DEFAULT_SAMPLE_LENGTH),
    acc_threshold: float = DEFAULT_ACC_THRESHOLD,
    require_accept_preserving: bool = True,
    cross_limit: float = 1.5e-7,
    acceptable_fnr: float = 0.01,
    max_coverage_error: float = DEFAULT_MAX_COVERAGE_ERROR,
    tracker: SynthesisTracker = SynthesisTracker(),
):
    """Learn a DFA from `oracle_creator`.  Failure raises
    `NoAcceptPreservingFamily` rather than returning.

    `oracle_creator(noise_model, seed)` builds the oracle to query; it is a
    factory rather than an oracle so callers can count or wrap the queries.
    `sampler` draws the probe strings (see `build_pst`).  Pass a `tracker` (see
    `SynthesisTracker`) to receive each round's family, hypothesis and
    consistency as they are made; the default one discards them.
    """
    pst = build_pst(
        oracle_creator,
        min_signal_strength=min_signal_strength,
        seed=seed,
        noise_model=noise_model,
        min_suffix_frequency=min_suffix_frequency,
        sampler=sampler,
        require_accept_preserving=require_accept_preserving,
        cross_limit=cross_limit,
        acceptable_fnr=acceptable_fnr,
        max_coverage_error=max_coverage_error,
    )
    dfa = do_counterexample_driven_synthesis(
        pst, acc_threshold=acc_threshold, tracker=tracker
    )
    return dfa
