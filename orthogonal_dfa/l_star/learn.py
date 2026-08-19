"""Run a learner against an oracle: the configuration entry point.

`synthesize_direct_lstar_fnr` in `counterexample_synthesis` learns from a populated
PrefixSuffixTracker. Getting to one means sizing the suffix population, the
evidence margin and the prefix count from the noise level, which is what this
module does -- so a caller only has to say how much signal the oracle carries.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

import numpy as np

from .counterexample_synthesis import synthesize_direct_lstar_fnr
from .prefix_suffix_tracker import PrefixSuffixTracker, SearchConfig
from .sampler import UniformSampler
from .statistics import (
    compute_suffix_size_counterexample_gen,
    population_size_and_evidence_margin,
)
from .structures import SymmetricBernoulli

#: All of E-L*'s signal comes from words drawn at this length.
DEFAULT_SAMPLE_LENGTH = 40

#: Accuracy the synthesis loop drives the hypothesis to before stopping.
DEFAULT_ACC_THRESHOLD = 0.98

#: Prefixes to start from.
NUM_PREFIXES = 200


def build_pst(
    oracle_creator: Callable[[Any, int], Any],
    *,
    min_signal_strength: float,
    seed: int,
    noise_model: Optional[Any] = None,
    min_suffix_frequency: float = 0.02,
    sample_length: int = DEFAULT_SAMPLE_LENGTH,
    fnr_limit: float = 0.02,
) -> PrefixSuffixTracker:
    """A PrefixSuffixTracker sized for an oracle carrying `min_signal_strength`.

    The oracle answers correctly with probability `0.5 + min_signal_strength`,
    and every population size here is derived from that: too small and the
    row statistics cannot separate states, too large and every query is spent
    on evidence nobody needs.
    """
    effective_p_acc = 0.5 + min_signal_strength
    if noise_model is None:
        noise_model = SymmetricBernoulli(p_correct=effective_p_acc)
    oracle = oracle_creator(noise_model, seed)
    n, eps = population_size_and_evidence_margin(
        signal_strength=min_signal_strength, acceptable_fpr=0.01, acceptable_fnr=0.01
    )
    config = SearchConfig(
        suffix_family_size=n,
        evidence_margin=eps,
        suffix_size_counterexample_gen=compute_suffix_size_counterexample_gen(
            0.01, effective_p_acc
        ),
        min_signal_strength=min_signal_strength,
        num_addtl_prefixes=NUM_PREFIXES,
        min_suffix_frequency=min_suffix_frequency,
        fnr_limit=fnr_limit,
    )
    return PrefixSuffixTracker.create(
        UniformSampler(sample_length),
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
    sample_length: int = DEFAULT_SAMPLE_LENGTH,
    acc_threshold: float = DEFAULT_ACC_THRESHOLD,
    fnr_limit: float = 0.02,
):
    """Learn a DFA from `oracle_creator`, and return it.

    `oracle_creator(noise_model, seed)` builds the oracle to query; it is a
    factory rather than an oracle so callers can count or wrap the queries.
    Returns None when synthesis produced no hypothesis.
    """
    pst = build_pst(
        oracle_creator,
        min_signal_strength=min_signal_strength,
        seed=seed,
        noise_model=noise_model,
        min_suffix_frequency=min_suffix_frequency,
        sample_length=sample_length,
        fnr_limit=fnr_limit,
    )
    dfa, _ = synthesize_direct_lstar_fnr(pst, acc_threshold=acc_threshold)
    return dfa
