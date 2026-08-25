# pylint: disable=duplicate-code  # forwarding build_pst's kwargs, not copied logic

from typing import Any, Optional, Tuple

from orthogonal_dfa.l_star.learn import (
    DEFAULT_ACC_THRESHOLD,
    DEFAULT_SAMPLE_LENGTH,
    learn_dfa,
)
from orthogonal_dfa.l_star.structures import NoiseModel, Oracle

from .oracle import LiftedOracle
from .sampler import SuperSampler
from .vocabulary import KmerVocabulary


def learn_superlanguage(
    base_oracle: Oracle,
    vocabulary: KmerVocabulary,
    *,
    min_signal_strength: float,
    seed: int,
    num_symbols: int = DEFAULT_SAMPLE_LENGTH,
    num_compilations: int = 1,
    noise_model: Optional[NoiseModel] = None,
    min_suffix_frequency: float = 0.02,
    acc_threshold: float = DEFAULT_ACC_THRESHOLD,
) -> Tuple[Any, KmerVocabulary]:
    """Noise reaches the learner only through noise_model, compile being invertible
    and the lifted oracle therefore deterministic. Raising num_compilations pays
    only when the base oracle can see the wildcard fill. dfa is None if synthesis
    reached no hypothesis.
    """

    def oracle_creator(nm, s):
        return LiftedOracle(
            base_oracle,
            vocabulary,
            num_compilations=num_compilations,
            seed=s,
            noise_model=nm,
        )

    dfa, _ = learn_dfa(
        oracle_creator,
        min_signal_strength=min_signal_strength,
        seed=seed,
        noise_model=noise_model,
        min_suffix_frequency=min_suffix_frequency,
        sampler=SuperSampler(vocabulary, num_symbols),
        acc_threshold=acc_threshold,
    )
    return dfa, vocabulary
