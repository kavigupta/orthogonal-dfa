# pylint: disable=duplicate-code  # forwarding build_pst's kwargs, not copied logic

from typing import Any, List, Optional, Tuple

from orthogonal_dfa.l_star.counterexample_synthesis import RoundClassifier
from orthogonal_dfa.l_star.learn import (
    DEFAULT_ACC_THRESHOLD,
    DEFAULT_SAMPLE_LENGTH,
    learn_dfa,
)
from orthogonal_dfa.l_star.structures import NoiseModel, NoisyOracle, Oracle

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
    noise_model: Optional[NoiseModel] = None,
    min_suffix_frequency: float = 0.02,
    acc_threshold: float = DEFAULT_ACC_THRESHOLD,
) -> Tuple[Any, List[RoundClassifier]]:
    """Where base_oracle is deterministic and blind to how the wildcards were
    filled, so is the lifted one, and noise reaches the learner through noise_model
    alone. dfa is None if synthesis reached no hypothesis.
    """

    def oracle_creator(nm, s):
        return NoisyOracle(LiftedOracle(base_oracle, vocabulary, seed=s), nm, s)

    dfa, classifiers = learn_dfa(
        oracle_creator,
        min_signal_strength=min_signal_strength,
        seed=seed,
        noise_model=noise_model,
        min_suffix_frequency=min_suffix_frequency,
        sampler=SuperSampler(vocabulary, num_symbols),
        acc_threshold=acc_threshold,
    )
    return dfa, classifiers
