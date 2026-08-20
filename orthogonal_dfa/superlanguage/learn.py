"""Learn a DFA over the superlanguage of a vocabulary.

A :class:`SuperSampler` draws probe strings over the super alphabet, a
:class:`LiftedOracle` answers membership against ``base_oracle``, and ``learn_dfa``
does the rest.  Returns ``(dfa, vocabulary)`` so the super-symbol transitions can
be read back as kmers / ``X``.

Compile is invertible, so the lifted oracle is a deterministic function of the
super-string; the framework's symmetric noise (``SymmetricBernoulli(0.5 +
min_signal_strength)``) is injected as usual so E-L*'s screening has signal to
size itself against, exactly as for a base-alphabet oracle.
"""

# pylint: disable=duplicate-code  # forwarding build_pst's kwargs, not copied logic

from typing import Any, Iterable, Optional, Sequence, Tuple

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
    """Learn a DFA over ``vocabulary``'s superlanguage against ``base_oracle``.

    ``num_symbols`` is the number of super-symbols per sampled string;
    ``num_compilations`` is how many base compilations are majority-voted per
    membership query; one is enough whenever the base oracle only reads features
    the kmers carry (compilation cannot forge them), which is the usual case.
    ``noise_model`` defaults to the framework's symmetric
    noise (see :func:`~orthogonal_dfa.l_star.learn.build_pst`).  Returns
    ``(dfa, vocabulary)`` (``dfa`` is ``None`` when synthesis produced no
    hypothesis).
    """

    def oracle_creator(nm, s):
        return LiftedOracle(
            base_oracle,
            vocabulary,
            num_compilations=num_compilations,
            seed=s,
            noise_model=nm,
        )

    dfa = learn_dfa(
        oracle_creator,
        min_signal_strength=min_signal_strength,
        seed=seed,
        noise_model=noise_model,
        min_suffix_frequency=min_suffix_frequency,
        sampler=SuperSampler(vocabulary, num_symbols),
        acc_threshold=acc_threshold,
    )
    return dfa, vocabulary


def learn_superlanguage_from_corpus(
    corpus: Iterable[Sequence[int]],
    base_oracle: Oracle,
    *,
    min_signal_strength: float,
    seed: int,
    lengths: Sequence[int] = (3, 4, 5, 6),
    top_n: int = 10,
    **kwargs,
) -> Tuple[Any, KmerVocabulary]:
    """Build a :class:`KmerVocabulary` from ``corpus`` (top-``top_n`` kmers over
    ``lengths``) and learn its superlanguage.  ``**kwargs`` are forwarded to
    :func:`learn_superlanguage`."""
    vocabulary = KmerVocabulary.from_corpus(
        corpus,
        base_alphabet_size=base_oracle.alphabet_size,
        lengths=lengths,
        top_n=top_n,
    )
    return learn_superlanguage(
        base_oracle,
        vocabulary,
        min_signal_strength=min_signal_strength,
        seed=seed,
        **kwargs,
    )
