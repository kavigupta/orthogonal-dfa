"""Learn a DFA over the superlanguage of a vocabulary.

This ties the three pieces together: a :class:`SuperSampler` draws probe strings
over the super alphabet, a :class:`LiftedOracle` answers membership by compiling
them back to the base alphabet and querying ``base_oracle``, and E-L*'s
``learn_dfa`` does the rest.  The returned DFA's transitions are labelled with
super-symbols; the vocabulary is returned alongside so they can be read back as
kmers / ``X``.

The lifted oracle is already stochastic (compilation realizes each ``X``
randomly, so different strings reaching the same state disagree), and that
variation *is* the noise E-L* denoises over.  So no extra symmetric noise is
injected by default -- ``noise_model`` defaults to the identity and
``min_signal_strength`` is the independent knob that sizes the search for the
lifted oracle's effective signal.  Raise ``num_compilations`` to sharpen that
signal (majority-voting more compilations per query) at proportional query cost.
"""

# The learn_dfa call below forwards the same configuration kwargs build_pst
# already names, which pylint flags as duplicate-code; the overlap is inherent
# to forwarding, not copied logic.
# pylint: disable=duplicate-code

from typing import Any, Iterable, Optional, Sequence, Tuple

from orthogonal_dfa.l_star.learn import DEFAULT_ACC_THRESHOLD, DEFAULT_SAMPLE_LENGTH
from orthogonal_dfa.l_star.learn import learn_dfa
from orthogonal_dfa.l_star.structures import NoiseModel, Oracle, SymmetricBernoulli

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
    num_compilations: int = 8,
    noise_model: Optional[NoiseModel] = None,
    min_suffix_frequency: float = 0.02,
    acc_threshold: float = DEFAULT_ACC_THRESHOLD,
) -> Tuple[Any, KmerVocabulary]:
    """Learn a DFA over ``vocabulary``'s superlanguage against ``base_oracle``.

    ``num_symbols`` is the number of super-symbols per sampled string;
    ``num_compilations`` is how many base compilations are majority-voted per
    membership query.  Returns ``(dfa, vocabulary)`` (``dfa`` is ``None`` when
    synthesis produced no hypothesis).
    """
    # Identity noise: the lifted oracle carries its own compilation noise, so
    # the framework's symmetric-noise default would double-count it.
    if noise_model is None:
        noise_model = SymmetricBernoulli(p_correct=1.0)

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
