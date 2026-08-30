import itertools
from dataclasses import dataclass
from typing import List

import numpy as np

from orthogonal_dfa.l_star.sampler import Sampler

from .vocabulary import KmerVocabulary


def _stationary_kmer_rates(vocabulary: KmerVocabulary) -> np.ndarray:
    """How often parse emits each kmer, once it has read enough to settle.

    Between emissions the parse carries the base symbols it has looked ahead at
    and not consumed -- ``longest - 1`` of them -- and each step reads one more,
    emits whichever kmer starts there or a wildcard, and advances by what it
    emitted.  A uniform base stream makes that window a Markov chain, and the
    rates are the emissions its stationary distribution produces.
    """
    base = vocabulary.base_alphabet_size
    longest = max((len(k) for k in vocabulary.kmers), default=1)
    windows = list(itertools.product(range(base), repeat=longest - 1))
    index = {w: i for i, w in enumerate(windows)}
    transfer = np.zeros((len(windows), len(windows)))
    emitted = np.zeros((len(windows), len(vocabulary.kmers)))
    for window in windows:
        for symbol in range(base):
            read = window + (symbol,)
            kmer = next(
                (i for i, k in enumerate(vocabulary.kmers) if read[: len(k)] == k),
                None,
            )
            if kmer is None:
                # A wildcard consumes one symbol, leaving the rest of the window.
                transfer[index[window], index[read[1:]]] += 1 / base
                continue
            emitted[index[window], kmer] += 1 / base
            # A kmer consumes its own length, so the window refills by that much
            # less one, each refill equally likely.
            kept = read[len(vocabulary.kmers[kmer]) :]
            refill = longest - 1 - len(kept)
            for tail in itertools.product(range(base), repeat=refill):
                transfer[index[window], index[kept + tail]] += 1 / base / base**refill
    values, vectors = np.linalg.eig(transfer.T)
    stationary = np.real(vectors[:, np.argmin(np.abs(values - 1))])
    return (stationary / stationary.sum()) @ emitted


@dataclass(frozen=True)
class SuperSampler(Sampler):
    """Draws by parsing a uniform base stream, which is what leaves the compiled
    strings uniform; drawing super-symbols independently would not.
    """

    vocabulary: KmerVocabulary
    length: int

    def symbol_weights(self, alphabet_size: int) -> List[float]:
        """How often parse emits each symbol, once it has read enough to settle.

        Not B**-L for a kmer of length L over B base symbols: that is the rate at
        the first emission alone, which is the only one reading a stream nothing
        is known about.  Every later one follows a parse decision -- after a
        wildcard, the knowledge that no kmer started where it did.
        """
        vocab = self.vocabulary
        assert alphabet_size == vocab.alphabet_size, (
            f"alphabet size mismatch: the vocabulary has {vocab.alphabet_size} "
            f"super-symbols but the learner asked for {alphabet_size}"
        )
        kmers = list(_stationary_kmer_rates(vocab))
        return kmers + [(1 - sum(kmers)) / vocab.num_wildcards] * vocab.num_wildcards

    def sample(self, rng: np.random.Generator, alphabet_size: int) -> List[int]:
        assert alphabet_size == self.vocabulary.alphabet_size, (
            f"alphabet size mismatch: the vocabulary has "
            f"{self.vocabulary.alphabet_size} super-symbols but the learner asked "
            f"for {alphabet_size}"
        )
        vocab = self.vocabulary
        # A super-symbol eats at most one kmer's worth of base symbols, so this
        # many always parses to at least `length` of them; the tail is discarded.
        longest = max((len(k) for k in vocab.kmers), default=1)
        stream = rng.integers(vocab.base_alphabet_size, size=self.length * longest)
        parsed = vocab.parse(stream.tolist())[: self.length]
        # parse only ever emits the canonical wildcard. Spreading the neutral
        # positions over all of them is what makes such strings plentiful.
        wildcards = rng.integers(vocab.num_wildcards, size=self.length)
        return [
            vocab.num_kmers + int(w) if vocab.is_unknown(symbol) else symbol
            for symbol, w in zip(parsed, wildcards)
        ]
