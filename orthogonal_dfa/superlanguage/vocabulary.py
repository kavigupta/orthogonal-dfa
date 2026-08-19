"""The superlanguage vocabulary: a small alphabet of frequent kmers plus a
wildcard, with a compilation back to the base alphabet.

The *base* alphabet is the integers ``0 .. base_alphabet_size - 1`` (genomic
ACGT is size 4).  A :class:`KmerVocabulary` picks ``K`` kmers -- each a tuple of
base symbols, possibly of *different* lengths -- and adds one wildcard symbol
``X``.  The resulting *super* alphabet has ``K + 1`` symbols: super-symbol ``i``
for ``i < K`` is ``kmers[i]``, and super-symbol ``K`` (``unknown_symbol``) is
``X``.

Compilation from the super alphabet back to the base alphabet is
*nondeterministic*: a kmer symbol emits exactly its kmer, while ``X`` emits a
single base symbol drawn uniformly.  This is the "X -> uniform distribution over
the original language" rule; the sampler and the lifted oracle are both built on
top of it.

The emission distribution ``P`` is the **uniform null model** with
shortest-match-wins tie-breaking: under a uniform random base string,
``P(kmer)`` is the probability the string *starts with* that kmer and with no
strictly shorter family kmer, i.e. ``(1 / base) ** len(kmer)`` unless a strictly
shorter family kmer is a prefix of it (in which case that shorter kmer claims the
mass and this one gets 0).  ``P(X)`` is whatever mass is left over.  Because the
minimal (non-preempted) kmers form a prefix-free set, Kraft's inequality keeps
their probabilities summing to at most 1, so ``P(X) >= 0``.
"""

from collections import Counter
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import numpy as np

Kmer = Tuple[int, ...]


def _prefix_related(a: Sequence[int], b: Sequence[int]) -> bool:
    """True when the shorter of ``a``/``b`` is a prefix of the longer (so keeping
    both would break the prefix-free property the null model relies on)."""
    m = min(len(a), len(b))
    return tuple(a[:m]) == tuple(b[:m])


@dataclass(frozen=True)
class KmerVocabulary:
    """A frozen kmer vocabulary over a base alphabet.

    ``kmers`` are ordered (the order fixes the super-symbol indices) and each is a
    tuple of base symbols.  Construct one from a corpus with
    :meth:`from_corpus`, or directly when the kmers are already known.
    """

    kmers: Tuple[Kmer, ...]
    base_alphabet_size: int

    def __post_init__(self):
        assert self.base_alphabet_size >= 1, "base alphabet must be non-empty"
        for kmer in self.kmers:
            assert len(kmer) >= 1, "kmers must be non-empty"
            assert all(
                0 <= c < self.base_alphabet_size for c in kmer
            ), f"kmer {kmer} has symbols outside the base alphabet"
        assert len(set(self.kmers)) == len(self.kmers), "duplicate kmers"

    # -- shape ---------------------------------------------------------------

    @property
    def num_kmers(self) -> int:
        return len(self.kmers)

    @property
    def unknown_symbol(self) -> int:
        """The index of ``X`` -- the last super-symbol."""
        return self.num_kmers

    @property
    def alphabet_size(self) -> int:
        """Size of the *super* alphabet: one symbol per kmer, plus ``X``."""
        return self.num_kmers + 1

    def is_unknown(self, symbol: int) -> bool:
        return symbol == self.unknown_symbol

    # -- distribution --------------------------------------------------------

    def probabilities(self) -> np.ndarray:
        """The emission distribution over the ``K + 1`` super-symbols.

        Uniform null model with shortest-match-wins: a kmer preempted by a
        strictly shorter family prefix gets 0, everything else gets
        ``base ** -len``, and ``X`` collects the remainder.
        """
        base = self.base_alphabet_size
        probs = np.zeros(self.alphabet_size)
        for i, kmer in enumerate(self.kmers):
            preempted = any(
                len(other) < len(kmer) and tuple(kmer[: len(other)]) == other
                for other in self.kmers
            )
            probs[i] = 0.0 if preempted else base ** (-len(kmer))
        # 1 - sum keeps the vector normalized exactly; the max guards the (only
        # non-antichain) case where floating-point rounding pushes the sum just
        # over 1.
        probs[self.unknown_symbol] = max(0.0, 1.0 - probs[: self.num_kmers].sum())
        return probs

    # -- compilation ---------------------------------------------------------

    def compiled_length(self, symbol: int) -> int:
        """How many base symbols ``symbol`` compiles to (``X`` -> 1)."""
        return 1 if self.is_unknown(symbol) else len(self.kmers[symbol])

    def compile_symbol(self, symbol: int, rng: np.random.Generator) -> List[int]:
        """Compile one super-symbol to base symbols: ``X`` draws a uniform base
        symbol, a kmer symbol returns its (deterministic) kmer."""
        if self.is_unknown(symbol):
            return [int(rng.integers(self.base_alphabet_size))]
        return list(self.kmers[symbol])

    def compile(
        self, super_string: Iterable[int], rng: np.random.Generator
    ) -> List[int]:
        """Compile a whole super-string to a base string by concatenating the
        per-symbol compilations (each ``X`` realized independently)."""
        out: List[int] = []
        for symbol in super_string:
            out.extend(self.compile_symbol(symbol, rng))
        return out

    # -- construction --------------------------------------------------------

    @classmethod
    def from_corpus(
        cls,
        corpus: Iterable[Sequence[int]],
        base_alphabet_size: int,
        *,
        lengths: Sequence[int] = (3, 4, 5, 6),
        top_n: int = 10,
        prune_non_minimal: bool = True,
    ) -> "KmerVocabulary":
        """Pick the ``top_n`` most frequent kmers over ``corpus``.

        Every contiguous substring whose length is in ``lengths`` is counted
        across the corpus and the kmers are ranked by count (ties broken by the
        kmer itself, for determinism).  With ``prune_non_minimal`` (the default),
        a candidate is skipped when it is prefix-related to an already-selected
        kmer, keeping the vocabulary prefix-free so every kept kmer has non-zero
        emission probability; the higher-count member of any prefix conflict
        wins.  Set it to ``False`` to keep the raw top-``n`` (preempted kmers then
        appear with probability 0).
        """
        counts: Counter = Counter()
        for string in corpus:
            symbols = list(string)
            for k in lengths:
                for i in range(len(symbols) - k + 1):
                    counts[tuple(symbols[i : i + k])] += 1
        ranked = sorted(counts, key=lambda km: (-counts[km], km))
        selected: List[Kmer] = []
        for kmer in ranked:
            if len(selected) >= top_n:
                break
            if prune_non_minimal and any(_prefix_related(kmer, o) for o in selected):
                continue
            selected.append(kmer)
        return cls(kmers=tuple(selected), base_alphabet_size=base_alphabet_size)
