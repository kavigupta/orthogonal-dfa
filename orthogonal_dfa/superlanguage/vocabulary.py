"""The superlanguage vocabulary: ``K`` prefix-free kmers plus interchangeable
wildcards (``X``, ``Y``, ...), with a compilation to the base alphabet
(``0 .. base-1``) that is invertible and distribution-preserving.

Super-symbol ``i < K`` is ``kmers[i]``; the rest are wildcards.  A base string is
read back with :meth:`parse` (greedy longest match: a kmer if one starts here,
else one wildcard).  :meth:`compile` is its inverse: kmers emit their symbols, and
each wildcard emits a single base symbol chosen so the greedy parse reproduces the
original super-string -- ``parse(compile(s)) == canonicalize(s)`` (the wildcards
compile identically, so parse cannot tell which one was used).

A wildcard may emit any base symbol that does not start a kmer given what follows
it, so :meth:`compile` works right to left (what follows is then already decided)
and draws from a precomputed per-state table.  The table holds the *Perron* weights
of the fiber-counting transfer matrix, which is the exact uniform law over
``parse**-1(s)`` in the interior of a wildcard run -- see
:func:`_transfer_tables`.  Only within a few positions of a kmer or the string end
does it deviate, geometrically.  So ``compile(parse(x))`` is uniform on the base
alphabet up to that boundary term, at the cost of a table lookup per symbol rather
than a per-string DP.  The kmers must be prefix-free so the parse is unambiguous.

Several wildcards exist for the learner's benefit: wildcard-only suffixes have the
same membership column as the empty suffix, so they are what the suffix-family
clustering locks onto -- and with a single wildcard there is only *one* such
suffix per length, too few to fill a family.
"""

import bisect
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable, List, Sequence, Tuple

import numpy as np

Kmer = Tuple[int, ...]


def _prefix_related(a: Sequence[int], b: Sequence[int]) -> bool:
    """True when the shorter of ``a``/``b`` is a prefix of the longer."""
    m = min(len(a), len(b))
    return tuple(a[:m]) == tuple(b[:m])


@lru_cache(maxsize=None)
def _transfer_tables(kmers: Tuple[Kmer, ...], base: int):
    """Compilation tables, built once per vocabulary.

    A state encodes the ``w = max_kmer_length - 1`` base symbols *following* a
    position, in radix ``base + 1`` so that ``base`` itself can act as an
    end-of-string sentinel (it matches no kmer).  ``shift[c][state]`` is the state
    seen one position to the left after emitting ``c``, and ``choice[state]`` is
    ``(symbols, cumulative_weights)`` for the symbols a wildcard may emit there.

    The weights come from the transfer matrix ``T`` of the counting recurrence
    ``ways[p][s] = sum over allowed c of ways[p-1][shift(c, s)]``: a run of ``L``
    wildcards applies ``T**L``, and since ``T`` is primitive, ``T**L`` grows like
    ``lambda_1**L w`` with ``w`` its right Perron eigenvector.  The ``lambda_1**L``
    cancels between candidates, so the exact fiber-uniform law in the interior of a
    run is just ``p(c | state) proportional to w[shift(c, state)]`` -- one table,
    no per-string DP.  (Weighting the candidates *equally* instead is the ``w ==
    const`` approximation, which visibly skews the base composition.)  Only within
    a few positions of a kmer block or the string end does the truth differ, and
    there by a factor decaying like ``(lambda_2 / lambda_1)`` per position.

    Plain lists, not arrays: compiling is a tight loop where numpy indexing costs
    more than it saves.
    """
    w = max((len(k) for k in kmers), default=1) - 1
    radix = base + 1
    num_states = radix**w

    shift = [
        [
            c + radix * (state % radix ** (w - 1)) if w >= 1 else 0
            for state in range(num_states)
        ]
        for c in range(base)
    ]

    allowed = []
    for state in range(num_states):
        following = [(state // radix**i) % radix for i in range(w)]
        starts = {k[0] for k in kmers if following[: len(k) - 1] == list(k[1:])}
        allowed.append([c for c in range(base) if c not in starts])

    transfer = np.zeros((num_states, num_states))
    for state in range(num_states):
        for c in allowed[state]:
            transfer[state, shift[c][state]] += 1.0
    values, vectors = np.linalg.eig(transfer)
    perron = np.abs(vectors[:, int(np.argmax(values.real))].real)

    choice = []
    for state in range(num_states):
        options = allowed[state]
        weights = np.array([perron[shift[c][state]] for c in options])
        total = weights.sum()
        # A degenerate vocabulary can leave a state with nothing legal to emit;
        # compile asserts on it rather than silently emitting a kmer.
        cumulative = (
            np.cumsum(weights / total).tolist() if options and total > 0 else []
        )
        choice.append((options, cumulative))

    initial = num_states - 1 if w else 0  # all sentinels: nothing follows the end
    return initial, choice, shift


@dataclass(frozen=True)
class KmerVocabulary:
    """A frozen, prefix-free kmer vocabulary over a base alphabet.

    ``kmers`` are ordered (the order fixes the super-symbol indices), each a tuple
    of base symbols, and no kmer may be a prefix of another.  Construct one from a
    corpus with :meth:`from_corpus`, or directly when the kmers are already known.
    """

    kmers: Tuple[Kmer, ...]
    base_alphabet_size: int
    num_wildcards: int = 2

    def __post_init__(self):
        assert self.base_alphabet_size >= 1, "base alphabet must be non-empty"
        assert self.num_wildcards >= 1, "need at least one wildcard"
        for kmer in self.kmers:
            assert len(kmer) >= 1, "kmers must be non-empty"
            assert all(
                0 <= c < self.base_alphabet_size for c in kmer
            ), f"kmer {kmer} has symbols outside the base alphabet"
        assert len(set(self.kmers)) == len(self.kmers), "duplicate kmers"
        for i, a in enumerate(self.kmers):
            for b in self.kmers[i + 1 :]:
                assert not _prefix_related(a, b), f"{a} and {b} are prefix-related"

    # -- shape ---------------------------------------------------------------

    @property
    def num_kmers(self) -> int:
        return len(self.kmers)

    @property
    def unknown_symbol(self) -> int:
        """The canonical wildcard (``X``) -- the first of the wildcard symbols."""
        return self.num_kmers

    @property
    def wildcard_symbols(self) -> Tuple[int, ...]:
        """All wildcard indices.  They are interchangeable: each compiles the same
        way, so the base oracle cannot tell them apart.  Several of them exist so
        the learner has many *distinct* wildcard-only suffixes to build a suffix
        family from -- with a single wildcard there is only one such suffix per
        length, and the family cannot be filled.
        """
        return tuple(range(self.num_kmers, self.alphabet_size))

    @property
    def alphabet_size(self) -> int:
        """Size of the *super* alphabet: one symbol per kmer, plus the wildcards."""
        return self.num_kmers + self.num_wildcards

    @property
    def max_kmer_length(self) -> int:
        return max((len(k) for k in self.kmers), default=1)

    def is_unknown(self, symbol: int) -> bool:
        return symbol >= self.num_kmers

    def canonicalize(self, super_string: Iterable[int]) -> List[int]:
        """Map every wildcard to the canonical one.  :meth:`parse` cannot recover
        which wildcard was used (they compile identically), so round-tripping is
        exact only up to this relabelling."""
        return [self.unknown_symbol if self.is_unknown(s) else s for s in super_string]

    # -- distribution --------------------------------------------------------

    def probabilities(self) -> np.ndarray:
        """The per-position emission law at a fresh parse position: a uniform base
        string starts with ``kmers[i]`` with probability ``base ** -len``, and the
        wildcards split the remainder evenly.  (This is the conditional at one
        position, not the marginal of :class:`SuperSampler`, a Markov process.)
        """
        base = self.base_alphabet_size
        probs = np.zeros(self.alphabet_size)
        for i, kmer in enumerate(self.kmers):
            probs[i] = base ** (-len(kmer))
        # max guards against floating-point rounding pushing the sum over 1.
        remainder = max(0.0, 1.0 - probs[: self.num_kmers].sum())
        for w in self.wildcard_symbols:
            probs[w] = remainder / self.num_wildcards
        return probs

    # -- parse / compile -----------------------------------------------------

    def compiled_length(self, symbol: int) -> int:
        """How many base symbols ``symbol`` compiles to (``X`` -> 1)."""
        return 1 if self.is_unknown(symbol) else len(self.kmers[symbol])

    def parse(self, base_string: Sequence[int]) -> List[int]:
        """Read a base string back into super-symbols by greedy longest match:
        at each position emit the kmer that starts there (unique, since prefix-free)
        or else one ``X``."""
        b = list(base_string)
        out: List[int] = []
        i, n = 0, len(b)
        while i < n:
            matched = None
            for idx, km in enumerate(self.kmers):
                if b[i : i + len(km)] == list(km):
                    matched = (idx, len(km))
                    break
            if matched is not None:
                out.append(matched[0])
                i += matched[1]
            else:
                out.append(self.unknown_symbol)
                i += 1
        return out

    def compile(
        self, super_string: Iterable[int], rng: np.random.Generator
    ) -> List[int]:
        """Compile a super-string to a base string that :meth:`parse` reads back as
        the original (up to which wildcard was used).

        Kmer symbols emit their kmer; each wildcard emits one base symbol that does
        not start a kmer given what follows it, drawn with the Perron weights of
        :func:`_transfer_tables` so the result is fiber-uniform away from the
        boundaries.  Going right to left makes "what follows" already known, so this
        is a single pass of table lookups.
        """
        template = self._template(super_string)
        state, choice, shift = _transfer_tables(self.kmers, self.base_alphabet_size)
        out: List[int] = [0] * len(template)
        draws = rng.random(len(template))
        for pos in range(len(template) - 1, -1, -1):
            chosen = template[pos]
            if chosen == -1:
                options, cumulative = choice[state]
                assert cumulative, "super-string has no valid compilation"
                chosen = options[bisect.bisect_left(cumulative, draws[pos])]
            out[pos] = chosen
            state = shift[chosen][state]
        return out

    def _template(self, super_string: Iterable[int]) -> List[int]:
        """Base-string layout: the fixed symbol at each kmer position, ``-1`` at
        each free wildcard slot."""
        template: List[int] = []
        for symbol in super_string:
            if self.is_unknown(symbol):
                template.append(-1)
            else:
                template.extend(self.kmers[symbol])
        return template

    # -- construction --------------------------------------------------------

    @classmethod
    def from_corpus(
        cls,
        corpus: Iterable[Sequence[int]],
        base_alphabet_size: int,
        *,
        lengths: Sequence[int] = (3, 4, 5, 6),
        top_n: int = 10,
    ) -> "KmerVocabulary":
        """Pick the ``top_n`` most frequent kmers over ``corpus``, pruned prefix-free.

        Every contiguous substring with length in ``lengths`` is counted and the
        kmers are ranked by count (ties broken by the kmer, for determinism).  A
        candidate prefix-related to an already-kept kmer is skipped, keeping the
        vocabulary prefix-free (the higher-count member of a conflict wins).
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
            if any(_prefix_related(kmer, o) for o in selected):
                continue
            selected.append(kmer)
        return cls(kmers=tuple(selected), base_alphabet_size=base_alphabet_size)
