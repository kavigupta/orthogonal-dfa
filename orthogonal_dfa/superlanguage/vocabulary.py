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
it.  Those choices are coupled, and :meth:`compile` weights them so that the result
is drawn *uniformly over the fiber* ``parse**-1(s)`` -- which makes
``compile(parse(x))`` exactly uniform when ``x`` is a uniform base string.  That
costs a pass over the string, so :meth:`compile_many` runs the pass for a whole
batch at once.  The kmers must be prefix-free so the parse is unambiguous.

Several wildcards exist for the learner's benefit: wildcard-only suffixes have the
same membership column as the empty suffix, so they are what the suffix-family
clustering locks onto -- and with a single wildcard there is only *one* such
suffix per length, too few to fill a family.
"""

from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable, List, Sequence, Tuple

import numpy as np

Kmer = Tuple[int, ...]

# Template slots that hold no fixed base symbol: a wildcard to fill, and padding
# left of a right-aligned string in a batch.
_FREE = -1
_PAD = -2


def _prefix_related(a: Sequence[int], b: Sequence[int]) -> bool:
    """True when the shorter of ``a``/``b`` is a prefix of the longer."""
    m = min(len(a), len(b))
    return tuple(a[:m]) == tuple(b[:m])


@lru_cache(maxsize=None)
def _transfer_tables(kmers: Tuple[Kmer, ...], base: int):
    """Compilation tables, built once per vocabulary.

    A state encodes the ``w = max_kmer_length - 1`` base symbols *following* a
    position, in radix ``base + 1`` so that ``base`` itself can act as an
    end-of-string sentinel (it matches no kmer).  ``shift[c, state]`` is the state
    seen one position to the left after emitting ``c``, and ``allowed[state, c]``
    says whether a wildcard may emit ``c`` there without starting a kmer.
    """
    w = max((len(k) for k in kmers), default=1) - 1
    radix = base + 1
    num_states = radix**w

    shift = np.array(
        [
            [
                c + radix * (state % radix ** (w - 1)) if w >= 1 else 0
                for state in range(num_states)
            ]
            for c in range(base)
        ],
        dtype=np.int64,
    )

    allowed = np.ones((num_states, base), dtype=bool)
    for state in range(num_states):
        following = [(state // radix**i) % radix for i in range(w)]
        for kmer in kmers:
            if following[: len(kmer) - 1] == list(kmer[1:]):
                allowed[state, kmer[0]] = False

    initial = num_states - 1 if w else 0  # all sentinels: nothing follows the end
    return initial, allowed, shift


def _compile_chunk(templates, rngs, tables):
    """One batch of :meth:`KmerVocabulary.compile_many`.

    Templates are right-aligned into a ``(batch, width)`` grid padded with ``_PAD``
    so every string's last position lands in the final column; the sampling pass
    then starts them all in the same end-of-string state and walks left in lockstep.
    """
    initial, allowed, shift = tables
    base, num_states = shift.shape
    size = len(templates)
    width = max(len(t) for t in templates)
    grid = np.full((size, width), _PAD, dtype=np.int64)
    start = np.empty(size, dtype=np.int64)
    draws = np.zeros((size, width))
    for row, (template, rng) in enumerate(zip(templates, rngs)):
        start[row] = width - len(template)
        grid[row, start[row] :] = template
        draws[row, start[row] :] = rng.random(len(template))

    # ways[j][row, s]: number of fillings of the columns left of j, given that the
    # window following column j-1 is s.  Rescaled per step; the true counts grow
    # like the transfer matrix's leading eigenvalue and would overflow.
    ways = np.ones((width + 1, size, num_states))
    for j in range(width):
        previous, column = ways[j], grid[:, j]
        free = (previous[:, shift] * allowed.T[None]).sum(axis=1)
        fixed = np.take_along_axis(
            previous, shift[np.where(column >= 0, column, 0)], axis=1
        )
        row = np.where((column == _FREE)[:, None], free, fixed)
        row = np.where((column == _PAD)[:, None], previous, row)
        peak = np.max(row, axis=1)[:, None]
        ways[j + 1] = np.divide(row, peak, out=row.copy(), where=peak > 0)

    state = np.full(size, initial, dtype=np.int64)
    out = np.zeros((size, width), dtype=np.int64)
    for j in range(width - 1, -1, -1):
        column = grid[:, j]
        live = column != _PAD
        weights = (
            np.take_along_axis(ways[j], shift[:, state].T, axis=1) * allowed[state]
        )
        running = np.cumsum(weights, axis=1)
        assert (running[live, -1] > 0).all(), "super-string has no valid compilation"
        picked = (running < (draws[:, j] * running[:, -1])[:, None]).sum(axis=1)
        chosen = np.where(
            column == _FREE, np.clip(picked, 0, base - 1), np.where(live, column, 0)
        )
        out[:, j] = chosen
        state = np.where(live, shift[chosen, state], state)
    return [out[row, start[row] :].tolist() for row in range(size)]


@dataclass(frozen=True)
class KmerVocabulary:
    """A frozen, prefix-free kmer vocabulary over a base alphabet.

    ``kmers`` are ordered (the order fixes the super-symbol indices), each a tuple
    of base symbols, and no kmer may be a prefix of another.
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
        """Compile one super-string; see :meth:`compile_many`, which this calls.

        Prefer :meth:`compile_many` when compiling a batch: the work per string is
        a pass over its length, and batching amortizes it.
        """
        return self.compile_many([super_string], [rng])[0]

    def compile_many(
        self,
        super_strings: Sequence[Iterable[int]],
        rngs: Sequence[np.random.Generator],
    ) -> List[List[int]]:
        """Compile super-strings to base strings that :meth:`parse` reads back as
        the originals (up to which wildcard was used), each drawn **uniformly over
        its fiber** ``parse**-1(s)`` -- so ``compile(parse(x))`` is uniform when
        ``x`` is.

        Kmer symbols emit their kmer; a wildcard may emit any symbol that does not
        start a kmer given what follows it.  Those choices are coupled, and the
        uniform law weights each by how many ways the rest can then be filled, so a
        backward pass counts fillings (``ways[j][s]``, renormalized per step since
        the true counts overflow) and a forward pass samples right to left against
        those counts.

        Both passes are a loop over *positions*, so the whole batch advances
        together and the per-step cost is paid once rather than once per string.
        """
        assert len(super_strings) == len(rngs), "need one rng per super-string"
        tables = _transfer_tables(self.kmers, self.base_alphabet_size)
        num_states = tables[2].shape[1]
        templates = [self._template(s) for s in super_strings]
        if not templates:
            return []
        width = max(len(t) for t in templates)
        # Cap the backward pass's (width, chunk, num_states) table at ~64MB.
        chunk = int(np.clip(8_000_000 // max((width + 1) * num_states, 1), 1, 4096))

        out_all: List[List[int]] = []
        for lo in range(0, len(templates), chunk):
            out_all.extend(
                _compile_chunk(
                    templates[lo : lo + chunk], rngs[lo : lo + chunk], tables
                )
            )
        return out_all

    def _template(self, super_string: Iterable[int]) -> List[int]:
        """Base-string layout: the fixed symbol at each kmer position, ``_FREE`` at
        each wildcard slot."""
        template: List[int] = []
        for symbol in super_string:
            if self.is_unknown(symbol):
                template.append(_FREE)
            else:
                template.extend(self.kmers[symbol])
        return template
