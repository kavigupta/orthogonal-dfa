"""An alphabet of K kmers plus interchangeable wildcards, built over a base
alphabet, with a translation to and from it that is invertible and preserves
uniformity.

parse reads a base string back by longest match. compile is its inverse, up to
which wildcard was used: the wildcards translate identically, so parse cannot
tell them apart and only canonicalize(s) is recovered.

Two things have to hold for a base string to parse back to the super-string that
produced it: a wildcard must not emit a symbol that starts a kmer, and a kmer
must not be extended by what follows it into a longer kmer. Both are conditions
on the symbols after a position, so compile fills the string right to left, where
those are already decided.

Which symbols are legal varies by position, so drawing evenly among them would
over-represent the constrained ones. compile instead weights each by how many
ways the rest of the string can then be filled, which is the uniform law on the
fiber and makes compile(parse(x)) uniform whenever x is.

When no kmer is a prefix of another, every super-string is encodable. When one
is, some are not -- (0,1) then (2,0) is forced to spell 0,1,2,0, which reads back
as the longer kmer (0,1,2) -- and compile raises for those.

Extra wildcards buy no expressive power, only strings: with one wildcard there is
a single wildcard-only super-string of each length, with two there are 2**n.
"""

from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable, List, Sequence, Tuple

import numpy as np

Kmer = Tuple[int, ...]

_FREE = -1  # a wildcard, to be filled in
_PAD = -2  # not part of this string: it is shorter than others in its batch


@lru_cache(maxsize=None)
def _transfer_tables(kmers: Tuple[Kmer, ...], base: int):
    """Whether a wildcard may emit a symbol depends on the symbols after it, so a
    state here is the next max_kmer_length - 1 of them, packed in radix base + 1.
    The extra digit is a sentinel for running off the end of the string, which
    matches no kmer.

    shift[c, state] is the state one position further left after emitting c,
    allowed[state, c] whether a wildcard may emit c there, and extendable[k, state]
    whether placing kmer k there would let a longer kmer swallow it.
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

    # A kmer is only safe where no longer one matches the same position, since
    # parse takes the longest. Only kmers extending it can, so the check needs
    # its first symbol and the window, which is exactly what is on hand.
    extendable = np.zeros((max(len(kmers), 1), num_states), dtype=bool)
    for index, kmer in enumerate(kmers):
        for state in range(num_states):
            ahead = [kmer[0]] + [(state // radix**i) % radix for i in range(w)]
            extendable[index, state] = any(
                len(other) > len(kmer) and ahead[: len(other)] == list(other)
                for other in kmers
            )

    initial = num_states - 1 if w else 0  # all sentinels: nothing follows the end
    return initial, allowed, shift, extendable


def _compile_chunk(templates, rngs, tables):
    """Templates are right-aligned so that every string's last position shares the
    final column: sampling runs right to left, so they then all start in the
    end-of-string state and stay in step.
    """
    initial, allowed, shift, extendable = tables
    base, num_states = shift.shape
    size = len(templates)
    width = max(len(t) for t, _ in templates)
    grid = np.full((size, width), _PAD, dtype=np.int64)
    heads = np.full((size, width), -1, dtype=np.int64)
    start = np.empty(size, dtype=np.int64)
    draws = np.zeros((size, width))
    for row, ((template, kmer_starts), rng) in enumerate(zip(templates, rngs)):
        start[row] = width - len(template)
        grid[row, start[row] :] = template
        heads[row, start[row] :] = kmer_starts
        draws[row, start[row] :] = rng.random(len(template))

    # ways[j][row, s] counts the fillings of the columns left of j when s follows
    # column j-1. Rescaled each step: the true counts grow geometrically and
    # overflow, and only ratios within a row are ever read.
    ways = np.ones((width + 1, size, num_states))
    for j in range(width):
        previous, column = ways[j], grid[:, j]
        free = (previous[:, shift] * allowed.T[None]).sum(axis=1)
        fixed = np.take_along_axis(
            previous, shift[np.where(column >= 0, column, 0)], axis=1
        )
        # Where a kmer starts, the states a longer kmer would swallow it in are
        # dead ends, so they count nothing.
        head = heads[:, j]
        fixed = np.where(
            (head >= 0)[:, None] & extendable[np.where(head >= 0, head, 0)], 0.0, fixed
        )
        row = np.where((column == _FREE)[:, None], free, fixed)
        row = np.where((column == _PAD)[:, None], previous, row)
        peak = np.max(row, axis=1)[:, None]
        ways[j + 1] = np.divide(row, peak, out=row.copy(), where=peak > 0)

    # Nothing follows the last column, so that is where a whole string's count
    # sits. Zero there means the constraints cannot all be met at once.
    impossible = np.flatnonzero(ways[width][:, initial] <= 0)
    if impossible.size:
        raise ValueError(
            f"super-string {impossible[0]} of this batch has no compilation that "
            "parses back to it: a kmer in it is a prefix of a longer one that "
            "what follows completes"
        )

    state = np.full(size, initial, dtype=np.int64)
    out = np.zeros((size, width), dtype=np.int64)
    for j in range(width - 1, -1, -1):
        column = grid[:, j]
        live = column != _PAD
        free_here = column == _FREE
        weights = (
            np.take_along_axis(ways[j], shift[:, state].T, axis=1) * allowed[state]
        )
        running = np.cumsum(weights, axis=1)
        assert (running[free_here, -1] > 0).all(), "sampled into a dead end"
        picked = (running < (draws[:, j] * running[:, -1])[:, None]).sum(axis=1)
        chosen = np.where(
            free_here, np.clip(picked, 0, base - 1), np.where(live, column, 0)
        )
        out[:, j] = chosen
        state = np.where(live, shift[chosen, state], state)
    return [out[row, start[row] :].tolist() for row in range(size)]


@dataclass(frozen=True)
class KmerVocabulary:
    """Super-symbol i is kmers[i]; the symbols past those are the wildcards, so
    the kmer order fixes the encoding.
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

    @property
    def num_kmers(self) -> int:
        return len(self.kmers)

    @property
    def unknown_symbol(self) -> int:
        """The wildcard that parse emits, and that canonicalize maps the rest to."""
        return self.num_kmers

    @property
    def wildcard_symbols(self) -> Tuple[int, ...]:
        """Interchangeable: each compiles the same way, so nothing reading the base
        string can distinguish them.
        """
        return tuple(range(self.num_kmers, self.alphabet_size))

    @property
    def alphabet_size(self) -> int:
        return self.num_kmers + self.num_wildcards

    @property
    def max_kmer_length(self) -> int:
        return max((len(k) for k in self.kmers), default=1)

    def is_unknown(self, symbol: int) -> bool:
        return symbol >= self.num_kmers

    def canonicalize(self, super_string: Iterable[int]) -> List[int]:
        """Collapse the wildcards, which is as much as parse can recover."""
        return [self.unknown_symbol if self.is_unknown(s) else s for s in super_string]

    def compiled_length(self, symbol: int) -> int:
        return 1 if self.is_unknown(symbol) else len(self.kmers[symbol])

    def parse(self, base_string: Sequence[int]) -> List[int]:
        """At each position take the longest kmer starting there, or else one
        wildcard. Longest rather than first keeps this independent of the order
        the kmers were given in, but it is still leftmost-first: an occurrence
        overlapping an earlier match is not seen.
        """
        b = list(base_string)
        out: List[int] = []
        i, n = 0, len(b)
        while i < n:
            index, size = None, 0
            for idx, km in enumerate(self.kmers):
                if len(km) > size and b[i : i + len(km)] == list(km):
                    index, size = idx, len(km)
            if index is None:
                out.append(self.unknown_symbol)
                i += 1
            else:
                out.append(index)
                i += size
        return out

    def compile(
        self, super_string: Iterable[int], rng: np.random.Generator
    ) -> List[int]:
        """Prefer compile_many for more than one: the cost is a pass over the
        string, which batching amortizes.
        """
        return self.compile_many([super_string], [rng])[0]

    def compile_many(
        self,
        super_strings: Sequence[Iterable[int]],
        rngs: Sequence[np.random.Generator],
    ) -> List[List[int]]:
        """Each result is drawn uniformly over the base strings that parse back to
        its super-string, so compile(parse(x)) is uniform when x is.

        Reaching that uniformity means weighting a wildcard's choices by how many
        ways the rest of the string can then be filled, which a backward pass
        counts and a forward pass samples against. Both walk positions, not
        strings, so a batch advances in step and pays the per-position cost once.
        """
        assert len(super_strings) == len(rngs), "need one rng per super-string"
        tables = _transfer_tables(self.kmers, self.base_alphabet_size)
        num_states = tables[2].shape[1]
        templates = [self._template(s) for s in super_strings]
        if not templates:
            return []
        width = max(len(t) for t, _ in templates)
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
        """The base string with the kmers filled in and the wildcards left open,
        alongside which column each kmer starts at -- the only columns where a
        kmer could be swallowed by a longer one."""
        template: List[int] = []
        starts: List[int] = []
        for symbol in super_string:
            if self.is_unknown(symbol):
                template.append(_FREE)
                starts.append(-1)
            else:
                starts.extend([symbol] + [-1] * (len(self.kmers[symbol]) - 1))
                template.extend(self.kmers[symbol])
        return template, starts
