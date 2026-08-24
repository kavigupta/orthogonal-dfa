"""
An alphabet of K prefix-free kmers plus interchangeable wildcards, that abstracts
a base alphabet, with a "compiler" that moves from the abstracted language to the
original, and a "parser" that moves back.

The compiler is a randomized function that fills in the wildcards with base symbols
so that the result parses back to the original super-string.

The parser is deterministic and greedy: it reads the base string left to right, taking
the first kmer it sees, or else a wildcard.

We guarantee that parse(compile(y)) = y up to wildcard identity, and that compile(y) is
uniform over the base strings that parse back to y.

We require two restrictions on the kmers: they must be prefix-free, and they must not leave any
position in the base string with no symbol a wildcard could take. E.g., over the
alphabet {A, C, G, T} the kmers {AAA, CAA, GAA, TAA} are not allowed, because
the string X AAA can't be compiled, as whatever X resolves to will get merged with
AA from AAA.

We allow multiple wildcards to ensure we can simulate a diversity of strings, they are,
in fact, interchangeable.
"""

from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable, List, Sequence, Tuple

import numpy as np

Kmer = Tuple[int, ...]

_FREE = -1  # a wildcard, to be filled in
_PAD = -2  # not part of this string: it is shorter than others in its batch


def _prefix_related(a: Sequence[int], b: Sequence[int]) -> bool:
    """True when one of a, b is a prefix of the other, so both cannot be kmers."""
    m = min(len(a), len(b))
    return tuple(a[:m]) == tuple(b[:m])


@lru_cache(maxsize=None)
def _transfer_tables(kmers: Tuple[Kmer, ...], base: int):
    """Whether a wildcard may emit a symbol depends on the symbols after it, so a
    state here is the next max_kmer_length - 1 of them, packed in radix base + 1.
    The extra digit is a sentinel for running off the end of the string, which
    matches no kmer.

    shift[c, state] is the state one position further left after emitting c, and
    allowed[state, c] whether a wildcard may emit c there.
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
    """Templates are right-aligned so that every string's last position shares the
    final column: sampling runs right to left, so they then all start in the
    end-of-string state and stay in step.

    The restrictions on the kmers leave every super-string with a compilation, so
    sampling never reaches a position with nothing to choose.
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
        picked = (running <= (draws[:, j] * running[:, -1])[:, None]).sum(axis=1)
        chosen = np.where(
            column == _FREE, np.clip(picked, 0, base - 1), np.where(live, column, 0)
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
        # Together these two make compile total. Without the first, what follows a
        # kmer can grow it into a longer one and the super-string it came from is
        # then unencodable; without the second there is a run of symbols before
        # which no wildcard can go, since every symbol would start a kmer.
        for i, a in enumerate(self.kmers):
            for b in self.kmers[i + 1 :]:
                assert not _prefix_related(a, b), f"{a} and {b} are prefix-related"
        _, allowed, _ = _transfer_tables(self.kmers, self.base_alphabet_size)
        assert allowed.any(axis=1).all(), (
            "the kmers leave some position with no symbol a wildcard could take, "
            "so a super-string using one there could not be compiled"
        )

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
        """At each position take the kmer starting there, or else one wildcard.
        Prefix-freeness makes at most one kmer match, so this is unambiguous, but
        it is still leftmost-first: an occurrence overlapping an earlier match is
        not seen.
        """
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
        """The base string with the kmers filled in and the wildcards left open."""
        template: List[int] = []
        for symbol in super_string:
            # a negative symbol is not a wildcard, and would index kmers from the end
            assert (
                0 <= symbol < self.alphabet_size
            ), f"{symbol} is outside the super alphabet"
            if self.is_unknown(symbol):
                template.append(_FREE)
            else:
                template.extend(self.kmers[symbol])
        return template
