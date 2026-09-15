"""
An alphabet of K prefix-free kmers plus interchangeable wildcards, that abstracts
a base alphabet, with a "compiler" that moves from the abstracted language to the
original, and a "parser" that moves back.

The compiler is a randomized function that fills in the wildcards with base symbols
so that the result parses back to the original super-string.

The parser is deterministic and greedy: it reads the base string left to right, taking
the first kmer it sees, or else a wildcard. Prefix-freeness is what makes that
unambiguous.

We guarantee that parse(compile(y)) = y up to wildcard identity, and that compile(y) is
uniform over the base strings that parse back to y.

Compile needs a second restriction beyond prefix-freeness: no context -- no run of the
next (longest kmer - 1) symbols -- may leave a wildcard with no symbol it could take.
E.g., over the alphabet {A, C, G, T} the kmers {AAA, CAA, GAA, TAA} are not allowed,
because the string X AAA can't be compiled, as whatever X resolves to will get merged
with AA from AAA. Both restrictions are stricter than strictly necessary; the second
rules out contexts compile would never have had to put a wildcard in.

We allow multiple wildcards to ensure we can simulate a diversity of strings, they are,
in fact, interchangeable.
"""

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import numpy as np

from .template_fill import FREE, TemplateFiller

Kmer = Tuple[int, ...]


def _prefix_related(a: Sequence[int], b: Sequence[int]) -> bool:
    """True when one of a, b is a prefix of the other, so both cannot be kmers."""
    m = min(len(a), len(b))
    return tuple(a[:m]) == tuple(b[:m])


@dataclass(frozen=True)
class KmerVocabulary:
    """Super-symbol i is kmers[i]; the symbols past those are the wildcards, so
    the kmer order fixes the encoding.
    """

    kmers: Tuple[Kmer, ...]
    base_alphabet_size: int
    num_wildcards: int = 2

    def __post_init__(self):
        # frozen, so the caller's lists would survive as unhashable fields and only
        # fail once something tried to cache on the vocabulary
        object.__setattr__(self, "kmers", tuple(tuple(k) for k in self.kmers))
        assert self.base_alphabet_size >= 1, "base alphabet must be non-empty"
        assert self.num_wildcards >= 1, "need at least one wildcard"
        for kmer in self.kmers:
            assert len(kmer) >= 1, "kmers must be non-empty"
            assert all(
                0 <= c < self.base_alphabet_size for c in kmer
            ), f"kmer {kmer} has symbols outside the base alphabet"
        assert len(set(self.kmers)) == len(self.kmers), "duplicate kmers"
        # compile asks only whether a wildcard starts a kmer, never whether it
        # extends the kmer to its left, which is what prefix-related ones need.
        for i, a in enumerate(self.kmers):
            for b in self.kmers[i + 1 :]:
                assert not _prefix_related(a, b), f"{a} and {b} are prefix-related"
        # Stricter than needed: some rejected contexts compile could steer around.
        assert self._filler.every_context_is_fillable, (
            "the kmers leave some position with no symbol a wildcard could take, "
            "so a super-string using one there could not be compiled"
        )

    @property
    def _filler(self) -> TemplateFiller:
        """A wildcard is a hole that must not start a kmer; everything about how one
        gets filled lives there.
        """
        return TemplateFiller(self.kmers, self.base_alphabet_size)

    @property
    def num_kmers(self) -> int:
        return len(self.kmers)

    @property
    def unknown_symbol(self) -> int:
        """The wildcard that parse emits, and that canonicalize maps the rest to."""
        return self.num_kmers

    @property
    def alphabet_size(self) -> int:
        return self.num_kmers + self.num_wildcards

    def is_unknown(self, symbol: int) -> bool:
        assert (
            0 <= symbol < self.alphabet_size
        ), f"{symbol} is outside the super alphabet"
        return symbol >= self.num_kmers

    def canonicalize(self, super_string: Iterable[int]) -> List[int]:
        """Collapse the wildcards, which is as much as parse can recover."""
        return [self.unknown_symbol if self.is_unknown(s) else s for s in super_string]

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
        """Prefer compile_many for more than one; the per-string pass amortizes."""
        return self.compile_many([super_string], [rng])[0]

    def compile_many(
        self,
        super_strings: Sequence[Iterable[int]],
        rngs: Sequence[np.random.Generator],
    ) -> List[List[int]]:
        """Each result is uniform over the base strings that parse back to its
        super-string.
        """
        templates = [self._template(s) for s in super_strings]
        return self._filler.fill_many(templates, rngs)

    def _template(self, super_string: Iterable[int]) -> List[int]:
        """The base string with the kmers filled in and the wildcards left open."""
        template: List[int] = []
        for symbol in super_string:
            if self.is_unknown(symbol):
                template.append(FREE)
            else:
                template.extend(self.kmers[symbol])
        return template
