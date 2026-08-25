"""
An alphabet of K prefix-free kmers plus interchangeable wildcards, that abstracts
a base alphabet, with a "parser" that moves a base string into it.

The parser is deterministic and greedy: it reads the base string left to right, taking
the first kmer it sees, or else a wildcard. Prefix-freeness is what makes that
unambiguous.

We allow multiple wildcards to ensure we can simulate a diversity of strings, they are,
in fact, interchangeable.
"""

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

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
