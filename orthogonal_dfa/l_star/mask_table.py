"""The prefix x suffix membership table.

A single object that owns the prefixes, the suffixes, and the membership matrix
between them.  It is the *only* place the raw table lives; everything else works
through the small interface below and never touches the underlying arrays.

Each cell is ``membership_query(prefix + suffix)`` as a float32 ``0``/``1``.  A
suffix's whole column is queried when the suffix is interned, and every column is
extended when new prefixes are added, so the matrix is always fully populated.
"""

from typing import List

import numpy as np


class MaskTable:
    def __init__(self, oracle, prefixes: List[List[int]], representative: List[bool]):
        assert len(prefixes) == len(representative)
        self._oracle = oracle
        self._prefixes = [list(p) for p in prefixes]
        self._prefix_keys = {tuple(p) for p in self._prefixes}
        self._representative = list(representative)
        self._suffixes: List[List[int]] = []
        self._suffix_index = {}  # tuple(suffix) -> row
        self._masks: List[np.ndarray] = []  # one float32 column per suffix

    # -- sizes / prefix side ------------------------------------------------

    @property
    def num_prefixes(self) -> int:
        return len(self._prefixes)

    @property
    def num_suffixes(self) -> int:
        return len(self._suffixes)

    @property
    def prefixes(self) -> tuple:
        """The prefixes, for read-only iteration."""
        return tuple(self._prefixes)

    @property
    def representative(self) -> np.ndarray:
        """Boolean mask selecting the representative (non-core) prefixes."""
        return np.array(self._representative, dtype=bool)

    def contains_prefix(self, prefix: List[int]) -> bool:
        return tuple(prefix) in self._prefix_keys

    def add_prefixes(self, new_prefixes: List[List[int]]) -> None:
        assert new_prefixes, "No new prefixes to add"
        assert all(not self.contains_prefix(p) for p in new_prefixes) and len(
            new_prefixes
        ) == len({tuple(p) for p in new_prefixes}), "Prefixes must be unique"
        # Extend every existing suffix column to cover the new prefixes.
        self._masks = [
            np.concatenate([col, self._query(suffix, new_prefixes)])
            for suffix, col in zip(self._suffixes, self._masks)
        ]
        self._prefixes.extend(list(p) for p in new_prefixes)
        self._prefix_keys.update(tuple(p) for p in new_prefixes)
        # Prefixes added after construction (counterexamples, leaf enrichment)
        # are full-length probe prefixes, hence representative.
        self._representative.extend([True] * len(new_prefixes))

    # -- suffix side --------------------------------------------------------

    def intern_suffix(self, v: List[int]) -> int:
        """Return the row index for suffix ``v``, registering it -- and querying
        its whole column against the oracle -- if it is new."""
        key = tuple(v)
        if key in self._suffix_index:
            return self._suffix_index[key]
        row = len(self._suffixes)
        self._suffixes.append(list(v))
        self._masks.append(self._query(v, self._prefixes))
        self._suffix_index[key] = row
        return row

    def contains_suffix(self, v: List[int]) -> bool:
        return tuple(v) in self._suffix_index

    def suffix(self, row: int) -> List[int]:
        return self._suffixes[row]

    # -- reads --------------------------------------------------------------

    def observed_masks(self, rows, prefix_mask) -> np.ndarray:
        """The ``(len(rows), prefix_mask.sum())`` block for suffix ``rows`` over
        the prefixes selected by ``prefix_mask``."""
        return np.array([self._masks[r][prefix_mask] for r in rows])

    def column(self, row: int) -> np.ndarray:
        """The full membership column of suffix ``row`` over every prefix."""
        return self._masks[row].copy()

    # -- internal -----------------------------------------------------------

    def _query(self, suffix: List[int], prefixes: List[List[int]]) -> np.ndarray:
        return np.array(
            [self._oracle.membership_query(p + suffix) for p in prefixes],
            dtype=np.float32,
        )
