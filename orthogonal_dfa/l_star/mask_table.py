"""The prefix x suffix membership table.

A single object that owns the prefixes, the suffixes, and the (lazily observed)
membership matrix between them.  It is the *only* place the raw table lives;
everything else works through the small interface below and never touches the
underlying arrays.

Each cell is int8: ``0`` (reject), ``1`` (accept), or ``UNOBSERVED (-1)`` for a
``(prefix, suffix)`` pair whose membership query has not been issued yet.  A new
suffix (``intern_suffix``) reserves an all-``UNOBSERVED`` column and queries
nothing; a cell is filled the first time some read (``observed_masks`` /
``column``) actually needs it.  ``add_prefixes`` reserves ``UNOBSERVED`` cells
for partially-observed columns but does query the new prefixes for already
fully-observed (family) columns, to keep them clustering candidates.  Because the
oracle is deterministic per string, lazy filling returns exactly the values eager
filling would, so callers cannot tell the difference except in query count.
"""

from typing import Dict, List

import numpy as np

from .memoized_oracle import MemoizedOracle

# Sentinel for a not-yet-queried cell.  Private to this module: callers ask about
# observation through ``fully_observed`` / ``observed_masks`` and never see it.
UNOBSERVED = np.int8(-1)


class MaskTable:
    def __init__(self, oracle, prefixes: List[bytes], *, population):
        # Memoize membership per string.  The matrix already dedups by (prefix,
        # suffix) cell; this additionally dedups across cells that spell the same
        # string, and allows us to remove the matrix caching in future.
        self.memo = MemoizedOracle(oracle)
        self._oracle = self.memo
        self._prefixes = list(prefixes)
        self._prefix_index: Dict[bytes, int] = {
            p: i for i, p in enumerate(self._prefixes)
        }
        self._populations: Dict[object, set] = {
            population: set(self._prefix_index.values())
        }
        self._suffixes: List[bytes] = []
        self._suffix_index = {}  # suffix -> row
        self._masks: List[np.ndarray] = []  # one int8 column per suffix

    # -- sizes / prefix side ------------------------------------------------

    @property
    def num_prefixes(self) -> int:
        return len(self._prefixes)

    @property
    def prefixes(self) -> tuple:
        """The prefixes, for read-only iteration."""
        return tuple(self._prefixes)

    @property
    def representative(self) -> np.ndarray:
        """Boolean mask selecting the prefixes clustering reads: every prefix in
        some live population."""
        mask = np.zeros(len(self._prefixes), dtype=bool)
        for columns in self._populations.values():
            mask[list(columns)] = True
        return mask

    def drop_population(self, label) -> None:
        """Retire a population.  Its prefixes stay in the table -- they are
        still columns, and may be in other populations -- but nothing is held to
        a rate over them any more.

        Rounds redefine what a state is, so the population that named one has to
        stop being live when the round that drew it ends."""
        self._populations.pop(label, None)

    def _column_of(self, prefix: bytes) -> int:
        column = self._prefix_index.get(prefix)
        assert (
            column is not None
        ), f"population names a prefix not in the table: {prefix!r}"
        return column

    def population_masks(self):
        """``label -> mask`` over the representative prefixes, in table order.

        Masks overlap where populations do, so a prefix can be evidence in more
        than one and the rates are read over all of each."""
        scoped = np.flatnonzero(self.representative)
        where = {column: i for i, column in enumerate(scoped)}
        out = {}
        for label, columns in self._populations.items():
            mask = np.zeros(len(scoped), dtype=bool)
            mask[[where[c] for c in columns]] = True
            if mask.any():
                out[label] = mask
        return out

    def contains_prefix(self, prefix: bytes) -> bool:
        return prefix in self._prefix_index

    def add_prefixes(self, prefixes: List[bytes], *, population) -> None:
        """Put ``prefixes`` in ``population``, adding any the table lacks.

        A prefix the table already holds joins the population without being
        added again: being a uniform draw does not stop it also being what is
        known about a state."""
        assert prefixes, "No prefixes to add"
        assert len(prefixes) == len(set(prefixes)), "Prefixes must be unique"
        new_prefixes = [p for p in prefixes if not self.contains_prefix(p)]
        if new_prefixes:
            self._add_columns(new_prefixes)
        self._populations.setdefault(population, set()).update(
            self._prefix_index[p] for p in prefixes
        )

    def _add_columns(self, new_prefixes: List[bytes]) -> None:
        # A column that is already fully observed is a family suffix: keep it
        # fully observed by querying the new prefixes, so it stays a clustering
        # candidate.  A partially-observed column (a transition distinguisher)
        # gets UNOBSERVED cells, filled later on demand only if some read needs
        # them.
        pad = np.full(len(new_prefixes), UNOBSERVED, dtype=np.int8)
        # Flatten out the pairs to update
        full_cols = [
            i for i, col in enumerate(self._masks) if (col != UNOBSERVED).all()
        ]
        adds = {}
        if full_cols:
            strings = [p + self._suffixes[i] for i in full_cols for p in new_prefixes]
            observed = np.asarray(
                self._oracle.membership_queries(strings), dtype=np.int8
            ).reshape(len(full_cols), len(new_prefixes))
            adds = {i: observed[k] for k, i in enumerate(full_cols)}
        updated = [
            np.concatenate([col, adds.get(i, pad)]) for i, col in enumerate(self._masks)
        ]
        self._masks = updated
        fresh = range(len(self._prefixes), len(self._prefixes) + len(new_prefixes))
        self._prefixes.extend(new_prefixes)
        self._prefix_index.update(zip(new_prefixes, fresh))

    # -- suffix side --------------------------------------------------------

    def intern_suffix(self, v: bytes) -> int:
        """Return the row index for suffix ``v``, registering it (with an
        all-``UNOBSERVED`` column, no queries) if it is new."""
        if v in self._suffix_index:
            return self._suffix_index[v]
        row = len(self._suffixes)
        self._suffixes.append(v)
        self._masks.append(np.full(self.num_prefixes, UNOBSERVED, dtype=np.int8))
        self._suffix_index[v] = row
        return row

    def contains_suffix(self, v: bytes) -> bool:
        return v in self._suffix_index

    def suffix(self, row: int) -> bytes:
        return self._suffixes[row]

    # -- observation / reads ------------------------------------------------

    def _ensure(self, rows, prefix_mask) -> None:
        """Fill any UNOBSERVED cells for ``rows`` over the boolean
        ``prefix_mask``.  Cells already observed are reused, so no
        ``(prefix, suffix)`` pair is queried twice."""
        assert len(set(rows)) == len(rows), "rows must be distinct"
        idx = np.flatnonzero(prefix_mask)
        strings, targets = [], []
        for r in rows:
            col = self._masks[r]
            suffix = self._suffixes[r]
            for p in idx[col[idx] == UNOBSERVED]:
                strings.append(self._prefixes[p] + suffix)
                targets.append((r, p))
        if not strings:
            return
        results = self._oracle.membership_queries(strings)
        assert len(results) == len(strings), "oracle dropped answers"
        for (r, p), val in zip(targets, results):
            self._masks[r][p] = val

    def observed_masks(self, rows, prefix_mask) -> np.ndarray:
        """The ``(len(rows), prefix_mask.sum())`` int8 block for ``rows`` over the
        prefixes selected by ``prefix_mask``, querying any cells not yet
        observed."""
        self._ensure(rows, prefix_mask)
        return np.array([self._masks[r][prefix_mask] for r in rows])

    def column(self, row: int) -> np.ndarray:
        """Fully observe suffix ``row`` over every prefix and return its column.
        Also used to promote a suffix to "fully observed" (a clustering
        candidate)."""
        self._ensure([row], np.ones(self.num_prefixes, dtype=bool))
        return self._masks[row].copy()

    def fully_observed(self) -> np.ndarray:
        """Row indices of the suffixes whose whole column is observed -- the
        sampled acceptance-family suffixes.  Partially-observed transition
        distinguishers are excluded."""
        if not self._masks:
            return np.array([], dtype=int)
        matrix = np.array(self._masks)
        return np.flatnonzero((matrix != UNOBSERVED).all(axis=1))
