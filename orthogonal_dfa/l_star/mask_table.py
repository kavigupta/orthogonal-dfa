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

from typing import List

import numpy as np

# Sentinel for a not-yet-queried cell.  Private to this module: callers ask about
# observation through ``fully_observed`` / ``observed_masks`` and never see it.
UNOBSERVED = np.int8(-1)


class MaskTable:
    def __init__(self, oracle, prefixes: List[List[int]], representative: List[bool]):
        assert len(prefixes) == len(representative)
        self._oracle = oracle
        self._prefixes = [list(p) for p in prefixes]
        self._prefix_keys = {tuple(p) for p in self._prefixes}
        self._representative = list(representative)
        self._suffixes: List[List[int]] = []
        self._suffix_index = {}  # tuple(suffix) -> row
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
    def probe_prefixes(self) -> tuple:
        """The representative prefixes -- the probe sample.  Every per-round pass
        over "the pool" means these; transient scratch rows are excluded so adding
        scratch never lengthens a pool scan."""
        return tuple(p for p, rep in zip(self._prefixes, self._representative) if rep)

    @property
    def representative(self) -> np.ndarray:
        """Boolean mask selecting the representative (non-core) prefixes."""
        return np.array(self._representative, dtype=bool)

    def set_representative(self, prefixes: List[List[int]]) -> None:
        """Make *exactly* ``prefixes`` the representative set (all others become
        non-representative).  Lets a driver curate the calibration/check
        population -- e.g. a balanced per-state resample -- without discarding the
        accumulated (scratch) prefixes the sift cache still needs."""
        keys = {tuple(p) for p in prefixes}
        self._representative = [tuple(p) in keys for p in self._prefixes]

    def prefix_rows(self, prefixes: List[List[int]]) -> np.ndarray:
        """A boolean mask selecting exactly ``prefixes``, which must be present."""
        wanted = {tuple(p) for p in prefixes}
        return np.array([tuple(p) in wanted for p in self._prefixes], dtype=bool)

    def ensure_prefixes(
        self, prefixes: List[List[int]], representative: bool = True
    ) -> None:
        """Add any of ``prefixes`` not already present, forwarding
        ``representative`` to :meth:`add_prefixes`."""
        missing = [p for p in prefixes if not self.contains_prefix(p)]
        if missing:
            deduped = list({tuple(p): list(p) for p in missing}.values())
            self.add_prefixes(deduped, representative=representative)

    def contains_prefix(self, prefix: List[int]) -> bool:
        return tuple(prefix) in self._prefix_keys

    def add_prefixes(
        self, new_prefixes: List[List[int]], representative: bool = True
    ) -> None:
        assert new_prefixes, "No new prefixes to add"
        assert all(not self.contains_prefix(p) for p in new_prefixes) and len(
            new_prefixes
        ) == len({tuple(p) for p in new_prefixes}), "Prefixes must be unique"
        # A column that is already fully observed is a family suffix: keep it
        # fully observed by querying the new prefixes, so it stays a clustering
        # candidate.  A partially-observed column (a transition distinguisher)
        # gets UNOBSERVED cells, filled later on demand only if some read needs
        # them.
        # ``representative=False`` marks transient scratch -- a string some caller
        # only needs a few cells of.  Such a row is added wholly unobserved: the
        # eager family-column observation below is what keeps a column a
        # clustering candidate, and :meth:`fully_observed` no longer counts
        # scratch rows, so there is nothing to keep up to date.
        pad = np.full(len(new_prefixes), UNOBSERVED, dtype=np.int8)
        rep_mask = self.representative
        full_cols = (
            [
                i
                for i, col in enumerate(self._masks)
                if (col[rep_mask] != UNOBSERVED).all()
            ]
            if representative
            else []
        )
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
        self._prefixes.extend(list(p) for p in new_prefixes)
        self._prefix_keys.update(tuple(p) for p in new_prefixes)
        # Prefixes added after construction (counterexamples, leaf enrichment)
        # are full-length probe prefixes, hence representative by default.
        self._representative.extend([representative] * len(new_prefixes))

    # -- suffix side --------------------------------------------------------

    def intern_suffix(self, v: List[int]) -> int:
        """Return the row index for suffix ``v``, registering it (with an
        all-``UNOBSERVED`` column, no queries) if it is new."""
        key = tuple(v)
        if key in self._suffix_index:
            return self._suffix_index[key]
        row = len(self._suffixes)
        self._suffixes.append(list(v))
        self._masks.append(np.full(self.num_prefixes, UNOBSERVED, dtype=np.int8))
        self._suffix_index[key] = row
        return row

    def contains_suffix(self, v: List[int]) -> bool:
        return tuple(v) in self._suffix_index

    def suffix(self, row: int) -> List[int]:
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
        """Observe suffix ``row`` over the *representative* prefixes and return
        its column.  Also used to promote a suffix to "fully observed" (a
        clustering candidate).

        Non-representative rows are skipped because nothing reads them: every
        consumer of a column slices to ``representative`` first (clustering, the
        give-up check, the FNR estimate), so observing the rest buys cells no
        reader looks at -- and it is what would make transient scratch prefixes
        ruinously expensive to keep in the pool."""
        self._ensure([row], self.representative)
        return self._masks[row].copy()

    def fully_observed(self) -> np.ndarray:
        """Row indices of the suffixes observed over every *representative*
        prefix -- the sampled acceptance-family suffixes.  Partially-observed
        transition distinguishers are excluded.

        Scratch rows are deliberately not required: :meth:`column` does not
        observe them, so demanding them would make a column's candidacy hinge on
        cells no reader ever reads."""
        rep = self.representative
        if not self._masks or not rep.any():
            return np.array([], dtype=int)
        matrix = np.array(self._masks)[:, rep]
        return np.flatnonzero((matrix != UNOBSERVED).all(axis=1))
