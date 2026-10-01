import hashlib
from typing import List

import numpy as np

from .structures import Oracle

#: Digest bytes per key.  At 128 bits a collision -- which would silently hand
#: one string another's answer -- runs at ~1e-23 for 10^8 distinct strings.
DIGEST_SIZE = 16

#: The value a slot holds while empty; answers are 0 or 1.
_EMPTY = 255

#: Slots per bucket, compared in one step, so a lookup rarely has to probe on.
_BUCKET = 8

#: Share of slots filled, as tenths, past which the table grows by half again.
_MAX_LOAD_TENTHS = 8


class MemoizedOracle(Oracle):
    """Membership of arbitrary strings, memoized per string and batched.

    Keyed by a digest of the string rather than the string, in flat arrays
    rather than a dict: at low signal strength this cache holds tens of millions
    of entries, and a dict entry costs ~100 bytes to the arrays' ~30.
    """

    def __init__(self, oracle):
        self._oracle = oracle
        self._table = _DigestTable()

    @property
    def alphabet_size(self) -> int:
        return self._oracle.alphabet_size

    def membership_queries(self, strings: List[bytes]) -> List[int]:
        if not strings:
            return []
        keys = _keys(strings)
        answers = self._table.get(keys)
        missing = np.flatnonzero(answers == _EMPTY)
        if missing.size:
            _, first, inverse = np.unique(
                keys[missing].view(_KEY_ROW).ravel(),
                return_index=True,
                return_inverse=True,
            )
            # Asked in the order the strings first appear, as the dict did.
            order = np.argsort(first)
            asked = missing[first[order]]
            bits = np.asarray(
                self._oracle.membership_queries([strings[i] for i in asked]),
                dtype=np.uint8,
            )
            assert len(bits) == len(asked), "oracle dropped answers"
            self._table.put(keys[asked], bits)
            by_first = np.empty_like(bits)
            by_first[order] = bits
            answers[missing] = by_first[inverse.ravel()]
        return answers.tolist()

    def membership_query(self, string: bytes) -> bool:
        return bool(self.membership_queries([string])[0])


#: A key as one row: the digest's two 64-bit halves.
_KEY_ROW = np.dtype([("lo", np.uint64), ("hi", np.uint64)])


def _keys(strings: List[bytes]) -> np.ndarray:
    """``(len(strings), 2)`` digest halves."""
    digests = b"".join(
        hashlib.blake2b(s, digest_size=DIGEST_SIZE).digest() for s in strings
    )
    return np.frombuffer(digests, dtype=np.uint64).reshape(-1, 2)


class _DigestTable:
    """Digests to answer bits, in buckets of ``_BUCKET`` slots, probing on to the
    next bucket when one is full."""

    def __init__(self):
        self._allocate(128)
        self._size = 0

    def __len__(self) -> int:
        return self._size

    def _allocate(self, buckets: int) -> None:
        self._lo = np.zeros((buckets, _BUCKET), dtype=np.uint64)
        self._hi = np.zeros((buckets, _BUCKET), dtype=np.uint64)
        self._values = np.full((buckets, _BUCKET), _EMPTY, dtype=np.uint8)

    def _home(self, keys: np.ndarray) -> np.ndarray:
        # The digest is uniform, so it is a hash already.
        return (keys[:, 0] % np.uint64(len(self._values))).astype(np.int64)

    def get(self, keys: np.ndarray) -> np.ndarray:
        """Each key's answer, or ``_EMPTY`` where it has none."""
        answers = np.full(len(keys), _EMPTY, dtype=np.uint8)
        buckets = self._home(keys)
        probing = np.arange(len(keys))
        while probing.size:
            at = buckets[probing]
            hit = (self._lo[at] == keys[probing, 0, None]) & (
                self._hi[at] == keys[probing, 1, None]
            )
            found = hit.any(axis=1)
            answers[probing[found]] = self._values[at[found], hit[found].argmax(axis=1)]
            # A key not in a bucket with room to spare is in none further on.
            full = (self._values[at] != _EMPTY).all(axis=1)
            probing = probing[~found & full]
            buckets[probing] = (buckets[probing] + 1) % len(self._values)
        return answers

    def put(self, keys: np.ndarray, values: np.ndarray) -> None:
        """Store distinct keys the table does not hold."""
        needed = self._size + len(keys)
        if needed * 10 > self._values.size * _MAX_LOAD_TENTHS:
            buckets = len(self._values)
            while needed * 10 > buckets * _BUCKET * _MAX_LOAD_TENTHS:
                buckets += buckets // 2
            held = self._values != _EMPTY
            old = np.stack([self._lo[held], self._hi[held]], axis=1)
            old_values = self._values[held]
            self._allocate(buckets)
            self._insert(old, old_values)
        self._insert(keys, values)
        self._size = needed

    def _insert(self, keys: np.ndarray, values: np.ndarray) -> None:
        buckets = self._home(keys)
        pending = np.arange(len(keys))
        while pending.size:
            order = np.argsort(buckets[pending], kind="stable")
            pending = pending[order]
            at = buckets[pending]
            # The r-th key headed for a bucket takes its r-th empty slot.
            starts = np.flatnonzero(np.r_[True, at[1:] != at[:-1]])
            rank = np.arange(len(at)) - np.repeat(
                starts, np.diff(np.r_[starts, len(at)])
            )
            empty = self._values[at] == _EMPTY
            nth = (np.cumsum(empty, axis=1) == rank[:, None] + 1) & empty
            fits = nth.any(axis=1)
            slot = nth[fits].argmax(axis=1)
            placed, rows = pending[fits], at[fits]
            self._lo[rows, slot] = keys[placed, 0]
            self._hi[rows, slot] = keys[placed, 1]
            self._values[rows, slot] = values[placed]
            # Past a full bucket, a key goes on to the next.
            pending = pending[~fits]
            buckets[pending] = (buckets[pending] + 1) % len(self._values)
