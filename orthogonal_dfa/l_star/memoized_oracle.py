"""
A memoized, batched membership wrapper for off-grid strings.

The prefix-suffix grid (MaskTable) holds membership for (prefix, suffix) cells.
A sift touches fresh strings the grid never holds -- one per tree node -- so
direct-L* memoizes those here instead, keeping the grid free of transient rows.
The oracle is deterministic per string, so a cached bit is exactly what a fresh
query would return; misses are issued as one batched call.
"""

from typing import List


class MemoizedOracle:
    """Membership of arbitrary strings, memoized per string and batched."""

    def __init__(self, oracle):
        self._oracle = oracle
        self._cache: dict = {}

    def membership(self, strings) -> List[int]:
        cache = self._cache
        keys = [tuple(s) for s in strings]
        misses = list(dict.fromkeys(k for k in keys if k not in cache))
        if misses:
            answers = self._oracle.membership_queries([list(k) for k in misses])
            assert len(answers) == len(misses), "oracle dropped answers"
            for key, bit in zip(misses, answers):
                cache[key] = int(bit)
        return [cache[k] for k in keys]
