"""
A memoized, batched membership wrapper for arbitrary strings.

Wraps an oracle transparently -- the ``membership_queries`` interface is
unchanged, so it drops in wherever a raw oracle is used -- but a string is
queried at most once no matter how many callers (or how many prefix/suffix
cells) reference it.  The oracle is deterministic per string, so a cached bit is
exactly what a fresh query would return; misses are issued as one batched call.
"""

from typing import List


class MemoizedOracle:
    """Membership of arbitrary strings, memoized per string and batched."""

    def __init__(self, oracle):
        self._oracle = oracle
        self.alphabet_size = oracle.alphabet_size
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

    # Oracle drop-in: same signature as the wrapped oracle's batched query.
    membership_queries = membership
