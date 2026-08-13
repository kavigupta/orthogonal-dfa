from typing import List

from .structures import Oracle


class MemoizedOracle(Oracle):
    """Membership of arbitrary strings, memoized per string and batched."""

    def __init__(self, oracle):
        self._oracle = oracle
        self._cache: dict = {}

    @property
    def alphabet_size(self) -> int:
        return self._oracle.alphabet_size

    def membership_queries(self, strings: List[List[int]]) -> List[int]:
        cache = self._cache
        keys = [tuple(s) for s in strings]
        misses = list(dict.fromkeys(k for k in keys if k not in cache))
        if misses:
            answers = self._oracle.membership_queries([list(k) for k in misses])
            assert len(answers) == len(misses), "oracle dropped answers"
            for key, bit in zip(misses, answers):
                cache[key] = int(bit)
        return [cache[k] for k in keys]

    def membership_query(self, string: List[int]) -> bool:
        return bool(self.membership_queries([string])[0])
