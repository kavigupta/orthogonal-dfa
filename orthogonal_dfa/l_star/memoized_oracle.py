import hashlib
from typing import List

from .structures import Oracle

#: Digest bytes per key.  At 128 bits a collision -- which would silently hand
#: one string another's answer -- runs at ~1e-23 for 10^8 distinct strings.
DIGEST_SIZE = 16


class MemoizedOracle(Oracle):
    """Membership of arbitrary strings, memoized per string and batched.

    Keyed by a digest of the string rather than the string: at low signal
    strength this cache holds tens of millions of entries, and a digest is a
    third cheaper per entry than the string.
    """

    def __init__(self, oracle):
        self._oracle = oracle
        self._cache: dict = {}

    @property
    def alphabet_size(self) -> int:
        return self._oracle.alphabet_size

    def membership_queries(self, strings: List[bytes]) -> List[int]:
        cache = self._cache
        keys = [_key(s) for s in strings]
        # Valued by the string the key came from: a digest cannot be inverted,
        # so this is what the underlying oracle gets handed.
        misses = {k: s for k, s in zip(keys, strings) if k not in cache}
        if misses:
            answers = self._oracle.membership_queries(list(misses.values()))
            assert len(answers) == len(misses), "oracle dropped answers"
            for key, bit in zip(misses, answers):
                cache[key] = int(bit)
        return [cache[k] for k in keys]

    def membership_query(self, string: bytes) -> bool:
        return bool(self.membership_queries([string])[0])


def _key(string: bytes) -> int:
    """An int rather than the digest bytes: same object size at 128 bits, and
    smaller than the equivalent ``bytes``."""
    return int.from_bytes(
        hashlib.blake2b(string, digest_size=DIGEST_SIZE).digest(), "big"
    )
