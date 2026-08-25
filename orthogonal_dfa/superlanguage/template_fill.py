"""
A template is a list of base symbols with holes punched in it, and a filling is legal
when no forbidden pattern starts at a hole; patterns starting at a fixed position are
none of our business, so a caller that wants one to survive puts it in the template.

Legality at a hole turns only on the next max_pattern_length - 1 symbols, so that run
is all a filler has to carry along, and the tables here say how it moves and what it
permits.
"""

from dataclasses import dataclass
from functools import lru_cache
from typing import Tuple

import numpy as np

Pattern = Tuple[int, ...]


@lru_cache(maxsize=None)
def _transfer_tables(forbidden: Tuple[Pattern, ...], base: int):
    """Whether a hole may take a symbol depends on the symbols after it, so a state
    here is the next max_pattern_length - 1 of them, packed in radix base + 1. The
    extra digit is a sentinel for running off the end, which matches no pattern.

    shift[c, state] is the state one position further left after taking c, and
    allowed[state, c] whether a hole may take c there.
    """
    w = max((len(p) for p in forbidden), default=1) - 1
    radix = base + 1
    num_states = radix**w

    shift = np.array(
        [
            [
                c + radix * (state % radix ** (w - 1)) if w >= 1 else 0
                for state in range(num_states)
            ]
            for c in range(base)
        ],
        dtype=np.int64,
    )

    allowed = np.ones((num_states, base), dtype=bool)
    for state in range(num_states):
        following = [(state // radix**i) % radix for i in range(w)]
        for pattern in forbidden:
            if following[: len(pattern) - 1] == list(pattern[1:]):
                allowed[state, pattern[0]] = False

    # Cached and shared between fillers over the same patterns.
    allowed.flags.writeable = False
    shift.flags.writeable = False
    initial = num_states - 1 if w else 0  # all sentinels: nothing follows the end
    return initial, allowed, shift


@dataclass(frozen=True)
class TemplateFiller:
    """Nothing here reads the patterns as anything but strings to keep out of the
    holes, so duplicate or prefix-related ones are fine.
    """

    forbidden: Tuple[Pattern, ...]
    base_alphabet_size: int

    def __post_init__(self):
        # frozen, so the caller's lists would survive as unhashable fields and only
        # fail once _tables tried to key the cache on them
        object.__setattr__(self, "forbidden", tuple(tuple(p) for p in self.forbidden))
        assert self.base_alphabet_size >= 1, "base alphabet must be non-empty"
        for pattern in self.forbidden:
            assert len(pattern) >= 1, "patterns must be non-empty"
            assert all(
                0 <= c < self.base_alphabet_size for c in pattern
            ), f"pattern {pattern} has symbols outside the base alphabet"

    @property
    def every_context_is_fillable(self) -> bool:
        """Whether every context of that many symbols leaves a hole something to
        take. Sufficient for every template to be fillable, not necessary: a context
        no template forces the sampler into still counts against it here.
        """
        _, allowed, _ = self._tables
        return bool(allowed.any(axis=1).all())

    @property
    def _tables(self):
        return _transfer_tables(self.forbidden, self.base_alphabet_size)
