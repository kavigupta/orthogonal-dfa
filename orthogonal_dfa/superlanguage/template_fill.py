"""
Over a base alphabet A = {0, ..., base - 1}, a template is a string t over A + {hole},
and a filling of it is an s in A^|t| agreeing with t off the holes:

    s[j] = t[j]   wherever t[j] != hole.

Such an s is legal when no forbidden pattern starts at a hole:

    s[j : j + |p|] != p   for every hole j and every p in forbidden.

(note that this only applies to strings that start in hole locations j).

Legality at j therefore constrains s only through s[j] and the w symbols after it,

    w = max |p| - 1,

and the tables here give the transitions on that window and the symbols it permits.
"""

from dataclasses import dataclass
from functools import lru_cache
from typing import Tuple

import numpy as np

Pattern = Tuple[int, ...]


@lru_cache(maxsize=None)
def _transfer_tables(forbidden: Tuple[Pattern, ...], base: int):
    """A state is the window of w symbols following a position j, packed in radix
    r = base + 1 with the nearest symbol as the least significant digit:

        state(j) = sum_{0 <= i < w} r^i * s[j + 1 + i].

    The digit `base` is a sentinel for j + 1 + i running past the end of the string.
    It is not in A, so it equals no symbol of any pattern.

    shift[c, state] is state(j - 1) once s[j] = c is chosen: c becomes digit 0 and
    the far symbol falls off,

        shift[c, state] = c + r * (state mod r^(w - 1)).

    allowed[state, c] is False exactly when c would start a forbidden pattern at j,

        p[0] = c and p[1:] = s[j + 1 : j + |p|]   for some p in forbidden.
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
