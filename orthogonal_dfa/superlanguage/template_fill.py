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
    """
    Effectively a DFA that traverses through windows of size w, backwards
    through the string, telling you when prepending a specific symbol would start
    a forbidden pattern.

    Let a state be the window of w symbols following a position j.
        We pack this into a radix r = base + 1 number so we can handle running past
        the end of the string.

    initial is the state at the end of the string, which is all sentinels.

    shift[c, state] is state(j - 1) once s[j] = c is chosen.

    allowed[c, state] is False exactly when c would start a forbidden pattern at j.
    """
    w = max((len(p) - 1 for p in forbidden), default=0)
    radix = base + 1
    num_states = radix**w

    if w == 0:
        # the window is empty, so there is one state and nothing to shift out of it
        shift = np.zeros((base, 1), dtype=np.int64)
    else:
        shift = np.array(
            [
                [c + radix * (state % radix ** (w - 1)) for state in range(num_states)]
                for c in range(base)
            ],
            dtype=np.int64,
        )

    allowed = np.ones((base, num_states), dtype=bool)
    for state in range(num_states):
        following = [(state // radix**i) % radix for i in range(w)]
        for pattern in forbidden:
            if following[: len(pattern) - 1] == list(pattern[1:]):
                allowed[pattern[0], state] = False

    # Cached and shared between fillers over the same patterns.
    allowed.flags.writeable = False
    shift.flags.writeable = False
    initial = num_states - 1 if w else 0  # all sentinels: nothing follows the end
    return initial, allowed, shift


@dataclass(frozen=True)
class TemplateFiller:

    #: forbidden need not be prefix-free, or even duplicate-free: each pattern is
    #: an independent ban, and allowed is their conjunction.
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
        """
        Whether every context leaves a hole some symbol it could take. Sufficient
        for every template to be fillable, not necessary: a context no template ever
        puts a hole in front of still counts against it here.
        """
        _, allowed, _ = self._tables
        return bool(allowed.any(axis=0).all())

    @property
    def _tables(self):
        return _transfer_tables(self.forbidden, self.base_alphabet_size)
