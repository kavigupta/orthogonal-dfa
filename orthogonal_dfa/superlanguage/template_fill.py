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
from typing import List, Sequence, Tuple

import numpy as np

Pattern = Tuple[int, ...]

FREE = -1  # a hole, to be filled in
_PAD = -2  # not part of this template: it is shorter than others in its batch


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


def _fill_chunk(templates, rngs, tables):
    """Templates are right-aligned, so that sampling right to left starts every one
    in the end-of-string state.
    """
    initial, allowed, shift = tables
    base, num_states = shift.shape
    size = len(templates)
    width = max(len(t) for t in templates)
    grid = np.full((size, width), _PAD, dtype=np.int64)
    start = np.empty(size, dtype=np.int64)
    draws = np.zeros((size, width))
    for row, (template, rng) in enumerate(zip(templates, rngs)):
        start[row] = width - len(template)
        grid[row, start[row] :] = template
        draws[row, start[row] :] = rng.random(len(template))

    # ways[j][row, s] counts the fillings of the columns left of j when s follows
    # column j-1. Rescaled each step: the true counts grow geometrically and
    # overflow, and only ratios within a row are ever read.
    ways = np.ones((width + 1, size, num_states))
    for j in range(width):
        previous, column = ways[j], grid[:, j]
        free = (previous[:, shift] * allowed[None]).sum(axis=1)
        fixed = np.take_along_axis(
            previous, shift[np.where(column >= 0, column, 0)], axis=1
        )
        row = np.where((column == FREE)[:, None], free, fixed)
        row = np.where((column == _PAD)[:, None], previous, row)
        peak = np.max(row, axis=1)[:, None]
        ways[j + 1] = np.divide(row, peak, out=row.copy(), where=peak > 0)

    state = np.full(size, initial, dtype=np.int64)
    out = np.zeros((size, width), dtype=np.int64)
    for j in range(width - 1, -1, -1):
        column = grid[:, j]
        live = column != _PAD
        weights = (
            np.take_along_axis(ways[j], shift[:, state].T, axis=1) * allowed[:, state].T
        )
        running = np.cumsum(weights, axis=1)
        # Only holes are constrained, so only they can run out of choices. Asking
        # this of a fixed column would reject templates that are perfectly fillable.
        assert (running[column == FREE, -1] > 0).all(), "template has no legal filling"
        picked = (running <= (draws[:, j] * running[:, -1])[:, None]).sum(axis=1)
        chosen = np.where(
            column == FREE, np.clip(picked, 0, base - 1), np.where(live, column, 0)
        )
        out[:, j] = chosen
        state = np.where(live, shift[chosen, state], state)
    return [out[row, start[row] :].tolist() for row in range(size)]


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

    def fill(self, template: Sequence[int], rng: np.random.Generator) -> List[int]:
        """Prefer fill_many for more than one; the per-template pass amortizes."""
        return self.fill_many([template], [rng])[0]

    def fill_many(
        self,
        templates: Sequence[Sequence[int]],
        rngs: Sequence[np.random.Generator],
    ) -> List[List[int]]:
        """Draws each result uniformly over its template's legal fillings, by
        weighting a hole's choices by how many ways the rest can then be filled.
        Drawing evenly instead skews toward the constrained continuations.

        One rng per template, spent one draw per position, so a result does not
        depend on what it was batched with.
        """
        assert len(templates) == len(rngs), "need one rng per template"
        if not templates:
            return []
        _, _, shift = self._tables
        width = max(len(t) for t in templates)
        # Batch to keep the backward pass's (width, chunk, num_states) table near
        # ~64MB. A single template wider than that is still passed through whole.
        chunk = int(np.clip(8_000_000 // max((width + 1) * shift.shape[1], 1), 1, 4096))

        out: List[List[int]] = []
        for lo in range(0, len(templates), chunk):
            out.extend(
                _fill_chunk(
                    templates[lo : lo + chunk], rngs[lo : lo + chunk], self._tables
                )
            )
        return out

    @property
    def _tables(self):
        return _transfer_tables(self.forbidden, self.base_alphabet_size)
