"""The prefix populations a round holds, and drawing more of one.

The rounds carry these across each other, and the family search in `cluster`
reads and grows them, so they live apart from the round loop that fills them.
"""

from .mask_table import UNIFORM
from .prefix_sources import UniformSource, draw_many


class PoolState:
    """The pool state carried across rounds: the initial uniform sample (kept in
    the representative set every round so global calibration stays anchored to the
    sampling distribution even if the per-state sample is skewed), and the
    populations the rounds have made, with a ``seen`` set to dedup the boundary
    strings across them."""

    def __init__(self, uniform):
        self.uniform = list(uniform)
        self.held = {}
        #: What draws more of each population, for the round that asks.
        self.sources = {}
        self.seen = set()
        #: Boundary populations named so far, which is what numbers them.
        self.named = 0
        #: The one this round is filling, or None before it strands anything.
        self.harvesting = None
        #: Labels the table holds, so a round retires what it does not renew.
        self.published = set()

    def retire_states(self) -> None:
        """Forget last round's state populations: this round's states are the
        ones there are, and a leaf it does not have is not one to go on
        publishing."""
        for stale in [label for label in self.held if label[0] == "state"]:
            self.held.pop(stale)
            self.sources.pop(stale, None)

    def harvest(self) -> list:
        """This round's boundary population, named on the first string to reach
        it."""
        if self.harvesting is None:
            self.named += 1
            self.harvesting = ("boundary", self.named)
            self.held[self.harvesting] = []
        return self.held[self.harvesting]


def grow_population(pst, state, label) -> bool:
    """Draw more prefixes for one population, or retire it, table and all, when
    nothing draws for it any more."""
    if label == UNIFORM:
        pst.sample_more_prefixes()
        return True
    source = state.sources.get(label)
    if source is None or not source.worth_drawing():
        # Forgotten rather than held aside, so a later round that strands one
        # of these again can pool it behind a source that does draw.
        state.seen.difference_update(state.held.pop(label, ()))
        state.sources.pop(label, None)
        pst.table.drop_population(label)
        return False
    # As many as the uniform draw this stands in for adds.
    drawn = [source.draw() for _ in range(pst.config.num_addtl_prefixes)]
    state.held.setdefault(label, []).extend(drawn)
    pst.table.add_prefixes(sorted(set(drawn)), population=label)
    return True


def population_labels(state) -> list:
    """This round's populations, and the uniform pool the table keeps across
    rounds."""
    return [UNIFORM, *state.held]


def prefixes_for_split(pst, state, label, wanted: int) -> list:
    """Prefixes for one population, to read the split on and not to keep.

    Empty where nothing draws for it, so the split goes ahead without it.
    """
    source = UniformSource(pst) if label == UNIFORM else state.sources.get(label)
    # Not proved here: proving costs a round's worth of probes, which the
    # split is not worth.
    if source is None or not source.proven:
        return []
    return draw_many(source, wanted)
