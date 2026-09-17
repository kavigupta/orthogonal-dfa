"""A stand-in for the prefix/suffix table, for tests about clustering."""

import numpy as np


class Table:
    """``masks[suffix, prefix]``, every prefix representative, and
    ``populations`` (``label -> mask``) over them."""

    def __init__(self, masks, populations):
        self._masks = masks
        self._populations = populations
        self.representative = np.ones(masks.shape[1], dtype=bool)

    def fully_observed(self):
        return np.arange(self._masks.shape[0])

    def observed_masks(self, rows, prefixes):
        return self._masks[np.asarray(rows)][:, prefixes]

    def population_masks(self):
        return self._populations
