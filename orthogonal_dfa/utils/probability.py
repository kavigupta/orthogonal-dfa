import math
from dataclasses import dataclass


@dataclass
class ZeroProbability:
    """Class representing zero probability in log-space."""

    probability: float

    def __post_init__(self):
        assert 0 < self.probability < 1, "Value must be in the range (0, 1)"

    @property
    def log_probability(self) -> float:
        """Returns the log probability."""
        return math.log(self.probability)

    @property
    def logit_probability(self) -> float:
        """Returns the logit of the probability."""
        return math.log(self.probability) - math.log(1 - self.probability)
