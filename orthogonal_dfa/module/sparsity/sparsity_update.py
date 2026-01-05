from abc import ABC, abstractmethod
from dataclasses import dataclass


class SparsityUpdateOptimizer(ABC):
    """
    Not a torch optimizer object, but it does support the basic zero_grad/step interface.
    """

    def __init__(self):
        pass

    @abstractmethod
    def update_sparsity(self, sparsity_value, update_sparsity_value, step, acc):
        """
        Updates the model's sparsity given the current step and accuracy information
        """
        pass


class NoopSUO(SparsityUpdateOptimizer):

    def update_sparsity(self, sparsity_value, update_sparsity_value, step, acc):
        """
        Does nothing
        """
        pass


@dataclass
class LinearThresholdAdaptiveSUO(SparsityUpdateOptimizer):
    threshold: float
    minimal_threshold: float
    maximal_threshold: float
    threshold_decrease_per_iter: float
    step: float
    minimal_update_frequency: float
    information_multiplier: float

    @classmethod
    def of(
        cls,
        initial_threshold,
        minimal_threshold,
        maximal_threshold,
        threshold_decrease_per_iter,
        minimal_update_frequency,
        information_multiplier,
        initial_step=0,
    ):
        """
        The Linear Threshold Adaptive SUO uses an algorithm where a current accuracy threshold
            is mantained, and slowly decreased over time. If the model exceeds the threshold,
            the threshold is increased to match the model and the model's information is reduced
            (information + sparsity = 1).

        Arguments:
            initial_threshold: the initial accuracy threshold [0-1]
            minimal_threshold: the minimal accuracy threshold to use [0-1]
            threshold_decrease_per_iter: the amount you decrease the threshold every iteration
            minimal_update_frequency: the minimal number of iterations between updates
            information_multiplier: how much to change the amount of information (1 - sparsity)
        """
        return cls(
            threshold=initial_threshold,
            minimal_threshold=minimal_threshold,
            maximal_threshold=maximal_threshold,
            threshold_decrease_per_iter=threshold_decrease_per_iter,
            step=initial_step,
            minimal_update_frequency=minimal_update_frequency,
            information_multiplier=information_multiplier,
        )

    def update_sparsity(self, sparsity_value, update_sparsity_value, step, acc):
        time_since_last = step - self.step
        assert time_since_last >= 0
        if time_since_last < self.minimal_update_frequency:
            return
        self.step = step
        self.threshold -= self.threshold_decrease_per_iter * time_since_last
        self.threshold = max(self.threshold, self.minimal_threshold)
        self.threshold = min(self.threshold, self.maximal_threshold)
        print(f"Accuracy: {acc:.2%}; Threshold: {self.threshold:.2%}")
        if acc > self.threshold:
            print(
                f"Originally using information (1 - sparsity) = {1 - sparsity_value:.10%}"
            )
            sparsity_value = 1 - (1 - sparsity_value) * self.information_multiplier

            update_sparsity_value(sparsity_value)
            print(
                f"Now        using information (1 - sparsity) = {1 - sparsity_value:.10%}"
            )
        self.threshold = max(self.threshold, acc)
        return sparsity_value


def suo_types():
    return dict(
        NoopSUO=NoopSUO, LinearThresholdAdaptiveSUO=LinearThresholdAdaptiveSUO.of
    )
