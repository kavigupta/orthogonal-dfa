from abc import ABC, abstractmethod
from typing import Optional


class NotifiableByLoss(ABC):
    @abstractmethod
    def notify_epoch_loss(
        self, epoch_idx: int, epoch_loss: list[float]
    ) -> Optional[object]:
        """
        Notify the module of the loss at the end of an epoch.

        :param epoch_idx: int, the index of the epoch.
        :param epoch_loss: List[float], the loss values for the epoch.
        :return: Optional information for logging or further processing.
        """
