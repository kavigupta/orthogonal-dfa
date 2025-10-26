from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import cached_property

import numpy as np
import torch

from orthogonal_dfa.spliceai.load_model import load_lssi


class DataLoader(ABC):
    @abstractmethod
    def next(self, iteration):
        pass


@dataclass
class SpliceDataLoader(DataLoader):
    """
    DataLoader that generates random inputs and labels them using a pretrained LSSI model.
    """

    which_one: str
    seed: int
    count: int
    length: int
    model_threshold: float = -10

    @cached_property
    def load_model(self):
        mod = load_lssi(self.which_one, self.seed)
        mod.conv_layers[0].clipping = "natural"
        return mod

    @property
    def channel(self):
        return {"donor": 2, "acceptor": 1}[self.which_one]

    def next(self, iteration):
        rng = np.random.default_rng(iteration)
        random_inputs = rng.integers(0, 4, size=(self.count, self.length))
        random_inputs = torch.tensor(np.eye(4, dtype=np.float32)[random_inputs]).cuda()
        with torch.no_grad():
            yp = self.load_model(random_inputs).log_softmax(-1)
            yp = yp[:, 0, self.channel]
            yp = yp.cpu()
        return random_inputs, yp >= self.model_threshold
