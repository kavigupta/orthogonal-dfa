from abc import ABC
from dataclasses import dataclass
import torch
from torch import nn

from dconstruct import construct


class Sparsity(ABC, nn.Module):

    def __init__(self, starting_sparsity, channels):
        super().__init__()
        self._sparsity = starting_sparsity
        self.channels = channels

    def notify_sparsity(self):
        pass

    @property
    def sparsity(self):
        return self._sparsity

    @property
    def density(self):
        return 1 - self.sparsity

    @sparsity.setter
    def sparsity(self, sparsity):
        self._sparsity = sparsity
        self.notify_sparsity()


class EnforceSparsityPerChannel(Sparsity):
    """
    Enforces sparsity across the last index in the given tensor, where C is num_channels.

    Takes an input of size (N, C) and enforces sparsities for each channel C independently.

    Arguments
        sparsity: inital sparsity to enforce
        num_channels: the number of channels C to enforce sparsity on
        momentum: the momentum of the collected percentile statistic. The default value of 0.1 indicates that
            at each batch update we use 90% existing thresholds and 10% the percentile statistic thresholds.
    """

    def __init__(
        self,
        starting_sparsity,
        channels,
        momentum=0.1,
    ):
        super().__init__(starting_sparsity, channels)
        self.thresholds = torch.nn.parameter.Parameter(
            torch.zeros(channels), requires_grad=False
        )
        self.momentum = momentum

    def update_with_batch(self, x):
        N, _ = x.shape
        to_drop = max(1, int(N * self.sparsity))
        thresholds, _ = torch.kthvalue(x, k=to_drop, dim=0)

        self.thresholds.data = (
            self.thresholds.data * (1 - self.momentum) + thresholds * self.momentum
        )

    def forward(self, x, disable_relu=False):
        _, C = x.shape

        assert [C] == list(self.thresholds.shape), f"{[C]} != {self.thresholds.shape}"
        if self.training:
            self.update_with_batch(x)
        x = x - self.thresholds
        if disable_relu:
            return x
        return torch.nn.functional.relu(x)


class EnforceSparsityPerChannelAccumulated(EnforceSparsityPerChannel):
    """
    Like EnforceSparsityPerChannel, but accumulates the values across batches.
    """

    def __init__(self, *args, accumulation_stop_strategy, **kwargs):
        super().__init__(*args, **kwargs)
        self.accumulation_stop_strategy = construct(
            accumulation_stop_strategy_types(),
            accumulation_stop_strategy,
        )
        self.accumulated_batches = []
        self.num_elements = 0

    def update_with_batch(self, x):
        self.accumulated_batches.append(x.detach())
        self.num_elements += x.shape[0]
        if self.accumulation_stop_strategy.should_stop(self):
            x = torch.cat(self.accumulated_batches, dim=0)
            super().update_with_batch(x)
            self.accumulated_batches = []
            self.num_elements = 0


@dataclass
class StopAtFixedNumberElements:
    num_elements: int

    def should_stop(self, enforcer):
        return enforcer.num_elements >= self.num_elements


@dataclass
class StopAtFixedNumberMotifs:
    num_motifs: int

    def should_stop(self, enforcer):
        return enforcer.num_elements * enforcer.density >= self.num_motifs


def accumulation_stop_strategy_types():
    return dict(
        StopAtFixedNumber=StopAtFixedNumberElements,
        StopAtFixedNumberMotifs=StopAtFixedNumberMotifs,
    )


def enforce_sparsity_per_channel_types():
    return dict(
        EnforceSparsityPerChannel=EnforceSparsityPerChannel,
        EnforceSparsityPerChannelAccumulated=EnforceSparsityPerChannelAccumulated,
    )
