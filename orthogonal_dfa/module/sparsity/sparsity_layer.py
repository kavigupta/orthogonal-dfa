from abc import ABC

from torch import nn

from dconstruct import construct

from .enforcer import Sparsity, enforce_sparsity_per_channel_types

class SparseLayerWithBatchNorm(Sparsity):
    """
    Wraps a sparsity enforcer, adding a batch norm layer before it.
    """

    def __init__(
        self,
        underlying_sparsity_spec,
        starting_sparsity,
        channels,
        affine,
        input_dimensions=2,
    ):
        super().__init__(starting_sparsity, channels)
        self.input_dimensions = input_dimensions
        self.batch_norm = {1: nn.BatchNorm1d, 2: nn.BatchNorm2d}[input_dimensions](
            channels, affine=affine
        )
        self.underlying_enforcer = construct(
            sparsity_types(),
            underlying_sparsity_spec,
            starting_sparsity=starting_sparsity,
            channels=channels,
        )

    def notify_sparsity(self):
        super().notify_sparsity()
        self.underlying_enforcer.sparsity = self.sparsity

    def forward(self, x, disable_relu=False):
        if getattr(self, "input_dimensions", 2) == 1:
            x = x.transpose(1, 2)
        x = self.batch_norm(x)
        x = self.underlying_enforcer(x, disable_relu=disable_relu)
        if getattr(self, "input_dimensions", 2) == 1:
            x = x.transpose(1, 2)
        return x


class EnforceSparsityPerChannel2D(Sparsity):
    """
    Like EnforceSparsityPerChannel, but handling 2d inputs.
    """

    def __init__(
        self,
        starting_sparsity,
        channels,
        momentum=0.1,
        *,
        enforce_sparsity_per_channel_spec=dict(type="EnforceSparsityPerChannel"),
    ):
        super().__init__(starting_sparsity, channels)
        self.channels = channels
        self.underlying_enforcer = construct(
            enforce_sparsity_per_channel_types(),
            enforce_sparsity_per_channel_spec,
            starting_sparsity=starting_sparsity,
            channels=channels,
            momentum=momentum,
        )

    def notify_sparsity(self):
        super().notify_sparsity()
        self.underlying_enforcer.sparsity = self.sparsity

    def forward(self, x, disable_relu=False):
        N, C, H, W = x.shape
        assert C == self.channels, str((C, self.channels))
        # x : (N, C, H, W)
        x = x.permute(0, 2, 3, 1)
        # x : (N, H, W, C)
        x = x.reshape(-1, self.channels)
        # x : (N * H * W, C)
        assert x.shape == (N * H * W, C)
        x = self.underlying_enforcer(x, disable_relu=disable_relu)
        # x : (N * H * W, C)
        x = x.reshape(N, H, W, self.channels)
        # x : (N, H, W, C)
        x = x.permute(0, 3, 1, 2)
        # x : (N, C, H, W)
        assert x.shape == (N, C, H, W)
        return x


class EnforceSparsityPerChannel1D(EnforceSparsityPerChannel2D):
    def forward(self, x, **kwargs):
        x = x.unsqueeze(2)
        x = super().forward(x, **kwargs)
        x = x.squeeze(2)
        return x


def sparsity_types():
    return dict(
        EnforceSparsityPerChannel1D=EnforceSparsityPerChannel1D,
        EnforceSparsityPerChannel2D=EnforceSparsityPerChannel2D,
        SparseLayerWithBatchNorm=SparseLayerWithBatchNorm,
    )
