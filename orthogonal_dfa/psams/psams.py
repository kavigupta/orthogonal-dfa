import torch
from torch import nn


class TorchPSAMs(nn.Module):
    """
    Represents a set of Position-Specific Affinity Matrices (PSAMs) using convolutional layers.

    :param two_r: int, the radius of the PSAM (the PSAM will have size (two_r + 1) x (two_r + 1)).
        This can be any positive integer, even an odd one.
    :param channels: int, the number of channels in the input data.
    :param num_psams: int, the number of PSAMs to represent.

    :field conv_logit: nn.Parameter of shape (num_psams, channels, two_r + 1, two_r + 1),
        which represents log(sigmoid(probabilities)) for each PSAM.s
    """

    def __init__(self, two_r, channels, num_psams):
        super().__init__()
        self.two_r = two_r
        self.channels = channels
        self.conv_logit = nn.Parameter(torch.randn((num_psams, channels, two_r + 1)))
        self.log_sigoid = nn.LogSigmoid()

    @property
    def conv_logprob(self):
        """
        Computes the log probabilities of the PSAMs by applying the log sigmoid function to conv_logit.
        """
        return self.log_sigoid(self.conv_logit)

    def forward(self, x):
        # x: (batch_size, length, channels)
        assert torch.isfinite(self.conv_logit).all()
        assert torch.isfinite(self.conv_logprob).all()
        return nn.functional.conv1d(
            x.permute(0, 2, 1),  # (batch_size, channels, length)
            self.conv_logprob,  # (num_psams, channels, two_r + 1)
        ).permute(
            0, 2, 1
        )  # (batch_size, length - two_r, num_psams)


def union_log_probs(x, axis):
    """
    Unions log probabilities along a given axis, assuming they are independent. I.e., along the given axis,
    compute the aggregation

    y = log(1 - prod_i (1 - exp(x_i)))

    This is accomplished by computing

    y = log1p(-exp(sum_i log1p(-exp(x_i))))

    log probability outcomes are clipped at -100 to avoid -inf values.
    """
    assert torch.isfinite(x).all()
    y = flip_log_probs(torch.sum(flip_log_probs(x), axis))
    assert torch.isfinite(y).all()
    return y


def flip_log_probs(x):
    """
    Flips log probabilities to represent the complementary event. I.e., computes log(1 - exp(x)).

    Makes sure to avoid error cases by clipping log the initial probabilities at -0.001
    """
    x = torch.clamp(x, max=-1e-7)
    y = torch.log1p(-torch.exp(x))
    assert torch.isfinite(y).all()
    return y


class UnionedPSAMs(nn.Module):
    """
    Represents a union of multiple PSAMs by combining their log probabilities.

    :param psams_list: list of TorchPSAMs instances to be unioned.
    """

    def __init__(self, psams):
        super().__init__()
        self.psams = psams

    def forward(self, x):
        return union_log_probs(self.psams(x), axis=-1)
