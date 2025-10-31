import torch
from torch import nn

from orthogonal_dfa.utils.bases import parse_nucleotides_as_one_hot_logit
from orthogonal_dfa.utils.probability import ZeroProbability


class TorchPSAMs(nn.Module):
    """
    Represents a set of Position-Specific Affinity Matrices (PSAMs) using convolutional layers.

    :param conv_logit: nn.Parameter of shape (num_psams, channels, two_r + 1, two_r + 1),
        which represents log(sigmoid(probabilities)) for each PSAM.s
    """

    @classmethod
    def from_literal_strings(cls, *psam_strings, zero_prob: ZeroProbability):
        """
        Creates a TorchPSAMs instance that matches the precise patterns defined in the given literal strings.
        """
        return cls(
            conv_logit=torch.stack(
                [
                    parse_nucleotides_as_one_hot_logit(psam_string, zero_prob).permute(
                        1, 0
                    )
                    for psam_string in psam_strings
                ],
                dim=0,
            )
        )

    @classmethod
    def create(cls, two_r, channels, num_psams):
        """
        Creates a TorchPSAMs instance with randomly initialized parameters.

        :param two_r: int, the radius of the PSAM (the PSAM will have size two_r + 1).
            This can be any positive integer, even an odd one.
        :param channels: int, the number of channels in the input data.
        :param num_psams: int, the number of PSAMs to represent.
        :return: TorchPSAMs instance with randomly initialized parameters.
        """
        conv_logit = torch.randn((num_psams, channels, two_r + 1))
        return cls(conv_logit)

    def __init__(self, conv_logit):
        super().__init__()

        assert (
            len(conv_logit.shape) == 3
        ), f"Expected conv_logit to have 3 dimensions, got {conv_logit.shape}"

        self.two_r = conv_logit.shape[2] - 1
        self.channels = conv_logit.shape[1]
        self.conv_logit = nn.Parameter(conv_logit)
        self.log_sigoid = nn.LogSigmoid()

    @property
    def conv_logprob(self):
        """
        Computes the log probabilities of the PSAMs by applying the log sigmoid function to conv_logit.
        """
        return self.log_sigoid(self.conv_logit)

    @property
    def sequence_logos(self):
        """
        Computes the sequence logos for the PSAMs. These are normalized log probabilities, i.e.,
            the log probability for each base above or below the mean probability at that position.

        :return: numpy array of shape (num_psams, channels, two_r + 1) representing the sequence logos.
        """
        actual_psams = self.conv_logprob.transpose(1, 2).detach().cpu().numpy()
        logo = actual_psams - actual_psams.mean(-1, keepdims=True)
        return logo

    def forward(self, x):
        assert len(x.shape) == 3
        # x: (batch_size, length, channels)
        assert torch.isfinite(self.conv_logprob).all()
        # pylint: disable=not-callable
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


def conditional_cascade_log_probs(x, axis):
    """
    Given a list of probabilities `x`, treats each element along axis as a step in a cascade, where the
    probability of reaching the end is the product of the probabilities at each step. Add an additional
    element to the front representing the probability of not having stopped at any previous step.

    i.e., computes
          q_0 = prod_{j < n} (1 - p_j)
          q_{i+1} = p_i * prod_{j < i} (1 - p_j)
    """
    assert torch.isfinite(x).all()
    cascaded_product_flip = torch.cumsum(flip_log_probs(x), axis)
    cascaded_product_flip_shift = torch.roll(cascaded_product_flip, 1, axis)
    cascaded_product_flip_shift.index_fill_(axis, torch.tensor(0, device=x.device), 0.0)
    y = x + cascaded_product_flip_shift
    y = torch.cat([cascaded_product_flip.select(axis, -1).unsqueeze(axis), y], dim=axis)
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
