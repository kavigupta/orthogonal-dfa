import torch
from torch import nn


class Monotonic1DFixedRange(nn.Module):
    """
    Represents a 1D monotonic function using a piecewise linear approximation.

    Breaks are placed uniformly across the input range, from -input_range to +input_range, with
    the number of breaks specified by num_input_breaks. I.e., for input_range=1.0 and num_input_breaks=5,
    the breaks will be at [-1.0, -0.5, 0.0, 0.5, 1.0].

    The output will always map [-input_range, +input_range] to itself, i.e., the output will also be in the range
    [-input_range, +input_range], and values outside this range will have different linear functions applied to them.

    This is always continuous, piecewise linear, and strictly monotonic.

    :param input_range: float, the range of the input and output values.
    :param num_input_breaks: int, the number of breaks to use in the piecewise linear function.

    :field inv_softplus_slopes: nn.Parameter of shape (num_input_breaks + 1,). softplus(inv_softplus_slopes)
        gives a tensor proportional to the slopes of the piecewise linear segments. There are
        `num_input_breaks + 1` segments because we include segments for inputs less than the first break
        and greater than the last break.
    """

    def __init__(self, input_range: float, num_input_breaks: int):
        super().__init__()
        self.input_range = input_range
        self.num_input_breaks = num_input_breaks

        self.inv_softplus_slopes = nn.Parameter(torch.randn(num_input_breaks + 1))
        self.softplus = nn.Softplus()

    @property
    def dx(self):
        """
        Computes the distance between consecutive input breaks.
        """
        return (2.0 * self.input_range) / (self.num_input_breaks - 1)

    @property
    def slopes(self):
        """
        Computes the slopes of the piecewise linear segments by applying the softplus function to
        inv_softplus_slopes. This ensures that all slopes are positive, maintaining monotonicity.
        """
        slopes_prop = self.softplus(self.inv_softplus_slopes)
        dy_internal = slopes_prop[1:-1] * self.dx
        dy_internal_total = dy_internal.sum()
        correction = (2.0 * self.input_range) / dy_internal_total
        slopes = slopes_prop * correction
        return slopes

    @property
    def values_at_breaks(self):
        """
        Computes the values at each break point based on the slopes of the segments.

        v_i = v_0 + sum_{j < i} slope_j * dx
            = -input_range + sum_{j < i} slopes[j] * dx
            = -input_range + dx * sum_{j < i} slopes[j]
        v = -input_range + dx * [0, *cumsum(slopes)[:-1]]
        """

        cumsum_slopes = torch.cumsum(self.slopes[1:-1], dim=0)
        cumsum_slopes = torch.cat(
            [torch.zeros(1, device=cumsum_slopes.device), cumsum_slopes], dim=0
        )
        values = -self.input_range + self.dx * cumsum_slopes
        return values

    def forward(self, x):
        """
        Applies the monotonic piecewise linear function to the input tensor x.

        :param x: torch.Tensor of arbitrary shape, the input values to transform.
        :return: torch.Tensor of the same shape as x, the transformed values.
        """
        # which to use.
        which = torch.clamp(
            ((x + self.input_range) / self.dx).floor().long(),
            min=-1,
            max=self.num_input_breaks - 1,
        )
        slopes = self.slopes[which + 1]
        nearest_break = torch.clamp(which, min=0, max=self.num_input_breaks - 1)
        values_at_breaks = self.values_at_breaks[nearest_break]
        xanchor = -self.input_range + nearest_break.float() * self.dx
        # distance from the left break, except for which == -1, where we use the first break
        return values_at_breaks + slopes * (x - xanchor)


class Monotonic1D(nn.Module):
    """
    Wraps Monotonic1DFixedRange to handle arbitrary input ranges by scaling inputs and outputs.

    Inputs are scaled via an unnormalized batch normalization, hence the input range being labeled
    max_z_abs.

    :param max_z_abs: float, the z score range to consider
    :param num_input_breaks: int, the number of breaks to use in the piecewise linear function.
    """

    def __init__(self, max_z_abs: float, num_input_breaks: int):
        super().__init__()
        self.max_z_abs = max_z_abs
        self.monotonic_fixed = Monotonic1DFixedRange(
            input_range=max_z_abs, num_input_breaks=num_input_breaks
        )
        self.batch_norm = nn.BatchNorm1d(1, affine=False)
        self.output_m = nn.Parameter(torch.tensor(1.0))
        self.output_b = nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        """
        Applies the monotonic piecewise linear function to the input tensor x after normalizing it.

        :param x: torch.Tensor of shape (batch_size, seq_len, 1), the input values to transform.
        :return: torch.Tensor of the same shape as x, the transformed values.
        """
        original_shape = x.shape
        x = x.view(-1, 1)
        x = self.batch_norm(x)
        x = self.monotonic_fixed(x)
        x = x * self.output_m + self.output_b
        x = x.view(original_shape)
        return x

    def plot_function(self, num_points: int = 1000, extra_range: float = 0.1):
        """
        Plots the monotonic function by evaluating it at evenly spaced points in the input range.

        :param num_points: int, the number of points to evaluate.
        :return: (torch.Tensor, torch.Tensor), the input points and their corresponding output values.
        """
        assert not self.training
        with torch.no_grad():
            z_range = self.max_z_abs * (1.0 + extra_range)
            input_to_underlying = torch.linspace(
                -z_range, z_range, num_points, device=self.output_m.device
            )
            input_overall = inv_batch_norm(
                input_to_underlying.unsqueeze(1), self.batch_norm
            ).squeeze(1)
            output_underlying = self(input_overall.unsqueeze(1)).squeeze(1)
        return input_overall.cpu().numpy(), output_underlying.cpu().numpy()


def inv_batch_norm(x: torch.Tensor, batch_norm: nn.BatchNorm1d) -> torch.Tensor:
    """
    Inverts the batch normalization on the input tensor x.

    :param x: the normalized input values.
    :param batch_norm: nn.BatchNorm1d, the batch normalization layer used for normalization.
    :return: torch.Tensor of the same shape as x, the unnormalized values.
    """
    assert not batch_norm.affine, "Only supports non-affine batch norm inversion"
    assert not batch_norm.training, "Batch norm must be in eval mode to invert"

    mu, var = batch_norm.running_mean, batch_norm.running_var
    std = torch.sqrt(var + batch_norm.eps)
    x_unnormalized = x * std.unsqueeze(0) + mu.unsqueeze(0)
    return x_unnormalized
