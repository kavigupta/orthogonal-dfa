import torch
from torch import nn
from torch._refs import cumsum_


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


class Monotonic2DFixedRange(nn.Module):
    """
    Represents a 2D monotonic function as the double integral of a piecewise step function.

    The integral is computed by a double cumulative sum of the step function values for each
    of the grid squares, and then the result is normalized to ensure that F(input_range, input_range) = input_range.

    Bilinear interpolation is used to compute the function value at non-grid points, with extrapolation
    of the nearest grid square for points outside the grid.

    This is different from how the 1D monotonic function is defined and has slightly fewer parameters,
    but is the same general concept.
    """

    def __init__(self, input_range: float, num_input_breaks: int):
        super().__init__()
        self.input_range = input_range
        self.num_input_breaks = num_input_breaks

        # Step function values for each grid square (all must be positive)
        # Shape: (num_input_breaks, num_input_breaks)
        # We use (num_input_breaks) * (num_input_breaks) because we have (num_input_breaks + 1) * (num_input_breaks + 1) grid squares
        # and we also add a bit of padding to the left and bottom to ensure that the cumulative integral is not exactly 0 at the edges
        self.inv_softplus_step_values = nn.Parameter(
            torch.randn(num_input_breaks, num_input_breaks)
        )
        self.softplus = nn.Softplus()

    @property
    def dx(self):
        """
        Computes the distance between consecutive input breaks.
        """
        return (2.0 * self.input_range) / (self.num_input_breaks - 1)

    def cumulative_integral(self):
        """
        Computes the cumulative double integral at each grid point.
        """
        step_vals = self.softplus(self.inv_softplus_step_values)
        # Each cell [i, i+1] x [j, j+1] contributes g[i, j] * dx * dx to the integral
        cell_integrals = step_vals * self.dx * self.dx

        # Compute 2D cumulative sum
        # First cumsum along x, then along y
        cumsum_x = torch.cumsum(cell_integrals, dim=1)
        cumsum_xy = torch.cumsum(cumsum_x, dim=0)

        range_total = cumsum_xy[-1, -1] - cumsum_xy[0, 0]
        cumsum_xy = -self.input_range + (cumsum_xy - cumsum_xy[0, 0]) * (2.0 * self.input_range) / range_total

        return cumsum_xy

    def forward(self, x, y):
        """
        Applies the monotonic 2D function to input tensors x and y.

        :param x: torch.Tensor of arbitrary shape, the x input values.
        :param y: torch.Tensor of the same shape as x, the y input values.
        :return: torch.Tensor of the same shape as x and y, the transformed values.
        """
        # idx of the bottom left of the grid square containing the point. is 0 if you are left of the first break,
        # and num_input_breaks - 2 if you are right of the last break (because we need to move one to the left so
        # we have a grid square to base the interpolation on)
        low_x_idx = torch.clamp(
            ((x + self.input_range) / self.dx).floor().long(),
            min=0,
            max=self.num_input_breaks - 2,
        )
        # analogous for y
        low_y_idx = torch.clamp(
            ((y + self.input_range) / self.dx).floor().long(),
            min=0,
            max=self.num_input_breaks - 2,
        )

        low_x = -self.input_range + low_x_idx.float() * self.dx
        low_y = -self.input_range + low_y_idx.float() * self.dx

        high_x = low_x + self.dx
        high_y = low_y + self.dx

        cint = self.cumulative_integral()
        z_at_grid = cint[low_y_idx, low_x_idx]
        z_at_grid_right = cint[low_y_idx, low_x_idx + 1]
        z_at_grid_bottom = cint[low_y_idx + 1, low_x_idx]
        z_at_grid_bottom_right = cint[low_y_idx + 1, low_x_idx + 1]

        # bilinear interpolation
        result = (high_x - x) * (high_y - y) * z_at_grid + (x - low_x) * (high_y - y) * z_at_grid_right + (high_x - x) * (y - low_y) * z_at_grid_bottom + (x - low_x) * (y - low_y) * z_at_grid_bottom_right
        return result / (self.dx ** 2)


class Monotonic2D(nn.Module):
    """
    Wraps Monotonic2DFixedRange to handle arbitrary input ranges by scaling inputs and outputs.

    Inputs are scaled via batch normalization, similar to Monotonic1D.

    :param max_z_abs: float, the z score range to consider
    :param num_input_breaks: int, the number of breaks to use in each dimension.
    """

    def __init__(self, max_z_abs: float, num_input_breaks: int):
        super().__init__()
        self.max_z_abs = max_z_abs
        self.monotonic_fixed = Monotonic2DFixedRange(
            input_range=max_z_abs, num_input_breaks=num_input_breaks
        )
        self.batch_norm_x = nn.BatchNorm1d(1, affine=False)
        self.batch_norm_y = nn.BatchNorm1d(1, affine=False)
        self.output_m = nn.Parameter(torch.tensor(1.0))
        self.output_b = nn.Parameter(torch.tensor(0.0))

    def forward(self, x, y):
        """
        Applies the monotonic 2D function to input tensors x and y after normalizing them.

        :param x: torch.Tensor of shape (batch_size, seq_len, 1) or similar, the x input values.
        :param y: torch.Tensor of the same shape as x, the y input values.
        :return: torch.Tensor of the same shape as x and y, the transformed values.
        """
        original_shape = x.shape
        x_flat = x.view(-1, 1)
        y_flat = y.view(-1, 1)
        x_norm = self.batch_norm_x(x_flat)
        y_norm = self.batch_norm_y(y_flat)
        result = self.monotonic_fixed(x_norm, y_norm)
        result = result * self.output_m + self.output_b
        return result.view(original_shape)
