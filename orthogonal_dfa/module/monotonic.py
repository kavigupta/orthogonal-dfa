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


class Monotonic2DFixedRange(nn.Module):
    """
    Represents a 2D monotonic function as the double integral of a piecewise step function.

    The function is defined as F(x, y) = ∫_{-input_range}^{y} ∫_{-input_range}^{x} g(s, t) ds dt,
    where g(s, t) is a piecewise step function that's constant on each grid square.

    The grid is uniform from -input_range to +input_range in both dimensions, with
    num_input_breaks specifying the number of breaks in each dimension.

    This ensures the function is monotonic in both x and y directions.

    :param input_range: float, the range of the input and output values.
    :param num_input_breaks: int, the number of breaks to use in each dimension.
    """

    def __init__(self, input_range: float, num_input_breaks: int):
        super().__init__()
        self.input_range = input_range
        self.num_input_breaks = num_input_breaks

        # Step function values for each grid square (all must be positive)
        # Shape: (num_input_breaks, num_input_breaks)
        # We use num_input_breaks x num_input_breaks because we have that many grid squares
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

    @property
    def step_values(self):
        """
        Computes the step function values by applying softplus to ensure positivity.
        These represent the constant value of g(x, y) in each grid square.
        No normalization here - normalization happens in cumulative_integral.
        """
        return self.softplus(self.inv_softplus_step_values)

    @property
    def cumulative_integral(self):
        """
        Computes the cumulative double integral at each grid point.
        This is F(x_i, y_j) = ∫_{-input_range}^{y_j} ∫_{-input_range}^{x_i} g(s, t) ds dt
        where (x_i, y_j) are the grid break points.
        """
        step_vals = self.step_values
        # Each cell [i, i+1] x [j, j+1] contributes g[i, j] * dx * dx to the integral
        cell_integrals = step_vals * self.dx * self.dx

        # Compute 2D cumulative sum
        # First cumsum along x, then along y
        cumsum_x = torch.cumsum(cell_integrals, dim=1)
        cumsum_xy = torch.cumsum(cumsum_x, dim=0)

        # Create result array with shape (num_input_breaks, num_input_breaks)
        # result[i, j] = F(breaks[i], breaks[j]) where breaks are the grid break points
        result = torch.zeros(
            self.num_input_breaks,
            self.num_input_breaks,
            device=step_vals.device,
        )

        # Fill interior: result[i, j] = cumulative integral up to break point (i, j)
        # cumsum_xy[i, j] is the integral up to cell [i, j], which corresponds to break point [i+1, j+1]
        # So result[i, j] = cumsum_xy[i-1, j-1] for i, j > 0
        result[1:, 1:] = -self.input_range + cumsum_xy[:-1, :-1]

        # Set boundary values
        result[0, 0] = -self.input_range

        # Fill first row and column from step values (before normalization)
        # These must be computed to ensure monotonicity
        if self.num_input_breaks > 1:
            # First row (y = breaks[0] = -input_range): integrate step values along x only
            # At y = -input_range, we integrate only in x direction
            x_cumsum = torch.cumsum(step_vals[0, :] * self.dx, dim=0)
            result[0, 1:] = result[0, 0] + x_cumsum[:-1]

            # First column (x = breaks[0] = -input_range): integrate step values along y only
            # At x = -input_range, we integrate only in y direction
            y_cumsum = torch.cumsum(step_vals[:, 0] * self.dx, dim=0)
            result[1:, 0] = result[0, 0] + y_cumsum[:-1]

        # Normalize so F(breaks[-1], breaks[-1]) = input_range
        # Normalize everything together to preserve relationships and monotonicity
        total_range = result[-1, -1] - result[0, 0]
        if total_range > 1e-10:
            scale = (2.0 * self.input_range) / total_range
            result = result[0, 0] + (result - result[0, 0]) * scale

        # Verify and fix monotonicity: ensure result[i+1, j] >= result[i, j] for all i, j
        # This ensures the function is monotonic in y
        for j in range(self.num_input_breaks):
            for i in range(self.num_input_breaks - 1):
                if result[i + 1, j] < result[i, j]:
                    # Force monotonicity by setting to previous value
                    result[i + 1, j] = result[i, j]

        # Ensure result[i, j+1] >= result[i, j] for all i, j (monotonic in x)
        for i in range(self.num_input_breaks):
            for j in range(self.num_input_breaks - 1):
                if result[i, j + 1] < result[i, j]:
                    # Force monotonicity by setting to previous value
                    result[i, j + 1] = result[i, j]

        return result

    def forward(self, x, y):
        """
        Applies the monotonic 2D function to input tensors x and y.

        :param x: torch.Tensor of arbitrary shape, the x input values.
        :param y: torch.Tensor of the same shape as x, the y input values.
        :return: torch.Tensor of the same shape as x and y, the transformed values.
        """
        original_shape = x.shape
        x_flat = x.flatten()
        y_flat = y.flatten()

        # Find which grid cell each point belongs to
        # Grid points are at: -input_range + i * dx for i = 0, 1, ..., num_input_breaks
        x_normalized = (x_flat + self.input_range) / self.dx
        y_normalized = (y_flat + self.input_range) / self.dx

        # Check if we're exactly at grid points (within numerical tolerance)
        x_is_grid = torch.abs(x_normalized - x_normalized.round()) < 1e-6
        y_is_grid = torch.abs(y_normalized - y_normalized.round()) < 1e-6

        # Get cumulative integral
        cum_int = self.cumulative_integral

        which_x = torch.clamp(
            x_normalized.floor().long(),
            min=-1,
            max=self.num_input_breaks - 1,
        )
        which_y = torch.clamp(
            y_normalized.floor().long(),
            min=-1,
            max=self.num_input_breaks - 1,
        )

        # Get the grid cell indices (clamped to valid range for step function)
        idx_x_cell = torch.clamp(which_x, min=0, max=self.num_input_breaks - 1)
        idx_y_cell = torch.clamp(which_y, min=0, max=self.num_input_breaks - 1)

        # Get grid indices for the four corners of the cell
        idx_x_grid = torch.clamp(which_x, min=0, max=self.num_input_breaks - 1)
        idx_y_grid = torch.clamp(which_y, min=0, max=self.num_input_breaks - 1)
        idx_x_next = torch.clamp(idx_x_grid + 1, min=0, max=self.num_input_breaks - 1)
        idx_y_next = torch.clamp(idx_y_grid + 1, min=0, max=self.num_input_breaks - 1)

        f_00 = cum_int[idx_y_grid, idx_x_grid]  # bottom-left
        f_10 = cum_int[idx_y_next, idx_x_grid]  # top-left
        f_01 = cum_int[idx_y_grid, idx_x_next]  # bottom-right
        f_11 = cum_int[idx_y_next, idx_x_next]  # top-right

        # Get the anchor point (bottom-left corner of the grid cell)
        x_anchor = -self.input_range + idx_x_cell.float() * self.dx
        y_anchor = -self.input_range + idx_y_cell.float() * self.dx

        # Compute normalized coordinates within the cell [0, 1]
        alpha = (x_flat - x_anchor) / self.dx
        beta = (y_flat - y_anchor) / self.dx
        alpha = torch.clamp(alpha, min=0.0, max=1.0)
        beta = torch.clamp(beta, min=0.0, max=1.0)

        # Use bilinear interpolation from corner values
        # This ensures consistency with grid points and maintains monotonicity
        result_local = (
            f_00 * (1 - alpha) * (1 - beta)
            + f_01 * alpha * (1 - beta)
            + f_10 * (1 - alpha) * beta
            + f_11 * alpha * beta
        )

        return result_local.view(original_shape)


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
