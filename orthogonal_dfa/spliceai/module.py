from frozendict import frozendict

import numpy as np
from torch import nn
from dconstruct import construct


def no_preprocess(x):
    return x, {}


class ResidualUnit(nn.Module):
    """
    Residual unit proposed in "Identity mappings in Deep Residual Networks"
    by He et al.

    Shapes: (N, C, L) -> (N, C, L)

    True width is 2 * w - 1, when ar = 1.
    """

    def __init__(self, l, w, ar):
        super().__init__()
        self.normalize1 = nn.BatchNorm1d(l)
        self.normalize2 = nn.BatchNorm1d(l)
        self.act1 = self.act2 = nn.ReLU()

        padding = (ar * (w - 1)) // 2

        self.conv1 = nn.Conv1d(l, l, w, dilation=ar, padding=padding)
        self.conv2 = nn.Conv1d(l, l, w, dilation=ar, padding=padding)

    def forward(self, input_node):
        bn1 = self.normalize1(input_node)
        act1 = self.act1(bn1)
        conv1 = self.conv1(act1)
        assert conv1.shape == act1.shape
        bn2 = self.normalize2(conv1)
        act2 = self.act2(bn2)
        conv2 = self.conv2(act2)
        assert conv2.shape == act2.shape
        output_node = conv2 + input_node
        return output_node


class SpliceAI(nn.Module):
    """
    Original spliceai code; translated from the original tensorflow
        code.
    """

    # pylint: disable=too-many-positional-arguments
    def __init__(
        self,
        l,
        w,
        ar,
        preprocess=no_preprocess,
        starting_channels=4,
        output_size=3,
        *,
        module_spec=frozendict(type="ResidualUnit"),
    ):
        super().__init__()
        assert len(w) == len(ar)
        self.w = w
        self.cl = 2 * sum(ar * (w - 1))
        self.conv1 = nn.Conv1d(starting_channels, l, 1)
        self.conv2 = nn.Conv1d(l, l, 1)

        def get_mod(i):
            res = construct(
                dict(ResidualUnit=ResidualUnit), module_spec, l=l, w=w[i], ar=ar[i]
            )
            return res

        self.convstack = nn.ModuleList([get_mod(i) for i in range(len(self.w))])
        self.skipconv = nn.ModuleList(
            [
                nn.Conv1d(l, l, 1) if self._skip_connection(i) else None
                for i in range(len(self.w))
            ]
        )
        self.output = nn.Conv1d(l, output_size, 1)

        self.preprocess = preprocess

        self.hooks_handles = []

    def forward(self, x):
        if isinstance(x, dict):
            x = x["x"]
        x = x.transpose(1, 2)
        if hasattr(self, "preprocess"):
            x, _ = self.preprocess(x)

        conv = self.conv1(x)
        skip = self.conv2(conv)

        for i in range(len(self.w)):
            conv = self.convstack[i](conv)

            if self._skip_connection(i):
                # Skip connections to the output after every 4 residual units
                skip = skip + self.skipconv[i](conv)

        skip = skip[:, :, self.cl // 2 : -self.cl // 2]

        y = self.output(skip)

        y = y.transpose(1, 2)
        return y

    def _skip_connection(self, i):
        return ((i + 1) % 4 == 0) or ((i + 1) == len(self.w))


class SpliceAIModule(nn.Module):
    """
    Wrapper around the original spliceai module to make it
        compatible with the rest of the codebase.

    Only accepts windows of 80, 400, 2000, or 10000.
    """

    def __init__(
        self,
        *,
        L=32,
        window,
        CL_max=10_000,
        input_size=4,
        output_size=3,
        spliceai_spec=frozendict(type="SpliceAI"),
    ):
        super().__init__()
        W, AR, _, _ = get_hparams(window, CL_max=CL_max)
        self.spliceai = construct(
            dict(SpliceAI=SpliceAI),
            spliceai_spec,
            l=L,
            w=W,
            ar=AR,
            starting_channels=input_size,
            output_size=output_size,
        )

    def forward(self, x):
        return self.spliceai(x)


def get_hparams(window, CL_max):
    """
    Get hyperparameters, based on the window size.

    This is based on the original SpliceAI code.
    """
    if window == 80:
        W = np.asarray([11, 11, 11, 11])
        AR = np.asarray([1, 1, 1, 1])
        BATCH_SIZE = 18
    elif window == 400:
        W = np.asarray([11, 11, 11, 11, 11, 11, 11, 11])
        AR = np.asarray([1, 1, 1, 1, 4, 4, 4, 4])
        BATCH_SIZE = 18
    elif window == 2000:
        W = np.asarray([11, 11, 11, 11, 11, 11, 11, 11, 21, 21, 21, 21])
        AR = np.asarray([1, 1, 1, 1, 4, 4, 4, 4, 10, 10, 10, 10])
        BATCH_SIZE = 12
    elif window == 10000:
        W = np.asarray([11, 11, 11, 11, 11, 11, 11, 11, 21, 21, 21, 21, 41, 41, 41, 41])
        AR = np.asarray([1, 1, 1, 1, 4, 4, 4, 4, 10, 10, 10, 10, 25, 25, 25, 25])
        BATCH_SIZE = 6
    else:
        raise AssertionError(f"Invalid window: {window})

    # Hyper-parameters:
    # L: Number of convolution kernels
    # W: Convolution standard_args.window size in each residual unit
    # AR: Atrous rate in each residual unit

    CL = 2 * np.sum(AR * (W - 1))
    assert CL <= CL_max and CL == window

    return W, AR, BATCH_SIZE, CL
