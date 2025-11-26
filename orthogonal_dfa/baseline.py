import numpy as np
import torch
from torch import nn

from orthogonal_dfa.psams.psams import TorchPSAMs


class MonolithicLinearLayer(nn.Module):
    def __init__(self, num_input_channels, input_length: int):
        super().__init__()
        self.num_input_channels = num_input_channels
        self.input_length = input_length
        self.linear = nn.Linear(num_input_channels * input_length, 1)
        self.log_sigmoid = nn.LogSigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.reshape(x.size(0), -1)
        x = self.linear(x)
        assert x.shape[1] == 1
        x = x.squeeze(1)
        x = self.log_sigmoid(x)
        return x[None]

    @property
    def weight_arr(self) -> np.ndarray:
        return (
            self.linear.weight.detach()
            .cpu()
            .numpy()
            .reshape(self.input_length, self.num_input_channels)
        )


class PSAMsFollowedByLinear(nn.Module):
    def __init__(self, num_input_channels, num_psams, two_r, input_length):
        super().__init__()
        self.num_input_channels = num_input_channels
        self.num_psams = num_psams
        self.two_r = two_r
        self.input_length = input_length
        self.psams = TorchPSAMs.create(
            two_r=two_r, channels=num_input_channels, num_psams=num_psams
        )
        self.linear = MonolithicLinearLayer(
            num_input_channels=num_psams, input_length=input_length - two_r
        )
        self.nonlinearity = "exp"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        log_input_probs = self.psams(x)
        if self.nonlinearity == "exp":
            log_input_probs = log_input_probs.exp()
        return self.linear(log_input_probs)
