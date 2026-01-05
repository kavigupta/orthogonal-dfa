from abc import ABC, abstractmethod
from typing import Union

import torch
from torch import nn

from orthogonal_dfa.module.monotonic import Monotonic1D


class ResidualGate(ABC):
    """
    Abstract base class for residual gates.
    """

    @abstractmethod
    def compute_output(
        self,
        x: torch.Tensor,
        residual_prev: torch.Tensor,
        do_not_train_phi: Union[bool, None] = None,
    ) -> torch.Tensor:
        """
        Given input x and previous residual, computes the next residual.
        """

    @abstractmethod
    def compute_input(
        self,
        x: torch.Tensor,
        residual_next: torch.Tensor,
        do_not_train_phi: Union[bool, None] = None,
    ) -> torch.Tensor:
        """
        Given input x and next residual, computes the previous residual.
        """


class InputMonotonicModelingGate(ResidualGate, nn.Module):
    """
    Combines an input phi with a prior residual r using a monotonic transformation
    on the phi. Adds this to the prior residual r to produce the final output.

    I.e., r_next = r_prev + monotonic(phi)
    """

    def __init__(
        self, phi: nn.Module, max_z_abs: float, num_input_breaks: int, batch_norm=True
    ):
        super().__init__()
        self.monotonic = Monotonic1D(
            max_z_abs=max_z_abs,
            num_input_breaks=num_input_breaks,
            batch_norm=batch_norm,
        )
        self.phi = phi

    def run_phi(
        self, x: torch.Tensor, do_not_train_phi: Union[bool, None] = None
    ) -> torch.Tensor:
        """
        Runs the phi module on input x.
        """
        assert do_not_train_phi in (None, True, False)
        if do_not_train_phi is None:
            assert (
                not self.training
            ), "do_not_train_phi must be specified during training"
            do_not_train_phi = True
        if do_not_train_phi:
            with torch.no_grad():
                p = self.phi(x)
        else:
            p = self.phi(x)
        if len(p.shape) > 1 and p.shape[0] == 1:
            p = p.squeeze(0)
        return p

    def compute_output(
        self,
        x: torch.Tensor,
        residual_prev: torch.Tensor,
        do_not_train_phi: Union[bool, None] = None,
    ) -> torch.Tensor:
        """
        Given x and r_prev, computes r_next = r_prev + monotonic(phi(x))
        """
        phi_output = self.run_phi(x, do_not_train_phi)
        monotonic_output = self.monotonic(phi_output)
        return residual_prev + monotonic_output

    def compute_input(
        self,
        x: torch.Tensor,
        residual_next: torch.Tensor,
        do_not_train_phi: Union[bool, None] = None,
    ) -> torch.Tensor:
        """
        Given x and r_next, computes r_prev = r_next - monotonic(phi(x))
        """
        phi_output = self.run_phi(x, do_not_train_phi)
        monotonic_output = self.monotonic(phi_output)
        return residual_next - monotonic_output

    def notify_epoch_loss(self, epoch_idx, epoch_loss):
        return self.phi.notify_epoch_loss(epoch_idx, epoch_loss)
