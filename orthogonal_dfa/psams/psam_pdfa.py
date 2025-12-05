import torch
from torch import nn

from orthogonal_dfa.psams.psams import (
    TorchPSAMs,
    conditional_cascade_log_probs,
    flip_log_probs,
)
from orthogonal_dfa.utils.pdfa import PDFA


class PSAMPDFA(nn.Module):

    @classmethod
    def create(cls, num_input_channels, num_psams, two_r, num_states, *, pdfa_typ=PDFA):
        """
        Creates a PSAM-PDFA model with randomly initialized parameters.

        :param num_input_channels: int, the number of input channels.
        :param num_psams: int, the number of PSAMs.
        :param two_r: int, the PSAM radius times two.
        :param num_states: int, the number of states in the PDFA.
        :return: PSAMPDFA instance with randomly initialized parameters.
        """
        psam = TorchPSAMs.create(
            two_r=two_r, channels=num_input_channels, num_psams=num_psams
        )
        pdfa = pdfa_typ.create(input_channels=num_psams + 1, num_states=num_states)
        return cls(psam, pdfa)

    def __init__(self, psam: nn.Module, pdfa: nn.Module):
        """
        A PSAM followed by a PDFA.

        :param psam: nn.Module that outputs log probabilities over input symbols.
        :param pdfa: PDFA instance that computes acceptance probabilities.
        """
        super().__init__()
        self.psam = psam
        self.pdfa = pdfa

    def forward(self, x):
        log_input_probs = self.psam(x)
        log_input_probs = conditional_cascade_log_probs(log_input_probs, axis=-1)
        return self.pdfa(log_input_probs)


class PSAMPDFAWithTemperature(nn.Module):
    def __init__(self, psam_pdfa: PSAMPDFA, temperature: float):
        """
        A PSAM-PDFA model with an adjustable temperature parameter.

        :param psam_pdfa: PSAMPDFA instance.
        :param temperature: float, the temperature parameter for scaling log probabilities.
        """
        super().__init__()
        self.psam_pdfa = psam_pdfa
        self.temperature = nn.Parameter(torch.tensor(float(temperature)))

    def forward(self, x):
        log_probs = self.psam_pdfa(x)
        logit = log_probs - flip_log_probs(log_probs)
        scaled_logit = logit / self.temperature
        # pylint: disable=not-callable
        scaled_log_probs = nn.functional.logsigmoid(scaled_logit)
        return scaled_log_probs
