import torch
from permacache import stable_hash
from torch import nn


class RNNProcessor(nn.Module):
    def __init__(self, num_inputs, hidden_size, num_layers=1):
        """
        An RNN-based processor for sequential data. Produces only the last output.

        :param num_inputs: int, the number of input features.
        :param hidden_size: int, the number of features in the hidden state.
        :param num_layers: int, number of recurrent layers.
        :param bidirectional: bool, if True, becomes a bidirectional RNN.
        """
        super().__init__()
        self.rnn = nn.RNN(
            input_size=num_inputs,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=False,
            batch_first=True,
        )
        self.linear_out = nn.Linear(hidden_size, 1)

    def forward(self, x):
        rnn_out, _ = self.rnn(x)
        rnn_out = rnn_out[:, -1, :]
        output = self.linear_out(rnn_out)
        output = output.T
        return output

    def __permacache_hash__(self):
        return stable_hash(
            ("RNNProcessor", self.rnn.state_dict(), self.linear_out.state_dict())
        )

    def notify_epoch_loss(self, epoch_idx, epoch_loss):
        # no sparsity here
        return None


class LSTMProcessor(nn.Module):
    def __init__(self, num_inputs, hidden_size, num_layers=1, use_last_state=True):
        super().__init__()
        self.rnn = nn.LSTM(
            input_size=num_inputs,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=False,
            batch_first=True,
        )
        self.linear_out = nn.Linear(hidden_size * (2 if use_last_state else 1), 1)
        self.use_last_state = use_last_state

    def forward(self, x):
        rnn_out, (h, c) = self.rnn(x)
        if self.use_last_state:
            rnn_out = torch.cat([h, c], dim=2)[-1]
        else:
            rnn_out = rnn_out[:, -1, :]
        output = self.linear_out(rnn_out)
        output = output.T
        return output

    def __permacache_hash__(self):
        return stable_hash(
            ("RNNProcessor", self.rnn.state_dict(), self.linear_out.state_dict())
        )

    def notify_epoch_loss(self, epoch_idx, epoch_loss):
        # no sparsity here
        return None


class RNNPSAMProcessorNoise(nn.Module):
    def __init__(self, psams, rnn, *, noise_level):
        """
        An RNN-based processor for PSAM outputs, the PSAM outputs are noised to ensure
        they are treated as actual probabilities.
        """
        super().__init__()
        self.psams = psams
        self.rnn = rnn
        self.noise_level = noise_level

    def forward(self, x):
        psam_out = self.psams(x).exp()
        if self.training:
            psam_out = psam_out + torch.randn_like(psam_out) * self.noise_level
        rnn_out = self.rnn(psam_out)
        return rnn_out

    def notify_epoch_loss(self, epoch_idx, epoch_loss):
        # no sparsity here
        return None


class RNNPSAMProcessorSparse(nn.Module):
    def __init__(self, psams, rnn, *, asl):
        super().__init__()
        self.psams = psams
        self.rnn = rnn
        self.asl = asl

    def forward(self, x):
        psam_out = self.psams(x)
        psam_out = self.asl(psam_out)
        rnn_out = self.rnn(psam_out)
        return rnn_out

    def notify_epoch_loss(self, epoch_idx, epoch_loss):
        return self.asl.notify_epoch_loss(epoch_idx, epoch_loss)
