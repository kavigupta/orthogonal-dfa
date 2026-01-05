import numpy as np
from dconstruct import construct
from torch import nn

from orthogonal_dfa.module.sparsity.sparsity_layer import sparsity_types
from orthogonal_dfa.module.sparsity.sparsity_update import suo_types


class AutomaticSparseLayer(nn.Module):
    def __init__(self, sparse_spec, suo_spec):
        super().__init__()
        self.sparsity = construct(sparsity_types(), sparse_spec)
        self.suo = construct(suo_types(), suo_spec)

    def forward(self, x):
        return self.sparsity(x)

    def notify_epoch_loss(self, epoch_idx, epoch_loss):
        epoch_loss = np.mean(epoch_loss)
        epoch_reward = -epoch_loss
        return self.suo.update_sparsity(
            self.sparsity.sparsity, self.update_sparsity, epoch_idx, epoch_reward
        )

    def update_sparsity(self, new_sparsity):
        self.sparsity.sparsity = new_sparsity
